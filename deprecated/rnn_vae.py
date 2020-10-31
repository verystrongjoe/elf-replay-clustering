import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import timeit
import random
import datetime
from pytictoc import TicToc


class RnnType:
    GRU = 1
    LSTM = 2


class ActivationFunction:
    RELU = 1
    TANH = 2
    SIGMOID = 3


class Parameters:
    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec("self.%s=%s" % (k, v))



class Encoder(nn.Module):

    def __init__(self, device, params):
        super(Encoder, self).__init__()
        self.device = device
        self.params = params
        if self.params.rnn_type not in [RnnType.LSTM,  RnnType.GRU]:
            raise Exception("Unknown RNN type.")
        self.num_directions = 2 if self.params.bidirectional_encoder == True else 1
        if self.params.rnn_type == RnnType.GRU:
            self.num_hidden_states = 1
            rnn = nn.GRU
        elif self.params.rnn_type == RnnType.LSTM:
            self.num_hidden_states = 2
            rnn = nn.LSTM
        self.rnn = rnn(self.params.input_dim,
                       self.params.rnn_hidden_dim,
                       num_layers= self.params.num_layers,
                       bidirectional=self.params.bidirectional_encoder,
                       dropout=self.params.dropout,
                       batch_first=True)
        self.hidden = None
        self.linear_dims = params.linear_dims
        self.linear_dims = [self.params.rnn_hidden_dim * self.num_directions * self.params.num_layers * self.num_hidden_states] + self.linear_dims

        # for vae
        self.last_dropout = nn.Dropout(p=self.params.dropout)
        self.hidden_to_mean = nn.Linear(self.linear_dims[-1], self.params.z_dim)
        self.hidden_to_logv = nn.Linear(self.linear_dims[-1], self.params.z_dim)

        self._init_weights()

    def init_hidden(self, batch_size):
        if self.params.rnn_type == RnnType.GRU:
            return torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device)
        elif self.params.rnn_type == RnnType.LSTM:
            return (torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device),
                    torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device))

    def forward(self, inputs):
        batch_size, _, _ = inputs.shape
        _, self.hidden = self.rnn(inputs, self.hidden)
        X = self._flatten_hidden(self.hidden, batch_size)
        # vae
        # mean = self.hidden_to_mean(inputs)
        # logv = self.hidden_to_logv(inputs)
        # z = self._sample(mean, logv)
        # return mean, logv, z
        return X

    def _flatten_hidden(self, h, batch_size):
        if h is None:
            return None
        elif isinstance(h, tuple): # for lstm
            X = torch.cat(
                [self._flatten(h[0], batch_size),
                 self._flatten(h[1], batch_size)],
                1
            )
        else:  # for gru
            X = self._flatten(h, batch_size)
        return X

    def _flatten(self, h, batch_size):
        return h.transpose(0, 1).contiguous().view(batch_size, -1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def _sample(self, mean, logv):
        std = torch.exp(0, 5 * logv)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return z


class Decoder(nn.Module):
    def __init__(self, device, params):
        super(Decoder, self).__init__()
        self.device = device
        self.params = params

        if self.params.rnn_type not in [RnnType.GRU, RnnType.LSTM]:
            raise Exception("Unknown RNN type.")
        self.num_directions = 2 if self.params.bidirectional_encoder == True else 1

        if self.params.rnn_type == RnnType.GRU:
            self.num_hidden_states = 1
            rnn = nn.GRU
        elif self.params.rnn_type == RnnType.LSTM:
            self.num_hidden_states = 2
            rnn = nn.LSTM
        self.rnn = rnn(self.params.input_dim,
                       self.params.rnn_hidden_dim*self.num_directions,
                       num_layers=self.params.num_layers,
                       dropout=self.params.dropout,
                       batch_first=True)
        self.linear_dims = self.params.linear_dims + [self.params.rnn_hidden_dim * self.num_directions * self.params.num_layers * self.num_hidden_states]
        self.z_to_hidden = nn.Linear(self.params.z_dim, self.linear_dims[0])

        self.out = nn.Linear(self.params.rnn_hidden_dim * self.num_directions, self.params.input_dim)
        self._init_weights()

    def forward(self, inputs, z, return_objects=False):
        batch_size, num_steps, input_dim = inputs.shape

        # for vae
        # X = self.z_to_hidden(z)
        X = z

        hidden = self._unflatten_hidden(X, batch_size)
        hidden = self._init_hidden_state(hidden)

        # # Unflatten hidden state for GRU or LSTM
        # hidden = self._unflatten(X, batch_size)
        # hidden = self._init_hidden_state(hidden)

        loss = 0
        outputs = torch.zeros((batch_size, num_steps), dtype=torch.long).to(self.device)

        for i in range(num_steps):
            output, hidden = self._step(input, hidden)
            topv, topi = output.topk(1)
            input = topi.detach()
            outputs[:, i] = topi.detach().squeeze()
            loss += self.criterion(output, inputs[:, i])

        if return_objects == True:
            return loss, outputs
        else:
            return loss

    def generate(self, z, max_steps):
        decoded_sequence = []
        X = self.z_to_hidden(z)

        hidden = self._unflaten_hidden(X,1)
        hidden = self._init_hidden_state(hidden)

        for i in  range(max_steps):
            output, hidden = self._step(input, hidden)
            topv, topi = output.data.topk(1)

            decoded_sequence.append(topi.item())
            input = topi.detach()

        return decoded_sequence

    def _step(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = F.log_softmax(self.out(output.squeeze(dim=1)), dim=-1)

        return output, hidden

    def _init_hidden_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        elif isinstance(encoder_hidden, tuple): # LSTM
            return tuple([self._concat_directions(h) for h in encoder_hidden])
        else:
            return self._concat_directions(encoder_hidden)

    def _concat_directions(self, hidden):
        if self.params.bidirectional_encoder:
            hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
            # h = hidden.view(self.params.num_layers, self.num_directions, hidden.size(1), self.params.rnn_hidden_dim)
            # h_fwd = h[:, 0, :, :]
            # h_bwd = h[:, 1, :, :]
            # hidden = torch.cat([h_fwd, h_bwd], 2)
        return hidden

    def _unflatten_hidden(self, X, batch_size):
        if X is None:
            return None
        elif self.params.rnn_type == RnnType.LSTM:  # LSTM
            X_split = torch.split(X, int(X.shape[1] / 2), dim=1)
            h = (self._unflatten(X_split[0], batch_size), self._unflatten(X_split[1], batch_size))
        else:  # GRU
            h = self._unflatten(X, batch_size)
        return h

    # def _flatten(self, h, batch_size):
    #     return h.transpose(0, 1).contiguous().view(batch_size, -1)

    def _unflatten(self, X, batch_size):
        return X.view(batch_size, self.params.num_layers * self.num_directions, self.params.rnn_hidden_dim).transpose(0,1).contiguous()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class RnnVae:

    def __init__(self, device, params, criterion):
        self.device = device
        self.params = params

        self.encoder = Encoder(device, params)
        self.decoder = Decoder(device, params)
        self.encoder.to(device)
        self.decoder.to(device)

        self.encoder_lr = self.params.encoder_lr
        self.decoder_lr = self.params.decoder_lr

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.encoder_lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.decoder_lr)


    def update_learning_rates(self, encoder_factor, decoder_factor):
        self.encoder_lr = self.encoder_lr * self.encoder_factor
        self.decoder_lr = self.decoder_lr * self.decoder_factor
        self.set_learning_rates(self.encoder_lr, self.decoer_lr)

    def set_learning_rates(self, encoder_lr, decoder_lr):
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr

        for param_group in self.encoder_optimizer.param_groups:
            param_group['lr'] = encoder_lr

        for param_group in self.decoder_optimizer.param_groups:
            param_group['lr'] = decoder_lr

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def train_epoch(self, epoch, X_iter, verbose=0):
        t = TicToc()
        start = t.tic()
        epoch_loss = 0.0
        num_batches = 5

        for idx, inputs in enumerate(X_iter):
            inputs = inputs['input']
            batch_size = inputs.shape[0]

            # Convert to tensors and move to device
            inputs = torch.tensor(inputs).to(self.device)

            # Train batch and get batch loss
            batch_loss = self.train_batch(inputs)
            # Update epoch loss given als batch loss
            epoch_loss += batch_loss

            if verbose != 0:
                print('[{}] Epoch: {} #batches {}/{}, loss: {:.8f}, learning rates: {:.6f}/{:.6f}'.format(
                    datetime.timedelta(seconds=int(t.toc() - start)), epoch + 1, idx + 1, num_batches,
                    (batch_loss / ((idx + 1) * batch_size)), self.encoder_lr, self.decoder_lr), end='\r')

        return epoch_loss

    def test_epoch(self, epoch, X_iter, verbose=0):
        start = timeit.default_timer()
        epoch_loss = 0.0
        num_batches = X_iter.batch_sampler.batch_count()
        for idx, inputs in enumerate(X_iter):
            batch_size = inputs.shape[0]
            inputs = torch.tensor(inputs).to(self.device)
            batch_loss = self.test_batch(inputs)
            epoch_loss += batch_loss

            if verbose !=0:
                print('[{}] Epoch: {} #batches {}/{}, loss: {:.8f}, learning rates: {:.6f}/{:.6f}'.format(
                    datetime.timedelta(seconds=int(timeit.default_timer() - start)), epoch + 1, idx + 1, num_batches,
                    (batch_loss / ((idx + 1) * batch_size)), self.encoder_lr, self.decoder_lr), end='\r')
        return epoch_loss

    @staticmethod
    def are_equal_tensor(a,b):
        if torch.all(torch.eq(a,b)).data.cpu().numpy() == 0:
            return False
        else:
            return True

    def train_batch(self, inputs):
        batch_size, num_steps, input_dim = inputs.shape
        self.encoder.hidden = self.encoder.init_hidden(batch_size)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # mean, logv, z = self.encoder(inputs)
        z = self.encoder(inputs)

        loss = self.decoder(inputs, z)
        # kld_loss = (-0.5 * torch.sum((logv - torch.pow(mean, 2) - torch.exp(logv) + 1), 1)).mean()
        # loss += (kld_loss * 0.1)

        # backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.params.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.params.clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / (num_steps)

    def test_batch(self, inputs, max_steps=100, use_mean=False):
        batch_size, num_steps = inputs.shape
        self.encoder.hideen = self.encoder.init_hidden(batch_size)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        mean, logv, z = self.encoder(inputs)
        loss = self.decoder(inputs, z)

        return loss.item() / (num_steps)

    def evaluate(self, input, max_steps= 100, use_mean=False):
        batch_size, _ = input.shape
        self.encoder.hidden = self.encoder.init_hidden(batch_size)

        mean, logv, z = self.encoder(input)

        if use_mean == True:
            decoded_sequence = self.decoder.generate(mean, max_steps=max_steps)
        else:
            decoded_sequence = self.decoder.generate(z, max_steps=max_steps)

        def save_models(self, encoder_file_name, decoder_file_name):
            torch.save(self.encoder, encoder_file_name)
            torch.save(self.decoder, decoder_file_name)

        def save_state_dicts(self, encoder_file_name, decoder_file_name):
            torch.save(self.encoder.state_dict(), encoder_file_name)
            torch.save(self.decoder.state_dict(), decoder_file_name)

        def load_models(self, encoder_file_name, decoder_file_name):
            self.encoder = torch.load(encoder_file_name)
            self.decoder = torch.load(decoder_file_name)

        def load_state_dicts(self, encoder_file_name, decoder_file_name):
            self.encoder.load_state_dict(torch.load(encoder_file_name))
            self.encoder.load_state_dict(torch.load(decoder_file_name))

