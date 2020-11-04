import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import timeit
import random
import datetime
from pytictoc import TicToc
import argslist


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CATEGORICAL_INDEX_SIZES = {}
for i in argslist.categorical_feature_idxes:
    CATEGORICAL_INDEX_SIZES[i] = 2
# NUMERICAL_INDEX_SIZES = {}


class ScalarEmbedding(nn.Module):
    """Embedding layer for a scalar type spatial feature from pysc2."""
    def __init__(self, embedding_dim, name=None):
        super(ScalarEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embed_fn = nn.Conv2d(
            in_channels=1,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__

    def forward(self, x):
        """
        Arguments:
            x: 4d tensor of shape (B, T, H, W)
        Returns a 5d tensor of shape (B, T, embedding_dim, H, W).
        """
        inputs = x.permute(1, 0, 2, 3)  # (T, B, H, W)
        outputs = []
        for inp in inputs:
            inp = inp.unsqueeze(1)    # (B, H, W) -> (B, 1, H, W)
            out = self.embed_fn(inp.float())  # (B, embedding_dim, H, W)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)  # (B, T, embedding_dim, H, W)

        return outputs


class CategoricalEmbedding(nn.Module):
    """Embedding layer for a categorical spatial feature from pysc2."""
    def __init__(self, category_size, embedding_dim, name=None):
        super(CategoricalEmbedding, self).__init__()
        self.category_size = category_size
        self.embedding_dim = embedding_dim
        self.embed_fn = nn.Embedding(
            num_embeddings=category_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__

    def forward(self, x):
        """
        Arguments:
            x: 4d tensor of shape (B, T, H, W)
        Returns a 5d tensor of shape (B, T, embedding_dim, H, W).
        """
        try:
            out = self.embed_fn(x.long())  # (B, T, H, W, emb_dim)
        except RuntimeError as e:
            print(f"Name: {self.name}")
            print(f"MAX: {x.max()}, MIN: {x.min()}")
            raise RuntimeError(str(e))
        out = out.permute(0, 1, 4, 2, 3)

        return out


class BasicBlock3D(nn.Module):
    """Add class docstring."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv3d(self.in_ch, self.out_ch, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.out_ch)
        self.conv2 = nn.Conv3d(self.out_ch, self.out_ch, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(self.out_ch)

        self.conv_skip = nn.Conv3d(self.in_ch, self.out_ch, 1, 1, padding=0, bias=True)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += self.conv_skip(residual)
        out = F.relu(out)

        return out


class ResNet3D(nn.Module):
    in_planes = [32, 64, 128]

    def __init__(self, in_channels: int):
        super(ResNet3D, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv3d(self.in_channels, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.res1 = nn.Sequential(
            BasicBlock3D(32, 64, stride=1, downsample=None),
            nn.MaxPool3d((5, 2, 2), padding=0)  # (B, 64, T'=20, 10, 10)
        )
        self.res2 = nn.Sequential(
            BasicBlock3D(64, 128, stride=1, downsample=None),
            nn.MaxPool3d((1, 2, 2), padding=0)  # (B, 128, T''=4, 5, 5)
        )
        self.res3 = nn.Sequential(
            BasicBlock3D(128, 256, stride=1, downsample=None),
            nn.MaxPool3d((4, 1, 1), padding=0)  # (B, 256, T'''=1, 5, 5)
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)  # (B, 256, 1, 1, 1)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x: torch.FloatTensor):
        """x: (B, T, C * emb_dim, H, W)"""
        x = x.permute(0, 2, 1, 3, 4)  # (B, C *emb_dim, T=20, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.avgpool(x).squeeze()
        return self.mlp(x)


class MoCo(nn.Module):
    def __init__(self):
        super(MoCo, self).__init__()
        self.embeddings = nn.ModuleDict()
        for i in argslist.categorical_feature_idxes:
            name = f'cf_{i}'
            self.embeddings[name] = CategoricalEmbedding(
                category_size=CATEGORICAL_INDEX_SIZES[i],
                embedding_dim=argslist.embed_dim,
                name=name
            )
        for i in argslist.numerical_feature_idxes:
            name = f'nf_{i}'
            self.embeddings[name] = ScalarEmbedding(argslist.embed_dim, name=name)

        self.device = argslist.device
        self.queue = torch.rand(argslist.proj_dim, argslist.num_negatives, device=argslist.device)
        self.queue = F.normalize(self.queue, dim=0)
        self.ptr = torch.zeros(1, device=argslist.device)

        self.net_q = nn.ModuleDict(
            {
                'embeddings': self.embeddings,
                'encoder': ResNet3D(argslist.num_raw_channels * argslist.embed_dim)
            }
        )
        self.net_k = copy.deepcopy(self.net_q)
        for p_k in self.net_k.parameters():
            p_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_net(self):
        key_momentum = 0.999
        for p_k, p_q in zip(self.net_k.parameters(), self.net_q.parameters()):
            p_k.data = key_momentum * p_k.data + (1 - key_momentum) * p_q.data

    def forward(self, batch: dict):

        pos_1 = batch['pos_1'].to(self.device)  # (B, T, C, H, W)
        pos_2 = batch['pos_2'].to(self.device)  # (B, T, C, H, W)

        # Query net
        x_q = []
        pos_1 = pos_1.permute(2, 0, 1, 3, 4)
        for i, p1 in enumerate(pos_1):
            x_q += [self.net_q['embeddings'][i].forward(p1)]  # (B, T, emb_dim, H, W)
        x_q = torch.cat(x_q, dim=2)                           # (B, T, C * emb_dim, H, W)
        z_q = F.normalize(self.net_q['encoder'][x_q], dim=1)  # (B, f)

        # Key net
        x_k = []
        pos_2 = pos_2.permute(2, 0, 1, 3, 4)
        with torch.no_grad():
            for i, p2 in enumerate(pos_2):
                x_k += [self.net_k['embeddings'][i].forward(p2)]  # (B, T, emb_dim, H, W)
                x_k = torch.cat(x_k, dim=2)  # (B, T, C * emb_dim, H, W)
                z_k = F.normalize(self.net_k['encoder'][x_k], dim=1)  # (B, f)

        logits_pos = torch.einsum('bf,bf->b', [z_q, z_k]).view(-1, 1)  # (B, 1)
        logits_neg = torch.einsum('bf,fk->bk', [z_q, self.queue])      # (B, K); K=negative examples
        logits = torch.cat([logits_pos, logits_neg], dim=0)            # (B, 1+K)
        logits.div_(0.2)  # TODO: add to argslist, argslist.temperature
        target = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)  # indicator of pos example

        self._update_queue(z_k)

        return logits, target

    @staticmethod
    def train_epoch(model, data_loader, optimizer):
        """Train defined for a single epoch."""
        steps = len(data_loader)
        train_loss = torch.zeros(steps, device=argslist.device)

        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            logits, target = model(batch)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            train_loss[i] = loss.detach()

        return train_loss.mean()

    @torch.no_grad()
    def _update_queue(self, z_k: torch.FloatTensor):
        assert self.queue.size(0) == z_k.size(1)  # (f, K), (B, f)
        ptr = int(self.ptr)
        b = z_k.size(0)
        self.queue[:, ptr:ptr+b] = z_k.T
        new_ptr = (ptr + b) % self.queue.size(1)
        self.ptr[0] = new_ptr


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.n_features = input_dim
        self.output_dim = output_dim
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=argslist.dropout
        )
        self.rnn2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=output_dim,
            num_layers=1,
            batch_first=True,
            dropout=argslist.dropout
        )

    def forward(self, x, batch_size):
        x = x.reshape((batch_size, -1, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        return x.reshape((batch_size, -1, self.output_dim))


class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=argslist.dropout
        )
        self.rnn2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=argslist.dropout
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch_size):
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        x = x.reshape((-1, self.hidden_dim))
        x = self.output_layer(x)
        x = x.reshape((batch_size, -1, self.output_dim))
        return x


class RecurrentAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim,  embedding_dim).to(device)
        self.decoder = Decoder(embedding_dim, hidden_dim,  input_dim).to(device)

    def forward(self, x, batch_size):
        x = self.encoder(x, batch_size)
        dx = self.decoder(x, batch_size)
        return dx, x
