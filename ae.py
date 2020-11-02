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
