import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

# define input sequence
sequence = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# reshape input into [samples, timesteps, features]
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))

class AutoEncoderDecoder(nn.Module):
    def __init__(self, input_shape, num_layers, lstm_hidden=100, activation='relu', bidirectional=True):
        self.input_shape = input_shape
        self.lstm_hidden = lstm_hidden
        self.activation = activation
        self.lstm1 = nn.LSTM(hidden_size=lstm_hidden, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size=lstm_hidden, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        encoder_output = self.lstm1(x)
        encoder_output


# define model
model = models.Sequential()
model.add(layers.LSTM(100, activation='relu', input_shape=(n_in, 1)))
model.add(layers.RepeatVector(n_in))
model.add(layers.LSTM(100, activation='relu', return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(1)))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(sequence, sequence, epochs=300, verbose=0)

# predict
yhat = model.predict(sequence)
yhat