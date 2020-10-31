import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from deprecated.rnn_vae import RnnVae, RnnType, Parameters
from utils.my_dataset import CustomDataset
from argslist import *
from ae import *
import copy

seq_len = 22
n_features = 400

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
model = RecurrentAutoencoder(seq_len, n_features, 10)
model = model.to(device)

def train_model(model, train_dataset_iter, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  for epoch in range(1, n_epochs + 1):
    model = model.train()
    train_losses = []

    for seq_true in train_dataset_iter:
      seq_true = seq_true['input']
      optimizer.zero_grad()
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      loss = criterion(seq_pred, seq_true)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())

  max_loss = np.max(train_losses)
  losses_normalized = train_losses / max_loss
  return losses_normalized, model

def eval_model(model, val_dataset):
  criterion = nn.L1Loss(reduction='sum').to(device)
  best_loss = 10000.0
  val_losses = []
  model = model.eval()
  with torch.no_grad():
    for seq_true in val_dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      loss = criterion(seq_pred, seq_true)
      val_losses.append(loss.item())
  # train_loss = np.mean(train_losses)
  val_loss = np.mean(val_losses)
  # history['train'].append(train_loss)
  if val_loss < best_loss:
    best_loss = val_loss
    best_model_wts = copy.deepcopy(model.state_dict())
  # print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
  model.load_state_dict(best_model_wts)
  return val_loss, model.eval()


def make_batch(samples):
  inputs = [torch.tensor(sample) for sample in samples]
  padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
  return {'input': padded_inputs.contiguous()}


dataset = CustomDataset('model_45252', "simple", 50)
data_loader = DataLoader(dataset, batch_size=5, collate_fn=make_batch)
data_loader_iterator = iter(data_loader)

X_train_iter = DataLoader(dataset, batch_size=5, collate_fn=make_batch)

loss, model = train_model(
  model,
  X_train_iter,
  n_epochs=150
)

print(f'loss : {loss} ')