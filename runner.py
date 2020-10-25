import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from rnn_vae import RnnVae

use_cuda = torch.cuda.is_available()
use_cuda = True
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

path = f'{home_dir}/data/ml-toolkit/pytorch-models/rnn-ae/'

params = { 'rnn_type': RnnType.LSTM,
           'rnn_hidden_dim': 512,
           'num_layers': 1,
           'bidirectional_encoder': True,
           'dropout': 0.0,
           'embed_dim': 300,
           'clip': 0.5,
           'encoder_lr': 0.001,
           'decoder_lr': 0.001,
           'linear_dims': [],
           'z_dim': 1024}

print(params)
with open(path+'params.json', 'w') as outfile:
    json.dump(params, outfile)

params = Parameters(params)

criterion = nn.NLLLoss()
rnn_ae = RnnAE(device, params, criterion)

print(rnn_ae.encoder)
print(rnn_ae.decoder)

losses = []

num_epochs = 100
safe_after_epoch = False

encoder_file_name = '{}/data/ml-toolkit/pytorch-models/rnn-ae/rnnae-encoder.model'.format(home_dir)
decoder_file_name = '{}/data/ml-toolkit/pytorch-models/rnn-ae/rnnae-decoder.model'.format(home_dir)

rnn_ae.train()

dataset = CustomDataset('model_45252', "simple", 50)
data_loader = DataLoader(dataset, batch_size=5, collate_fn=make_batch)
data_loader_iterator = iter(data_loader)

for idx in range(5):
    sample = data_loader_iterator.next()
    print(f"{idx} shape =  {sample['input'].shape}")

def make_batch(samples):
    inputs = [torch.tensor(sample) for sample in samples]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return {'input': padded_inputs.contiguous()}


X_train_iter = DataLoader(dataset, batch_size=5, collate_fn=make_batch)

# rnn_ae.set_learning_rates(0.001, 0.001)
for epoch in range(num_epochs):
    epoch_loss = rnn_ae.train_epoch(epoch, X_train_iter, verbatim=True)
    print(epoch_loss)
    losses.append(epoch_loss)
    if safe_after_epoch:
        rnn_ae.save_models(encoder_file_name, decoder_file_name)
    rnn_ae.update_learning_rates(0.99, 0.99)

rnn_ae.eval()

max_loss = np.max(losses)
losses_normalized = losses / max_loss

plt.plot(losses_normalized, label='loss')
plt.legend(loc='upper right')
plt.ylabel('RNN-AE (e_dim={}, h_dim={})'.format(params.embed_dim, params.rnn_hidden_dim))
plt.show()


def check_sequence(sequence, model, vectorizer, max_length=100):
    original_sequence = vectorizer.sequence_to_text(sequence)
    X = torch.tensor([sequence], dtype=torch.long).to(model.device)
    decoded_indices = model.evaluate(X)
    decoded_sequence = vectorizer.sequence_to_text(decoded_indices)
    return ' '.join(original_sequence), ' '.join(decoded_sequence)


print(check_sequence(X_train[0], text_rnn_ae, vectorizer, max_length=50))


for idx, s in enumerate(X_train):
    original, decoded = check_sequence(s, rnn_ae, vectorizer)
    print("================================================")
    print()
    print(original)
    print(">>>")
    print(decoded)
    print()
    if idx > 200:
        break


