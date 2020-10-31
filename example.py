import torch
import torch.nn as nn
import numpy as np

np.random.seed(123)
batch_data = ["I love Mom ' s cooking", "I love you too !", "No way", "This is the shit", "Yes"]
input_seq = [s.split() for s in batch_data]
max_len = 0

for s in input_seq:
    if len(s) >= max_len:
        max_len = len(s)

vocab = {w:i for i, w in enumerate(set([t for s in input_seq for t in s]), 1)}
vocab["<pad>"] = 0

input_seq = [s + ["<pad>"] *(max_len - len(s)) if len(s) < max_len else s for s in input_seq]
input_seq2idx = torch.LongTensor([list(map(vocab.get, s)) for s in input_seq])

from torch.nn.utils.rnn import pack_padded_sequence

input_lengths = torch.LongTensor(
    [torch.max(input_seq2idx[i, :].data.nonzero())+1 for i in range(input_seq2idx.size(0))]
)

input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
input_seq2idx = input_seq2idx[sorted_idx]

packed_input = pack_padded_sequence(input_seq2idx, input_lengths.tolist(), batch_first=True)

print(type(packed_input))
print(packed_input[0])  # packed data
print(packed_input[1])  # batch_sizes

vocab_size = len(vocab)
hidden_size = 1
embedding_size = 5
num_layers = 3

embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
gru = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
             bidirectional=False, batch_first=True)

embeded = embed(input_seq2idx)
packed_input = pack_padded_sequence(embeded, input_lengths.tolist(), batch_first=True)
packed_output, hidden = gru(packed_input)

packed_output[0].size(), packed_output[1]

from torch.nn.utils.rnn import pad_packed_sequence
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
output.size(), output_lengths

packed_output[0]
output


import pandas as pd

def color_white(val):
    color = 'white' if val == 0 else 'black'
    return 'color: {}'.format(color)
def color_red(data):
    max_len = len(data)
    fmt = 'color: red'
    lst = []
    for i, v in enumerate(data):
        if (v != 0) and (i == max_len-1):
            lst.append(fmt)
        elif (v != 0) and (data[i+1] == 0):
            lst.append(fmt)
        else:
            lst.append('')
    return lst

df = pd.DataFrame(
    np.concatenate(
        [o.detach().numpy() for o in output.transpose(0,1)]
        , axis=1
    ).round(4)
)

df.index.name = 'batch'
df.columns.name = 'hidden_step'
df.style.applymap(color_white).apply(color_red, axis=1)

hidden[-1]

packed_output[0], packed_output[1]

