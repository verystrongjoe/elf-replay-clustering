import torch
import argslist
import os
import glob
import numpy as np


file_list = sorted(glob.glob(os.path.join(argslist.home_dir, "**/*.npz"), recursive=True))
data_list = [np.load(fname)['state'] for fname in file_list]
total = np.concatenate(data_list, axis=0)
print(f'shape : {total.shape} and type : {type(total)}')
total = torch.from_numpy(total)
print(f'shape : {total.shape} and type : {type(total)}')

for idx in range(22):
    embed = total[:, idx, :, :]
    embedding_size = torch.unique(embed).size().numel()
    print(f'{idx} embed size : {embedding_size}')

