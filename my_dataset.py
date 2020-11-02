import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import warnings
warnings.filterwarnings("ignore")
import argslist


class CustomDataset(Dataset):

    def __init__(self):
        self.root_dir = argslist.home_dir + "/features"
        self.datalist = os.listdir(self.root_dir)
        print(f"{len(self.datalist)} files is loaded..")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        sample = np.load(os.path.join(self.root_dir, self.datalist[idx]))

        type = 0

        splits = self.datalist[idx].split('_')
        if len(splits) == 5:
            if splits[3] == '20':
                type = 1
            elif splits[3] == '50':
                type = 2
            elif splits[3] == '80':
                type = 3
        elif splits[1] == 'simple':
            if splits[2] == '20':
                type = 4
            elif splits[2] == '50':
                type = 5
            elif splits[2] == '80':
                type = 6

        return torch.tensor(sample['state']).sum(axis=1).flatten(start_dim=1), torch.tensor(type)