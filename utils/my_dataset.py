import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import warnings
warnings.filterwarnings("ignore")

base_dir = "D:/workspace/clustering/"

class CustomDataset(Dataset):

    def __init__(self, model_name, style, frame_skip):
        self.root_dir = base_dir + f"{model_name}/save_npz/{style}/{frame_skip}/"
        self.datalist = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        sample = np.load(os.path.join(self.root_dir, self.datalist[idx]))
        return torch.tensor(sample['state'].sum(1))