import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import warnings
warnings.filterwarnings("ignore")
from argslist import *


class CustomDataset(Dataset):

    def __init__(self, model_names=["model_45252","model_9996"], styles=["hit_run","simple"], frame_skips=["20", "50","80"]):
        self.datalist = []
        for m in model_names:
            for s in styles:
                for f in frame_skips:
                    self.root_dir = home_dir + f"/{m}/save_npz/{s}/{f}/"
                    self.datalist.extend(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        sample = np.load(os.path.join(self.root_dir, self.datalist[idx]))
        # return torch.tensor(sample['state'].sum(1))
        return torch.tensor(sample['state']).flatten(start_dim=1)