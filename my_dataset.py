import os
import glob
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
import argslist
import pickle

class CustomDataset(Dataset):

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.root_dir = os.path.join(argslist.home_dir, 'features')
        self.filelist = sorted(glob.glob(os.path.join(self.root_dir, "**/*.npz"), recursive=True))
        self.datalist = []

        # for i in range(80):
        #     self.datalist.append(np.random.random((70, 22, 20, 20)))

        if not os.path.isfile('list.data.npy'):
            for f in self.filelist:
                sample = np.load(f)['state']  # TODO: add action here
                if sample.shape[0] > argslist.sample_len_threshold:
                    self.datalist.append(sample)
                if len(self.datalist) > 100:
                    continue
            print(f'loading {len(self.datalist)} npz files is finished.')
            d = np.concatenate(self.datalist, axis=0)
            np.save('list.dat', d)
        else:
            self.datalist = np.load('list.dat.npy')
            print(f'loading {len(self.datalist)} npz files is finished.')

    def __getitem__(self, idx):
        sample = self.datalist[idx]
        replay_name = os.path.basename(self.filelist[idx]).split('.')[0]
        try:
            model_name, scenario, tmp, frame_skip, _ = replay_name.split('_')
            scenario = '-'.join([scenario, tmp])  # hit, run -> hit-run
        except ValueError:
            model_name, scenario, frame_skip, _ = replay_name.split('_')

        # model_name = random.choice(['9996', '45552'])
        # scenario = random.choice(['simple', 'hit-run'])
        # frame_skip = random.choice(['20', '50', '60'])

        #pos_1 = sample[argslist.sampling_tuple_idx_1:argslist.sampling_tuple_idx_1+self.window_size]  # (T, C, h, w); C=channels
        #pos_2 = sample[argslist.sampling_tuple_idx_2:argslist.sampling_tuple_idx_2+self.window_size]  # (T, C, h, w); T=window_size

        len = sample.shape[0]
        first = 0
        middle = int(len/2)
        last = len

        idx_1 = random.randint(first, middle - argslist.window_size)
        idx_2 = random.randint(middle, last - argslist.window_size)

        pos_1 = sample[idx_1:idx_1+self.window_size, [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19]]  # (T, C, h, w); C=channels
        pos_2 = sample[idx_2:idx_2+self.window_size, [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19]]  # (T, C, h, w); T=window_size
        pos_1 = torch.from_numpy(pos_1)
        pos_2 = torch.from_numpy(pos_2)

        assert pos_1.shape[0] == pos_2.shape[0]

        return dict(pos_1=pos_1, pos_2=pos_2, model_name=model_name, scenario=scenario, frame_skip=frame_skip)

    def _dry_run(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.datalist)
