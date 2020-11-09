from torch.utils.data import DataLoader
from my_dataset import CustomDataset
from ae import *
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import argslist
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('t', type=float, help='moco temperature')
parser.add_argument('lr', type=float, help='learning rate')
parser.add_argument('num_gpu', type=int, help='num of gpu be used')

args = parser.parse_args()
argslist.temperature = args.t
argslist.lr = args.lr
num_gpu = args.num_gpu
experiment = f"moco_{args.t}_{args.lr}_{args.num_gpu}"

# # GPU 할당 변경하기
GPU_NUM = num_gpu  # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
print('Current cuda device ', torch.cuda.current_device())  # check
argslist.device = device

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

print(f'loading pickle file is finished.')
datalist = np.load('list.dat.npy')

print(f'loading {len(self.datalist)} npz files is finished.')

model_fn = 'moco-epoch_moco_0.05_0.1_1_2600.pt'
moco = MoCo()
model_dic = torch.load(model_fn)
moco.load_state_dict(model_dic)
moco.to(device)


feature_list = []

for idx in range(len(datalist)):
    sample = datalist[idx]
    replay_name = os.path.basename(self.filelist[idx]).split('.')[0]
    try:
        model_name, scenario, tmp, frame_skip, _ = replay_name.split('_')
        scenario = '-'.join([scenario, tmp])  # hit, run -> hit-run
    except ValueError:
        model_name, scenario, frame_skip, _ = replay_name.split('_')
    len = sample.shape[0]
    first = 0
    middle = int(len / 2)
    last = len

    idx_1 = random.randint(first, middle - argslist.window_size)
    idx_2 = random.randint(middle, last - argslist.window_size)

    pos_1 = sample[idx_1:idx_1 + self.window_size, [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19]]  # (T, C, h, w); C=channels
    pos_2 = sample[idx_2:idx_2 + self.window_size, [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19]]  # (T, C, h, w); T=window_size
    pos_1 = torch.from_numpy(pos_1)
    pos_2 = torch.from_numpy(pos_2)

    assert pos_1.shape[0] == pos_2.shape[0]

    data = dict(pos_1=pos_1, pos_2=pos_2, model_name=model_name, scenario=scenario, frame_skip=frame_skip)

    feature = moco.forward(data)

    feature_list.append(feature)
    print(f'{idx} sample got feature.')

with open('fetures.dat', 'wb') as f:
    f.write(feature_list)


