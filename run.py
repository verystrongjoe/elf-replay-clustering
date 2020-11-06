from torch.utils.data import DataLoader
from my_dataset import CustomDataset
from ae import *
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import argslist
from torch.optim.lr_scheduler import LambdaLR
import math

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

writer = SummaryWriter(f'runs/{argslist.temperature}_{argslist.lr}_moco')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup & cosine decay.
       Implementation from `pytorch_transformers.optimization.WarmupCosineSchedule`.
       Assuming that the initial learning rate of `optimizer` is set to 1., this scheduler
       linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
       Decreases learning rate for 1. to 0. over remaining `t_total - warmup_steps` following a cosine curve.
       If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, min_lr=1e-4, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.min_lr = min_lr
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return self.min_lr + float(step) / float(max(1.0, self.warmup_steps))
        # Progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return self.min_lr + max(0., 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))



def main():
    moco = MoCo()
    # optimizer = torch.optim.Adam(moco.parameters(), lr=argslist.lr)
    optimizer = torch.optim.SGD(moco.parameters(), lr=argslist.lr, momentum=0.9)
    scheduler = WarmupCosineSchedule(optimizer, t_total=argslist.n_epoch, warmup_steps=0)

    train_set = CustomDataset(window_size=argslist.window_size)

    print(f'argslist.temperature : {argslist.temperature}')

    print('created dataset')
    train_loader = DataLoader(
        train_set,
        batch_size=argslist.batch_size,  # 256
        shuffle=True,
        num_workers=1,
        drop_last=True,
        pin_memory=True,
    )
    print('created dataloader')
    for e in range(argslist.n_epoch+1):
        epoch_loss = MoCo.train_epoch(model=moco, data_loader=train_loader, optimizer=optimizer)
        scheduler.step()
        print(f"Epoch [{e}/{argslist.n_epoch}]: {epoch_loss.item():.4f}")
        if e % 100 == 0: 
            print(f'{e} epoch trained.')
            torch.save(moco.state_dict(), f'moco-epoch_{experiment}_{e}.pt')
        writer.add_scalar('training_epoch_loss', epoch_loss, e)
    print('trained finished..')
    torch.save(moco.state_dict(), f'moco_final_{experiment}.pt')

if __name__ == '__main__':
    main()
