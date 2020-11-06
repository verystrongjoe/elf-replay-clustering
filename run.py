from torch.utils.data import DataLoader
from my_dataset import CustomDataset
from ae import *
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import argslist



parser = argparse.ArgumentParser()

parser.add_argument('t', type=float, help='moco temperature')
parser.add_argument('lr', type=float, help='learning rate')
parser.add_argument('num_gpu', type=int, help='num of gpu be used')

args = parser.parse_args()

argslist.temperature = args.t
argslist.lr = args.lr
num_gpu = args.num_gpu


# # GPU 할당 변경하기
GPU_NUM = 1  # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
print('Current cuda device ', torch.cuda.current_device())  # check
argslist.device = device


writer = SummaryWriter(f'runs/{}_{}_moco')


# Additional Infos
if device.type == 'cuda':
    print(torch.cuda.get_device_name(GPU_NUM))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')

# # GPU 할당 변경하기
# GPU_NUM = 0  # 원하는 GPU 번호 입력
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device)  # change allocation of current GPU
# print('Current cuda device ', torch.cuda.current_device())  # check
#
# # Additional Infos
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(GPU_NUM))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')


def main():
    moco = MoCo()
    optimizer = torch.optim.Adam(moco.parameters(), lr=argslist.lr)
    train_set = CustomDataset(window_size=argslist.window_size)

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
        print(f"Epoch [{e}/{argslist.n_epoch}]: {epoch_loss.item():.4f}")
        writer.add_scalar('training_epoch_loss', epoch_loss, e)
        if e % 100 == 0: 
            print(f'{e} epoch trained.')
            torch.save(moco.state_dict(), f'moco-epoch_{e}.pt')
    print('trained finished..')
    torch.save(moco.state_dict(), 'moco_final.pt')

if __name__ == '__main__':
    main()
