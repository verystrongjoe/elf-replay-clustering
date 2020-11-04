from torch.utils.data import DataLoader
from my_dataset import CustomDataset
from ae import *
import argslist
import logging


def main():
    moco = MoCo()
    optimizer = torch.optim.Adam(moco.parameters(), lr=argslist.lr)
    train_set = CustomDataset(window_size=argslist.window_size)
    train_loader = DataLoader(
        train_set,
        batch_size=argslist.batch_size,  # 256
        shuffle=True,
        num_workers=1,
        drop_last=True,
        pin_memory=True,
    )
    for e in range(argslist.n_epoch):
        epoch_loss = MoCo.train_epoch(model=moco, data_loader=train_loader, optimizer=optimizer)
        logging.info(f"Epoch [{e}/{argslist.n_epoch}]: {epoch_loss.item():.4f}")


if __name__ == '__main__':
    main()
