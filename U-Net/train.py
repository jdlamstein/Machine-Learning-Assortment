import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from util import check_accuracy, save_predictions_as_imgs
from unet import UNET
from pipeline import FileDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from param_unet import Param
import wandb
import datetime
import argparse
now = datetime.datetime.now()

timestamp = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Train:
    def __init__(self, p, root_dir, use_wandb=True):
        self.p = p
        data_dir = os.path.join(root_dir, 'data')
        self.fig_dir = os.path.join(root_dir, 'figures', 'training')
        self.train_dir = dict(images=os.path.join(data_dir, 'train', 'images'),
                              masks=os.path.join(data_dir, 'train', 'masks'))
        self.val_dir = dict(images=os.path.join(data_dir, 'val', 'images'),
                            masks=os.path.join(data_dir, 'val', 'masks'))
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        if use_wandb:
            self.wand = wandb.init(
                project="segmentation",
                config={
                    "learning_rate": self.p.learning_rate,
                    "epochs": self.p.epochs,
                    "batch_size": self.p.batch_size,
                    "timestamp": timestamp
                })

    def train_one_epoch(self, loader, model, optimizer, loss_fn, scaler):
        loop = tqdm(loader)

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
            wandb.log({"train_loss": loss})

    def run(self):
        train_dataset = FileDataset(self.train_dir['images'], mask_dir=self.train_dir['masks'],
                                           transform=A.Compose([
                                               A.HorizontalFlip(0.5),
                                               A.VerticalFlip(0.5),
                                               ToTensorV2()]))
        val_dataset = FileDataset(self.val_dir['images'], mask_dir=self.val_dir['masks'],
                                         transform=A.Compose([ToTensorV2()]))
        model = UNET(in_channels=1, out_channels=1).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.p.learning_rate)
        train_loader = DataLoader(train_dataset, batch_size=self.p.batch_size, shuffle=True,
                                  num_workers=self.p.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.p.batch_size, shuffle=False,
                                num_workers=self.p.num_workers, pin_memory=True)

        check_accuracy(val_loader, model, device=DEVICE)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.p.epochs):
            print(f'Epoch: {epoch}')
            self.train_one_epoch(train_loader, model, optimizer, criterion, scaler)
            val_acc, dice_score = check_accuracy(val_loader, model, device=DEVICE)
            wandb.log({"val_accuracy": val_acc, "val_dice": dice_score})

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        ckpt_filename = f"unet_checkpoint_{timestamp}.pth.tar"
        wandb.log({"cpkt_filename": ckpt_filename})

        torch.save(checkpoint, ckpt_filename)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, self.fig_dir, device=DEVICE
        )
        wandb.finish()


if __name__ == "__main__":
    p = Param()
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, help='Path to project directory.')
    args = parser.parse_args()
    Tr = Train(p, root_dir=args.project_dir)
    Tr.run()
