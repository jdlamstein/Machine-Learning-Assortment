import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import wandb

from pipeline import MILdataset

Date = datetime.now()
timestamp = f'{Date.year}_{Date.month}_{Date.day}_{Date.hour}_{Date.minute}_{Date.second}'


class Train:

    def run(self, args):
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        wandb.init(project="MIL")
        best_acc = 0
        wandb.config = {
            "timestamp": timestamp,
            "epochs": args.nepochs,
            "learning_rate": learning_rate,
            'weight_decay': weight_decay,
            "batch_size": args.batch_size,
            "train_dir": args.train_lib,
            "val_dir": args.val_lib
        }
        # cnn
        model = models.resnet34(True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.cuda()

        if args.weights == 0.5:
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            w = torch.Tensor([1 - args.weights, args.weights])
            criterion = nn.CrossEntropyLoss(w).cuda()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        cudnn.benchmark = True

        # normalization
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
        trans = transforms.Compose([transforms.ToTensor(), normalize])

        # load data
        train_dset = MILdataset(args.train_lib, trans)
        train_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)
        if args.val_lib:
            val_dset = MILdataset(args.val_lib, trans)
            val_loader = torch.utils.data.DataLoader(
                val_dset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)

        # open output file
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'w')
        fconv.write('epoch,metric,value\n')
        fconv.close()

        # loop throuh epochs
        for epoch in range(args.nepochs):
            train_dset.setmode(1)
            probs = self.inference(epoch, train_loader, model)
            topk = self.group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
            train_dset.maketraindata(topk)
            train_dset.shuffletraindata()
            train_dset.setmode(2)
            loss = self.train_one_epoch(train_loader, model, criterion, optimizer)
            wandb.log({'train_epoch_loss': loss})
            print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch + 1, args.nepochs, loss))
            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},loss,{}\n'.format(epoch + 1, loss))
            fconv.close()

            # Validation
            if args.val_lib and (epoch + 1) % args.test_every == 0:
                val_dset.setmode(1)
                probs = self.inference(epoch, val_loader, model)
                maxs = self.group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
                pred = [1 if x >= 0.5 else 0 for x in maxs]
                err, fpr, fnr = self.calc_err(pred, val_dset.targets)
                acc = 1 - (fpr + fnr) / 2.
                wandb.log({'val_err': err,
                           'val_fpr': fpr,
                           'val_fnr': fnr,
                           'val_acc': acc})
                print(
                    'Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch + 1, args.nepochs, err, fpr,
                                                                                     fnr))
                fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
                fconv.write('{},error,{}\n'.format(epoch + 1, err))
                fconv.write('{},fpr,{}\n'.format(epoch + 1, fpr))
                fconv.write('{},fnr,{}\n'.format(epoch + 1, fnr))
                fconv.close()
                # Save best model
                err = (fpr + fnr) / 2.
                if 1 - err >= best_acc:
                    best_acc = 1 - err
                    obj = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(obj, os.path.join(args.output, 'checkpoint_best.pth'))
        wandb.finish()

    def inference(self, run, loader, model):
        model.eval()
        probs = torch.FloatTensor(len(loader.dataset))
        with torch.no_grad():
            for i, input in enumerate(loader):
                if not i % 10:
                    print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run + 1, args.nepochs, i + 1, len(loader)))
                input = input.cuda()
                output = F.softmax(model(input), dim=1)
                probs[i * args.batch_size:i * args.batch_size + input.size(0)] = output.detach()[:, 1].clone()
        return probs.cpu().numpy()

    def train_one_epoch(self, loader, model, criterion, optimizer):
        model.train()
        running_loss = 0.
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            wandb.log({"train_batch_loss": loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input.size(0)
        return running_loss / len(loader.dataset)

    def calc_err(self, pred, real):
        pred = np.array(pred)
        real = np.array(real)
        neq = np.not_equal(pred, real)
        err = float(neq.sum()) / pred.shape[0]
        fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum()
        fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum()
        return err, fpr, fnr

    def group_argtopk(self, groups, data, k=1):
        order = np.lexsort((data, groups))
        groups = groups[order]
        data = data[order]
        index = np.empty(len(groups), 'bool')
        index[-k:] = True
        index[:-k] = groups[k:] != groups[:-k]
        return list(order[index])

    def group_max(self, groups, data, nmax):
        out = np.empty(nmax)
        out[:] = np.nan
        order = np.lexsort((data, groups))
        groups = groups[order]
        data = data[order]
        index = np.empty(len(groups), 'bool')
        index[-1] = True
        index[:-1] = groups[1:] != groups[:-1]
        out[groups[index]] = data[index]
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
    parser.add_argument('--train_lib', type=str,
                        help='path to train MIL library binary')
    parser.add_argument('--val_lib', type=str,
                        help='path to validation MIL library binary. If present.')
    parser.add_argument('--output', type=str,
                        help='name of output file')
    parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for adam optimizer')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_every', default=1, type=int, help='test on val every (default: 10)')
    parser.add_argument('--weights', default=0.5, type=float,
                        help='unbalanced positive class weight (default: 0.5, balanced classes)')
    parser.add_argument('--k', default=1, type=int,
                        help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
    args = parser.parse_args()
    Tr = Train()
    Tr.run(args)
