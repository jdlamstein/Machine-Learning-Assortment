import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model import Discriminator, Generator, weights_init

class Train:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self, args):
        try:
            os.makedirs(args.outf)
        except OSError:
            pass

        if args.manualSeed is None:
            args.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", args.manualSeed)
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)

        cudnn.benchmark = True
        if torch.cuda.is_available() and not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        if torch.backends.mps.is_available() and not args.mps:
            print("WARNING: You have mps device, to enable macOS GPU run with --mps")

        if args.dataroot is None and str(args.dataset).lower() != 'fake':
            raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % args.dataset)

        if args.dataset in ['imagenet', 'folder', 'lfw']:
            # folder dataset
            dataset = dset.ImageFolder(root=args.dataroot,
                                       transform=transforms.Compose([
                                           transforms.Resize(args.imageSize),
                                           transforms.CenterCrop(args.imageSize),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
            nc = 3
        elif args.dataset == 'lsun':
            classes = [c + '_train' for c in args.classes.split(',')]
            dataset = dset.LSUN(root=args.dataroot, classes=classes,
                                transform=transforms.Compose([
                                    transforms.Resize(args.imageSize),
                                    transforms.CenterCrop(args.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
            nc = 3
        elif args.dataset == 'cifar10':
            dataset = dset.CIFAR10(root=args.dataroot, download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(args.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
            nc = 3

        elif args.dataset == 'mnist':
            dataset = dset.MNIST(root=args.dataroot, download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(args.imageSize),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,)),
                                 ]))
            nc = 1

        elif args.dataset == 'fake':
            dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                                    transform=transforms.ToTensor())
            nc = 3
        return dataset, nc



    def run(self, opt):
        dataset, nc = self.setup(opt)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                                 shuffle=True, num_workers=int(opt.workers))
        netG = Generator(opt.ngpu, opt.nz, nc, opt.ngf).to(self.device)
        netG.apply(weights_init)
        if opt.netG != '':
            netG.load_state_dict(torch.load(opt.netG))
        print(netG)
        netD = Discriminator(opt.ngpu, nc, opt.ndf).to(self.device)
        netD.apply(weights_init)
        if opt.netD != '':
            netD.load_state_dict(torch.load(opt.netD))
        print(netD)

        criterion = nn.BCELoss()

        fixed_noise = torch.randn(opt.batchSize, opt.nz, 1, 1, device=self.device)
        real_label = 1
        fake_label = 0

        # setup optimizer
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        if opt.dry_run:
            opt.niter = 1

        for epoch in range(opt.niter):
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                netD.zero_grad()
                real_cpu = data[0].to(self.device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label,
                                   dtype=real_cpu.dtype, device=self.device)

                output = netD(real_cpu)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, opt.nz, 1, 1, device=self.device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                output = netD(fake)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, opt.niter, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                                      '%s/real_samples.png' % opt.outf,
                                      normalize=True)
                    fake = netG(fixed_noise)
                    vutils.save_image(fake.detach(),
                                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                                      normalize=True)

                if opt.dry_run:
                    break
            # do checkpointing
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='folder', help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', default='/Users/gandalf/CelebAMask-HQ', required=False, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
    parser.add_argument('--mps', action='store_true', default=True, help='enables macOS GPU training')
    args = parser.parse_args()
    print(f'args {args}')
    Tr = Train()
    Tr.run(args)