import torch.utils.data as data
import torch
import sys
import os
import openslide
import numpy as np
from glob import glob
from PIL import Image
import random

class MILdataset(data.Dataset):
    """Dataset for handling library files of digital pathology slides"""
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        slides = []
        for i, name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i + 1, len(lib['slides'])))
            sys.stdout.flush()
            assert os.path.exists(name), f'{name} DNE'

            slides.append(openslide.OpenSlide(name))
        print('')
        # Flatten grid
        grid = []
        slideIDX = []
        for i, g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i] * len(g))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        print('Number of slides: {}'.format(len(self.slidenames)))
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.mult = lib['mult']
        self.size = int(np.round(224 * lib['mult']))
        self.level = lib['level']

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]]) for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self, index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].read_region(coord, self.level, (self.size, self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224, 224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord, self.level, (self.size, self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224, 224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

class CropMILDataset(data.Dataset):
    """Dataset Object for image directories for multiple instance learning
    Directory and label structure:
    --imagedir
    ----class 1
        --img1
        --img2
    ----class 2
        --img3
        --img4
    ----class 3
        --img5
        --img6"""
    def __init__(self, imagedir, transform=None):
        self.targets = []
        self.files = []
        class_directories = glob(os.path.join(imagedir, '*'))
        for cls in class_directories:
            files = os.path.join(cls, '*.png')
            self.targets.extend([cls.split('/')[-1]] * len(files))
            self.files.extend(files)
        self.transform = transform

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        self.t_data = [(self.files[x], self.targets[x]) for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self, index):
        f, target = self.t_data[index]
        img = Image.open(f)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.t_data)