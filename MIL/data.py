"""Save data as torch dict"""

import torch
import os
from glob import glob

"""Process slides to crops and save or generate"""

import openslide
import os
from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import random


class SlideData:
    def __init__(self, datadir, savedir, label_csv, plotbool, token='png', debug_count=None):
        """
        Slices whole slide images to crops. Avoids background with otsu thresholding on thumbnail
        :param datadir: directory with whole slide image
        :param savedir: save directory
        :param plotbool: to plot or not to plot
        :param token: image type
        :param debug_count: None for all. Set to integer for cropping only that number of slices
        """
        self.datadir = datadir
        self.savedir = savedir
        self.files = []
        self.slidenames = []
        self.labels = []
        self.boundaries = {}
        self.h = 0
        self.w = 0
        self.wsmall = 0
        self.hsmall = 0
        self.sf = 16
        self.level = 1
        self.area_limit = 30000
        self.target_sf = {0:1, 1:4, 2:16}
        self.target_sf= self.target_sf[self.level]
        self.debug_count = debug_count
        self.label_df = pd.read_csv(label_csv)
        self.densities = []
        for i in range(4):
            self.densities.append(self.label_df[self.label_df['Density.Score3'] == i])
        # sample_size = 9
        # for i in range(1, 4):
        #     self.densities[i] = self.densities[i][:sample_size // 3]
        if self.debug_count is not None:
            self.densities[3] = self.densities[3][:self.debug_count]
        self.densities[0] = self.densities[0][:len(self.densities[3])]
        for i, df in enumerate(self.densities):
            if i==0 or i==3:
                for j, row in df.iterrows():
                    self.labels.append(1 if i > 0 else 0)
                    self.slidenames.append(row['Consensus Filepaths'])

        for s in self.slidenames:
            f = os.path.join(datadir, s + f'.{token}')
            assert os.path.exists(f), f'{f} path DNE'
            self.files.append(f)

        self.N = len(self.files)
        print(f'{self.N} slides')

        random.seed(11)
        random.shuffle(self.files)
        random.seed(11)
        random.shuffle(self.labels)

        self.split = [.7, .85, 1.]


        self.cropsize = 224
        self.plot = plotbool
        self.points = []
        self.train = {'slides': [], 'grid': [], 'targets': [], 'mult': [], 'level': []}
        self.val = {'slides': [], 'grid': [], 'targets': [], 'mult': [], 'level': []}
        self.test = {'slides': [], 'grid': [], 'targets': [], 'mult': [], 'level': []}
        self.phase = {'train':self.train, 'val':self.val, 'test':self.test}

    def run(self):
        for i, (label, file) in enumerate(zip(self.labels, self.files)):
            if i < self.N * self.split[0]:
                self.readSlide(label, file, 'train')
            elif np.floor(self.N * self.split[0]) <= i < self.N * self.split[1]:
                self.readSlide(label, file, 'val')
            else:
                self.readSlide(label, file, 'test')

        # save dict as pytorch
        # for _, d in self.phase.items():
        #     for key in d:
        #         d[key] = np.array(d[key])
        torch.save(self.train, os.path.join(self.savedir, 'train.pt'))
        torch.save(self.val, os.path.join(self.savedir, 'val.pt'))
        torch.save(self.test, os.path.join(self.savedir, 'test.pt'))
        print('Done.')


        # read slide at low res
        # get boundaries of each slide, store in dict
        # expand boundaries to original res
        # sample crops in grid, no overlap

    def readSlide(self, label, file, phase_str):
        slide = openslide.OpenSlide(file)
        self.w, self.h = slide.dimensions
        self.wsmall, self.hsmall = self.w // self.sf, self.h // self.sf
        thumb = slide.get_thumbnail((self.wsmall, self.hsmall))
        contours = self.get_boundary(thumb)  # get contour of thumbnail
        points = self.get_points_in_contours(contours, self.hsmall, self.wsmall,
                                             step=self.target_sf * self.cropsize // self.sf)  # get points

        if self.plot:
            self.crop_slide(slide, points, file)  # crop slide within contour
        # save points in dict

        self.phase[phase_str]['slides'].append(file)
        self.phase[phase_str]['grid'].append(points)
        self.phase[phase_str]['targets'].append(label)
        self.phase[phase_str]['level'] = 0  # 0 is highest level
        self.phase[phase_str]['mult'] = 1  # 1 for no scaling


    def crop_slide(self, slide, points, file):
        """Crop slide from points with cropsize"""
        # name = file.split('/')[-1].split('.')[0]
        # savesubdir = os.path.join(self.savedir, name)

        # if not os.path.exists(savesubdir):
        #     os.makedirs(savesubdir, exist_ok=False)
        for i, (x, y) in enumerate(points):
            # savecroppath = os.path.join(savesubdir, f'{name}_x_{x}_y_{y}.png')
            # row is y, col is x
            if y + self.cropsize < self.h and x + self.cropsize < self.w:
                crop = slide.read_region((x, y), level=self.level, size=(self.cropsize, self.cropsize))
                plt.imshow(crop)
                plt.title(f'crop with level: {self.level} at x:{x} y:{y}')
                plt.show()

    def get_boundary(self, img):
        """
        img: PIL image
        """

        rgb = np.array(img)
        # if self.plot:
        #     plt.imshow(rgb)
        #     plt.title('thumbnail')
        #     plt.show()
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        ret,thresh = cv2.threshold(grey,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        if self.plot:
            plt.imshow(thresh)
            plt.title('otsu')
            plt.show()
        thresh = cv2.dilate(thresh, kernel=np.ones((5,5,), np.uint8))
        thresh = cv2.dilate(thresh, kernel=np.ones((5,5,), np.uint8))
        # hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # low_H = 50
        # low_S = 0
        # low_V = 0
        # high_H = 180
        # high_S = 255
        # high_V = 255
        # frame_threshold = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
        # if self.plot:
        #     plt.imshow(frame_threshold)
        #     plt.title('frame threshold')
        #
        #     plt.show()
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        ctrs = []
        for cc in contours:
            if cv2.contourArea(cc) > self.area_limit:
                ctrs.append(cc)
                # areas.append(cv2.contourArea(cc))
        areas = sorted(areas, reverse=True)
        # c = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(rgb, ctrs,-1, (0, 255, 0), 2)
        # cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if self.plot:
            plt.imshow(rgb)
            plt.title('Contour')

            plt.show()
        # frame_threshold /= 255
        # loop through frame threhsold
        return ctrs

    def get_points_in_contours(self, contours, h, w, step):
        for x in range(0, w, step):
            for y in range(0, h, step):
                for contour in contours:
                    inside = cv2.pointPolygonTest(contour, (x, y), False)
                    if inside >= 0:
                        self.points.append((x * self.sf, y * self.sf))  # points time scale factor
        print(f'Collected {len(self.points)} points')
        return self.points


if __name__ == '__main__':
    plotbool = False
    datadir = '/gladstone/finkbeiner/linsley/MJFOX/Submandibular_Asyn_Images'
    savedir = '/gladstone/finkbeiner/linsley/MIL_MJFOX'
    label_csv = '/gladstone/finkbeiner/linsley/josh/MJFOX/Random_Josh_Data_almost_all.csv'
    SD = SlideData(datadir, savedir, label_csv, plotbool, 'svs', debug_count=None)
    SD.run()
