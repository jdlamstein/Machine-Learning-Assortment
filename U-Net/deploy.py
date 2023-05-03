"""Deploy models"""
import torch
import torch.optim as optim
from unet import UNET
from util import check_accuracy, save_predictions_as_imgs
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pipeline import FileDataset
from param_unet import Param
import imageio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
import numpy as np
import argparse
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Deploy:
    def __init__(self, p, root_dir):
        self.p = p
        self.root_dir = root_dir
        data_dir = os.path.join(root_dir, 'data')
        self.fig_dir = os.path.join(root_dir, 'figures')

        self.val_dir = dict(images=os.path.join(data_dir, 'val', 'images'),
                            masks=os.path.join(data_dir, 'val', 'masks'))
        self.locations = dict(frame_number=[], centroid_x=[], centroid_y=[])
        self.val_locations = dict(frame_number=[], centroid_x=[], centroid_y=[], gt_centroid_x=[], gt_centroid_y=[],
                                  distance=[])

    def localize_squares(self, img, frame_number):
        """Get centroids of detected squares"""
        labelled_image = measure.label(img, background=0)
        # plt.imshow(labelled_image, cmap='nipy_spectral')
        # plt.show()
        regions = measure.regionprops_table(labelled_image, properties=('centroid',))
        print('Centroids:')
        for y0, x0 in zip(regions['centroid-0'], regions['centroid-1']):
            self.locations['frame_number'].append(frame_number)
            self.locations['centroid_x'].append(x0)
            self.locations['centroid_y'].append(y0)
            print(f'x: {x0:.2f}, y: {y0:.2f}')

    def check_locations(self, loader, model, device="cuda"):
        """Compare centroids of ground truth masks to prediction centroids."""
        for frame_num, (x, y) in enumerate(loader):
            # batch size needs to be same size as number of crops per image
            print('frame num: ', frame_num)
            x = x.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            np_preds = preds.cpu().detach().numpy()
            np_ys = y.cpu().detach().numpy() * 255
            np_preds = np_preds.squeeze() * 255

            for tile, (np_pred, np_y) in enumerate(zip(np_preds, np_ys)):
                labelled_pred = measure.label(np_pred > 100, background=0)
                labelled_y = measure.label(np_y > 100, background=0)
                lbls = np.unique(labelled_pred)
                lbls = [i for i in lbls if i > 0]
                lbl_mapping = {}
                lbl_mapping_check = []
                for lbl in lbls:
                    found_lbls = np.unique((labelled_pred == lbl) * labelled_y)
                    found_lbls = [i for i in found_lbls if i > 0]
                    if len(found_lbls) == 1:
                        if found_lbls[0] not in lbl_mapping_check:  #
                            lbl_mapping[lbl] = found_lbls[0]
                            lbl_mapping_check.append(found_lbls[0])
                for lbl, found_lbl in lbl_mapping.items():
                    regions_pred = measure.regionprops_table(lbl * (labelled_pred == lbl), properties=('centroid',))
                    regions_y = measure.regionprops_table(lbl * (labelled_y == found_lbl), properties=('centroid',))
                    for y0, x0, gt_y0, gt_x0 in zip(regions_pred['centroid-0'], regions_pred['centroid-1'],
                                                    regions_y['centroid-0'], regions_y['centroid-1']):
                        self.val_locations['frame_number'].append(f'{frame_num}_{tile}')
                        self.val_locations['centroid_x'].append(x0)
                        self.val_locations['gt_centroid_x'].append(gt_x0)
                        self.val_locations['centroid_y'].append(y0)
                        self.val_locations['gt_centroid_y'].append(gt_y0)
                        dist = np.sqrt((x0 - gt_x0) ** 2 + (y0 - gt_y0) ** 2)
                        if dist > 10:
                            plt.figure()
                            plt.imshow(labelled_pred == lbl)
                            plt.figure()
                            plt.imshow(labelled_y == found_lbl)
                            plt.show()
                        self.val_locations['distance'].append(dist)

    def run(self, savebool=False):
        # Load models
        model = UNET(in_channels=1, out_channels=1).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=self.p.learning_rate)
        # checkpoint = torch.load('unet_checkpoint_2023_4_9_12_56_13.pth.tar')  # models: min intensity=2
        checkpoint = torch.load('unet_checkpoint_2023_4_9_13_7_16.pth.tar', map_location=torch.device(DEVICE))  # models: min intensity=1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Loading images from validation directory {self.val_dir}')
        val_dataset = FileDataset(self.val_dir['images'], mask_dir=self.val_dir['masks'],
                                         transform=A.Compose([ToTensorV2()]))
        deploy_loader = DataLoader(val_dataset, batch_size=self.p.batch_size, shuffle=False,
                                   num_workers=self.p.num_workers, pin_memory=True)
        self.fig_dir = os.path.join(self.fig_dir, 'val')

        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        print('Deploying models\n')
        model.eval()
        if image_stack_path is None:
            check_accuracy(deploy_loader, model, device=DEVICE)
            model.eval()
            self.check_locations(deploy_loader, model, device=DEVICE)
            val_df = pd.DataFrame(self.val_locations)
            print('mean distance', val_df.distance.mean())
            print('stdev distance', val_df.distance.std())
            print('max distance', val_df.distance.max())
            print('min distance', val_df.distance.min())
            val_df.to_csv('val_locations.csv')
            if savebool:
                save_predictions_as_imgs(
                    deploy_loader, model, self.fig_dir, device=DEVICE,
                    save_masks=image_stack_path is None
                )

        else:
            with torch.no_grad():
                for frame_num, (x, y) in enumerate(deploy_loader):
                    # batch size needs to be same size as number of crops per image
                    print(f'Running Frame Number: {frame_num}')
                    x = x.to(DEVICE)
                    preds = torch.sigmoid(model(x))
                    preds = (preds > 0.5).float()
                    np_preds = preds.cpu().detach().numpy()
                    np_preds = np_preds.squeeze() * 255
                    montage = montage_crops(np_preds, frame_num, self.p.batch_size, crop_size=200,
                                            orig_shape=(600, 800), mapping=mapping)
                    self.localize_squares(montage > 100, frame_num)

                    if savebool:
                        savepath = os.path.join(self.fig_dir, f'montage_{frame_num}.png')
                        imageio.imwrite(savepath, montage)
                        print(f'Saved image to {savepath}')
            df = pd.DataFrame(self.locations)
            df.to_csv('locations.csv')
            print('Saved locations.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_stack_path', action="store",
                        default=r'C:\Users\hyrule\take_home_movie_compresed.tiff',
                        help='Path to image stack',
                        dest='image_stack_path')
    args = parser.parse_args()

    # data_dir = '/Users/gandalf/PycharmProjects/Segmentation/data_intensity_3'
    # root_dir = r'C:\Users\hyrule\PycharmProjects\Segmentation'
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.getcwd()
    print(f'Current Working Directory: {root_dir}')
    p = Param()
    Dep = Deploy(p, root_dir=root_dir)
    Dep.run(image_stack_path=args.image_stack_path, savebool=True)
