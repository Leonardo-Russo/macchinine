import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils import getImage

import pandas as pd


class MacchinineDataset(Dataset):
    def __init__(self, data_path= None, num_samples=60000):
        self.num_samples = num_samples
        self.data_path= data_path
        if data_path is not None:
            self.getSample=self._readSample
            print("No datapath have been selected. I will generate random samples")
        else:
            self.getSample= self._generateImage 

    def __len__(self):
        return self.num_samples

    def _generateImage(self, idx):
        sample=getImage(False)

        left_corner= sample["bounding_box"][0]
        right_corner= sample["bounding_box"][2]

        image_size_x=sample["image_size"][0]
        image_size_y=sample["image_size"][1]

        bb_height=abs(right_corner[1]-left_corner[1])/ image_size_y
        bb_lenght=abs(right_corner[0]-left_corner[0])/ image_size_x

        bbcx=sample["bb_center"][0][0]/ image_size_x
        bbcy=sample["bb_center"][0][1]/ image_size_y

        image_center_x=sample["image_center"][0][0]/ image_size_x
        image_center_y=sample["image_center"][0][1]/ image_size_y

    
    def _readSample(self, idx):
        desired_row_index = idx+1

        # Use `skiprows` to skip all rows directly up to the row before your desired row
        # You should not skip the header, so use a lambda to skip all but the desired row
        df = pd.read_csv(self.data_path, skiprows=lambda x: x not in [0, desired_row_index])

        # df now contains only the desired row if indexed correctly
        # Selecting specific columns
        result = df[['bbox_x_center', 'bbox_y_center', 'bbox_width', 'bbox_height']]
        bbox_x_center = df.at[0, 'bbox_x_center']
        bbox_y_center = df.at[0, 'bbox_y_center']
        bbox_width = df.at[0, 'bbox_width']
        bbox_height = df.at[0, 'bbox_height']

        #todo image center 

        print(bbox_x_center, bbox_y_center, bbox_width, bbox_height)
        return bbox_x_center, bbox_y_center, bbox_width, bbox_height

        #return getImage(False)

    def __getitem__(self, idx):
        # Generate a single sample using the create_sample function
        sample=self.getSample(idx)

        
        error=[image_center_x-bbcx , image_center_y-bbcy]

        return (np.array([bb_lenght,bb_height, sample["phi"],sample["azimuth"]], dtype='f'),
               np.array(error, dtype='f'),
               {'bb_center':sample['bb_center'],
                'focal_length':np.array([sample['focal_length']]),
                'sensor_size':np.array([sample['sensor_size']]) ,
                'image_size': np.array([sample['image_size']]),
                'r': sample[ 'r'],
                'true_center':np.array([sample['true_center']]),
                'camera_position':sample['camera_position']
                }
        )


#focal_length, sensor_size, image_size, cm_projected, r
class MachinineDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_samples=300000, data_path= None):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.data_path= data_path 

    def setup(self, stage=None):
        self.train = MacchinineDataset(num_samples=self.num_samples, data_path= self.data_path)
        self.val = MacchinineDataset(num_samples=5000,data_path= self.data_path)
        self.test = MacchinineDataset(num_samples=10000,data_path= self.data_path)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=1)

