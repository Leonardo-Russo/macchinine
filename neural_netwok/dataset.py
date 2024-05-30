import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from .utils import getImage

class MacchinineDataset(Dataset):
    def __init__(self, data_path= None, num_samples=60000):
        self.num_samples = num_samples
        self.data_path= data_path
        if data_path is not None:
            self.getSample=self._readSample
        else:
            self.getSample= self._generateImage 

    def __len__(self):
        return self.num_samples

    def _generateImage(idx):
        return getImage(False)
    
    def _readSample(idx):
        pass
        #return getImage(False)

    def __getitem__(self, idx):
        # Generate a single sample using the create_sample function
        sample=self.getSample(idx)

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
    def __init__(self, batch_size=64, num_samples=300000):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples

    def setup(self, stage=None):
        self.train = MacchinineDataset(num_samples=self.num_samples)
        self.val = MacchinineDataset(num_samples=5000)
        self.test = MacchinineDataset(num_samples=10000)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=1)

