import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from utils import  getAzimuthElevation, lookat
from pytransform3d.rotations import  matrix_from_axis_angle
import numpy as np
import json

from utils import getImage

import pandas as pd


class MacchinineDataset(Dataset):
    def __init__(self, data_path= None, num_samples=60000, debug=False):
        self.num_samples = num_samples
        self.data_path= data_path
        self.debug = debug

        if data_path is not None:
            self.df = pd.read_csv(data_path)
            self.df = self.df.dropna(subset=['y_center'])
            with open("./camera_parameters.json") as json_file:
                self.camera_params=json.load(json_file)

            self.getSample = self._readSample           # read Data from File
            
            self.focal_length=  float(self.camera_params['focal_length'])/10000
            self.sensor_size=self.camera_params['sensor_size'] 
            self.image_size=self.camera_params['image_size'] 
            self.from_point = np.array([self.camera_params['camera_position']])
            self.from_point = self.from_point.reshape(1, 3)

            self.to_point = np.array([self.camera_params['camera_target']])  
            self.to_point = self.to_point.reshape(1, 3)

        else:
            self.getSample = self._generateImage        # generate Synthetic Data for Training

    def __len__(self):
        if self.data_path is None:
            return self.num_samples
        else:
            num_rows_shape = self.df.shape[0]
            print(f"{num_rows_shape=}")
            return num_rows_shape-1
            
    def _generateImage(self, idx):
        sample=getImage(False)

        left_corner = sample["bounding_box"][0]
        right_corner = sample["bounding_box"][2]

        image_size_x = sample["image_size"][0]
        image_size_y = sample["image_size"][1]

        bbox_height_normalized = abs(right_corner[1]-left_corner[1]) / image_size_y
        bbox_width_normalized = abs(right_corner[0]-left_corner[0]) / image_size_x

        bbox_x_center_normalized = sample["bb_center"][0][0] / image_size_x
        bbox_y_center_normalized = sample["bb_center"][0][1] / image_size_y

        x_center_normalized = sample["image_center"][0][0] / image_size_x
        y_center_normalized = sample["image_center"][0][1] / image_size_y

        # x :   y = f(x)
        inputs = np.array([bbox_width_normalized, bbox_height_normalized, sample["phi"], sample["azimuth"]], dtype='f')
        
        # y
        error = [x_center_normalized - bbox_x_center_normalized , y_center_normalized - bbox_y_center_normalized]   # y
        label = np.array(error, dtype='f')

        # infos
        info={'bb_center':sample['bb_center'],
                'focal_length':np.array([sample['focal_length']]),
                'sensor_size':np.array([sample['sensor_size']]) ,
                'image_size': np.array([sample['image_size']]),
                'r': sample[ 'r'],
                'true_center':np.array([sample['true_center']]),
                'camera_position':sample['camera_position'],
                'azimuth': sample['azimuth'],
                'elevation': sample['elevation']
                }


        return inputs, label, info

    def _readSample(self, idx=1):
        nth_row = self.df.iloc[idx-1] 

        # Use `skiprows` to skip all rows directly up to the row before your desired row
        # You should not skip the header, so use a lambda to skip all but the desired row
        
        image_size_x = self.image_size[0]
        image_size_y = self.image_size[1]
        
        bbox_x_center = nth_row['bbox_x_center'] 
        bbox_y_center =nth_row['bbox_y_center'] 
        bbox_width = nth_row['bbox_width']
        bbox_height =nth_row['bbox_height']
        
        x_center = nth_row['x_center']
        y_center = nth_row['y_center']

        true_x=nth_row['x']
        true_y=nth_row['y'] 

        if self.debug:
            print((image_size_x,image_size_y, bbox_x_center, bbox_y_center))
        
        center_coordinates=np.array([bbox_x_center, bbox_y_center])
        center_coordinates = center_coordinates.reshape(1, 2)

        bb_center=np.array([bbox_x_center, bbox_y_center])
        bb_center = bb_center.reshape(1, 2)

        up = np.array([0, 0, 1])

        error=np.array([(x_center-bbox_x_center)/ image_size_x , (y_center-bbox_y_center)/ image_size_y])
        
        R_C2W, t_C2W = lookat(self.from_point, self.to_point, up)       # these are the rotation and translation matrices
        R_C2W = R_C2W @ matrix_from_axis_angle((1, 0, 0, np.pi))        # flips about axis 1 to obtain Camera Frame
        R_C2W=R_C2W.reshape(3, 3)

        azimuth, elevation, b_hat= getAzimuthElevation(
            self.focal_length, 
            self.sensor_size, 
            self.image_size, 
            bb_center, 
            R_C2W.reshape(3, 3)
        )
        phi = np.pi/2 - elevation
        
        # TODO: image center ???????
        
        inputs = np.array([bbox_width/image_size_x, bbox_height/image_size_y, phi, azimuth], dtype='f')
        label=error
        true_center=np.array((true_x, true_y, 0)).reshape(1,3)
        info  = {'bb_center': bb_center,
                'focal_length': np.array([self.focal_length]),
                'sensor_size': np.array([self.sensor_size]),
                'image_size': np.array([ self.image_size]),
                'r': R_C2W,
                'true_center': true_center,
                'camera_position': self.from_point,
                'trun_image_center': np.array([[x_center, y_center]])
                }

        if self.debug:
            print(f"{x_center=}")
            print(f"{bbox_x_center=}")

            print(f"{y_center=}")
            print(f"{bbox_y_center=}")
            print(f"{inputs=}")
        
        return inputs, label.astype(np.float32), info

    def __getitem__(self, idx):
        x, y, info = self.getSample(idx)
        return x,y,info


class MacchinineDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_samples=300000, num_workers=1, train_data_path=None, eval_data_path=None, debug=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.train_data_path = train_data_path 
        self.eval_data_path = eval_data_path
        self.num_workers = num_workers
        self.debug = debug

    def setup(self, stage=None):
        self.train = MacchinineDataset(num_samples=self.num_samples, data_path=self.train_data_path)
        self.val = MacchinineDataset(num_samples=int(self.num_samples * 2/7), data_path=self.eval_data_path)
        self.test = MacchinineDataset(num_samples=int(self.num_samples * 1/7), data_path=self.eval_data_path)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
