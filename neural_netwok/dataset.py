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
    def __init__(self, data_path= None, num_samples=60000):
        self.num_samples = num_samples
        self.data_path= data_path
        if data_path is not None:
            self.df = pd.read_csv(data_path)
            with open("./camera_parameters.json") as json_file:
                self.camera_params=json.load(json_file)

            self.getSample=self._readSample2
            
            self.focal_length=  float(self.camera_params['focal_length'])/10000
            self.sensor_size=self.camera_params['sensor_size'] 
            self.image_size=self.camera_params['image_size'] 
            self.from_point = np.array([self.camera_params['camera_position']])
            self.from_point = self.from_point.reshape(1, 3)

            self.to_point = np.array([self.camera_params['camera_target']])  
            self.to_point = self.to_point.reshape(1, 3)

            # #print("point_on_image=array([[354, 233]]) ")
            # #print(f"{self.from_point=}")
            # #print(f"{self.to_point=}")

        else:
            self.getSample= self._generateImage 

    def __len__(self):
        if self.data_path is None:
            return self.num_samples
        else:
            num_rows_shape = self.df.shape[0]
            print(f"{num_rows_shape=}")
            return num_rows_shape-1
            

    def _generateImage(self, idx):
        sample=getImage(False)

        left_corner= sample["bounding_box"][0]
        right_corner= sample["bounding_box"][2]

        image_size_x=sample["image_size"][0]
        image_size_y=sample["image_size"][1]

        bbox_height=abs(right_corner[1]-left_corner[1])/ image_size_y
        bbox_width=abs(right_corner[0]-left_corner[0])/ image_size_x

        bbox_x_center=sample["bb_center"][0][0]/ image_size_x
        bbox_y_center=sample["bb_center"][0][1]/ image_size_y

       

        x_center=sample["image_center"][0][0]/ image_size_x
        y_center=sample["image_center"][0][1]/ image_size_y

        error=[x_center-bbox_x_center , y_center-bbox_y_center]

        inputs=np.array([bbox_width,bbox_height, sample["phi"],sample["azimuth"]], dtype='f')
        label=np.array(error, dtype='f')
        info={'bb_center':sample['bb_center'],
                'focal_length':np.array([sample['focal_length']]),
                'sensor_size':np.array([sample['sensor_size']]) ,
                'image_size': np.array([sample['image_size']]),
                'r': sample[ 'r'],
                'true_center':np.array([sample['true_center']]),
                'camera_position':sample['camera_position']
                }


        return inputs, label, info
    
    def _readSample(self, idx):
        desired_row_index =  idx+1


        # Use `skiprows` to skip all rows directly up to the row before your desired row
        # You should not skip the header, so use a lambda to skip all but the desired row
        df = pd.read_csv(self.data_path, skiprows=lambda x: x not in [0, desired_row_index])
        image_size_x = self.image_size[0]
        image_size_y = self.image_size[0]
        
        bbox_x_center = df.at[0, 'bbox_x_center']
        bbox_y_center = df.at[0, 'bbox_y_center']
        bbox_width = df.at[0, 'bbox_width']
        bbox_height =df.at[0, 'bbox_height']
        
        x_center = df.at[0, 'x_center']
        y_center = df.at[0, 'y_center']
        center_coordinates=np.array([bbox_x_center, bbox_y_center])
        center_coordinates = center_coordinates.reshape(1, 2)

        up = np.array([0, 0, 1])

        R_C2W, t_C2W = lookat(self.from_point, self.to_point, up)     # these are the rotation and translation matrices
        R_C2W = R_C2W @ matrix_from_axis_angle((1, 0, 0, np.pi))     # flips about axis 1 to obtain Camera Frame
        R_C2W=R_C2W.reshape(3, 3)
        print(R_C2W)


        azimuth, elevation, b_hat= getAzimuthElevation(
            self.focal_length, 
            self.sensor_size, 
            self.image_size, 
            center_coordinates, 
            R_C2W.reshape(3, 3)
        )
        phi = np.pi/2 - elevation
        #todo image center 


        error=np.array([float((x_center-bbox_x_center)/image_size_x) ,float((y_center-bbox_y_center)/image_size_y)],  dtype='f')

        inputs=np.array([bbox_width/image_size_x,bbox_height/image_size_y, phi,azimuth], dtype='f')
        label=error
        
        info={'bb_center':center_coordinates,
                'focal_length':np.array([self.focal_length]),
                'sensor_size':np.array([self.sensor_size]),
                'image_size':np.array([ self.image_size]),
                'r': R_C2W,
                'true_center':np.array([center_coordinates]),
                'camera_position':self.from_point[0]
                }
        
        return inputs,label,info






    def _readSample2(self, idx):
        #idx= 211 
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


        #print((image_size_x,image_size_y, bbox_x_center, bbox_y_center))
        center_coordinates=np.array([bbox_x_center, bbox_y_center])
        center_coordinates = center_coordinates.reshape(1, 2)
        
        
        # sample=getImage(False)

        # left_corner= sample["bounding_box"][0]
        # right_corner= sample["bounding_box"][2]

        # image_size_x=sample["image_size"][0]
        # image_size_y=sample["image_size"][1]

        # bbox_height=abs(right_corner[1]-left_corner[1])
        # bbox_width=abs(right_corner[0]-left_corner[0])

        # bbox_x_center=sample["bb_center"][0][0]
        # bbox_y_center=sample["bb_center"][0][1]

       

        # x_center=sample["image_center"][0][0]
        # y_center=sample["image_center"][0][1]

        # true_x,true_y,_=sample['true_center']
        # self.from_point=sample['camera_position']
        # self.to_point=np.array([0, 0, 0])



       
        



     
        

        bb_center=np.array([bbox_x_center, bbox_y_center])
        bb_center = bb_center.reshape(1, 2)

        up = np.array([0, 0, 1])

        error=np.array([(x_center-bbox_x_center)/ image_size_x , (y_center-bbox_y_center)/ image_size_y])
        
        R_C2W, t_C2W = lookat(self.from_point, self.to_point, up)     # these are the rotation and translation matrices
        R_C2W = R_C2W @ matrix_from_axis_angle((1, 0, 0, np.pi))     # flips about axis 1 to obtain Camera Frame
        R_C2W=R_C2W.reshape(3, 3)
       # print(self.from_point)


        azimuth, elevation, b_hat= getAzimuthElevation(
            self.focal_length, 
            self.sensor_size, 
            self.image_size, 
            bb_center, 
            R_C2W.reshape(3, 3)
        )
        phi = np.pi/2 - elevation
        #todo image center 


        
        inputs=np.array([bbox_width/image_size_x,bbox_height/image_size_y, phi,azimuth], dtype='f')
        
        # print(f"{x_center=}")
        # print(f"{bbox_x_center=}")

        # print(f"{y_center=}")
        # print(f"{bbox_y_center=}")
        #inputs_generated=np.array([bbox_width,bbox_height, sample["phi"],sample["azimuth"]], dtype='f')
        # print(f"{inputs=}")
        # print(f"{inputs_generated=}")
        label=error
        true_center=np.array((true_x,true_y,0)).reshape(1,3)
        info={'bb_center':bb_center,
                'focal_length':np.array([self.focal_length]),
                'sensor_size':np.array([self.sensor_size]),
                'image_size':np.array([ self.image_size]),
                'r': R_C2W,
                'true_center':true_center,
                'camera_position':self.from_point,
                'trun_image_center':np.array([[x_center,y_center]])
                }
        
        # info_generated={'bb_center':sample['bb_center'],
        #         'focal_length':np.array([sample['focal_length']]),
        #         'sensor_size':np.array([sample['sensor_size']]) ,
        #         'image_size': np.array([sample['image_size']]),
        #         'r': sample[ 'r'],
        #         'true_center':np.array([sample['true_center']]),
        #         'camera_position':sample['camera_position']
        #         }
        # print(f"{info=}")
        # print(f"{info_generated=}")
        # print(f"{(x_center,bbox_x_center, image_size_x)=}")
        
        return inputs,label.astype(np.float32),info








    def __getitem__(self, idx):
        # Generate a single sample using the create_sample function
        x,y,info=self.getSample(idx)
        return x,y,info
     

        # return (np.array([bbox_width,bbox_height, sample["phi"],sample["azimuth"]], dtype='f'),
        #        np.array(error, dtype='f'),
        #        {'bb_center':sample['bb_center'],
        #         'focal_length':np.array([sample['focal_length']]),
        #         'sensor_size':np.array([sample['sensor_size']]) ,
        #         'image_size': np.array([sample['image_size']]),
        #         'r': sample[ 'r'],
        #         'true_center':np.array([sample['true_center']]),
        #         'camera_position':sample['camera_position']
        #         }
        # )


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

