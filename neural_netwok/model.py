import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch 
import torch.nn as nn
import torch.nn.functional as F
from  utils import getAzimuthElevation, findRoadIntersection
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.LeakyReLU())

        # Hidden layers
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        for h1, h2 in layer_sizes:
            layers.append(nn.Linear(h1, h2))
            layers.append(nn.LeakyReLU())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        #x=x.view(x.size(0), -1)
       # print(x.shape)
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y, info = batch
        print(f"{x=}")
        print(f"{y=}")
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        projected_errors= []
        for b in range(len(batch)):
          bb_center= info['bb_center'][b].cpu().detach().numpy()
          image_size= info['image_size'][b][0].cpu().detach().numpy().tolist()
          focal_length= info['focal_length'][b].cpu().detach().numpy().tolist()[0]
          sensor_size= info['sensor_size'][b][0].cpu().detach().numpy().tolist()
          camera_position = info['camera_position'][b].cpu().detach().numpy()
          true_center = info['true_center'][b][0].cpu().detach().numpy()
          trun_image_center = info['trun_image_center'][b][0].cpu().detach().numpy()
          
          r= info['r'][b].cpu().detach().numpy()
   

        #   print(f"{bb_center=}")
        #   print(f"{image_size=}")
        #   print(f"{focal_length=}")
        #   print(f"{sensor_size=}")
        #   print(f"{true_center=}")
        #   print(f"{r=}")

          error_hat= y_hat[b].cpu().detach().numpy()
          cm_projected= bb_center+ error_hat* image_size #np.array([[543.3329202855937,82.19877841484055]])
          #cm_projected= bb_center#+ error_hat* image_size
          _, _, cm_hat = getAzimuthElevation(focal_length,sensor_size, image_size, cm_projected, r, check_flag=False)
          center_info = findRoadIntersection(camera_position, cm_hat)

          #print(f"{true_center.shape=}  ||| {center_info.shape=}")


          if(batch_idx>1500):
            print(f"{true_center[:2]=} {center_info=}")
            print(f"{cm_projected=}")
            print(f"{trun_image_center=}")
            print(f"{error_hat=}")
            print(f"{bb_center=}")
            print(f"{error_hat* image_size=}")
          
          projected_errors.append(np.mean(((true_center[:2] - center_info)**2), axis=0))




        projected_errors_mean= np.array(projected_errors).mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('error', projected_errors_mean, prog_bar=True)

        # print(f"{projected_errors_mean=}")
        # print(f"{true_center[:2]=}")
        # print(f"{center_info=}")
        # print(f"{np.linalg.norm(true_center[:2]- center_info)=}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, info = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # Create the ReduceLROnPlateau scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=10, verbose=True)

        # Scheduler configuration dictionary expected by PyTorch Lightning
        scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'val_loss',  # Name of the metric to monitor
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,  # For ReduceLROnPlateau scheduler
            'strict': True,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}