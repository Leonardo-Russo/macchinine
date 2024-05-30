import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch 
import torch.nn as nn
import torch.functional as F
from .utils import getAzimuthElevation, findRoadIntersection


class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        for h1, h2 in layer_sizes:
            layers.append(nn.Linear(h1, h2))
            layers.append(nn.ReLU())

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
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)


        projected_errors_fixed= []
        projected_errors= []
        for b in range(len(batch)):
          bb_center= info['bb_center'][b].cpu().detach().numpy()
          image_size= info['image_size'][b][0].cpu().detach().numpy().tolist()
          focal_length= info['focal_length'][b].cpu().detach().numpy().tolist()[0]
          sensor_size= info['sensor_size'][b][0].cpu().detach().numpy().tolist()
          camera_position = info['camera_position'][b].cpu().detach().numpy()
          true_center = info['true_center'][b][0].cpu().detach().numpy()
          r= info['r'][b].cpu().detach().numpy()

          error_hat= y_hat[b].cpu().detach().numpy()
          cm_projected= bb_center+ error_hat* image_size
          cm_projected= bb_center#+ error_hat* image_size
          _, _, cm_hat = getAzimuthElevation(focal_length,sensor_size, image_size, cm_projected, r)
          center_info = findRoadIntersection(camera_position, cm_hat)
          projected_errors.append(np.linalg.norm(true_center[:2]- center_info))

          cm_projected= bb_center+ error_hat* image_size
          _, _, cm_hat = getAzimuthElevation(focal_length,sensor_size, image_size, cm_projected, r)
          center_info = findRoadIntersection(camera_position, cm_hat)
          projected_errors_fixed.append(np.linalg.norm(true_center[:2]- center_info))

        projected_errors_mean= np.array(projected_errors).mean()
        projected_errors_fixed_mean= np.array(projected_errors_fixed).mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('error', projected_errors_mean, prog_bar=True)
        self.log('error_fixed', projected_errors_fixed_mean, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, info = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
