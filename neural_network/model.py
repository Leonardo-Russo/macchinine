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
    def __init__(self, input_size, hidden_sizes, output_size, debug=False):
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
        self.train_projected_errors = []
        self.val_projected_errors = []
        self.debug = debug

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, true_centers, info = batch
        computed_centers = self(x)

        loss = F.l1_loss(computed_centers, true_centers)

        if self.debug:
            print("computed_centers: ", computed_centers)
            print("true_centers: ", true_centers)

        # projected_errors= []
        # for b in range(len(batch)):
        #     bb_center = info['bb_center'][b].cpu().detach().numpy()
        #     image_size = info['image_size'][b][0].cpu().detach().numpy().tolist()
        #     focal_length = info['focal_length'][b].cpu().detach().numpy().tolist()[0]
        #     sensor_size = info['sensor_size'][b][0].cpu().detach().numpy().tolist()
        #     camera_position = info['camera_position'][b].cpu().detach().numpy()
        #     true_center = info['true_center'][b][0].cpu().detach().numpy()
        #     az = info['azimuth'][b].cpu().detach().numpy()
        #     el = info['elevation'][b].cpu().detach().numpy()
            
        #     r = info['r'][b].cpu().detach().numpy()

        #     error_hat = y_hat[b].cpu().detach().numpy()
        #     if self.debug:
        #         print("error hat: ", error_hat)
        #         print("image size: ", image_size)
        #     cm_projected = bb_center + error_hat * image_size 
        #     _, _, cm_hat = getAzimuthElevation(focal_length, sensor_size, image_size, cm_projected, r, check_flag=False)
        #     center_info = findRoadIntersection(camera_position, cm_hat)

        #     projected_error = np.mean(((true_center[:2] - center_info)**2), axis=0)
        #     threshold = 2
        #     if projected_error > threshold:
        #         print(f"az = {np.rad2deg(az):3.2f}\tel = {np.rad2deg(el):3.2f}\terr = {projected_error:2.2f}")
            
        #     projected_errors.append(projected_error)

        # projected_errors = np.array(projected_errors)
        # projected_errors_mean = projected_errors.mean()
        # projected_errors_median = np.median(projected_errors)
        # self.train_projected_errors.append(projected_errors_mean)
        self.log('train_loss', loss, prog_bar=True)
        # self.log('mean_error', projected_errors_mean, prog_bar=True)
        # self.log('median_error', projected_errors_median, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # Average the accumulated projected errors
        avg_train_projected_errors = torch.mean(torch.tensor(self.train_projected_errors))
        self.log('avg_train_projected_errors', avg_train_projected_errors, prog_bar=True)
        # Clear the list for next epoch
        self.train_projected_errors = []

    def validation_step(self, batch, batch_idx):
        x, true_centers, info = batch
        computed_centers = self(x)

        loss = F.l1_loss(computed_centers, true_centers)
        self.log('val_loss', loss, prog_bar=True)


        # projected_errors= []
        # for b in range(len(batch)):
        #   bb_center= info['bb_center'][b].cpu().detach().numpy()
        #   image_size= info['image_size'][b][0].cpu().detach().numpy().tolist()
        #   focal_length= info['focal_length'][b].cpu().detach().numpy().tolist()[0]
        #   sensor_size= info['sensor_size'][b][0].cpu().detach().numpy().tolist()
        #   camera_position = info['camera_position'][b].cpu().detach().numpy()
        #   true_center = info['true_center'][b][0].cpu().detach().numpy()
        # #   trun_image_center = info['trun_image_center'][b][0].cpu().detach().numpy()
          
        #   r= info['r'][b].cpu().detach().numpy()
   
        #   error_hat= y_hat[b].cpu().detach().numpy()
        #   cm_projected= bb_center+ error_hat* image_size 
        #   _, _, cm_hat = getAzimuthElevation(focal_length,sensor_size, image_size, cm_projected, r, check_flag=False)
        #   center_info = findRoadIntersection(camera_position, cm_hat)
        #   projected_errors.append(np.mean(((true_center[:2] - center_info)**2), axis=0))

        # projected_errors_mean= np.array(projected_errors).mean()
        # self.val_projected_errors.append(projected_errors_mean)

        # self.val_projected_errors.append(projected_errors_mean)
        # self.log('val_error', projected_errors_mean, prog_bar=True)

        return loss
    def on_validation_epoch_end(self):
        # Average the accumulated projected errors
        avg_val_projected_errors = torch.mean(torch.tensor(self.val_projected_errors))
        self.log('avg_val_projected_errors', avg_val_projected_errors, prog_bar=True)
        # Clear the list for next epoch
        self.val_projected_errors = []


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        
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