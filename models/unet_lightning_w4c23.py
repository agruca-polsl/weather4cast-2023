# Weather4cast 2023 Starter Kit
#
# The files from this repository make up the Weather4cast 2023 Starter Kit.
# 
# It builds on and extends the Weather4cast 2022 Starter Kit, the
# original copyright and GPL license notices for which are included
# below.
#
# In line with the provisions of that license, all changes and
# additional code are also released under the GNU General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.

# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
# 
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluate import *
import numpy as np;

#models
from models.baseline_UNET3D import UNet as Base_UNET3D # 3_3_2 model selection

VERBOSE = False
# VERBOSE = True

class UNet_Lightning(pl.LightningModule):
    def __init__(self, UNet_params: dict, params: dict,
                 **kwargs):
        super(UNet_Lightning, self).__init__()

        self.in_channels = params['in_channels']
        self.start_filts = params['init_filter_size']
        self.dropout_rate = params['dropout_rate']
        self.model = Base_UNET3D(in_channels=self.in_channels, start_filts =  self.start_filts, dropout_rate = self.dropout_rate)

        self.save_hyperparameters()
        self.params = params
        #self.example_input_array = np.zeros((44,252,252))
        
        self.val_batch = 0
        
        self.prec = 7

        pos_weight = torch.tensor(params['pos_weight']);
        if VERBOSE: print("Positive weight:",pos_weight);

        self.loss = params['loss']
        self.bs = params['batch_size']
        self.loss_fn = {
            'smoothL1': nn.SmoothL1Loss(), 
            'L1': nn.L1Loss(), 
            'mse': F.mse_loss,
            'BCELoss': nn.BCELoss()
            }[self.loss]
        self.main_metric = {
            'smoothL1':          'Smooth L1',
            'L1':                'L1',
            'mse':               'MSE',  # mse [log(y+1)-yhay]'
            'BCELoss':           'BCE',  # binary cross-entropy
            'BCEWithLogitsLoss': 'BCE with logits',
            'CrossEntropy':      'cross-entropy',
            }[self.loss]

        self.relu = nn.ReLU() # None
        t = f"============== n_workers: {params['n_workers']} | batch_size: {params['batch_size']} \n"+\
            f"============== loss: {self.loss} | weight: {pos_weight} (if using BCEwLL)"
        print(t)
    
    def on_fit_start(self):
        """ create a placeholder to save the results of the metric per variable """
        metric_placeholder = {self.main_metric: -1}
        self.logger.log_hyperparams(self.hparams, metric_placeholder)
        
    def forward(self, x):
        x = self.model(x)
        #if self.loss =='BCELoss':
        #x = self.relu(x)
        return x

    def retrieve_only_valid_pixels(self, x, m):
        """ we asume 1s in mask are invalid pixels """
        ##print(f"x: {x.shape} | mask: {m.shape}")
        return x[~m]

    def get_target_mask(self, metadata):
        mask = metadata['target']['mask']
        #print("mask---->", mask.shape)
        return mask
    
    def _compute_loss(self, y_hat, y, agg=True, mask=None):
        
        if mask is not None:
            y_hat[mask] = 0
            y[mask] = 0
        # print("================================================================================")
        # print(y_hat.shape, y_hat.min(), y_hat.max())
        # print(y.shape, y.min(), y.max())
        if agg:
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none')
        return loss
    
    def training_step(self, batch, batch_idx, phase='train'):
        x, y, metadata  = batch
        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')
        y_hat = self.forward(x)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')
        mask = self.get_target_mask(metadata)
        loss = self._compute_loss(y_hat, y, mask=mask)

        #LOGGING
        self.log(f'{phase}_loss', loss,batch_size=self.bs, sync_dist=True)
        return loss
                
    def validation_step(self, batch, batch_idx, phase='val'):
        #data_start = timer()
        x, y, metadata  = batch
        #data_end = timer()
        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')
        y_hat = self.forward(x)
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')

        loss = self._compute_loss(y_hat, y, mask=mask)

        if mask is not None:
            y_hat[mask]=0
            y[mask]=0

        #LOGGING
        self.log(f'{phase}_loss', loss, batch_size=self.bs, sync_dist=True)
        values  = {'val_mse': loss} 
        self.log_dict(values, batch_size=self.bs, sync_dist=True)
    
        return {'loss': loss.cpu(), 'N': x.shape[0],
                'mse': loss.cpu()}
        

    def validation_epoch_end(self, outputs, phase='val'):
        print("Validation epoch end average over batches: ",
              [batch['N'] for batch in outputs]);
        avg_loss = np.average([batch['loss'] for batch in outputs],
                              weights=[batch['N'] for batch in outputs]);
        avg_mse  = np.average([batch['mse'] for batch in outputs],
                              weights=[batch['N'] for batch in outputs]);
        values={f"{phase}_loss_epoch": avg_loss,
                f"{phase}_mse_epoch":  avg_mse}
        self.log_dict(values, batch_size=self.bs, sync_dist=True)
        self.log(self.main_metric, avg_loss, batch_size=self.bs, sync_dist=True)


    def test_step(self, batch, batch_idx, phase='test'):
        x, y, metadata = batch
        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')
        y_hat = self.forward(x)
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')
        loss = self._compute_loss(y_hat, y, mask=mask)
        
        if mask is not None:
            y_hat[mask]=0
            y[mask]=0

        #LOGGING
        self.log(f'{phase}_loss', loss, batch_size=self.bs, sync_dist=True)
        values = {'test_mse': loss}
        self.log_dict(values, batch_size=self.bs, sync_dist=True)
        
        return 0, y_hat

    def predict_step(self, batch, batch_idx, phase='predict'):
        x, y, metadata = batch
        y_hat = self.model(x)
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')

        return y_hat

    def configure_optimizers(self):
        if VERBOSE: print("Learning rate:",self.params["lr"], "| Weight decay:",self.params["weight_decay"])
        optimizer = torch.optim.AdamW(self.parameters(),
                                     lr=float(self.params["lr"]),weight_decay=float(self.params["weight_decay"])) 
        return optimizer


def main():
    print("running")
if __name__ == 'main':
    main()
