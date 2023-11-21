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

#imports for plotting
import os
import datetime
import matplotlib.pyplot as plt
from utils.viz import plot_sequence, save_pdf


#models
from models.baseline_UNET3D import UNet as Base_UNET3D # 3_3_2 model selection

VERBOSE = False
# VERBOSE = True

class UNet_Lightning(pl.LightningModule):
    def __init__(self, UNet_params: dict, params: dict,
                 **kwargs):
        super(UNet_Lightning, self).__init__()

        self.plot_results = params['plot_results']
        self.in_channel_to_plot = params['in_channel_to_plot']
        self.in_channels = params['in_channels']
        self.start_filts = params['init_filter_size']
        self.dropout_rate = params['dropout_rate']
        self.out_channels = params['len_seq_predict']
        self.model = Base_UNET3D(in_channels=self.in_channels, start_filts =  self.start_filts, 
                                 dropout_rate = self.dropout_rate, out_channels = self.out_channels)

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

        if(self.plot_results):
            title = f'batch {self.val_batch} | mse: {loss.cpu():.3f}'
            self.plot_batch(x, y, y_hat, metadata, title , phase, vmax=1.)

        return 0, y_hat

    def predict_step(self, batch, batch_idx, phase='predict'):
        x, y, metadata = batch
        y_hat = self.model(x)
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')

        if(self.plot_results):
            self.plot_batch(x, y, y_hat, metadata, f'batch: {self.val_batch} | prediction results', phase, vmax=1.)
        
        return y_hat

    def configure_optimizers(self):
        if VERBOSE: print("Learning rate:",self.params["lr"], "| Weight decay:",self.params["weight_decay"])
        optimizer = torch.optim.AdamW(self.parameters(),
                                     lr=float(self.params["lr"]),weight_decay=float(self.params["weight_decay"])) 
        return optimizer

    def plot_batch(self, xs, ys, y_hats, metadata, loss, phase, vmax=0.01, vmin=0):
        figures = []

        # pytorch to numpy
        xs, y_hats = [o.cpu() for o in [xs, y_hats]]
        xs, y_hats = [np.asarray(o) for o in [xs, y_hats]]

        if(phase == "test"):
            ys = ys.cpu()
            ys = np.asarray(ys) 
        else:
            ys = y_hats     # it's going to be empty - just to make life easier while passing values to other functions

        print(f"\nplot batch of size {len(xs)}")
        for i in range(len(xs)):
            print(f"plot, {i+1}/{len(xs)}")
            texts_in = [t[i] for t in metadata['input']['timestamps']]
            texts_ta = [t[i] for t in metadata['target']['timestamps']]
            #title = self.seq_metrics(ys[i].ravel(), y_hats[i].ravel())            
            if VERBOSE:
                print("inputs")
                print(np.shape(xs[i]))
                if(phase == "test"):
                    print("target")
                    print(np.shape(ys[i]))
                print("prediction")
                print(np.shape(y_hats[i]))
            self.collapse_time = True   

            fig = plot_sequence(xs[i], ys[i], y_hats[i], texts_in, texts_ta, 
                                self.params, phase, self.collapse_time, vmax=vmax, vmin=vmin, 
                                channel=self.in_channel_to_plot, title=loss)
            figures.append(fig)
            # save individual image to tensorboard
            self.logger.experiment.add_figure(f"preds_{self.trainer.global_step}_{self.val_batch}_{i}", fig)
        # save all figures to disk
        date_time = datetime.datetime.now().strftime("%m%d-%H:%M")
        fname = f"batch_{self.val_batch}_{date_time}"
        dir_path = os.path.join('plots',f"{self.params['name']}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f_path = os.path.join(dir_path,fname)
        save_pdf(figures, f_path)
        if(phase == "test"):
            print(f'saved figures at: {fname} | {loss}')
        else:
            print(f'saved figures at: {fname}')
        self.val_batch += 1
        return figures

def main():
    print("running")
if __name__ == 'main':
    main()
