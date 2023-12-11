'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

import os
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
import timeit
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg') # It disables interactive GUI backend. Commented by default but in case of config.train.plot_save_steps << config.train.num_epochs, please uncomment this line (otherwise, may throw matplotlib error). 

from trainers.BaseTrainer import BaseTrainer
from DFINE import DFINE 
from python_utils import carry_to_device
from time_series_utils import get_nrmse_error
from metrics import Mean

torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)


class TrainerDFINE(BaseTrainer):
    '''
    Trainer class for DFINE model. 
    '''

    def __init__(self, config):
        '''
        Initializer for a TrainerDFINE object. Note that TrainerDFINE is a subclass of trainers.BaseTrainer.
        
        Parameters
        ------------
        - config: yacs.config.CfgNode, yacs config which contains all hyperparameters required to create and train the DFINE model
                                       Please see config.py for the hyperparameters, their default values and definitions. 
        '''

        super(TrainerDFINE, self).__init__(config)

        # Initialize training time statistics
        self.training_time = 0
        self.training_time_epochs = []

        # Initialize best validation losses
        self.best_val_loss = torch.inf
        if self.config.model.supervise_behv:
            self.best_val_behv_loss = torch.inf

        # Initialize logger 
        self.logger = self._get_logger(prefix='dfine')

        # Set device
        self.device = 'cpu' if self.config.device == 'cpu' or not torch.cuda.is_available() else 'cuda:0' 
        self.config.device = self.device # if cuda is asked in config but it's not available, config is also updated

        # Initialize the model, optimizer and learning rate scheduler
        self.dfine = DFINE(self.config); 
        self.dfine.to(self.device) # carry the model to the desired device
        self.optimizer = self._get_optimizer(params=self.dfine.parameters())
        self.lr_scheduler = self._get_lr_scheduler()

        # Load ckpt if asked, model with best validation model loss can be loaded as well, which is saved with name 'best_loss_ckpt.pth'
        if (isinstance(self.config.load.ckpt, int) and self.config.load.ckpt > 1) or isinstance(self.config.load.ckpt, str): 
            self.dfine, self.optimizer, self.lr_scheduler = self._load_ckpt(model=self.dfine,
                                                                             optimizer=self.optimizer,
                                                                             lr_scheduler=self.lr_scheduler)

        # Get the metrics 
        self.metric_names, self.metrics = self._get_metrics()

        # Save the config
        self._save_config()


    def _get_metrics(self):
        '''
        Creates the metric names and nested metrics dictionary. 

        Returns: 
        ------------
        - metric_names: list, Metric names to log in Tensorboard, which are the keys of train/valid defined below
        - metrics_dictionary: dict, nested metrics dictionary. Keys (and metric_names) are (e.g. for config.loss.steps_ahead = [1,2]): 
            - train: 
                - steps_{k}_mse: metrics.Mean, Training {k}-step ahead predicted MSE         
                - model_loss: metrics.Mean, Training negative sum of {k}-step ahead predicted MSEs (e.g. steps_1_mse + steps_2_mse)
                - reg_loss: metrics.Mean, L2 regularization loss for DFINE encoder and decoder weights
                - behv_mse: metrics.Mean, Exists if config.model.supervise_behv is True, Training behavior MSE
                - behv_loss: metrics.Mean, Exists if config.model.supervise_behv is True, Training behavior reconstruction loss
                - total_loss: metrics.Mean, Sum of training model_loss, reg_loss and behv_loss (if config.model.supervise_behv is True)
            - valid: 
                - steps_{k}_mse: metrics.Mean, Validation {k}-step ahead predicted MSE
                - model_loss: metrics.Mean, Validation negative sum of {k}-step ahead predicted MSEs (e.g. steps_1_mse + steps_2_mse)
                - reg_loss: metrics.Mean, L2 regularization loss for DFINE encoder and decoder weights
                - behv_mse: metrics.Mean, Exists if config.model.supervise_behv is True, Validation behavior MSE
                - behv_loss: metrics.Mean, Exists if config.model.supervise_behv is True, Validation behavior reconstruction loss
                - total_loss: metrics.Mean, Sum of validation model_loss, reg_loss and behv_loss (if config.model.supervise_behv is True)
        '''

        metric_names = []
        for k in self.config.loss.steps_ahead:
            metric_names.append(f'steps_{k}_mse')
            
        if self.config.model.supervise_behv:
            metric_names.append('behv_mse')
            metric_names.append('behv_loss')
        metric_names.append('model_loss')
        metric_names.append('reg_loss')
        metric_names.append('total_loss')
        
        metrics = {}
        metrics['train'] = {}
        metrics['valid'] = {}

        for key in metric_names:
            metrics['train'][key] = Mean()
            metrics['valid'][key] = Mean()
        
        return metric_names, metrics
    
    
    def _get_log_str(self, epoch, train_valid='train'):
        '''
        Creates the logging/printing string of training/validation statistics at each epoch

        Parameters: 
        ------------
        - epoch: int, Number of epoch to log the statistics for 
        - train_valid: str, Training or validation prefix to log the statistics, 'train' by default

        Returns: 
        ------------
        - log_str: str, Logging string 
        '''

        log_str = f'Epoch {epoch}, {train_valid.upper()}\n'

        # Logging k-step ahead predicted MSEs
        for k in self.config.loss.steps_ahead:
            if k == 1:
                log_str += f"{k}_step_mse: {self.metrics[train_valid][f'steps_{k}_mse'].compute():.5f}\n"
            else:
                log_str += f"{k}_steps_mse: {self.metrics[train_valid][f'steps_{k}_mse'].compute():.5f}\n"

        # Logging L2 regularization loss and L2 scale 
        log_str += f"reg_loss: {self.metrics[train_valid]['reg_loss'].compute():.5f}, scale_l2: {self.dfine.scale_l2:.5f}\n"

        # If model is behavior-supervised, log behavior reconstruction loss
        if self.config.model.supervise_behv:
            log_str += f"behv_loss: {self.metrics[train_valid]['behv_loss'].compute():.5f}, scale_behv_recons: {self.dfine.scale_behv_recons:.5f}\n"

        # Finally, log model_loss and total_loss to optimize
        log_str += f"model_loss: {self.metrics[train_valid]['model_loss'].compute():.5f}, total_loss: {self.metrics[train_valid]['total_loss'].compute():.5f}\n"
        return log_str


    def train_epoch(self, epoch, train_loader, verbose=True):
        '''
        Performs single epoch training over batches, logging to Tensorboard and plot generation

        Parameters: 
        ------------
        - epoch: int, Number of epoch to perform training iteration
        - train_loader: torch.utils.data.DataLoader, Training dataloader
        '''

        # Take the model into training mode
        self.dfine.train()

        # Reset the metrics at the beginning of each epoch
        self._reset_metrics(train_valid='train')

        # Keep track of update step for logging the gradient norms
        step = (epoch - 1) * len(train_loader) + 1

        # Keep the time which training epoch starts
        start_time = timeit.default_timer()

        # Start iterating over batches
        with tqdm(train_loader, unit='batch') as tepoch:
            for _, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}, TRAIN")

                # Carry data to device
                batch = carry_to_device(data=batch, device=self.device)
                y_batch, behv_batch, mask_batch = batch

                # Perform forward pass and compute loss
                model_vars = self.dfine(y=y_batch, mask=mask_batch)
                loss, loss_dict = self.dfine.compute_loss(y=y_batch, 
                                                          model_vars=model_vars, 
                                                          mask=mask_batch, 
                                                          behv=behv_batch)

                # Compute model gradients
                self.optimizer.zero_grad()
                loss.backward()

                # Log UNCLIPPED model gradients after gradient computations
                self.write_model_gradients(self.dfine, step=step, prefix='unclipped')

                # Skip gradient clipping for the first epoch
                if epoch > 1:
                    clip_grad_norm_(self.dfine.parameters(), self.config.optim.grad_clip)

                # Log CLIPPED model gradients after gradient computations
                self.write_model_gradients(model=self.dfine, step=step, prefix='clipped')

                # Update model parameters
                self.optimizer.step()

                # Update metrics
                self._update_metrics(loss_dict=loss_dict, 
                                     batch_size=y_batch.shape[0], 
                                     train_valid='train', 
                                     verbose=False)
            
                # Update the step 
                step += 1
        
        # Get the runtime for the training epoch
        epoch_time = timeit.default_timer() - start_time
        self.training_time += epoch_time
        self.training_time_epochs.append(epoch_time)
                
        # Save model, optimizer and learning rate scheduler (we save the initial and the last model no matter what config.model.save_steps is)
        if epoch % self.config.model.save_steps == 0 or epoch == 1 or epoch == self.config.train.num_epochs:
            self._save_ckpt(epoch=epoch, 
                            model=self.dfine, 
                            optimizer=self.optimizer, 
                            lr_scheduler=self.lr_scheduler)
        
        # Write model summary
        self.write_summary(epoch, prefix='train')

        # Create and save plots from the last batch
        if epoch % self.config.train.plot_save_steps == 0 or epoch == 1 or epoch == self.config.train.num_epochs:
            self.create_plots(y_batch=y_batch, 
                              behv_batch=behv_batch, 
                              model_vars=model_vars, 
                              epoch=epoch, 
                              prefix='train')

        # Logging the training step information for last batch
        if verbose and (epoch % self.config.train.print_log_steps == 0 or epoch == 1 or epoch == self.config.train.num_epochs):
            log_str = self._get_log_str(epoch=epoch, train_valid='train')
            self.logger.info(log_str)

        # Update LR 
        self.lr_scheduler.step()


    def valid_epoch(self, epoch, valid_loader, verbose=True):
        '''
        Performs single epoch validation over batches, logging to Tensorboard and plot generation

        Parameters: 
        ------------
        - epoch: int, Number of epoch to perform validation 
        - valid_loader: torch.utils.data.DataLoader, Validation dataloader
        '''

        with torch.no_grad():
            # Take the model into evaluation mode
            self.dfine.eval()

            # Reset metrics at the beginning of each epoch
            self._reset_metrics(train_valid='valid')

            # Start iterating over the batches
            y_all, mask_all = [], []
            with tqdm(valid_loader, unit='batch') as tepoch:
                for _, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}, VALID")

                    # Carry data to device
                    batch = carry_to_device(data=batch, device=self.device)
                    y_batch, behv_batch, mask_batch = batch
                    y_all.append(y_batch)
                    mask_all.append(mask_batch)

                    # Perform forward pass and compute loss
                    model_vars = self.dfine(y=y_batch, mask=mask_batch)
                    _, loss_dict = self.dfine.compute_loss(y=y_batch, 
                                                        model_vars=model_vars, 
                                                        mask=mask_batch, 
                                                        behv=behv_batch)

                    # Update metrics 
                    self._update_metrics(loss_dict=loss_dict, 
                                        batch_size=y_batch.shape[0], 
                                        train_valid='valid', 
                                        verbose=False)
            
            # Perform one-step-ahead prediction on the provided validation data, for evaluation
            y_all = torch.cat(y_all, dim=0)
            mask_all =  torch.cat(mask_all, dim=0)
            model_vars_all = self.dfine(y=y_all, mask=mask_all)
            y_pred_all = model_vars_all['y_pred']
            _, one_step_ahead_nrmse = get_nrmse_error(y_all[:, 1:, :], y_pred_all)
            self.training_valid_one_step_nrmses.append(one_step_ahead_nrmse)

            # Write model summary
            self.write_summary(epoch, prefix='valid')

            # Save the best validation loss model (and best behavior reconstruction loss model if supervised)
            if self.metrics['valid']['model_loss'].compute() < self.best_val_loss:
                self.best_val_loss = self.metrics['valid']['model_loss'].compute()
                self._save_ckpt(epoch='best_loss', 
                                model=self.dfine, 
                                optimizer=self.optimizer, 
                                lr_scheduler=self.lr_scheduler)

            if self.config.model.supervise_behv: 
                if self.metrics['valid']['behv_loss'].compute() < self.best_val_behv_loss:
                    self.best_val_behv_loss = self.metrics['valid']['behv_loss'].compute()
                    self._save_ckpt(epoch='best_behv_loss', 
                                    model=self.dfine, 
                                    optimizer=self.optimizer, 
                                    lr_scheduler=self.lr_scheduler)
                                    
            # Create and save plots from last batch
            if epoch % self.config.train.plot_save_steps == 0 or epoch == 1 or epoch == self.config.train.num_epochs:
                self.create_plots(y_batch=y_batch, 
                                behv_batch=behv_batch, 
                                model_vars=model_vars,
                                epoch=epoch, 
                                prefix='valid')
            
            if verbose and (epoch % self.config.train.print_log_steps == 0 or epoch == 1 or epoch == self.config.train.num_epochs):
                # Logging the validation step information for last batch
                log_str = self._get_log_str(epoch=epoch, train_valid='valid')
                self.logger.info(log_str)


    def train(self, train_loader, valid_loader=None):
        '''
        Performs full training of DFINE model

        Parameters: 
        ------------
        - train_loader: torch.utils.data.DataLoader, Training dataloader
        - valid_loader: torch.utils.data.DataLoader, Validation dataloader, None by default (if no valid_loader is provided, validation is skipped)
        '''

        # Bookkeeping the validation NRMSEs over the course of training
        self.training_valid_one_step_nrmses = []

        # Start iterating over the epochs
        for epoch in range(self.start_epoch, self.config.train.num_epochs + 1):
            # Perform validation with the initialized model
            if epoch == self.start_epoch:
                self.valid_epoch(epoch, valid_loader, verbose=False)

            # Perform training iteration over train_loader
            self.train_epoch(epoch, train_loader)

            # Perform validation over valid_loader if it's not None and we're at validation epoch
            if (epoch % self.config.train.valid_step == 0) and isinstance(valid_loader, torch.utils.data.dataloader.DataLoader):
                self.valid_epoch(epoch, valid_loader)


    def create_plots(self, y_batch, model_vars, behv_batch=None, mask_batch=None, epoch=1, trial_num=0, prefix='train'):
        '''
        Creates training/validation plots of neural reconstruction, manifold latent factors and dynamic latent factors

        Parameters: 
        ------------
        - y_batch: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of high-dimensional neural observations
        - model_vars: dict, Dictionary which contains inferrred latents, predictions and reconstructions. See DFINE.forward for further details. 
        - epoch: int, Number of epoch for which to create plot
        - behv_batch: torch.Tensor, shape: (num_seq, num_steps, dim_behv), Batch of behavior, None by default
        - trial_num: int, Trial number to plot
        - prefix: str, Plotname prefix to save plots
        '''

        # Create the mask if it's None
        if mask_batch is None: 
            mask_batch = torch.ones(y_batch.shape[:-1], dtype=torch.float32).unsqueeze(dim=-1)

        # Generate and save reconstructed neural observation plot
        self.create_y_plot(y_batch=y_batch, y_hat_batch=model_vars['y_hat'], mask_batch=mask_batch, epoch=epoch, trial_num=trial_num, prefix=f'{prefix}', feat_name='y_hat')
        # Generate and save smoothed neural observation plot
        self.create_y_plot(y_batch=y_batch, y_hat_batch=model_vars['y_smooth'], mask_batch=mask_batch, epoch=epoch, trial_num=trial_num, prefix=f'{prefix}', feat_name='y_smooth')

        # Generate and save smoothed manifold latent factor plot
        self.create_k_step_ahead_plot(y_batch=y_batch, model_vars=model_vars, mask_batch=mask_batch, epoch=epoch, trial_num=trial_num, prefix=prefix)

        # Generate and save projected (encoder output directly) manifold latent factor plot
        self.create_latent_factor_plot(f=model_vars['a_hat'], epoch=epoch, trial_num=trial_num, prefix=prefix, feat_name='a_hat')
        # Generate and save smoothed manifold latent factor plot
        self.create_latent_factor_plot(f=model_vars['a_smooth'], epoch=epoch, trial_num=trial_num, prefix=prefix, feat_name='a_smooth')
        # Generate and save smoothed dynamic latent factor plot
        self.create_latent_factor_plot(f=model_vars['x_smooth'], epoch=epoch, trial_num=trial_num, prefix=prefix, feat_name='x_smooth')
        # Generate and save reconstructed behavior if model is behavior-supervised
        if self.config.model.supervise_behv and (behv_batch is not None):
            self.create_behv_recons_plot(behv_batch=behv_batch, behv_hat_batch=model_vars['behv_hat'], epoch=epoch, trial_num=trial_num, prefix=prefix)

        plt.close('all')


    def create_y_plot(self, y_batch, y_hat_batch, mask_batch=None, epoch=1, trial_num=0, prefix='train', feat_name='y_hat'):
        '''
        Creates true and estimated neural observation plots during training and validation

        Parameters:
        ------------
        - y_batch: torch.Tensor, shape: (num_seq, num_steps, dim_y), True high-dimensional neural observation
        - y_hat_batch: torch.Tensor, shape: (num_seq, num_steps, dim_y), Reconstructed high-dimensional neural observation, smoothed/filtered/reconstructed neural observation can be provided
        - mask_batch: torch.Tensor, shape: (num_seq, num_steps, 1), Mask for manifold latent factors which shows whether 
                                                                    observations at each timestep exists (1) or are missing (0)
        - epoch: int, Number of epoch for which to create the plot
        - trial_num:, int, Trial number in the batch to plot
        - prefix: str, Plotname prefix to save the plot
        - feat_name: str, Feature name of y_hat_batch (e.g. y_hat/y_smooth) used in plotname
        '''
        
        # Create the mask if it's None
        if mask_batch is None: 
            mask_batch = torch.ones(y_batch.shape[:-1], dtype=torch.float32).unsqueeze(dim=-1)
        
        # Detach tensors for plotting
        y_batch = y_batch.detach().cpu()
        y_hat_batch = y_hat_batch.detach().cpu()
        mask_batch = mask_batch.detach().cpu()

        # Mask y_batch and y_hat_batch 
        num_seq, _, dim_y = y_batch.shape
        mask_bool_batch = mask_batch.type(torch.bool).tile(1, 1, self.dfine.dim_y)
        y_batch = y_batch[mask_bool_batch].reshape(num_seq, -1, self.dfine.dim_y)
        y_hat_batch = y_hat_batch[mask_bool_batch].reshape(num_seq, -1, self.dfine.dim_y)

        # Create the figure
        fig = plt.figure(figsize=(20,15))
        num_samples = y_batch.shape[1]
        color_index = range(num_samples)
        color_map = plt.cm.get_cmap('viridis')

        # Plot the true observations and noiseless observations (if it's provided)
        if dim_y >= 3:
            ax = fig.add_subplot(321, projection='3d')
            ax_m = ax.scatter(y_batch[trial_num, :, 0], y_batch[trial_num, :, 1], y_batch[trial_num, :, 2], c=color_index, vmin=0, vmax=num_samples, s=35, cmap=color_map, label='y_true')

            ax.set_xlabel('Dim 0')
            ax.set_ylabel('Dim 1')
            ax.set_zlabel('Dim 2')
            ax.set_title(f'True observations in 3d')
            ax.legend()
            fig.colorbar(ax_m)
            
            # Plot the reconstructed observation and noiseless observations (if it's provided)
            ax = fig.add_subplot(322, projection='3d')
            ax_m = ax.scatter(y_hat_batch[trial_num, :, 0], y_hat_batch[trial_num, :, 1], y_hat_batch[trial_num, :, 2], c=color_index, vmin=0, vmax=num_samples, s=35, cmap=color_map, label='y_hat')
            ax.set_title(f'Reconstructed observations in 3d')
            ax.set_xlabel('Dim 0')
            ax.set_ylabel('Dim 1')
            ax.set_zlabel('Dim 2')
            ax.legend()
            fig.colorbar(ax_m)

        # Plot Dim 0
        ax = fig.add_subplot(324)
        ax.plot(range(num_samples), y_batch[trial_num, :, 0], 'g', label='y_true')
        ax.plot(range(num_samples), y_hat_batch[trial_num, :, 0], 'b', label='y_hat')
        ax.set_title('Dim 0')

        if dim_y >= 2:
            # Plot Dim 1
            ax = fig.add_subplot(325)
            ax.plot(range(num_samples), y_batch[trial_num, :, 1], 'g', label='y_true')
            ax.plot(range(num_samples), y_hat_batch[trial_num, :, 1] ,'b', label='y_hat')
            ax.set_title('Dim 1')

        if dim_y >= 3:
            # Plot Dim 2
            ax = fig.add_subplot(326)
            ax.plot(range(num_samples), y_batch[trial_num, :, 2], 'g', label='y_true')
            ax.plot(range(num_samples), y_hat_batch[trial_num, :,2], 'b', label='y_hat')
            ax.set_title('Dim 2')
            ax.legend()

        # Save the plot under plot_save_dir
        plot_name = f'{prefix}_{feat_name}_{epoch}.png'
        plt.savefig(os.path.join(self.plot_save_dir, plot_name))
        plt.close('all')


    def create_k_step_ahead_plot(self, y_batch, model_vars, mask_batch=None, epoch=1, trial_num=0, prefix='train'):
        '''
        Creates true and k-step ahead predicted neural observation plots during training and validation

        Parameters:
        ------------
        - y_batch: torch.Tensor, shape: (num_seq, num_steps, dim_y), True high-dimensional neural observation
        - model_vars: dict, Dictionary which contains inferrred latents, predictions and reconstructions. See DFINE.forward for further details. 
        - mask_batch: torch.Tensor, shape: (num_seq, num_steps, 1), Mask for manifold latent factors which shows whether 
                                                                    observations at each timestep exists (1) or are missing (0)
        - epoch: int, Number of epoch for which to create the plot
        - trial_num:, int, Trial number in the batch to plot
        - prefix: str, Plotname prefix to save the plot
        '''
        
        num_total_steps = y_batch.shape[1]

        # Create the mask if it's None
        if mask_batch is None: 
            mask_batch = torch.ones(y_batch.shape[:-1], dtype=torch.float32).unsqueeze(dim=-1)

        # Get the number of steps ahead for which DFINE is optimized and create the figure
        num_k = len(self.config.loss.steps_ahead)
        fig = plt.figure(figsize=(20, 20))
        fig_num = 1
        
        # Start iterating over steps ahead for plotting
        for k in self.config.loss.steps_ahead:
            # Get the k-step ahead prediction 
            y_pred_k_batch, _, _ = self.dfine.get_k_step_ahead_prediction(model_vars, k)

            # Detach tensors for plotting and take timesteps from k to T (since we're plotting k-step ahead predictions)
            y_batch_k = y_batch[:, k:, ...].detach().cpu()  
            mask_batch_k = mask_batch[:, k:, :].detach().cpu()
            y_pred_k_batch = y_pred_k_batch.detach().cpu()

            # Mask y and y_hat 
            num_seq = y_batch.shape[0]
            mask_bool = mask_batch_k.type(torch.bool).tile(1, 1, self.dfine.dim_y)
            y_batch_k = y_batch_k[mask_bool].reshape(num_seq, -1, self.dfine.dim_y)
            y_pred_k_batch = y_pred_k_batch[mask_bool].reshape(num_seq, -1, self.dfine.dim_y)
            
            # Plot dimension 0
            ax = fig.add_subplot(num_k, 2, fig_num)
            ax.plot(range(k, num_total_steps), y_batch_k[trial_num, :, 0], 'g', label=f'{k}-step y_true')
            ax.plot(range(k, num_total_steps), y_pred_k_batch[trial_num, :, 0], 'b', label=f'{k}-step predicted y')
            ax.set_title(f'k={k} step ahead')
            ax.set_xlabel('Time')
            ax.set_ylabel('Dim 0')
            ax.legend()
            fig_num += 1
            
            # Plot first 3 dimensions of k-step prediction as 3D scatter plot (mostly not useful visualization unless the manifold is obvious in first 3 dimensions)
            color_index = range(y_batch_k.shape[1])
            color_map = plt.cm.get_cmap('viridis')
            ax = fig.add_subplot(num_k, 2, fig_num, projection='3d')
            ax_m = ax.scatter(y_pred_k_batch[trial_num, :, 0], y_pred_k_batch[trial_num, :, 1], y_pred_k_batch[trial_num, :, 2], c=color_index, vmin=0, vmax=y_batch.shape[1], s=35, cmap=color_map, label=f'{k}-step predicted y')
            ax.set_title(f'k={k} step ahead')
            ax.set_xlabel('Dim 0')
            ax.set_ylabel('Dim 1')
            ax.set_zlabel('Dim 2')
            ax.legend()
            fig.colorbar(ax_m)
            fig_num += 1
        
        # Save the plot under plot_save_dir
        plot_name = f'{prefix}_k_step_obs_{epoch}.png'
        plt.savefig(os.path.join(self.plot_save_dir, plot_name))
        plt.close('all')
    

    def create_latent_factor_plot(self, f, epoch=1, trial_num=0, prefix='train', feat_name='x_smooth'):
        '''
        Creates dynamic latent factor plots during training/validation

        Parameters:
        ------------
        - f: torch.Tensor, shape: (num_seq, num_steps, dim_x/dim_a), Batch of inferred dynamic/manifold latent factors, smoothed/filtered factors can be provided
        - epoch: int, Number of epoch for which to create dynamic latent factor plot
        - trial_num: int, Trial number to plot
        - prefix: str, Plotname prefix to save plots
        - feat_name: str, Feature name of y_hat_batch (e.g. y_hat/y_smooth) used in plotname
        '''

        # Detach the tensor for plotting
        f = f.detach().cpu()
        
        # From feat_name, get whether it's manifold or dynamic latent factors
        if feat_name[0].lower() == 'x':
            factor_name = 'Dynamic'
        else:
            factor_name = 'Manifold' 

        # Create the figure and colormap
        fig = plt.figure(figsize=(10,8))
        _, num_steps, dim_f = f.shape
        color_index = range(num_steps)
        color_map = plt.cm.get_cmap('viridis')
        
        if dim_f > 2:
            # Scatter first 3 dimensions of dynamic latent factors 
            ax = fig.add_subplot(221, projection='3d')
            ax_m = ax.scatter(f[trial_num, :, 0], f[trial_num, :, 1], f[trial_num, :, 2], c=color_index, vmin=0, vmax=num_steps, s=35, cmap=color_map)
            ax.set_xlabel('Dim 0')
            ax.set_ylabel('Dim 1')
            ax.set_zlabel('Dim 2')
            ax.set_title(f'{factor_name} latent factors in 3D')
            fig.colorbar(ax_m)

            # Scatter first 2 dimensions of dynamic latent factors, top view
            ax = fig.add_subplot(222)
            ax_m = ax.scatter(f[trial_num, :, 0], f[trial_num, :, 1], c=color_index, vmin=0, vmax=num_steps, s=35, cmap=color_map)
            ax.set_xlabel('Dim 0')
            ax.set_ylabel('Dim 1')
            ax.set_title(f'{factor_name} latent factors from top')
            fig.colorbar(ax_m)

            # Plot the first dimension of dynamic latent factors
            ax = fig.add_subplot(223)
            ax.plot(range(num_steps), f[trial_num, :, 0])
            ax.set_xlabel('Time')
            ax.set_ylabel('Dim 0')

            # Plot the second dimension of dynamic latent factors
            ax = fig.add_subplot(224)
            ax.plot(range(num_steps), f[trial_num, :, 1])
            ax.set_xlabel('Time')
            ax.set_ylabel('Dim 1')

        elif dim_f == 2:
            # Scatter first 2 dimensions of dynamic latent factors, top view
            ax = fig.add_subplot(221)
            ax_m = ax.scatter(f[trial_num, :, 0], f[trial_num, :, 1], c=color_index, vmin=0, vmax=num_steps, s=35, cmap=color_map)
            ax.set_xlabel('Dim 0')
            ax.set_ylabel('Dim 1')
            ax.set_title(f'{factor_name} latent factors from top')
            fig.colorbar(ax_m)

            # Plot the first dimension of dynamic latent factors
            ax = fig.add_subplot(222)
            ax.plot(range(num_steps), f[trial_num, :, 0])
            ax.set_xlabel('Time')
            ax.set_ylabel('Dim 0')

            # Plot the second dimension of dynamic latent factors
            ax = fig.add_subplot(223)
            ax.plot(range(num_steps), f[trial_num, :, 1])
            ax.set_xlabel('Time')
            ax.set_ylabel('Dim 1')

        else:
            # Plot the first dimension of dynamic latent factors
            ax = fig.add_subplot(111)
            ax.plot(range(num_steps), f[trial_num, :, 0])
            ax.set_xlabel('Time')
            ax.set_ylabel('Dim 0')
        fig.suptitle(f'{factor_name} latent factors info', fontsize=16)
        
        # Save the plot under plot_save_dir
        plot_name = f'{prefix}_{feat_name}_{epoch}.png'
        plt.savefig(os.path.join(self.plot_save_dir, plot_name))
        plt.close('all')

    
    def create_behv_recons_plot(self, behv_batch, behv_hat_batch, epoch=1, trial_num=0, prefix='train'):
        '''
        Creates behavior reconstruction plots during training/validation

        Parameters:
        ------------
        - behv_batch: torch.Tensor, shape: (num_seq, num_steps, dim_behv), Batch of true behavior
        - behv_hat_batch: torch.Tensor, shape: (num_seq, num_steps, dim_behv), Batch of reconstructed behavior
        - epoch: int, Number of epoch for which to create dynamic latent factor plot
        - trial_num: int, Trial number to plot
        - prefix: str, Plotname prefix to save plots
        '''

        # Create the figure and detach the tensors for plotting 
        fig = plt.figure(figsize=(15,20))
        behv_batch = behv_batch.detach().cpu()
        behv_hat_batch = behv_hat_batch.detach().cpu()

        # Plot the desired behavior dimension
        for k_i, i in enumerate(self.config.model.which_behv_dims):
            ax = fig.add_subplot(self.dfine.dim_behv, 1, k_i+1)
            ax.plot(behv_batch[trial_num, :, i], label='True Behavior', color='green')
            ax.plot(behv_hat_batch[trial_num, :, k_i], label='Decoded Behavior', color='red')
            ax.set_xlabel(f'Time')
            ax.set_ylabel(f'Dim {i+1}')
            ax.legend()
        
        # Save the plot under plot_save_dir
        plot_name = f'{prefix}_behv_{epoch}.png'
        plt.savefig(os.path.join(self.plot_save_dir, plot_name))
        plt.close('all')


    def save_encoding_results(self, train_loader, valid_loader=None, do_full_inference=True, save_results=True):
        '''
        Performs inference, reconstruction and predictions for training data and validation data (if provided), and saves training and inference time statistics.
        Then, encoding results are saved under {config.model.save_dir}/encoding_results.pt.

        Parameters:
        ------------
        - train_loader: torch.utils.data.DataLoader, Training dataloader
        - valid_loader: torch.utils.data.DataLoader, Validation dataloader, None by default (if no valid_loader is provided, validation inference is skipped)
        - do_full_inference: bool, Whether to perform inference on flattened trials of batches of segments
        '''
        self.dfine.eval()

        with torch.no_grad():
            ############################################################################ BATCH INFERENCE ############################################################################
            # Create the keys for encoding results dictionary
            encoding_dict = {}
            encoding_dict['training_time'] = self.training_time
            encoding_dict['training_time_epochs'] = self.training_time_epochs
            encoding_dict['latent_inference_time'] = dict(train=0, valid=0)

            encoding_dict['x_pred'] = dict(train=[], valid=[])
            encoding_dict['x_filter'] = dict(train=[], valid=[])
            encoding_dict['x_smooth'] = dict(train=[], valid=[])
            
            encoding_dict['a_hat'] = dict(train=[], valid=[])
            encoding_dict['a_pred'] = dict(train=[], valid=[])
            encoding_dict['a_filter'] = dict(train=[], valid=[])
            encoding_dict['a_smooth'] = dict(train=[], valid=[])

            encoding_dict['mask'] = dict(train=[], valid=[]) 

            y_key_list = ['y', 'y_hat', 'y_filter', 'y_smooth', 'y_pred']
            for k in self.config.loss.steps_ahead:
                if k != 1:
                    y_key_list.append(f'y_{k}_pred')

            for y_key in y_key_list:
                encoding_dict[y_key] = dict(train=[], valid=[]) 

            # If model is behavior-supervised, create the keys for behavior reconstruction 
            if self.config.model.supervise_behv:
                encoding_dict['behv'] = dict(train=[], valid=[])
                encoding_dict['behv_hat'] = dict(train=[], valid=[])

            # Dump train_loader and valid_loader into a dictionary 
            loaders = dict(train=train_loader, valid=valid_loader)

            # Start iterating over dataloaders
            for train_valid, loader in loaders.items():
                if isinstance(loader, torch.utils.data.dataloader.DataLoader):
                    # If loader is not None, start iterating over the batches
                    for _, batch in enumerate(loader):
                        # Keep track of latent inference start time
                        start_time = timeit.default_timer()

                        batch = carry_to_device(batch, device=self.device)
                        y_batch, behv_batch, mask_batch = batch
                        model_vars = self.dfine(y=y_batch, mask=mask_batch)

                        # Add to the latent inference time over the batches
                        encoding_dict['latent_inference_time'][train_valid] += timeit.default_timer() - start_time

                        # Append the inference variables to the empty lists created in the beginning
                        encoding_dict['x_pred'][train_valid].append(model_vars['x_pred'].detach().cpu())
                        encoding_dict['x_filter'][train_valid].append(model_vars['x_filter'].detach().cpu())
                        encoding_dict['x_smooth'][train_valid].append(model_vars['x_smooth'].detach().cpu())

                        encoding_dict['a_hat'][train_valid].append(model_vars['a_hat'].detach().cpu())
                        encoding_dict['a_pred'][train_valid].append(model_vars['a_pred'].detach().cpu())
                        encoding_dict['a_filter'][train_valid].append(model_vars['a_filter'].detach().cpu())
                        encoding_dict['a_smooth'][train_valid].append(model_vars['a_smooth'].detach().cpu())

                        encoding_dict['mask'][train_valid].append(mask_batch.detach().cpu())
                        encoding_dict['y'][train_valid].append(y_batch.detach().cpu())
                        encoding_dict['y_hat'][train_valid].append(model_vars['y_hat'].detach().cpu())
                        encoding_dict['y_pred'][train_valid].append(model_vars['y_pred'].detach().cpu())
                        encoding_dict['y_filter'][train_valid].append(model_vars['y_filter'].detach().cpu())
                        encoding_dict['y_smooth'][train_valid].append(model_vars['y_smooth'].detach().cpu())

                        for k in self.config.loss.steps_ahead:
                            if k != 1:
                                y_pred_k, _, _ = self.dfine.get_k_step_ahead_prediction(model_vars, k)
                                encoding_dict[f'y_{k}_pred'][train_valid].append(y_pred_k)

                        if self.config.model.supervise_behv:
                            encoding_dict['behv'][train_valid].append(behv_batch.detach().cpu())
                            encoding_dict['behv_hat'][train_valid].append(model_vars['behv_hat'].detach().cpu())

                    # Convert lists to tensors
                    encoding_dict['x_pred'][train_valid] = torch.cat(encoding_dict['x_pred'][train_valid], dim=0)
                    encoding_dict['x_filter'][train_valid] = torch.cat(encoding_dict['x_filter'][train_valid], dim=0)
                    encoding_dict['x_smooth'][train_valid] = torch.cat(encoding_dict['x_smooth'][train_valid], dim=0)

                    encoding_dict['a_hat'][train_valid] = torch.cat(encoding_dict['a_hat'][train_valid], dim=0)
                    encoding_dict['a_pred'][train_valid] = torch.cat(encoding_dict['a_pred'][train_valid], dim=0)
                    encoding_dict['a_filter'][train_valid] = torch.cat(encoding_dict['a_filter'][train_valid], dim=0)
                    encoding_dict['a_smooth'][train_valid] = torch.cat(encoding_dict['a_smooth'][train_valid], dim=0)

                    encoding_dict['mask'][train_valid] = torch.cat(encoding_dict['mask'][train_valid], dim=0)
                    for y_key in y_key_list:
                        encoding_dict[y_key][train_valid] = torch.cat(encoding_dict[y_key][train_valid], dim=0)

                    if self.config.model.supervise_behv:
                        encoding_dict['behv'][train_valid] = torch.cat(encoding_dict['behv'][train_valid], dim=0)
                        encoding_dict['behv_hat'][train_valid] = torch.cat(encoding_dict['behv_hat'][train_valid], dim=0)

            ############################################################################ FULL INFERENCE w/ FLATTENED SEQUENCE ############################################################################
            encoding_dict_full_inference = {}

            if do_full_inference:
                # Create the keys for encoding results dictionary
                encoding_dict_full_inference = {}
                encoding_dict_full_inference['latent_inference_time'] = dict(train=0, valid=0)

                encoding_dict_full_inference['x_pred'] = dict(train=[], valid=[])
                encoding_dict_full_inference['x_filter'] = dict(train=[], valid=[])
                encoding_dict_full_inference['x_smooth'] = dict(train=[], valid=[])
                
                encoding_dict_full_inference['a_hat'] = dict(train=[], valid=[])
                encoding_dict_full_inference['a_pred'] = dict(train=[], valid=[])
                encoding_dict_full_inference['a_filter'] = dict(train=[], valid=[])
                encoding_dict_full_inference['a_smooth'] = dict(train=[], valid=[])

                encoding_dict_full_inference['mask'] = dict(train=[], valid=[]) 
            
                for y_key in y_key_list:
                    encoding_dict_full_inference[y_key] = dict(train=[], valid=[]) 

                # If model is behavior-supervised, create the keys for behavior reconstruction 
                if self.config.model.supervise_behv:
                    encoding_dict_full_inference['behv'] = dict(train=[], valid=[])
                    encoding_dict_full_inference['behv_hat'] = dict(train=[], valid=[])

                # Dump variables to encoding_dict_full_inference
                for train_valid, loader in loaders.items():
                    if isinstance(loader, torch.utils.data.dataloader.DataLoader):
                        # Flatten the batches of neural observations, corresponding mask and behavior if model is supervised
                        encoding_dict_full_inference['y'][train_valid] = encoding_dict['y'][train_valid].reshape(1, -1, self.dfine.dim_y) 
                        encoding_dict_full_inference['mask'][train_valid] = encoding_dict['mask'][train_valid].reshape(1, -1, 1) 

                        if self.config.model.supervise_behv:
                            total_dim_behv = encoding_dict['behv'][train_valid].shape[-1]
                            encoding_dict_full_inference['behv'][train_valid] = encoding_dict['behv'][train_valid].reshape(1, -1, total_dim_behv)   
                        
                        # Keep track of latent inference start time
                        start_time = timeit.default_timer()
                        model_vars = self.dfine(y=encoding_dict_full_inference['y'][train_valid].to(self.device), mask=encoding_dict_full_inference['mask'][train_valid].to(self.device))
                        encoding_dict_full_inference['latent_inference_time'][train_valid] += timeit.default_timer() - start_time

                        # Append the inference variables to the empty lists created in the beginning
                        encoding_dict_full_inference['x_pred'][train_valid] = model_vars['x_pred'].detach().cpu()
                        encoding_dict_full_inference['x_filter'][train_valid] = model_vars['x_filter'].detach().cpu()
                        encoding_dict_full_inference['x_smooth'][train_valid] = model_vars['x_smooth'].detach().cpu()

                        encoding_dict_full_inference['a_hat'][train_valid] = model_vars['a_hat'].detach().cpu()
                        encoding_dict_full_inference['a_pred'][train_valid] = model_vars['a_pred'].detach().cpu()
                        encoding_dict_full_inference['a_filter'][train_valid] = model_vars['a_filter'].detach().cpu()
                        encoding_dict_full_inference['a_smooth'][train_valid] = model_vars['a_smooth'].detach().cpu()

                        encoding_dict_full_inference['y_hat'][train_valid] = model_vars['y_hat'].detach().cpu()
                        encoding_dict_full_inference['y_pred'][train_valid] = model_vars['y_pred'].detach().cpu()
                        encoding_dict_full_inference['y_filter'][train_valid] = model_vars['y_filter'].detach().cpu()
                        encoding_dict_full_inference['y_smooth'][train_valid] = model_vars['y_smooth'].detach().cpu()

                        for k in self.config.loss.steps_ahead:
                            if k != 1:
                                y_pred_k, _, _ = self.dfine.get_k_step_ahead_prediction(model_vars, k)
                                encoding_dict_full_inference[f'y_{k}_pred'][train_valid] = y_pred_k

                        if self.config.model.supervise_behv:
                            encoding_dict_full_inference['behv_hat'][train_valid] = model_vars['behv_hat'].detach().cpu()

            # Dump batch and full inference encoding dictionaries into encoding_results
            encoding_results = dict(batch_inference=encoding_dict, full_inference=encoding_dict_full_inference)

            # Save encoding dictionary as .pt file
            if save_results:
                torch.save(encoding_results, os.path.join(self.config.model.save_dir, 'encoding_results.pt'))
            
            return encoding_results


    def write_summary(self, epoch, prefix='train'):
        '''
        Logs metrics to Tensorboard

        Parameters:
        ------------
        - epoch: int, Number of epoch for which to log metrics 
        - prefix: str, Prefix to log metrics 
        '''

        for key, val in self.metrics[prefix].items():
                self.writer.add_scalar(f'{prefix}/{key}', val.compute(), epoch)

        # Rest below is for logging scale values in the loss, will be same for all prefices, so log them only for 'train'
        if prefix != 'valid':
            self.writer.add_scalar(f'scale_l2', self.dfine.scale_l2, epoch)
            self.writer.add_scalar(f'learning_rate', self.lr_scheduler.get_last_lr()[0], epoch)
            if self.config.model.supervise_behv:
                self.writer.add_scalar(f'scale_behv_recons', self.dfine.scale_behv_recons, epoch)

