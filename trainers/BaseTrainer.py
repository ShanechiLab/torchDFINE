'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

import os 
import torch 
import datetime
import logging
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer:
    '''
    Base trainer class which is overwritten by TrainerDFINE.
    '''

    def __init__(self, config):
        '''
        Initializer of BaseTrainer. 

        Parameters:
        ------------
        - config: yacs.config.CfgNode, yacs config which contains all hyperparameters required to create the DFINE model
                                       Please see config.py for the hyperparameters, their default values and definitions. 
        '''
        
        self.config = config

        # Checkpoint and plot save directories, create directories if they don't exist
        self.ckpt_save_dir = os.path.join(self.config.model.save_dir, 'ckpts'); os.makedirs(self.ckpt_save_dir, exist_ok=True)
        self.plot_save_dir = os.path.join(self.config.model.save_dir, 'plots'); os.makedirs(self.plot_save_dir, exist_ok=True)

        # Training can be continued where it was left of, by default, training start epoch is 1
        self.start_epoch = 1

        # Tensorboard summary writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.config.model.save_dir, 'summary'))
    

    def _save_config(self, config_save_name='config.yaml'):
        '''
        Saves the config inside config.model.save_dir 

        Parameters:
        ------------
        - config_save_name: str, Fullfile name to save the config, 'config.yaml' by default
        '''

        config_save_path = os.path.join(self.config.model.save_dir, config_save_name)
        with open(config_save_path, 'w') as outfile:
            outfile.write(self.config.dump())
    

    def _get_optimizer(self, params):
        '''
        Creates the Adam optimizer with initial learning rate and epsilon specified inside config by config.lr.init and config.optim.eps, respectively
        
        Parameters:
        ------------
        - params: Parameters to be optimized by the optimizer

        Returns:
        ------------
        - optimizer: Adam optimizer with desired learning rate, epsilon to optimize parameters specified by params
        '''

        optimizer = torch.optim.Adam(params=params, 
                                     lr=self.config.lr.init, 
                                     eps=self.config.optim.eps)
        return optimizer


    def _get_lr_scheduler(self):
        '''
        Creates the learning rate scheduler based on scheduler type specified in config.lr.scheduler. Options are constrained by StepLR (explr), CyclicLR (cyclic) and LambdaLR (which is used as constantlr).
        '''

        if self.config.lr.scheduler.lower() == 'explr':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=self.config.lr.explr.gamma, step_size=self.config.lr.explr.step_size)
        elif self.config.lr.scheduler.lower() == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.config.lr.cyclic.base_lr, max_lr=self.config.lr.cyclic.max_lr, mode=self.config.lr.cyclic.mode, gamma=self.config.lr.cyclic.gamma, step_size_up=self.config.lr.cyclic.step_size_up, cycle_momentum=False)
        elif self.config.lr.scheduler.lower() == 'constantlr':
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: 1)
        else:
            assert False, 'Only these learning rate schedulers are available: StepLR (explr), CyclicLR (cyclic) and LambdaLR (which is constantlr)!'
        return scheduler


    def _get_metrics(self):
        '''
        Empty function, overwritten function must return metric names as list metrics as nested dictionary. Keys are: 
            - train: dict, Training Mean metrics
            - valid: dict, Validation Mean metrics
        '''
        pass


    def _reset_metrics(self, train_valid='train'):
        '''
        Resets the metrics 

        Parameters:
        ------------
        - train_valid: str, Which metrics to reset, 'train' by default
        '''

        for _, metric in self.metrics[train_valid].items():
            metric.reset()


    def _update_metrics(self, loss_dict, batch_size, train_valid='train', verbose=True):
        '''
        Updates the metrics

        Parameters:
        ------------
        - loss_dict: dict, Dictionary with loss values to log in Tensorboard
        - batch_size: int, Number of trials for which the metrics are computed for
        - train_valid, str, Which metrics to update, 'train' by default
        - verbose: bool, Whether to print the warning if a key in metric_names doesn't exist in loss_dict
        '''

        for key in self.metric_names:
            if key not in loss_dict:
                if verbose:
                    self.logger.warning(f'{key} does not exist in loss_dict, metric cannot be updated!')
                else:
                    pass
            else:
                self.metrics[train_valid][key].update(loss_dict[key], batch_size)


    def _get_logger(self, prefix='dfine'):
        '''
        Creates the logger which is saved as .log file under config.model.save_dir

        Parameters:
        ------------
        - prefix: str, Prefix which is used as logger's name and .log file's name, 'dfine' by default 

        Returns:
        ------------
        - logger: logging.Logger, Logger object to write logs into .log file
        '''

        os.makedirs(self.config.model.save_dir, exist_ok=True)
        date_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        log_path = os.path.join(self.config.model.save_dir, f'{prefix}_{date_time}.log')

        # from: https://stackoverflow.com/a/56689445/16228104
        logger = logging.getLogger(f'{prefix.upper()} Logger')
        logger.setLevel(logging.DEBUG)

        # Remove old handlers from logger (since logger is static object) so that in several calls, it doesn't overwrite to previous log files
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        
        # Create file handler which logs even debug messages
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.DEBUG)
        
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m/%d/%Y %I:%M:%S %p')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add the handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)

        return logger

    
    def _load_ckpt(self, model, optimizer, lr_scheduler=None):
        '''
        Loads the checkpoint whose number is specified in the config by config.load.ckpt

        Parameters:
        ------------
        - model: torch.nn.Module, Initialized DFINE model to load the parameters to
        - optimizer: torch.optim.Adam, Initialized Adam optimizer to load optimizer parameters to (loading is skipped if config.load.resume_train is False)
        - lr_scheduler: torch.optim.lr_scheduler, Initialized learning rate scheduler to load learning rate scheduler parameters to, None by default (loading is skipped if config.load.resume_train is False)

        Returns:
        ------------
        - model: torch.nn.Module, Loaded DFINE model
        - optimizer: torch.optim.Adam, Loaded Adam optimizer (if config.load.resume_train is True, otherwise, initialized optimizer is returned)
        - lr_scheduler: torch.optim.lr_scheduler, Loaded learning rate scheduler (if config.load.resume_train is True, otherwise, initialized learning rate scheduler is returned)
        '''

        self.logger.warning('Optimizer and LR scheduler can be loaded only in resume_train mode, else they are re-initialized')
        load_path = os.path.join(self.config.model.save_dir, 'ckpts', f'{self.config.load.ckpt}_ckpt.pth')
        self.logger.info(f'Loading model from: {load_path}...')

        # Load the checkpoint 
        try: 
            ckpt = torch.load(load_path)
        except:
            self.logger.error('Ckpt path does not exist!')
            assert False, ''

        # If config.load.resume_train is True, load optimizer and learning rate scheduler
        if self.config.load.resume_train:
            self.start_epoch = ckpt['epoch'] + 1 if isinstance(ckpt['epoch'], int) else 1
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except:
                self.logger.error('Optimizer cannot be loaded!, check if optimizer type is consistent!')
                assert False, ''
            
            if lr_scheduler is not None:
                try:
                    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                except:
                    self.logger.error('LR scheduler cannot be loaded, check if scheduler type is consistent!')
                    assert False, ''

        try:
            model.load_state_dict(ckpt['state_dict'])
        except:
            self.logger.error('Given architecture in config does not match the architecture of given checkpoint!')
            assert False, ''
        
        self.logger.info(f'Checkpoint succesfully loaded from {load_path}!')
        return model, optimizer, lr_scheduler


    def write_model_gradients(self, model, step, prefix='unclipped'):
        '''
        Logs the gradient norms to Tensorboard

        Parameters:
        ------------
        - model: torch.nn.Module, DFINE model whose gradients are to be logged
        - step: int, Step to log gradients for
        - prefix: str, Prefix for gradient norms to be logged, it can be 'clipped' or 'unclipped', 'unclipped' by default
        '''

        total_norm = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norm =  p.grad.detach().data.cpu().norm(2)
                total_norm += grad_norm ** 2
                self.writer.add_scalar('grads/' + name + f"/{prefix}_grad", grad_norm, step)
            
        total_norm = total_norm ** 0.5
        self.writer.add_scalar(f'grads/total_grad_{prefix}_norm', total_norm, step)
        

    def _save_ckpt(self, epoch, model, optimizer, lr_scheduler=None):
        '''
        Saves the checkpoint under ckpt_save_dir (see __init__) with filename {epoch}_ckpt.pth

        Parameters:
        ------------
        - epoch: int, Epoch number for which the checkpoint is to be saved for
        - model: torch.nn.Module, DFINE model to be saved
        - optimizer: torch.optim.Adam, Adam optimizer to be saved
        - lr_scheduler: torch.optim.lr_scheduler, Learning rate scheduler to be saved
        '''

        save_path = os.path.join(self.ckpt_save_dir, f'{epoch}_ckpt.pth')
        if lr_scheduler is not None:
            torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch
                        }, save_path)
        else:
            torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch
                        }, save_path)


    def train_epoch(self, epoch, train_loader):
        '''
        Empty function, overwritten function performs single epoch training

        Parameters:
        ------------
        - epoch: int, Epoch number for which the training iterations are performed
        - train_loader: torch.utils.data.DataLoader, Training dataloader
        '''

        pass


    def valid_epoch(self, epoch, valid_loader): 
        '''
        Empty function, overwritten function performs single epoch validation

        Parameters:
        ------------
        - epoch: int, Epoch number for which the training iterations are performed
        - valid_loader: torch.utils.data.DataLoader, Validation dataloader
        '''

        pass


    def train(self, train_loader, valid_loader):
        '''
        Empty function, overwritten function performs DFINE training for number of epochs specified in config.train.num_epochs

        Parameters:
        ------------
        - train_loader: torch.utils.data.DataLoader, Training dataloader
        - valid_loader: torch.utils.data.DataLoader, Validation dataloader
        '''
        
        pass