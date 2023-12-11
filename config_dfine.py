'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

from python_utils import flatten_dict, unflatten_dict

from yacs.config import CfgNode as CN
import torch


#### Initialization of default and recommended (except dimensions and hidden layer lists, set them suitable for data to fit) config 
_config = CN() 

## Set device and seed
_config.device = 'cpu'
_config.seed = int(torch.randint(low=0, high=100000, size=(1,)))

## Dump model related settings
_config.model = CN() 

# Hidden layer list where each element is the number of neurons for that hidden layer of DFINE encoder/decoder. Please use [20,20,20,20] for nonlinear manifold simulations.
_config.model.hidden_layer_list = [32,32,32] 
# Activation function used in encoder and decoder layers
_config.model.activation = 'tanh'
# Dimensionality of neural observations
_config.model.dim_y = 30
# Dimensionality of manifold latent factor, a choice higher than dim_y (above) may lead to overfitting
_config.model.dim_a = 16
# Dimensionality of dynamic latent factor, it's recommended to set it same as dim_a (above), please see Extended Data Fig. 8
_config.model.dim_x = 16
# Initialization scale of LDM state transition matrix
_config.model.init_A_scale = 1
# Initialization scale of LDM observation matrix
_config.model.init_C_scale = 1
# Initialization scale of LDM process noise covariance matrix
_config.model.init_W_scale = 0.5
# Initialization scale of LDM observation noise covariance matrix
_config.model.init_R_scale = 0.5
# Initialization scale of dynamic latent factor estimation error covariance matrix
_config.model.init_cov = 1
# Boolean for whether process noise covariance matrix W is learnable or not
_config.model.is_W_trainable = True
# Boolean for whether observation noise covariance matrix R is learnable or not
_config.model.is_R_trainable = True
# Initialization type of LDM parameters, see nn.get_kernel_initializer_function for detailed definition and supported types
_config.model.ldm_kernel_initializer = 'default'
# Initialization type of DFINE encoder and decoder parameters, see nn.get_kernel_initializer_function for detailed definition and supported types
_config.model.nn_kernel_initializer = 'xavier_normal'
# Boolean for whether to learn a behavior-supervised model or not. It must be set to True if supervised model will be trained.  
_config.model.supervise_behv = False
# Hidden layer list for the behavior mapper where each element is the number of neurons for that hidden layer of the mapper
_config.model.hidden_layer_list_mapper = [20,20,20]
# Activation function used in mapper layers
_config.model.activation_mapper = 'tanh'
# List of dimensions of behavior data to be decoded by mapper, check for any dimensionality mismatch 
_config.model.which_behv_dims = [0,1,2,3]
# Boolean for whether to decode behavior from a_smooth
_config.model.behv_from_smooth = True
# Main save directory for DFINE results, plots and checkpoints
_config.model.save_dir = 'D:/DATA/DFINE_results'
# Number of steps to save DFINE checkpoints
_config.model.save_steps = 10

## Dump loss related settings
_config.loss = CN()

# L2 regularization loss scale (we recommend a grid-search for the best value, i.e., a grid of [1e-4, 5e-4, 1e-3, 2e-3]). Please use 0 for nonlinear manifold simulations as it leads to a better performance. 
_config.loss.scale_l2 = 2e-3
# List of number of steps ahead for which DFINE is optimized. For unsupervised and supervised versions, default values are [1,2,3,4] and [1,2], respectively. 
_config.loss.steps_ahead = [1,2,3,4]
# If _config.model.supervise_behv is True, scale for MSE of behavior reconstruction (We recommend a grid-search for the best value. It should be set to a large value).
_config.loss.scale_behv_recons = 20

## Dump training related settings
_config.train = CN()

# Batch size 
_config.train.batch_size = 32
# Number of epochs for which DFINE is trained
_config.train.num_epochs = 200
# Number of steps to check validation data performance
_config.train.valid_step = 1 
# Number of steps to save training/validation plots
_config.train.plot_save_steps = 50
# Number of steps to print training/validation logs
_config.train.print_log_steps = 10

## Dump loading settings
_config.load = CN()

# Number of checkpoint to load
_config.load.ckpt = -1
# Boolean for whether to resume training from the epoch where checkpoint is saved
_config.load.resume_train = False

## Dump learning rate related settings
_config.lr = CN()

# Learning rate scheduler type, options are explr (StepLR, purely exponential if explr.step_size == 1), cyclic (CyclicLR) or constantlr (constant learning rate, no scheduling)
_config.lr.scheduler = 'explr'
# Initial learning rate
_config.lr.init = 0.02

# Dump cyclic LR scheduler related settings, check https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html for details
_config.lr.cyclic = CN()
# Minimum learning rate for cyclic LR scheduler
_config.lr.cyclic.base_lr = 0.005
# Maximum learning rate for cyclic LR scheduler
_config.lr.cyclic.max_lr = 0.02
# Envelope scale for exponential cyclic LR scheduler mode
_config.lr.cyclic.gamma = 1
# Mode for cyclic LR scheduler
_config.lr.cyclic.mode = 'triangular'
# Number of iterations in the increasing half of the cycle
_config.lr.cyclic.step_size_up = 10

# Dump exponential LR scheduler related settings, check https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html for details
_config.lr.explr = CN()
# Multiplicative factor of learning rate decay
_config.lr.explr.gamma = 0.9
# Steps to decay the learning rate, becomes purely exponential if step is 1
_config.lr.explr.step_size = 15

## Dump optimizer related settings
_config.optim = CN()

# Epsilon for Adam optimizer
_config.optim.eps = 1e-8
# Gradient clipping norm    
_config.optim.grad_clip = 1


def get_default_config():
    '''
    Creates the default config

    Returns: 
    ------------
    - config: yacs.config.CfgNode, default DFINE config
    '''

    return _config.clone()


def update_config(config, new_config):
    '''
    Updates the config

    Parameters:
    ------------
    - config: yacs.config.CfgNode or dict, Config to update 
    - new_config: yacs.config.CfgNode or dict, Config with new settings and appropriate keys

    Returns: 
    ------------
    - unflattened_config: yacs.config.CfgNode, Config with updated settings
    '''

    # Flatten both configs
    flat_config = flatten_dict(config)
    flat_new_config = flatten_dict(new_config)

    # Update and unflatten the config to return
    flat_config.update(flat_new_config)
    unflattened_config = CN(unflatten_dict(flat_config))

    return unflattened_config





    

    

    

    

    
















                    

                    
















