'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

import torch 
import torch.nn as nn


def compute_mse(y_flat, y_hat_flat, mask_flat=None):
    '''
    Returns average Mean Square Error (MSE) 

    Parameters:
    ------------
    - y_flat: torch.Tensor, shape: (num_samp, dim_y), True data to compute MSE of  
    - y_hat_flat: torch.Tensor, shape: (num_samp, dim_y), Predicted/Reconstructed data to compute MSE of  
    - mask_flat: torch.Tensor, shape: (num_samp, 1), Mask to compute MSE loss which shows whether 
                                                     observations at each timestep exists (1) or are missing (0)

    Returns:
    ------------    
    - mse: torch.Tensor, Average MSE 
    '''

    if mask_flat is None: 
        mask_flat = torch.ones(y_flat.shape[:-1], dtype=torch.float32)

    # Make sure mask is 2D
    if len(mask_flat.shape) != len(y_flat.shape):
        mask_flat = mask_flat.unsqueeze(dim=-1)

    # Compute the MSEs and mask the timesteps where observations are missing
    mse = (y_flat - y_hat_flat) ** 2
    mse = torch.mul(mask_flat, mse)
    
    # Return the mean of the mse (over available observations)
    if mask_flat.shape[-1] != y_flat.shape[-1]: # which means shape of mask_flat is of dimension 1
        num_el = mask_flat.sum() * y_flat.shape[-1]
    else:
        num_el = mask_flat.sum()

    mse = mse.sum() / num_el
    return mse


def get_activation_function(activation_str):
    '''
    Returns activation function given the activation function's name

    Parameters:
    ----------------------
    - activation_str: str, Activation function's name

    Returns:
    ----------------------
    - activation_fn: torch.nn, Activation function  
    '''

    if activation_str.lower() == 'elu':
        return nn.ELU()
    elif activation_str.lower() == 'hardtanh':
        return nn.Hardtanh()
    elif activation_str.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation_str.lower() == 'relu':
        return nn.ReLU()
    elif activation_str.lower() == 'rrelu':
        return nn.RReLU()
    elif activation_str.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation_str.lower() == 'mish':
        return nn.Mish()
    elif activation_str.lower() == 'tanh':
        return nn.Tanh()
    elif activation_str.lower() == 'tanhshrink':
        return nn.Tanhshrink()
    elif activation_str.lower() == 'linear':
        return lambda x: x

def get_kernel_initializer_function(kernel_initializer_str):
    '''
    Returns kernel initialization function given the kernel initialization function's name

    Parameters:
    ----------------------
    - kernel_initializer_str: str, Kernel initialization function's name

    Returns:
    ----------------------
    - kernel_initializer_fn: torch.nn.init, Kernel initialization function  
    '''

    if kernel_initializer_str.lower() == 'uniform':
        return nn.init.uniform_
    elif kernel_initializer_str.lower() == 'normal':
        return nn.init.normal_
    elif kernel_initializer_str.lower() == 'xavier_uniform':
        return nn.init.xavier_uniform_
    elif kernel_initializer_str.lower() == 'xavier_normal':
        return nn.init.xavier_normal_
    elif kernel_initializer_str.lower() == 'kaiming_uniform':
        return nn.init.kaiming_uniform_
    elif kernel_initializer_str.lower() == 'kaiming_normal':
        return nn.init.kaiming_normal_
    elif kernel_initializer_str.lower() == 'orthogonal':
        return nn.init.orthogonal_
    elif kernel_initializer_str.lower() == 'default':
        return lambda x:x