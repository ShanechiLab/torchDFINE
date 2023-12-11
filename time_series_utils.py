'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

from python_utils import convert_to_tensor
import torch
from scipy.stats import pearsonr


def get_nrmse_error(y, y_hat, version_calculation='modified'):
    '''
    Computes normalized root-mean-squared error between two 3D tensors. Note that this operation is not symmetric. 

    Parameters: 
    ------------
    - y: torch.Tensor/np.ndarray, shape: (num_seq, num_steps, dim_y), Tensor with true observations
    - y_hat: torch.Tensor/np.ndarray, shape: (num_seq, num_steps, dim_y), Tensor with reconstructed/estimated observations
    - version_calculation: str, Version to calculate the variance. If 'regular', variance of each sequence is computed separately, 
                                which may result in unstable nrmse value since some sequences may be constant or close to being constant and
                                results in ~0 variance, so high/unreasonable nrmse. To prevent that, variance is computed across flattened sequence in 
                                'modified' mode. 'modified' by default.

    Returns: 
    ------------
    - normalized_error: torch.Tensor, shape: (dim_y,), Normalized root-mean-squared error for each data dimension
    - normalized_error_mean: torch.Tensor, shape: (), Average normalized root-mean-squared error for each data dimension
    '''

    # Check if dimensions are consistent
    assert y.shape == y_hat.shape, f'dimensions of y {y.shape} and y_hat {y_hat.shape} do not match'
    assert len(y.shape) == 3, 'mismatch in x dimension: x should be in the format of (num_seq, num_steps, dim_x)'

    y = convert_to_tensor(y).detach().cpu()
    y_hat = convert_to_tensor(y_hat).detach().cpu()
    
    # carry time to first dimension
    y = torch.permute(y, (1,0,2)) # (num_steps, num_seq, dim_x)
    y_hat = torch.permute(y_hat, (1,0,2)) # (num_steps, num_seq, dim_x)

    recons_error = torch.mean(torch.square(y - y_hat), dim=0)

    # way 1 to calculate variance
    if version_calculation == 'regular':
        var_y = torch.mean(torch.square(y - torch.mean(y, dim=0)), dim=0)

    # way 2 to calculate variance (sometime data in a batch is flat, it's more robust to calculate variance globally)
    elif version_calculation == 'modified':
        y_resh = torch.reshape(y, (-1, y.shape[2]))
        var_y = torch.mean(torch.square(y_resh - torch.mean(y_resh, dim=0)), dim=0)
        var_y = torch.tile(var_y.unsqueeze(dim=0), (y.shape[1], 1))
    normalized_error = torch.mean((torch.sqrt(recons_error) / torch.sqrt(var_y)), dim=0) # mean across batches 
    normalized_error_mean = torch.mean(normalized_error)

    return normalized_error, normalized_error_mean


def get_rmse_error(y, y_hat):    
    '''
    Computes root-mean-squared error between two 3D tensors
    
    Parameters: 
    ------------
    - y: torch.Tensor/np.ndarray, shape: (num_seq, num_steps, dim_y), Tensor with true observations
    - y_hat: torch.Tensor/np.ndarray, shape: (num_seq, num_steps, dim_y), Tensor with reconstructed/estimated observations

    Returns: 
    ------------
    - rmse: torch.Tensor, shape: (dim_y,), Root-mean-squared error for each data dimension
    - rmse_mean: torch.Tensor, shape: (), Average root-mean-squared error for each data dimension
    '''

    # Check if dimensions are consistent
    assert y.shape == y_hat.shape, f'dimensions of y {y.shape} and y_hat {y_hat.shape} do not match'

    if len(y.shape) == 3:
        dim_y = y.shape[-1]
        y = y.reshape(-1, dim_y)
        y_hat = y_hat.reshape(-1, dim_y)

    y = convert_to_tensor(y).detach().cpu()
    y_hat = convert_to_tensor(y_hat).detach().cpu()

    rmse = torch.sqrt(torch.mean(torch.square(y-y_hat), dim=0))
    rmse_mean = torch.nanmean(rmse.nan_to_num(posinf=torch.nan, neginf=torch.nan)) # for stability purposes
    return rmse, rmse_mean


def get_pearson_cc(y, y_hat):     
    '''
     Computes Pearson correlation coefficient across two 2D (If 3D tensors are given, they're reshaped across 1st and 2nd dimensions) tensors across first (time) dimension.

     Parameters: 
     ------------
     - y: torch.Tensor/np.ndarray, shape: (num_seq, num_steps, dim_y) or (num_steps, dim_y), Tensor with true observations
     - y_hat: torch.Tensor/np.ndarray, shape: (num_seq, num_steps, dim_y) or (num_steps, dim_y), Tensor with reconstructed/estimated observations

     Returns: 
     ------------
     - ccs: torch.Tensor, shape: (dim_y,), Pearson correlation coefficients computed across first (time) dimension
     - ccs_mean: torch.Tensor, shape: (), Pearson correlation coefficients computed across first (time) dimension and averaged across data dimensions
     '''

    assert y.shape == y_hat.shape, f'dimensions of x {y.shape} and xhat {y_hat.shape} do not match'

    if len(y.shape) == 3:
        dim_y = y.shape[-1]
        y = y.reshape(-1, dim_y)
        y_hat = y_hat.reshape(-1, dim_y)

    y = convert_to_tensor(y).detach().cpu().numpy() # make sure every array/tensor has .numpy() function, pearsonr works on ndarrays
    y_hat = convert_to_tensor(y_hat).detach().cpu().numpy()

    ccs = []
    for dim in range(y.shape[-1]):
        cc, _ = pearsonr(y[:, dim], y_hat[:, dim])
        ccs.append(cc)

    ccs = torch.tensor(ccs, dtype=torch.float32)
    ccs_mean = torch.nanmean(ccs.nan_to_num(posinf=torch.nan, neginf=torch.nan))
    return ccs, ccs_mean



def z_score_tensor(y, fit=True, **kwargs):
    '''
    Performs z-scoring fitting and transformation.

    Parameters: 
    ------------
    - y: torch.Tensor/np.ndarray, shape: (num_seq, num_steps, dim_y) or (num_steps, dim_y), Tensor/array to z-score 
                                                                                           (and if fit is True, to learn mean and standard deviation)
    - fit: bool, Whether to learn mean and standard deviation from y. If False, learnt 'mean' and 'std' should be provided as keyword arguments. 
    - mean: torch.Tensor, shape: (), Mean to transform y. If fit is True, it's not necessary to provide since mean is going to be learnt. 0 by default.  
    - std: torch.Tensor, shape: (), Standard deviation to transform y. If fit is True, it's not necessary to provide since std is going to be learnt. 1 by default.  

    Returns: 
    ------------
    y_z_scored: torch.Tensor/np.ndarray, shape: (num_seq, num_steps, dim_y) or (num_steps, dim_y), Z-scored tensor/array
    mean: torch.Tensor, shape: (), Learnt mean. If fit is True, it's the mean provided via keyword, or default
    mean: torch.Tensor/np.ndarray, Learnt standard deviation. If fit is True, it's the std provided via keyword, or default
    '''
    
    # Make sure that gradients are turned off
    with torch.no_grad():
        y = convert_to_tensor(y)

        y_resh = y.reshape(-1, y.shape[-1])
        if fit:
            mean = torch.mean(y_resh, dim=0)
            std = torch.std(y_resh, dim=0)
        else:
            mean = kwargs.pop('mean', 0)
            std = kwargs.pop('std', 1)  

        # to prevent nan values
        std[std==0] = 1
        
        y_resh = (y_resh - mean) / std
        y_z_scored = y_resh.reshape(y.shape)
        return y_z_scored, mean, std  

