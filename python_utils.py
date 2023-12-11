'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

import torch
import numpy as np


def carry_to_device(data, device, dtype=torch.float32):    
    '''
    Carries dict/list of torch Tensors/numpy arrays to desired device recursively

    Parameters: 
    ------------
    - data: torch.Tensor/np.ndarray/dict/list: Dictionary/list of torch Tensors/numpy arrays or torch Tensor/numpy array to be carried to desired device
    - device: str, Device name to carry the torch Tensors/numpy arrays to
    - dtype: torch.dtype, Data type for torch.Tensor to be returned, torch.float32 by default

    Returns: 
    ------------
    - data: torch.Tensor/dict/list: Dictionary/list of torch.Tensors or torch Tensor carried to desired device
    '''

    if torch.is_tensor(data):
        return data.to(device)
    
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=dtype).to(device)

    elif isinstance(data, dict):
        for key in data.keys():
            data[key] = carry_to_device(data[key], device)
        return data
    
    elif isinstance(data, list):
        for i, d in enumerate(data):
            data[i] = carry_to_device(d, device)
        return data

    else:
        return data


def convert_to_tensor(x, dtype=torch.float32):
    '''
    Converts numpy.ndarray to torch.Tensor

    Parameters: 
    ------------
    - x: np.ndarray, Numpy array to convert to torch.Tensor (if it's of type torch.Tensor already, it's returned without conversion)
    - dtype: torch.dtype, Data type for torch.Tensor to be returned, torch.float32 by default

    Returns: 
    ------------
    - y: torch.Tensor, Converted tensor
    '''

    if isinstance(x, torch.Tensor):
        y = x
    elif isinstance(x, np.ndarray):
        y = torch.tensor(x, dtype=dtype) # use np.ndarray as middle step so that function works with tf tensors as well
    else:
        assert False, 'Only Numpy array can be converted to tensor'
    return y


def flatten_dict(dictionary, level=[]):
    '''
    Flattens nested dictionary by putting '.' between nested keys, reference: https://stackoverflow.com/questions/6037503/python-unflatten-dict
    
    Parameters: 
    ------------
    - dictionary: dict, Nested dictionary to be flattened
    - level: list, List of strings for recursion, initialized by empty list

    Returns: 
    ------------
    - tmp_dict: dict, Flattened dictionary
    '''

    tmp_dict = {}
    for key, val in dictionary.items():
        if isinstance(val, dict):
            tmp_dict.update(flatten_dict(val, level + [key]))
        else:
            tmp_dict['.'.join(level + [key])] = val
    return tmp_dict
    

def unflatten_dict(dictionary):    
    '''
    Unflattens a flattened dictionary whose keys are joint string of nested keys separated by '.', reference: https://stackoverflow.com/questions/6037503/python-unflatten-dict
    
    Parameters: 
    ------------
    - dictionary: dict, Flat dictionary to be unflattened

    Returns: 
    ------------
    - resultDict: dict, Unflattened dictionary
    '''

    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict

