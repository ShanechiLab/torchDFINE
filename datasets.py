'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

from torch.utils.data import Dataset
import torch 


class DFINEDataset(Dataset):
    '''
    Dataset class for DFINE. 
    '''
    
    def __init__(self, y, behv=None, mask=None):
        '''
        Initializer for DFINEDataset. Note that this is a subclass of torch.utils.data.Dataset. \

        Parameters: 
        ------------
        - y: torch.Tensor, shape: (num_seq, num_steps, dim_y), High dimensional neural observations. 
        - behv: torch.Tensor, shape: (num_seq, num_steps, dim_behv), Behavior data. None by default.
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask for manifold latent factors which shows whether 
                                                              observations at each timestep exists (1) or are missing (0). 
                                                              None by default. 
        '''
        
        self.y = y

        # If behv is not provided, initialize it by zeros. 
        if behv is None:
            self.behv = torch.zeros(y.shape[:-1], dtype=torch.float32).unsqueeze(dim=-1)
        else:
            self.behv = behv
        
        # If mask is not provided, initialize it by ones. 
        if mask is None:
            self.mask = torch.ones(y.shape[:-1], dtype=torch.float32).unsqueeze(dim=-1)
        else:
            self.mask = mask


    def __len__(self):
        '''
        Returns the length of the dataset 
        '''

        return self.y.shape[0]


    def __getitem__(self, idx):
        '''
        Returns a tuple of neural observations, behavior and mask segments
        '''

        return self.y[idx, :, :], self.behv[idx, :, :], self.mask[idx, :, :]