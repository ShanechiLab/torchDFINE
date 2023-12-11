'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

from torchmetrics import Metric
import torch 


class Mean(Metric):
    '''
    Mean metric class to log batch-averaged metrics to Tensorboard. 
    '''

    def __init__(self):
        '''
        Initializer for Mean metric. Note that this class is a subclass of torchmetrics.Metric.
        '''

        super().__init__(dist_sync_on_step=False)
        
        # Define total sum and number of samples that sum is computed over
        self.add_state("sum", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")


    def update(self, value, batch_size):
        '''
        Updates the total sum and number of samples

        Parameters: 
        ------------
        - value: torch.Tensor, shape: (), Value to add to sum
        - batch_size: torch.Tensor, shape: (), Number of samples that 'value' is averaged over
        '''

        value = value.clone().detach()
        batch_size = torch.tensor(batch_size, dtype=torch.float32)
        self.sum += value.cpu() * batch_size
        self.num_samples += batch_size


    def reset(self):
        '''
        Resets the total sum and number of samples to 0
        '''

        self.sum = torch.tensor(0, dtype=torch.float32)
        self.num_samples = torch.tensor(0, dtype=torch.float32)


    def compute(self):
        '''
        Computes the mean metric.

        Returns: 
        ------------
        - avg: Average value for the metric
        '''

        avg = self.sum / self.num_samples
        return avg
