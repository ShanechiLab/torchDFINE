'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

from modules.LDM import LDM
from modules.MLP import MLP
from nn import get_kernel_initializer_function, compute_mse, get_activation_function

import torch
import torch.nn as nn


class DFINE(nn.Module):
    '''
    DFINE (Dynamical Flexible Inference for Nonlinear Embeddings) Model. 

    DFINE is a novel neural network model of neural population activity with the ability to perform 
    flexible inference while modeling the nonlinear latent manifold structure and linear temporal dynamics. 
    To model neural population activity, two sets of latent factors are defined: the dynamic latent factors 
    which characterize the linear temporal dynamics on a nonlinear manifold, and the manifold latent factors 
    which describe this low-dimensional manifold that is embedded in the high-dimensional neural population activity space. 
    These two separate sets of latent factors together enable all the above flexible inference properties 
    by allowing for Kalman filtering on the manifold while also capturing embedding nonlinearities.
    Here are some mathematical notations used in this repository:
    - y: The high dimensional neural population activity, (num_seq, num_steps, dim_y). It must be Gaussian distributed, e.g., Gaussian-smoothed firing rates, or LFP, ECoG, EEG
    - a: The manifold latent factors, (num_seq, num_steps, dim_a).
    - x: The dynamic latent factors, (num_seq, num_steps, dim_x).


    * Please note that DFINE can perform learning and inference both for continuous data or trial-based data or segmented continuous data. In the case of continuous data,
    num_seq and batch_size can be set to 1, and we let the model be optimized from the long time-series (this is basically gradient descent and not batch-based gradient descent). 
    In case of trial-based data, we can just pass the 3D tensor as the shape (num_seq, num_steps, dim_y) suggests. In case of segmented continuous data,
    num_seq can be the number of segments and DFINE provides both per-segment and concatenated inference at the end for the user's convenience. In the concatenated inference, 
    the assumption is the concatenation of segments form a continuous time-series (single time-series with batch size of 1).
    '''

    def __init__(self, config):
        '''
        Initializer for an DFINE object. Note that DFINE is a subclass of torch.nn.Module. 

        Parameters: 
        ------------

        - config: yacs.config.CfgNode, yacs config which contains all hyperparameters required to create the DFINE model
                                       Please see config_dfine.py for the hyperparameters, their default values and definitions. 
        '''

        super(DFINE, self).__init__()

        # Get the config and dimension parameters
        self.config = config

        # Set the seed, seed is by default set to a random integer, see config_dfine.py
        torch.manual_seed(self.config.seed)

        # Set the factor dimensions and loss scales
        self._set_dims_and_scales()

        # Initialize LDM parameters
        A, C, W_log_diag, R_log_diag, mu_0, Lambda_0 = self._init_ldm_parameters()

        # Initialize the LDM
        self.ldm = LDM(dim_x=self.dim_x, dim_a=self.dim_a, 
                       A=A, C=C, 
                       W_log_diag=W_log_diag, R_log_diag=R_log_diag,
                       mu_0=mu_0, Lambda_0=Lambda_0,
                       is_W_trainable=self.config.model.is_W_trainable,
                       is_R_trainable=self.config.model.is_R_trainable)

        # Initialize encoder and decoder(s)
        self.encoder = self._get_MLP(input_dim=self.dim_y, 
                                     output_dim=self.dim_a, 
                                     layer_list=self.config.model.hidden_layer_list, 
                                     activation_str=self.config.model.activation)

        self.decoder = self._get_MLP(input_dim=self.dim_a, 
                                     output_dim=self.dim_y, 
                                     layer_list=self.config.model.hidden_layer_list[::-1], 
                                     activation_str=self.config.model.activation)
        
        # If asked to train supervised model, get behavior mapper
        if self.config.model.supervise_behv:
            self.mapper = self._get_MLP(input_dim=self.dim_a, 
                                        output_dim=self.dim_behv, 
                                        layer_list=self.config.model.hidden_layer_list_mapper, 
                                        activation_str=self.config.model.activation_mapper)


    def _set_dims_and_scales(self):
        '''
        Sets the observation (y), manifold latent factor (a) and dynamic latent factor (x)
        (and behavior data dimension if supervised model is to be trained) dimensions,
        as well as behavior reconstruction loss and regularization loss scales from config. 
        '''

        # Set the dimensions
        self.dim_y = self.config.model.dim_y
        self.dim_a = self.config.model.dim_a
        self.dim_x = self.config.model.dim_x

        if self.config.model.supervise_behv:
            self.dim_behv = len(self.config.model.which_behv_dims)
        
        # Set the loss scales for behavior component and for the regularization
        if self.config.model.supervise_behv:
            self.scale_behv_recons = self.config.loss.scale_behv_recons
        self.scale_l2 = self.config.loss.scale_l2


    def _get_MLP(self, input_dim, output_dim, layer_list, activation_str='tanh'):
        '''
        Creates an MLP object

        Parameters:
        ------------
        - input_dim: int, Dimensionality of the input to the MLP network
        - output_dim: int, Dimensionality of the output of the MLP network
        - layer_list: list, List of number of neurons in each hidden layer
        - activation_str: str, Activation function's name, 'tanh' by default

        Returns: 
        ------------
        - mlp_network: an instance of MLP class with desired architecture
        '''

        activation_fn = get_activation_function(activation_str)
        kernel_initializer_fn = get_kernel_initializer_function(self.config.model.nn_kernel_initializer)
    
        mlp_network = MLP(input_dim=input_dim,
                          output_dim=output_dim,
                          layer_list=layer_list,
                          activation_fn=activation_fn,
                          kernel_initializer_fn=kernel_initializer_fn
                          )
        return mlp_network


    def _init_ldm_parameters(self):
        '''
        Initializes the LDM Module parameters

        Returns:
        ------------
        - A: torch.Tensor, shape: (self.dim_x, self.dim_x), State transition matrix of LDM
        - C: torch.Tensor, shape: (self.dim_a, self.dim_x), Observation matrix of LDM
        - W_log_diag: torch.Tensor, shape: (self.dim_x, ), Log-diagonal of dynamics noise covariance matrix (W, therefore it is diagonal and PSD)
        - R_log_diag: torch.Tensor, shape: (self.dim_a, ), Log-diagonal of observation noise covariance matrix  (R, therefore it is diagonal and PSD)
        - mu_0: torch.Tensor, shape: (self.dim_x, ), Dynamic latent factor prediction initial condition (x_{0|-1}) for Kalman filtering
        - Lambda_0: torch.Tensor, shape: (self.dim_x, self.dim_x), Dynamic latent factor estimate error covariance initial condition (P_{0|-1}) for Kalman filtering

        * We learn the log-diagonal of matrix W and R to satisfy the PSD constraint for cov matrices. Diagnoal W and R are used for the stability of learning 
        similar to prior latent LDM works, see (Kao et al., Nature Communications, 2015) & (Abbaspourazad et al., IEEE TNSRE, 2019) for further info
        '''

        kernel_initializer_fn = get_kernel_initializer_function(self.config.model.ldm_kernel_initializer)
        A = kernel_initializer_fn(self.config.model.init_A_scale * torch.eye(self.dim_x, dtype=torch.float32)) 
        C = kernel_initializer_fn(self.config.model.init_C_scale * torch.randn(self.dim_a, self.dim_x, dtype=torch.float32)) 

        W_log_diag = torch.log(kernel_initializer_fn(torch.diag(self.config.model.init_W_scale * torch.eye(self.dim_x, dtype=torch.float32))))
        R_log_diag = torch.log(kernel_initializer_fn(torch.diag(self.config.model.init_R_scale * torch.eye(self.dim_a, dtype=torch.float32))))
        
        mu_0 = kernel_initializer_fn(torch.zeros(self.dim_x, dtype=torch.float32))
        Lambda_0 = kernel_initializer_fn(self.config.model.init_cov * torch.eye(self.dim_x, dtype=torch.float32))

        return A, C, W_log_diag, R_log_diag, mu_0, Lambda_0

    
    def forward(self, y, mask=None):
        '''
        Forward pass for DFINE Model

        Parameters: 
        ------------
        - y: torch.Tensor, shape: (num_seq, num_steps, dim_y), High-dimensional neural observations
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether 
                                                              observations at each timestep exist (1) or are missing (0)

        Returns: 
        ------------
        - model_vars: dict, Dictionary which contains learned parameters, inferrred latents, predictions and reconstructions. Keys are: 
            - a_hat: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of projected manifold latent factors. 
            - a_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_a), Batch of predicted estimates of manifold latent factors (last index of the second dimension is removed)
            - a_filter: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of filtered estimates of manifold latent factors 
            - a_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of smoothed estimates of manifold latent factors 
            - x_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_x), Batch of predicted estimates of dynamic latent factors
            - x_filter: torch.Tensor, shape: (num_seq, num_steps, dim_x), Batch of filtered estimates of dynamic latent factors
            - x_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_x), Batch of smoothed estimates of dynamic latent factors
            - Lambda_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_x, dim_x), Batch of predicted estimates of dynamic latent factor estimation error covariance
            - Lambda_filter: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Batch of filtered estimates of dynamic latent factor estimation error covariance
            - Lambda_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Batch of smoothed estimates of dynamic latent factor estimation error covariance
            - y_hat: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of projected estimates of neural observations
            - y_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_y), Batch of predicted estimates of neural observations
            - y_filter: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of filtered estimates of neural observations
            - y_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of smoothed estimates of neural observations
            - A: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Repeated (tile) state transition matrix of LDM, same for each time-step in the 2nd axis
            - C: torch.Tensor, shape: (num_seq, num_steps, dim_y, dim_x), Repeated (tile) observation matrix of LDM, same for each time-step in the 2nd axis
            - behv_hat: torch.Tensor, shape: (num_seq, num_steps, dim_behv), Batch of reconstructed behavior. None if unsupervised model is trained

        * Terminology definition:
            projected: noisy estimations of manifold latent factors after nonlinear manifold embedding via encoder 
            predicted: one-step ahead predicted estimations (t+1|t), the first and last time indices are (1|0) and (T|T-1)
            filtered: causal estimations (t|t)
            smoothed: non-causal estimations (t|T)
        '''

        # Get the dimensions from y
        num_seq, num_steps, _ = y.shape

        # Create the mask if it's None
        if mask is None:
            mask = torch.ones(y.shape[:-1], dtype=torch.float32).unsqueeze(dim=-1)

        # Get the encoded low-dimensional manifold factors (project via nonlinear manifold embedding) -> the outputs are (num_seq * num_steps, dim_a)
        a_hat = self.encoder(y.view(-1, self.dim_y))

        # Reshape the manifold latent factors back into 3D structure (num_seq, num_steps, dim_a)
        a_hat = a_hat.view(-1, num_steps, self.dim_a)

        # Run LDM to infer filtered and smoothed dynamic latent factors
        x_pred, x_filter, x_smooth, Lambda_pred, Lambda_filter, Lambda_smooth = self.ldm(a=a_hat, mask=mask, do_smoothing=True)
        A = self.ldm.A.repeat(num_seq, num_steps, 1, 1)
        C = self.ldm.C.repeat(num_seq, num_steps, 1, 1)
        a_pred = (C @ x_pred.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)
        a_filter = (C @ x_filter.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)
        a_smooth = (C @ x_smooth.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)
        
        # Remove the last timestep of predictions since it's T+1|T, which is not of our interest
        x_pred = x_pred[:, :-1, :]
        Lambda_pred = Lambda_pred[:, :-1, :, :]
        a_pred = a_pred[:, :-1, :]

        # Supervise a_seq or a_smooth to behavior if requested -> behv_hat shape: (num_seq, num_steps, dim_behv)
        if self.config.model.supervise_behv:
            if self.config.model.behv_from_smooth:
                behv_hat = self.mapper(a_smooth.view(-1, self.dim_a))
            else:
                behv_hat = self.mapper(a_hat.view(-1, self.dim_a))
            behv_hat = behv_hat.view(-1, num_steps, self.dim_behv)
        else:
            behv_hat = None

        # Get filtered and smoothed estimates of neural observations. To perform k-step-ahead prediction, 
        # get_k_step_ahead_prediction(...) function should be called after the forward pass. 
        y_hat = self.decoder(a_hat.view(-1, self.dim_a))
        y_pred = self.decoder(a_pred.reshape(-1, self.dim_a))
        y_filter = self.decoder(a_filter.view(-1, self.dim_a))
        y_smooth = self.decoder(a_smooth.view(-1, self.dim_a))

        y_hat = y_hat.view(num_seq, -1, self.dim_y)
        y_pred = y_pred.view(num_seq, -1, self.dim_y)
        y_filter = y_filter.view(num_seq, -1, self.dim_y)
        y_smooth = y_smooth.view(num_seq, -1, self.dim_y)

        # Dump inferrred latents, predictions and reconstructions to a dictionary
        model_vars = dict(a_hat=a_hat, a_pred=a_pred, a_filter=a_filter, a_smooth=a_smooth, 
                          x_pred=x_pred, x_filter=x_filter, x_smooth=x_smooth,
                          Lambda_pred=Lambda_pred, Lambda_filter=Lambda_filter, Lambda_smooth=Lambda_smooth,
                          y_hat=y_hat, y_pred=y_pred, y_filter=y_filter, y_smooth=y_smooth, 
                          A=A, C=C, behv_hat=behv_hat)
        return model_vars


    def get_k_step_ahead_prediction(self, model_vars, k):
        '''
        Performs k-step ahead prediction of manifold latent factors, dynamic latent factors and neural observations. 

        Parameters: 
        ------------
        - model_vars: dict, Dictionary returned after forward(...) call. See the definition of forward(...) function for information. 
            - x_filter: torch.Tensor, shape: (num_seq, num_steps, dim_x), Batch of filtered estimates of dynamic latent factors
            - A: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x) or (dim_x, dim_x), State transition matrix of LDM
            - C: torch.Tensor, shape: (num_seq, num_steps, dim_y, dim_x) or (dim_y, dim_x), Observation matrix of LDM
        - k: int, Number of steps ahead for prediction

        Returns: 
        ------------
        - y_pred_k: torch.Tensor, shape: (num_seq, num_steps-k, dim_y), Batch of predicted estimates of neural observations, 
                                                                           the first index of the second dimension is y_{k|0}
        - a_pred_k: torch.Tensor, shape: (num_seq, num_steps-k, dim_a), Batch of predicted estimates of manifold latent factor, 
                                                                        the first index of the second dimension is a_{k|0}                                                              
        - x_pred_k: torch.Tensor, shape: (num_seq, num_steps-k, dim_x), Batch of predicted estimates of dynamic latent factor, 
                                                                        the first index of the second dimension is x_{k|0}  
        '''

        # Check whether provided k value is valid or not
        if k <= 0 or not isinstance(k, int):
            assert False, 'Number of steps ahead prediction value is invalid or of wrong type, k must be a positive integer!'

        # Extract the required variables from model_vars dictionary
        x_filter = model_vars['x_filter']
        A = model_vars['A']
        C = model_vars['C']

        # Get the required dimensions
        num_seq, num_steps, _ = x_filter.shape

        # Check if shapes of A and C are 4D where first 2 dimensions are (number of trials/time segments) and (number of steps)
        if len(A.shape) == 2:
            A = A.repeat(num_seq, num_steps, 1, 1)

        if len(C.shape) == 2:
            A = A.repeat(num_seq, num_steps, 1, 1)

        # Here is where k-step ahead prediction is iteratively performed
        x_pred_k = x_filter[:, :-k, ...] # [x_k|0, x_{k+1}|1, ..., x_{T}|{T-k}]
        for i in range(1, k+1):
            if i != k:
                x_pred_k = (A[:, i:-(k-i), ...] @ x_pred_k.unsqueeze(dim=-1)).squeeze(dim=-1)  
            else:
                x_pred_k = (A[:, i:, ...] @ x_pred_k.unsqueeze(dim=-1)).squeeze(dim=-1)
        a_pred_k = (C[:, k:, ...] @ x_pred_k.unsqueeze(dim=-1)).squeeze(dim=-1)

        # After obtaining k-step ahead predicted manifold latent factors, they're decoded to obtain k-step ahead predicted neural observations
        y_pred_k = self.decoder(a_pred_k.view(-1, self.dim_a))

        # Reshape mean and variance back to 3D structure after decoder (num_seq, num_steps, dim_y)
        y_pred_k = y_pred_k.reshape(num_seq, -1, self.dim_y)

        return y_pred_k, a_pred_k, x_pred_k

    
    def compute_loss(self, y, model_vars, mask=None, behv=None):
        '''
        Computes k-step ahead predicted MSE loss, regularization loss and behavior reconstruction loss
        if supervised model is being trained. 

        Parameters: 
        ------------
        - y: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of high-dimensional neural observations
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether 
                                                              observations at each timestep exists (1) or are missing (0)
                                                              if None it will be set to ones.
        - model_vars: dict, Dictionary returned after forward(...) call. See the definition of forward(...) function for information. 
        - behv: torch.tensor, shape: (num_seq, num_steps, dim_behv), Batch of behavior data

        Returns: 
        ------------
        - loss: torch.Tensor, shape: (), Loss to optimize, which is sum of k-step-ahead MSE loss, L2 regularization loss and 
                                         behavior reconstruction loss if model is supervised
        - loss_dict: dict, Dictionary which has all loss components to log on Tensorboard. Keys are (e.g. for config.loss.steps_ahead = [1, 2]): 
            - steps_{k}_mse: torch.Tensor, shape: (), {k}-step ahead predicted masked MSE, k's are determined by config.loss.steps_ahead
            - model_loss: torch.Tensor, shape: (), Negative of sum of all steps_{k}_mse
            - behv_loss: torch.Tensor, shape: (), Behavior reconstruction loss, 0 if model is unsupervised
            - reg_loss: torch.Tensor, shape: (), L2 Regularization loss for DFINE encoder and decoder weights
            - total_loss: torch.Tensor, shape: (), Sum of model_loss, behv_loss and reg_loss
        '''

        # Create the mask if it's None
        if mask is None:
            mask = torch.ones(y.shape[:-1], dtype=torch.float32).unsqueeze(dim=-1)

        # Dump individual loss values for logging or Tensorboard
        loss_dict = dict()
        
        # Iterate over multiple steps ahead
        k_steps_mse_sum = 0  
        for _, k in enumerate(self.config.loss.steps_ahead):
            y_pred_k, _, _ = self.get_k_step_ahead_prediction(model_vars, k=k)
            mse_pred = compute_mse(y_flat=y[:, k:, :].reshape(-1, self.dim_y), 
                                   y_hat_flat=y_pred_k.reshape(-1, self.dim_y),
                                   mask_flat=mask[:, k:, :].reshape(-1,))
            k_steps_mse_sum += mse_pred
            loss_dict[f'steps_{k}_mse'] = mse_pred

        model_loss = k_steps_mse_sum
        loss_dict['model_loss'] = model_loss

        # Get MSE loss for behavior reconstruction, 0 if we dont supervise our model with behavior data
        if self.config.model.supervise_behv:
            behv_mse = compute_mse(y_flat=behv[..., self.config.model.which_behv_dims].reshape(-1, self.dim_behv), 
                                   y_hat_flat=model_vars['behv_hat'].reshape(-1, self.dim_behv),
                                   mask_flat=mask.reshape(-1,))
            behv_loss = self.scale_behv_recons * behv_mse
        else:
            behv_mse = torch.tensor(0, dtype=torch.float32, device=model_loss.device)
            behv_loss = torch.tensor(0, dtype=torch.float32, device=model_loss.device)
        loss_dict['behv_mse'] = behv_mse
        loss_dict['behv_loss'] = behv_loss

        # L2 regularization loss 
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + self.scale_l2 * torch.norm(param)
        loss_dict['reg_loss'] = reg_loss

        # Final loss is summation of model loss (sum of k-step ahead MSEs), behavior reconstruction loss and L2 regularization loss
        loss = model_loss + behv_loss + reg_loss
        loss_dict['total_loss'] = loss
        return loss, loss_dict

        


                    
 
        
        

