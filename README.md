# Torch DFINE
PyTorch implementation of DFINE: Dynamical Flexible Inference for Nonlinear Embeddings 

DFINE is a neural network model of neural population activity that is developed to enable accurate and
flexible inference, whether causally in real time, non-causally, or even in the presence of missing neural observations. 
Also, DFINE enables recursive and thus computationally efficient inference for real-time implementation.
DFINE's capabilities are important for applications such as neurotechnologies and brain-computer interfaces.

More information about the model and its training and inference methods can be found inside [tutorial.ipynb](tutorial.ipynb) and in our manuscript below.

## Publication

Abbaspourazad, H.\*, Erturk, E.\*, Pesaran, B., & Shanechi, M. M. Dynamical flexible inference of nonlinear latent factors and structures in neural population activity. _Nature Biomedical Engineering_ (2023). https://www.nature.com/articles/s41551-023-01106-1

Original preprint: https://www.biorxiv.org/content/10.1101/2023.03.13.532479v1

## Installation 
Torch DFINE requires Python version 3.8.* or 3.9.*. After the virtual environment with compatible Python version is set up, 
navigate to project folder and simply run the following command:

```
pip install -r requirements.txt
```

Then, navigate to the virtual environment's site-packages directory, create a file with .pth extension and copy the 
main project directory path (e.g. .../torchDFINE) into that .pth file. This will allow importing the desired modules by using subdirectories, 
for instance, TrainerDFINE class can be imported by ```from trainers.TrainerDFINE import TrainerDFINE```.

## DFINE Tutorial
Please see [tutorial.ipynb](tutorial.ipynb) for further information and guidelines on DFINE's model, training, and inference. 

## Licence
Copyright (c) 2023 University of Southern California  <br />
See full notice in [LICENSE.md](LICENSE.md)  <br />
Hamidreza Abbaspourazad\*, Eray Erturk\* and Maryam M. Shanechi  <br />
Shanechi Lab, University of Southern California





