import warnings

from loss_function import convolution
warnings.filterwarnings('ignore')
import torch
import os
import urllib
import zipfile
from torch.distributions import normal
import matplotlib.pyplot as plt, numpy as np, pickle
from scipy.stats import norm
from tifffile import imread
import sys
from sklearn.feature_extraction import image
from tqdm import tqdm
sys.path.append('../../')

import numpy as np
import time
from glob import glob
from tifffile import imsave
from sklearn.cluster import MeanShift
from matplotlib import pyplot as plt
from IPython.display import clear_output
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import pytorch_lightning as pl

import logging
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger



import network


def get_split_data(x,split_fraction=0.85):
        
    """Split the data 85% fro training anf 15% for the trianing. 
    """
    np.random.shuffle(x)
    train_images = x[:int(split_fraction*x.shape[0])]
    val_images = x[int(split_fraction*x.shape[0]):]
    return train_images, val_images
    


def preprocess(train_patches,val_patches):

    """Conver the data into tensors to support for trianing. 
    """
    data_mean, data_std = getMeanStdData(train_patches, val_patches)
    x_train, x_val = convertToFloat32(train_patches,val_patches)
    x_train_extra_axis = x_train[:,np.newaxis]
    x_val_extra_axis = x_val[:,np.newaxis]
    x_train_tensor = convertNumpyToTensor(x_train_extra_axis)
    x_val_tensor = convertNumpyToTensor(x_val_extra_axis)
    return x_train_tensor, x_val_tensor, data_mean, data_std


def getMeanStdData(train_images,val_images):
    """Compute mean and standrad deviation of data. 
    """
    x_train_ = train_images.astype('float32')
    x_val_ = val_images.astype('float32')
    data = np.concatenate((x_train_,x_val_), axis=0)
    mean, std = np.mean(data), np.std(data)
    return mean, std


def convertToFloat32(train_images,val_images):

    """Converts the data to float 32 bit type. 
        """
    x_train = train_images.astype('float32')
    x_val = val_images.astype('float32')
    return x_train, x_val


def convertNumpyToTensor(numpy_array):

    """Convert numpy array to PyTorch tensor. 
    """
    return torch.from_numpy(numpy_array)


def create_model_and_train(basedir,data_mean,data_std,gaussian_noise_std,
                           noise_model,n_depth,max_epochs,logger,
                           checkpoint_callback,train_loader,val_loader,
                           kl_annealing, weights_summary):
    
#     for filename in glob.glob(basedir+"/*"):
#             os.remove(filename) 
    
    vae = network.VAELightning(data_mean = data_mean,
                                      data_std = data_std, 
                                      gaussian_noise_std = gaussian_noise_std,
                                      noise_model = noise_model,
                                      n_depth=n_depth,
                                      kl_annealing = kl_annealing)
#     print("instnce vae created")
#     print(vae)
    if torch.cuda.is_available():
        trainer = network.pl.Trainer(gpus=1, max_epochs=max_epochs, logger=logger,
                             callbacks=
                             [EarlyStopping(monitor='val_loss', min_delta=1e-6, 
                              patience = 100, verbose = True, mode='min'),checkpoint_callback], weights_summary=weights_summary)
    else:
        trainer = network.pl.Trainer(max_epochs=max_epochs, logger=logger,
                             callbacks=
                             [EarlyStopping(monitor='val_loss', min_delta=1e-6, 
                              patience = 100, verbose = True, mode='min'),checkpoint_callback], weights_summary=weights_summary)
#     print("fitting strated")
    trainer.fit(vae, train_loader, val_loader)
#     print('fitting ended')
    collapse_flag = vae.collapse
    return collapse_flag , vae


