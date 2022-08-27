import warnings
warnings.filterwarnings('ignore')
import torch
import os
import urllib
import zipfile
from torch.distributions import normal
import matplotlib.pyplot as plt, numpy as np, pickle
from scipy.stats import norm
import sys
from sklearn.feature_extraction import image
from tqdm import tqdm
sys.path.append('../../')
device = torch.device("cuda:0")


def loss_fn(predicted_s, x, mu, logvar, gaussian_noise_std, data_mean, data_std, noiseModel):
    reg_parameter = 1e-10
    hf = convolution(3,256)
    regulariser = reg_parameter*torch.sum(torch.abs(predicted_s))
    # print(regulariser)
    predicted_s = torch.real(torch.fft.ifft2(hf*torch.fft.fft2(predicted_s))).to(device)

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    reconstruction_loss = torch.mean(((predicted_s)-x)**2) / (2.0* (gaussian_noise_std/data_std)**2 )
#     print(reconstruction_loss)
    return reconstruction_loss, kl_loss /float(x.numel()), regulariser


def positivitycontrain(u):
        u[u > 0] = 0
        return torch.sum(torch.square(u))

#Convolution
def convolution(sigma, shape):
        sigma = sigma
        n = shape
        t = np.concatenate( (np.arange(0,n/2+1), np.arange(-n/2,-1)) )
        [Y,X] = np.meshgrid(t,t)
        h = np.exp( -(X**2+Y**2)/(2.0*float(sigma)**2) )
        h = h/np.sum(h)
        hf = np.real(np.fft.fft2(h))
        hf = torch.from_numpy(hf).to(device='cuda') 
        return hf