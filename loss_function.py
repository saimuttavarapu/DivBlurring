import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from scipy.stats import norm
import sys
sys.path.append('../../')
device = torch.device("cuda:0")


def loss_fn(method, reg_parameter,hf,predicted_s, x, mu, logvar, gaussian_noise_std, data_mean, data_std, noiseModel):
    """The loss function for divblurring aling with regularisers.
    """
    method = method[0]
    reg_parameter= reg_parameter[0]

    if(method=='DivBlurring'):  # DivBlurring with no regulariser.
        regulariser = reg_parameter

    elif(method=='DivBlurring_l1'): # DivBlurring with l1 regulariser.
        regulariser = reg_parameter*torch.sum(torch.abs(predicted_s))

    elif(method=='DivBlurring_l2'): # DivBlurring with l2 regulariser.
        regulariser = reg_parameter*torch.linalg.norm(predicted_s)**2

    elif(method=='DivBlurring_PCReg_1e3'): # DivBlurring with positivity contraint regulariser.
        sample = predicted_s.clone()
        regulariser = reg_parameter*positivitycontraint(sample)

    elif(method=='DivBlurring_PCReg_1e5'): # DivBlurring with positivity contraint regulariser.
        sample = predicted_s.clone()
        regulariser = reg_parameter*positivitycontraint(sample)

    elif(method=='DivBlurring_PCReg_l1'): # DivBlurring with positivity contraint regulariser and l1 regulariser..
        sample = predicted_s.clone()
        regulariser = reg_parameter[0]*positivitycontraint(sample) + reg_parameter[1]*torch.sum(torch.abs(predicted_s))

    else:
        regulariser = 0

    hf = torch.from_numpy(np.asarray(hf)).to(device='cuda') 
    predicted_s = torch.real(torch.fft.ifft2(hf*torch.fft.fft2(predicted_s))).to(device)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    reconstruction_loss = torch.mean(((predicted_s)-x)**2) / (2.0* (gaussian_noise_std/data_std)**2 )

    return reconstruction_loss, kl_loss /float(x.numel()), regulariser


def positivitycontraint(u):
    """positivity contraint regulariser for one of the mentioned regularisers.
    """
    u[u > 0] = 0
    return torch.sum(torch.square(u))

