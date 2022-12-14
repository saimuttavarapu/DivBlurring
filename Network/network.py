import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import pytorch_lightning as pl
from loss_function import loss_fn
from tqdm import tqdm

class VAELightning(pl.LightningModule):
    """ The model architecture with encoder and decoder.
    """
    def __init__(self, method, reg_parameter,hf, data_mean, data_std,  gaussian_noise_std=None, noise_model=None,  kl_annealing = False,
                 kl_start=0, kl_annealtime=10, z_dim=64, lr=0.001, in_channels = 1, init_filters = 32, 
                 n_filters_per_depth=2, n_depth=2,kernel_size=3, stride=1, padding=1, bias=True, groups=1, 
                 num_samples=100, kl_min=1e-7):
        super(VAELightning, self).__init__()
        self.method= method,
        self.reg_parameter = reg_parameter,
        self.hf = hf,
        self.data_mean = data_mean
        self.data_std = data_std
        self.gaussian_noise_std = gaussian_noise_std
        self.noise_model = noise_model
        self.kl_start = kl_start
        self.kl_annealtime = kl_annealtime
        self.z_dim = z_dim
        self.lr = lr
        self.in_channels = in_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.n_depth = n_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.kl_annealing = kl_annealing
        self.groups = groups
        self.num_samples=num_samples
        self.kl_min=kl_min
        self.collapse = False
        
        self.encoder = Encoder(z_dim=self.z_dim, in_channels = self.in_channels, init_filters = self.init_filters, n_filters_per_depth=self.n_filters_per_depth, n_depth=self.n_depth, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups=self.groups)
               
        self.decoder = Decoder(z_dim=self.z_dim, in_channels = self.in_channels, init_filters = self.init_filters, n_filters_per_depth=self.n_filters_per_depth, n_depth=self.n_depth, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups=self.groups)
        
        self.save_hyperparameters('method','reg_parameter','hf','data_mean',
                                  'data_std',
                                  'gaussian_noise_std',
                                  'kl_annealing',
                                  'kl_start',
                                  'kl_annealtime',
                                  'z_dim',
                                  'in_channels',
                                  'n_depth',
                                  'init_filters',
                                  'n_filters_per_depth',
                                  'lr')
        
        
    def encode(self, x):
#         print("encode in vaelighing")
        x_= (x-self.data_mean) / self.data_std
        return self.encoder(x_)

    def decode(self, z):
#         print("decode in vaelighing")
        out = (self.decoder(z)* self.data_std)+self.data_mean 
        return out

    def reparameterize(self, mu, logvar):
#         print("reparametrize in vaelighing")
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(mu)
        z = mu + epsilon*std
        return z

    def forward(self, x, tqdm_bar=False):
#         print("forward in vaelighing")
        x = (x-self.data_mean) / self.data_std
        mu, logvar = self.encoder(x)
#         print('mu and logvar')
#         print(mu.shape, logvar.shape)
        sample_list = []
        for i in range(self.num_samples):
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            sample_list.append(recon.detach().cpu().numpy())
        return sample_list
        
    def configure_optimizers(self):
#         print("config params in vaelighing")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=30,factor=0.5,min_lr=1e-12,verbose=False)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
       }
    
    def step(self, x):
#         print("step in vaelighing")

        x_normalized = (x-self.data_mean) / self.data_std
        mu, logvar = self.encoder(x_normalized)
        z = self.reparameterize(mu, logvar)
        recon_normalized = self.decoder(z)

        reconstruction_loss, kl_loss, regulariser = loss_fn(self.method ,self.reg_parameter, self.hf, recon_normalized, x_normalized, mu, logvar, self.gaussian_noise_std, self.data_mean, self.data_std, self.noise_model)
        return reconstruction_loss, kl_loss, regulariser, recon_normalized, x_normalized
    
    def get_kl_weight(self):
        if(self.kl_annealing==True):
            #calculate weight
            kl_weight = (self.current_epoch-self.kl_start) * (1.0/self.kl_annealtime)
            # clamp to [0,1]
            kl_weight = min(max(0.0,kl_weight),1.0)
            
        else:
            kl_weight = 1.0 
        
        return kl_weight

    def training_step(self, batch, batch_idx):
#         print("traiing_step in vaelighing")
        x_denormalized, y_denormalized = batch
        reconstruction_loss, kl_loss,regulariser, pred, x = self.step(x_denormalized)
        kl_weight = self.get_kl_weight()
        loss = reconstruction_loss + kl_weight*kl_loss + regulariser
         #check for posterior collapse
        # if kl_loss < self.kl_min:
        #     print('postersior collapse: aborting')
        #     self.trainer.should_stop=True
        #     self.collapse=True
            
        self.log('reconstruction_loss', reconstruction_loss, on_epoch=True)
        self.log('kl_loss', kl_weight*kl_loss, on_epoch=True)
        self.log('training_loss', loss, on_epoch=True)
        self.log('kl_weight', kl_weight, on_epoch=True)
        self.log('regulariser', regulariser, on_epoch=True)
        self.log('lr', self.lr, on_epoch=True)
        output = {
          'loss': loss,  
          'reconstruction_loss': reconstruction_loss,
          'kl_loss': kl_weight*kl_loss,
          'regulariser': regulariser,
          
          'kl_weight': kl_weight
    }
        return output         
    
    def training_epoch_end(self, training_step_outputs):

        reco_loss = training_step_outputs[-1]['reconstruction_loss'].item()
        kl_loss = training_step_outputs[-1]['kl_loss'].item()      
        loss = training_step_outputs[-1]['loss'].item()
        kl_weight = training_step_outputs[-1]['kl_weight']
        if(self.method[0] !='DivBlurring'):
            regulariser = training_step_outputs[-1]['regulariser'].item()


        
        # RC_loss_global.append(reco_loss)
        # regulariser_global.append(regulariser)
        
        # KL_loss_global.append(kl_loss)
        
        # total_loss_global.append(loss)
        
        # KL_weight_global.append(kl_weight)
        # print("kl_weight:"+str(kl_weight))
        # print("reco_loss:"+str(reco_loss))
        # print("kl_loss:"+str(kl_loss))
        # print("regulariser:"+str(regulariser))
        # print("loss:"+str(loss))
        
    def log_images_for_tensorboard(self, pred, x, img_mmse):
        clamped_input = torch.clamp((x - x.min()) / (x.max() - x.min()), 0, 1)
        clamped_pred = torch.clamp((pred - pred.min()) / (pred.max() - pred.min()), 0, 1)
        clamped_mmse = torch.clamp((img_mmse - img_mmse.min()) / (img_mmse.max() - img_mmse.min()), 0, 1)
        self.trainer.logger.experiment.add_image('inputs/img', clamped_input[0],
                                                     self.current_epoch)
        for i in range(7):
            self.trainer.logger.experiment.add_image('predcitions/sample_{}'.format(i), clamped_pred[i],
                                                     self.current_epoch)
        self.trainer.logger.experiment.add_image('predcitions/mmse (100 samples)', clamped_mmse[0],
                                                     self.current_epoch)
    def validation_step(self, batch, batch_idx):
        x_denormalized, y_denormalized = batch
        reconstruction_loss, kl_loss, regulariser, pred, x  = self.step(x_denormalized)
        val_loss = reconstruction_loss + kl_loss + regulariser
        self.log('val_loss', val_loss, on_epoch=True)
        if batch_idx == 0:
            all_samples = self(x_denormalized[0:1,...])
            img_mmse = torch.from_numpy(np.mean(np.array(all_samples)[:,0,...],axis=0,keepdims=True))
            self.log_images_for_tensorboard(torch.from_numpy(np.array(all_samples)[:,0,...]), x[0:1,...], img_mmse)
        return val_loss
    
    def validation_epoch_end(self, validation_step_output):
        val_loss = validation_step_output[0].item()
        


class DownConv(nn.Module):
    """Down covolutoin model for the enocder in main model.
    """
    def __init__(self, in_channels = 1, out_channels = 32, init_filters = 32, n_filters_per_depth=2,
                 kernel_size=3, stride=1, padding=1, bias=True, groups=1):

        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        
        ins = self.in_channels
        outs = self.out_channels
        
        self.conv1 = nn.Conv2d(in_channels=ins, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.conv2 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        if(self.n_filters_per_depth==3):
            self.conv3 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)    
    
    def forward(self,x):
#         print("forward in downcon")
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if(self.n_filters_per_depth==3):
            x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x
    
    
class UpConv(nn.Module):
    """Up covolutoin model for the decoder in main model.
    """
    def __init__(self, in_channels = 1, out_channels = 32, init_filters = 32, n_filters_per_depth=2, 
                 kernel_size=3, stride=1, padding=1, bias=True, groups=1):

        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        
        ins = self.in_channels
        outs = self.out_channels
        
        self.conv1 = nn.Conv2d(in_channels=ins, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.conv2 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        if(self.n_filters_per_depth==3):
            self.conv3 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.convtranspose = nn.ConvTranspose2d(in_channels=outs, out_channels=outs, kernel_size=2, stride=2)    
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if(self.n_filters_per_depth==3):
            x = F.relu(self.conv3(x))
        x = self.convtranspose(x)
        return x

    
class Encoder(nn.Module):
    def __init__(self, z_dim=4, in_channels = 1, init_filters = 32, n_filters_per_depth=2, n_depth=2,
                 kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        """
        Encoder pathway. It performs encoding operation and returns 
        latent space mean and log varaiance of the encoder distribution.
        """
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.n_depth = n_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.down_convs = []
        
        self.convmu = nn.Conv2d(in_channels=self.init_filters*(2**(self.n_depth-1)), out_channels=self.z_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.convlogvar = nn.Conv2d(in_channels=self.init_filters*(2**(self.n_depth-1)), out_channels=self.z_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        
        for i in range(n_depth):
            ins = self.in_channels if i==0 else outs
            outs = self.init_filters*(2**i)
            down_conv = DownConv(in_channels=ins, out_channels = outs, init_filters = self.init_filters, n_filters_per_depth=self.n_filters_per_depth)
            self.down_convs.append(down_conv)
     
        self.down_convs = nn.ModuleList(self.down_convs)
   
            
    def forward(self, x):
        # encoder pathway
        
        for i, module in enumerate(self.down_convs):
            x = module(x)
        
        mu = self.convmu(x)
        logvar = self.convlogvar(x)
        return mu, logvar
    
    
class Decoder(nn.Module):
    def __init__(self, z_dim=4, in_channels = 1, init_filters = 32, n_filters_per_depth=2, n_depth=2,
                 kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        """
        Decoder pathway. It performs decoding operation using the latent sample, 
        latent mean and latent log variance and returns the output image.
        """
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.n_depth = n_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.up_convs = []
        
        self.convrecon = nn.Conv2d(in_channels=self.init_filters, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups=self.groups)

        for i in reversed(range(n_depth)):
            ins = self.z_dim if i==(n_depth-1) else outs
            outs = self.init_filters*(2**i)
            up_conv = UpConv(in_channels=ins, out_channels = outs, init_filters = self.init_filters, n_filters_per_depth=self.n_filters_per_depth)
            self.up_convs.append(up_conv)
            
        self.up_convs = nn.ModuleList(self.up_convs)
        
    def forward(self, x):
        for i, module in enumerate(self.up_convs):
            x = module(x)

        recon = self.convrecon(x)
        return recon


