import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from Network import network
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
import glob
import os

# DataLoader class
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def create_dataloaders(x_train_tensor,x_val_tensor,batch_size):
    """Convert the data into dataloader.
    """
    train_dataset = MyDataset(x_train_tensor,x_train_tensor)
    val_dataset = MyDataset(x_val_tensor,x_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader,val_loader

def get_split_data(x,split_fraction=0.85):
        
    """Split the data 85% for training anf 15% for the validation. 
    """
    np.random.shuffle(x)
    train_images = x[:int(split_fraction*x.shape[0])]
    val_images = x[int(split_fraction*x.shape[0]):]
    return train_images, val_images
    


def preprocess(train_patches,val_patches):

    """Convert the data into tensors to support for trianing. 
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
                           noise_model, hf,reg_parameter,method,n_depth,max_epochs,logger,
                           checkpoint_callback,train_loader,val_loader,
                           kl_annealing, weights_summary):
    """Create instance for VAE network and fit the model.
    """
    
    for filename in glob.glob(basedir+"/*"):
            os.remove(filename) 
    
    vae = network.VAELightning(method, reg_parameter,hf= hf, data_mean = data_mean,
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


