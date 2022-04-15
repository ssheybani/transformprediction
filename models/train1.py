# defines the actual training loop for the model. 
# This code interacts with the optimizer and handles logging during training.

# Lots of relevant content in nma_models.py
#%% Imports
import copy, os, itertools
from functools import partialmethod
import warnings

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import torch, torchvision
from torch import nn
from tqdm import tqdm as tqdm


#%% Network Classes
torch.set_default_dtype(torch.float32)

def to_static(img_seq):
    n_samples, seq_len, *other_dims = img_seq.shape
    return img_seq.reshape((n_samples*seq_len, *other_dims))

def to_seq(img_flt, seq_len):
    n_samplesXseq_len, *other_dims = img_flt.shape
    n_samples = int(n_samplesXseq_len/seq_len)
    return img_flt.reshape((n_samples, seq_len, *other_dims))

#%%
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from pytorchvideo.transforms.functional import uniform_temporal_subsample

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

transform = Compose(
        [
            Lambda(lambda x: x/255.0),
            Lambda(lambda x: torch.permute(x, (3,0,1,2))),
            NormalizeVideo(mean, std),
            Lambda(lambda x: torch.permute(x, (1,0,2,3)))
        ])

# Notes on ShortSideScale
# Determines the shorter spatial dim of the video (i.e. width or height) and scales it to the given size. To maintain aspect ratio, the longer side is then scaled accordingly. :param x: A video tensor of shape (C, T, H, W) and type torch.float32. :type x: torch.Tensor :param size: The size the shorter side is scaled to. :type size: int :param interpolation: Algorithm used for upsampling,
# https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/transforms/functional.html#short_side_scale

#%%% Random Video Dataset


#%%% Create the dataloader
vid_root = r'C:/Users/Saber/Documents/tmp_dir/CharadesEgo_preproc/'
# sample_paths = os.listdir(vid_root)
# exclude the non-ego videos
sample_paths = [s for s in os.listdir(vid_root)]

def train_test_split_idx(n_samples, fraction_train=0.8):
    sample_idx = np.random.permutation(n_samples)
    n_train = int(n_samples*fraction_train)
    train_idx, test_idx = sample_idx[:n_train], sample_idx[n_train:]
    return train_idx, test_idx
n_samples = len(sample_paths)
train_idx, test_idx = train_test_split_idx(n_samples, fraction_train=0.8)

train_data = [sample_paths[i] for i in train_idx]

class VidTorchDataset(torch.utils.data.Dataset):
    def __init__(self, root, fnames, transform=None):
        self.root = root
        self.fnames = fnames
        self.transform = transform
    def __getitem__(self, idx):
        # assuming idx is an int
        vid = torchvision.io.read_video(
            self.root+self.fnames[idx])[0]
        if self.transform is not None:
            vid = self.transform(vid)
        return vid
    def __len__(self):
        return len(self.fnames)

train_data_te = VidTorchDataset(vid_root, train_data, 
                                transform=transform)
train_dataloader = torch.utils.data.DataLoader(
    train_data_te, batch_size=3, shuffle=True)


for xclips in train_dataloader:
    break

xtt = to_static(xclips)

# ytt = xmodel(xtt)
# xtt = torchvision.io.read_video('005BU_tr.mp4')[0]




#%% Instantiation
xmodel = torchvision.models.resnet50(pretrained=False)

xmodel.float()
# xmodel.train()

#%%
# optimizer = torch.optim.Adam([
#     {'params': xmodel.shared_encoder.parameters()},
#     {'params': xmodel.ventral.parameters(), 'lr': 1e-3},
#     {'params': xmodel.dorsal.parameters(), 'lr': 1e-4}
#     ], lr=1e-1, weight_decay=0, amsgrad=False)

optimizer = torch.optim.SGD([
    {'params': xmodel.shared_encoder.parameters()},
    {'params': xmodel.ventral.parameters(), 'lr': 1e-3},
    {'params': xmodel.dorsal.parameters(), 'lr': 1e-4}
    ], lr=1e-1, weight_decay=0, amsgrad=False)
    
# optimizer_name="SGD":optimizer_hparams={
    # "lr": 0.1,"momentum": 0.9,"weight_decay": 1e-4})  
    
MSELoss = nn.MSELoss()
# for clip, (dlabel, clabel) in train_dataloader:

  # optimizer.zero_grad()
  # yv, yd = xmodel(clip)
  
  # loss_d = 10*torch.mean(yd - clabel, dim=(0,1,2))
  # # loss_v = nn.CrossEntropyLoss(
  # #     yv.view(batch_size*seq_len, -1), 
  # #     dlabel.view(batch_size*seq_len, -1))
  # loss = loss_d
  # print('loss: ',loss.item())
  
  # loss.backward()
  # optimizer.step()
  # break
  

#%% Training loop

seq_len = 10
batch_size = 32

num_epochs = 20
loss_list = []
iteration_list = []
test_loss_list = [] ## accuracy_list = []
count = 0
for epoch in range(num_epochs):
    for i, (clips, (dlabels, clabels)) in enumerate(train_dataloader):
        xmodel.train()
        # train  = images.view(-1, seq_len, input_dim)
            
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        yv, yd = xmodel(clips)
        
        # loss_d = 10*torch.mean(yd - clabels, dim=(0,1,2))
        loss_d = MSELoss(yd, clabels)#.type(torch.FloatTensor)
        
        # loss_v = nn.CrossEntropyLoss(
        #     yv.view(batch_size*seq_len, -1), 
        #     dlabel.view(batch_size*seq_len, -1))
        loss = loss_d
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 5 == 0:#250 == 0:
            # Calculate Accuracy         
            # correct = 0
            # total = 0; 
            test_loss = [] #@@
            xmodel.eval()
            
            # Iterate through test dataset
            for clips, (dlabels, clabels) in test_dataloader:
                # images = images.view(-1, seq_len, input_dim)
                
                # Forward propagation
                yv, yd = xmodel(clips)
                
                loss_d = 10*torch.mean(abs(yd - clabels), dim=(0,1))
                test_loss.append(loss_d.detach().numpy())
                # # Get predictions from the maximum value
                # predicted = torch.max(outputs.data, 1)[1]
                
                # # Total number of labels
                # total += 1#labels.shape[0]
                
                # correct += (predicted == labels).sum()
            test_loss = np.mean(np.asarray(test_loss), axis=0)
            # accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data.item())
            iteration_list.append(count)
            # accuracy_list.append(accuracy)
            test_loss_list.append(test_loss)
            if count % 5 == 0:
                # Print Loss
                print('Iteration: {}  Train Loss: {}  Test Loss: {} %'.format(count, loss.data.item(), test_loss))

# from . import data, plot_util


# Other methods in nma_models.py :
# def plot_model_RSMs(encoders, dataset, sampler, titles=None,
# class EncoderCore(nn.Module):

# Classification-related
# def train_classifier(encoder, dataset, train_sampler, test_sampler,
# def train_clfs_by_fraction_labelled(encoder, dataset, train_sampler,

# VAE-specific
# class VAE_decoder(nn.Module):
# def vae_loss_function(recon_X_logits, X, mu, logvar, beta=1.0):
# def train_vae(encoder, dataset, train_sampler, num_epochs=100, batch_size=500,
# def plot_vae_reconstructions(encoder, decoder, dataset, indices, title=None,

# SimCLR specific
# def contrastive_loss(proj_feat1, proj_feat2, temperature=0.5, neg_pairs="all"):
# def train_simclr(encoder, dataset, train_sampler, num_epochs=50,