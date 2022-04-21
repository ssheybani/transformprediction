# defines the actual training loop for the model. 
# This code interacts with the optimizer and handles logging 
# during training.

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
# from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     ShortSideScale,
#     UniformTemporalSubsample
# )

# from pytorchvideo.transforms.functional import uniform_temporal_subsample

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

# Given the conflict between pytorchvideo and other packages, it'd be better to write
# uniform_temporal_subsample and ShortSideScale manually.


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
    train_data_te, batch_size=8, shuffle=True)


for xclips in train_dataloader:
    break

xtt = to_static(xclips)


# xtt = torchvision.io.read_video('005BU_tr.mp4')[0]






#%% Try a contrastive loss

# import torch
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, reduction='none')
anchor = torch.randn(100, 128, requires_grad=True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)
output = triplet_loss(anchor, positive, negative).detach().numpy()
# the loss values are distributed exponentially decaying from zero, with 60% 
# being less than 1 and 85% being less than 2.

ytt = np.histogram(output)
# import matplotlib.pyplot as plt
# _ = plt.hist(output)
# output.backward()

# With a network
ytt = xmodel(xtt)

ytt = to_seq(ytt, seq_len)

a_t = np.random.randint(0,seq_len)
p_dif = 2
n_dif = int(seq_len/2)
anchor = ytt[:, a_t, :]
positive = ytt[:, a_t-p_dif, :]
negative = ytt[:, a_t-n_dif, :]

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
cpc_loss = triplet_loss(anchor, positive, negative)#.detach().numpy()

# xmodel.train()

def get_feat_extr_output_size(model, input_dim):
    dummy_tensor = torch.ones(1, *input_dim)
    # reset_training = model.training
    with torch.no_grad():   
        output_dim = model(dummy_tensor).shape
    return np.product(output_dim), output_dim[1:]

input_dim = (3,64,64)
print(get_feat_extr_output_size(xmodel, input_dim))

#%% Instantiation
xmodel = torchvision.models.resnet50(pretrained=False)
hdim = 1000
seq_len = xclips.shape[1]
xmodel.fc = nn.Linear(512, hdim)
# nn.init.kaiming_normal_(xmodel.fc.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

xmodel.float()
xmodel.train()

# optimizer = torch.optim.Adam([
#     {'params': xmodel.shared_encoder.parameters()},
#     {'params': xmodel.ventral.parameters(), 'lr': 1e-3},
#     {'params': xmodel.dorsal.parameters(), 'lr': 1e-4}
#     ], lr=1e-1, weight_decay=0, amsgrad=False)

optimizer = torch.optim.Adam(xmodel.parameters(), lr=1e-3, 
                            weight_decay=0)

# optimizer = torch.optim.SGD(xmodel.parameters(), lr=1e-2, 
#                             weight_decay=1e-4)
    
# optimizer_name="SGD":optimizer_hparams={
    # "lr": 0.1,"momentum": 0.9,"weight_decay": 1e-4})  
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')

# MSELoss = nn.MSELoss()

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

seq_len = 14
batch_size = 8

num_epochs = 1
loss_list = []
iteration_list = []
test_loss_list = [] ## accuracy_list = []
count = 0
for epoch in range(num_epochs):
    for i, clips in enumerate(train_dataloader):
        
        xmodel.train()
        # clips = to_static()
        # train  = images.view(-1, seq_len, input_dim)
            
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        yh = to_seq( 
            xmodel(to_static(xclips)),
            seq_len)
        
        a_t = np.random.randint(0,seq_len)
        p_dif = 2
        n_dif = int(seq_len/2)
        anchor = yh[:, a_t, :]
        positive = yh[:, a_t-p_dif, :]
        negative = yh[:, a_t-n_dif, :]

        cpc_loss = triplet_loss(anchor, positive, negative)#.detach().numpy()
        
        
        # loss_v = nn.CrossEntropyLoss(
        #     yv.view(batch_size*seq_len, -1), 
        #     dlabel.view(batch_size*seq_len, -1))
        loss = cpc_loss
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        print('Iteration: {}  Train Loss: {}  '.format(count, loss.data.item()))
        
        if count % 5 == 0:#250 == 0:
            # Calculate Accuracy         
            # correct = 0
            # total = 0; 
            # test_loss = [] #@@
            # xmodel.eval()
            
            # Iterate through test dataset
            # for clips in test_dataloader:
            #     yh = to_seq( 
            #         xmodel(to_static(clips)),
            #         seq_len)
                
                
            #     loss_d = 10*torch.mean(abs(yd - clabels), dim=(0,1))
            #     test_loss.append(loss_d.detach().numpy())
            #     # # Get predictions from the maximum value
            #     # predicted = torch.max(outputs.data, 1)[1]
                
            #     # # Total number of labels
            #     # total += 1#labels.shape[0]
                
            #     # correct += (predicted == labels).sum()
            # test_loss = np.mean(np.asarray(test_loss), axis=0)
            # # accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data.item())
            iteration_list.append(count)
            # accuracy_list.append(accuracy)
            # test_loss_list.append(test_loss)
            # if count % 5 == 0:
                # Print Loss
            print('Iteration: {}  Train Loss: {}  '.format(count, loss.data.item()))

#%%

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