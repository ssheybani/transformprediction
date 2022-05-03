#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import socket
# print(socket.gethostname())


import sys
sys.path.insert(0,'/geode2/home/u080/sheybani/BigRed200/spenv/lib/python3.10/site-packages')
print(sys.path)

import torch
if not torch.cuda.is_available():
    raise ValueError('Cuda not available!')


# In[98]:


# Load the Basic files

import numpy as np
import torch, torchvision
from torchvision.transforms import ToTensor, Normalize, ConvertImageDtype
from torchvision import transforms as T
from torch import nn
from torch.nn import functional as F
import os
import random
#import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# In[3]:


vid_root = '/N/slate/sheybani/ego4ddata/pp_images/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


# fnames = os.listdir(vid_root)
fnames = []
for froot, _, fname in os.walk(vid_root):
    if len(fname)>0:
        new_names = [os.path.join(froot,item) for item in fname]
        fnames +=new_names
# def get_picnum(fpath):
#     return int(fpath.split('/')[-1].split('.')[0])
fnames.sort()#key=get_picnum)


# In[54]:


# # get mean, std of the colors
# s_rate = 2
# transform0 = T.Compose([T.ConvertImageDtype(torch.float32)])
# train_data_te = TimeContrastiveTorchDataset(train_data, pdif=1, ndif=5*s_rate,
#                                 transform=transform0)

# means = []
# stds = []

# n_subset = 1000
# rand_idx = np.random.randint(len(train_data_te), size=n_subset)
# t0 = time.time()
# for i in range(n_subset):
#     img, _,_ = train_data_te[rand_idx[i]]
#     means.append(img.view(3,-1).mean(dim=1).numpy())
    
#     if i%100==0:
#         print('i=',i)

# print(time.time()-t0)
# mean = np.asarray(means).mean(axis=0)
# print('mean=',mean)

# mean_t = torch.FloatTensor(mean).view(3,1)
# sse = []
# for i in range(n_subset):
#     img, _,_ = train_data_te[rand_idx[i]]
#     sse.append(((img.view(3,-1)-mean_t)**2).mean(dim=1).numpy())
# #     means.append(img.view(3,-1).mean(dim=1).numpy())
# #     stds.append(img.view(3,-1).std(dim=1).numpy())
# #     stds.append(img.std(dim=1).std(dim=1).numpy())
    
#     if i%100==0:
#         print('i=',i)
# sse = np.asarray(sse)
# std = np.sqrt(sse.mean(axis=0))
# print('std=',std)


# In[102]:


# Create a Dataloader for the Time contrastive loss.
s_rate = 2

mean, std = (0.413, 0.373, 0.340), (0.251, 0.236, 0.235)
# transforms_ego4d = nn.Sequential(
#     ConvertImageDtype(torch.float32),
#     Normalize(mean,std),
#     )

transforms_ego4d = nn.Sequential(
    ConvertImageDtype(torch.float32),
    Normalize(mean,std),
    )

#     [
#      NormalizeVideo(mean, std),
#      Lambda(lambda x: torch.permute(x, (1,2,3,0)))
#      ])


class TimeContrastiveTorchDataset(torch.utils.data.Dataset):
    def __init__(self, fnames, pdif=1, ndif=5*s_rate, transform=None):
#         self.root = root
        self.fnames = fnames
        self.transform = transform
        self.pdif = pdif
        self.ndif = ndif #depends on the activity (perhaps some inherent period of the chore being done) 
        # 5*s_rate seems like a good number for the activities in ego4d
        
    def __getitem__(self, idx):
        # assuming idx is an int
        anchor_path = self.fnames[idx]
        pos_path = self.fnames[idx-self.pdif] # has an incorrect edge case at idx being the first in the folder,
        # which happens very rarely given that there are ~500 images in each folder
#         neg_path = random.choice(self.fnames)
        neg_path = self.fnames[idx-self.ndif]
        
        a = torchvision.io.read_image(anchor_path)
        p = torchvision.io.read_image(pos_path)
        n = torchvision.io.read_image(neg_path)
        
        if self.transform is not None:
            a,p,n = self.transform(a), self.transform(p), self.transform(n)
            
        return torch.stack((a,p,n))
    
    def __len__(self):
        return len(self.fnames)

train_data = fnames
train_data_te = TimeContrastiveTorchDataset(train_data, pdif=1, ndif=5*s_rate,
                                transform=transforms_ego4d)
train_dataloader = torch.utils.data.DataLoader(
    train_data_te, batch_size=128, shuffle=True, pin_memory=True, num_workers=2)


# In[99]:


#%% Instantiation
xmodel = torchvision.models.resnet18(pretrained=False)
hdim = 1000
# seq_len = xclips.shape[1]
xmodel.fc = nn.Linear(512, hdim)
# nn.init.kaiming_normal_(xmodel.fc.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

xmodel.float()
xmodel.to(device)
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
# triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
# cosine_sim = nn.CosineSimilarity(dim=0)
# cosine_sim_f = lambda inp1,inp2: 1-cosine_sim(inp1,inp2)
cosine_sim_f = lambda x, y: 1.0 - F.cosine_similarity(x, y)
triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=cosine_sim_f, margin=0.2, reduction='mean')


# In[100]:


def to_static(img_seq):
    n_samples, seq_len, *other_dims = img_seq.shape
    return img_seq.reshape((n_samples*seq_len, *other_dims))

def to_seq(img_flt, seq_len):
    n_samplesXseq_len, *other_dims = img_flt.shape
    n_samples = int(n_samplesXseq_len/seq_len)
    return img_flt.view((n_samples, seq_len, *other_dims))

def to_batchseq(img_flt, seq_len):
    n_samplesXseq_len, *other_dims = img_flt.shape
    n_samples = int(n_samplesXseq_len/seq_len)
    return img_flt.view((seq_len, n_samples, *other_dims))


# In[101]:


seq_len = 3
loss_list = []
iteration_list = []

for i, triplet in enumerate(tqdm(train_dataloader)):
    xmodel.train()
    optimizer.zero_grad()
    
    triplet = triplet.to(device)
    out = xmodel(to_static(triplet))
    ytt = to_seq(out, seq_len)
    a = ytt[:,0,...]
    p = ytt[:,1,...]
    n = ytt[:,2,...]
    
    cpc_loss = triplet_loss(a, p, n)
    loss = cpc_loss
    loss.backward()
    optimizer.step()
    
    loss_list.append(loss.data.item())
    iteration_list.append(i)
        
    if i%10==0:
        print('Iteration: {}  Train Loss: {}  '.format(i, loss.data.item()))
    


# In[50]:


# Additional information
EPOCH = 1
PATH = "/N/slate/sheybani/tmp_dir/trainedmodels/ego4d_cpc0/model_may3_1900.pt"
LOSS = loss_list

torch.save({
            'epoch': EPOCH,
            'model_state_dict': xmodel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

# https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html





