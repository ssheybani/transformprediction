# Definition of nn models
# When testing them, the module load_dataset is needed.
# new in models2: ConvRNN

import copy
from functools import partialmethod
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import pandas as pd




# xmodel = Reservoir0(1, n_hidden=20)
# xh_all = []

# tt = np.arange(0, 3.14, 0.1)
# for i in tt:
#     # print('model state = ',xmodel.state)
#     xin = torch.sin(torch.tensor(i)).reshape(1,1)
#     # print('xin = ', xin)
#     xh = xmodel(xin)
#     xh_all.append(xh.detach().numpy())

# xh_all = np.asarray(xh_all).squeeze()

# x0 = np.sin(tt)
# plt.plot(tt, x0, label='input: sin(t)')
# for h_idx in range(5):
#     plt.plot(tt, xh_all[:,h_idx])

# plt.legend()

#--------------------------------
# Test 2: full model without training
# Run the file load_dataset to define train_data_te

# def change_range(data, old_range, new_range):
#     assert len(old_range)==2 and len(new_range)==2
#     old_loc = np.mean(old_range)
#     old_scale = old_range[1]-old_loc

#     new_loc = np.mean(new_range)
#     new_scale = new_range[1]-new_loc

#     ratio = new_scale/old_scale
#     return (data-old_loc)*ratio

# def clip_torchfloat2npimage(clip_te, from_range=[0,1], to_range=[0,1], to_dtype=np.float32):
#     # clip_te: range=[-1.,1.]. shape=(N,C,W,H)
#     clip2 = clip_te.detach().numpy().transpose(0,2,3,1)
#     # img3 = img2*2-1
#     if from_range!=to_range:
#         clip2 = change_range(clip2, from_range, to_range)#(clip2/2)+0.5
#     return clip2.astype(to_dtype)

# def clip_npimage2torchfloat(clip_np, from_range=[0,1], to_range=[0,1]):
#     # clip_np: shape=(N,W,H,C)
#     if from_range!=to_range:
#         clip_np = change_range(clip_np.astype(np.float32), 
#             from_range, to_range)
#     clip2 = torch.permute(
#         torch.as_tensor(clip_np, dtype=torch.float32),
#             (0, 3, 1, 2))
#     return clip2

# def img_torchfloat2ubyte(img_te):
#     img2 = img_te.detach().numpy().transpose(0,2,3,1)
#     # img3 = img2*2-1
#     img3 = 255*img2
#     return img3.astype(np.uint8)#skutil.img_as_ubyte(img3)
# default `log_dir` is "runs" - we'll be more specific here


train_dataloader = torch.utils.data.DataLoader(
    train_data_te, batch_size=1, shuffle=True)

for xclips, (xdlabels, xclabels) in train_dataloader:
    break
# xencoder = Encoder0(feat_size=84, input_dim=(3,64,64))
# xreservoir = Reservoir0(84, n_hidden=20)

xmodel = FullModel0(input_dim=(3,64,64), output_dim=7, encoder_h=84, 
                 reservoir_h=20, debug=True)
xmodel.eval()

len_clip = 4
batch_size = len(xclips)

xencs = []
xress = []
xouts = []
for i in range(len_clip):
    batch_img = xclips[:,i,...]
    # xmodel.init_state()
    xouttmp, xenctmp, xrestmp = xmodel(batch_img)
    xouts.append(xouttmp.detach().numpy())
    xencs.append(xenctmp.detach().numpy())
    xress.append(xrestmp.detach().numpy())
    
xclips_np = xclips.detach().numpy()
    
xouts = np.asarray(xouts).transpose(1,0,2)
xencs = np.asarray(xencs).transpose(1,0,2)
xress = np.asarray(xress).transpose(1,0,2)

# from skimage import util as skutil
    
sample_nums = np.random.randint(low=0, high=batch_size, size=3)

fig, ax = plt.subplots(len_clip+3, len(sample_nums), figsize=(3* len(sample_nums),12))

for i, sample_num in enumerate(sample_nums):
    ax[0,i].set_xlabel('Time')
    
    ax[0,i].plot(xencs[sample_num,...])
    ax[0,i].set_ylabel('Conv Encoder output')
    
    ax[1,i].plot(xress[sample_num,...])
    ax[1,i].set_ylabel('RNN output')
    
    ax[2,i].plot(xouts[sample_num,...])
    ax[2,i].set_ylabel('Final output')

    for j in range(3, 3+len_clip):
        xclip = clip_torchfloat2npimage(
            xclips[sample_num,...], from_range=[0,1], to_range=[0,1])
        
        # xclip = xclip-np.mean(xclip, axis=0)
        ax[j,i].imshow(xclip[j-3,...])


# Verify the statistical properties of the signals
fig, ax = plt.subplots(4, 1, figsize=(3,7))
_ = ax[0].hist(xclips_np.flatten(), label='Input images')
_ = ax[1].hist(xencs.flatten(), label='Conv encoder output')
_ = ax[2].hist(xress.flatten(), label='RNN output')
_ = ax[3].hist(xouts.flatten(), label='Final output')

for i, axij in enumerate(ax):
    axij.legend()
    
    
# Visualizing on Tensorboard
# Run in a terminal:
# cd C:\Users\Saber\Documents\GitHub\transformprediction\data
# tensorboard --logdir=runs

writer = SummaryWriter('runs/models_test2')
writer.add_embedding(xouttmp,
                    metadata=xdlabels[:,-1,:],
                    label_img=batch_img)

for i, var in enumerate([xencs, xress, xouts]):
    writer.add_histogram('layer activation distributions', var.flatten(), i)
writer.close()



#--------------------------------
# test1: torch rnn with image sequence
class Reservoir1(nn.Module):

    # A fully linear unit
    # Does not accept sequences, but does maintain a state
    
  def __init__(self, n_in, n_hidden=20):
    super().__init__()
    
    self.n_in = n_in
    self.n_hidden = n_hidden
    self.state = torch.zeros((n_in, n_hidden))
    self.w_in = nn.Parameter(torch.rand(n_in, n_hidden))
    # self.w_out = nn.Parameter(torch.rand(n_in, n_hidden))
    
    # a fixed auxilliary tensor, used in every forward() call.
    self.ones = torch.ones_like(self.state)

  def init_state(self):
      self.state = torch.zeros((self.n_in, self.n_hidden))
      
  def forward(self, xin):
    assert len(xin.shape)==2
    # assumes xin is of shape (N, n_in)
    # ultimately, it must be the same shape as xclips: (N, L, C, W, H)
    # For every pixel in (C,W,H) we have a nh dimensional time serie. That is 250k dimensions.
    xin = torch.unsqueeze(xin, dim=-1)
    # xin: (N, n_in, 1)
    self.state = torch.mul(xin, self.w_in)+ \
                 torch.mul(self.state, (self.ones-self.w_in))
    # self.out = torch.mul(self.state, self.w_out).sum(dim=-1)
    return self.state


class RNNLattice(nn.Module):
    def __init__(self, n_in, n_h):
        self.n_in = n_in
        self.n_h = n_h
        
        self.lattice = [nn.RNN(n_in, n_in, nonlinearity='relu', num_layers=1, 
                              bias=False, batch_first=True)
                        for i in range(n_h)]
        for rnn in self.lattice:
            _ = nn.init
        nn.init.zeros_(self.weight_hh_l)

    def forward()

n_in = 64*64*3
n_h = 2
lattice = [nn.RNN(n_in, n_in, nonlinearity='tanh', num_layers=1, 
                              bias=False, batch_first=True)
                        for i in range(n_h)]
for rnn in lattice:
    _ = nn.init.zeros_(rnn.weight_hh_l0)
    rnn.eval()

h0 = torch.zeros(1,1,n_in)
inp = torch.flatten(xclips, start_dim=2, end_dim=-1) 
inp = torch.rand(1,10,n_in)
outp, _ = lattice[1](inp, h0) 

chosen_h = np.random.randint(0, n_in, size=15)
xts = outp[0,:,chosen_h].detach().numpy()
_ = plt.plot(xts)

xdf = pd.DataFrame(xts)
xcorr = xdf.corr()
_ = plt.hist(xcorr.to_numpy().flatten(), bins=40)
