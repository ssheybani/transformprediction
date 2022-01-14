import copy
from functools import partialmethod
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F



class Encoder0(nn.Module):
    def __init__(self, feat_size=84, input_dim=(3, 64, 64)):
        """
        Adapted from Neuromatch Academy encoder for VAE and SimCLR
        from https://github.com/colleenjg/neuromatch_ssl_tutorial/blob/main/modules/models.py


        Initializes the core encoder network.

        Optional args:
        - feat_size (int): size of the final features layer (default: 84)
        - input_dim (tuple): input image dimensions (channels, width, height) 
            (default: (3, 64, 64))
        """

        super().__init__()
        self._untrained = True

        # check input dimensions provided
        self.input_dim = tuple(input_dim)
        # if len(self.input_dim) == 3:
        #     self.input_dim = (1, *input_dim)            
        if len(self.input_dim) != 3:
            raise ValueError("input_dim should have length 3 "
                f"3 (ch x wid x hei), but has length ({len(self.input_dim)}).")
        self.input_ch = self.input_dim[0]

        # convolutional component of the feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_ch, out_channels=6, kernel_size=5, 
                stride=1
                ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(6, affine=False),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(16, affine=False)
        )

        # calculate size of the convolutional feature extractor output
        self.feat_extr_output_size = \
            self._get_feat_extr_output_size(self.input_dim)
        self.feat_size = feat_size

        # linear component of the feature extractor
        self.linear_projections = nn.Sequential(
            nn.Linear(self.feat_extr_output_size, 120),
            nn.ReLU(),
            nn.BatchNorm1d(120, affine=False),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.BatchNorm1d(84, affine=False),
        )

        self.linear_projections_output = nn.Sequential(
            nn.Linear(84, self.feat_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.feat_size, affine=False)
        )

    def _get_feat_extr_output_size(self, input_dim):
        dummy_tensor = torch.ones(1, *input_dim)
        reset_training = self.training
        self.eval()
        with torch.no_grad():   
            output_dim = self.feature_extractor(dummy_tensor).shape
        if reset_training:
            self.train()
        return np.product(output_dim)
        
    @property
    def untrained(self):
        return self._untrained

    def forward(self, X):
        if self.untrained and self.training:
            self._untrained = False
        
        assert len(X.shape)==4
        # assumes the input is of shape (N, C, W, H)
        
        feats_extr = self.feature_extractor(X)
        feats_flat = torch.flatten(feats_extr, 1)
        feats_proj = self.linear_projections(feats_flat)
        feats = self.linear_projections_output(feats_proj)
        return feats

    def get_features(self, X):
        with torch.no_grad():
            feats_extr = self.feature_extractor(X)
            feats_flat = torch.flatten(feats_extr, 1)
            feats_proj = self.linear_projections(feats_flat)
            feats = self.linear_projections_output(feats_proj)
        return feats


class Reservoir0(nn.Module):
  
  def __init__(self, n_in, n_hidden=20):
    super().__init__()
    
    self.n_in = n_in
    self.n_hidden = n_hidden
    self.state = torch.zeros((n_in, n_hidden))
    self.w_in = nn.Parameter(torch.rand(n_in, n_hidden))
    # self.w_out = nn.Parameter(torch.rand(n_in, n_hidden))

    self.ones = torch.ones_like(self.state)
    

  def init_state(self):
      self.state = torch.zeros((self.n_in, self.n_hidden))
      
  def forward(self, xin):
    assert len(xin.shape)==2
    # assumes xin is of shape (N, n_in)
    xin = torch.unsqueeze(xin, dim=-1)
    # xin: (N, n_in, 1)
    self.state = torch.mul(xin, self.w_in)+ \
                 torch.mul(self.state, (self.ones-self.w_in))
    # self.out = torch.mul(self.state, self.w_out).sum(dim=-1)
    return self.state


class FullModel0(nn.Module):
    def __init__(self, input_dim=(3,64,64), output_dim=7, encoder_h=84, 
                 reservoir_h=20, debug=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_h = encoder_h
        self.reservoir_h = reservoir_h
        self.debug = debug
        
        self.encoder = Encoder0(feat_size=encoder_h, input_dim=input_dim)
        self.fc_enc = nn.Linear(encoder_h, encoder_h)
        self.reservoir = Reservoir0(encoder_h, n_hidden=reservoir_h)
        self.res_w_out = nn.Parameter(torch.rand(encoder_h, reservoir_h))
        # self.fc_out = nn.Linear(encoder_h, output_dim)
    
        self.linear_projections = nn.Sequential(
                nn.Linear(encoder_h, 120),
                nn.ReLU(),
                nn.BatchNorm1d(120, affine=False),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.BatchNorm1d(84, affine=False),
            )
    
        self.linear_projections_output = nn.Sequential(
                nn.Linear(84, output_dim),
                nn.ReLU(),
                nn.BatchNorm1d(output_dim, affine=False)
            )
        
    def init_state(self):
      self.reservoir.init_state()
        
    def forward(self, batch_img):
        enc_out = self.encoder(batch_img) #out dim=(N,encoder_h)
        # enc_out = 0.1*torch.rand(batch_img.shape[0], self.encoder_h) #debug @@@@
        # enc_out = 0.1*torch.ones(batch_img.shape[0], self.encoder_h) #debug @@@@
        res_state = self.reservoir(enc_out)#dim=(N,reservoir_h, encoder_h)
        print('res_state shape: ', res_state.shape)
        res_out = F.relu(
            torch.mul(res_state, self.res_w_out).sum(dim=-1)
            )#dim=(N,encoder_h)
        res_flat_expansion = self.linear_projections(res_out)
        out = self.linear_projections_output(res_flat_expansion)
        
        if self.debug:
            return out, enc_out, res_out
        return out
    
    

import matplotlib.pyplot as plt

# test1: reservoir with sin input
xmodel = Reservoir0(1, n_hidden=20)
xh_all = []

tt = np.arange(0, 3.14, 0.1)
for i in tt:
    # print('model state = ',xmodel.state)
    xin = torch.sin(torch.tensor(i)).reshape(1,1)
    # print('xin = ', xin)
    xh = xmodel(xin)
    xh_all.append(xh.detach().numpy())

xh_all = np.asarray(xh_all).squeeze()

x0 = np.sin(tt)
plt.plot(tt, x0, label='input: sin(t)')
for h_idx in range(5):
    plt.plot(tt, xh_all[:,h_idx])

plt.legend()

# Test 2: full model without training
# Run the file load_dataset to define train_data_te

def clip_torchfloat2float(clip_te):
    # clip_te: range=[-1.,1.]. shape=(N,C,W,H)
    clip2 = clip_te.detach().numpy().transpose(0,2,3,1)
    # img3 = img2*2-1
    clip3 = (clip2/2)+0.5
    return clip3#.astype(np.uint8)#skutil.img_as_ubyte(img3)

def clip_float2torchfloat(clip_np):
    # clip_np: range=[0.,1.]. shape=(N,W,H,C)
    clip2 = clip_np*2 -1.
    clip3 = torch.permute(
        torch.as_tensor(clip2, dtype=torch.float32),
            (0, 3, 1, 2))
    return clip3

# def img_torchfloat2ubyte(img_te):
#     img2 = img_te.detach().numpy().transpose(0,2,3,1)
#     # img3 = img2*2-1
#     img3 = 255*img2
#     return img3.astype(np.uint8)#skutil.img_as_ubyte(img3)

train_dataloader = torch.utils.data.DataLoader(
    train_data_te, batch_size=64, shuffle=True)

for xclips, (xdlabels, xclabels) in train_dataloader:
    break
# xencoder = Encoder0(feat_size=84, input_dim=(3,64,64))
# xreservoir = Reservoir0(84, n_hidden=20)

xmodel = FullModel0(input_dim=(3,64,64), output_dim=7, encoder_h=84, 
                 reservoir_h=20, debug=True)
xmodel.eval()

len_clip = 4

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
xouts = np.asarray(xouts).transpose(1,0,2)
xencs = np.asarray(xencs).transpose(1,0,2)
xress = np.asarray(xress).transpose(1,0,2)

from skimage import util as skutil
    
sample_nums = np.random.randint(low=0, high=xouts.shape[0], size=3)

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
        xclip = clip_torchfloat2float(
            xclips[sample_num,...])
        
        # xclip = xclip-np.mean(xclip, axis=0)
        ax[j,i].imshow(xclip[j-3,...])


# Verify the statistical properties of the signals
fig, ax = plt.subplots(3, 1, figsize=(3,5))
_ = ax[0].hist(xencs.flatten(), label='Conv encoder output')
_ = ax[1].hist(xress.flatten(), label='RNN output')
_ = ax[2].hist(xouts.flatten(), label='Final output')

for i, axij in enumerate(ax):
    axij.legend()