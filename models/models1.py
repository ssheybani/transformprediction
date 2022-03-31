# Definition of nn models
# When testing them, the module load_dataset is needed.
# Includes SUmmaryWriter boilerplate for sending the signals to tensorboard
import copy
from functools import partialmethod
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter

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
    
class DorsalPathway0(nn.Module):
    def __init__(self, n_in, n_out, n_layers=2, n_hidden=30):
      super().__init__()
      self.n_in = n_in
      self.n_out = n_out
      self.n_hidden = n_hidden
      self.n_layers = n_layers
      
      self.rnn = nn.LSTM(n_in, n_hidden, n_layers, batch_first=True)
      self.res_w_out = nn.Linear(n_hidden, n_out)
      
    def init_state(self, batch_size):
        h0 = torch.randn(self.n_layers, batch_size, self.n_hidden)
        c0 = torch.randn(self.n_layers, batch_size, self.n_hidden)
        return h0,c0
        
    def forward(self, xin):
      assert len(xin.shape)==3
      # assumes xin is of shape (N, L, n_in)
      batch_size, seq_len, _ = xin.shape
      # xin = torch.unsqueeze(xin, dim=-1)
      h0, c0 = self.init_state(batch_size)
      rnn_out, (hn, cn) = self.rnn(xin, (h0, c0))
      
      # dorsal_out = self.res_w_out(rnn_out)
      # xin: (N, n_in, 1)
      return rnn_out   
     
      
class DualModel0(nn.Module):
    def __init__(self, input_dim=(10,3,64,64), output_dimv=10, output_dimd=7, encoder_h=84, 
                 rnn_h=30, debug=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dimv = output_dimv
        self.output_dimd = output_dimd
        self.encoder_h = encoder_h
        self.rnn_h = rnn_h
        self.debug = debug
        
        self.encoder = Encoder0(feat_size=encoder_h, input_dim=input_dim[1:])
        self.fc_enc = nn.Linear(encoder_h, encoder_h)
        
        self.ventral = nn.Linear(encoder_h, output_dimv)
        
        self.dorsal = DorsalPathway0(encoder_h, 30, n_layers=2, n_hidden=30)
        
        
        # self.res_w_out = nn.Parameter(torch.rand(encoder_h, reservoir_h))
        # self.fc_out = nn.Linear(encoder_h, output_dim)
    
        # self.linear_projections = nn.Sequential(
        #         nn.Linear(encoder_h, 120),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(120, affine=False),
        #         nn.Linear(120, 84),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(84, affine=False),
        #     )
    
        # self.linear_projections_output = nn.Sequential(
        #         nn.Linear(84, output_dim),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(output_dim, affine=False)
        #     )
        
    def init_state(self):
      self.reservoir.init_state()
        
    def forward(self, batch_seq):
        
        orig_shape = batch_seq.shape
        # For the static part, collapse the sequence dim into the batch dim.
        batch_img = batch_seq.reshape(-1, *orig_shape[2:])
        
        enc_out = self.encoder(batch_img) #out dim=(N,encoder_h)
        
        ventral_out = self.ventral(enc_out)
        ventral_out = ventral_out.reshape(*orig_shape[:2], -1)
        
        dorsal_in = enc_out.reshape(*orig_shape[:2], -1)
        dorsal_out = self.dorsal(dorsal_in)
        return ventral_out, dorsal_out
        # # enc_out = 0.1*torch.rand(batch_img.shape[0], self.encoder_h) #debug @@@@
        # # enc_out = 0.1*torch.ones(batch_img.shape[0], self.encoder_h) #debug @@@@
        # res_state = self.reservoir(enc_out)#dim=(N,reservoir_h, encoder_h)
        # print('res_state shape: ', res_state.shape)
        # res_out = F.relu(
        #     torch.mul(res_state, self.res_w_out).sum(dim=-1)
        #     )#dim=(N,encoder_h)
        # res_flat_expansion = self.linear_projections(res_out)
        # out = self.linear_projections_output(res_flat_expansion)
        
        # if self.debug:
        #     return out, enc_out, res_out
        # return out
 

xmodel0 = DualModel0(input_dim=(10,3,64,64), output_dimv=10, output_dimd=7, encoder_h=84, 
             rnn_h=30, debug=False)

yv, yd = xmodel0(xclips)

yv_np = yv.detach().numpy()
yd_np = yd.detach().numpy()

plt.plot(yv_np[0, :,0])
plt.plot(yd_np[0, :,0])


import matplotlib.pyplot as plt

#--------------------------------
# test1: reservoir with sin input
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
    train_data_te, batch_size=64, shuffle=True)

for xclips, (xdlabels, xclabels) in train_dataloader:
    break
# xencoder = Encoder0(feat_size=84, input_dim=(3,64,64))
# xreservoir = Reservoir0(84, n_hidden=20)

xmodel = FullModel0(input_dim=(3,64,64), output_dim=7, encoder_h=84, 
                 reservoir_h=20, debug=True)
xmodel.eval()

len_clip = 10 #4
batch_size = xclips.shape[0]

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
