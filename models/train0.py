# defines the actual training loop for the model. 
# This code interacts with the optimizer and handles logging during training.

# Lots of relevant content in nma_models.py
#%% Imports
import copy
from functools import partialmethod
import warnings

import numpy as np
import torch
from torch import nn
from tqdm.notebook import tqdm as tqdm
from matplotlib import pyplot as plt

#%% Network Classes
torch.set_default_dtype(torch.float32)

class Encoder0(nn.Module):
    def __init__(self, input_dim=(3, 64, 64), training=False):
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
        # self._untrained = True
        # self.training = training

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

    def forward(self, X):
        # if self.untrained and self.training:
        #     self._untrained = False
        
        assert len(X.shape)==4
        # assumes the input is of shape (N, C, W, H)
        
        feats_extr = self.feature_extractor(X)
        return feats_extr

class VentralPathway0(nn.Module):
    def __init__(self, input_dim, output_dim, training=False):

        super().__init__()
        # self.training = training
        # self._untrained = True

        # check input dimensions provided
        self.input_dim = tuple(input_dim)
        self.output_dim = output_dim
        # if len(self.input_dim) == 3:
        #     self.input_dim = (1, *input_dim)            
        if len(self.input_dim) != 3:
            raise ValueError("input_dim should have length 3 "
                f"3 (ch x wid x hei), but has length ({len(self.input_dim)}).")
        self.input_ch = self.input_dim[0]

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_ch, out_channels=16, kernel_size=3, 
                stride=1
                ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(16, affine=False),
        )
        
        # calculate size of the convolutional feature extractor output
        self.feat_extr_output_size = \
            self._get_feat_extr_output_size(self.input_dim)
        
        
        self.linear_projections = nn.Sequential(
            nn.Linear(self.feat_extr_output_size, 120),
            nn.ReLU(),
            nn.BatchNorm1d(120, affine=False),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.BatchNorm1d(84, affine=False),
        )

        self.linear_projections_output = nn.Sequential(
            nn.Linear(84, self.output_dim)
        )
    
    def _get_feat_extr_output_size(self, input_dim):
        dummy_tensor = torch.ones(1, *input_dim)
        reset_training = self.training
        
        print('dummy_tensor shape: ', dummy_tensor.shape) #@@@
        self.eval()
        with torch.no_grad():   
            output_dim = self.feature_extractor(dummy_tensor).shape
        if reset_training:
            self.train()
        return np.product(output_dim)

    def forward(self, X):
        # if self.untrained and self.training:
        #     self._untrained = False
        
        assert len(X.shape)==4
        # assumes the input is of shape (N, C, W, H)
        
        feats_extr = self.feature_extractor(X)
        feats_flat = torch.flatten(feats_extr, 1)
        feats_proj = self.linear_projections(feats_flat)
        ventral_out = self.linear_projections_output(feats_proj)
        return ventral_out
    
class DorsalPathway0(nn.Module):
    def __init__(self, n_in, n_out, n_layers=2, n_hidden=30):
      super().__init__()
      self.n_in = n_in
      self.n_out = n_out
      self.n_hidden = n_hidden
      self.n_layers = n_layers
      
      self.n_rnnin = 30
      
      self.fc1 = nn.Sequential(
            nn.Linear(n_in, self.n_rnnin),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_rnnin, affine=False),
        )
      self.rnn = nn.LSTM(self.n_rnnin, n_hidden, n_layers, batch_first=True)
      self.res_w_out = nn.Linear(n_hidden, n_out)
      
    def init_state(self, batch_size):
        h0 = torch.randn(self.n_layers, batch_size, self.n_hidden)
        c0 = torch.randn(self.n_layers, batch_size, self.n_hidden)
        return h0,c0
        
    def forward(self, xin):
      assert len(xin.shape)==3
      # assumes xin is of shape (N, L, n_in)
      # print('rnn xin shape: ', xin.shape)
      batch_size, seq_len, _ = xin.shape
      
      rnnin = self.fc1(xin.view(batch_size*seq_len, -1))
      rnnin = rnnin.view(batch_size, seq_len, -1)
      # xin = torch.unsqueeze(xin, dim=-1)
      h0, c0 = self.init_state(batch_size)
      rnn_out, (hn, cn) = self.rnn(rnnin, (h0, c0))
      
      
      dorsal_out = self.res_w_out(
          rnn_out.reshape(batch_size*seq_len, -1)) #@@@slower because of reshape instead of view
      dorsal_out = dorsal_out.view(batch_size, seq_len, -1)
      # xin: (N, n_in, 1)
      return dorsal_out, rnn_out
  
class DualModel0(nn.Module):
    def __init__(self, input_dim=(10,3,64,64), output_dimv=10, output_dimd=7, encoder_h=84, 
                 rnn_h=30, debug=False, training=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dimv = output_dimv
        self.output_dimd = output_dimd
        self.encoder_h = encoder_h
        self.rnn_h = rnn_h
        self.debug = debug
        # self.training = training
        
        self.shared_encoder = Encoder0(input_dim=input_dim[1:])
        
        # # calculate size of the convolutional feature extractor output
        self.shared_enc_size, self.shared_enc_shape = \
            self._get_feat_extr_output_size(self.input_dim[1:])

        # # linear component of the feature extractor
        # self.linear_projections = 
        
        # self.fc_enc = nn.Linear(encoder_h, encoder_h)
        print('self.shared_enc_shape: ', self.shared_enc_shape)
        self.ventral = VentralPathway0(self.shared_enc_shape, output_dimv)
        
        self.dorsal = DorsalPathway0(self.shared_enc_size, self.output_dimd, 
                                     n_layers=2, n_hidden=30)
        
        
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
        
    def _get_feat_extr_output_size(self, input_dim):
        dummy_tensor = torch.ones(1, *input_dim)
        reset_training = self.training
        self.eval()
        with torch.no_grad():   
            output_dim = self.shared_encoder(dummy_tensor).shape
        if reset_training:
            self.train()
        return np.product(output_dim), output_dim[1:]
            
    @property
    def untrained(self):
        return self._untrained
        
    def forward(self, batch_seq):
        
        orig_shape = batch_seq.shape
        # For the static part, collapse the sequence dim into the batch dim.
        batch_img = batch_seq.reshape(-1, *orig_shape[2:])
        
        enc_out = self.shared_encoder(batch_img) #out dim=(N,encoder_h)
        
        ventral_out = self.ventral(enc_out)
        ventral_out = ventral_out.reshape(*orig_shape[:2], -1)
        
        dorsal_in = enc_out.reshape(*orig_shape[:2], -1)
        
        dorsal_out,_ = self.dorsal(dorsal_in)
        return ventral_out, dorsal_out
 

# xmodel = FullModel0(input_dim=(3,64,64), output_dim=7, encoder_h=84, 
#                  reservoir_h=20, debug=True)
  
#%% Instantiation
xmodel = DualModel0(input_dim=(10,3,64,64), 
                    output_dimv=10, output_dimd=3, encoder_h=84, 
             rnn_h=30, debug=False)
xmodel.float()
# xmodel.train()

optimizer = torch.optim.Adam([
    {'params': xmodel.shared_encoder.parameters()},
    {'params': xmodel.ventral.parameters(), 'lr': 1e-3},
    {'params': xmodel.dorsal.parameters(), 'lr': 1e-4}
    ], lr=1e-1, weight_decay=0, amsgrad=False)
    
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