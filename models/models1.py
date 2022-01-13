import copy
from functools import partialmethod
import warnings

import numpy as np
import torch
from torch import nn

class Encoder0(nn.Module):
    def __init__(self, feat_size=84, input_dim=(1, 64, 64)):
        """
        Adapted from Neuromatch Academy encoder for VAE and SimCLR
        from https://github.com/colleenjg/neuromatch_ssl_tutorial/blob/main/modules/models.py


        Initializes the core encoder network.

        Optional args:
        - feat_size (int): size of the final features layer (default: 84)
        - input_dim (tuple): input image dimensions (channels, width, height) 
            (default: (1, 64, 64))
        """

        super().__init__()

        # check input dimensions provided
        self.input_dim = tuple(input_dim)
        if len(self.input_dim) == 2:
            self.input_dim = (1, *input_dim)            
        elif len(self.input_dim) != 3:
            raise ValueError("input_dim should have length 2 (wid x hei) or "
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

    def forward(self, X):
        if self.untrained and self.training:
            self._untrained = False
        feats_extr = self.feature_extractor(X)
        feats_flat = torch.flatten(feats_extr, 1)
        feats_proj = self.linear_projections(feats_flat)
        feats = self.linear_projections_output(feats_proj)
        if self.vae:
            logvars = self. linear_projections_logvar(feats_proj)
            return feats, logvars
        return feats

    def get_features(self, X):
        with torch.no_grad():
            feats_extr = self.feature_extractor(X)
            feats_flat = torch.flatten(feats_extr, 1)
            feats_proj = self.linear_projections(feats_flat)
            feats = self.linear_projections_output(feats_proj)
        return feats




class Encoder1_noBN(EncoderCore):
    # Removed batchnorm from the feature extractor of EncoderCore.
        def __init__(self, feat_size=84, input_dim=(1, 64, 64), vae=False):
        """
        Optional args:
        - feat_size (int): size of the final features layer (default: 84). 
        If feat_size>84, it might not work.
        - input_dim (tuple): input image dimensions (channels, width, height) 
            (default: (1, 64, 64))
        - vae (bool): if True, a VAE encoder is initialized with a second 
            feature head for the log variances. (default: False)
        """

        super().__init__()

        self._vae = vae
        self._untrained = True

        # check input dimensions provided
        self.input_dim = tuple(input_dim)
        if len(self.input_dim) == 2:
            self.input_dim = (1, *input_dim)            
        elif len(self.input_dim) != 3:
            raise ValueError("input_dim should have length 2 (wid x hei) or "
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
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

        # calculate size of the convolutional feature extractor output
        self.feat_extr_output_size = \
            self._get_feat_extr_output_size(self.input_dim)
        self.feat_size = feat_size

        # linear component of the feature extractor
        self.linear_projections = nn.Sequential(
            nn.Linear(self.feat_extr_output_size, 120),
            nn.ReLU()
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.linear_projections_output = nn.Sequential(
            nn.Linear(84, self.feat_size),
            nn.ReLU()
        )

        if self.vae:
            self.linear_projections_logvar = nn.Sequential(
                nn.Linear(84,self.feat_size),
                nn.ReLU()
            )