# defines the actual training loop for the model. 
# This code interacts with the optimizer and handles logging during training.

# Lots of relevant content in nma_models.py
import copy
from functools import partialmethod
import warnings

import numpy as np
import torch
from torch import nn
from tqdm.notebook import tqdm as tqdm
from matplotlib import pyplot as plt

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