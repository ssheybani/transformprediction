import copy, os, itertools, math
from functools import partialmethod
import warnings

# import numpy as np
# from matplotlib import pyplot as plt
# import pandas as pd
# import seaborn as sns

import torch, torchvision
from torch import nn
from tqdm import tqdm 
import traceback

#%% Network Classes
torch.set_default_dtype(torch.float32)



#%% Preproc functions
from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.transforms import _functional_video as torchvid

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
ss_rate = 15
crop_size = 256

subsample = lambda img_seq, ss_rate: img_seq[0:-1:ss_rate, ...]
to_float = lambda x: (x/255.0).type(torch.float32)
to_uint = lambda x: (255*x).type(torch.uint8)
# assumes the frame dimension is the first.

def short_side_scale(
    x: torch.Tensor,
    size: int,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.
    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
        backend (str): backend used to perform interpolation. Options includes
            `pytorch` as default, and `opencv`. Note that opencv and pytorch behave
            differently on linear interpolation on some versions.
            https://discuss.pytorch.org/t/pytorch-linear-interpolation-is-different-from-pil-opencv/71181
    Returns:
        An x-like Tensor with scaled spatial dims.
    """  # noqa
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    t, c, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))

    return torch.nn.functional.interpolate(
        x, size=(new_h, new_w), mode=interpolation, align_corners=False,
        antialias=True
    )
        
transforms_ego4d = Compose(
    [
     Lambda(to_float),
     Lambda(lambda x: torch.permute(x, (0,3,1,2))),
     Lambda(lambda x: short_side_scale(x, crop_size)),
     Lambda(lambda x: torch.permute(x, (1,0,2,3))),
     CenterCropVideo(crop_size=(crop_size, crop_size)),
     Lambda(lambda x: torch.permute(x, (1,2,3,0))),
     Lambda(to_uint)
     ])
#T,H,W,C
#%%% Ego4D Dataset: collect metadata



vid_root = r"/N/slate/sheybani/ego4ddata/v1/clips/"
out_root = r"/N/slate/sheybani/ego4ddata/preprocessed_clips/"
sample_paths = os.listdir(vid_root)
    
n_samples = 3
for i in tqdm(range(n_samples)):
    vp = sample_paths[i]
    try:
        vid_in = torchvision.io.read_video(vid_root+vp)[0]
    except Exception as exception:
        traceback.print_exc()
        continue
    if len(vid_in)==0:
        print('Empty input tensor at '+vp)
        continue
    vid_ss = subsample(vid_in, ss_rate)
    vid_out = transforms_ego4d(vid_ss)
    torchvision.io.write_video(out_root+vp, vid_out, 1, video_codec='libx264')


    
