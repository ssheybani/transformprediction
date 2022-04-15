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



#%%
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ShortSideScale
)

from pytorchvideo.transforms.functional import uniform_temporal_subsample

#%%% ChradesEgo Dataset: collect metadata

vid_root = r'C:/Users/Saber/Documents/tmp_dir/CharadesEgo_v1_480/'
sample_paths = os.listdir(vid_root)


def train_test_split_idx(n_samples, fraction_train=0.8):
    sample_idx = np.random.permutation(n_samples)
    n_train = int(n_samples*fraction_train)
    train_idx, test_idx = sample_idx[:n_train], sample_idx[n_train:]
    return train_idx, test_idx
n_samples = len(sample_paths)
train_idx, test_idx = train_test_split_idx(n_samples, fraction_train=0.8)

train_data = [sample_paths[i] for i in train_idx]

class VidTorchDataset(torch.utils.data.Dataset):
    def __init__(self, root, fnames):
        self.root = root
        self.fnames = fnames
    def __getitem__(self, idx):
        # assuming idx is an int
        return torchvision.io.read_video(self.root+self.fnames[idx])
    def __len__(self):
        return len(self.fnames)

train_data_te = VidTorchDataset(vid_root, train_data)
train_dataloader = torch.utils.data.DataLoader(
    train_data_te, batch_size=1, shuffle=False)


for xclips in train_dataloader:
    break


tmp_fps = []
tmp_shape = []
i=0
for xclips in train_dataloader:
    tmp_fps.append(xclips[2]['video_fps'].item())
    tmp_shape.append(np.asarray(xclips[0].shape))
    print('fps: ', xclips[2]['video_fps'].item(), ' shape:', xclips[0].shape)
    if i >100:
        break
    i+=1

tmp_fps = np.asarray(tmp_fps).reshape(-1,1)
tmp_shape = np.asarray(tmp_shape)
tmp_shape2 = tmp_shape[:,1:]
tmp_meta = np.concatenate((tmp_fps, tmp_shape2), axis=1)

df_meta = pd.DataFrame(tmp_meta, columns=['fps','seq_len','W','H','C'])
df_meta.to_csv('chradesego_meta.csv')
# sns.pairplot(df_meta)


#%% Preprocess the videos and store them
# - subsample the frames
# - crop to a square frame
# - same length sequences


# absolute location of the video files.
vid_root = r'C:/Users/Saber/Documents/tmp_dir/CharadesEgo_v1_480/'
# Load the filenames, excluding the non-ego videos
sample_paths = [s for s in os.listdir(vid_root) if s.split('.')[0][-3:]=='EGO']
# relative location of saving the preprocessed videos
save_dir = 'data_charades_preproc/'

#%% Preprocessing script
side_size = 256
crop_size = side_size
new_fps = 2
max_len = 7*new_fps #7 is the minimum length of many videos


# mean = [0.45, 0.45, 0.45]
# std = [0.225, 0.225, 0.225]


            # Lambda(lambda x: x/255.0),
            # NormalizeVideo(mean, std),
transform1 = Compose(
        [
            Lambda(lambda x: x.type(torch.float32)),
            Lambda(lambda x: torch.permute(x, (3,0,1,2))),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size=(crop_size, crop_size)),
            Lambda(lambda x: x.type(torch.uint8)),
            Lambda(lambda x: torch.permute(x, (1,2,3,0)))
        ])
        
n_samples = 500
train_data = [sample_paths[i] for i in range(n_samples)]


# i_vid=1
# n_newvids = 50
start_i = 100
end_i = 500
for i_sample in tqdm(range(start_i, end_i)):
    sample_path = sample_paths[i_sample]
    
    full_path = vid_root + sample_path
    sample = torchvision.io.read_video(full_path)
    
    c_fps = sample[2]['video_fps']
    c_duration = sample[0].shape[1]/c_fps
    num_frames = int(c_duration*new_fps)
    sample_tr = uniform_temporal_subsample(sample[0], num_frames, temporal_dim=0)
    sample_tr = transform1(sample_tr)
    new_len = sample_tr.shape[0]
    # if new_len>= max_len: split into a new video sequence
    ip = 0
    while 1:
        start_f, end_f = ip*max_len, (ip+1)*max_len
        if end_f>new_len:
            break
        c_sample = sample_tr[start_f:end_f,...]
        new_fname = 'fps'+str(new_fps)+'_'+str(ip)+'_'+sample_path
        # write the videos
        torchvision.io.write_video(save_dir+new_fname, c_sample, fps=new_fps)
        ip+=1
    

# xtt = torchvision.io.read_video('005BU_tr.mp4')[0]