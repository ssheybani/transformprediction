#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.path)


# In[2]:


# import torch
import os, re
import cv2
from tqdm import tqdm
import numpy as np
# import torchvision


# In[3]:


def center_crop(image, crop_size):
    center = np.array(image.shape) / 2
    x = center[1] - crop_size/2
    y = center[0] - crop_size/2

    crop_img = image[int(y):int(y+crop_size), int(x):int(x+crop_size), :]
    return crop_img


# In[4]:


vid_root = r"/N/slate/sheybani/ego4ddata/v1/clips/"
out_root = r"/N/slate/sheybani/ego4ddata/pp_images/"
sample_paths = os.listdir(vid_root)

for i,item in enumerate(sample_paths):
    suffix = item.split('.')[1]
    if (suffix !='mp4') and (suffix !='MP4'):
        del sample_paths[i]


# In[5]:


new_fps = 2
crop_size= 256

start_sample = 0
end_sample = 500
for i in tqdm(range(start_sample, end_sample)):
    clip_name = sample_paths[i]
    new_dir = clip_name.split('.')[0]
#     new_dir = re.sub(r'\W+', '', new_dir) #remove non-alphanumeric characters
    try:
        chunks_dir = os.path.join(out_root, new_dir)
        os.mkdir(chunks_dir)
    except OSError as error: 
        print(error)
        
    fnum_o = 0
    fnum_n = 0
    vidcap = cv2.VideoCapture(vid_root+clip_name)
    orig_fps = vidcap.get(cv2.CAP_PROP_FPS)
    ds_rate = int(orig_fps/new_fps)
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret: 
            w,h,_ = frame.shape
            s_factor = crop_size/min(w,h)
            frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation= cv2.INTER_CUBIC)
            frame = center_crop(frame, crop_size)
            fname = os.path.join(chunks_dir, '{:06d}.jpg'.format(fnum_n))
            cv2.imwrite(fname, frame)
            fnum_o += ds_rate 
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, fnum_o)
            fnum_n +=1
        else:
            vidcap.release()
            break
#     fnum_o +=1


# In[53]:


# import matplotlib.pyplot as plt
# plt.imshow(resize_down)


# In[ ]:




