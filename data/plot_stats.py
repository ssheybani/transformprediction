import numpy as np
import pandas as pd
import json
from skimage import io

from matplotlib import pyplot as plt

from json2array import json2df, get_all_samples
from pathlib import Path

def spherical2cartesian(d, a, e):
    # d: distance
    # a: azimuth (radians)
    # e: elevation (radians)
    # paste from the implemented version
    tmp = d*np.cos(e)
    x = tmp* np.cos(a)
    y = d* np.sin(e)
    z = tmp*np.sin(a)
    return x,y,z

def plot_pose(sample_arr):
    dist, azim, elev = sample_arr[:,0], sample_arr[:,1], sample_arr[:,2]
    # z,y axes are different in Unity3D.
    rotVecs = 0.02*np.cos(sample_arr[:,3:])
    rotX, rotZ, rotY = rotVecs[:,0],rotVecs[:,1],rotVecs[:,2]
    xx, xz, xy = spherical2cartesian(dist, azim, elev)
    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    
    figsize = (8,8)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(45.,45.)
    xlim, ylim, zlim = [-1,1],[-1,1],[-0.5,1.5]

    ax.quiver(xx, xy, xz, rotX, rotY, rotZ,
              linewidths=1.)
    ax.set_xlim(xlim)
    ax.set_xlim(ylim)
    ax.set_xlim(zlim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def plot_position(sample_arr):
    dist, azim, elev = sample_arr[:,0], sample_arr[:,1], sample_arr[:,2]
    # z,y axes are different in Unity3D.
    # rotVecs = 0.02*np.cos(sample_arr[:,3:])
    # rotX, rotZ, rotY = rotVecs[:,0],rotVecs[:,1],rotVecs[:,2]
    xx, xz, xy = spherical2cartesian(dist, azim, elev)
    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    
    figsize = (8,8)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(45.,45.)
    xlim, ylim, zlim = [-1,1],[-1,1],[-0.5,1.5]

    ax.scatter(xx, xy, xz)
    # ax.quiver(xx, xy, xz, rotX, rotY, rotZ,
    #           linewidths=1.)
    ax.set_xlim(xlim)
    ax.set_xlim(ylim)
    ax.set_xlim(zlim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def plot_orientation(sample_arr):
    # dist, azim, elev = sample_arr[:,0], sample_arr[:,1], sample_arr[:,2]
    # z,y axes are different in Unity3D.
    rotVecs = 0.02*np.cos(sample_arr)
    rotX, rotZ, rotY = rotVecs[:,0],rotVecs[:,1],rotVecs[:,2]
    # xx, xz, xy = spherical2cartesian(dist, azim, elev)
    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    
    figsize = (8,8)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(45.,45.)
    # xlim, ylim, zlim = [-1,1],[-1,1],[-0.5,1.5]

    ax.scatter(rotX, rotY, rotZ)
    # ax.set_xlim(xlim)
    # ax.set_xlim(ylim)
    # ax.set_xlim(zlim)
    ax.set_xlabel('RotX')
    ax.set_ylabel('RotY')
    ax.set_zlabel('RotZ')


path2json = '../saved_images/dataset_Dec20/RubberDuck_01/'+ \
    '0000030/labels.json'

rename_dict = {'pos.x': 'posD', 
            'pos.y': 'posA',
            'pos.z': 'posE',
            'rot.x': 'rotX', 
            'rot.y': 'rotY',
            'rot.z': 'rotZ',
            'dPos.x': 'dPosD', 
            'dPos.y': 'dPosA', 
            'dPos.z': 'dPosE', 
            'dRotAx.x': 'dRotAxX',
            'dRotAx.y': 'dRotAxY',
            'dRotAx.z': 'dRotAxZ'
            }
fields = ['shape', 'posD', 'posA', 'posE', 'rotX', 'rotY', 'rotZ',
          'dPosD', 'dPosA', 'dPosE', 'dRotAxX', 'dRotAxY', 'dRotAxZ',
          'dRot']
hist_fields = ['posD', 'posA', 'posE', 'rotX', 'rotY', 'rotZ',
          'dPosD', 'dPosA', 'dPosE', 'dRotAxX', 'dRotAxY', 'dRotAxZ',
          'dRot']

arr_fields = ['posD', 'posA', 'posE', 'rotX', 'rotY', 'rotZ']


sample_df = json2df(path2json, rename_dict=rename_dict, 
                    field_list=fields)
sample_arr = sample_df[arr_fields].to_numpy()

plot_pose(sample_arr)

path2imgs = '../saved_images/dataset_Dec20/RubberDuck_01/'+ \
    '0000030/*.png'

images = io.ImageCollection(path2imgs)

fig, ax = plt.subplots(1,1)
for i in range(len(images)):
    ax.imshow(images[i], alpha=0.3)


ds_dir = r'../saved_images/dataset_Dec20/RubberDuck_01/'
ds_path_P = Path(ds_dir) 

all_df = get_all_samples(ds_path_P)

_ = all_df.hist(column=hist_fields, bins=30, figsize=(10,10))