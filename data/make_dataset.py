"""
- Save the dataset as a collection of numpy arrays following the example of dSprites:
    - clips:
        (clips x timesteps x height x width x channel))
        (5000 x 4 x 256x 256, 3, uint8) Images in RGB.
    - latents_values
        (5000 x 4 x 8, float64) Values of the latent factors.
    - latents_classes
        (5000 x 4 x 8, int64) Integer index of the latent factor values. 
    - metadata
        'title': 'Moving 3D Toys Dataset',
        'author': 'sheybani.saber@gmail.com',
        'date': ds_date,
        'version': self.version,
        'description': 'Fill Later!',
        'latents_names': 
            ('shape', 'dPosD', 'dPosA', 'dPosE', 'dRotAxX', 'dRotAxY', 'dRotAxZ', 'dRot'), 
        'latents_sizes': 10
"""

import numpy as np
import pandas as pd
import json
import os
import skimage
from skimage import io

from matplotlib import pyplot as plt

from pathlib import Path
import glob

import datetime
import h5py
from copy import deepcopy

# By the end, you should have a path to a valid NPZ file, containing the datset specified in the format above.


def json2df(path2json, rename_dict=None, field_list=None):
    with open(path2json,'r') as f:
        js_data = json.loads(f.read())# Normalizing data
    df = pd.json_normalize(js_data, record_path =['trajLabels'])
    if rename_dict is not None:
        df = df.rename(columns=rename_dict)
    if field_list is not None:
        df = df[fields]
    return df

    
class MakeDataset():
    hp = {}
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
    all_fields = ['shape', 'posD', 'posA', 'posE', 'rotX', 'rotY', 'rotZ',
          'dPosD', 'dPosA', 'dPosE', 'dRotAxX', 'dRotAxY', 'dRotAxZ',
          'dRot']
    target_fields = ['shape', 'dPosD', 'dPosA', 'dPosE', 'dRotAxX', 'dRotAxY', 'dRotAxZ', 'dRot']
    clips = []
    latents_values = []
    latents_classes = []
    metadata = {}
    bins = 10
    sample_fnames = [] #for debugging

    def __init__(self, **kwargs):
        self.hp = kwargs
        if 'target_fields' in kwargs:
            self.target_fields = kwargs['target_fields']
        if 'version' in kwargs:
            self.version = kwargs['version']
        else:
            self.version = '1.0'
        # if 'clip_len' in kwargs:
        #     self.clip_len = kwargs['clip_len']
        # else:
        #     self.clip_len = 4
        if 'bins' in kwargs:
            self.bins = kwargs['bins']
        # else:
        #     self.bins = 10
        self.latents_names = self.target_fields
        self.latents_sizes = self.bins

        self.value_to_shape_name_map = {
            0: "RubberDuck_01",
            1: "ButterflyToy_01",
            2: "BabyMug_01",
            3: "MilkBottle_01",
            4: "Spoon_01"
        } # add the rest after creating the first version of the class.

        self.shape_name_to_value_map = {
            value: key for key, value in self.value_to_shape_name_map.items()
            }

    def __repr__(self):
        return f"Moving Toys dataset"

    
    def _make_clips_and_values(self, dsdir):
        ds_path = Path(dsdir)
        # Go through the directories one-by-one
        for categ_path in ds_path.iterdir():
            if categ_path.is_dir() is False:
                continue
            clips_in_categ = []
            latent_vals_in_categ = []
            # inside the main dir
            for clip_path in categ_path.iterdir():
                if clip_path.is_dir() is False:
                    continue
                # inside the dir of one object category

                self.sample_fnames.append(clip_path)
                # Append the images in the directory to self.clips. 
                # Follow numbers to make sure the order is kept.
                if clip_path.as_posix()[-2:]=='00':
                    print('clip_path: ', clip_path)
                    
                # if clip_path.as_posix()[-2:]=='02': #@@@@@@ early stop for debugging
                #     break
                clips_in_categ.append(
                    skimage.io.ImageCollection(
                        clip_path.absolute().as_posix()+'/*.png').concatenate() 
                    )
                tmp_labels_df = json2df(clip_path/'labels.json', 
                        rename_dict=self.rename_dict, 
                        field_list=self.target_fields
                    )
                tmp_labels_df = tmp_labels_df.replace({'shape':self.shape_name_to_value_map})
                # return tmp_labels_df
                # print(tmp_labels_df.dtypes)
                # identity = [
                #     self.shape_name_to_value_map[item] 
                #         for item in tmp_labels_df['shape']]

                # print('Make sure the dtype is correct')
                
                latent_vals_in_categ.append(
                    tmp_labels_df.to_numpy(copy=True).astype(float)
                    )
                # Append the labels in the directory to self.latents_values
                # self.latents_classes.append(
                #     discretize(self.latents_values[-1])
                #     )
            self.clips.append(deepcopy(clips_in_categ))
            self.latents_values.append(deepcopy(latent_vals_in_categ))
            
        self.clips = np.asarray(self.clips)
        self.latents_values = np.asarray(self.latents_values, dtype=float)
        print ('--------------------------------')


    def _make_latents_classes(self, dsdir):
        return
    def _make_metadata(self):
        tnow = datetime.datetime.now()
        ds_date = tnow.strftime("%c")

        self.metadata = {
        'title': 'Moving 3D Toys Dataset',
        'author': 'sheybani.saber@gmail.com',
        'date': ds_date,
        'version': self.version,
        'description': r"Moving3dToys is a dataset of 3D toys moving over random trajectories "+\
            "in the space of position and orientation. The dataset is creating with the aim of "+\
            "creating a representation learning dataset inspired by the visual input of toddlers "+\
            "as they play with toys, an experience which is believed to play an important role in "+\
            "the development of visual perception and motor control in humans.",
        'latents_names': self.latents_names,
        'latents_sizes': self.latents_sizes
        }

    def make(self, unitydsdir, savefname):
        # Warning: the metadata field may work incorrectly, 
        # because we're not taking special care of how the 
        # dictionaries in self.metadata are serialized.

        # Make sure unitydsdir is a valid directory
        assert os.path.isdir(unitydsdir)

        self._make_clips_and_values(unitydsdir)
        self._make_latents_classes(unitydsdir)
        self._make_metadata()
        
        np.savez_compressed(savefname+'.npz', clips=self.clips, 
                 latents_values=self.latents_values)

        with h5py.File(savefname+'.hdf', 'w') as outfile:
            outfile.create_dataset('clips', data=self.clips, compression="gzip")
            outfile.create_dataset('latents_values', data=self.latents_values, compression="gzip")
            outfile.create_dataset('latents_classes', data=self.latents_classes, compression="gzip")

            metadata_ds = outfile.create_dataset('metadata', (1,))
            for key, val in self.metadata.items():
                metadata_ds.attrs[key] = val
                

fields = ['shape', 'posD', 'posA', 'posE', 'rotX', 'rotY', 'rotZ',
          'dPosD', 'dPosA', 'dPosE', 'dRotAxX', 'dRotAxY', 'dRotAxZ',
          'dRot']
arr_fields = ['posD', 'posA', 'posE', 'rotX', 'rotY', 'rotZ']
hist_fields = ['posD', 'posA', 'posE', 'rotX', 'rotY', 'rotZ',
          'dPosD', 'dPosA', 'dPosE', 'dRotAxX', 'dRotAxY', 'dRotAxZ',
          'dRot']

mk_const = MakeDataset(target_fields=fields)

# ds_dir= '../saved_images/dataset_Dec20/'
ds_dir= '../saved_images/dataset_Jan12/'
mk_const.make(ds_dir, 'ds_jan12')


# This code correctly generates the hdf file. 
# next: can work on moving it to pytorch.
# but later, we may come back and add a digitizer.
# a potential bug can be how we used '*/.png', which may or may 
# not break on another operating system. Instead, we may use glob or pathlib tools.

