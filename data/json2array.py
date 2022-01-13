import numpy as np
import pandas as pd
import json
from skimage import io

from matplotlib import pyplot as plt

from pathlib import Path
import glob




def json2df(path2json, rename_dict=None, field_list=None):
    with open(path2json,'r') as f:
        js_data = json.loads(f.read())# Normalizing data
    df = pd.json_normalize(js_data, record_path =['trajLabels'])
    if rename_dict is not None:
        df = df.rename(columns=rename_dict)
    if field_list is not None:
        df = df[fields]
    return df

    

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
arr_fields = ['posD', 'posA', 'posE', 'rotX', 'rotY', 'rotZ']
hist_fields = ['posD', 'posA', 'posE', 'rotX', 'rotY', 'rotZ',
          'dPosD', 'dPosA', 'dPosE', 'dRotAxX', 'dRotAxY', 'dRotAxZ',
          'dRot']
sample_df = json2df(path2json, rename_dict=rename_dict, 
                    field_list=fields)
sample_arr = sample_df[arr_fields].to_numpy()

def sample_gen(ds_dir, file_pattern='*.json'):
    ds_path_P = Path(ds_dir)
    for fname in ds_path_P.glob('**/'+file_pattern):
        yield json2df(fname, rename_dict=rename_dict, 
                field_list=fields)

def get_all_samples(ds_dir, return_type='flat_df'):
    # Written for only one object
    # Internally calls sample_gen which uses the following 
    # global variables: rename_dict, fields
    df_list = []
    for xdf in sample_gen(ds_dir):
        df_list.append(xdf)
    
    if return_type=='flat_df':
        return pd.concat(df_list, axis=0)
    return df_list
    

ds_dir = r'../saved_images/dataset_Dec20/RubberDuck_01/'
ds_path_P = Path(ds_dir) 

all_df = get_all_samples(ds_path_P)

_ = all_df.hist(column=hist_fields, bins=30, figsize=(10,10))

xtt = all_df['posD']
ytt = xtt[xtt!=1]
_ = ytt.hist(bins=100)

from plot_stats import plot_position, plot_orientation
xpos = all_df[['posD', 'posA', 'posE']].to_numpy()
xori = all_df[['rotX', 'rotY', 'rotZ']].to_numpy()
plot_position(xpos)
plot_orientation(xori)
