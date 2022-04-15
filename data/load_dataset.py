# Functions for properly loading the HDF files into a Torch dataset for a training task.
#%%
import numpy as np
import torch, torchvision
import skimage 
# from skimage import color as skcolor
# from skimage import util as skutil
import h5py
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

#%%
def weighted_mse_loss(pred, target, weight):
    return torch.sum(weight * (pred - target) ** 2)

class movingToysDataset():

# preprocess_images:
    # for now, just do rgba2rgb
    # to_lab:
        # new_clips = np.zeros(clips.shape[0], clips.shape[1], clips.shape[2], 3)
        # for j in range(4)
            # new_clips[:,j, ...] = skimage.io.rgb2lab(rgba2rgb(image))
    # to_gs
    # to_gabor

# normalize_targets:
    # use sklearn Scalers

# split train and test
    # copy from nma_datasets

# to_torchDataset:
    # torch.Tensor()
    # returns one tensor for input and another for target.


# Then we'd create a basic torch dataset from it.


    def __init__(self, dataset_path, filetype='hdf'):
        """
        Initializes movingToysDataset instance, sets basic attributes and 
        metadata attributes.

        Optional args:
        - dataset_path (str): path to dataset 
            (default: global variable DEFAULT_DATASET_NPZ_PATH)

        Attributes:
        - dataset_path (str): path to the dataset
        - npz (np.lib.bpyio.NpzFile): zipped numpy data file
        - num_images (int): number of images in the dataset
        """

        self.dataset_path = dataset_path
        # self.npz = np.load(
        #     self.dataset_path, allow_pickle=True, encoding="latin1"
        #     )
        # self._load_metadata()
        if filetype=='hdf':
            dsfile = h5py.File(dataset_path, 'r')
        elif filetype=='npz':
            dsfile = np.load(
                dataset_path, allow_pickle=True, encoding="latin1"
            )
        else:
            raise ValueError('filetype must be either hdf or npz')
        # self.clips = np.zeros_like(dsfile['clips'])
        print('Loading data')
        self.clips = dsfile['clips'][()] #shape: (10, 502, 4, 256, 256, 4)
        self.latents_values = dsfile['latents_values'][()] #shape: 10, 502, 4, 14)
        
        self.latents_classes = dsfile['latents_classes'][()]
        # Create attributes for different pathways
        # latents_translation_rotation
        # latents_identity

        if filetype=='hdf':
            self.metadata = dsfile['metadata']
        else:
            self.metadata = None
        
        self.num_objects, self.num_clips, self.clip_len, self.img_w, self.img_h = \
            self.clips.shape[:5]

        # self.num_clatents = self.
        # self.value_to_shape_name_map = {
        #     1: "RubberDuck_01",
        #     2: "BabyMug_01"
        # } # add the rest after creating the first version of the class.

        # self.shape_name_to_value_map = {
        #     value: key for key, value in self.value_to_shape_name_map.items()
        #     }

    def __repr__(self):
        return f"Moving 3D Toys dataset"
    
    def __getitem__(self, idx):
        if type(idx)==int:
            return (self.clips[idx,...], (self.dtargets[idx, ...], 
                                          self.ctargets[idx,...]))
        elif len(idx)==2:
            return (self.clips[idx[0], idx[1],...], 
                    (self.dtargets[idx[0], idx[1], ...],
                     self.ctargets[idx[0], idx[1],...]))
        else:
            raise ValueError
        # elif type(idx)==np.ndarray and idx.shape[0]==self.num_objects:
            
    def _choose_targets(self, target_names):
        if self.metadata is not None:
            self.latents_names = list(self.metadata.attrs['latents_names'])
        dtarget_names, ctarget_names = target_names
        dtarget_inds = [self.latents_names.index(tname) for tname in dtarget_names]
        ctarget_inds = [self.latents_names.index(tname) for tname in ctarget_names]
        return dtarget_inds, ctarget_inds

    def _preprocess_continuous_targets(self, ctargets):
        print('Normalizing the target variables')
        ctargets_shape = ctargets.shape
        ctargets_reshaped = ctargets.reshape(-1, ctargets.shape[-1])
        self.scaler = StandardScaler().fit(ctargets_reshaped)
        # self.scaler = MinMaxScaler(feature_range=(-1, 1), clip=False) #if we later want 
        ctargets_tr_reshaped = self.scaler.transform(ctargets_reshaped)
        ctargets_tr = ctargets_tr_reshaped.reshape(ctargets_shape)
        return ctargets_tr
    
    def _preprocess_digitize(self, ctargets):
        # assumes ctargets is already centered and scaled.
        ctargets_shape = ctargets.shape
        ctargets_reshaped = ctargets.reshape(-1, ctargets.shape[-1])
        # bins = [-1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1]
        bins = [-2, -1, -0.5, -0.25, 0., 0.25, 0.5, 1., 2.]
        ctargets_tr_reshaped = np.digitize(ctargets_reshaped, bins)
        ctargets_tr = ctargets_tr_reshaped.reshape(ctargets_shape)
        return ctargets_tr
        
        
    def make_targets(self):
        target_names = (
            ['shape'],
            ['posD',
            'posA',
            'posE']
            )
        # target_names = (
        #     ['shape'],
        #     ['dPosD',
        #     'dPosA',
        #     'dPosE',
        #     'dRotAxX',
        #     'dRotAxY',
        #     'dRotAxZ',
        #     'dRot']
        #     )
        # dtarget_inds, ctarget_inds = dset._choose_targets(target_names)
        dtarget_inds, ctarget_inds = self._choose_targets(target_names)
        
        self.dtargets = self.latents_values[...,dtarget_inds]
        self.ctargets = self.latents_values[...,ctarget_inds]
        # xtt = dset.latents_values[...,ctarget_inds]
        # ytt = dset._preprocess_continuous_targets(xtt)
        # _ = plt.hist(ytt[...,1].flatten())
        self.ctargets = self._preprocess_continuous_targets(
            self.ctargets)
        self.ctargets = self._preprocess_digitize(self.ctargets)
        return self.dtargets, self.ctargets

    def preprocess_images(self, **kwargs):
        # clips_shape = list(self.clips.shape)
        # clips_shape[-1] = 3
        # new_clips = np.zeros(tuple(clips_shape), dtype='uint8')
        # print('Converting images from RGBA to RGB')
        # # self.clips = skcolor.rgba2rgb(
        # #     self.clips, background=(1, 1, 1), channel_axis=-1)
        # # The above code gives memory error (needs ~3GB for each 500 clips
        # # because the output of rgba2rgb are float64 images instead of uint8)
        
        # for i in range(self.num_objects):
        #     print('...Object ',i)
        #     for j in range(self.num_clips):
        #         if j%100==0:
        #             print('... Sample ', j)
        #         tmp_slice = self.clips[i,j,...]
        #         tmp_new = skcolor.rgba2rgb(
        #             tmp_slice, background=(1, 1, 1), channel_axis=-1)
        #         new_clips[i,j,...] = skutil.img_as_ubyte(tmp_new)
        # self.clips = new_clips
        return self.clips
    
    

    def train_test_split_idx(self, fraction_train=0.8, randseed=None,
        melt=True, **kwargs):
        """
         scenario 0: simplest test: 
            # learning the same dataset
                # test: for each object, 20% of the clips are withheld for testing
         scenario 1: rotation and translation of unseen objects
        
        - Returns
            train_indices: numpy array of shape (num_objects, num_clips)
            test_indices: numpy array of shape (num_objects, num_clips)

        """
        if isinstance(randseed, int):
            np.random.seed(randseed)
        elif randseed is None:
            pass
        else:
            raise ValueError('randseed must be an integer')

        all_inds = np.zeros((self.num_objects, self.num_clips))

        for i in range(self.num_objects):
            obj_i_inds = np.random.permutation(self.num_clips)
            all_inds[i,:] = deepcopy(obj_i_inds)
        
        train_size = int(fraction_train * self.num_clips)
        train_indices = all_inds[:,:train_size]
        test_indices = all_inds[:,train_size:]
        
        if melt:
            melt_func = lambda arr: [(index[0], int(item)) 
                                  for index, item in np.ndenumerate(arr)]
            return melt_func(train_indices), melt_func(test_indices)
        return train_indices, test_indices
    
    def get_sample_set(self, sample_set_idx):
        return [self[idx] for idx in sample_set_idx]
    
    @staticmethod
    def get_torchtensor(sample_set):
        n_samples = len(sample_set)
        clips_shape = tuple([n_samples, 
                             *sample_set[0][0].shape])
        dlabels_shape = tuple([n_samples, 
                             *sample_set[0][1][0].shape])
        clabels_shape = tuple([n_samples, 
                             *sample_set[0][1][1].shape])
        clips_te = torch.zeros(clips_shape, dtype=torch.float32)
        dlabels_te = torch.zeros(dlabels_shape, dtype=torch.int32)
        clabels_te = torch.zeros(clabels_shape, dtype=torch.float32)
        
        for i in range(n_samples):
            tmp_clip, tmp_labels = sample_set[i]
            clips_te[i,...] = torch.as_tensor(skutil.img_as_float32(tmp_clip))
            dlabels_te[i,...], clabels_te[i,...] = \
                torch.as_tensor(tmp_labels[0]), torch.as_tensor(tmp_labels[1])
        clips_te = torch.permute(clips_te, (0, 1, 4, 2, 3))
        return clips_te, (dlabels_te, clabels_te)

#%%
def change_range(data, old_range, new_range):
    assert len(old_range)==2 and len(new_range)==2
    old_loc = np.mean(old_range)
    old_scale = old_range[1]-old_loc

    new_loc = np.mean(new_range)
    new_scale = new_range[1]-new_loc

    ratio = new_scale/old_scale
    return ((data-old_loc)*ratio)+new_loc

def clip_torchfloat2npimage(clip_te, from_range=[0,1], to_range=[0,1], to_dtype=np.float32):
    # clip_te: range=[-1.,1.]. shape=(N,C,W,H)
    clip2 = clip_te.detach().numpy().transpose(0,2,3,1)
    # img3 = img2*2-1
    if from_range!=to_range:
        clip2 = change_range(clip2, from_range, to_range)#(clip2/2)+0.5
    return clip2.astype(to_dtype)

def clip_npimage2torchfloat(clip_np, from_range=[0,1], to_range=[0,1]):
    # clip_np: shape=(N,W,H,C)
    if from_range!=to_range:
        clip_np = change_range(clip_np.astype(np.float32), 
            from_range, to_range)
    clip2 = torch.permute(
        torch.as_tensor(clip_np, dtype=torch.float32),
            (0, 3, 1, 2))
    return clip2

# def clip_uint2torchfloat(clip_np, from_range):
#     clip2 = 2*(clip_np.astype(np.float32)/255 -0.5)
#     clip3 = torch.permute(
#         torch.as_tensor(clip2, dtype=torch.float32),
#             (0, 3, 1, 2))
#     return clip3

#%%
class movingToysTorchDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        # assuming idx is an int
        xclips, (dlabels, clabels) = self.data[idx]
        xclips2 = clip_npimage2torchfloat(xclips, 
            from_range=(0,255), to_range=(0,1))
        return xclips2, (torch.as_tensor(dlabels, dtype=torch.float32), 
                         torch.as_tensor(clabels, dtype=torch.float32))
    def __len__(self):
        return len(self.data)
    
#%%
dataset_path = 'ds_jan25.hdf'
filetype = 'hdf'

dset = movingToysDataset(dataset_path, filetype=filetype)

_ = dset.make_targets()
_ = dset.preprocess_images()

train_idx, test_idx = dset.train_test_split_idx()
train_data = dset.get_sample_set(train_idx)
train_data_te = movingToysTorchDataset(train_data)

xds = train_data
def get_narr_mem(narr):
    return narr.size*narr.itemsize
size_on_mem = len(xds)*(
    get_narr_mem(xds[0][0])+\
        get_narr_mem(xds[0][1][0])+\
            get_narr_mem(xds[0][1][1]))

print('Train dataset size on memory (in megabytes): ', size_on_mem/(1024*1024))

# Creating a torch dataloader
train_dataloader = torch.utils.data.DataLoader(
    train_data_te, batch_size=32, shuffle=True)

test_data = dset.get_sample_set(test_idx)
test_data_te = movingToysTorchDataset(test_data)
test_dataloader = torch.utils.data.DataLoader(
    test_data_te, batch_size=32, shuffle=True)

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels[0].size()}{train_labels[1].size()}")
# xclip = train_features[0].squeeze()

# xclip, (xdlabel, xclabel)=train_data[101]
# plt.imshow(xclip[1,...])

# clips_te, (dlabels_te, clabels_te) = dset.get_torchtensor(train_data)


