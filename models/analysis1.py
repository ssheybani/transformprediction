
#%% Imports
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import torch


#%% Some utility functions

def to_static(img_seq):
    n_samples, seq_len, *other_dims = img_seq.shape
    return img_seq.reshape((n_samples*seq_len, *other_dims))

def to_seq(img_flt, seq_len):
    n_samplesXseq_len, *other_dims = img_flt.shape
    n_samples = int(n_samplesXseq_len/seq_len)
    return img_flt.reshape((n_samples, seq_len, *other_dims))


#%% Prepare the model and the test data

# Run the following scripts before this:
    # load_dataset.py: defines train_dataloader, test_dataloader
    # train0.py: defines xmodel
# xmodel is defined in train0.py 

n_samples = 10

train_dataloader = torch.utils.data.DataLoader(
    train_data_te, batch_size=n_samples, shuffle=True)

model = xmodel

# for xclips, (xdlabels, xclabels) in train_dataloader:
for xclips in train_dataloader:
    break

images = xclips[:n_samples,...]
# clabels = xclabels[:n_samples,...]
# dlabels = xdlabels[:n_samples,...]
seq_len = images.shape[1]

#%% Setup the forward hooks and collect the activations.
model.train()
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

# h1 = model.layer1[2].relu.register_forward_hook(get_activation('l1'))
h2 = model.avgpool.register_forward_hook(get_activation('l4'))


out = model(to_static(images))

# xtt1 = activation['l1'].detach().numpy()
xtt2 = activation['l4'].detach().numpy().squeeze()
# h1.remove()
h2.remove()

#%% Analyze the collected data
import pandas as pd
import seaborn as sns
from sklearn import metrics, decomposition
import matplotlib.pyplot as plt

embeddings = xtt2#images.detach().numpy()#

target_signals = np.concatenate((
    dlabels.detach().numpy().reshape((n_samples*seq_len, -1)),
    clabels.detach().numpy().reshape((n_samples*seq_len, -1))), axis=1)
# target_signals = targets.reshape((n_samples*seq_len, -1))

embedding_signals = embeddings.reshape((n_samples*seq_len, -1))

xdata = np.hstack((target_signals, embedding_signals))

target_labels = ['shape','posD', 'posA', 'posE']
embedding_labels = [str(i) for i in range(embedding_signals.shape[1])]
columns= target_labels+embedding_labels
    
xdf = pd.DataFrame(data=xdata, columns=columns)
xdf = pd.DataFrame(data=embedding_signals)
xdf.to_csv('tmp_resnet_untrained_charadesego2.csv', index=False)
# xdf.to_csv('tmp_raw_images_charadesego.csv', index=False)
xdf = pd.read_csv('tmp_resnet_untrained_charadesego2.csv')

sample_pwdist = metrics.pairwise_distances(xdf, metric='correlation')

fig,ax = plt.subplots(1,1)
ax = sns.heatmap(sample_pwdist, vmax=1.75)#, cmap='seismic', vmin=-1, vmax=1)
for i in range(n_samples):
    ax.axvline(x=seq_len*i, ymin=0, ymax=seq_len*n_samples)
    ax.axhline(y=seq_len*i, xmin=0, xmax=seq_len*n_samples)
#%% Similarity by Class 

byvar = 'shape'
# tmp_xdf = xdf.sort_values(by=byvar, axis=0)
tmp_xdf = xdf
sample_pwdist = metrics.pairwise.cosine_similarity(tmp_xdf[embedding_labels])
sample_pwdist = pd.DataFrame(sample_pwdist, columns=tmp_xdf[byvar].astype(int))
sns.heatmap(sample_pwdist)

# matrix rank
# xpca = decomposition.PCA().fit(xdf)
# plt.plot(xpca.singular_values_, label='Singular values'); plt.legend()


#%% Send the embeddings to Tensorboard

i=0
writer = SummaryWriter('runs/test'+str(i))
writer.add_embedding(xdf.to_numpy(),
                     label_img=images.reshape((n_samples*seq_len, *images.shape[2:])), 
                     tag='resnet50')
writer.close()

# Create metadata file to load as labels in the Tensorboard Projector

df_metadata = pd.DataFrame(target_signals, columns=['shape','posD','posA', 'posE'])
df_metadata.to_csv('sample_labels.tsv', sep="\t", index=False)

