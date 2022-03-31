
#%% Imports
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import matplotlib.pyplot as plt

#%% Prepare the model and the test data

# Run the following scripts before this:
    # load_dataset.py: defines train_dataloader, test_dataloader
    # train0.py: defines xmodel
# xmodel is defined in train0.py 

train_dataloader = torch.utils.data.DataLoader(
    train_data_te, batch_size=100, shuffle=False)

model = xmodel

for xclips, (xdlabels, xclabels) in train_dataloader:
    break
n_samples = 100
images = xclips[:n_samples,...]
clabels = xclabels[:n_samples,...]
dlabels = xdlabels[:n_samples,...]
seq_len = images.shape[1]

#%% Setup the forward hooks and collect the activations.
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

h1 = model.shared_encoder.feature_extractor[7].register_forward_hook(get_activation('l_shared'))
h2 = model.dorsal.rnn.register_forward_hook(get_activation('l_dorsal'))
# h3 = model.shared_encoder.feature_extractor[7].register_forward_hook(get_activation('l_shared'))

out = model(images)

xtt1 = activation['l_shared'].detach().numpy()
xtt1 = xtt1.reshape((n_samples, seq_len, *xtt1.shape[1:]))
xtt2 = activation['l_dorsal'][0].detach().numpy()
xtt2_state = activation['l_dorsal'][1][0].detach().numpy()
h1.remove()
h2.remove()

#%% Analyze the collected data
import pandas as pd
import seaborn as sns
from sklearn import metrics, decomposition


#%% Correlation between the positional target variables and the activations

# Pairwise correlation between the targets and the memory units

targets = clabels.detach().numpy()
target_signals = targets.reshape((n_samples*seq_len, -1))

memory_signals = xtt2.reshape((n_samples*seq_len, -1))

xdata = np.hstack((target_signals, memory_signals))

columns=['posD', 'posA', 'posE']+\
    [str(i) for i in range(30)]
xdf = pd.DataFrame(data=xdata, columns=columns)

corrs = xdf.corr()
# np.corrcoef(target_signals, memory_signals)
sns.heatmap(corrs,cmap='seismic', vmin=-1, vmax=1)

#%% Class separability

# shared_enc_flt = 
xdf_base = pd.DataFrame(data=xtt1.reshape((n_samples*seq_len, -1)))
sample_pwdist = metrics.pairwise.cosine_similarity(xdf_base)
sns.heatmap(sample_pwdist)

# matrix rank
xpca = decomposition.PCA().fit(xdf_base)
plt.plot(xpca.singular_values_, label='Singular values'); plt.legend()


#%% Send the embeddings to Tensorboard

i=0
writer = SummaryWriter('runs/test'+str(i))
writer.add_embedding(xdf_base.to_numpy(),
                     label_img=images.reshape((n_samples*seq_len, *images.shape[2:])), 
                     tag='shared_encoder')
writer.close()

# Create metadata file to load as labels in the Tensorboard Projector
dig_bins = [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2]
target_dig = np.digitize(target_signals, dig_bins)
metadata = np.concatenate((target_dig, 
                           dlabels.detach().numpy().reshape((-1,1))), axis=1)
df_metadata = pd.DataFrame(metadata, columns=['posD','posA', 'posE', 'class'])
df_metadata.to_csv('sample_labels.tsv', sep="\t", index=False)

