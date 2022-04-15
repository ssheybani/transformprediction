# Following the example at: https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/


#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
import json
import urllib



# from pytorchvideo.data.encoded_video import EncodedVideo

# from torchvision.transforms import Compose, Lambda
# from torchvision.transforms._transforms_video import (
#     CenterCropVideo,
#     NormalizeVideo,
# )
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     ShortSideScale,
#     UniformTemporalSubsample
# )

from torch.utils.tensorboard import SummaryWriter
#%% Load the model
# Choose the `slow_r50` model 
# model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

model = torchvision.models.video.r2plus1d_18(pretrained=True)

# downloaded the R3D (SLOW_8x8_R50) model trained on Kinetics dataset.
# download size: 250mb
# Saved in C:\Users\Saber/.cache\torch\hub\checkpoints\SLOW_8x8_R50.pyth


#%%
# Set to GPU or CPU
device = "cpu"
model = model.eval()
model = model.to(device)

# Download the id to label mapping for the Kinetics 400 dataset on 
# which the torch hub models were trained. This will be used to get the category 
# label names from the predicted class ids.

json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try: urllib.URLopener().retrieve(json_url, json_filename)
except: urllib.request.urlretrieve(json_url, json_filename)

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

#%% Define input transform

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 8
sampling_rate = 8
frames_per_second = 30

# Note that this transform is specific to the slow_R50 model.
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second


#%% Run Inference

# Download an example video.
url_link = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
video_path = 'archery.mp4' #'sample-mp4-file-small.mp4'#
try: urllib.URLopener().retrieve(url_link, video_path)
except: urllib.request.urlretrieve(url_link, video_path)

#%% Load the video and transform it to the input format required by the model.

# Select the duration of the clip to load by specifying the start and end duration
# The start_sec should correspond to where the action occurs in the video
start_sec = 0
end_sec = start_sec + clip_duration

# Initialize an EncodedVideo helper class and load the video

video = EncodedVideo.from_path(video_path)

# Load the desired clip
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
# video_data = video.get_clip(start_sec=start_sec, end_sec=3)

# Apply a transform to normalize the video input
video_data = transform(video_data)

# Move the inputs to the desired device
inputs = video_data["video"]
inputs = inputs.to(device)


#%% Get predictions

preds = model(inputs[None, ...])

# Get the predicted classes
post_act = torch.nn.Softmax(dim=1)
preds = post_act(preds)
pred_classes = preds.topk(k=5).indices[0]

# Map the predicted classes to the label names
pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))


#%% Send the model to Tensorboard
images = inputs[None, ...]#inputs.permute(1,0,2,3)
# images = torch.cat((images, inputs[None, ...]),dim=0)

i = 0
writer = SummaryWriter('runs/test'+str(i))
writer.add_graph(model, images) #model_alexnet
writer.close()


#%% Visualize some of the embeddings


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

# h1 = model.blocks[1].res_blocks[2].activation.register_forward_hook(get_activation('l_r1'))
# h2 = model.blocks[4].res_blocks[2].activation.register_forward_hook(get_activation('l_r4'))
# h3 = model.blocks[5].pool.register_forward_hook(get_activation('l_pool'))

h1 = model.blocks[1].res_blocks[2].activation.register_forward_hook(get_activation('l_r1'))
h2 = model.blocks[4].res_blocks[2].activation.register_forward_hook(get_activation('l_r4'))
h3 = model.avgpool.register_forward_hook(get_activation('l_pool'))


out = model(images)

xtt1 = activation['l_r1'].detach().numpy()
# xtt1 = xtt1.reshape((n_samples, seq_len, *xtt1.shape[1:]))
xtt2 = activation['l_r4'].detach().numpy()
xtt3 = activation['l_pool'].detach().numpy()

# xtt2_state = activation['l_dorsal'][1][0].detach().numpy()
h1.remove()
h2.remove()
h3.remove()

#%%
t=0
# i = 3
# plt.imshow(xtt2[0, i, t, ...])

# lots of zero 8x8 outputs
xtt2_nz = []
for i in range(2048):
    tmp = xtt2[0, i, t, ...]
    if np.sum(tmp)>5:
        xtt2_nz.append(tmp)

xtt11 = images.detach().squeeze().permute(1,2,3,0).numpy()

fig, ax = plt.subplots(10,10, figsize=(10,10))
ax[0,0].imshow(xtt11[t,...])
for j in range(1,10):
    for k in range(10):
        ax[j,k].imshow(xtt2_nz[j*10+k])

#%%

#MDS of the time frames
n_samples, seq_len = 2, 8
img_perm = images.permute(0,2,1,3,4)
label_img = img_perm.reshape(
    (n_samples*seq_len, *img_perm.shape[2:]))

xh2 = xtt2.transpose(0,2,1,3,4).reshape((n_samples*seq_len, -1))

i=0
writer = SummaryWriter('runs/test'+str(i))
writer.add_embedding(xh2,
                     label_img=label_img, 
                     tag='shared_encoder')
writer.close()


# xtt = model.stem[3].weight.detach().numpy().reshape(-1,3)

xtt = model.layer2[1].conv2[0][3].weight.detach().numpy().reshape(-1,3)
plt.imshow(xtt[np.random.randint(0, xtt.shape[0], 20), ...])
