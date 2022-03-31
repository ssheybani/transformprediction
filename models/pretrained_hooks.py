
#%% Imports
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import functional as F
import torchvision
import numpy as np

#%% Data loading
train_dataloader = torch.utils.data.DataLoader(
    train_data_te, batch_size=1, shuffle=True)

#%% Rest of the code

# Models are downloaded to:
    # C:\Users\Saber/.cache\torch\hub\checkpoints
model_regnet = torchvision.models.regnet_x_400mf(pretrained=True)
model_alexnet = torchvision.models.alexnet(pretrained=True)
model_resnet = torchvision.models.resnet18(pretrained=True)
model_resnet50 = torchvision.models.resnet50(pretrained=True)
model_vgg = torchvision.models.vgg16_bn(pretrained=True)
model_vgg11 = torchvision.models.vgg11(pretrained=True)
model_squeezenet = torchvision.models.squeezenet1_1(pretrained=True)


model_resnet50._version

for xclips, (xdlabels, xclabels) in train_dataloader:
    break

xclips_np = clip_torchfloat2npimage(
    xclips[0,...], from_range=[0,1], to_range=[0,1], to_dtype=np.float32)

fig,ax = plt.subplots(10,1)
for i in range(10):
    ax[i].imshow(xclips_np[i,...])

# images = torch.rand(8, 3, 64, 64)
images = xclips[0,...]
# len_clip = 4
# batch_size = len(xclips)

# xencs = []
# xress = []
# xouts = []
# for i in range(len_clip):
#     batch_img = xclips[:,i,...]
#     # xmodel.init_state()
#     xouttmp, xenctmp, xrestmp = xmodel(batch_img)
#     xouts.append(xouttmp.detach().numpy())
#     xencs.append(xenctmp.detach().numpy())
#     xress.append(xrestmp.detach().numpy())

i = 6
writer = SummaryWriter('runs/test'+str(i))
writer.add_graph(model_resnet50, images) #model_alexnet
writer.close()

model = model_resnet









activation = {}
model = model_squeezenet
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
# For RegNet
# h1 = model_regnet.trunk_output[0][0].activation.register_forward_hook(get_activation('l1'))
# h2 = model_regnet.trunk_output[1][0].activation.register_forward_hook(get_activation('l2'))
# h3 = model_regnet.trunk_output[2][0].activation.register_forward_hook(get_activation('l3'))

# For ResNet18
# h1 = model.layer1[1].relu.register_forward_hook(get_activation('l1'))
# h2 = model.layer2[1].relu.register_forward_hook(get_activation('l2'))
# h3 = model.layer3[1].relu.register_forward_hook(get_activation('l3'))

# For SqueezeNet
h1 = model_squeezenet.features[3].squeeze_activation.register_forward_hook(get_activation('l1'))
h2 = model_squeezenet.features[6].squeeze_activation.register_forward_hook(get_activation('l2'))
h3 = model_squeezenet.features[9].squeeze_activation.register_forward_hook(get_activation('l3'))

# forward pass -- getting the outputs
out = model(images)

# print(activation)

xtt1 = activation['l1'].detach().numpy()
xtt2 = activation['l2'].detach().numpy()
xtt3 = activation['l3'].detach().numpy()

ytt = xtt1[9, np.random.randint(low=0,high=16, size=10),...]

fig,ax = plt.subplots(3,3)
for i in range(9):
    ax[i//3, i%3].imshow(ytt[i,...])

plt.imshow(images[9,...].permute(1,2,0))
# detach the hooks
h1.remove()
h2.remove()
h3.remove()


xtt11 = activation['l3']
xtt11_mp = torch.flatten(F.max_pool2d(xtt11, 2), start_dim=-2, end_dim=-1)

xtt11_mp_np = xtt11_mp.detach().numpy()[9,...]
plt.imshow(xtt11_mp_np)


#L1 seems enough

# Give the container as an argument. 
# It's not really necessary if we go with the following procedure:
    # collect all samples in the batch dimension.
    # run the model only once on the batch.
    # you'll get a dictionary, with one key per layer.
    # for each layer, the activation for each sample will be collected in the first dimension of the value.
    
# def get_activation2(name, tmp_storage):
#     def hook(model, input, output):
#         tmp_storage[name] = output
#     return hook

# container1 = {}
# container2 = {}
# h12 = model.shared_encoder.feature_extractor[7].register_forward_hook(
#     get_activation2('l_shared', container1))
# h22 = model.dorsal.rnn.register_forward_hook(
#     get_activation2('l_dorsal', container2))