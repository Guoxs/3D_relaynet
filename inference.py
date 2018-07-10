import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torch.autograd import Variable

data01 = sio.loadmat('./datasets/2015_BOE_Chiu/Subject_01.mat')

images = data01['images']
automaticFluidDME = data01['automaticFluidDME']
automaticLayersDME = data01['automaticLayersDME']
automaticLayersNormal = data01['automaticLayersNormal']
manualFluid1 = data01['manualFluid1']
manualFluid2 = data01['manualFluid2']
manualLayers1 = data01['manualLayers1']
manualLayers2 = data01['manualLayers2']

SEG_LABELS_LIST = [
    {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    {"id": 0, "name": "Region above the retina (RaR)", "rgb_values": [128, 0, 0]},
    {"id": 1, "name": "ILM: Inner limiting membrane", "rgb_values": [0, 128, 0]},
    {"id": 2, "name": "NFL-IPL: Nerve fiber ending to Inner plexiform layer", "rgb_values": [128, 128, 0]},
    {"id": 3, "name": "INL: Inner Nuclear layer", "rgb_values": [0, 0, 128]},
    {"id": 4, "name": "OPL: Outer plexiform layer", "rgb_values": [128, 0, 128]},
    {"id": 5, "name": "ONL-ISM: Outer Nuclear layer to Inner segment myeloid", "rgb_values": [0, 128, 128]},
    {"id": 6, "name": "ISE: Inner segment ellipsoid", "rgb_values": [128, 128, 128]},
    {"id": 7, "name": "OS-RPE: Outer segment to Retinal pigment epithelium", "rgb_values": [64, 0, 0]},
    {"id": 8, "name": "Region below RPE (RbR)", "rgb_values": [192, 0, 0]}]


# {"id": 9, "name": "Fluid region", "rgb_values": [64, 128, 0]}];

def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1, 2, 0)
    for l in label_infos:
        mask =   label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


relaynet_model =  torch.load('models/relaynet_good.model')
input_data = np.array(images[:,:,30],dtype=np.float).T
input_data = input_data.reshape((1,1,768,496))


out = relaynet_model(Variable(torch.Tensor(input_data).cuda(),volatile=True))
out = F.softmax(out)
max_val, idx = torch.max(out,1)
idx = idx.data.cpu().numpy()
idx = label_img_to_rgb(idx)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.imshow(idx)

img_test = input_data
img_test = np.squeeze(img_test)

ax2 = fig.add_subplot(221)
ax2.imshow(img_test,cmap=plt.cm.gray)
plt.show()
