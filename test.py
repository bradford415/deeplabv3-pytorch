import torch.utils.model_zoo as model_zoo
from torchvision import models
from scipy.io import loadmat
import numpy as np

print('hey')
#pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')


#resnet = models.resnet101(pretrained=True)

#resnet

#print(resnet.state_dict())

#tester = [1 , 2, 3, 4, 5, 6]
#print(tester[3:])

cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
print(cmap)
cmap = (cmap * 255).astype(np.uint8).flatten().tolist()
print(cmap)
