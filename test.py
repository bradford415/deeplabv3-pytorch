import torch.utils.model_zoo as model_zoo
from torchvision import models
from scipy.io import loadmat
import numpy as np
from pathlib import Path

print('hey')
#pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')


#resnet = models.resnet101(pretrained=True)

#resnet

#print(resnet.state_dict())

#tester = [1 , 2, 3, 4, 5, 6]
#print(tester[3:])

#cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
#print(cmap)
#cmap = (cmap * 255).astype(np.uint8).flatten().tolist()
#print(cmap)

lister1 = [1, 2, 3, 4]
lister2 = ['a', 'b', 'c', 'd']

start_index = 0
for i in range(3):
    for start_index, (x, y) in enumerate(zip(lister1, lister2), start_index):
        print("{0} and {1} at index {2}".format(x, y, start_index))

#Path('test/test3/').mkdir(parents=True, exist_ok=True)


