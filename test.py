import torch.utils.model_zoo as model_zoo
from torchvision import models


print('hey')
#pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')


resnet = models.resnet101(pretrained=True)

resnet

print(resnet.state_dict())

