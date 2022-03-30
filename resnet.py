"""ResNet class implementation but slightly modified for Deeplab.
Most of the modifications are for the dilation hyperparameters

ResNet Paper: https://arxiv.org/pdf/1512.03385.pdf
PyTorch ResNet w/ Dilation: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.module):
    
    def __init__(self, block, layers, num_classes):
        super().init()
        self.inplanes = 64 # PyTorch refers to planes as the number of channels for some reason
        self.dilation = 1

        # No bias because batch normalization, next layer, takes care of it
        # Padding and stride values set of the first conv layer set by ResNet paper
        # This first layer downsamples the input by 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False) # [(W−K+2P)/S]+1 = Z
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # inplace can save a small amount of memory by not creating a new object
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0]) # defining conv blocks after the first conv layer
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2) 
        self.layer1 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer1 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=2)

    def _make_layer(self, block, planes, num_blocks, stride, dilation=False):
        """Create convolutional blocks w/ hyperparameters specified by resnet.
        These conv blocks start after the first conv and max pooling layer.

        Args:
            block: The type of block to add, in this case it is the 'BottleNeck' class (#######read bottleneck class and make sure this is the case)
            planes: Refers to the number of channels (I'm not sure why they call them planes)
            num_blocks: The number of blocks to add per 'building', this will depend on the
                        type of resnetxxx (ex. 'resnet101' uses [3, 4, 23, 3]). these values
                        can be found in Table 1 in the resnet paper.
        """
        #############################################Start here, add dilation conditions and understand bottle necks
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes *block.expansion
        return nn.sequential(*layers) 
        # * operator expands the list into positional arguments to create a model in 1 line
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)