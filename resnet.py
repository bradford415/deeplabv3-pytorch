"""ResNet class implementation but slightly modified for Deeplab.
Deeplabv3 modifies resnet so that block 4 uses atrous convolutions.
This file also only implements ResNet101.From this, ResNet50/101/152 can be
easily created. ResNet18/34 need an additional class called 'BuildingBlock'
or 'BasicBlock' as they use this instead of the 'BottleNeck' class.

ResNet Paper: https://arxiv.org/pdf/1512.03385.pdf
PyTorch ResNet w/ Dilation: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck():
    """Modified BuildingBlock for deeper ResNets (50, 101, 152).
    Each BottleNeck has 3 layers instead of 2. The 3 layers are 1x1, 3x3, and 1x1 convolutions.
    The 1x1 layers are responsible for reducing and then increasing (retoring) dimensions
    
    """"

    # Value to multiply the output channels by in the last 1x1 BottleNeck layer (Fig. 5 & Table 1)
    expansion = 4  


class ResNet(nn.module):
    
    def __init__(self, block, layers, num_classes, replace_stride_with_dilation=None):
        super().init()
        self.inplanes = 64 # PyTorch refers to planes as the number of channels for some reason
        self.dilation = 1
        self.conv = nn.Conv2d # Mostly use for 1x1 convolutions
        self.norm_layer = nn.BatchNorm2d

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
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]) # start downsampling here => stride=2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
         # Apply dilation only for block 4. If replace_stride_with_dilation[i]=True,
         # _make_layer() will set to stride to 1 and dilation will equal the stride argument value, this case 2

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Create convolutional blocks w/ hyperparameters specified by resnet.
        These conv blocks start after the first conv and max pooling layer.

        Args:
            block: The type of block to add, ResNet-18/34 uses 'BuildingBlock', ResNet-50/101/152 
                   uses 'BottleNeck' block
            planes: Refers to the number of channels, (I'm not sure why they call them planes).
                    Inplanes is the number of input channels to the conv layer I think.
            num_blocks: The number of blocks to add per 'building', this will depend on the
                        type of resnetxxx (ex. 'resnet101' uses [3, 4, 23, 3]). these values
                        can be found in Table 1 in the resnet paper.
        """
        
        downsample = None
        previous_dilation = self.dilation
        # if using atrous convolutions, swap dilation with stride value and set
        if dilate:
            self.dilation *= stride  
            stride = 1
        # If you are adding the last layer in the BottleNeck (1x1 conv)
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes*block.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion)
            )   
        
        """From the original ResNet paper, after the block of 64 kernels, starting at 128, 
        the first block will always downsample after the 1x1 convolution, every block after 
        will not. We start the  for loop at 1 to skip this first downsampling layer we 
        already appended. According to a new paper, https://arxiv.org/abs/1512.03385, 
        downsampling, w/ stridee > 1, in the BottleNeck during every 3x3 convolution improves 
        accuracy and is what the PyTorch github implements. This variant is called ResNet V1.5
        Similarily, they downsample in the 'BasicBlock' on every first 3x3 conv layer, but not the second
        """
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, previous_dilation, self.conv, self.norm_layer))
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes *block.expansion
        return nn.sequential(*layers) 
        # * operator expands the list into positional arguments to create a model in 1 line
    
    #############################Finish impelementing _make_layer, forward function and bottleneck class. Do BasicBlock class if have time 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)

