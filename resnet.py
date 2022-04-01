"""ResNet class implementation but slightly modified for Deeplab.
Deeplabv3 modifies resnet so that block 4 uses atrous convolutions.
This file also only implements ResNet101.From this, ResNet50/101/152 can be
easily created. ResNet18/34 need an additional class called 'BuildingBlock'
or 'BasicBlock' as they use this instead of the 'BottleNeck' class.

ResNet Paper: https://arxiv.org/pdf/1512.03385.pdf
PyTorch ResNet w/ Dilation: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py 
"""
from os import defpath
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.module):

    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        """
        Args:
            C: number of in_channels
            depth: number of out_channels
        """
        super().init()
        self._C = C
        self._depth = depth
        self._num_classes=num_classes
        # ##### need to look for the upsampling part in the pytorch github to verify##### #
        # Global average pooling is used to help the problem of losing information
        # when atrous rates are large enough, close to the size of the feature map.
        # This is explained in the deeplabv3 paper section 3.3. This is  performed 
        # in the x5 step
        self.global_pooling = nn.AdaptiveAvgPooling2d(1) # Global pooling, output size 1
        self.relu = nn.ReLU(inplace=True)
        # Defining convolutions with atrous rates [6, 12, 18]
        self.aspp1 = conv(C,depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                          dilation=int(6*mult), padding=int(6*mult), bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                          dilation=int(12*mult), padding=int(12*mult), bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                          dilation=int(18*mult), padding=int(12*mult), bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(depth, momentum)
        self.conv2 = conv(depth*5, depth, kernel_size=1, stride=1, bias=False)
        self.bn2 = norm(depth, momentum)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)
    
    def forward(self, x):
        # perform 5 different convolutions, each using the orignal data x
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        # x5 is used to incorporate global context information
        # AdapativeAvgPooling(1) returns only 1 pixel from each feature map 
        # so they need to be upsampled to the original feature map size of x
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        # Upsampling to orignal feature map (x) size
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', 
                         align_corners=True)(x5) # shape[2]=height, shape[3]=width
        # Concatenate feature maps from each convolution.
        # This will append feature maps on the 1st dimension which effectively 'stacks'
        # them like normal Conv2d with multiple output channels.
        # The feature maps are of size [batch_size, num_channels, height, width]
        # so dim=1 will append the feature maps to the num_channels dimension
        x = torch.cat((x1, x2, x3, x4 ,x5), dim=1) 
        # Fuse concatenated feature maps using 1x1 convolutions
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # General the final logits using 1x1 convolution
        x = self.conv3(x)
        
        return x



class BottleNeck():
    """Modified BuildingBlock for deeper ResNets (50, 101, 152).
    Each BottleNeck has 3 layers instead of 2. The 3 layers are 1x1, 3x3, and 1x1 convolutions.
    The 1x1 layers are responsible for reducing and then increasing (retoring) dimensions
    
    """

    # Value to multiply the output channels by in the last 1x1 BottleNeck layer (Fig. 5 & Table 1)
    expansion = 4  


class ResNet(nn.module):
    
    def __init__(self, block, layers, num_classes, replace_stride_with_dilation=None):
        super().init()
        self.inplanes = 64 # PyTorch refers to planes as the number of channels for some reason
        self.dilation = 1
        self.conv = nn.Conv2d # Mostly use for 1x1 convolutions
        self.norm_layer = nn.BatchNorm2d

        if replace_stride_with_dilation is None:
            # each element in the list indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            # for deeplab you want the third element to be true,
            # for the normal resnet without dilation they all should be false
            # I think. This should be set when calling ResNet#########################################################
            replace_stride_with_dilation = [False, True, True]

        # No bias because batch normalization, next layer, takes care of it
        # Padding and stride values set of the first conv layer set by ResNet paper
        # This first layer downsamples the input by 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False) # [(W−K+2P)/S]+1 = Z
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # inplace can save a small amount of memory by not creating a new object
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
        #                       padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0]) # defining conv blocks after the first conv layer
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]) # start downsampling here => stride=2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
         # Apply dilation only for block 4. If replace_stride_with_dilation[i]=True,
         # _make_layer() will set to stride to 1 and dilation will equal the stride argument value, this case 2
        self.aspp = ASPP(512*block.expansion, 256, num_classes, conv=self.conv, norm=self.norm)

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

