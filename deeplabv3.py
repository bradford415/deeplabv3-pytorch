"""Classes and functions for the deeplabv3 model and backbones.

"""
import torch
import torch.nn as nn
import torch.tuils.model_zoo as model_zoo # Importing pytorch pre-trained models
from torch.nn import functional as F

# List of backbones
# Double underscore is just naming convention, 
# allows you to use all as a variable and ignores the 'all' keyword/function
__all___ = ['resnet101'] 

model_urls = {

}


        