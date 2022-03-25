"""Main file to run this project

"""
import argparse
import os
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training iterations')
parser.add_argument('--batch_size', type=int, default=16, 
                    help='Number of samples to train at a time per iteration')

def main():
    # Finds best algorithm and can speed up runtime
    # Only use if your input size does not vary and 
    # your model stays the same for every interation
    torch.backends.cudnn.benchmark = True 
    
    # In this case, getattr() is calling a function from deeplab.py file to return the model
    # and the following parenthesis pass arguments to this 'resnet101' function
    # I am not sure the advantage over this rather than just calling the function itself
    # w/o getattr()
    if args.backbone == 'resnet101':
        model = getattr(deeplab, 'resnet101')(
        pretrained=(not args.scratch),
        num_classes=len(dataset.CLASSES),
        num_groups=args.groups,
        weight_std=args.weight_std,
        beta=args.beta)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

###################### start at line 79 on main.py on github
    if args.train:
        


if __name__ == '__main__':
    main()