"""Main file to run this project

"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training iterations')
parser.add_argument('--batch_size', type=int, default=16, 
                    help='Number of samples to train at a time per iteration')
args = parser.parse_args()

def main():
    # cudnn.benchmark finds best algorithm and can speed up runtime
    # Only use if your input size does not vary and 
    # your model stays the same for every interation
    torch.backends.cudnn.benchmark = True 
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    
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

    """Notes:
    - ignore_index ignores the 255 value bc indices go from 0-254, I think.
      Or it is used for the background class which you will ignore
    - DataParallel splits input across devices. During forward pass the model is 
      replicated on each device and each device handles a portion of the input.
    - .cuda(), the same as, .to(device) used to put models/tensors on gpu 
      .to(device) is more flexible and should probably be used more
    """
    if args.train:
        criterion = nn.CrossEntropyLoss(ignore_index=255) 
        model = nn.DataParallel(model)
        model = model.to(device)

        backbone_params = (
            list(model.module.conv1.parameters()) +
            list(model.module.bn1.parameters()) +
            list(model.module.layer1.parameters()) +
            list(model.module.layer2.parameters()) +
            list(model.module.layer3.parameters()) +
            list(model.module.layer4.parameters())
        )

        # create a list of dictionaries to store the backbone and other layer parameters
        ###########################3 look into deep lab file and how it gets the backbone_params###################33
        params_to_optimize = [
            {'params': filter(lambda p: p.requires_grad, backbone_params)},
            {'params': filter(lambda p: p.requires_grad, last_params)}
        ]
        optimizer = optim.SGD(params_to_optimize, lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
        
        start_epoch = 0 # defined if loading a checkpoint and need to start a specific epoch 
        for epoch in range(start_epoch, args.epochs):
            model.train()
            for index, (data, target) in enumerate(dataset_loader):



if __name__ == '__main__':
    main()