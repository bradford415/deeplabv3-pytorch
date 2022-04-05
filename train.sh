#!/bin/bash

python main.py --train \
               --experiment bn_lr7e-3 \
               --backbone resnet101 \
               --dataset pascal \
               --epochs 50 \
               --batch_size 4 \
               --base_lr 0.007 \
               --crop_size 513 
