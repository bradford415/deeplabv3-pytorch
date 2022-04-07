# DeepLabv3 Implementation in PyTorch
Implementation of DeepLabv3 semantic segmentation model in PyTorch for ECE-8550 final project. The goal of this project is to implement and improve a research paper. A lot of code is referenced and used from [PyTorch's offical github page](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py) and an [independent approach from chenxi116](https://github.com/chenxi116/DeepLabv3.pytorch). While my version is very similar to other approaches, I have commented a majority of lines for my own understanding and for others who may be confused. Some comments could be inaccurate as I may have misunderstood some of the code but I believe a majority, if not all, are correct.

After understanding and implementing DeepLabv3, the model was trained and inferenced on the PASCAL VOC 2012 dataset augmented with SBD like in the DeepLabv3 paper. To further test and improve the paper, I trained and tested the model on [RELLIS-3D off-road terrain dataset](https://unmannedlab.github.io/research/RELLIS-3D).

This project includes most aspects of the DeepLabv3 paper such as a ResNet backbone modified for DeepLab and atrous spatial pyramid pooling (ASPP).  Unfortunately, I was not able to implement the multi-grid due to confusion and time constraints for my project. The model still achieves an mIoU score of __73.90%__ on the augmented pascal voc 2012 dataset.

## Setup for a local machine
### Installation
Clone this github repository
```bash
git clone https://github.com/bradford415/deeplabv3-pytorch.git
```
### Anaconda environment
