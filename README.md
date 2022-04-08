# DeepLabv3 Implementation in PyTorch
Implementation of DeepLabv3 semantic segmentation model in PyTorch for ECE-8550 final project. The goal of this project is to implement and improve a research paper. A lot of code is referenced and used from [PyTorch's offical github page](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py) and an [independent approach from chenxi116](https://github.com/chenxi116/DeepLabv3.pytorch). While my version is very similar to other approaches, I have commented a majority of lines for my own understanding and for others who may be confused. Some comments could be inaccurate as I may have misunderstood some of the code but I believe a majority, if not all, are correct.

After understanding and implementing DeepLabv3, the model was trained and inferenced on the PASCAL VOC 2012 dataset augmented with SBD like in the DeepLabv3 paper. To further test and improve the paper, I plan to train and test the model on [RELLIS-3D off-road terrain dataset](https://unmannedlab.github.io/research/RELLIS-3D).

This project includes most aspects of the DeepLabv3 paper such as a ResNet backbone modified for DeepLab and atrous spatial pyramid pooling (ASPP).  Unfortunately, I was not able to implement the multi-grid due to confusion and time constraints for my project. The model still achieves an mIoU score of __73.90%__ on the augmented pascal voc 2012 dataset.

## Setup for a local machine
### Installation
Clone this github repository
```bash
git clone https://github.com/bradford415/deeplabv3-pytorch.git
```

### Anaconda environment
Create a virtual environment with the required dependencies.
```bash
conda create -n deeplabv3-pytorch python=3.7 pytorch torchvision numpy scipy pillow
```
### Traning the model
After preparing a dataset, to train the model run the ```train.sh``` script or run ```main.py``` and specify the command line arguments. A brief explanation of the command line arguments are at the top of ```main.py```. Instructions on preparing the augemented pascal dataset are written below.
```bash
bash train.sh
```
or
```bash
python main.py --train --experiment bn_lr7e-3 --backbone resnet101 --dataset pascal --epochs 50 --batch_size 4 --base_lr 0.007 --crop_size 513
```
A directory is created named after the command line arguments: backbone, experiment, dataset, and epochs. The trained model and checkpoints are stored here.

### Testing the model
Once you have a trained model, to test the model run the ```inference.sh``` script or remove the ```--train``` argument from the command line argument. The trained model that is loaded is named by the hyperparameters used to train it.
```
bash inference.sh
```
During inference, the epoch, iteration, and loss will be printed. When inference is finished, the segmented images will be saved, the IoU of each class and the final mIoU for the test set are printed out and saved to a text file. This is all stored in the ```data/<experiment>``` directory.
## Preparing augmented pascal voc 2012 dataset
