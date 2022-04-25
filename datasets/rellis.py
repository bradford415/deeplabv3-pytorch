"""Dataset file for the Rellis-3D. Used to preprocess the data and 
prepare it for the data loader. The Rellis dataset can be found here:
https://unmannedlab.github.io/research/RELLIS-3D

Notes:

"""
import torch.utils.data as Dataset
import os
import PIL import Image
from utils import preprocess


class Rellis3D(Dataset):
    CLASSES = [
        'void', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object',
        'asphalt', 'building', 'log', 'person', 'fence', 'bush', 'concrete',
        'barrier', 'puddle', 'mud', 'rubble'
    ]

    def __init__(self, root, train=True, crop_size=None):
        self.root = root
        self.train = train
        self.images = []
        self.masks = []
        self.rgb_img_train = 
        self.id_img_train = 
        self.rgb_img_test = os.path.join(self.root, 'test', 'rgb')
        
        #######################################################Ask max how he handled the miou for test set if he used all images for training
        
        if self.train:
            img_path = os.path.join(self.root, 'train', 'rgb')
            label_path = os.path.join(self.root, 'train', 'id')
        else:
            img_path = os.path.join(self.root, 'train', 'id')

        # Used to map the current id labels (masks) to the right class.
        # Function is used to transform the mask
        self.label_mapping = {0: 0,
                              1: 0,
                              3: 1,
                              4: 2,
                              5: 3,
                              6: 4,
                              7: 5,
                              8: 6,
                              9: 7,
                              10: 8,
                              12: 9,
                              15: 10,
                              17: 11,
                              18: 12,
                              19: 13,
                              23: 14,
                              27: 15,
                              29: 1,
                              30: 1,
                              31: 16,
                              32: 4,
                              33: 17,
                              34: 18}

    def convert_label(self, label, inverse=False):
        """Transform mask labels to class values 0-34 => 0-18
           
        Args
            label: The ground truth image mask to be transformed
            inverse: Bool variable to swap the label_mapping to transform image
                     from class ids to original label values (currently not used)
        """
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp==k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp==k] = v
        return label

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.masks[index])

        _img, _target = preprocess(_img,
                                _target,
                                flip=True if self.train else False,
                                scale=(0.5, 2.0) if self.train else None,
                                crop=(self.crop_size, self.crop_size))

        return _img, _target

    def __len__(self):
        return len(self.images)




####### Code for converting the labels is here i think: https://github.com/unmannedlab/RELLIS-3D/blob/main/benchmarks/GSCNN-master/datasets/rellis.py#L155