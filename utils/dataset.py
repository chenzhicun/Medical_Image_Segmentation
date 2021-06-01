from genericpath import isdir
from os.path import splitext
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2

from .argumentation import data_argument


class MIS_Dataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, argument=True, type='train', val_percent=0.1, in_channel=1) -> None:
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.argument = argument
        self.in_channel=in_channel

        self.ids = [splitext(file)[0] for file in listdir(
            imgs_dir) if not file.startswith('.')]
        if type == 'train':
            self.ids = self.ids[:int(len(self.ids)*(1-val_percent))]
        elif type == 'val':
            self.ids = self.ids[int(len(self.ids)*(1-val_percent)):]
        elif type == 'test':
            self.argument=False
            pass
        else:
            raise NotImplementedError
        logging.info(f'Creating {type} dataset with {len(self.ids)} examples with argument is {self.argument}')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def img_transform(cls, img):
        '''
        transform an image to ndarray.
        '''
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img/255
        img=1-img
        return img

    def __getitem__(self, index):
        idx = self.ids[index]
        mask_file = f'{self.masks_dir}/{idx}.png'
        img_file = f'{self.imgs_dir}/{idx}.png'
        if self.in_channel==1:
            mask = Image.open(mask_file)
            img = Image.open(img_file)
            if self.argument:
                img, mask = data_argument(img, mask)
            mask = self.img_transform(mask)
            img = self.img_transform(img)
            img = (img - 0.5052) / 0.1678
        if self.in_channel==3:
            mask = Image.open(mask_file)
            img = Image.open(img_file)
            if self.argument:
                img, mask = data_argument(img, mask)
            mask = self.img_transform(mask)
            img = np.array(img)
            img = 255 - img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img / 255
            
            _std = np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
            _mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))

            img = (img - _mean) / _std
            img = img.transpose((2,0,1))


        assert img.shape[1:] == mask.shape[1:], f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
