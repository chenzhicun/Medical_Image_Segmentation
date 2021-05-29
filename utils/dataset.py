from genericpath import isdir
from os.path import splitext
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

from .argumentation import data_argument


class MIS_Dataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, argument=True, type='train', val_percent=0.1) -> None:
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.argument = argument

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

    # @classmethod
    # def preprocess(cls, pil_img, scale):
    #     w, h = pil_img.size
    #     new_w, new_h = int(scale*w), int(scale*h)
    #     assert new_h > 0 and new_w > 0, 'new image size is too small!'
    #     pil_img = pil_img.resize((new_w, new_h))

    #     img_nd = np.array(pil_img)

    #     if len(img_nd.shape) == 2:
    #         img_nd = np.expand_dims(img_nd, axis=2)

    #     # H x W x C => C x H x W
    #     img_trans = img_nd.transpose((2, 0, 1))
    #     if img_trans.max() > 1:
    #         img_trans = img_trans/255

    #     return img_trans

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

        mask = Image.open(mask_file)
        img = Image.open(img_file)
        # if self.argument:
        #     img, mask = data_argument(img, mask)
        mask = self.img_transform(mask)
        img = self.img_transform(img)

        assert img.size == mask.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
