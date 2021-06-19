from PIL import Image
import numpy as np
from imgaug import augmenters as iaa


def data_argument(img, gt_mask):
    img = np.expand_dims(img, axis=2)
    gt_mask = np.expand_dims(gt_mask, axis=2)
    img = np.expand_dims(img, axis=0)
    gt_mask = np.expand_dims(gt_mask, axis=0)
    seq = iaa.Sequential([
        iaa.Affine(scale=(0.8, 1.2), shear=(-30, 30), rotate=(-45, 45), translate_percent=(0, 0.1)),
        iaa.flip.Fliplr(p=0.5),
        iaa.flip.Flipud(p=0.5)
    ], random_order=True)
    img, gt_mask = seq(images=img, segmentation_maps=gt_mask)

    return img[0, :, :, 0], gt_mask[0, :, :, 0]
