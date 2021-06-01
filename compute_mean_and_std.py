import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from config.model_config import get_args
from utils.dataset import MIS_Dataset
from utils.get_model import get_model

from torch.utils.data import DataLoader
# define the dir of dataset
dir_img = 'data/train_img/'
dir_mask = 'data/train_label/'


def compute_mean_and_std():
    mean, std= 0,0

    train_dataset = MIS_Dataset(
        dir_img, dir_mask, argument=False, type='train', val_percent=0.1)
    val_dataset = MIS_Dataset(
        dir_img, dir_mask, argument=False, type='val', val_percent=0.1)
    n_val = len(val_dataset)
    n_train = len(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    test_dataset = MIS_Dataset('data/test_img', 'data/test_label',argument=False,type='test')
    test_dataloader = DataLoader(test_dataset)
    print(len(train_loader) + len(val_loader) + len(test_dataloader))

    for batch in train_loader:
        img = batch['image']
        mean += img[:,0,:,:].mean()
        std += img[:,0,:,:].std()

    for batch in val_loader:
        img = batch['image']
        mean += img[:,0,:,:].mean()
        std += img[:,0,:,:].std()

    for batch in test_dataloader:
        img = batch['image']
        mean += img[:,0,:,:].mean()
        std += img[:,0,:,:].std()

    mean = mean / (len(train_loader) + len(val_loader) + len(test_dataloader))
    std = std / (len(train_loader) + len(val_loader) + len(test_dataloader))
    print(f'mean is {mean}\nstd is {std}')


if __name__=='__main__':
    compute_mean_and_std()