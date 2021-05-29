import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from model.unet_model import UNet
from config.model_config import get_args
from utils.dataset import MIS_Dataset
from utils.get_model import get_model

from torch.utils.data import DataLoader
# define the dir of dataset
dir_img = 'data/train_img/'
dir_mask = 'data/train_label/'


def print_model_info(name):
    '''
    print some key information of model.
    '''
    if name=='unet':
        logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    else:
        raise NotImplementedError


def train_net(net,
              device,
              epochs=50,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              threshold=0.5,
              argument=True):
    # generate the train dataloader and validation dataloader.
    train_dataset = MIS_Dataset(
        dir_img, dir_mask, argument=argument, type='train', val_percent=val_percent)
    val_dataset = MIS_Dataset(
        dir_img, dir_mask, argument=False, type='val', val_percent=val_percent)
    n_val = len(val_dataset)
    n_train = len(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    global_step = 0
    # print some key information for training.
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Mask Threshold:  {threshold}
        Argument:        {argument}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_score = 0

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

        val_score = eval_net(net, val_loader, device, threshold)
        scheduler.step(val_score)
        logging.info('Current lr:{}'.format(
            optimizer.state_dict()['param_groups'][0]['lr']))

        if net.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_score))
        else:
            logging.info('Validation Dice Coeff: {}'.format(val_score))

        if val_score > best_score:
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_best.pth')
            logging.info(f'Best checkpoint {epoch} saved!')
            best_score = val_score
            best_epoch = epoch

        if save_cp:
            if epoch % 5 == 0:
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch}.pth')
                logging.info(f'Checkpoint {epoch} saved !')

    torch.save(net.state_dict(),
               dir_checkpoint + f'CP_final.pth')
    logging.info(f'Final checkpoint saved !')
    logging.info(f'Best score on validation set is {best_score}, during epoch {best_epoch}.')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    dir_checkpoint = f'checkpoints/{args.model}_{args.exp_id}/'
    try:
        os.mkdir(dir_checkpoint)
        logging.info('Created checkpoint directory')
    except:
        pass

    net=get_model(args.model)
    print_model_info(args.model)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  threshold=args.threshold,
                  argument=args.argument)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
