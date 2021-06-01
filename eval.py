from numpy import dtype
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from utils.criterion import dice_coeff, IOU, vrand_vinfo, accuracy
from sklearn.metrics import roc_curve


def search_threshold(predict, gt):
    '''
    search the best output threshold.
    '''
    predict, gt = deepcopy(predict), deepcopy(gt)
    try:
        predict = predict.cpu().numpy()
        gt = gt.cpu().numpy()
    except:
        pass
    predict = predict.reshape(-1)
    gt = gt.reshape(-1)

    fpr, tpr, threshold = roc_curve(gt, predict)
    y = tpr - fpr
    optimal_index = np.argmax(y)
    optimal_threshold = threshold[optimal_index]

    return optimal_threshold



def eval_net(net, loader, device, threshold=0.5, criterion='dice'):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    num_val = len(loader)
    tot, tot_vrand, tot_vinfo = 0, 0, 0
    criterion_dict = {'dice': dice_coeff,
                      'iou': IOU,
                      'vrand&vinfo':vrand_vinfo,
                      'acc': accuracy}
    try:
        criterion = criterion_dict[criterion]
    except:
        raise NotImplementedError

    with tqdm(total=num_val, desc="validation round", unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, gt_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            gt_masks = gt_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                pred_masks = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(pred_masks, gt_masks).item()
            else:
                pred = torch.sigmoid(pred_masks)
                # threshold = search_threshold(pred, gt_masks)
                # print(threshold)
                # x=input()
                if criterion==vrand_vinfo:
                    pred = (pred > threshold)
                    gt_masks = gt_masks.to(device=device, dtype=torch.bool)
                    tmp_vrand, tmp_vinfo = vrand_vinfo(pred[0,0,:,:],gt_masks[0,0,:,:])
                    tot_vrand += tmp_vrand.item()
                    tot_vinfo += tmp_vinfo.item()
                else:
                    pred = (pred > threshold).float()
                    tot += criterion(pred, gt_masks).item()
            pbar.update()

    net.train()
    if criterion==vrand_vinfo:
        return tot_vrand / num_val, tot_vinfo / num_val
    return tot / num_val
