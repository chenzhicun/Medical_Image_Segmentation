from numpy import dtype
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.criterion import dice_coeff, IOU


def eval_net(net, loader, device, threshold=0.5, criterion='dice'):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    num_val = len(loader)
    tot = 0
    criterion_dict = {'dice': dice_coeff,
                      'iou': IOU}
    criterion = criterion_dict[criterion]

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
                pred = (pred > threshold).float()
                tot += criterion(pred, gt_masks).item()
            pbar.update()

    net.train()
    return tot / num_val
