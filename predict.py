import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from os.path import splitext
from os import listdir
from tqdm import tqdm

from model.unet_model import UNet
from utils.dataset import MIS_Dataset
from eval import eval_net
from utils.get_model import get_model


def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(MIS_Dataset.img_transform(full_img))
    img = (img - 0.5052) / 0.1678
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(full_img.size[1]),
        #         transforms.ToTensor()
        #     ]
        # )

        # probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--saved_model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input_dir', '-i', type=str,
                        help='dirs of input images', required=True)
    parser.add_argument('--output_dir', '-o', type=str,
                        help='dirs of ouput images')
    parser.add_argument('--mask_dir', '-d', type=str, help='dirs of gt mask')
    parser.add_argument('--mask_threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--model_name', type=str, default='unet', help='model name.')
    parser.add_argument('--in_channel', type=int, default=1, help='in channels of image.')

    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((255 - mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_dir = args.input_dir
    try:
        os.mkdir(f'{args.output_dir}/{args.model_name}')
    except:
        pass
    out_dir = f'{args.output_dir}/{args.model_name}'

    img_ids = [splitext(file)[0] for file in listdir(in_dir) if not file.startswith('.')]

    net = get_model(args.model_name)

    logging.info("Loading model from {}".format(args.saved_model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.saved_model, map_location=device))

    logging.info("Model loaded !")

    for i in tqdm(range(len(img_ids))):
        img_name = f'{in_dir}/{img_ids[i]}.png'
        logging.info("\nPredicting image {} ...".format(img_name))

        img = Image.open(img_name)

        mask = predict_img(net=net,
                           full_img=img,
                           out_threshold=args.mask_threshold,
                           device=device)

        result = mask_to_image(mask)
        result.save(f'{out_dir}/{img_ids[i]}_output.png')

        logging.info("Mask saved to {}".format(f'{out_dir}/{img_ids[i]}_output.png'))

    test_dataset = MIS_Dataset(in_dir, args.mask_dir, argument=False, type='test', in_channel=args.in_channel)
    test_dataloader = DataLoader(test_dataset)
    dice_score = eval_net(net, test_dataloader, device, args.mask_threshold, criterion='dice')
    print(f'The dice score on test dataset is {dice_score}.')
    iou_score = eval_net(net, test_dataloader, device, args.mask_threshold, criterion='iou')
    print(f'The IOU score on test dataset is {iou_score}.')
    acc_score = eval_net(net, test_dataloader, device, args.mask_threshold, criterion='acc')
    print(f'The accuracy score on test dataset is {acc_score}.')
    # Computing vrand and Vinfo costs a long time to compute.
    vrand_score, vinfo_score = eval_net(net, test_dataloader, device, args.mask_threshold, criterion='vrand&vinfo')
    print(f'The V^rand score on test dataset is {vrand_score}.')
    print(f'The V^info score on test dataset is {vinfo_score}.')
