from model.unet_model import UNet
from model.CEnet import CE_Net_OCT
from model.DCUnet import DCUNet
from model.MultiresUnet import MultiResUnet
from model.NestedUnet import NestedUNet
from model.AttentionUnet import AttU_Net
import logging


def get_model(name):
    if name=='unet':
        net = UNet(n_channels=1, n_classes=1, bilinear=True)
        logging.info(f'Network:Unet\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    elif name =='cenet_1_channel':
        net = CE_Net_OCT(num_classes=1,num_channels=1)
        logging.info(f'Network: CEnet_1_channel\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)\n')
    elif name =='cenet_3_channel':
        net = CE_Net_OCT(num_classes=1,num_channels=3)
        logging.info(f'Network: CEnet_3_channel\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)\n')
    elif name == 'dcunet':
        net = DCUNet(num_classes=1,num_channels=1)
        logging.info(f'Network: DCUnet\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)\n')
    elif name =='multiresunet':
        net = MultiResUnet(num_classes=1,num_channels=1)
        logging.info(f'Network: MultiResUNet\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)\n')
    elif name =='unet++':
        net = NestedUNet(num_classes=1,num_channels=1)
        logging.info(f'Network: Unet++\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)\n')
    elif name =='att_unet':
        net = AttU_Net(num_classes=1,num_channels=1)
        logging.info(f'Network: attention unet\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)\n')
    else:
        raise NotImplementedError

    return net