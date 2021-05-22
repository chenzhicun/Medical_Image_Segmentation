from model.unet_model import UNet


def get_model(name):
    if name=='unet':
        net = UNet(n_channels=1, n_classes=1, bilinear=True)
    else:
        raise NotImplementedError

    return net