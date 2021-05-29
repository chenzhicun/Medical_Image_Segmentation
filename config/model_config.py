import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',type=str,default='unet',help='model name.')
    parser.add_argument('--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('--batch_size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('--learning_rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('--validation_percent', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--exp_id', dest='exp_id', type=str, default=None,
                        help='ID of experiment.')
    parser.add_argument('--mask_threshold', dest='threshold', type=float,
                            default=0.5, help='the probability threshold of mask')
    parser.add_argument('--argument', action='store_true',
                            help='Whether do argument on train dataset.')

    return parser.parse_args()
