#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-25 14:58:24
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

from ..register import REGISTER
from ..loggers.std_logger import STDLogger as logger
from ..utils import tuple_or_list, get_valid_size
from ..config import CONFIG as cfg

import transforms as custom_transforms
import torchvision.transforms as transforms

def require_args():
    """all args for dataset objects
    
    Arguments:
        parser {argparse} -- current version of argparse object
    """

    known_args, _ = cfg.parse_known_args()

    # basic args for all datasets
    cfg.add_argument('--data-root', default=None, type=str,
                        help='root path to dataset')
    cfg.add_argument('--resize', default="256", type=str,
                        help='resize into. (default: 256)')
    cfg.add_argument('--size', default="224", type=str,
                        help='crop into. (default: 224)')
    cfg.add_argument('--scale', default=None, type=str,
                        help='scale for random resize crop. (default: None)')
    cfg.add_argument('--ratio', default="(0.75, 1.3333333333333)", type=str,
                        help='ratio for random resize crop. (default: (0.75, 1.333)')
    cfg.add_argument('--colorjitter', default=None, type=str,
                        help='color jitters for input. (default: None)')
    cfg.add_argument('--random-grayscale', default=0, type=float,
                        help='transform input to gray scale. (default: 0)')
    cfg.add_argument('--random-horizontal-flip', action='store_true',
                        help='random horizontally flip for input. (default: False)')

    # basic args for all dataloader
    cfg.add_argument('--batch-size', default=128, type=int,
                        help='batch size for input data. (default: 128)')
    cfg.add_argument('--workers-num', default=4, type=int,
                        help='number of workers being used to load data. (default: 4)')

    # get args for datasets
    if (REGISTER.is_package_registered(__name__) and
        REGISTER.is_class_registered(__name__, known_args.dataset)):
    
        dataset = get(known_args.dataset)
    
        if hasattr(dataset, 'require_args'):
            dataset.require_args()

def get(name):
    return REGISTER.get_class(__name__, name)

def get_transforms(stage='train', means=None, stds=None):

    transform = []

    stage = stage.lower()
    assert stage in ['train', 'test'], ('arg [stage]'
                                ' should be one of ["train", "test"]')
    resize = get_valid_size(cfg.resize)
    size = get_valid_size(cfg.size)
    if stage == 'train':
        # size transform
        scale = tuple_or_list(cfg.scale)
        if scale:
            ratio = tuple_or_list(cfg.ratio)
            logger.debug('Training samples will be random resized and crop '
                'with size %s, scale %s and ratio %s'
                % (size, scale, ratio))
            transform.append(custom_transforms.RandomResizedCrop(
                                size=size, scale=scale, ratio=ratio))
        else:
            logger.debug('Training samples will be resized to %s and then '
                'random cropped into %s' % (resize, size))
            transform.append(transforms.Resize(size=resize))
            transform.append(transforms.RandomCrop(size))
        # color jitter transform
        colorjitter = tuple_or_list(cfg.colorjitter)
        if colorjitter is not None:
            logger.debug('Training samples will use color jitter to enhance '
                'with args: %s' % (colorjitter,))
            transform.append(transforms.ColorJitter(*colorjitter))

        # gray scale
        if cfg.random_grayscale > 0:
            logger.debug('Training samples will be randomly convert to '
                'grayscale with probability %.2f' % cfg.random_grayscale)
            transform.append(transforms.RandomGrayscale(
                                            p=cfg.random_grayscale))

        # random horizontal flip
        if cfg.random_horizontal_flip:
            logger.debug('Training samples will be random horizontally flip')
            transform.append(transforms.RandomHorizontalFlip())

    else:
        logger.debug('Testing samples will be resized to %s and then center '
            'crop to %s' % (resize, size))
        transform.extend([transforms.Resize(resize),
                          transforms.CenterCrop(size)])

    # to tensor
    transform.append(transforms.ToTensor())

    # normalize
    means = tuple_or_list(means)
    stds = tuple_or_list(stds)
    if not (means is None or stds is None):
        logger.debug('Samples will be normalized with means: %s and stds: %s' %
            (means, stds))
        transform.append(transforms.Normalize(means, stds))
    else:
        logger.debug('Input images will not be normalized')

    return transforms.Compose(transform)

def register(name, cls):
    REGISTER.set_class(__name__, name, cls)









