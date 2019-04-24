#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-14 19:31:42
# @Author  : Jiabo (Raymond) Huang (jiabo.huang@qmul.ac.uk)
# @Link    : https://github.com/Raymond-sci
from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np

from packages.config import CONFIG as cfg

class SVHNInstance(datasets.SVHN):
    """SVHNInstance Dataset.
    """
    
    @staticmethod
    def require_args():
        cfg.add_argument('--means', default='(0.4377, 0.4438, 0.4728)',
                        type=str, help='channel-wise means')
        cfg.add_argument('--stds', default='(0.1201, 0.1231, 0.1052)',
                        type=str, help='channel-wise stds')

    def __init__(self, root, train=True, transform=None, target_trainsform=None, download=False):
        self.train = train
        super(SVHNInstance, self).__init__(root, split=('train' if train else 'test'),
            transform=transform, target_transform=target_trainsform, download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index