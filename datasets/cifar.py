from __future__ import print_function

import os

from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data

from packages.config import CONFIG as cfg

class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """

    @staticmethod
    def require_args():
        cfg.add_argument('--means', default='(0.4914, 0.4822, 0.4465)',
                        type=str, help='channel-wise means')
        cfg.add_argument('--stds', default='(0.2023, 0.1994, 0.2010)',
                        type=str, help='channel-wise stds')

    def __init__(self, *args, **kwargs):
        super(CIFAR10Instance, self).__init__(*args, **kwargs)
        self.labels = self.targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class CIFAR100Instance(CIFAR10Instance):
    """CIFAR100Instance Dataset.

    This is a subclass of the `CIFAR10Instance` Dataset.
    """

    @staticmethod
    def args(parser):
        parser.add_argument('--means', default='(0.5071, 0.4866, 0.4409)',
                        type=str, help='channel-wise means')
        parser.add_argument('--stds', default='(0.2009, 0.1984, 0.2023)',
                        type=str, help='channel-wise stds')
        return parser

    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
