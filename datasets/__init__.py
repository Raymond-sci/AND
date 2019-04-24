#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-28 12:34:35
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

from .cifar import CIFAR10Instance, CIFAR100Instance
from .svhn import SVHNInstance

import torch

from packages import datasets as cmd_datasets
from packages.config import CONFIG as cfg

__all__ = ('CIFAR10Instance', 'CIFAR100Instance', 'SVHNInstance')

def get(name, instant=False):
    """
    Get dataset instance according to the dataset string and dataroot
    """
    
    # get dataset class
    dataset_cls = cmd_datasets.get(name)

    if not instant:
        return dataset_cls

    # get transforms for training set
    transform_train = cmd_datasets.get_transforms('train', cfg.means, cfg.stds)

    # get transforms for test set
    transform_test = cmd_datasets.get_transforms('test', cfg.means, cfg.stds)

    # get trainset and trainloader
    trainset = dataset_cls(root=cfg.data_root, train=True, download=True,
                            transform=transform_train)
    # filter trainset if necessary
    trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.workers_num)

    # get testset and testloader
    testset = dataset_cls(root=cfg.data_root, train=False, download=True,
                            transform=transform_test)
    # filter testset if necessary
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                        shuffle=False, num_workers=cfg.workers_num)

    return trainset, trainloader, testset, testloader


cmd_datasets.register('cifar10', CIFAR10Instance)
cmd_datasets.register('cifar100', CIFAR100Instance)
cmd_datasets.register('svhn', SVHNInstance)



