#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-03-13 21:25:39
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

from .resnet_cifar import *

from packages.register import REGISTER
from packages.config import CONFIG as cfg
from packages import networks as cmd_networks

def require_args():

    cfg.add_argument('--low-dim', default=128, type=int, help='feature dimension')

def get(name, instant=False):
    cls = cmd_networks.get(name)
    if not instant:
        return cls
    return cls(low_dim=cfg.low_dim)

REGISTER.set_package(__name__)
cmd_networks.register('ResNet18', ResNet18)
cmd_networks.register('ResNet50', ResNet50)
cmd_networks.register('ResNet101', ResNet101)
