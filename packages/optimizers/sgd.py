#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-26 19:42:22
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

import sys

import torch

from ..loggers.std_logger import STDLogger as logger
from ..config import CONFIG as cfg

def get(params):
    logger.debug('Going to use [SGD] optimizer for training with momentum %.2f, '
        'dampening %f, weight decay %f %s nesterov' % (cfg.momentum,
            cfg.dampening, cfg.weight_decay,
            ('with' if cfg.nesterov else 'without')))
    return torch.optim.SGD(params, lr=cfg.base_lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay, nesterov=cfg.nesterov,
            dampening=cfg.dampening)

def require_args():
        
    cfg.add_argument('--momentum', default=0, type=float,
                        help='momentum factor')
    cfg.add_argument('--dampening', default=0, type=float,
                        help='dampening for momentum')
    cfg.add_argument('--nesterov', action='store_true',
                        help='enables Nesterov momentum')
    
from ..register import REGISTER
REGISTER.set_class(REGISTER.get_package_name(__name__), 'sgd', __name__)