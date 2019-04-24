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
    logger.debug('Going to use [RMSprop] optimizer for training with alpha %.2f, '
        'eps %f, weight decay %f, momentum %f %s centered' % (cfg.alpha,
            cfg.eps, cfg.weight_decay, cfg.momentum,
            ('with' if cfg.centered else 'without')))
    return torch.optim.RMSprop(params, lr=cfg.base_lr,
            alpha=cfg.alpha, eps=cfg.eps, momentum=cfg.eps,
            centered=cfg.centered, weight_decay=cfg.weight_decay)

def require_args():
        
    cfg.add_argument('--alpha', default=0.99, type=float,
            help='smoothing constant')
    cfg.add_argument('--eps', default=1e-8, type=float,
            help=('term added to the denominator to improve'
                  ' numerical stability'))
    cfg.add_argument('--momentum', default=0, type=float,
            help='momentum factor')
    cfg.add_argument('--centered', action='store_true',
            help='whether to compute the centered RMSProp')

from ..register import REGISTER
REGISTER.set_class(REGISTER.get_package_name(__name__), 'rmsprop', __name__)