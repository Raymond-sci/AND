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
    logger.debug('Going to use [Adam] optimizer for training with betas %s, '
        'eps %f, weight decay %f %s amsgrad' % ((cfg.beta1, cfg.beta2),
            cfg.eps, cfg.weight_decay,
            ('with' if cfg.amsgrad else 'without')))
    return torch.optim.Adam(params, lr=cfg.base_lr,
            betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
            weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)

def require_args():
        
    cfg.add_argument('--beta1', default=0.9, type=float,
            help=('coefficients used for computing running'
                  ' averages of gradient'))
    cfg.add_argument('--beta2', default=0.999, type=float,
            help=('coefficients used for computing running'
                  ' averages of gradient\'s square'))
    cfg.add_argument('--eps', default=1e-8, type=float,
            help='term added to the denominator to improve numerical stability')
    cfg.add_argument('--amsgrad', action='store_true',
            help='whether to use the AMSGrad variant of this algorithm')

from ..register import REGISTER
REGISTER.set_class(REGISTER.get_package_name(__name__), 'adam', __name__)