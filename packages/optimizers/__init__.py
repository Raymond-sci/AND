#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-24 23:26:56
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

from . import sgd
from . import adam
from . import rmsprop

from ..register import REGISTER
from ..config import CONFIG as cfg

def require_args():
    """all args for optimizer objects

    Arguments:
        parser {argparse} -- current version of argparse object
    """

    cfg.add_argument('--weight-decay', default=0, type=float,
                        help='weight decay (L2 penalty)')

    known_args, _ = cfg.parse_known_args()

    if (REGISTER.is_package_registered(__name__) and
        REGISTER.is_class_registered(__name__, known_args.optimizer)):
    
        optimizer = get(known_args.optimizer)

        if hasattr(optimizer, 'require_args'):
            return optimizer.require_args()

def get(name, instant=False, params=None):
    cls = REGISTER.get_class(__name__, name)
    if instant:
        return cls.get(params)
    return cls

def register(name, cls):
    REGISTER.set_class(__name__, name, cls)


