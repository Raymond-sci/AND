#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-26 20:11:16
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

from . import step
from . import multistep
from . import fixed

from ..register import REGISTER
from ..config import CONFIG as cfg

def require_args():
    """all args for optimizer objects

    Arguments:
        parser {argparse} -- current version of argparse object
    """

    cfg.add_argument('--base-lr', default=1e-1, type=float,
                        help='base learning rate. (default: 1e-1)')

    known_args, _ = cfg.parse_known_args()

    if (REGISTER.is_package_registered(__name__) and
        REGISTER.is_class_registered(__name__, known_args.lr_policy)):
    
        policy = get(known_args.lr_policy)

        if hasattr(policy, 'require_args'):
            return policy.require_args()

def get(name, instant=False):
    cls = REGISTER.get_class(__name__, name)
    if instant:
        return cls()
    return cls

def register(name, cls):
    REGISTER.set_class(__name__, name, cls)