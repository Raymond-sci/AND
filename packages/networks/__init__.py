#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-24 23:26:56
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

from ..register import REGISTER
from ..config import CONFIG as cfg

def require_args():
    """all args for network objects
    
    Arguments:
        parser {argparse} -- current version of argparse object
    """

    known_args, _ = cfg.parse_known_args()

    if (REGISTER.is_package_registered(__name__) and
        REGISTER.is_class_registered(__name__, known_args.network)):

        network = get(known_args.network)
    
        if  hasattr(network, 'require_args'):
            # get args for network
            return network.require_args()

def get(name, instant=False):
    cls = REGISTER.get_class(__name__, name)
    if instant:
        return cls()
    return cls

def register(name, cls):
    REGISTER.set_class(__name__, name, cls)