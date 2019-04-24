#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-24 23:26:16
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

from . import std_logger
from . import tf_logger

from ..register import REGISTER

def require_args():
    """all args for logger objects
    
    Arguments:
        parser {argparse} -- current version of argparse object
    """
    if not REGISTER.is_package_registered(__name__):
        return parser

    classes = REGISTER.get_classes(__name__)

    for (name, cls) in classes.items():
        if hasattr(cls, 'require_args'):
            cls.require_args()

def get(name):
    return REGISTER.get_class(__name__, name)

def register(name, cls):
    REGISTER.set_class(__name__, name, cls)