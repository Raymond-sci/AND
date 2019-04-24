#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-25 17:16:34
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

from tensorboardX import SummaryWriter as TFBWriter

from ..config import CONFIG as cfg

class TFLogger():

    @staticmethod
    def require_args():
        
        cfg.add_argument('--log-tfb', action='store_true',
                        help='use tensorboard to log training process. '
                        '(default: False)')

    def __init__(self, debugging, *args, **kwargs):
        self.debugging = debugging
        if not self.debugging and cfg.log_tfb:
            self.writer = TFBWriter(*args, **kwargs)

    def __getattr__(self,attr):
        if self.debugging or not cfg.log_tfb:
            return do_nothing
        return self.writer.__getattribute__(attr)

def do_nothing(*args, **kwargs):
    pass


from ..register import REGISTER
REGISTER.set_class(REGISTER.get_package_name(__name__), 'tf_logger', TFLogger)
