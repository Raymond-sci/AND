#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-26 20:19:00
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

from ..loggers.std_logger import STDLogger as logger
from ..config import CONFIG as cfg

class StepPolicy:
    """Caffe style step decay learning rate policy
    
    Decay learning rate at every `lr-decay-step` steps after the
    first `lr-decay-offset` ones at the rate of `lr-decay-rate`
    """

    def __init__(self, *args, **kwargs):
        if len(args) + len(kwargs) == 0:
            self.__init_by_cfg()
        else:
            self.__init(*args, **kwargs)

    def __init(self, base_lr, offset, step, rate):
        self.base_lr = base_lr
        self.offset = offset
        self.step = step
        self.rate = rate
        logger.debug('Going to use [step] learning policy for optimization with '
            'base learning rate %.5f, offset %d, step %d and decay rate %f' %
            (base_lr, offset, step, rate))

    def __init_by_cfg(self):
        self.__init(cfg.base_lr, cfg.lr_decay_offset,
                    cfg.lr_decay_step, cfg.lr_decay_rate)

    @staticmethod
    def require_args():

        cfg.add_argument('--lr-decay-offset', default=0, type=int,
                        help='learning rate will start to decay at which step')

        cfg.add_argument('--lr-decay-step', default=0, type=int,
                        help='learning rate will decay at every n round')


        cfg.add_argument('--lr-decay-rate', default=0.1, type=float,
                        help='learning rate will decay at what rate')

    def update(self, steps):
        """decay learning rate according to current step
        
        Decay learning rate at a fixed ratio
        
        Arguments:
            steps {int} -- current steps
        
        Returns:
            int -- updated learning rate
        """

        if steps < self.offset:
            return self.base_lr
        
        return self.base_lr * (self.rate ** ((steps - self.offset) // self.step))


from ..register import REGISTER
REGISTER.set_class(REGISTER.get_package_name(__name__), 'step', StepPolicy)
