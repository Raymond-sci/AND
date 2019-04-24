#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-26 20:19:00
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

from ..loggers.std_logger import STDLogger as logger
from ..config import CONFIG as cfg

class MultiStepPolicy:
    """Caffe style step decay learning rate policy
    
    Decay learning rate at every `lr-decay-step` steps after the
    first `lr-decay-offset` ones at the rate of `lr-decay-rate`
    """

    def __init__(self, *args, **kwargs):
        if len(args) + len(kwargs) == 0:
            self.__init_by_cfg()
        else:
            self.__init(*args, **kwargs)

    def __init(self, base_lr, schedule):
        self.base_lr = base_lr
        self.schedule = schedule
        logger.debug('Going to use [multistep] learning policy for optimization '
            'with base learing rate %.5f and schedule from %s'
            % (base_lr, schedule))

    def __init_by_cfg(self):
        schedule = cfg.lr_schedule
        assert schedule is not None and os.path.exists(schedule), ('Schedule '
                                            'file not found: [%s]' % schedule)
        self.__init(cfg.base_lr, schedule)

    @staticmethod
    def require_args():

        cfg.add_argument('--lr-schedule', default=None, type=str,
                        help='learning rate schedule')

    def update(self, steps):
        """update learning rate
        
        Update learning rate according to current steps and schedule file
        
        Arguments:
            steps {int} -- current steps
        
        Returns:
            float -- updated file
        """

        lines = filter(lambda x:not x.startswith('#'),
                       open(self.schedule, 'r').readlines())
        assert len(lines) > 0, 'Invalid schedule file'

        learning_rate = self.base_lr
        for line in lines:
            
            line = line.split('#')[0]
            anchor, target = line.strip().split(':')
            
            if target.startswith('-'):
                lr = -1
            else:
                lr = float(target)
                if steps <= anchor:
                    learning_rate = lr
                else:
                    break

        return learning_rate


from ..register import REGISTER
REGISTER.set_class(REGISTER.get_package_name(__name__), 'multistep', MultiStepPolicy)
