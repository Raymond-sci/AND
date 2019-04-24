#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-12 21:37:13
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.config import CONFIG as cfg
from packages.register import REGISTER
from packages.loggers.std_logger import STDLogger as logger


def require_args():
    cfg.add_argument('--ANs-select-rate', default=0.25, type=float,
                        help='ANs select rate at each round')
    cfg.add_argument('--ANs-size', default=1, type=int,
                        help='ANs size discarding the anchor')

class ANsDiscovery(nn.Module):
    """Discovery ANs
    
    Discovery ANs according to current round, select_rate and most importantly,
    all sample's corresponding entropy
    """

    def __init__(self, nsamples):
        """Object used to discovery ANs
        
        Discovery ANs according to the total amount of samples, ANs selection
        rate, ANs size
        
        Arguments:
            nsamples {int} -- total number of sampels
            select_rate {float} -- ANs selection rate
            ans_size {int} -- ANs size
        
        Keyword Arguments:
            device {str} -- [description] (default: {'cpu'})
        """
        super(ANsDiscovery, self).__init__()

        # not going to use ``register_buffer'' as 
        # they are determined by configs
        self.select_rate = cfg.ANs_select_rate
        self.ANs_size = cfg.ANs_size
        # number of samples
        self.register_buffer('samples_num', torch.tensor(nsamples))
        # indexes list of anchor samples
        self.register_buffer('anchor_indexes', torch.LongTensor([]))
        # indexes list of instance samples
        self.register_buffer('instance_indexes', torch.arange(nsamples).long())
        # anchor samples' and instance samples' position
        self.register_buffer('position', -1 * torch.arange(nsamples).long() - 1)
        # anchor samples' neighbours
        self.register_buffer('neighbours', torch.LongTensor([]));
        # each sample's entropy
        self.register_buffer('entropy', torch.FloatTensor(nsamples));
        # consistency
        self.register_buffer('consistency', torch.tensor(0.));

    def get_ANs_num(self, round):
        """Get number of ANs
        
        Get number of ANs at target round according to the select rate
        
        Arguments:
            round {int} -- target round
        
        Returns:
            int -- number of ANs
        """
        return int(self.samples_num.float() * self.select_rate * round)

    def update(self, round, npc, cheat_labels=None):
        """Update ANs
        
        Discovery new ANs and update `anchor_indexes`, `instance_indexes` and
        `neighbours`
        
        Arguments:
            round {int} -- target round
            npc {Module} -- non-parametric classifier
            cheat_labels {list} -- used to compute consistency of chosen ANs only
        
        Returns:
            number -- [updated consistency]
        """
        with torch.no_grad():
            batch_size = 100
            ANs_num = self.get_ANs_num(round)
            logger.debug('Going to choose %d samples as anchors' % ANs_num)
            features = npc.memory

            logger.debug('Start to compute each sample\'s entropy')
            for start in xrange(0, self.samples_num, batch_size):
                logger.progress(start, self.samples_num, 'processing %d/%d samples...')

                end = start + batch_size
                end = min(end, self.samples_num)

                preds = F.softmax(npc(features[start:end], None), 1)
                self.entropy[start:end] = -(preds * preds.log()).sum(1)

            logger.debug('Compute entropy done, max(%.2f), min(%.2f), mean(%.2f)'
                % (self.entropy.max(), self.entropy.min(), self.entropy.mean()))

            # get the anchor list and instance list according to the computed
            # entropy
            self.anchor_indexes = self.entropy.topk(ANs_num, largest=False)[1]
            self.instance_indexes = (torch.ones_like(self.position)
                                        .scatter_(0, self.anchor_indexes, 0)
                                        .nonzero().view(-1))
            anchor_entropy = self.entropy.index_select(0, self.anchor_indexes)
            instance_entropy = self.entropy.index_select(0, self.instance_indexes)
            if self.anchor_indexes.size(0) > 0:
                logger.debug('Entropies of anchor samples: max(%.2f), '
                                'min(%.2f), mean(%.2f)' % (anchor_entropy.max(),
                                anchor_entropy.min(), anchor_entropy.mean()))
            if self.instance_indexes.size(0) > 0:
                logger.debug('Entropies of instance sample: max(%.2f), '
                                'min(%.2f), mean(%.2f)' % (instance_entropy.max(),
                                instance_entropy.min(), instance_entropy.mean()))

            # get position
            # if the anchor sample x whose index is i while position is j, then
            # sample x_i is the j-th anchor sample at current round
            # if the instance sample x whose index is i while position is j, then
            # sample x_i is the (-j-1)-th instance sample at current round
            logger.debug('Start to get the position of both anchor and '
                          'instance samples')
            instance_cnt = 0
            for i in xrange(self.samples_num):
                logger.progress(i, self.samples_num, 'processing %d/%d samples...')

                # for anchor samples
                if (i == self.anchor_indexes).any():
                    self.position[i] = (self.anchor_indexes == i).max(0)[1]
                    continue
                # for instance samples
                instance_cnt -= 1
                self.position[i] = instance_cnt

            logger.debug('Start to find %d neighbours for each anchor sample'
                             % self.ANs_size)
            anchor_features = features.index_select(0, self.anchor_indexes)
            self.neighbours = (torch.LongTensor(ANs_num, self.ANs_size)
                                                            .to(cfg.device))
            for start in xrange(0, ANs_num, batch_size):
                logger.progress(start, ANs_num, 'processing %d/%d samples...')

                end = start + batch_size
                end = min(end, ANs_num)

                sims = torch.mm(anchor_features[start:end], features.t())
                sims.scatter_(1, self.anchor_indexes[start:end].view(-1, 1), -1.)
                _, self.neighbours[start:end] = (
                            sims.topk(self.ANs_size, largest=True, dim=1))
            logger.debug('ANs discovery done')

            # if cheat labels is provided, then compute consistency
            if cheat_labels is None:
                return 0.
            logger.debug('Start to compute ANs consistency')
            anchor_label = cheat_labels.index_select(0, self.anchor_indexes)
            neighbour_label = cheat_labels.index_select(0,
                    self.neighbours.view(-1)).view_as(self.neighbours)
            self.consistency = ((anchor_label.view(-1, 1) == neighbour_label)
                                                            .float().mean())

            return self.consistency

REGISTER.set_package(__name__)
REGISTER.set_class(__name__, 'ans', ANsDiscovery)
