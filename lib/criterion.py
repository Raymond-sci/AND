#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-07 22:32:15
# @Author  : Jiabo (Raymond) Huang (jiabo.huang@qmul.ac.uk)
# @Link    : https://github.com/Raymond-sci

import torch
import torch.nn.functional as F
import torch.nn as nn

class Criterion(nn.Module):

    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, x, y, ANs):
        batch_size, _ = x.shape
        
        # split anchor and instance list
        anchor_indexes, instance_indexes = self.__split(y, ANs)
        preds = F.softmax(x, 1)

        l_ans = 0.
        if anchor_indexes.size(0) > 0:
            # compute loss for anchor samples
            y_ans = y.index_select(0, anchor_indexes)
            y_ans_neighbour = ANs.position.index_select(0, y_ans)
            neighbours = ANs.neighbours.index_select(0, y_ans_neighbour)
            # p_i = \sum_{j \in \Omega_i} p_{i,j}
            x_ans = preds.index_select(0, anchor_indexes)
            x_ans_neighbour = x_ans.gather(1, neighbours).sum(1)
            x_ans = x_ans.gather(1, y_ans.view(-1, 1)).view(-1) + x_ans_neighbour
            # NLL: l = -log(p_i)
            l_ans = -1 * torch.log(x_ans).sum(0)

        l_inst = 0.
        if instance_indexes.size(0) > 0:
            # compute loss for instance samples
            y_inst = y.index_select(0, instance_indexes)
            x_inst = preds.index_select(0, instance_indexes)
            # p_i = p_{i, i}
            x_inst = x_inst.gather(1, y_inst.view(-1, 1))
            # NLL: l = -log(p_i)
            l_inst = -1 * torch.log(x_inst).sum(0)

        return (l_inst + l_ans) / batch_size

    def __split(self, y, ANs):
        pos = ANs.position.index_select(0, y.view(-1))
        return (pos >= 0).nonzero().view(-1), (pos < 0).nonzero().view(-1)
