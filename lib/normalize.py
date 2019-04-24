#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-12 21:37:13
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

import torch
from torch.autograd import Variable
from torch import nn

class Normalize(nn.Module):
    """Normalize module
    
    Module used to normalize matrix
    
    Extends:
        nn.Module
    """

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out
