#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-28 21:45:01
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

import torchvision.transforms as transforms

class RandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(self, size, **kwargs):
        super(RandomResizedCrop, self).__init__(0, **kwargs)
        self.size = size