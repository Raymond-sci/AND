#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-12 21:37:13
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

import os
import sys
import shutil
import numpy as np
from datetime import timedelta

import torch

from packages.loggers.std_logger import STDLogger as logger

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def time_progress(elapsed_iters, tot_iters, elapsed_time):
    estimated_time = 1. * tot_iters / elapsed_iters * elapsed_time
    elapsed_time = timedelta(seconds=elapsed_time)
    estimated_time = timedelta(seconds=estimated_time)
    return tuple(map(lambda x:str(x).split('.')[0], [elapsed_time, estimated_time]))

def save_ckpt(state_dict, target, is_best=False):
    latest, best = map(lambda x:os.path.join(target, x), ['latest.ckpt', 'best.ckpt'])
    # save latest checkpoint
    torch.save(state_dict, latest)
    # if is best, then copy latest to best
    if not is_best:
        return
    shutil.copyfile(latest, best)

def traverse(net, loader, transform=None, tencrops=False, device='cpu'):

    bak_transform = loader.dataset.transform
    if transform is not None:
        loader.dataset.transform = transform

    features = None
    labels = torch.zeros(len(loader.dataset)).long().to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(loader):
            logger.progress(batch_idx, len(loader), 'processing %d/%d batch...')

            if tencrops:
                bs, ncrops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)
            inputs, targets, indexes = (inputs.to(device), targets.to(device),
                                                            indexes.to(device))

            feats = net(inputs)
            if tencrops:
                feats = torch.squeeze(feats.view(bs, ncrops, -1).mean(1))

            if features is None:
                features = torch.zeros(len(loader.dataset), feats.shape[1]).to(device)
            features.index_copy_(0, indexes, feats)
            labels.index_copy_(0, indexes, targets)

    loader.dataset.transform = bak_transform

    return features, labels
