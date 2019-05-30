#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-27 15:09:03
# @Author  : Jiabo (Raymond) Huang (jiabo.huang@qmul.ac.uk)
# @Link    : https://github.com/Raymond-sci

import torch
import torch.backends.cudnn as cudnn

import sys
import os
import time
from datetime import datetime

import models
import datasets

from lib import protocols
from lib.non_parametric_classifier import NonParametricClassifier
from lib.criterion import Criterion
from lib.ans_discovery import ANsDiscovery
from lib.utils import AverageMeter, time_progress, adjust_learning_rate

from packages import session
from packages import lr_policy
from packages import optimizers
from packages.config import CONFIG as cfg
from packages.loggers.std_logger import STDLogger as logger
from packages.loggers.tf_logger import TFLogger as SummaryWriter


def require_args():
    
    # dataset to be used
    cfg.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset to be used. (default: cifar10)')
    
    # network to be used
    cfg.add_argument('--network', default='resnet18', type=str,
                        help='backbone to be used. (default: ResNet18)')

    # optimizer to be used
    cfg.add_argument('--optimizer', default='sgd', type=str,
                        help='optimizer to be used. (default: sgd)')

    # lr policy to be used
    cfg.add_argument('--lr-policy', default='step', type=str,
                        help='lr policy to be used. (default: step)')

    # args for protocol
    cfg.add_argument('--protocol', default='knn', type=str,
                        help='protocol used to validate model')

    # args for network training
    cfg.add_argument('--max-epoch', default=200, type=int,
                        help='max epoch per round. (default: 200)')
    cfg.add_argument('--max-round', default=5, type=int, 
                        help='max iteration, including initialisation one. '
                             '(default: 5)')
    cfg.add_argument('--iter-size', default=1, type=int,
                        help='caffe style iter size. (default: 1)')
    cfg.add_argument('--display-freq', default=1, type=int,
                        help='display step')
    cfg.add_argument('--test-only', action='store_true', 
                        help='test only')


def main():

    logger.info('Start to declare training variables')
    cfg.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    start_round = 0 # start for iter 0 or last checkpoint iter

    logger.info('Start to prepare data')
    trainset, trainloader, testset, testloader = datasets.get(cfg.dataset, instant=True)
    # cheat labels are used to compute neighbourhoods consistency only
    cheat_labels = torch.tensor(trainset.labels).long().to(device)
    ntrain, ntest = len(trainset), len(testset)
    logger.info('Totally got %d training and %d test samples' % (ntrain, ntest))

    logger.info('Start to build model')
    net = models.get(cfg.network, instant=True)
    npc = NonParametricClassifier(cfg.low_dim, ntrain, cfg.npc_temperature, cfg.npc_momentum)
    ANs_discovery = ANsDiscovery(ntrain)
    criterion = Criterion()
    optimizer = optimizers.get(cfg.optimizer, instant=True, params=net.parameters())
    lr_handler = lr_policy.get(cfg.lr_policy, instant=True)
    protocol = protocols.get(cfg.protocol)
    
    # data parallel
    if device == 'cuda':
        if (cfg.network.lower().startswith('alexnet') or
            cfg.network.lower().startswith('vgg')):
            net.features = torch.nn.DataParallel(net.features,
                                    device_ids=range(len(cfg.gpus.split(','))))
        else:
            net = torch.nn.DataParallel(net, device_ids=range(
                                                    len(cfg.gpus.split(','))))
        cudnn.benchmark = True

    net, npc, ANs_discovery, criterion = (net.to(device), npc.to(device), 
        ANs_discovery.to(device), criterion.to(device))
    
    # load ckpt file if necessary
    if cfg.resume:
        assert os.path.exists(cfg.resume), "Resume file not found: %s" % cfg.resume
        logger.info('Start to resume from %s' % cfg.resume)
        ckpt = torch.load(cfg.resume)
        net.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        npc = npc.load_state_dict(ckpt['npc'])
        ANs_discovery.load_state_dict(ckpt['ANs_discovery'])
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
        start_round = ckpt['round']

    # test if necessary
    if cfg.test_only:
        logger.info('Testing at beginning...')
        acc = protocol(net, npc, trainloader, testloader, 200,
                            cfg.npc_temperature, True, device)
        logger.info('Evaluation accuracy at %d round and %d epoch: %.2f%%' %
                                        (start_round, start_epoch, acc * 100))
        sys.exit(0)

    logger.info('Start the progressive training process from round: %d, '
        'epoch: %d, best acc is %.4f...' % (start_round, start_epoch, best_acc))
    round = start_round
    global_writer = SummaryWriter(cfg.debug,
                                log_dir=os.path.join(cfg.tfb_dir, 'global'))
    while (round < cfg.max_round):

        # variables are initialized to different value in the first round
        is_first_round = True if round == start_round else False
        best_acc = best_acc if is_first_round else 0

        if not is_first_round:
            logger.info('Start to mining ANs at %d round' % round)
            ANs_discovery.update(round, npc, cheat_labels)
            logger.info('ANs consistency at %d round is %.2f%%' %
                        (round, ANs_discovery.consistency * 100))

        ANs_num = ANs_discovery.anchor_indexes.shape[0]
        global_writer.add_scalar('ANs/Number', ANs_num, round)
        global_writer.add_scalar('ANs/Consistency', ANs_discovery.consistency, round)

        # declare local writer
        writer = SummaryWriter(cfg.debug, log_dir=os.path.join(cfg.tfb_dir,
                                        '%04d-%05d' % (round, ANs_num)))
        logger.info('Start training at %d/%d round' % (round, cfg.max_round))


        # start to train for an epoch
        epoch = start_epoch if is_first_round else 0
        lr = cfg.base_lr
        while lr > 0 and epoch < cfg.max_epoch:

            # get learning rate according to current epoch
            lr = lr_handler.update(epoch)

            train(round, epoch, net, trainloader, optimizer, npc, criterion,
                ANs_discovery, lr, writer)

            logger.info('Start to evaluate...')
            acc = protocol(net, npc, trainloader, testloader, 200,
                            cfg.npc_temperature, False, device)
            writer.add_scalar('Evaluate/Rank-1', acc, epoch)

            logger.info('Evaluation accuracy at %d round and %d epoch: %.1f%%'
                                                % (round, epoch, acc * 100))
            logger.info('Best accuracy at %d round and %d epoch: %.1f%%'
                                            % (round, epoch, best_acc * 100))

            is_best = acc >= best_acc
            best_acc = max(acc, best_acc)
            if is_best and not cfg.debug:
                target = os.path.join(cfg.ckpt_dir, '%04d-%05d.ckpt'
                                                        % (round, ANs_num))
                logger.info('Saving checkpoint to %s' % target)
                state = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ANs_discovery' : ANs_discovery.state_dict(),
                    'npc' : npc.state_dict(),
                    'acc': acc,
                    'epoch': epoch + 1,
                    'round' : round,
                    'session' : cfg.session
                }
                torch.save(state, target)
            epoch += 1

        # log best accuracy after each iteration
        global_writer.add_scalar('Evaluate/best_acc', best_acc, round)
        round += 1

# Training
def train(round, epoch, net, trainloader, optimizer, npc, criterion,
            ANs_discovery, lr, writer):

    # tracking variables
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch the model to train mode
    net.train()
    # adjust learning rate
    adjust_learning_rate(optimizer, lr)

    end = time.time()
    start_time = datetime.now()
    optimizer.zero_grad()
    for batch_idx, (inputs, _, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, indexes = inputs.to(cfg.device), indexes.to(cfg.device)

        features = net(inputs)
        outputs = npc(features, indexes)
        loss = criterion(outputs, indexes, ANs_discovery) / cfg.iter_size

        loss.backward()
        train_loss.update(loss.item() * cfg.iter_size, inputs.size(0))

        if batch_idx % cfg.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % cfg.display_freq != 0:
            continue

        writer.add_scalar('Train/Learning_Rate', lr,
                        epoch * len(trainloader) + batch_idx)
        writer.add_scalar('Train/Loss', train_loss.val,
                        epoch * len(trainloader) + batch_idx)


        elapsed_time, estimated_time = time_progress(batch_idx + 1,
                                        len(trainloader), batch_time.sum)
        logger.info('Round: {round} Epoch: {epoch}/{tot_epochs} '
              'Progress: {elps_iters}/{tot_iters} ({elps_time}/{est_time}) '
              'Data: {data_time.avg:.3f} LR: {learning_rate:.5f} '
              'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
              round=round, epoch=epoch, tot_epochs=cfg.max_epoch,
              elps_iters=batch_idx, tot_iters=len(trainloader),
              elps_time=elapsed_time, est_time=estimated_time,
              data_time=data_time, learning_rate=lr,
              train_loss=train_loss))

if __name__ == '__main__':
    
    session.run(__name__)
