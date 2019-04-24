import torch
import datasets
from lib.utils import AverageMeter, traverse
import sys

from packages.register import REGISTER
from packages.loggers.std_logger import STDLogger as logger

def NN(net, npc, trainloader, testloader, K=0, sigma=0.1, 
        recompute_memory=False, device='cpu'):

    # switch model to evaluation mode
    net.eval()

    # tracking variables
    correct = 0.
    total = 0

    trainFeatures = npc.memory
    trainLabels = torch.LongTensor(trainloader.dataset.labels).to(device)

    # recompute features for training samples
    if recompute_memory:
        trainFeatures, trainLabels = traverse(net, trainloader, 
                                    testloader.dataset.transform, device)
    trainFeatures = trainFeatures.t()
    
    # start to evaluate
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            logger.progress(batch_idx, len(testloader), 'processing %d/%d batch...')
            inputs, targets = inputs.to(device), targets.to(device)
            batchSize = inputs.size(0)

            # forward
            features = net(inputs)

            # cosine similarity
            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()
            
    return correct/total

def kNN(net, npc, trainloader, testloader, K=200, sigma=0.1,
            recompute_memory=False, device='cpu'):

    # set the model to evaluation mode
    net.eval()

    # tracking variables
    total = 0

    trainFeatures = npc.memory
    trainLabels = torch.LongTensor(trainloader.dataset.labels).to(device)

    # recompute features for training samples
    if recompute_memory:
        trainFeatures, trainLabels = traverse(net, trainloader, 
                                    testloader.dataset.transform, device)
    trainFeatures = trainFeatures.t()
    C = trainLabels.max() + 1
    
    # start to evaluate
    top1 = 0.
    top5 = 0.
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C.item()).to(device)
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            logger.progress(batch_idx, len(testloader), 'processing %d/%d batch...')

            batchSize = inputs.size(0)
            targets, inputs = targets.to(device), inputs.to(device)

            # forward
            features = net(inputs)

            # cosine similarity
            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C),
                                        yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

    return top1/total

def get(name):
    return REGISTER.get_class(__name__, name)

REGISTER.set_package(__name__)
REGISTER.set_class(__name__, 'knn', kNN)
REGISTER.set_class(__name__, 'nn', NN)