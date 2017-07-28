#!/usr/bin/env python3
import sys
import os
import time
from collections import OrderedDict
import random
import argparse
import json

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable

from dataloader import DataLoader

### CONFIGURE ###
argparser = argparse.ArgumentParser()
argparser.add_argument('-g', '--gpu', action='store_true',
                       help='use GPU/CUDA insteaf of CPU')
argparser.add_argument('-d', '--double', action='store_true',
                       help='use fp64 instead of fp32')
argparser.add_argument('-m', '--maxiter', type=int, default=6400,
                       help='set maximum iterations of training',)
argparser.add_argument('-s', '--seed', type=int, default=1,
                       help='set manual seed')
argparser.add_argument('-n', '--numthreads', type=int, default=4,
                       help='set *_NUM_THREADS environment variable')
args = argparser.parse_args()
print('=> Dump configuration')
print(json.dumps(vars(args), indent=2))

### ENVIRONMENT ###
os.putenv('OPENBLAS_NUM_THREADS', str(args.numthreads))
os.putenv('OMP_NUM_THREADS', str(args.numthreads))
os.putenv('MKL_NUM_THREADS', str(args.numthreads))

### TIMER SETUP ###
perf_tm = {}
perf_getdiff = lambda d: d['end'] - d['start']

### TORCH SETUP ###
print('-> Using PyTorch', th.__version__)
th.manual_seed(args.seed)
if args.gpu: th.cuda.manual_seed(args.seed)
X_TENSOR = ''
if not args.gpu:
    X_TENSOR = 'torch.DoubleTensor' if args.double else 'torch.FloatTensor'
else:
    X_TENSOR = 'torch.cuda.DoubleTensor' if args.double else 'torch.cuda.FloatTensor'
    #th.set_default_tensor_type('torch.cuda.HalfTensor') # Bad Stability
th.set_default_tensor_type(X_TENSOR)

### DataLoader ###
dataloader = DataLoader()

### Model ###
class QuickNet(th.nn.Module):
    ''' Reference: caffe/examples/cifar10 # 70%
    https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
    '''
    def __init__(self):
        super(QuickNet, self).__init__()
        self.SEQ1 = th.nn.Sequential(OrderedDict([
('conv1', th.nn.Conv2d(3, 32, 5, stride=1, padding=2)), #64,32,32
('relu1', th.nn.ReLU()),
('pool1', th.nn.MaxPool2d(3, stride=2, padding=1)), #64,16,16
('bn1',   th.nn.BatchNorm2d(32)),
('conv2', th.nn.Conv2d(32, 64, 5, stride=1, padding=2)), #64,16,16
('relu2', th.nn.ReLU()),
('bn2',   th.nn.BatchNorm2d(64)),
('pool2', th.nn.MaxPool2d(3, stride=2, padding=1)), #64,8,8
]))
        self.SEQ2 = th.nn.Sequential(OrderedDict([
('fc4',   th.nn.Linear(4096, 384)),
('relu4', th.nn.ReLU()),
('bn4',   th.nn.BatchNorm1d(384)),
('fc5',   th.nn.Linear(384, 192)),
('bn6',   th.nn.BatchNorm1d(192)),
('fc6',   th.nn.Linear(192, 10)),
]))
    def forward(self, x):
        x = self.SEQ1(x)
        x = x.view(-1, 4096)
        x = self.SEQ2(x)
        return x

### Create Instances
net = QuickNet() if not args.gpu else QuickNet().cuda()
if not args.double: net = net.float()
print(net)
crit = th.nn.CrossEntropyLoss()
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)

### Data Transformation
def transform(images, labels):
    images = images.reshape(-1, 3, 32, 32) / 255.
    if random.choice((True,False)):
        images = np.flip(images, 3) # 77%->79%
    images = Variable(th.from_numpy(images.astype(np.double)), requires_grad=False)
    labels = Variable(th.from_numpy(labels.reshape(-1).astype(np.long)), requires_grad=False)
    if args.gpu: images, labels = images.cuda(), labels.cuda()
    if not args.double: images = images.float()
    return images, labels

### Evaluation on Validation set
def evaluate(i, net, dataloader):
    print('-> EVAL @ {} |'.format(i), end='')
    net.eval()
    correct = 0
    total = 0
    lossaccum = 0
    dataloader.reset('test')
    for j in range(dataloader.itersInEpoch('test', 100)):
        images, labels = dataloader.getBatch('test', 100)
        images, labels = transform(images, labels)
        out = net(images)
        loss = crit(out, labels)
        pred = out.data.max(1)[1]
        correct += pred.eq(labels.data).cpu().sum()
        total += 100
        lossaccum += loss.data[0]
        print('.', end=''); sys.stdout.flush()
    print('|')
    print('-> TEST @ {} |'.format(i),
            'Loss {:7.3f} |'.format(lossaccum),
            'Accu {:.5f}|'.format(correct / total))

### Training
perf_tm['start'] = time.time()
for i in range(args.maxiter+1):
    # read data
    images, labels = dataloader.getBatch('trainval', 100)
    images, labels = transform(images, labels)

    # decay the learning rate
    if i!=0 and i%1000==0 and i<5000:
        print('-> *0.7 the learning rate')
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.7
            print(param_group['lr'])

    # train
    net.train()
    out = net(images)
    loss = crit(out, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred = out.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    print('-> Iter {:5d} |'.format(i), 'loss {:7.3f} |'.format(loss.data[0]),
            'Bch Train Accu {:.2f}'.format(correct / out.size()[0]))

    # test
    if i%100==0: evaluate(i, net, dataloader)

perf_tm['end'] = time.time()
print('-> done, time elapsed', perf_getdiff(perf_tm))

'''
time:
    6400 Iterations
    TitanX Pascal / fp32 : 55.483641624450684 s, 621MiB
    TitanX Pascal / fp64 : 351.21968483924866 s, 749MiB
'''
