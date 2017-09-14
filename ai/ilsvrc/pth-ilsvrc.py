#!/usr/bin/env python3
'''
ref: https://github.com/pytorch/examples/blob/master/imagenet/main.py
'''
import sys
import os
import time
from collections import OrderedDict
import random
import argparse
import json
import math
from PIL import Image

import numpy as np
import torch as th
import torch.nn.init
import torch.nn.functional as F
from torch.autograd import Variable

from dataloader import DataLoader
import torchvision

### CONFIGURE ###
argparser = argparse.ArgumentParser()
argparser.add_argument('-g', '--gpu', action='store_true',
                       help='use GPU/CUDA insteaf of CPU')
argparser.add_argument('-d', '--double', action='store_true',
                       help='use fp64 instead of fp32')
argparser.add_argument('-m', '--maxiter', type=int,
                       default=1281167*90, # 90 Epoches.
                       help='set maximum iterations of training',)
argparser.add_argument('-s', '--seed', type=int, default=1,
                       help='set manual seed')
argparser.add_argument('-n', '--numthreads', type=int, default=4,
                       help='set *_NUM_THREADS environment variable')
argparser.add_argument('-t', '--testevery', type=int, default=200,
                       help='set model evaluation interval')
argparser.add_argument('-e', '--lr', type=float, default=1e-1,
                       help='set the initial learning rate')
argparser.add_argument('-b', '--batchsize', type=int, default=256,
                       help='set batch size for training')
argparser.add_argument('--testbatchsize', type=int, default=100,
                       help='set batch size for test')
argparser.add_argument('--arch', type=str, default='resnet18',
choices=('alexnet', 'resnet18','resnet34','resnet50', 'resnet101', 'resnet152'),
                       help='choose model architecture')
                       
args = argparser.parse_args()
print('=> Dump configuration')
print(json.dumps(vars(args), indent=2))

### ENVIRONMENT ###
os.putenv('OPENBLAS_NUM_THREADS', str(args.numthreads))
os.putenv('OMP_NUM_THREADS', str(args.numthreads))
os.putenv('MKL_NUM_THREADS', str(args.numthreads))

### Misc
def barX(colorcode):
    return lambda x,xmax,width: print('{:>4.0%}'.format(x/xmax)+\
        '|'+'\x1b[{};1m'.format(colorcode)+'*'*round(width*x/xmax)+\
        ' '*round(width-width*x/xmax)+'\x1b[;m'+'|')
# Tips : get terminal width like this -- os.get_terminal_size().columns-6
barG = barX('32') # Green for train Acc
barY = barX('33') # Yellow for train loss
barC = barX('36') # Cyan for test Acc
barR = barX('31') # Red for test loss

### TIMER SETUP ###
class Perf_TM(object):
    def __init__(self):
        self.d = {} # dict{'key': list[float]}
    def go(self, key):
        if key not in self.d.keys(): self.d[key] = []
        self.d[key].append(-time.time())
    def halt(self, key):
        self.d[key][-1] += time.time()
    def dump(self):
        s = dict(self.d)
        for key in s.keys():
            if len(s[key])>1:
                num_rec = len(s[key])
                s[key] = [sum(s[key])/num_rec, 'Average of the '+str(num_rec)+' records']
        print(json.dumps(s, indent=2))
perf_tm = Perf_TM()

### SETUP Recorder, for loss curve and etc.
class Perf_ML(dict):
    def go(self, name, it, value):
        if name not in self.keys():
            self[name] = []
        self[name].append((it, value))
    def dump(self, name=None):
        #print(json.dumps(self, indent=2)) # Useless
        for k,v in self['test/loss']:
            barR(v, self['test/loss'][0][1], 100)
        for k,v in self['test/acc']:
            barC(v, 1, 100)
perf_ml = Perf_ML()

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
th.backends.cudnn.benchmark=True # enable cuDNN auto-tuner

### DataLoader ###
dataloader = DataLoader()
dataloader.satellite(64, 'train', 100)

### Model ###
Model = eval('torchvision.models.{}'.format(args.arch))
net = Model() if not args.gpu else Model().cuda()
if not args.double: net = net.float()
print(net)
crit = th.nn.CrossEntropyLoss()
#optimizer = th.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-7)
optimizer = th.optim.SGD(net.parameters(), lr=args.lr,
                         momentum=0.9, weight_decay=1e-4)

### Data Transformation
def transform(images, labels, training=False):
    images = images.reshape(-1, 3, 224, 224) / 255.
    images = Variable(th.from_numpy(images.astype(np.float)), requires_grad=False)
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
    for j in range(dataloader.itersInEpoch('test', args.testbatchsize)):
        images, labels = dataloader.getBatch('test', args.testbatchsize)
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
    perf_ml.go('test/acc', i, correct/total)
    perf_ml.go('test/loss', i, lossaccum)
    barC(correct, total, 80)
    barR(lossaccum, perf_ml['test/loss'][0][1], 80)

### Training
perf_tm.go('all')
for i in range(args.maxiter+1):
    # read data
    perf_tm.go('data/fetch')
    #images, labels = dataloader.getBatch('train', args.batchsize)
    images, labels = dataloader.Q.get()
    images, labels = transform(images, labels, training=True)
    perf_tm.halt('data/fetch')

    # decay the learning rate
    curlr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = curlr

    # train
    perf_tm.go('train/flbu')
    net.train()
    out = net(images)
    loss = crit(out, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    perf_tm.halt('train/flbu')

    pred = out.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    print('-> Iter {:5d} ({:<5d}/{:5d} Eph {:>3d} ) |'.format(i,
            (i+1)*args.batchsize % dataloader.max['train'],
            dataloader.max['train'],
            int((i+1)*args.batchsize / dataloader.max['train'])),
            'loss {:7.3f} |'.format(loss.data[0]),
            'Bch Train Accu {:.2f}'.format(correct / out.size()[0]))
    perf_ml.go('train/loss', i, loss.data[0])
    perf_ml.go('train/acc', i, correct / out.size()[0])
    barG(correct, out.size()[0], 80)
    barY(loss.data[0], perf_ml['train/loss'][0][1], 80)

    # test
    #if i%args.testevery==0: evaluate(i, net, dataloader)

perf_tm.halt('all')
print('-> Complete. Time elapsed', perf_tm.d['all'])

print('=> Dump Summaries')
perf_tm.dump()
perf_ml.dump()

dataloader.landing()

# Save the model
modelpath = '{}.iter{}.pth'.format(__file__, i)
print('=> Saving model to {}'.format(modelpath))
th.save(net.state_dict(), modelpath) #th.save(net, modelpath)
print('=> Reload and test')
net = Model() if not args.gpu else Model().cuda() #net = th.load(modelpath)
net.load_state_dict(th.load(modelpath))
evaluate('final', net, dataloader)

'''
GPU mem usage: @TitanX Pascal *1
  Resnet18: 3800MiB @batch 256
  Resnet34: 5000MiB @batch 256
  Resnet50: 11100MiB @batch 256
  Resnet101: OOM
  Resnet152: OOM
dataloader:
time:
ref performance:
'''
