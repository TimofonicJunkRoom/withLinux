#!/usr/bin/env python3
'''
Informal repro of cv16-orcnn: MR-CNN
'''
import sys
import os
import time
from collections import OrderedDict, deque
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

### CONFIGURE ###
argparser = argparse.ArgumentParser()
argparser.add_argument('-g', '--gpu', action='store_true',
                       help='use GPU/CUDA insteaf of CPU')
argparser.add_argument('-d', '--double', action='store_true',
                       help='use fp64 instead of fp32')
argparser.add_argument('-m', '--maxiter', type=int, default=24000,
                       help='set maximum iterations of training',)
argparser.add_argument('-s', '--seed', type=int, default=1,
                       help='set manual seed')
argparser.add_argument('-n', '--numthreads', type=int, default=4,
                       help='set *_NUM_THREADS environment variable')
argparser.add_argument('-t', '--testevery', type=int, default=500,
                       help='set model evaluation interval')
argparser.add_argument('-o', '--decay0', type=int, default=975,
                       help='set the first iteration where the learning rate starts to decay')
argparser.add_argument('-T', '--decayT', type=int, default=1950,
                       help='set the learning rate decay period')
argparser.add_argument('-e', '--lr', type=float, default=1e-4,
                       help='set the initial learning rate')
argparser.add_argument('-b', '--batchsize', type=int, default=128,
                       help='set batch size for training')
argparser.add_argument('--testbatchsize', type=int, default=100,
                       help='set batch size for test')
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
        for k,v in self['test/loss']:
            barR(v, self['test/loss'][0][1], 100)
        for k,v in self['test/mae']:
            barC(v, self['test/mae'][0][1], 100)
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

### DataLoader ###
dataloader = DataLoader()

### Model ###
class Model(th.nn.Module):
    '''
    60x60 input image cropped from 64x64 image
    batchsize is 128
    label should be shifted by -16
    '''
    def __init__(self):
        super(Model, self).__init__()
        self.SEQ1 = th.nn.Sequential(OrderedDict([
          # 128x3x60x60
          ('conv1', th.nn.Conv2d(3, 20, 5, stride=1, padding=0)),
          # 128x20x56x56
#          ('bn1',   th.nn.BatchNorm2d(20)),
          ('relu1', th.nn.ReLU()),
          ('norm1', th.nn.CrossMapLRN2d(9, alpha=0.001, beta=0.75)),
          ('pool1', th.nn.MaxPool2d(3, stride=2, padding=1)),
          # 128x20x28x28

          ('conv2', th.nn.Conv2d(20, 40, 7, stride=1, padding=0)),
          # 128x40x22x22
#          ('bn2',   th.nn.BatchNorm2d(40)),
          ('relu2', th.nn.ReLU()),
          ('norm2', th.nn.CrossMapLRN2d(9, alpha=0.001, beta=0.75)),
          ('pool2', th.nn.MaxPool2d(3, stride=2, padding=1)),
          # 128x40x11x11
           
          ('conv3', th.nn.Conv2d(40, 80, 11, stride=1, padding=0)),
          # 128x80x1x1
#          ('bn3',   th.nn.BatchNorm2d(80)),
          ('relu3', th.nn.ReLU()),
          ('norm2', th.nn.CrossMapLRN2d(9, alpha=0.001, beta=0.75)),

        ]))
        th.nn.init.normal(self.SEQ1.conv1.weight, mean=0, std=0.01)
        th.nn.init.constant(self.SEQ1.conv1.bias, 0.05)
        th.nn.init.normal(self.SEQ1.conv2.weight, mean=0, std=0.01)
        th.nn.init.constant(self.SEQ1.conv2.bias, 0.05)
        th.nn.init.normal(self.SEQ1.conv3.weight, mean=0, std=0.01)
        th.nn.init.constant(self.SEQ1.conv3.bias, 0.05)

        self.SEQ2 = th.nn.Sequential(OrderedDict([
          # 128x80
          ('fc4',   th.nn.Linear(80, 80)),
          # 128x80
#          ('bn4',   th.nn.BatchNorm1d(80)),
          ('relu4', th.nn.ReLU()),
          
#          ('drop5', th.nn.Dropout(0.2)),
          ('fc5',   th.nn.Linear(80, 1)),
          # 128x1
        ]))
        th.nn.init.normal(self.SEQ2.fc4.weight, mean=0, std=0.005)
        th.nn.init.constant(self.SEQ2.fc4.bias, 0.05)
        th.nn.init.normal(self.SEQ2.fc5.weight, mean=0, std=0.005)
        th.nn.init.constant(self.SEQ2.fc5.bias, 0.05)

        self.DUMPSHAPE = False
    def forward(self, x):
        if not self.DUMPSHAPE:
            def psize(size):
                msg = ''
                for x in size: msg += '{} '.format(x)
                prod = 1
                for x in size[1:]: prod *= x
                msg += '({} x {})'.format(size[0], prod)
                return msg
            def dumpshape(module, x):
                for name, m in list(module.named_modules())[1:]:
                    print('*> pre ', name.ljust(10),
                          'SHAPE', psize(x.size()))
                    x = m(x)
                    print('*> post', name.ljust(10),
                          'SHAPE', psize(x.size()))
                return x
            x = dumpshape(self.SEQ1, x)
            x = x.view(-1, 80)
            x = dumpshape(self.SEQ2, x)
            self.DUMPSHAPE = True
        else:
            x = self.SEQ1(x)
            x = x.view(-1, 80)
            x = self.SEQ2(x)
        return x

### Create Instances
net = Model() if not args.gpu else Model().cuda()
if not args.double: net = net.float()
print(net)
crit = th.nn.MSELoss() # Caffe:EuclideanLoss
optimizer = th.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

### Data Transformation
image_mean = np.load('morph2.mean64x64.npz')['arr_0']
image_mean[[0,2],:,:] = image_mean[[2,0],:,:]
def transform(images, labels, evaluate=False):
    # cropping
    if evaluate==True:
        xoff, yoff = 2, 2
    else:
        xoff, yoff = random.randint(0,4), random.randint(0,4)
    # mean, scale and shift
    images = (images - image_mean)[:,:,yoff:yoff+60, xoff:xoff+60]/ 255.
    labels = labels - 16.
    # mirroring while training
    if evaluate==False and random.choice((True,False)):
        images = np.flip(images, 3)
    images = Variable(th.from_numpy(images.astype(np.double)), requires_grad=False)
    labels = Variable(th.from_numpy(labels.reshape(-1, 1).astype(np.double)), requires_grad=False)
    if args.gpu: images, labels = images.cuda(), labels.cuda()
    if not args.double: images, labels = images.float(), labels.float()
    return images, labels

#    # -1x3x64x64
#    #images = images - image_mean
#    images_np = np.zeros((images.shape[0], 3, 60, 60))
#    if not evaluate:
#        for i in range(images.shape[0]):
#            # Do random cropping
#            im = Image.fromarray(images[i,:,:,:].reshape(3,64,64).swapaxes(0,2), 'RGB')
#            xoff, yoff = random.randint(0,4), random.randint(0,4)
#            im = np.asarray(im.crop((xoff, yoff, xoff+60, yoff+60)))
#            images_np[i,:,:,:] = im.swapaxes(0,2)
#        if random.choice((True,False)):
#            images_np = np.flip(images_np, 3)
#        #print('---------------------', images_np.shape)
#    else:
#        for i in range(images.shape[0]):
#            im = Image.fromarray(images[i,:,:,:].reshape(3,64,64).swapaxes(0,2), 'RGB')
#            im = np.asarray(im.resize((60,60), Image.BILINEAR))
#            images_np[i,:,:,:] = im.swapaxes(0,2)
#        #print('---------------------', images_np.shape)
#
#    images = images_np.reshape(-1, 3, 60, 60) #/ 255.
#    labels = labels - 16.
#    images = Variable(th.from_numpy(images.astype(np.double)), requires_grad=False)
#    labels = Variable(th.from_numpy(labels.reshape(-1, 1).astype(np.double)), requires_grad=False)
#    if args.gpu: images, labels = images.cuda(), labels.cuda()
#    if not args.double: images, labels = images.float(), labels.float()
#    return images, labels

### Evaluation on Validation set
def evaluate(i, net, dataloader):
    print('-> EVAL @ {} |'.format(i), end='')
    net.eval()
    maeaccum = 0
    lossaccum = 0
    total = 0
    dataloader.reset('val')
    for j in range(dataloader.itersInEpoch('val', args.testbatchsize)):
        images, labels = dataloader.getBatch('val', args.testbatchsize)
        images, labels = transform(images, labels, True)
        out = net(images)
        loss = crit(out, labels)

        maeaccum += th.Tensor(out.data).add(-labels.data).abs().sum()
        lossaccum += loss.data[0]
        total += args.testbatchsize
        print('.', end=''); sys.stdout.flush()

    mae = maeaccum / total
    loss = lossaccum / total
    print('|')
    print('-> TEST @ {} |'.format(i),
            'Loss {:7.3f} |'.format(loss),
            'MAE {:.5f}|'.format(mae))
    perf_ml.go('test/mae', i, mae)
    perf_ml.go('test/loss', i, lossaccum / total)
    barC(mae, perf_ml['test/mae'][0][1], 80)
    #barR(lossaccum, perf_ml['test/loss'][0][1], 80)

### Training
perf_tm.go('all')
smthmae = deque([], 17) # ~5% of trainset
for i in range(args.maxiter+1):
    # read data
    perf_tm.go('data/fetch')
    images, labels = dataloader.getBatch('train', args.batchsize)
    #images, labels = dataloader.Q.get()
    images, labels = transform(images, labels)
    perf_tm.halt('data/fetch')

    # decay the learning rate
    # inv: return base_lr * (1 + gamma * iter) ^ (- power)
    curlr = args.lr * (1 + 1e-4 * i)**(- 0.75)
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

    mae = th.Tensor(out.data).add(-labels.data).abs().sum() / args.batchsize
    smthmae.append(mae)

    print('-> Iter {:5d} ({:<5d}/{:5d} Eph {:>3d} ) |'.format(i,
            (i+1)*args.batchsize % dataloader.max['train'],
            dataloader.max['train'],
            int((i+1)*args.batchsize / dataloader.max['train'])),
            'loss {:7.3f} |'.format(loss.data[0]),
            'Batch MAE {:.2f}'.format(mae),
            '(Smth {:.2f})'.format(sum(smthmae)/len(smthmae)))
    perf_ml.go('train/loss', i, loss.data[0])
    perf_ml.go('train/mae', i, mae)
    barG(mae, perf_ml['train/mae'][0][1], 80)
    barY(loss.data[0], perf_ml['train/loss'][0][1], 80)

    # test
    if i%args.testevery==0: evaluate(i, net, dataloader)

perf_tm.halt('all')
print('-> Complete. Time elapsed', perf_tm.d['all'])

print('=> Dump Summaries')
perf_tm.dump()
perf_ml.dump()
