import sys
import os
import time
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from dataloader import DataLoader
from collections import OrderedDict

perf_tm = {}
perf_getdiff = lambda d: d['end'] - d['start']

X_THNUM = '4'
os.putenv('OPENBLAS_NUM_THREADS', X_THNUM)
os.putenv('OMP_NUM_THREADS', X_THNUM)
os.putenv('MKL_NUM_THREADS', X_THNUM)

USE_GPU = False if len(sys.argv)>1 else True
X_MAXITER = 5000

print('-> Using TH', th.__version__)
print('-> USE_GPU: {}'.format(USE_GPU))

th.manual_seed(1)
if not USE_GPU:
    th.set_default_tensor_type('torch.DoubleTensor')
else:
    th.set_default_tensor_type('torch.cuda.DoubleTensor')
    th.cuda.manual_seed(1)

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

net = QuickNet() if not USE_GPU else QuickNet().cuda()
print(net)
crit = th.nn.CrossEntropyLoss()
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)

### Train
def transform(images, labels):
    images = images.reshape(-1, 3, 32, 32) / 255.
    images = Variable(th.from_numpy(images.astype(np.double)), requires_grad=False)
    labels = Variable(th.from_numpy(labels.reshape(-1).astype(np.long)), requires_grad=False)
    if USE_GPU: images, labels = images.cuda(), labels.cuda() 
    return images, labels

perf_tm['start'] = time.time()
for i in range(X_MAXITER+1):
    # read data
    images, labels = dataloader.getBatch('trainval', 100)
    images, labels = transform(images, labels)

    # half the learning rate @ iter 500
    if i!=0 and i%1000==0:
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

    # val
    if i%100==0:
        print('-> TEST @ {} |'.format(i), end='')
        net.eval()
        correct = 0
        total = 0
        lossaccum = 0
        dataloader.reset('val')
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

perf_tm['end'] = time.time()
print('-> done, time elapsed', perf_getdiff(perf_tm))
