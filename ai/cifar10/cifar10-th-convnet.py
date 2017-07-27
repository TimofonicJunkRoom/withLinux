import sys
import os
import time
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from dataloader import DataLoader

perf_tm = {}
perf_getdiff = lambda d: d['end'] - d['start']

X_THNUM = '4'
os.putenv('OPENBLAS_NUM_THREADS', X_THNUM)
os.putenv('OMP_NUM_THREADS', X_THNUM)
os.putenv('MKL_NUM_THREADS', X_THNUM)

USE_GPU = True if len(sys.argv)>1 else False # Append any argument to command line to toggle GPU mode

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
    ''' Reference: caffe/examples/cifar10
    data (-1,3,32,32) -> conv1 (-1,32,32,32) ->
    pool1 (-1,32,16,16) -> relu1 (.) ->
    conv2 (-1,32,16,16) -> relu2 (.) ->
    pool2 (-1,32,8,8) -> conv3 (-1,64,8,8) ->
    relu3 (.) -> pool3 (-1,64,4,4) (1024) ->
    ip1 (-1, 64) -> ip2 (-1,10) ->
    softmaxwithloss
    '''
    def __init__(self):
        super(QuickNet, self).__init__()
        self.conv1 = th.nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.conv2 = th.nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv3 = th.nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.fc1 = th.nn.Linear(1024, 64)
        self.fc2 = th.nn.Linear(64, 10)
    def forward(self, x):
        #q = lambda msg: print(msg, x.size())
        q = lambda msg: True
        q('data')
        x = self.conv1(x) # 3,32,32 -> 32,32,32
        q('conv1')
        x = F.max_pool2d(x, 3, stride=2, padding=1) # 32,16,16
        q('pool1')
        x = F.relu(x)
        q('relu')
        x = self.conv2(x) # 32,16,16
        q('conv2')
        x = F.relu(x)
        q('relu')
        x = F.max_pool2d(x, 3, stride=2, padding=1) # 32,8,8
        q('pool2')
        x = self.conv3(x) # 64,8,8
        q('conv3')
        x = F.relu(x)
        q('relu')
        x = F.max_pool2d(x, 3, stride=2, padding=1) # 64,4,4
        q('pool')
        x = x.view(-1, 1024)
        x = self.fc1(x)
        q('fc1')
        x = self.fc2(x)
        q('fc2')
        return x

net = QuickNet() if not USE_GPU else QuickNet().cuda()
print(net)
crit = th.nn.CrossEntropyLoss()
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)

### Train
def transform(images, labels):
    images = images.reshape(-1, 3, 32, 32) / 255.
    images = Variable(th.from_numpy(images.astype(np.double)), requires_grad=False)
    labels = Variable(th.from_numpy(labels.reshape(-1).astype(np.long)), requires_grad=False)
    if USE_GPU: images, labels = images.cuda(), labels.cuda() 
    return images, labels

#for i in range(100+1):
perf_tm['start'] = time.time()
for i in range(500+1):
    # read data
    images, labels = dataloader.getBatch('trainval', 100)
    images, labels = transform(images, labels)

    # half the learning rate @ iter 500
    if i==500 or i==1000 or i==2000:
        print('-> half the learning rate')
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
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
