#!/usr/bin/python3

import os

import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
print('-> Using CAFFE', caffe.__version__)

caffe.set_mode_cpu()

### Read Train-Val data and split ###
trainval = pd.read_csv("train.csv")
trainval_images = trainval.iloc[:, 1:].div(255)
trainval_labels = trainval.iloc[:, :1]
train_images, val_images, train_labels, val_labels = train_test_split(
        trainval_images, trainval_labels, train_size=0.8, random_state=0)
print('-> train set shape', train_images.shape)
print('-> val   set shape', val_images.shape)

### Read Test data ###
test = pd.read_csv('test.csv')
test_images = test.iloc[:,:].div(255)
print('-> test  set shape', test_images.shape)

### Create HDF5
if not os.path.exists('mnist.train.h5') and \
        not os.path.exists('mnist.val.h5'):
    ### Write HDF5 files
    print('-> writing hdf5 files')
    with h5py.File("mnist.train.h5", "w") as f:
        f.create_dataset('data', data=train_images,
                compression='gzip', compression_opts=1)
        f.create_dataset('label', data=train_labels.astype(np.float32),
                compression='gzip', compression_opts=1)
    open('mnist.train.h5.txt', 'w').write('mnist.train.h5\n')
    with h5py.File("mnist.val.h5", "w") as f:
        f.create_dataset('data', data=val_images,
                compression='gzip', compression_opts=1)
        f.create_dataset('label', data=val_labels.astype(np.float32),
                compression='gzip', compression_opts=1)
    open('mnist.val.h5.txt', 'w').write('mnist.val.h5\n')

### Network helper
print('-> creating network')
def model(h5path, batchsize):
    n = caffe.NetSpec()
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(source=h5path, batch_size=batchsize, ntop=2)
    n.ip1 = L.InnerProduct(n.data, num_output=10,
            weight_filler=dict(type='xavier'))
    n.accuracy = L.Accuracy(n.ip1, n.label)
    n.loss = L.SoftmaxWithLoss(n.ip1, n.label)
    return n.to_proto()

model_train_path = "model.cf.train.prototxt"
open(model_train_path, 'w').write(str(model('mnist.train.h5.txt', 64)))
model_val_path = 'model.cf.val.prototxt'
open(model_val_path, 'w').write(str(model('mnist.val.h5.txt', 100)))

### Solver helper
print('-> creating solver')
def solver(train_net_path, test_net_path):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path
    s.test_net.append(test_net_path)
    s.test_interval = 1000
    s.test_iter.append(250)
    s.max_iter = 10000
    s.base_lr = 0.01
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 5000
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.display = 1000
    s.snapshot = 10000
    s.snapshot_prefix = 'model.cf'
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    return s

solver_path = 'model.cf.solver.prototxt'
open(solver_path, 'w').write(str(solver(model_train_path, model_val_path)))

### Get solver and train!
solver = caffe.get_solver(solver_path)
solver.solve()

print('-> final accuracy', solver.test_nets[0].blobs['accuracy'].data)

# each output is (batch size, feature dim, spatial dim)
print([(k, v.data.shape) for k, v in solver.net.blobs.items()])
# just print the weight sizes (we'll omit the biases)
print([(k, v[0].data.shape) for k, v in solver.net.params.items()])
