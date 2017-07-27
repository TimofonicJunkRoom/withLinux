#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import h5py
import pickle
from sklearn.model_selection import train_test_split

class DataLoader(object):
    def __init__(self):
        self.name = 'CIFAR10'
        self.home = 'http://www.cs.toronto.edu/~kriz/cifar.html'
        self.rawpath = './cifar-10-batches-py/'
        self.h5path = 'cifar10.th.h5'
        print('=> Initializing {} DataLoader ...'.format(self.name))
        if not os.path.exists(self.h5path):
            self.create_hdf5()
        else:
            print('-> using Cached HDF5 ...')
        ### Cache all contents of HDF5 into Memory
        self.f = h5py.File(self.h5path, 'r')
        self.trainval_images = self.f['trainval/images'][:,:]
        self.trainval_labels = self.f['trainval/labels'][:,:]
        self.test_images = self.f['test/images'][:,:]
        self.test_labels = self.f['test/labels'][:,:]
        print(' -> trainval im shape', self.trainval_images.shape)
        print(' -> trainval lb shape', self.trainval_labels.shape)
        print(' -> test  im shape', self.test_images.shape)
        print(' -> test  lb shape', self.test_labels.shape)
        self.maxtrainval = self.trainval_images.shape[0]
        self.maxtest = self.test_images.shape[0]
        # Misc
        self.cur = {'trainval':0, 'test':0}
        self.max = {'trainval':self.maxtrainval, 'test':self.maxtest}
        print('=> Initializing {} DataLoader ... OK'.format(self.name))
    def reset(self, split):
        self.cur[split] = 0
    def inc(self, split):
        self.cur[split] += 1
    def itersInEpoch(self, split, batchsize):
        return int(self.max[split] / batchsize)
    def create_hdf5(self):
        # define helper functions
        def unpickle(fname):
            with open(fname, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
            return d
        def fill(ims, lbs, idx):
            print(' -> Reading blob num.', idx)
            data_batch = unpickle(self.rawpath + 'data_batch_'+str(idx))
            '''
            data_batch -> data is a numpy array
            data_batch -> labels is a list[int]
            '''
            ims[(idx-1)*10000:idx*10000, :] = data_batch[b'data']
            lbs[(idx-1)*10000:idx*10000, :] = \
                    np.array(data_batch[b'labels']).reshape(-1, 1)
            del(data_batch)
        print('=> Creating HDF5 for {} ...'.format(self.name))
        ### Read Train-Val data and split ###
        trainval_images = np.zeros([50000, 3072])
        trainval_labels = np.zeros([50000, 1])
        for i in range(1,6): fill(trainval_images, trainval_labels, i)
        ### Read Test data for validation use
        test_batch = unpickle(self.rawpath + 'test_batch')
        test_images = test_batch[b'data']
        test_labels = np.array(test_batch[b'labels']).reshape(-1, 1)
        ### Write HDF5
        newdata = lambda f,name,data: f.create_dataset(name,
                data=data, compression='gzip', compression_opts=1)
        with h5py.File(self.h5path,'w') as f:
            newdata(f, 'trainval/images', trainval_images)
            newdata(f, 'trainval/labels', trainval_labels)
            newdata(f, 'test/images', test_images)
            newdata(f, 'test/labels', test_labels)
        print('=> Creating HDF5 for {} ... OK'.format(self.name))
    def getBatch(self, split, batchsize):
        '''
        split: str, {trainval, test}
        batchsize: int
        '''
        batchim = np.zeros((batchsize, self.trainval_images.shape[1]))
        batchlb = np.zeros((batchsize, self.trainval_labels.shape[1]))
        batchids = []
        for i in range(batchsize):
            if self.cur[split] >= self.max[split]:
                self.reset(split)
            batchids.append(self.cur[split])
            self.inc(split)
        if split=='trainval':
            batchim[:,:] = self.trainval_images[batchids, :]
            batchlb[:,:] = self.trainval_labels[batchids, :]
        elif split=='test':
            batchim[:,:] = self.test_images[batchids, :]
            batchlb[:,:] = self.test_labels[batchids, :]
        else:
            raise Exception('Unexpected split')
        return batchim, batchlb

if __name__=='__main__':
    # test
    dataloader = DataLoader()
    images, labels = dataloader.getBatch('trainval', 1)
    print(type(images), type(labels))
    print(images, labels)
    for i in range(10000):
        print('get batch iter', i)
        images, labels = dataloader.getBatch('trainval', 64)
    print('test ok')
