#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

class DataLoader(object):
    def __init__(self):
        self.name = 'MNIST'
        self.h5path = 'mnist.th.h5'
        if not os.path.exists(self.h5path):
            self.create_hdf5()
        else:
            print('-> using Cached HDF5 ...')
        print('=> Initializing {} DataLoader ...'.format(self.name))
        ### Cache the whole HDF5 into Memory
        # The HDF5 way is faster than simply reading the whole
        # csv into memory.
        self.f = h5py.File(self.h5path, 'r')
        self.train_images = self.f['train/images'][:,:]
        self.train_labels = self.f['train/labels'][:,:]
        self.val_images = self.f['val/images'][:,:]
        self.val_labels = self.f['val/labels'][:,:]
        self.test_images = self.f['test/images'][:,:]
        print(' -> train im shape', self.train_images.shape)
        print(' -> train lb shape', self.train_labels.shape)
        print(' -> val   im shape', self.val_images.shape)
        print(' -> val   lb shape', self.val_labels.shape)
        print(' -> test  im shape', self.test_images.shape)
        self.maxtrain = self.train_images.shape[0]
        self.maxval = self.val_images.shape[0]
        self.maxtest = self.test_images.shape[0]
        # Misc
        self.cur = {'train':0, 'val':0, 'test':0}
        self.max = {'train':self.maxtrain,
                'val':self.maxval,
                'test':self.maxtest}
        print('=> Initializing {} DataLoader ... OK'.format(self.name))
    def create_hdf5(self):
        print('=> Creating HDF5 for {} ...'.format(self.name))
        ### Read Train-Val data and split ###
        trainval = pd.read_csv("train.csv")
        trainval_images = trainval.iloc[:, 1:]
        trainval_labels = trainval.iloc[:, :1]
        train_images, val_images, \
                train_labels, val_labels = \
                train_test_split(trainval_images, trainval_labels,
                train_size=0.9, random_state=0)
        ### Read Test data ###
        test = pd.read_csv('test.csv')
        test_images = test.iloc[:,:]
        ### Write HDF5
        newdata = lambda f,name,data: f.create_dataset(name,
                data=data, compression='gzip', compression_opts=1)
        with h5py.File(self.h5path,'w') as f:
            newdata(f, 'train/images', train_images)
            newdata(f, 'train/labels', train_labels)
            newdata(f, 'val/images', val_images)
            newdata(f, 'val/labels', val_labels)
            newdata(f, 'test/images', test_images)
        print('=> Creating HDF5 for {} ... OK'.format(self.name))
    def reset(self, split):
        self.cur[split] = 0
    def inc(self, split):
        self.cur[split] += 1
    def itersInEpoch(self, split, batchsize):
        return int(self.max[split] / batchsize)
    def getBatch(self, split, batchsize):
        '''
        split: str, {train, val, test}
        batchsize: int
        '''
        # batchfill, as a workaround for h5py's inability to
        # perform (random) fancy indexing, is TOO SLOW
        #def batchfill(batchim, batchlb, idxs, dataim, datalb):
        #    '''
        #    h5py doesn't support fancy indexing such as
        #    batchim[:,:] = self.train_images[batchids, :]
        #    https://stackoverflow.com/questions/38761878/indexing-a-large-3d-hdf5-dataset-for-subsetting-based-on-2d-condition
        #    https://stackoverflow.com/questions/44959604/subsampling-an-hdf5-file-with-multiple-datasets
        #    '''
        #    for k,idx in enumerate(idxs):
        #        batchim[k,:] = dataim[idx,:]
        #        batchlb[k,:] = datalb[idx,:]
        batchim = np.zeros((batchsize, self.train_images.shape[1]))
        batchlb = np.zeros((batchsize, self.train_labels.shape[1]))
        batchids = []
        for i in range(batchsize):
            if self.cur[split] >= self.max[split]:
                self.reset(split)
            batchids.append(self.cur[split])
            self.inc(split)
        #print(batchids)
        if split=='train':
            #batchfill(batchim, batchlb, batchids,
            #        self.train_images, self.train_labels)
            batchim[:,:] = self.train_images[batchids, :]
            batchlb[:,:] = self.train_labels[batchids, :]
        elif split=='val':
            batchim[:,:] = self.val_images[batchids, :]
            batchlb[:,:] = self.val_labels[batchids, :]
        elif split=='test':
            batchim[:,:] = self.test_images[batchids, :]
        else:
            raise Exception('Unexpected split')
        return batchim, batchlb

if __name__=='__main__':
    # test
    dataloader = DataLoader()
    images, labels = dataloader.getBatch('train', 1)
    print(type(images), type(labels))
    print(images, labels)
    for i in range(10000):
        print('get batch iter', i)
        images, labels = dataloader.getBatch('train', 64)
    print('test ok')
