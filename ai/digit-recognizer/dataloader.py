#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader(object):
    def __init__(self):
        self.name = 'MNIST'
        print('=> Initializing {} DataLoader ...'.format(self.name))
        ### Read Train-Val data and split ###
        trainval = pd.read_csv("train.csv")
        trainval_images = trainval.iloc[:, 1:]
        trainval_labels = trainval.iloc[:, :1]
        self.train_images, self.val_images, \
                self.train_labels, self.val_labels = \
                train_test_split(trainval_images, trainval_labels,
                train_size=0.9, random_state=0)
        print(' -> train im shape', self.train_images.shape)
        print(' -> train lb shape', self.train_labels.shape)
        print(' -> val   im shape', self.val_images.shape)
        print(' -> val   lb shape', self.val_labels.shape)
        self.maxtrain = self.train_images.shape[0]
        self.maxval = self.val_images.shape[0]
        ### Read Test data ###
        test = pd.read_csv('test.csv')
        self.test_images = test.iloc[:,:]
        print(' -> test  im shape', self.test_images.shape)
        self.maxtest = self.test_images.shape[0]
        ### Misc
        self.cur = {'train':0, 'val':0, 'test':0}
        self.max = {'train':self.maxtrain,
                'val':self.maxval,
                'test':self.maxtest}
        print('=> Initializing {} DataLoader ... OK'.format(self.name))
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
            batchim[:,:] = self.train_images.values[batchids, :]
            batchlb[:,:] = self.train_labels.values[batchids, :]
        elif split=='val':
            batchim[:,:] = self.val_images.values[batchids, :]
            batchlb[:,:] = self.val_labels.values[batchids, :]
        elif split=='test':
            batchim[:,:] = self.test_images.values[batchids, :]
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
        images, labels = dataloader.getBatch('train', 64)
