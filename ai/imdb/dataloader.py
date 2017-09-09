#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import h5py
import pickle
import random
from multiprocessing import Process, Queue
from sklearn.model_selection import train_test_split

class DataLoader(object):
    def __init__(self):
        self.name = 'Stanford IMDB'
        print('=> Initializing {} DataLoader ...'.format(self.name))
        ### Cache all data into Memory
        self.f = np.load('./imdb.npz')
        self.train_sents  = self.f['x_train'] # List[List[int]] 25000
        self.train_labels = self.f['y_train'] # Array[int]
        self.val_sents    = self.f['x_test']  # List[List[int]] 25000
        self.val_labels   = self.f['y_test']  # Array[int]

        print(' -> train st number', len(self.train_sents))
        print(' -> train lb number', len(self.train_labels))
        print(' -> val   st number', len(self.val_sents))
        print(' -> val   lb number', len(self.val_labels))
        self.maxtrain = len(self.train_sents)
        self.maxval = len(self.val_sents) 
        # Misc
        self.cur = {'train':0, 'val':0}
        self.max = {'train':self.maxtrain, 'val':self.maxval}
        print('=> Initializing {} DataLoader ... OK'.format(self.name))
    def reset(self, split):
        self.cur[split] = 0
    def inc(self, split):
        self.cur[split] += 1
    def itersInEpoch(self, split, batchsize):
        return int(self.max[split] / batchsize)
    def getBatch(self, split, batchsize):
        '''
        split: str, {train, val}
        batchsize: int
        '''
        batchids = []
        for i in range(batchsize):
            if self.cur[split] >= self.max[split]:
                self.reset(split)
            batchids.append(self.cur[split])
            self.inc(split)
        if split=='train':
            batchids = random.sample(range(self.maxtrain), batchsize)
            batchst = [ self.train_sents[i] for i in batchids ]
            batchlb = [ self.train_labels[i] for i in batchids ]
        elif split=='val':
            batchst = [ self.train_sents[i] for i in batchids ]
            batchlb = [ self.train_labels[i] for i in batchids ]
        else:
            raise Exception('Unexpected split')
        return batchst, batchlb
    def satellite(self, qbufsize, split, batchsize):
        '''
        Fork a worker and prefetch data,
        qbuf is a Queue used as data prefetching buffer
        '''
        self.Q = Queue(qbufsize)
        def _background(dataloader, split, batchsize):
            while True:
                #print(' *> {} : satellite putting data in qbuf'.format(os.getpid()))
                dataloader.Q.put(dataloader.getBatch(split, batchsize))
        self.worker = Process(target=_background,
                              args=(self, split, batchsize))
        self.worker.start()
    def landing(self):
        ''' the satellite is landing '''
        self.worker.join(timeout=0.1)
        print(' *> {} : pulling satellite to ground ...'.format(os.getpid()))
        self.worker.terminate()

if __name__=='__main__':
    # test
    dataloader = DataLoader()
    images, labels = dataloader.getBatch('train', 1)
    print(type(images), type(labels))
    print(images, labels)
    for i in range(10000):
        print('get batch iter', i)
        images, labels = dataloader.getBatch('train', 64)
        images, labels = dataloader.getBatch('val', 64)
    print('test ok')
    #print('mp test')
    #dataloader.satellite(3, 'trainval', 100)
    #for i in range(100):
    #    print('iter', i, 'getting data')
    #    images, labels = dataloader.Q.get()
    #    print(images.mean(), labels.mean())
    #dataloader.landing()
    #print('mp test ok')
