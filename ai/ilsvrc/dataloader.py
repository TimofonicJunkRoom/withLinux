#!/usr/bin/env python3
from typing import *
import sys
import os
import numpy as np
import pandas as pd
import h5py
import pickle
import random
import xmltodict
import json
from multiprocessing import Process, Queue
from sklearn.model_selection import train_test_split

class DataLoader(object):
    def createJSON(self, basepath:str='/niuzst/imagenet/ILSVRC2015/') -> None:
        ''' Create a JSON file that used by this dataloader
        JSON: Dict[
                "trainset": List[Tuple[str(path), str(label)]],
                "cls2name": Dict[str(class) ->
                                 Tuple[str(classname), str(description)]],
                "cls2label": Dict[str(class) -> int(label)]
              ] where class e.g. "n01440764", label e.g. int(1)
        '''
        print(' -> Creating JSON file ...')
        with open(os.path.join(basepath,
                  'ImageSets/CLS-LOC/train_cls.txt')) as f:
            trainset = [ l.strip().split()[0] for l in f.readlines() ]
            # e.g. trainset[0] ~ "n01440764/n01440764_10026"
        trainset_classes = [ os.path.dirname(x) for x in trainset ]
        # e.g. trainset_labels[0] ~ n01440764
        trainset_paths = [ os.path.join(basepath, 'Data/CLS-LOC/train',
                                  '{}.JPEG'.format(x)) for x in trainset ]
        # load meta data
        from scipy.io import loadmat
        meta_clsloc = loadmat(os.path.join(basepath, 'devkit/data/meta_clsloc.mat'))
        cls2label = { meta_clsloc['synsets'][0][i][1][0] : i # label in [0,999]
                      for i in range(len(meta_clsloc['synsets'][0])) }
        cls2name = { meta_clsloc['synsets'][0][i][1][0] :
                     tuple(meta_clsloc['synsets'][0][i][j][0] for j in (2,3))
                     for i in range(len(meta_clsloc['synsets'][0])) }
        # translate labels
        trainset_labels = [ cls2label[x] for x in trainset_classes ]
        # finalize and write json
        trainset = list(zip(trainset_paths, trainset_labels))
        J = {'trainset':trainset, 'cls2name':cls2name, 'cls2label':cls2label}
        with open(self.jsonpath, 'w+') as f:
            json.dump(J, f)
        print(' -> JSON saved to {}'.format(self.jsonpath))
    #def createHDF5(self): # copying data again wastes toooooo much disk space.
    def __init__(self):
        self.name = 'ILSVRC-2015'
        self.jsonpath = '{}.json'.format(__file__)
        print('=> Initializing {} DataLoader ...'.format(self.name))
        if not os.path.exists(self.jsonpath):
            self.createJSON()
        else:
            print('-> using Cached JSON ...')
        ### Cache all contents of HDF5 into Memory
        with open(self.jsonpath, 'r') as f:
            self.J = json.load(f)
        self.trainset = self.J['trainset']
        #self.valset =
        #self.testset =
        self.cls2name = self.J['cls2name']
        self.cls2label = self.J['cls2label']
        print(' -> train im number', len(self.trainset))
        print(' -> train lb number', len(self.trainset))
        #print(' -> val   im number', self.test_images.shape)
        #print(' -> val   lb number', self.test_labels.shape)
        #print(' -> test  im number', self.test_images.shape)
        #print(' -> test  lb number', self.test_labels.shape)
        self.maxtrain = len(self.trainset)
        self.maxval = None # FIXME
        self.maxtest = None # FIXME
        # Misc
        self.cur = {'train':0, 'val':0, 'test':0}
        self.max = {'train':self.maxtrain, 'val':self.maxval, 'test':self.maxtest}
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
        batchids = []
        for i in range(batchsize):
            if self.cur[split] >= self.max[split]:
                self.reset(split)
            batchids.append(self.cur[split])
            self.inc(split)
        if split=='train':
            # override batchids
            batch = random.sample(self.trainset, batchsize)
            batch_path, batch_lb = list(zip(*batch))
            return batch_path, batch_lb
        elif split=='val':
            raise NotImplementedError
        elif split=='test':
            raise NotImplementedError
        else:
            raise Exception('Unexpected split')
        return batchim, batchlb
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
        images, labels = dataloader.getBatch('train', 32)
        #print(images, labels)
    print('test ok')
    #print('mp test')
    #dataloader.satellite(3, 'trainval', 100)
    #for i in range(100):
    #    print('iter', i, 'getting data')
    #    images, labels = dataloader.Q.get()
    #    print(images.mean(), labels.mean())
    #dataloader.landing()
    #print('mp test ok')
