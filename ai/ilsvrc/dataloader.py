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
import math
import time
from PIL import Image, ImageOps
from functools import partial
from multiprocessing import Process, Queue, Pool
from sklearn.model_selection import train_test_split

class DataLoader(object):
    '''
    ref: torchvision
    '''
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
        self.P = Pool(128) # thread pool for processing images
        print('=> Initializing {} DataLoader ... OK'.format(self.name))
    def reset(self, split):
        self.cur[split] = 0
    def inc(self, split):
        self.cur[split] += 1
    def itersInEpoch(self, split, batchsize):
        return int(self.max[split] / batchsize)
    def getBatch(self, split, batchsize, tpool=None):
        '''
        split: str, {train, val, test}
        batchsize: int
        return ndarray (bsize, 3, 224, 224), array(bsize)
        '''
        batchim = np.zeros((batchsize,3,224,224), dtype=np.uint8)
        batchlb = np.zeros((batchsize), dtype=np.uint8)
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
            batchlb[:] = batch_lb
            # batch_images = list(map(pipeapply_train, batch_path))
            if not tpool:
                batch_images = list(self.P.map(pipeapply_train,
                                               batch_path))
            else:
                batch_images = list(tpool.map(pipeapply_train,
                                              batch_path))
            for i,v in enumerate(batch_images):
                batchim[i] = batch_images[i]
            return batchim, batchlb
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
            with Pool(128) as workerthreadpool:
                while True:
                    #print(' *> {} : satellite putting data in qbuf'.format(os.getpid()))
                    try:
                        dataloader.Q.put(dataloader.getBatch(split,
                                 batchsize, workerthreadpool))
                    except:
                        workerthreadpool.join()
        self.worker = Process(target=_background,
                              args=(self, split, batchsize))
        self.worker.start()
    def landing(self):
        ''' the satellite is landing '''
        self.worker.join(timeout=0.1)
        print(' *> {} : pulling satellite to ground ...'.format(os.getpid()))
        self.worker.terminate()


# -- START a bunch of image transformation helpers --
# -- Reference: torchvision:transforms:*
def imloader(path:str) -> Image.Image:
    ''' load PIL.Image by path string '''
    return Image.open(path).convert('RGB')
def imexpander(im:Image.Image) -> Image.Image:
    ''' padding the image so that the shortest side is
    greater or equal to 224 pixels '''
    if all((im.height>=224, im.width>=224)):
        return im
    else:
        return ImageOps.expand(im,
          math.ceil((224 - min((im.height, im.width)))/2),
          (127,127,127))
def imresizer(im:Image.Image) -> Image.Image:
    ''' resize the given image to (224,224) '''
    return im if all((im.height==224, im.width==224)) else \
           im.resize((224, 224), Image.BILINEAR)
def imhorizontalflip(im:Image.Image) -> Image.Image:
    ''' randomly mirror the given image '''
    return im if random.choice((True,False)) else \
           im.transpose(Image.FLIP_LEFT_RIGHT)
def imrandomcropper(im:Image.Image) -> Image.Image:
    ''' randomly crop a region of ratio 1:1 from the given im '''
    if all((im.height==224, im.width==224)):
        return im
    else:
        yoff = random.randint(0, im.height - 224)
        xoff = random.randint(0, im.width - 224)
        # (left, upper, right, lower)
        return im.crop((xoff, yoff, xoff+224, yoff+224))
def imcentercropper(im:Image.Image) -> Image.Image:
    ''' crop the center part of the given image '''
    if all((im.height==224, im.width==224)):
        return im
    else:
        yoff = int(round((im.height - 224)/2.))
        xoff = int(round((im.width - 224)/2.))
        return im.crop((xoff, yoff, xoff+224, yoff+224))
def imrandomsizedcropper(im:Image.Image) -> Image.Image:
    ''' (0.08 to 1.0) of the original area
        ratio of (3/4 to 4/3)
        popular for training inception networks.
    @ref torchvision::transforms::RandomSizedCrop '''
    raise NotImplementedError
def im2np(im:Image.Image) -> np.ndarray:
    ''' convert (HxWxC) PIL.Image instance into (CxHxW) numpy
        ndarray of uint8 in range[0,255] '''
    return np.array(im, copy=False).swapaxes(0,2)
def pipeapply(pipe:List[Callable[[Any], Any]], img:Any):
    for f in pipe:
        img = f(img)
    return img
# pipelines for training and validation image processing
#trainpipe = [ imloader, imexpander, imhorizontalflip, imrandomcropper, im2np ]
#valpipe = [ imloader, imexpander, imhorizontalflip, imcentercropper, im2np ]
trainpipe = [imloader, imresizer, imhorizontalflip, im2np]
valpipe = [imloader, imresizer, imhorizontalflip, im2np]
pipeapply_train = partial(pipeapply, trainpipe)
pipeapply_val   = partial(pipeapply, valpipe)
# -- END a bunch of image transformation helpers --


if __name__=='__main__':
    # test
    dataloader = DataLoader()
    TEST_BATCH = 32

    # load 1 sample
    images, labels = dataloader.getBatch('train', 1)
    print(type(images), type(labels))
    print(images.shape, labels.shape)
    print(images, labels)

    # load a batch
    images, labels = dataloader.getBatch('train', TEST_BATCH)
    print(type(images), type(labels))
    print(images.shape, labels.shape)
    print(images, labels)

    # constantly loading new batches
    for i in range(10):
        print('get batch iter', i)
        images, labels = dataloader.getBatch('train', TEST_BATCH)
        #print(images, labels)
    print('test ok')

    # FIXME: BrokenPipeError: [Errno 32] Broken pipe
    print('mp test')
    dataloader.satellite(50, 'train', TEST_BATCH) # queue64, batch32
    for i in range(10):
        print('iter', i, 'getting data')
        images, labels = dataloader.Q.get()
        print(images.mean(), labels.mean())
    dataloader.landing()
    print('mp test ok')

    # simulation : serial : 14 seconds.
    tm_start = time.time()
    for i in range(10):
        print(' *> serial iter', i)
        images, labels = dataloader.getBatch('train', TEST_BATCH)
        time.sleep(0.5)
    print('serial time elapsed', time.time() - tm_start)
    # simulation : background worker : 10 seconds.
    dataloader.satellite(50, 'train', TEST_BATCH)
    tm_start = time.time()
    for i in range(10):
        print(' *> worker', i)
        images, labels = dataloader.Q.get()
        time.sleep(0.5)
    print('workder time elapsed', time.time() - tm_start)
    dataloader.landing()
    print('-> simulation complete')
