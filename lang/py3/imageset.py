#!/usr/bin/python3
from PIL import Image
import numpy as np
import h5py
import argparse
import random
import os
from subprocess import call

#from torchvision import transforms
# TODO: read torchvision.transformations

'''
Convert an imageset from caffe list file to hdf5
See also: Caffe/convert_imageset
See also: Torchvision/tranformations
https://caffe2.ai/docs/tutorial-image-pre-processing.html
http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''

## Parse command line
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, action='store', dest='input',
                    required=True, help='image-label list file (train)')
parser.add_argument('--vali', type=str, action='store',
                    dest='vinput', help='validation set list file')
parser.add_argument('--test', type=str, action='store', dest='tinput',
                    help='test set list file')
parser.add_argument('-o', type=str, action='store', dest='output',
                    default=__file__+'.h5', help='output hdf5 path')
parser.add_argument('-p', type=int, action='store', dest='pixels',
                    required=True, help='output image size')
parser.add_argument('-s', action='store_true', dest='s',
                    default=False, help='shuffle the list?')
parser.add_argument('-c', action='store_true', dest='c',
                    default=False, help='compression?')
parser.add_argument('-f', action='store_true', dest='force',
                    default=False, help='force overwrite output hdf5')
parser.add_argument('-v', action='store_true', dest='view',
                    default=False, help='view example image')
args = parser.parse_args()

## Configure
compargs = {'compression':'gzip', 'compression_opts':1} if args.c else {}

## Helpers
def readlist(_fpath):
    # -> list[ list[ str(path), str(label) ] ]
    return [l.strip().split() for l in open(_fpath, 'r').readlines() ]

def fillhdf5(_h5, _list, _group):
    for i, line in enumerate(_list, 1):
        if i%1000==0: print(' *> processed {} images'.format(i))
        path, label = line
        if i < 10: print(repr(path), repr(label))
        image = Image.open(path).resize((args.pixels, args.pixels), Image.BILINEAR)
        image = image.convert('RGB') # RGBA/... -> RGB
        if i == 1 and args.view: image.show()
        if i < 10: print('\t', image)
        # image -> [0,255], H,W,C
        image = np.asarray(image) # Image -> Numpy
        # HWC -> CHW
        image = image.swapaxes(0,2) # or image.transpose((1,2,0))
        _h5[_group+'/images'][i-1,:,:,:] = image
        _h5[_group+'/labels'][i-1,:] = int(label)
        if i == 1 and args.view:
            Image.fromarray(_h5[_group+'/images'][i-1,:,:,:].swapaxes(2,0), mode='RGB').show()

def createdsets(_h5, _list, _impath, _lbpath):
    h5.create_dataset(_impath, # N x C x H x W, np.ubyte (not np.byte! that will cause problem)
        (len(_list), 3, args.pixels, args.pixels), dtype=np.ubyte, **compargs)
    h5.create_dataset(_lbpath, # N x 1, int
        (len(_list), 1), dtype=np.int, **compargs)

# Read list files
imagelist = readlist(args.input)
if args.s: random.shuffle(imagelist)
print('-> Found {} images for training'.format(len(imagelist)))
if args.vinput:
    imagelist_vali = readlist(args.vinput)
    print('-> Found {} images for validation'.format(len(imagelist_vali)))
if args.tinput:
    imagelist_test = readlist(args.tinput)
    print('-> Found {} images for test'.format(len(imagelist_test)))

# Create output file
if os.path.exists(args.output):
    if not args.force: raise SystemExit('HDF5 file {} already exists!'.format(args.output))
h5 = h5py.File(args.output, 'w')

# Fill HDF5
createdsets(h5, imagelist, 'train/images', 'train/labels')
fillhdf5(h5, imagelist, 'train')
print(' *> processed {} images for training'.format(len(imagelist)))
if args.vinput:
    createdsets(h5, imagelist_vali, 'val/images', 'val/labels')
    fillhdf5(h5, imagelist_vali, 'val')
    print(' *> processed {} images for validation'.format(len(imagelist_vali)))
if args.tinput:
    createdsets(h5, imagelist_test, 'test/images', 'test/labels')
    fillhdf5(h5, imagelist_test, 'test')
    print(' *> processed {} images for validation'.format(len(imagelist_test)))

# Write to disk
h5.close()
print('-> Dataset saved as {}'.format(args.output))
call(['h5ls', '-rv', args.output ])
