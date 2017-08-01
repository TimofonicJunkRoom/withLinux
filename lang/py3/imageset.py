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

# Parse command line
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, action='store', dest='input',
                    required=True, help='image-label list file (train)')
parser.add_argument('--vali', type=str, action='store',
                    dest='vinput', help='validation set list file')
parser.add_argument('--test', type=str, action='store', dest='test',
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
args = parser.parse_args()

# Read image list
# -> list[ list[ str(path), str(label) ] ]
imagelist = [l.strip().split() for l in open(args.input, 'r').readlines() ]
if args.s: random.shuffle(imagelist)
print('-> Found {} images'.format(len(imagelist)))

# Create output file
if os.path.exists(args.output):
    if not args.force: raise SystemExit('HDF5 file {} already exists!'.format(args.output))
h5 = h5py.File(args.output, 'w')
if args.c:
    h5.create_dataset('train/images',
        (len(imagelist), 3, args.pixels, args.pixels), # N,C,H,W
        dtype=np.byte, compression='gzip', compression_opts=1)
    h5.create_dataset('train/labels',
        (len(imagelist), 1), # N,1
        dtype=np.int, compression='gzip', compression_opts=1)
else:
    h5.create_dataset('train/images',
        (len(imagelist), 3, args.pixels, args.pixels), # N,C,H,W
        dtype=np.byte)
    h5.create_dataset('train/labels',
        (len(imagelist), 1), dtype=np.int)

# Traverse the image list
for i, line in enumerate(imagelist, 1):
    if i%1000==0: print(' *> processed {} images'.format(i))
    path, label = line
    if i < 10: print(repr(path), repr(label))
    image = Image.open(path).resize((args.pixels, args.pixels), Image.BILINEAR).convert('RGB')
    if i < 10: print('\t', image)
    # image -> [0,255], H,W,C
    image = np.asarray(image) # Image -> Numpy
    # HWC -> CHW
    image = image.swapaxes(0,2) # or image.transpose((1,2,0))
    h5['train/images'][i-1,:,:,:] = image
    h5['train/labels'][i-1,:] = int(label)
print(' *> processed {} images in total'.format(len(imagelist)))

# Write to disk
h5.close()
print('-> Dataset saved as {}'.format(args.output))
call(['h5ls', '-rv', args.output ])
