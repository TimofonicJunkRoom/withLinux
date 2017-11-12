#!/usr/bin/python3
import h5py
import numpy as np
import sys


if len(sys.argv)==1: # demo.h5
    f = h5py.File('demo.h5', 'w')
    f['data'] = np.random.rand(10, 784)
    f['label'] = np.arange(10)
else:
    f = h5py.File('mnist.fake.h5', 'w') # fake mnist for benchmarking
    f['/train/images'] = np.random.rand(37800, 784)*255
    f['/train/labels'] = np.random.rand(37800, 1  )* 10
f.flush()
f.close()
