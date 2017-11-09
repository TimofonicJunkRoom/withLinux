#!/usr/bin/python3
import h5py
import numpy as np

f = h5py.File('demo.h5', 'w')
f['data'] = np.random.rand(10, 784)
f['label'] = np.arange(10)
f.flush()
f.close()
