#!/usr/bin/python3
# > http://docs.h5py.org/en/latest/quick.html

import logging as log
log.basicConfig(
  format='\x1b[36;1mL%(asctime)s %(process)d %(filename)s:%(lineno)d]'
    +' %(message)s\x1b[m',
  datefmt='%m%d %I:%M:%S',
  level=log.DEBUG
)

import h5py
import numpy as np

log.info('generate src hdf5 file')
f = h5py.File ('junk.hdf5', 'w')

log.info('create data')
t_cnn   = np.random.random((16,1024))
t_lstm  = np.random.random((16,1024))
t_tree  = np.random.random((16,100))
t_label = np.random.random((16,100))

log.info('write data')
for i in range(1,20):
  f['/'+str(i)+'/cnn_embed'] = t_cnn
  f['/'+str(i)+'/lstm_embed'] = t_lstm
  f['/'+str(i)+'/tree_idx'] = t_tree
  f['/'+str(i)+'/label_idx'] = t_label

f.close()
log.info('done')
