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
for i in range(1,2):
  f['/'+str(i)+'/cnn_embed'] = t_cnn
  f['/'+str(i)+'/lstm_embed'] = t_lstm
  f['/'+str(i)+'/tree_idx'] = t_tree
  f['/'+str(i)+'/label_idx'] = t_label

log.info('write string')
f['/strings/1'] = bytes('write a string to hdf5'.encode("utf8"))
f['/strings/2'] = 'another string into hdf5'

f.close()
log.info('done')

'''
Inspecting the resulting hdf5, and repack hdf5 into compressed format.

    h5ls -r junk.h5
    h5ls -rfv junk.h5
    h5dump -g /strings junk.h5
    h5dump -d /strings/1 junk.h5
    h5stat junk.h5
    h5repack -i junk.hdf5 -o junk.repack.h5 -f GZIP=9 -v

Further reading:

    http://cyrille.rossant.net/moving-away-hdf5/
    http://cyrille.rossant.net/should-you-use-hdf5/
'''
