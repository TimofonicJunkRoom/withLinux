import h5py
import numpy as np

f = h5py.File('junk.h5', 'w')
f.create_dataset('/string', (1,50), dtype='a25')
f.create_dataset('/u8', (1,50), dtype=np.int32)

f['/test'] = np.arange(10)
f['test2'] = np.ones(10, dtype=np.uint8)

f['/string'][0,0] = bytes('asdf'.encode('utf8'))
print(f['/string'])

f['/u8'][:] = np.ones((1,50), dtype=np.int32)
f['u8'][0,:4] = np.zeros((1,4), dtype=np.int32)
print(f['/u8'])

f['zhs'] = '你好' # simply works, implicitly using utf8
f['zhs2'] = '你好'.encode('utf8')
print( f['zhs2'][...].all().decode() )

f.close()
