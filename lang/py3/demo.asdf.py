
# http://asdf.readthedocs.io/en/latest/asdf/examples.html

from asdf import AsdfFile
import numpy as np

af = AsdfFile()
af.tree['hello'] = 'world'
af.tree['data'] = np.random.rand(8,4096) # 'data': the main content
myarray = np.random.rand(1024)
af.tree['data_ascii'] = myarray
af.set_array_storage(myarray, 'inline') # specify ascii mode
af.write_to('example.af')

# compression: zlib / bzp2
# af.tree['data'] = np.random.rand(10000, 4096)
# af.write_to('example.af') # 313M, RAW, quick
# af.write_to('example.af', all_array_compression='zlib')
  # 295M, SLOW
# af.write_to('example.af', all_array_compression='bzp2')
  # 299M, VERY SLOW
