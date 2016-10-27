%! octave

data = rand(10,10);

save -hdf5 data.hdf5 data

clear;

load -hdf5 data.hdf5

% you can inspect the resulting HDF5 data file with
%  $ h5dump -H data.hdf5      # only dump header info
%  $ h5dump data.hdf5         # dump all data
