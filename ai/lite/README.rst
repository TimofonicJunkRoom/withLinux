Lite
====

A light weight neural network implementation in C++, inspired by Caffe and
Torch(lua). It only needs the HDF5 library for data input.

Design
------

There are 3 (4 in the future) Core concepts in this framework currently, they are:
*Tensor*, *Blob*, *Layer*, (*Graph* in the future).

*Tensor*

  Generalized container of numerical data. Vectors, matrices or any
  higher-dimensional number blocks are regarded as Tensor, where the
  data is stored in a contiguous memory block.

  See ``tensor.cc`` for detail.

*Blob*

  Combination of two Tensors, one for the value and another for its
  gradient. This is useful in the Caffe-styled computation graph,
  where the backward pass just uses the forward graph instead of
  extending the graph for gradient computation and parameter update.

  See ``blob.cc`` for detail.

*Layer*

  Network layers, including loss functions. Each of them takes some
  input Blobs and output Blobs as argument during forward and backward.

  See ``layer.cc`` for detail.

*Graph*

  To be implemented.

Details
-------

TODO:

* conv2 layer
* reshape layer
* container/graph

Dependencies:

* HDF5

See also:

* Memory leak issue: https://stackoverflow.com/questions/6261201/how-to-find-memory-leak-in-a-c-code-project

Changelog:

* Nov 8, draft.
* Nov 11, first finely working version.
