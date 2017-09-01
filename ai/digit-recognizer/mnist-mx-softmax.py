# coding: utf-8
# @ref http://mxnet.io/tutorials/python/mnist.html
import logging
logging.getLogger().setLevel(logging.DEBUG)

import mxnet as mx
mnist = mx.test_utils.get_mnist()
batchsize = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batchsize, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batchsize)
data = mx.sym.var('data')
data = mx.sym.flatten(data=data)

fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type='relu')

fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
act2 = mx.sym.Activation(data=fc2, act_type='relu')

fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
mlp = mx.sym.Softmax(data=fc3, name='softmax')

model = mx.mod.Module(symbol=mlp, context=mx.cpu())
model.fit(train_iter, eval_data=val_iter,
  optimizer='adam', optimizer_params={'learning_rate':0.01},
  eval_metric='acc',
  batch_end_callback = mx.callback.Speedometer(batchsize, 1),
  num_epoch=3)

test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batchsize)
prob = model.predict(test_iter)
print(prob.shape)
