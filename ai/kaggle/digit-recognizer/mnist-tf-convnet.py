# http://tensorfly.cn/tfdoc/tutorials/mnist_pros.html
# http://oldblog.fuyangzhen.com/bootstrap/blog/001484103400778194a18ec4c8e4e599d4df98c338590f7000#

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
print('-> Using TF', tf.__version__)

### Read Train-Val data and split ###
trainval = pd.read_csv("train.csv")
trainval_images = trainval.iloc[:, 1:].div(255)
trainval_labels = pd.get_dummies(trainval.iloc[:, :1].label)
train_images, val_images, train_labels, val_labels = train_test_split(
        trainval_images, trainval_labels, train_size=0.8, random_state=0)
print('-> train set shape', train_images.shape)
print('-> val   set shape', val_images.shape)

### Read Test data ###
test = pd.read_csv('test.csv')
test_images = test.iloc[:,:].div(255)
print('-> test  set shape', test_images.shape)

### Setup Graph ###
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1,28,28,1]) # localid, width, height, channel

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_ =tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y*tf.log(y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#for i in range(20000):
#    batch = mnist.train.next_batch(50)
#    if i%100 == 0:
#        train_accuracy = accuracy.eval(feed_dict={
#            x:batch[0], y_: batch[1], keep_prob: 1.0})
#        print "step %d, training accuracy %g"%(i, train_accuracy)
#    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
#print "test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

### Train and Val ###
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i in range(20000):
# could not set cudnn tensor descriptor: CUDNN_STATUS_BAD_PARAM
# batchsize 0 will cause the above zero, be careful when reading.
# when your index for reading data is wrong, the batch could be 0.
    batch_images = train_images.iloc[
        (i*50)%33600:
        (i+1)%672==0 and 33600 or ((i+1)*50)%33600].values
    batch_labels = train_labels.iloc[
        (i*50)%33600:
        (i+1)%672==0 and 33600 or ((i+1)*50)%33600].values

    loss, acc, _ = sess.run([cross_entropy, accuracy, train_step],
            feed_dict={x: batch_images, y: batch_labels, keep_prob: 0.5})
    if i % 20 == 0:
        #print('-> data range {} - {}'.format( (i*50)%33600,
        #    (i+1)%672==0 and 33600 or ((i+1)*50)%33600))
        print('-> step {:5d} | loss: {:5.2f} | train acc {:.03f} | test accuracy: {:.05f}'.format(
            i, loss, acc,
            sess.run(accuracy, feed_dict={x:val_images, y:val_labels, keep_prob: 1.0})))

print('Save file to path:', saver.save(sess, 'kaggle_MNIST_net.ckpt'))
### saver restore ### 
# W = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32, name='Weights')
# b = tf.Variable(tf.zeros([1, 10]), dtype=tf.float32, name='biases')
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess,'/Users/fuyangzhen/Desktop/kaggle_MNIST_net.ckpt')
#     print('W:',sess.run(W))
#     print('b:',sess.run(b))

### Test ###
test_predictions = sess.run( tf.argmax(y_, 1), feed_dict={x: test_images})
print('-> made predictions', test_predictions.shape)
df_predictions = pd.DataFrame(np.array([np.arange(1,28000+1), test_predictions]).T,
        columns=["ImageID", "Label"])
print(df_predictions.head())
df_predictions.to_csv("predictions.csv", header=True, index=False)
