import numpy as np
import tensorflow as tf
import os
import scipy

# training data
data=np.load('data.npy')

# begin interactive session with tensorflow
sess=tf.InteractiveSession()

# create input nodes and output node to run data through
x = tf.placeholder(tf.float32, shape=[None, 106*204*3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

def train_size(training_data, batch_size):
  r=np.random.permutation(np.ma.size(training_data, 0))
  batch=r[0:batch_size]
  x=training_data[batch, 1:]
  y=training_data[batch, 0]
  y=y[:, np.newaxis]
  return x, y

# create small, positive weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# initilize values
Convolutions_1=32
Convolutions_2=64
Fully_connected1=5000
Training_iters=10000
batch_size=35

# [filter_height, filter_width, input_channels, convolutions]
Convolution1_W = weight_variable([8, 8, 3, Convolutions_1]) 
bias1 = bias_variable([Convolutions_1])

# reshape image vector to actual shape
x_image = tf.reshape(x, [-1,106,204,3])

# apply first layer of convolutions and run through rectified
# linear activation function
activation_conv1 = tf.nn.relu(conv2d(x_image, Convolution1_W) + bias1) 

# max_pooling on first layer of convolutions
activation_pool1 = max_pool_2x2(activation_conv1) 

# layer 2 convolutions on max pooled layer
Convolution2_W = weight_variable([5, 5, Convolutions_1, Convolutions_2]) 
bias2 = bias_variable([Convolutions_2])

# apply second layer of convolutions and activate
activation_conv2 = tf.nn.relu(conv2d(activation_pool1, Convolution2_W) + bias2)

# max pool second layer of convolutions
activation_pool2 = max_pool_2x2(activation_conv2) 

# create weights and bias for fully-connected layer
W_fc1 = weight_variable([7*13*Convolutions_2, Fully_connected1])
b_fc1 = bias_variable([Fully_connected1])

# reshape 2nd pooling layer and run through fully connected layer
h_pool2_flat = tf.reshape(activation_pool2, [-1, 7*13*Convolutions_2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# create placeholder for dropout probability
keep_prob = tf.placeholder(tf.float32)

# cause dropout of activations from fully-connected layer 
# to avoid overfitting
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# create weights and biases for final fully-connected layer
W_fc2 = weight_variable([Fully_connected1, 1])
b_fc2 = bias_variable([1])

# compute output with softmax function
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# compute cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

# method of training and computing accuracy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
correct_prediction = tf.equal(tf.round(y_conv), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize all variables before running
sess.run(tf.initialize_all_variables())

# training
for i in range(Training_iters):
  train_data, train_labels=train_size(data, batch_size)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:train_data, y_:train_labels, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: train_data, y_: train_labels, keep_prob: 0.75})

#print("test accuracy %g"%accuracy.eval(feed_dict={
    #x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


