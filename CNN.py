
import numpy as np
import tensorflow as tf
import os
import scipy

# training data
data=np.load('data.npy')

# begin interactive session with tensorflow
sess=tf.InteractiveSession()

# image dimensions
rows=371
cols=716
input_channels=1
numel_pool=6*12
output=2

# create input nodes and output node to run data through
x = tf.placeholder(tf.float32, shape=[None, rows*cols*input_channels])
y_ = tf.placeholder(tf.float32, shape=[None, output])

def train_size(training_data, batch_size):
  r=np.random.permutation(np.ma.size(training_data, 0))
  batch=r[0:batch_size]
  x=training_data[batch, 2:]
  y=training_data[batch, 0:2]
  #y=y[:, np.newaxis]
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
Convolutions_1=30
Convolutions_2=60
Convolutions_3=30
Fully_connected1=1300
Training_iters=10000
batch_size=65


# [filter_height, filter_width, input_channels, convolutions]
Convolution1_W = weight_variable([5, 5, input_channels, Convolutions_1]) 
bias1 = bias_variable([Convolutions_1])

# reshape image vector to actual shape
x_image = tf.reshape(x, [-1,rows,cols,input_channels])

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

# layer 3 convolutions and bias
Convolution3_W=weight_variable([5, 5, Convolutions_2, Convolutions_3])
bias3=bias_variable([Convolutions_3])

# third layer of convolutions and activation
activation_conv3 = tf.nn.relu(conv2d(activation_pool2, Convolution3_W) + bias3)

# max pool third layer of convolutions
activation_pool3=max_pool_2x2(activation_conv3)
print activation_pool3

# create weights and bias for fully-connected layer
W_fc1 = weight_variable([numel_pool*Convolutions_3, Fully_connected1])
b_fc1 = bias_variable([Fully_connected1])

# reshape 2nd pooling layer and run through fully connected layer
h_pool3_flat = tf.reshape(activation_pool3, [-1, numel_pool*Convolutions_3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# create placeholder for dropout probability
keep_prob = tf.placeholder(tf.float32)

# cause dropout of activations from fully-connected layer 
# to avoid overfitting
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# create weights and biases for final fully-connected layer
W_fc2 = weight_variable([Fully_connected1, output])
b_fc2 = bias_variable([output])

# compute output with softmax function
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# compute cross entropy loss function
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

# method of training and computing accuracy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize all variables before running
sess.run(tf.initialize_all_variables())

# training
for i in range(Training_iters):
  train_data, train_labels=train_size(data, 50)
  if i%20 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:train_data, y_:train_labels, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: train_data, y_: train_labels, keep_prob: 0.8})

#print("test accuracy %g"%accuracy.eval(feed_dict={
    #x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
