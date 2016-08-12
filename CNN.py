import scipy
from scipy.io import loadmat
import scipy.misc
import numpy as np
import os 
import sys
sys.path.append('/home/mpcr/newproject/venv/lib/python2.7/site-packages')
import tensorflow as tf

# load data and ground truth
a=loadmat('lab.mat')
b=loadmat('datapatches.mat')
labels=a['labels']
data=b['patches']
patch_channels=3
patch_size=np.ma.size(data, 1)
output=1
batch=100
c1=75
c2=150
c3=200
c4=300
h1=2500
h2=2500

sess=tf.InteractiveSession() # begin tensorflow session

x = tf.placeholder(tf.float32, shape=[None, patch_size])
y = tf.placeholder(tf.float32, shape=[None, output])
keep_prob = tf.placeholder(tf.float32)

def training_set(X, y):
	td=np.empty([1, patch_size])
	labels=np.zeros([batch, 1])
	for i in range(batch):
		rand_ex=np.random.permutation(np.ma.size(X, 0))
		r=rand_ex[0:batch]
		train_data=X[r[i], :]
		train_data=train_data.flatten()
		train_data=train_data[np.newaxis, :]
		td=np.concatenate((td, train_data), axis=0)
		label=y[r[i]]
		labels[i]=label
	td=td[1:, :]
	if np.ma.size(td, 0)!=batch:
		print 'Data not loaded correctly. Goodbye'
		sys.exit()
	return td, labels

# create small, random weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# create biases
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

X=tf.reshape(x, [-1, 65, 65, patch_channels])
w1=weight_variable([2, 2, patch_channels, c1])
bias1=bias_variable([c1])
act1=max_pool_2x2(tf.nn.relu(conv2d(X, w1)+bias1))

w2=weight_variable([2, 2, c1, c2])
bias2=bias_variable([c2])
act2=max_pool_2x2(tf.nn.relu(conv2d(act1, w2)+bias2))

w3=weight_variable([2, 2, c2, c3])
bias3=bias_variable([c3])
act3=max_pool_2x2(tf.nn.relu(conv2d(act2, w3)+bias3))

w4=weight_variable([2, 2, c3, c4])
bias4=bias_variable([c4])
act4=max_pool_2x2(tf.nn.relu(conv2d(act3, w4)+bias4))
act4flat=tf.reshape(act4, [-1, 25*c4])

theta1=weight_variable([25*c4, h1])
bias1=bias_variable([h1])
activation2=tf.nn.relu(tf.matmul(act4flat, theta1) + bias1)
activation2 = tf.nn.dropout(activation2, keep_prob)

theta2=weight_variable([h1, output])
bias2=bias_variable([output])
out=tf.nn.softmax(tf.matmul(activation2, theta2) + bias2)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

training_iters=10000

# training
for i in range(training_iters):
  train_data, train_labels=training_set(data, labels)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:train_data, y:train_labels, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    #tf.Print(y_, train_labels)
  train_step.run(feed_dict={x: train_data, y: train_labels, keep_prob: 0.75})

