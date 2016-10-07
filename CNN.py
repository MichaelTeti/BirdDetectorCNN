import scipy
from scipy.io import loadmat
from scipy.io import savemat
import scipy.misc
import numpy as np
import os 
import sys
sys.path.append('/home/mpcr/newproject/venv/lib/python2.7/site-packages')
import tensorflow as tf

# load data and labels
a=loadmat('data.mat') 
data=a['patches'] # whitened data
b=loadmat('labels.mat')
labels=b['labels'] # labels

# Create training dataset and testing dataset and feature scaling
data=np.divide((data-np.mean(data, axis=0)), np.std(data, axis=0))
data=np.nan_to_num(data)
perm=np.random.permutation(np.ma.size(data, 0))
data=data[perm, :]
labels=labels[perm]
train_data=data[:(np.ma.size(data, 0)-np.ma.size(data, 0)*.1), :]
test_data=data[(np.ma.size(data, 0)-np.ma.size(data, 0)*.1):, :]
train_labels=labels[:(np.ma.size(data, 0)-np.ma.size(data, 0)*.1)]
test_labels=labels[(np.ma.size(data, 0)-np.ma.size(data, 0)*.1):]

patch_size=np.ma.size(data, 1) # 130x130x2 input 
output=1 # one output node
h1=np.round(25*50*.5, decimals=0).astype(int) # num. of fully connected layer nodes

# Create random training set each iteration of gradient descent 
def training_set(X, y, batch):
	rand_ex=np.round((np.random.rand(batch, 1)*(np.ma.size(X, 0)-1)), 
		decimals=0).astype(int)
	td=X[rand_ex.flatten(), :]
	labels=y[rand_ex.flatten()]
	if np.ma.size(td, 0)!=batch and np.ma.size(td, 1)!=patch_size:
		print 'Data not loaded correctly. Goodbye'
		sys.exit()
	return td, labels

sess=tf.InteractiveSession() # begin tensorflow session
x = tf.placeholder(tf.float32, shape=[None, patch_size]) # Input layer 
y = tf.placeholder(tf.float32, shape=[None, output]) # labels
keep_prob = tf.placeholder(tf.float32) # dropout probability for fc layer

# Network
out=tf.nn.conv2d(tf.reshape(x, [-1, 130, 130, 2]),
	tf.Variable(tf.truncated_normal([8, 8, 2, 15],
	stddev=0.1)), strides=[1, 2, 2, 1], padding='SAME')
out=tf.nn.relu(out+tf.Variable(tf.constant(0.1, shape=[15])))
out=tf.nn.max_pool(out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
	padding='SAME')
out=tf.nn.conv2d(out, tf.Variable(tf.truncated_normal([4, 4, 15, 25],
	stddev=0.1)), strides=[1, 1, 1, 1], padding='SAME')
out=tf.nn.relu(out+tf.Variable(tf.constant(0.1, shape=[25])))
out=tf.nn.max_pool(out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
	padding='SAME')
out=tf.nn.conv2d(out, tf.Variable(tf.truncated_normal([3, 3, 25, 35], 
	stddev=0.1)), strides=[1, 1, 1, 1], padding='SAME')
out=tf.nn.relu(out+tf.Variable(tf.constant(0.1, shape=[35])))
out=tf.nn.max_pool(out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
	padding='SAME')
out=tf.nn.conv2d(out, tf.Variable(tf.truncated_normal([3, 3, 35, 50],
	stddev=0.1)), strides=[1, 1, 1, 1], padding='SAME')
out=tf.nn.relu(out+tf.Variable(tf.constant(0.1, shape=[50])))
out=tf.nn.max_pool(out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
	padding='SAME')
out=tf.matmul(tf.reshape(out, [-1, 25*50]), tf.Variable(tf.truncated_normal
	([25*50, h1], stddev=0.1)))
out=tf.nn.dropout(tf.nn.relu(out+tf.Variable(tf.constant(0.1, shape=[h1]))), 
	keep_prob)
out=tf.matmul(out, tf.Variable(tf.truncated_normal([h1, h1], stddev=0.1)))
out=tf.nn.dropout(tf.nn.relu(out+tf.Variable(tf.constant(0.1, shape=[h1]))),
	keep_prob)
out=tf.matmul(out, tf.Variable(tf.truncated_normal([h1, output], stddev=0.1)))
out=tf.sigmoid(out+tf.Variable(tf.constant(0.1, shape=[output])))

# Mean squared error cost function
cost=tf.reduce_sum((y-out)**2, reduction_indices=0)

# Gradient Descent Optimizer and calculate accuracy for each output
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(out), y), tf.float32))

saver=tf.train.Saver() # Saves the model parameters after training

sess.run(tf.initialize_all_variables()) # initialize all variables

# Run and train graph
for i in range(15000): # training iterations
  training_data, training_labels=training_set(train_data, train_labels, 90)
  if i%1 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:training_data, y:training_labels, keep_prob: 1.0})
    print("step %d, training accuracy: %g"%(i, train_accuracy))
    oute=out.eval(feed_dict={x:training_data, y:training_labels, keep_prob: 1.0})
    print oute
  train_step.run(feed_dict={x: training_data, y: training_labels, keep_prob: 0.5})
  test_accuracy=accuracy.eval(feed_dict={x:test_data, y:test_labels, keep_prob:1.0})
  if test_accuracy > .99: # test accuracy on 10% of data
    print 'testing accuracy: %.4f' % (test_accuracy)
    # Save modified variables
    save_variables=saver.save(sess, '/home/mpcr/Desktop/lila_birds/bird.ckpt')
    sys.exit() # exit and save parameters if >= 99% accuracy reached
