import cv2
import sys
import numpy as np
import imutils
import functools
import tensorflow as tf
from scipy.misc import imshow


def get_contours(image, to_hsv=None, thresh=None, smooth=None, edge=None):
  """
  A function to get contours from a BGR image using opencv with optional thresholding
  and smoothing first.

  Args:
       to_hsv: if this value is set to True, the BGR image will be transformed
               to hsv colorspace. Default leaves it as BGR colorspace.

       thresh: If thresh is given, a threshold will be applied to the image. 
               thresh should be a 2 x 3 numpy array, where the first row is the
               lower bound and the second row is the upper bound of the threshold. 

       smooth: If smooth is given, cv2's bilateralFilter will be applied to 
               the image. smooth should be a 3-vector, indicating the arguments 
               for the smoothing function. 

       edge:   If edge is given, cv2's canny edge detector will be used on the 
               image. It should consist of a 2-vector signifying the lower bound 
               and upper bound. 
  """
  if to_hsv is True:
    image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  
  if thresh is not None:
    if np.shape(thresh)!=(2, 3):
      raise TypeError('thresh must be a 2x3 array.')
    else:
      mask=cv2.inRange(image, thresh[0, :].astype(int), thresh[1, :].astype(int))

  if smooth is not None:
    if 'mask' in locals():
      mask=cv2.bilateralFilter(np.uint8(mask), smooth[0], smooth[1], smooth[2])
    else:
      mask=cv2.bilateralFilter(np.uint8(image), smooth[0], smooth[1], smooth[2])
  
  if edge is not None:
    if 'mask' in locals():
      mask=cv2.Canny(mask, edge[0], edge[1])
    else:
      mask=cv2.Canny(image, edge[0], edge[1])

  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
  return cnts[0] if imutils.is_cv2() else cnts[1]
    


def contour_imcrop(im, contour, patch_size):
  """Creates a bounding box around contour and crops im around contour according
     to patch_size. """
  x, y, w, h = cv2.boundingRect(contour) 
  add_ht=patch_size-h
  add_wd=patch_size-w 
  assert((add_ht, add_wd)>=(0, 0)), 'added height: %d, added width: %d'%(add_ht, add_wd)
  im_pad=np.pad(im, ((patch_size/2, patch_size/2), (patch_size/2, patch_size/2), (0, 0)), 'edge')
  im_crop=im_pad[(y+patch_size/2)-add_ht/2:(y+patch_size/2)+h+add_ht/2,
                 			(x+patch_size/2)-add_wd/2:(x+patch_size/2)+w+add_wd/2, :]

  return x, y, w, h, im_crop

  

def doublewrap(function):
  """
  A decorator decorator, allowing to use the decorator to be used without 
  parentheses if not arguments are provided. All arguments must be optional.
  """
  @functools.wraps(function)
  def decorator(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
      return function(args[0])
    else:
      return lambda wrapee: function(wrapee, *args, **kwargs)
  return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
  """
  A decorator for functions that define TensorFlow operations. The wrapped
  function will only be executed once. Subsequent calls to it will directly
  return the result so that operations are added to the graph only once.
  The operations added by the function live within a tf.variable_scope(). If
  this decorator is used with arguments, they will be forwarded to the
  variable scope. The scope name defaults to the name of the wrapped
  function.
  """
  attribute = '_cache_' + function.__name__
  name = scope or function.__name__
  @property
  @functools.wraps(function)
  def decorator(self):
    if not hasattr(self, attribute):
      with tf.variable_scope(name, *args, **kwargs):
        setattr(self, attribute, function(self))
    return getattr(self, attribute)
  return decorator


class NN(object):
  ''' A convolutional neural network.
      
      Parameters:
                 x: input data in the form batch size x features
                 y: vector with the length batch size containing to the 
                    correct prediction for the corresponding image.
           num_out: the number of output nodes for the network.
                ps: Assumes the width and height of the input image are the
                    same. The width and height of each patch in x.
          batch_sz: the number of patches to use in each training iteration.
   feature_scaling: Whether or not to normalize the data before training. 
                    Defaults to None, in which no normalization will be
                    performed. ''' 

  def __init__(self, x, y, num_out, ps, batch_sz, feature_scaling=None):
    self.batch_sz=batch_sz # batch size
    self.feature_scale=feature_scaling
    self.ps=ps # input image size
    self.x=x # data
    self.y=y # labels
    self.num_out=num_out  # number of output nodes
    if self.num_out!=1:
      self.y=tf.one_hot(self.y, self.num_out)
    self.prediction 
    self.optimize
    self.error 

  @define_scope()
  def feature_scaling(self):
    if self.feature_scale is not None:
      mean, var=tf.nn.moments(self.x, axes=[0])
      mean=tf.tile(tf.reshape(mean, [1, self.ps**2*3]), [self.batch_sz, 1])
      std=tf.tile(tf.reshape(tf.sqrt(var), [1, self.ps**2*3]), [self.batch_sz, 1])
      return (self.x-mean)/std
    else:
      return self.x
  
  @define_scope(initializer=tf.contrib.slim.xavier_initializer())
  def prediction(self):
    self.x_=tf.reshape(self.feature_scaling, [-1, self.ps, self.ps, 3])
    slim=tf.contrib.slim
    net=slim.repeat(self.x_, 3, slim.conv2d, 100, [5, 5], padding='VALID')
    net=tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    net=slim.dropout(slim.fully_connected(slim.flatten(net), 1100), keep_prob=0.5)
    net=slim.dropout(slim.fully_connected(net, 1100), keep_prob=0.5)
    return tf.contrib.slim.fully_connected(net, int(self.num_out), tf.nn.softmax)
  
  @define_scope()
  def optimize(self):
    cost=-tf.reduce_sum(self.y * tf.log(self.prediction+1e-12))
    #optimizer = tf.train.RMSPropOptimizer(0.03)
    return tf.train.AdamOptimizer(1e-4).minimize(cost)
  
  @define_scope
  def error(self):
    if self.num_out!=1:
      max_labels=tf.argmax(self.y, 1)
    else:
      max_labels=self.y
    mistakes=tf.not_equal(max_labels, tf.to_float(tf.argmax(self.prediction, 1)))
    return tf.reduce_mean(tf.cast(mistakes, tf.float32))



def main():

  # initialize some stuff
  i=1
  e_plot=np.zeros([1, ]) 
  batch_sz=5
  patch_sz=100 
  num_out_nodes=1 

  data=np.zeros([1, patch_sz**2*3])
  labels=np.zeros([1, ]) 
  
  data_holder=tf.placeholder(tf.float32, shape=[None, data.shape[1]])
  label_holder=tf.placeholder(tf.float32, shape=[None, ])
 
  nn=NN(data_holder, 
        label_holder, 
        num_out_nodes, 
        patch_sz, 
        batch_sz, 
        feature_scaling=None)
    
  sess=tf.Session() 
  sess.run(tf.global_variables_initializer()) 

  im=cv2.imread('im.JPG') # read the image - going to have to loop through them all

  # get the contours 
  cnts=get_contours(im, 
                    thresh=np.array([[215, 215, 215], [255, 255, 255]]),
                    smooth=np.array([9, 95, 95]))
  

  for c in cnts: # loop through each contour
    if cv2.contourArea(c)>1200: 

      x, y, w, h, im_crop=contour_imcrop(im, c, patch_sz)
      im_crop=cv2.resize(im_crop, (patch_sz, patch_sz)) 
      data=np.concatenate((data, im_crop.reshape([1, patch_sz**2*3])), axis=0)

      imshow(im_crop)

      # take user input to create labels for the patch shown
      label=raw_input('0: no bird, 1: bird, 2: see whole map')
      try:
        label=int(label)
      except ValueError: 
        imshow(im_crop)
        label=int(raw_input('Please enter only an integer'))
      im_copy=im.copy()
      if label==2:
        while(1):
          cv2.rectangle(im_copy, (x, y), (x+w, y+h), (0, 0, 255), 4)
          cv2.namedWindow('rect contour', cv2.WINDOW_NORMAL)
          cv2.imshow('rect contour', im_copy)
          if cv2.waitKey(10000)==27:
            cv2.destroyAllWindows()
            break
        label=int(raw_input('0: no bird, 1: bird'))
      assert(label in [0, 1, 2]), 'input must be 0, 1, or 2.'
      labels=np.concatenate((labels, np.array([label, ])), axis=0)
      print np.shape(data)
      print labels

      # send labels/data to network when data/label input equals desired batch size
      if data.shape[0]%(batch_sz+1)==0:   
        e=sess.run(nn.error, {data_holder: data[1:, :], label_holder: labels[1:, ]})
        e_plot=np.concatenate((e_plot, np.array([e, ])), axis=0)
        print("""
              ===================================
              Iteration: %d, Training Error: %.2f
              ===================================
              """%(i, e))

        sess.run(nn.optimize, {data_holder: data[1:, :], label_holder: labels[1:, ]})
        data=np.zeros([1, patch_sz**2*3]) # empty the data array for next iter
        labels=np.zeros([1, ]) # empty the label array for next iter
        i+=1

if __name__ == '__main__':
  main()
