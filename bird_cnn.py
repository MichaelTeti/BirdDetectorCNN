#============================================================
#############################################################
# Author = Michael A. Teti
# Florida Atlantic University
# Center for Complex Systems and Brain Sciences
# Machine Perception and Cognitive Robotics Lab

#############################################################
#============================================================
#############################################################
# Time-lapse images taken of a cell at the LILA complex every 
# 3 minutes, which were taken by a standard game camera. 
# Images owned by Dr. Nathan Dorn, Biological Sciences Dept.
# of Florida Atlantic University. Deep convolutional neural 
# network using TensorFlow was constructed to detect and 
# identify birds appearing in the images. 

#############################################################
#============================================================

import tensorflow as tf
import numpy as np
import os
import glob
import scipy
import PIL
import scipy.misc



class CNN:
    def loadimages(self):
	os.chdir('/media/mpcr/5EBA074CBA071FDF/M4_3') # folder with images
        print 'Loading Images...'
	ims=np.zeros((1, 212*409*3))
        for filename in glob.glob("*.JPG"): # load all images with .JPG format
            im=scipy.misc.imread(filename) # turn image into numpy array
	    im=im[450:1510, 0:2200, :] # crop unecessary parts of image 
	    im=scipy.misc.imresize(im, .20) # resize image to 20% of original size
            im=im.flatten('C') # turn image into a vector
	    im=im[np.newaxis, :]
            ims=np.concatenate((ims, im), axis=0) # add current image to compilation
	self.images=ims
	self.images=self.images[1:, :]
	return self.images
       
    def featureScaling(self):
        self.images=instance.loadimages() 
        print 'Scaling image features...'
        (a, b)=np.where(self.images==0) # remove 0's to avoid returning NAN
        self.images[a, b]=0.0001
        mu=np.mean(self.images, axis=0)
        sigma=np.std(self.images, axis=0)
        a=np.where(sigma==0)
        sigma[a]=0.0001
        cols=np.ma.size(self.images, 1)
        for i in range(cols):
	    imcol=self.images[:, i]
 	    imcol=imcol-mu[i]
	    imcol=np.divide(imcol, sigma[i])
            self.images[:, i]=imcol # scale each pixel down to a mean of 0 and std of 1
        return self.images
   

    def load_labels(self):
        print 'Loading Labels...'
	os.chdir('/home/mpcr/Desktop/lila_birds')
        self.labels = np.loadtxt('labels.txt')
	self.labels = self.labels[0:np.ma.size(self.images, 0)]
        return self.labels


    def train_and_test(self):
        instance.featureScaling()
        instance.load_labels()
        self.labels=self.labels[:, np.newaxis]
	print 'Compiling data and saving...'
        all_data=np.concatenate((self.labels, self.images), axis=1)
	np.save('data.npy', all_data)
    


instance=CNN()
instance.train_and_test()






