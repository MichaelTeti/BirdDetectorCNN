# Deep Learning Bird Detector
  
## Programs

- [Bird identifier](https://github.com/MichaelTeti/BirdDetectorCNN/blob/master/CNN.py) - This project is in collaboration with Dr. Nathan Dorn of the FAU Biological Sciences Dept. to detect birds present in remote time-lapse images, and it is also my first experience with Tensorflow. 

- [Bird counter](https://github.com/MichaelTeti/BirdDetectorCNN/blob/master/aerial_counter.py) - This script is in collaboration with the South Florida Water Management District, and is an attempt to count the number of birds present in an aerial image. It uses Tensorflow, as well as OpenCV to get contours from the image. This is part of the Everglades Foundation Grant I received for September 2016 - September 2017.

## Background
Observing how populations of a species change due to some variable or fluctuate in size over time is a vital function of many ecologists. This may be particularly straightforward for those studying the population dynamics of microbes or other species that can be studied in a small, easily-defined area, yet this is not the case for most species. It is often very challenging, if not impossible in some cases, to monitor and observe certain species over a significant amount of time and space, whether this is due to the species' wide habitat or foraging range, skittishness, or some other difficulty. 

Wading birds, a good example of species that are difficult to monitor over long periods and large areas, are studied greatly due to their use as an indicator of ecosystem health and restoration success. However, due to their foraging habits and ability to fly relatively large distances to access food, they are not easy to study on a large scale. One relatively inexpensive solution is to deploy remote cameras in a study area over some period of time to capture images or video of the area. However, this method takes a substantial amount of time, as researchers are required to look through the vast amount of recorded data. 

Automated methods of classifying birds, among other animal species, have proven to be quite successful in recent years. One such method, convolutional neural networks (CNN's), have recently become widely used due to their remarkable ability to classify images as well as objects in them. Here, we use a convolutional neural network architecture to aid in the detection of bird species. 

The network takes each image as input and contains alternating convolution and pooling layers, and a subsequent fully-connected layer(s) with a sigmoid activation function is used to compute the output. The input images are often whitened or scaled so each value is between zero and one. A popular method used in feature scaling of images is to subtract the mean of the data and divide by the standard deviation. The number of convolutional filters in each convolutional layer, which can be chosen through trial and error or a pruning method, often depend on several factors, including the number of channels in the input and the size of the pooling kernels. Each successive layer in a CNN contains an equal or greater number of convolutional filters, and thus maps more abstract and complex features, than the previous convolutional layer. After the convolutional filters are applied, pooling layers then perform a non-linear subsampling function and return the maximum value from a defined subset of the input to the layer. Overlapping pooling describes an architecture in which the size of the pooling window is greater than the stride length and has been shown to aid in reducing overfitting in a CNN. Random dropout of a proportion of the fully-connected layer nodes has also been shown to dramatically reduce overfitting of the training data. 

We experiment with different methods of utilizing the temporal aspect of these time-lapsed images to predict whether a bird is present in the image being classified. A recurrent neural network could not have been used in this task, as it is impossible to reliably predict whether a bird is present in one image given the previous image. A number of different methods were attempted, each using grayscale images. In the first method, the image being classified was merely subtracted from the previous one. In a second trial, the previous image was placed behind the classification image along the third dimension, essentially acting like a color channel for the CNN. In the final method, the second order central finite difference between the previous image and the one being classified.

## Bird Detector
The ground-level, time-lapse images are of variable quality and lighting. Some contain lots of fog, and the lighting is very different throughout them. In addition, the birds are very far away in some, making them hard to see. Here is an example image:  
  
<p align="center">
  <br><br>
  <img src="https://github.com/MichaelTeti/BirdDetectorCNN/blob/master/bird.jpg">
</p>  
  
As is visible, it is very hard at times to distinguish these birds, and some are even farther away and smaller or appear behind the sawgrass. Since the images are very large as well, it is impossible to train a network on each image, so distinct patches were taken from each image and used as the training set. We achieved 99.05% validation accuracy when using these patches and hope to implement this network in a software system to detect and track birds in real time.  
  
## Bird Counter
The goal in these images is to count each bird in the images. Most times, the white color of the birds is contrasted well with the dark water, but some images contain dead cattails that appear white, making the problem a little difficult. Furthermore, all images contain dark colored birds (look closely and you can see them), which are not included in the SFWMD's surveys because they are too difficult to see. I also tried to see if those could be detected and counted.  
  
<p align="center">
  <br><br>
  <img src="https://github.com/MichaelTeti/BirdDetectorCNN/blob/master/WP11-Dec18-2013%20(3).JPG">
</p>  
  
As can be seen from this image, which is typical for this data, the problem is not as straightforward as it may seem. Therefore, I decided to use OpenCV to detect contours in each image, draw bounding boxes around each one, and use those patches as training to the network. This has returned variable success. The next thing to try might be a Cellular Nonlinear Network at the front end, as there are many that perform different functions. 


