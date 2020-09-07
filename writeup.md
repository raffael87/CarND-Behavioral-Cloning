# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/nvidia_cnn_arch.png "Nvidia CNN Architecture Visualization"
[image2]: ./images/histo_train.png "Histogram Training Data"
[image4]: ./images/recovery_left.png "Recovery left"
[image5]: ./images/recovery_middle.png "Recovery half"
[image6]: ./images/recovery_center.png "Recovery center"
[image7]: ./images/image_left.png "Original left"
[image8]: ./images/image_center.png "Original center"
[image9]: ./images/image_right.png "Original right"
[image10]: ./images/image_cropped.png "Horizon, car hood removed"
[image11]: ./images/image_resized.png "Resized image to fit input"
[image12]: ./images/image_gray.png "Image Gray"
[image13]: ./images/image_translate.png "Image translated"
[image14]: ./images/image_flip.png "Image flipped"
[image15]: ./images/image_bright.png "Image brightend"
[image16]: ./images/epoch.png "Image brightend"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md = README.md
* visualize_data.py
* video.mp4

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

During the last course lessons, some neural networks where discussed as AlexNet, GoogLeNet, Nvidia. The paper from Nvidia ([End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)) was a good source, because they trained also a car to use the behavior of a driver.
Also the architecutre seemed not so complicated to implement with Keras.
The model includes RELU layers to introduce nonlinearity (code line 117, 119, 122, 124, 126, 130, 132, 135, 137), and the data is normalized in the model using a Keras lambda layer (code line 114). 
At the beginning also Cropping2D was used to crop the image, but later it was replaced by a cropping which is done outside the model.

#### 2. Attempts to reduce overfitting in the model

Depending on the model, dropout layers where used to reduce overfitting. At the final stages it was not necessary anymore so it was removed (model.py lines 127, 133)

To ensure that there is no overfitting, the model was trained and validated on data sets. 80% of the data was used for training and 20% for validation.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 140).

#### 4. Appropriate training data

During this whole project, different apporaches where used in order to find good training data. This was not as easy because of the driving behavior of the simulator.
Nevertheless a combination of re entering the lane from outside, driving in the middle lane and also driving middle lane in reverse was performed.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the one from Nvidia in order to have a good base model and architecture which was used for the same purpose.

Modifications during the project on this model where adding a preprocessing layer for normalization, a preporcessing layer for cropping (which was later removed) and also drop out layers (which were removed too after no overfitting was seen).

The mean square errors for training and validation were quite good all the time, arround 0.03 and 0.02 depending on the data.

Looking into the epochs, it was clear that a good value would be smaller than 10 and after some tests 4 epochs were sufficient (model.py line 198).

After each model, a training run in the simulator was performed to see where the predictions were bad (curves, sometimes straight lane).

Getting good training data (driving in the simulator is more difficult than in real life) from the simulator is quite challenging. In order to have still training data, I augmented the data set similar to the last project.

For the training data from Udacity the distribution shows clearly, that the course has more left turns then right turns. Later on with data set augmentation we try to reduce this fact. Also with driving backwards it is possible to get a better distribution:

![alt text][image2]


#### 2. Final Model Architecture

Original image size: 160x320x3

| Layer         		|     Description	        									|
|:---------------------:|:---------------------------------------------:				|
| Input         		| 66x200x3 normalized using lambda function (x/127.5 - 1.0) 	|
| Convolutional 2d		| stride = 2 depth = 24 kernel = 5x5 							|
| ELU				    | 																|
| Convolutional 2d		| stride = 2 depth = 36 kernel = 5x5 							|
| ELU				    | 																|
| Convolutional 2d		| stride = 2 depth = 48 kernel = 5x5 							|
| ELU				    | 																|
| Convolutional 2d		| stride = 2 depth = 48 kernel = 5x5 							|
| ELU				    | 																|
| Convolutional 2d		| stride = 2 depth = 64 kernel = 3x3 							|
| ELU				    | 																|
| Convolutional 2d		| stride = 2 depth = 64 kernel = 3x3 							|
| ELU				    | 																|
| Flatten				|																|
| Dense					| output_space = 1164											|
| ELU				    | 																|
| Dense					| output_space = 100											|
| ELU				    | 																|
| Dense					| output_space = 50												|
| ELU				    | 																|
| Dense					| output_space = 10												|
| ELU				    | 																|
| Output: Dense			| output_space = 1												|


![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving and backwards.
Here an example driving from left side back to the center line

Left

![alt text][image4]

Going to the middle

![alt text][image5]

Center lane
![alt text][image6]


To augment the data set, I applied randomly different augmentation functions to the training data set. These augmentation technique was already used in the previous project.

During this project, some more hints were given what kind of augmentation could be used. I opted for the following:

| Function         		|     Description	        																										|
|:---------------------:|:---------------------------------------------:																					|
| No augmentation 		| Keeps the image and the steering wheel as it comes from the data																	|
| Grayscale				| Receives input image, converts it to grayscale and then back to rgb color space. This helps the model to not only rely on color 	|
| Translation			| As in the previous project, the image is moved a randomly a bit. The steering angle is recalculated depending on the movement 	|
| Flipped				| Flips the image randomly. This is mostly done because the track has more left curves. The steering angle is multiplied with -1 	|
| Brightness			| Changes the brightness of the image with a random value. This helps to simulate shades on the image etc							|

These functions are applied on each training image. So in the end the trainings data is much more divers.

Original image from left side:

![alt text][image7]

Original image from center:

![alt text][image8]

Original image from right side:

![alt text][image9]

Image cropped, part of horizon and the car hood was removed:

![alt text][image10]

Image resized to fit to the input layer:

![alt text][image11]

Grayscale image:

![alt text][image12]

Translated image:

![alt text][image13]

Flipped image:

![alt text][image14]

Brightned image:

![alt text][image15]


To save memory and to avoid memory problems two generators were used (model.py lines 72, 92). The technique was shown in the previous course classes. 
From previous experience the batch size was set to 64.
The data was split 80% for training data and 20% for validation (model.py line 180). In total for the last model after augmentation we get 48000 samples.

In the generators the sample is picked randomly (similar to the shuffle). The path is taken and a RGB image is created with the help of OpenCV. For the training samples, augmentation functions are applied (model.py line 84).

For training, all three images are used. Left, Right and Center. In the paper from Nvidias CNN it is described how we can use all of the tree images. For this, we apply to the steering angle for the left and the right a correction of 0.25.

Example of one training:

![alt text][image16]

In order to be able to use the model for prediction in the simulator, in drive.py (line 52)  the input data is cropped and resized to fit to the model.

Link to the video [One lap](https://github.com/raffael87/CarND-Behavioral-Cloning-P3/blob/master/video.mp4)
