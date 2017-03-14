#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[visualization]: ./report_images/visualization.png "Visualization"
[distribution]: ./report_images/distribution.png "Distribution"
[precision_recall]: ./report_images/precision_recall.png "Precision_Recall"
[state]: ./report_images/nn_state.png "Neural Network State - Yield Sign"
[state1]: ./report_images/nn_state1.png "Neural Network State - Speed Limit Sign"
[image1]: ./test_images_hard/german1.png "Traffic Sign 1"
[image2]: ./test_images_hard/german2.png "Traffic Sign 2"
[image3]: ./test_images_hard/german3.png "Traffic Sign 3"
[image4]: ./test_images_hard/german4.png "Traffic Sign 4"
[image5]: ./test_images_hard/german5.png "Traffic Sign 5"

## [Rubric](https://review.udacity.com/#!/rubrics/481/view) 

---
### README
Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34,799.
* The size of test set is 12,630.
* The size of validation set is 4,410.
* The shape of a traffic sign image is (32,32,3).
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set.
The first image shows an example for each class in the dataset.
The second image shows the distribution of images for each class in the training and test datasets.
Since the distribution plot shows that certain classes only have a few examples, the network may overfit those classes.

![alt text][visualization]
![alt text][distribution]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because it was recommend in original model used by Yann LeCun.
As a last step, I normalized the image data because certain images are too dark or too bright.
Pixel normalization enhances the features in the image.

I disabled the standard pre-processing because it seemed to produce worse results.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the standard training, validation, testing split provided by the dataset. I randomly shuffle the datasets during training.

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.  

My final training set had 32,799 number of images. My validation set and test set had 4,410 and 12,630 number of images.

Since the performance of my network was satisfactory, I did not augment the training data. Tensorflow has a suite of functions to distort and transform images to improve the robustness of the algorithm. 

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x128 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x128				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x256   |
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x256   |
| Max pooling	      	| 2x2 stride,  outputs 8x8x256 					|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x512     |
| Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x512     |
| Max pooling	      	| 2x2 stride,  outputs 4x4x512 					|
| Fully connected		| (8192 x 1024)        							|
| Softmax				| (1024 x NUM_CLASSES)  						|
 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the cells 7 and 10 of the ipython notebook. 
* Optimizer: Adam
* Default Learning Rate: 1e-3
* Batch size: 32
* Number of epochs: 100

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eighth cell of the Ipython notebook.

* What architecture was chosen?

My architecture is a derivative of the VGG-16 architecture used for the ImageNet dataset. I would add a dropout layer to the fully-connected layer of the network and an l2 weight decay to reduce over-fitting of the model. However, it was not necessary to reach the 93% accuracy threshold for the assignment.

* Why did you believe it would be relevant to the traffic sign application?

This architecture is commonly used for the Street View House Numbers (SVHN), CIFAR-10, and CIFAR-100 datasets. The German Traffic Sign dataset contains (32x32x3) RGB images and between 10-100 classes. Therefore, it is similar to those common datasets.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The training accuracy is 99.9%, so the model was trained for a sufficient amount of epochs. The gap between the training and testing accuracy is about 3-4%. Adding regularization such as Dropout or Weight Decay would improve the gap. 

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.969
* test set accuracy of 0.960
 
#### Precision and Recall 
![alt text][precision_recall]

I plot the precision and recall for each class in the test dataset. It measures how well my network generalizes and how I should augment my training dataset. Overall, my model achieves excellent precision and recall. However, certain classes report poor results. I suspect that augmenting those classes in the training dataset would improve my model's performance. The classes with poor precision or recall correspond with fewer training examples for those classes.
 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] ![alt text][image4] ![alt text][image5]

1. Roundabout Mandatory - An easy example
2. No Entry - The sign is turned toward an oblique angle.
3. Yield - This sign was taken from a street cone. It is one side of a pyramid.
4. Speed Limit (120km/h) - A simple example turned slightly.
5. Turn left Ahead - There are multiple signs in the image.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the twelth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout Mandatory  | Roundabout Mandatory   						| 
| No Entry  			| No Entry										|
| Yield					| Yield											|
| Speed Limit (120km/h)	| Speed Limit (120km/h)					 		|
| Turn Left Ahead		| Speed Limit (80km/h)      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the twelth cell of the Ipython notebook.

The top five softmax probabilities were always 1.0, so the model is very certain about its prediction. Although, it reports the same level of certainty even before training the network.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Roundabout Mandatory   						| 
| 1.0     				| No Entry 										|
| 1.0					| Yield											|
| 1.0	      			| Speed Limit (120km/h)					 		|
| 1.0				    | Speed Limit (80km/h)      					|


#### Visualize the Neural Network's State with Test Images
####1. Discuss how you used the visual output of your trained network's feature map to show that it has learned to look for interesting characteristics in traffic sign images

The code for visualizing the state of my neural network is located in the thirteenth cell of the Ipython notebook.
I visualised the feature activations from the 2nd convolutional layer of the network.

For the first image. I used the third test image - Yield Sign.
The primary features are the triangle outline of the sign and the exclamation point in the center. [See FeatureMaps 11 + 14]

In the second image, I showed the 120km/hr speed limit sign.
The feature activation highlights the number in the center and the circle's outline. [See FeatureMap 8]

![alt text][state]
![alt text][state1]