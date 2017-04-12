#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[TrainOccurrencesImage]: ./examples/TrafficSignsOccurrencesInTrainDataSet.png "sign classes occurrences from train data set"
[SampleTrafficSign]: ./examples/SampleTrafficSign.png "signs before grayscale and intensity correction"
[SampleTrafficSignGray]: ./examples/SampleTrafficSignGray.png "signs after grayscale and intensity correction"
[ReflectionOccurrencesImage]: ./examples/TrafficSignsOccurrenceWithReflections.png "sign classes occurrences with reflections"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You are reading my writeup file. My project code is at [project code](https://github.com/jeffpape/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images.
* The size of validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32 pixels x 32 pixels with 3 colors.
* The number of unique classes/labels in the data set is 43 sign classes.

####2. Include an exploratory visualization of the dataset.

The code for the exploratory visualization is contained in code cells from 3 to 5 (In[3] to In[5]) of the IPython notebook.  

10 random images for each sign class are shown in the output of code cell 4 (In[4])
The bar chart shows number of occurrences for each sign type in the training data set.

The number of sign class occurrences versus sign classes is shown below.
![sign class occurrences from train data set][TrainOccurrencesImage]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The code for preprocessing the images is contained in code cells from 6 to 21 (In[6] to In[21]) of the IPython notebook.

I converted the images to grayscale because the signs are not distinctive based on their color, but they are distinctive by their shapes and images. In other words, no 2 sign classes only differ by their colors, but their shapes.

I used the Contrast Limited Adaptive Histogram Equalization (CLAHE) in the skimage.exposure library to adjusted the images' intensities.

Here is an example of a traffic sign image before and after grayscaling and intensity correction.

![sign class before gray scale and intensity correct][SampleTrafficSign]
![sign class after gray scale and intensity correct][SampleTrafficSignGray]

I noticed several sign classes were under represented. I wrote a method, reflect (In[13]), to horizontally, vertically and both reflected some of the sign classes to generate additional images of the same sign class.
The sign classes that are the same when reflected are:

|reflected | sign classes |
|:---:|:---|
| horizontally | 11, 12, 13, 15, 17, 18, 22, 26, 30, 35 |
| vertically | 1, 5, 12, 15, 17
| both | 32, 40 |

Some sign classes become new sign classes when horizontally reflected are:

| source sign class | reflected sign class |
|:---:|:---:|
| 19 | 20 |
| 20 | 19 |
| 33 | 34 |
| 34 | 33 |
| 36 | 37 |
| 37 | 36 |
| 38 | 39 |
| 39 | 38 |

I wrote a method, rotate (In[17]), to slightly rotate the images to generate additional images, but I ran out of time to incorporate the method in the pipeline. Slightly rotating the images +/-10 and +/- 5 degrees would add new images to train with and generate more images for the underrepresented sign classes.

I wrote a method, normalize (In[19]), to normalize the images to prevent neurons from saturating when inputs have varying scale, and to aid generalization.

The number of sign class occurrences with reflections versus sign classes is shown below.

![sign class occurrences with reflections][ReflectionOccurrencesImage]


###2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model consisted of the following layers:

| Layer | Description	|
|:---|:---|
| `Input`  | 32x32x1 grayscale image  |
| `Layer 1`:  | `Input: 32x32x1, Output: 32x32x32`  |
| Convolutional 5x5.  | stride 1x1, output: 32x32x32 |
| RELU.  | |
| Max pooling 2x2.  | stride 2x2, output: 16x16x32 |
| `Layer 2`:  | `Input: 16x16x32, Output: 16x16x64` |
| Convolutional 5x5.  | stride 1x1, output: 16x16x64 |
| RELU.  | |
| Max pooling 2x2.  | stride 2x2, output: 8x8x64 |
| `Layer 3`:  | `Input: 8x8x64, Output: 8x8x128`	 |
| Convolutional 5x5.  | stride 1x1, output: 8x8x128 |
| RELU.  |	|
| Max pooling 2x2.  |	stride 2x2, output: 4x4x128 |
| Dropout.  | keep probability = 0.5 |
| `Flatten.`  | `Input: 4x4x128, Output: 2048.` |
| `Layer 4`:  | `Input: 2048, Output: 1024.` |
| Fully Connected.  |	output 1024 |
| RELU.  |	|
| Dropout.  | keep probability = 0.5 |
| `Layer 5`:  | `Input: 1024, Output: 43` |
| Fully Connected.  | output: 43 |


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I employed a training mechanism similar to that demonstrated in the LeNet lab example.
I added a check to only save the model data when the train accuracy improved from the previous iterations.

To train the model, I used:

| Parameter | Value |
|:---:|:---:|
| learning rate | 0.001 |
| number of epochs | 100 |
| batch size | 256 |
| optimizer | Adam optimizer |
| minimized | cross entropy |



####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My model accuracies are:
* training set accuracy = ?
* validation set accuracy = ?
* test set accuracy = ?

Since this is the first time I have implemented a tensorflow model on my own, I used the LeNet model provided in the course as a starting point.

I arrived at a model with 3 convolution layers because I did not want a drastic change in size of the dimensional size but a gradual change.

I added the dropout layers so that my model would have computational redundancy. This is accomplished by randomly removing some weights in the layer.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
