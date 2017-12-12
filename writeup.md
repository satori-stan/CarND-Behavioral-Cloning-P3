# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_driving.jpg "Center driving"
[image2]: ./images/center_driving_opposite.jpg "Center driving, opposite direction"
[image3]: ./images/recovery_left.jpg "Recovery Image from left side of track"
[image4]: ./images/recovery_right.jpg "Recovery Image from right side of track"
[image5]: ./images/track2_1.jpg "Driving in track 2"
[image6]: ./images/track2_2.jpg "More driving in track 2"
[image7]: ./images/polar.jpg "Vector addition"
[image8]: ./images/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.hd5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
The model.py file can be executed as
```sh
python clone.py data_folder/ model_name
```
The first parameter is the folder where a driving_log.csv file and IMG folders can be found. These files are the usual output of a training session of the provided simulator. The second parameter is the name with which the model will be saved. A third, named, parameter is available `--keep_one_in` that allows skipping some lines from the driving log to speed learning up. A last named parameter `--base_model` can be used to provide an existing model for finetuning.

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.hd5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is configured as a residual network. It consists of blocks of 32 3x3 asymetric bottleneck convolutions each followed by a 2x2 MaxPool operation. The blocks start with a 20% dropout on the input that is then passed to a downsampling 16 channel 1x1 convolution. A 3x1 and 1x3 convolution pair is then applied for the same 16 channels, before the data is upsampled to 32 channels by another 1x1 convoution. The last step is merging the input to the result of the convolutional block. All convolutions use ReLU activations and have bias added. Additionally, the asymetric convolution pair uses "same" padding to preserve the shape of the data. This section can be seen in the file clone.py (lines 123-138).

At the start of the network, the data is first cropped to a size of 75x320x3 to get rid of data which is not considered relevant to the calculation (line 141).

The data is then normalized in the model using Batch Normalization (code line 142). Batch normalizaton was preferred over global normalization to keep the calculations of each batch within the same range, which could provide a slight calculation benefit over the alternative.

An initial 32 1x1 convolution is used to have the data match the size of the residual block layer.

Finally, there is a fully connected layed of 64 nodes with ReLU activation, before the final fully connected layer of 1 node.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting inside every residual block (clone.py line 133). Initially an extra dropout layer was used in the fully connected layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 71). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (clone.py line 163). I did try a callback to reduce the Learning Rate by one fifth when the validation loss plateaued but I failed to see any improvement and removed it.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and images from the right and left camera with adjusted angles to augment the data set.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a residual network, which has proved to be a superior architecture and reasonably fast. To implement it, I looked at the source code for the Keras implementation of ResNet.

My first model had 3 residual blocks of 1x1 convolutions and 64 channel depth with 20% dropout and 4x4 maxpools after each block. 1x1 convolutions are used to have higher non-linearity and make the network faster by having less parameters per convolution. With this model, I had a hard time getting loss values under 0.05 for both training and validation sets.

When driving the car with this model, it was not recovering enough in the curves, so I augmented the dataset with the left and right images; compensating the angle with a trigonometric function that takes the (approximated) distance between cameras as a hyper parameter. Although the car kept more to the center, I still had issues with curves.

The next model had 5 residual blocks. The first problem was that it was no longer possible to use the 4x4 maxpool layers since the dimensions would drop too much after the third pooling, so I changed them for 2x2 maxpool layers. I also ran out of resources when running this architecture in a GPU, so I changed the convolution filters to 32. These changes made it possible to drop the losses to the 0.03 range and the car drove around almost perfectly but there were two curves where the car drove too close to the edge and I had to further improve the model. The problem now was the time taken to train with the 214,000 example size.

At this stage I was starting to question my original model idea and tried out the NVidia model. After a couple of tests, the result wasn't much better and the model had about 10x more parameters. I figured I was missing something in my model.

I added the bottleneck convolutions to increase the perceptive field of my network without increasing the parameter numbers too much. The idea came from the CS231n lectures on the practical aspects of implementing CNNs. I even added an extra residual block to make the network deeper.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle had trouble in previous attempts. Namely in the first curve after the bridge, where the road border changes from a curb to dirt. It also had problems the next curve after that where the water is directly in front. It seemed to make the model believe that the was more road ahead. To improve the driving behavior in these cases, I made recordings of only these sections where the model was having trouble and the new model drove around them perfectly.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. This means it is inside the yellow lines all the time and it never goes over the ledges.

#### 2. Final Model Architecture

The final model architecture (clone.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer              |     Description                                                   |
|:------------------:|:-----------------------------------------------------------------:|
| Input              | 160x320x3 RGB image                                               |
| Cropping           | Crop 60 pixels from the top and 25 off the bottom to get 75x320x3 |
| BatchNormalization |                                                                   |
| Convolution 1x1    | 1x1 stride, same padding, 32 filters outputs 75x320x32            |
| Dropout            | 20%                                                               |
| Convolution 1x1    | 1x1 stride, valid padding, 16 filters outputs 75x320x16           |
| Convolution 3x1    | 1x1 stride, same padding, 16 filters outputs 75x320x16            |
| Convolution 1x3    | 1x1 stride, same padding, 16 filters outputs 75x320x16            |
| Convolution 1x1    | 1x1 stride, valid padding, 32 filters outputs 75x320x32           |
| Merge              | Sum the input to the last dropout and the bottleneck result       |
| Max pooling        | 2x2 stride,  outputs 38x160x32                                    |
| Dropout            | 20%                                                               |
| Convolution 1x1    | 1x1 stride, valid padding, 16 filters outputs 38x160x16           |
| Convolution 3x1    | 1x1 stride, same padding, 16 filters outputs 38x160x16            |
| Convolution 1x3    | 1x1 stride, same padding, 16 filters outputs 38x160x16            |
| Convolution 1x1    | 1x1 stride, valid padding, 32 filters outputs 38x160x32           |
| Merge              | Sum the input to the last dropout and the bottleneck result       |
| Max pooling        | 2x2 stride,  outputs 19x80x32                                     |
| Dropout            | 20%                                                               |
| Convolution 1x1    | 1x1 stride, valid padding, 16 filters outputs 19x80x16            |
| Convolution 3x1    | 1x1 stride, same padding, 16 filters outputs 19x80x16             |
| Convolution 1x3    | 1x1 stride, same padding, 16 filters outputs 19x80x16             |
| Convolution 1x1    | 1x1 stride, valid padding, 32 filters outputs 19x80x32            |
| Merge              | Sum the input to the last dropout and the bottleneck result       |
| Max pooling        | 2x2 stride,  outputs 10x40x32                                     |
| Dropout            | 20%                                                               |
| Convolution 1x1    | 1x1 stride, valid padding, 16 filters outputs 10x40x16            |
| Convolution 3x1    | 1x1 stride, same padding, 16 filters outputs 10x40x16             |
| Convolution 1x3    | 1x1 stride, same padding, 16 filters outputs 10x40x16             |
| Convolution 1x1    | 1x1 stride, valid padding, 32 filters outputs 10x40x32            |
| Merge              | Sum the input to the last dropout and the bottleneck result       |
| Max pooling        | 2x2 stride,  outputs 5x20x32                                      |
| Dropout            | 20%                                                               |
| Convolution 1x1    | 1x1 stride, valid padding, 16 filters outputs 5x20x16             |
| Convolution 3x1    | 1x1 stride, same padding, 16 filters outputs 5x20x16              |
| Convolution 1x3    | 1x1 stride, same padding, 16 filters outputs 5x20x16              |
| Convolution 1x1    | 1x1 stride, valid padding, 32 filters outputs 5x20x32             |
| Merge              | Sum the input to the last dropout and the bottleneck result       |
| Max pooling        | 2x2 stride,  outputs 3x10x32                                      |
| Dropout            | 20%                                                               |
| Convolution 1x1    | 1x1 stride, valid padding, 16 filters outputs 3x10x16             |
| Convolution 3x1    | 1x1 stride, same padding, 16 filters outputs 3x10x16              |
| Convolution 1x3    | 1x1 stride, same padding, 16 filters outputs 3x10x16              |
| Convolution 1x1    | 1x1 stride, valid padding, 32 filters outputs 3x10x32             |
| Merge              | Sum the input to the last dropout and the bottleneck result       |
| Max pooling        | 2x2 stride,  outputs 2x5x32                                       |
| Fully connected    | 64 units + bias and RELU                                          |
| Fully connected    | 1 unit + bias                                                     |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded six laps on track one using center lane driving (although my driving could use some improvement). Here is an example image of center lane driving:

![Center lane driving from central car camera][image1]

I then recorded four laps of center lane driving in the opposite direction. Here is an example:

![Center lane driving in opposite direction from central car camera][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to the center if it ever found itself close to the edges. I did one full lap in the proper direction swerving around and turning the recording off for the portions where the car was leaving the center and restarting it when the car was on or close to the edge and was comming back. These images show what a recovery looks like:

![Recovery driving from left side of the lane from central car camera][image3]
![Recovery driving from right side of the lane from central car camera][image4]

I recorded at least 8 more full laps of the first track in the normal direction and half as many in the opposite direction. I also recorded additional data for sections that were more difficult for the model, like the bridge and the curves with dirt edges and the curve that drives away from the central lake.

Then I repeated this process on track two in order to get more data points. In this track, I attempted to stay on the right lane the whole time. The results can be seen when the car is driving around track one and it is mostly hanging to the right. One of the clearest differences was the need to break for some parts of the track and the magnitude of the steering angle. Here are some images of driving in track two:

![Right lane driving from central car camera in second track][image5]
![Right lane driving from central car camera in second track, different section][image6]

After the collection process, I had close to 40K data points. I then preprocessed this data by calculating adjusted angles for the left and right captures. This was done with the formula: `ATAN(Y/((Y/TAN(B*PI()/180))+W))*180/PI()` where:
* Y is a constant hyper parameter. A reference point in the horizon (a y axis when looking down on the car on the track from above). It was set to 75 pixels.
* B is the steering angle from the central camera.
* W is the distance between the central camera to the left/right camera. It was set to 65 pixels as measured by the offset of some relevant track features between the three captures.
* PI() is the mathematical constant Pi (3.1415...), used to convert the degrees to radians.
* TAN is the tangent of the angle.
* ATAN is the inverse tangent of the angle.

This formula was derived by assuming a vector X with given magnitude and angle can be calculated as the sum of another vector Y and a third one Z. The angle of X is the steering angle. The magnitude of X can be derived by choosing a point along the steering direction. The magnitude of Z is the distance between cameras and its angle is zero. With this information, the magnitude and angle of the missing vector Y can be calculated.

![A single vector with magnitude and angle can be calculated as the sum of two other vectors][image7]

To augment the data set, I also flipped images and angles thinking that this would help the model generalize better. For example, here is an image that has then been flipped:

![Flipped central driving from central car camera][image8]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was set to 30 with a batch size of 40K examples. Less examples per epoch would not yield relevant improvements to the validation loss. More examples would take a long time to train, without the ability to save checkpoints. It was possible to continue the training after 30 epochs and still see improvements, but the code includes a feature to resume a previous learning session if necessary. Thirty epochs would take a brand-new training session from infinite loss to close to the final one.
