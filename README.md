
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car_hog.png
[image2]: ./output_images/error_channel.png
[image3]: ./output_images/error_code.png
[image4]: ./output_images/draw_bboxes.png
[image5]: ./output_images/heat_map_result.png
[image6]: ./output_images/label.png
[image7]: ./output_images/heat_map4.png



[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook.  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tested the commonly used image types in fourth code cell, observed that RGB and YCrCb had better HOG characteristics, and at the same time, YCrCb was superior to RGB under the condition of light difference. So I chose YCrCb, Orient, which is commonly used in 9 directions, and the test result is also good.

I have tried different parameters. Pixels_per_cell and cells_per_block will not be able to display features too large, but too small, too much computation and too small is too easy to appear over fitting.

I using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, As in the previous course.

![alt text][image1]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a classifier 7th to 9th code cells. I tried every combination of channels and all channels in `YCrCb`, and found that the accuracy of the combination reached the highest level of `99.4`. The best 0 channels in other single channels were `99.0`, and the other 2 channels were all around `97.6`. But when i use the channel 0 i got the result is like this:
![alt text][image2]
I didn't find this cause, can you analyze it for me.
I only change the code to this:
![alt text][image3]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I found in the height of less than `400` of the area is basically the sky is too small or the vehicle, and more than 656 of the area is the car front cover, so I chose to use the `400 and 656` area for paddling window. 
The vehicle from far to near in the image is from small to large, so need different window sizes to sliding in order to better find vehicle.
Observed by `0.8 to 3.2` in the test images using a total of 24 scale, yeast scale have good effect in between `1.0 to 2.0`, and I used 5 of this 10 scales, i used [x / 10 for x in range(10, 20, 2)]. The result is like:

![alt text][image4]

There are a few frames in which the white car is clearly visible and yet it is not detected. I reduced cells_per_step.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I use the C=0.001 parameter to improve the generalization ability of the image in the LinerSVC function, and the use of `abs(svc.decision_function(test_features)[0] > 1.0)` in the find_cars function to further improve image generalization, while the use of heat map and `threshold = 3` bbox found in the region of the image to reduce the noise in the image, and finally the use of label to find the image region is the location of the vehicle。

![alt text][image5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In my 15th code, I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

![alt text][image6]
![alt text][image5]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the shade of the tree may produce false positive and false positives, sometimes the vehicle did not significantly detected problems, for false positive error, improve Heatmap value, add and use decision_function to solve the problem, not detected for false negative, with lower cells_per_step to increase the detection range



```python

```
