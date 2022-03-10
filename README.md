# Dance Classifier
Deepti Ramani, Gareth Coad, Dylin Stoen

---

## Abstract
In this project, we analyze videos of dance routines in order to classify them with the style/type of dance that they most closely fall under. 

We use a convolutional neural network to classify individual dance frames, then run that network over the frames of the video being analyzed in order to determine its overall style. We also explore different ways of retrieving frames from the videos to determine which method results in the most accurate overall classification.

REWRITE!!!

---

## Problem
For our project, we wanted to be able to classify a dance video with one of 16 different dance styles based on the movements performed in the video. We broke this down into three subproblems: classifying individual movements with a dance style, breaking the video down into frames, and classifying the video based on what category the majority of its frames fall into.

---

## Motivation
When initially discussing ideas for the project, we were planning to try the 'Bird Classification Competition' from Kaggle, since it provided an organized dataset that would allow us to explore image classification without worrying about wrangling messy or incomplete data.

However, we were also interested in image classification that could have more real-world applications, which is when we began looking into motion classification, and how we could classify different movements, like dances. We were able to find a large, well-organized dataset of dance movements, sorted into 16 categories, which allowed us to pursue this project without worrying about having to manually gather and process a dataset.

---

## Datasets
### Dataset 1: Individual movements
The first dataset we used is the *Let's Dance: Learning From Online Dance Videos* dataset from Georgia Tech. This dataset is a collection of over 300,000 high resolution images of various dance movements, which we used to train our model for classifying individual movements.
The images are sorted into 16 categories:
* Ballet (22,419 images)
* Break dance (25,622 images)
* Cha (28,098 images)
* Flamenco (24,755 images)
* Foxtrot (23,750 images)
* Jive (29,100 images)
* Latin dance (24,460 images)
* Pasodoble (26,607 images)
* Quickstep (24,042 images)
* Rumba (27,264 images)
* Samba (25,815 images)
* Square dance (27,453 images)
* Swing (26,337 images)
* Tango (24,020 images)
* Tap dance (28,544 images)
* Waltz (24,381 images)

We chose this dataset because the dataset was extensive, with many images for each category of dance, and because the images were pre-sorted into categories, and had the same aspect ratio.

However, we found that using the full datatset to test and train our model was inefficient: not only was loading the dataset time-consuming due to the large number of images, but training the model on those images was also very slow due to the large size of the images.  
To make loading the dataset more efficient, we used a bash script to sample only a portion of the images from the larger dataset (this script was run 16 times, replacing `category_name` with the name of the folder corresponding to a different cateory each time):
```
for file in `find ./rgb/category_name -type f | awk 'NR %28 == 0'`; do cp $file ./partial_dataset/category_name ; done
```
This gave us a dataset containing 14,730 images, sampled uniformly from the original dataset (not in clumps to avoid skewed data).

To make training the model more efficient, we scaled the images down to 640x360. Because every image had the same aspect ratio, regardless of size, this was fairly straightforward, and we didn't need to worry about any warping or distortion.  
Additionally, we noticed that the model was significantly more efficient with square images, so we also cropped the images (centered) to 360x360.

### Dataset 2: Full videos
The second dataset we used is a custom dataset that we created by retrieving and processing dance videos from Youtube.

We began by gathering a variety of dancing videos that fell both inside and outside of the 16 categories of dance that are included in our training dataset.
Using the pytube library, we downloaded Youtube videos as mp4 files with the desired resolution and frame rate. We decided on resolution of 1920x1080, which matched the aspect ratio and quality of the images in the training dataset, which would make it easier to run our model on them.

We also used the OpenCV library was then to split the newly downloaded video into indivual .png files, thus forming our testing set. We wanted to explore different ways of selecting the frames to use for the dataset in order to best split a video into its individual movements.
We considered selecting every frame or every n-th frame of the video, in addition to selecting frames of the video based on how different they were from the previously selected frame. The reasoning behind the last choice was that the frames for a single dance movement would all be relatively similar to each other, so we could determine when the video showed a different move by looking for frames that were above a certain threshold in terms of difference.

---

## Technique

---

## Additional Info

---

## References
* Code for classifying individual movements was adapted from a [PyTorch tutorial](https://github.com/pjreddie/uwimg/blob/main/tutorial1%20-%20pytorch-introduction.ipynb) for CSE 455 (Computer Vision) by Joe Redmon
* The training data for the movement classifier was taken from the [Let's Dance: Learning From Online Dance Videos dataset](https://www.cc.gatech.edu/cpl/projects/dance/) from Georgia Tech

## Demo video
