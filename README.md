# Dance Classifier
Deepti Ramani, Gareth Coad

---

## Abstract
In this project, we analyze videos of dance routines in order to classify them with the style/type of dance that they most closely fall under. 

We use a convolutional neural network to classify individual dance frames, then run that network over the frames of the video being analyzed in order to determine its overall style. We also explore different ways of retrieving frames from the videos to determine which method results in the most accurate overall classification.

Our project was ultimately unsuccessful: our model was unable to classify dance videos with substantial accuracy. In the future, we plan to train our model with a more extensive dataset to see if that improves the results at all.

---

## Problem
For our project, we wanted to be able to classify a dance video with one of 16 different dance styles based on the movements performed in the video. We broke this down into three subproblems:

  (1) Classifying individual movements by dance style with the use of a neural network, then training a model with a pre-existing data set  
  (2) Collecting a video from YouTube and breaking it into individual frames, allowing us to form our own data set  
  (3) Classifying the video data with our model and outputting a suggested dance style based on which category the majority of its frames fall into

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

However, we found that using the full datatset to test and train our model was inefficient: due to the large number of images in the dataset, and the high resolution of those images, it took a very long time to train and test the model (> 3 hours).  
To reduce the number of images in the dataset, we used a bash script to sample only a portion of the images from the larger dataset (this script was run 16 times, replacing `category_name` with the name of the folder corresponding to a different cateory each time):
```
for file in `find ./rgb/category_name -type f | awk 'NR %28 == 0'`; do cp $file ./partial_dataset/category_name ; done
```
This gave us a dataset containing 14,730 images, sampled uniformly from the original dataset (we chose not to sample in clumps in order to retrieve a variety of image data).
To reduce the resolution of the images, we scaled the images down to 64x36. Because every image had the same aspect ratio, regardless of size, this was fairly straightforward, and we didn't need to worry about any warping or distortion. Additionally, we noticed that the model was significantly more efficient with square images, so we also cropped the images (centered) to 36x36.

### Dataset 2: Full videos
The second dataset we used is a custom dataset that we created by retrieving and processing dance videos from Youtube.

We began by gathering a variety of dancing videos that fell both inside and outside of the 16 categories of dance that are included in our training dataset.
Using the pytube library, we downloaded Youtube videos as mp4 files with the desired resolution and frame rate. We decided on resolution of 1920x1080, which matched the aspect ratio and quality of the images in the training dataset, which would make it easier to run our model on them.

We also used the OpenCV library was then to split the newly downloaded video into indivual jpg files, thus forming our testing set. We wanted to explore different ways of selecting the frames to use for the dataset in order to best split a video into its individual movements. We ended up splitting the video by frame (keep every 40th frame)

---

## Techniques
### Neural Network
We used a convolutional neural network for this project. The model was trained using a partial version of the Let's Dance dataset, with the training code adapted from a PyTorch tutorial by Joe Redmon (see References for more details). The design of the neural network can be found in the `frame_nn` python script, and the training and testing code can be found in the `frame_classifier` python script.

`frame_nn` shows the break down of the neural network layers and the downsizing and reshaping that occurs at each filter. We attempted to optimize the layer count and the reshaping that occured but due to time constraints and the data set sizes this was a challenge and would need revisiting in future work.  
`frame_classifer` handles the majority of processing, by receiving the data, forming the model through training on the dancing data set, and finally saving the model for later use

### Video Processing

Our video processing was handled by the `video_processor` python script. The script consists of three functions: 

  (1) `download_video`, which accepts the URL of a YouTube video and processes the video with the aid of the `pytube` library, then downloads it as a 1080p mp4 file with a formatted filename  
  (2) `create_dir`, which accepts a path and creates a directory at the given path  
  (3) `save_frame`, which accepts a path to a video and a chosen gap between each saved frame, creates a directory to store the frames in, and saves every `gap`-th frame as a 1920x180 jpg image

### Generating Classification Output

The `video_classifier` python script is the final step of our project and uses the trained model to discern if an inputted YouTube video is one of our 16 dance styles. It is made up of three functions:

  (1) `get_frame_dirs`, which retrieves the paths to the directories that contain the frames for each saved video  
  (2) `predict_frame`, which takes a PIL image and returns its most likely classification (dance move) by running the saved model on it  
  (3) `predict_video`, which takes in a directory to the frames of a video and classifies the video with a dance style based on the most common classification of its frames

---

## Evaluation
Overall, the model was highly inaccurate: when calculating the accuracy of the model in `frame_classifier`, it had a training accuracy of ~0.21 and a testing accuracy of ~0.14. These are very low numbers, and are likely due to having to run the model on a partial version of the dataset with severely scaled-down images. Had we had the time to train the model on a more extensive dataset with higher resolution images, we believe the model would have performed significantly better, and plan to continue testing this theory in the future.

Additionally, when running the video classifier using the aforementioned model on our downloaded video samples, all of the videos were incorrectly classified.

---

## Additional Info
A lot of the time on this project was spent figuring out how to minimize runtime while maximizing accuracy. Ideally, we would train and test our model on the full dataset with relatively high-resolution images; however, due to our technical limitations, we decided to sacrifice accuracy in order to finish the project on time.

---

## References
* Code for classifying individual movements was adapted from a [PyTorch tutorial](https://github.com/pjreddie/uwimg/blob/main/tutorial1%20-%20pytorch-introduction.ipynb) for CSE 455 (Computer Vision) by Joe Redmon
* The training data for the movement classifier was taken from the [Let's Dance: Learning From Online Dance Videos dataset](https://www.cc.gatech.edu/cpl/projects/dance/) from Georgia Tech

---

## Demo video
Demo video can be found [here](https://github.com/deeptii-20/cse455-final-project/blob/main/cse455-final-project-video.mp4).

---

## Code
Code for this project can be found [here](https://github.com/deeptii-20/cse455-final-project).
