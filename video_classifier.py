import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

import torchvision
from torchvision import datasets, transforms

from PIL import Image

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('frame_model.pth')
model.eval()

video_list_path = './data/videos'
base_data_path = './data/video_frames/'

classes = ['ballet', 'break', 'cha', 'flamenco', 'foxtrot', 'pasodoble', 'quickstep', 'rumba', 'samba', 'square', 'swing', 'tango', 'tap']

# given a directory of videos,
# return a list of paths to the directories containing those videos' frames
def get_full_data_paths():
  full_data_paths = []
  video_list = os.listdir(video_list_path)
  for video in video_list:
    full_data_paths.append(base_data_path + video[:-4])
  return full_data_paths

# given a PIL image, predict its style
def predict_frame(image):
  # get tensor image from PIL image
  image_transform = transforms.Compose([
    transforms.Resize(36),       # 1920 x 1080 -> 64 x 36
    transforms.CenterCrop(36),   # 64 x 36 -> 36 x 36
    transforms.ToTensor()
  ])
  tensor_image = image_transform(image).float()

  # input/output -> get labels
  input = tensor_image.to(device)
  output = model(input)
  print(outputs)
  _, predicted = torch.max(outputs, 1)
  print(predicted)
  return "hi"
  

# given a path to a directory containing frames of a video,
# predict its dance style
def predict_video(data_path):
  images = os.listdir(data_path)
  labels = []
  for image in images:
    pil_image = Image.open(data_path + "/" + image)
    label = predict_frame(pil_image)
    labels.append(label)
  return labels

for data_path in get_full_data_paths():
  labels = predict_video(data_path)
  print(labels[0:10])