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

video_dir = os.path.join('data', 'videos')
frame_dir = os.path.join('data', 'video_frames')

classes = ['ballet', 'break', 'cha', 'flamenco', 'foxtrot', 'pasodoble', 'quickstep', 'rumba', 'samba', 'square', 'swing', 'tango', 'tap']

# return a list of paths to the directories containing the videos' frames
def get_frame_dirs():
  video_paths = [os.path.join(video_dir, filename) for filename in os.listdir(video_dir)]

  frame_paths = []
  for video_path in video_paths:
    name = "vf_" + os.path.basename(video_path)[:-4]
    save_path = os.path.join(frame_dir, name)
    frame_paths.append(save_path)
  return frame_paths

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
  outputs = model(input[None, ...])
  _, predicted = torch.max(outputs, 1)
  return classes[predicted[0]]
  

# given a path to a directory containing frames of a video,
# predict its dance style
def predict_video(frame_dir):
  print("using frames from", frame_dir)
  frames = os.listdir(frame_dir)
  labels = []
  for frame in frames:
    pil_frame = Image.open(os.path.join(frame_dir, frame))
    label = predict_frame(pil_frame)
    labels.append(label)
  return labels

for frame_dir in get_frame_dirs():
  video_name = os.path.basename(frame_dir).replace("_", " ").replace("vf", "").strip()

  labels = predict_video(frame_dir)
  # most_common_label = max(set(labels), key=labels.count)

  for label in np.unique(labels):
    num_curr_label = labels.count(label)
    label_perc = labels.count(label) / len(labels)
    print("%s: %.4f" % (label, label_perc))