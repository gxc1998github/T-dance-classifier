import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

import torchvision
from torchvision import datasets, transforms

import frame_nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = './data/datasets/full_set/rgb'
# data_path = './data/datasets/partial_dataset'

def load_datasets():
  # get full dataset
  dataset_transform = transforms.Compose([
    transforms.Resize(36),       # 1920 x 1080 OR 1280 x 720 -> 64 x 36
    transforms.CenterCrop(36),   # 64 x 36 -> 36 x 36
    transforms.ToTensor()
  ])
  dataset = datasets.ImageFolder(data_path, transform=dataset_transform)
  total_size = len(dataset)

  # randomize the indices for sampling from the dataset
  indices = list(range(total_size))
  np.random.shuffle(indices)

  # split the dataset for training (80%) and testing (20%) using random sampling
  midpoint = total_size // 5
  training_sampler = data.sampler.SubsetRandomSampler(indices[midpoint:])
  testing_sampler = data.sampler.SubsetRandomSampler(indices[:midpoint])

  # get training/testing dataset loaders
  training_dataset_loader = data.DataLoader(dataset, sampler=training_sampler)
  testing_dataset_loader = data.DataLoader(dataset, sampler=testing_sampler)

  # return training/testing dataset loaders
  return {'train': training_dataset_loader, 'test': testing_dataset_loader, 'classes': training_dataset_loader.dataset.classes} 

def train(net, dataloader, epochs=1, lr=0.01, momentum=0.9, decay=0.0):
  net.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

  # keep track of losses
  losses = []

  for epoch in range(epochs):
    # iterate over dataset in batches
    for i, batch in enumerate(dataloader, 0):
      # get inputs and corresponding labels
      inputs, labels = batch[0].to(device), batch[1].to(device)

      # set gradients to zero
      optimizer.zero_grad()

      # predict outputs -> calculate loss
      outputs = net(inputs)
      loss = criterion(outputs, labels)

      # optimize
      loss.backward()
      optimizer.step()

      # add loss to list
      losses.append(loss.item())

  # return losses
  return losses

def accuracy(net, dataloader):
  # keep track of correct predictions and total predictions
  correct = 0
  total = 0

  with torch.no_grad():
    # iterate over dataset in batches
    for batch in dataloader:
      # get inputs and corresponding labels
      images, labels = batch[0].to(device), batch[1].to(device)

      # predict outputs -> get most likely outcome
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)

      # increment total predictions
      total += labels.size(0)
 
      # update correct predictions based on results
      correct += (predicted == labels).sum().item()

  # return accuracy as proportion of correct vs incorrect proportions
  return correct / total


###############################################################################
#---------------------------------RUN MODEL-----------------------------------#
###############################################################################

# get training/testing datsets
print("getting datasets...")
data = load_datasets()

# get nn model
print("getting model...")
frame_cnn = frame_nn.CNN()

# train model on training dataset
print("training model...")
frame_losses = train(frame_cnn, data['train'], epochs=1, lr=0.01)
print("done!")

# check accuracy
print("checking accuracy...")
print("Training accuracy: %f" % accuracy(frame_cnn, data['train']))
print("Testing accuracy: %f" % accuracy(frame_cnn, data['test']))

# save model
torch.save(frame_cnn, 'frame_model.pth')