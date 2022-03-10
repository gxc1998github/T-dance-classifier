import torch
import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()

    # input: 640 x 360 x 3 image
    # filters: 16, 3 x 3 x 3
    # output: 320 x 180 x 16 image
    self.conv_1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
    self.batchnorm_1 = nn.BatchNorm2d(16)

    # input: 320 x 180 x 16 image
    # filters: 32, 3 x 3 x 16
    # output: 180 x 180 x 32 image
    self.conv_2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
    self.batchnorm_2 = nn.BatchNorm2d(32)

    # input: 160 x 90 x 32 image
    # filters: 64, 3 x 3 x 32
    # output: 80 x 45 x 64 image
    self.conv_3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
    self.batchnorm_3 = nn.BatchNorm2d(64)

    # input: 230400 vector
    # output: 16 vector
    self.fully_connected = nn.Linear(230400, 16)


  def forward(self, x):
    x = self.conv_1(x)
    x = self.batchnorm_1(x)
    x = f.relu(x)

    x = self.conv_2(x)
    x = self.batchnorm_2(x)
    x = f.relu(x)

    x = self.conv_3(x)
    x = self.batchnorm_3(x)
    x = f.relu(x)

    x = torch.flatten(x, 1)
    x = self.fully_connected(x)
    return x