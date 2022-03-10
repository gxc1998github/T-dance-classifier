import torch
import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()

    # input: 36 x 36 x 3 image
    # filters: 16, 3 x 3 x 3
    # output: 18 x 18 x 16 image
    self.conv_1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
    self.batchnorm_1 = nn.BatchNorm2d(16)

    # input: 18 x 18 x 16 image
    # filters: 32, 3 x 3 x 16
    # output: 9 x 9 x 32 image
    self.conv_2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
    self.batchnorm_2 = nn.BatchNorm2d(32)

    # input: 9 x 9 x 32 image
    # filters: 64, 3 x 3 x 32
    # output: 3 x 3 x 96 image
    self.conv_3 = nn.Conv2d(32, 96, 3, stride=3, padding=1)
    self.batchnorm_3 = nn.BatchNorm2d(96)

    # input: 864 vector
    # output: 16 vector
    self.fully_connected = nn.Linear(864, 16)


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