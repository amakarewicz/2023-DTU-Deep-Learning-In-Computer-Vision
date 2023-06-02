import torch
import torch.nn as nn
from config.baseline_config import *


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.nonLinearActivation = nn.LeakyReLU
        self.dropoutRate = DROP_OUT_RATE
        conv_output_size = int(IMG_SIZE/4)
        
        self.convolutional = nn.Sequential(
                nn.BatchNorm2d(3),
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same'),
                nn.Dropout(self.dropoutRate),
                self.nonLinearActivation(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
                nn.Dropout(self.dropoutRate),
                self.nonLinearActivation(),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
                nn.Dropout(self.dropoutRate),
                self.nonLinearActivation(),
                nn.BatchNorm2d(64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
                nn.Dropout(self.dropoutRate),
                self.nonLinearActivation(),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(64)
                )
    
        self.fully_connected = nn.Sequential(
                nn.Linear(conv_output_size*conv_output_size*64, 500),
                nn.Dropout(self.dropoutRate),
                self.nonLinearActivation(),
                nn.BatchNorm1d(500),
                nn.Linear(500, 10))
#                 nn.Softmax(dim=1))    
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x