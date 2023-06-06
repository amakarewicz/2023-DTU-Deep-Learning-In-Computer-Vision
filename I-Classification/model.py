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
    
    
class SimpleNET(nn.Module):
    def __init__(self):
        super(SimpleNET, self).__init__()
        
        self.nonLinearActivation = nn.LeakyReLU
        self.dropoutRate = DROP_OUT_RATE
        conv_output_size = int(IMG_SIZE/4)
        
        self.convolutional = nn.Sequential(
                nn.BatchNorm2d(3),
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same'),
                nn.Dropout(self.dropoutRate),
                self.nonLinearActivation(),
                nn.BatchNorm2d(8),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'),
                nn.Dropout(self.dropoutRate),
                self.nonLinearActivation(),
#                 nn.MaxPool2d(kernel_size=2),
#                 nn.BatchNorm2d(16),
#                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
#                 nn.Dropout(self.dropoutRate),
#                 self.nonLinearActivation(),
#                 nn.BatchNorm2d(32),
#                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
#                 nn.Dropout(self.dropoutRate),
#                 self.nonLinearActivation(),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(16)
                )
    
        self.fully_connected = nn.Sequential(
                nn.Linear(conv_output_size*conv_output_size*16, 32),
                nn.Dropout(self.dropoutRate),
                self.nonLinearActivation(),
                nn.BatchNorm1d(32),
                nn.Linear(32, 1),
                nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
    
    
class ResNetBlock(nn.Module):
    def __init__(self, n_features):
        super(ResNetBlock, self).__init__()
        
        self.nonLinearActivation = nn.ReLU
        self.nonLinearActivation_fwd = nn.ReLU()
        
        self.convolution = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, padding = 1),
            self.nonLinearActivation(),
            nn.Conv2d(n_features, n_features, 3, padding = 1)
        )
        
    def forward(self, x):
        out = x + self.convolution(x)
        out = self.nonLinearActivation_fwd(out)
        return out
    
    
class ResNet(nn.Module):
    def __init__(self, n_features, num_res_blocks, n_in=3):
        super(ResNet, self).__init__()
        
        self.dropoutRate=DROP_OUT_RATE
        self.nonLinearActivation = nn.ReLU
        
        conv_layers = [nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        for i in range(num_res_blocks):
            conv_layers.append(ResNetBlock(n_features))
        self.res_blocks = nn.Sequential(*conv_layers)
        
        self.fully_connected = nn.Sequential(
                nn.Linear(IMG_SIZE*IMG_SIZE*n_features, 32),
                nn.Dropout(self.dropoutRate),
                self.nonLinearActivation(),
                nn.BatchNorm1d(32),
                nn.Linear(32, 1),
                nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        out = self.fully_connected(x)
        return out