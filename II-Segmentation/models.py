import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config_baseline import IMAGE_SIZE

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 1, 3, padding=1)
        
    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(e0))
        e2 = F.relu(self.enc_conv2(e1))

        # no activation
        e3 = self.enc_conv3(e2)
        
        return e3

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(IMAGE_SIZE//8)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(IMAGE_SIZE//4)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(IMAGE_SIZE//2)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(IMAGE_SIZE)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        #print(f"e0 shape: {e0.shape}")
        #print(f"e1 shape: {e1.shape}")
        #print(f"e2 shape: {e2.shape}")
        #print(f"e3 shape: {e3.shape}")

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))
        #print(f"b shape: {b.shape}")

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        #print(f"d0 shape: {d0.shape}")
        d0 = torch.cat([d0, e2], 1)
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d1 = torch.cat([d1, e1], 1)
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d2 = torch.cat([d2, e0], 1)
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        #d3 = torch.cat([d3, x], 1)
        return F.sigmoid(d3)