import torch
from torchmetrics import Metric
import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt
import torch.nn.functional as F

#PyTorch
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer    
        inputs = F.sigmoid(inputs)
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs.to(float), targets.to(float), reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


class Sensitivity(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Convert predictions to binary values (0 or 1)
        preds = torch.round(preds)

        # Calculate true positives and false negatives
        tp = torch.sum((preds >= 0.5) & (target == 1))
        fn = torch.sum((preds < 0.5) & (target == 1))

        # Update state variables
        self.tp += tp
        self.fn += fn

    def compute(self):
        # Calculate sensitivity/recall
        sensitivity = self.tp.float() / (self.tp + self.fn + 1e-12)

        return sensitivity

def visualize_img_and_label(set_):

    
    plt.figure(figsize=(10, 4))
    for i in range(5):
        im, lb = set_.__getitem__(i)
        plt.subplot(2, 5, i+1)
        plt.imshow(im.permute(1,2,0))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 5, 6+i)
        plt.imshow(lb) # lab.permute(1,2,0)
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig('output.png')
    plt.show()
    

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = F.sigmoid(inputs.view(-1)).to(float)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return dice_loss
