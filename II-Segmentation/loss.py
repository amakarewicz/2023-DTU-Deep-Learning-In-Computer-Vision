import torch
import torch.nn
import torch.nn.functional as F

def dice_loss(y_real, y_pred):
    y_pred = F.sigmoid(y_pred)
    print(y_pred.max())
    num = torch.mean(2.*y_real*y_pred+1)
    denom = torch.mean(y_real+y_pred)+1
    return 1- (num/denom)


def IoU(y_real, y_pred):
    intersection = torch.logical_and(y_real, y_pred)
    union = torch.logical_or(y_real, y_pred)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score