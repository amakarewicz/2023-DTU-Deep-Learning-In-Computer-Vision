import torch
import torch.nn
import torch.nn.functional as F

def dice_loss(y_real, y_pred):
    y_pred = F.sigmoid(y_pred)
    num = torch.mean(2.*y_real*y_pred+1)
    denom = torch.mean(y_real+y_pred)+1
    return 1- (num/denom)

def focal_loss(y_real, y_pred, gamma=2):
    y_real = y_real.view(-1)
    y_pred = y_pred.view(-1)

    left = (1-F.sigmoid(y_pred))**gamma*y_real*torch.log(F.sigmoid(y_pred))
    right = (1-y_real)*torch.log(1-F.sigmoid(y_pred))

    return -1 * (left+right).sum()

def cross_entropy():
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).cuda())
    return loss_fn

def IoU(y_real, y_pred):
    intersection = torch.logical_and(y_real, y_pred)
    union = torch.logical_or(y_real, y_pred)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score

def InvIoU(y_real, y_pred):
    return 1-IoU(y_real, y_pred)