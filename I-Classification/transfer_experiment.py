import csv
from torchvision import datasets, models, transforms
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision

from augmentation import *
from train import train
from config.transfer_config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainset = datasets.ImageFolder('/dtu/datasets1/02514/hotdog_nothotdog/train', transform=transforms_0)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

testset = datasets.ImageFolder('/dtu/datasets1/02514/hotdog_nothotdog/test', transform=transforms_0)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)


def transfer_model_set(model, freeze_convs=False,):
    
    if freeze_convs:
        print('Freezing Convs')
        # freeze the feature extractors
        for param in model.parameters():
            param.requires_grad = False
    
    if type(model) == torchvision.models.densenet.DenseNet:
        in_features = model.classifier.in_features
    
    elif type(model) == torchvision.models.resnet.ResNet:
        in_features = model.fc.in_features
    
    
    size_hidden = 512
    out_features = 10
    
    head = nn.Sequential(
                    nn.Linear(in_features, size_hidden),
                    nn.Dropout(DROP_OUT_RATE),
                    nn.ReLU(),
                    nn.BatchNorm1d(size_hidden),
                    nn.Linear(size_hidden, out_features))
    
    if type(model) == torchvision.models.densenet.DenseNet:
        model.classifier = head
    
    elif type(model) == torchvision.models.resnet.ResNet:
        model.fc = head

    else:
        raise Exception('Not implemented the classifier for this type of model')

    model = model.to(device)

    return model


# Just head 

model = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2) # lets take v2 here 

model = transfer_model_set(model, freeze_convs=True)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=HEAD_LEARNING_RATE)

out_dict_resnet50 = train(model, train_loader, test_loader, loss, optimizer, NUM_EPOCHS)

# saving results
optim = 'Adam'
d = out_dict_resnet50
with open(f'results/results_transfer_resnet_head_{NUM_EPOCHS}_epochs_{HEAD_LEARNING_RATE:.0e}_lr_{optim}_optim.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(d.keys())
    writer.writerows(zip(*d.values()))

model = models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)

model = transfer_model_set(model, freeze_convs=True)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=HEAD_LEARNING_RATE)

out_dict = train(model, train_loader, test_loader, loss, optimizer, NUM_EPOCHS)

# saving results
optim = 'Adam'
d = out_dict
with open(f'results/results_transfer_densenet_head_{NUM_EPOCHS}_epochs_{HEAD_LEARNING_RATE:.0e}_lr_{optim}_optim.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(d.keys())
    writer.writerows(zip(*d.values()))


# All 

model = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2) # lets take v2 here 

model = transfer_model_set(model, freeze_convs=False)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=HEAD_LEARNING_RATE)

out_dict_resnet50 = train(model, train_loader, test_loader, loss, optimizer, NUM_EPOCHS)

# saving results
optim = 'Adam'
d = out_dict_resnet50
with open(f'results/results_transfer_resnet_full_{NUM_EPOCHS}_epochs_{HEAD_LEARNING_RATE:.0e}_lr_{optim}_optim.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(d.keys())
    writer.writerows(zip(*d.values()))

model = models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)

model = transfer_model_set(model, freeze_convs=False)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=HEAD_LEARNING_RATE)

out_dict = train(model, train_loader, test_loader, loss, optimizer, NUM_EPOCHS)

# saving results
optim = 'Adam'
d = out_dict
with open(f'results/results_transfer_densenet_full_{NUM_EPOCHS}_epochs_{HEAD_LEARNING_RATE:.0e}_lr_{optim}_optim.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(d.keys())
    writer.writerows(zip(*d.values()))