import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
import numpy as np
import glob
from PIL import Image, ExifTags

# pip install torchsummary
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from time import time
import pandas as pd
import random

import matplotlib.pyplot as plt
import albumentations as A
import cv2

# Obtain Exif orientation tag code
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

from config import *
from utils import *
    

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())


categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']
nr_cats = len(categories)
nr_annotations = len(anns)
nr_images = len(imgs)



image_paths = []
annotations = []
for i in range(NUM_IMAGES):
    img_path, ann = get_image_and_annotations(imgs, anns, i)
    image_paths.append(img_path)
    annotations.append(ann)
    

class TacoDataset(torch.utils.data.Dataset):
    def __init__(self, part, transforms, data_path=data_path, test_size=0.2):
        'Initialization'
        self.transforms = transforms

        # we can simply operate on indices 
        all_indices = np.arange(NUM_IMAGES)
        np.random.seed(420)
        train_indices = np.random.choice(all_indices, int((1-test_size)  * NUM_IMAGES), replace=False) 
  
        np.random.seed(420)
        val_indices = np.random.choice(train_indices, int(test_size  * NUM_IMAGES), replace=False) 

        test_indices = [f for f in all_indices if f not in train_indices]
        train_indices = [f for f in train_indices if f not in val_indices ]
        
        if part == 'train':
            indices_to_read = train_indices
        if part == 'val':
            indices_to_read = val_indices
        if part == 'test':
            indices_to_read = test_indices
        
        img_path = [data_path + image_paths[i] for i in indices_to_read]
        labels = [annotations[i] for i in indices_to_read]
        
        self.image_paths = img_path
        self.annotation_dict = labels
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        annotation_dict = self.annotation_dict[idx]
        
        I = Image.open(image_path)

        
        # Load and process image metadata
        if I._getexif():
            exif = dict(I._getexif().items())
            # Rotate portrait and upside down images if necessary
            if orientation in exif:
                if exif[orientation] == 3:
                    I = I.rotate(180,expand=True)
                if exif[orientation] == 6:
                    I = I.rotate(270,expand=True)
                if exif[orientation] == 8:
                    I = I.rotate(90,expand=True)

        image = np.asarray(I)

        # get bounding box coordinates for each mask
        num_objs = len(annotation_dict)
        boxes = []
        iscrowd = [] 
        for i in range(num_objs):
            pos = annotation_dict[i]
            boxes.append(pos['bbox'])
            iscrowd.append(pos['iscrowd'])
        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.float32)
        
        # binary class here
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            x = self.transforms(image=image,
                                bboxes=target['boxes'],
                                labels=target["labels"])

        return x['image'], x['bboxes'], x['labels']