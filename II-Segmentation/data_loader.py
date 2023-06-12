
import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from time import time
import pandas as pd
import random

import matplotlib.pyplot as plt


data_path = '/dtu/datasets1/02514/PH2_Dataset_images/'
class PH2(torch.utils.data.Dataset):
    def __init__(self, part, transform, data_path=data_path, test_size=0.2):
        'Initialization'
        self.transform = transform
        files = glob.glob(data_path + '*')
        
        np.random.seed(420)
        train_files = np.random.choice(files, int( (1-test_size)  * len(files)), replace=False) 
  
        np.random.seed(420)
        val_files = np.random.choice(train_files, int(test_size  * len(files)), replace=False) 

        test_files = [f for f in files if f not in train_files]
        train_files = [f for f in train_files if f not in val_files ]
        
        if part == 'train':
            files_to_read = train_files
        if part == 'val':
            files_to_read = val_files
        if part == 'test':
            files_to_read = test_files
        
        img_ids = pd.Series(files_to_read).str.split('/').apply(lambda x: x[-1])
        img_path   = data_path + img_ids +'/'+ img_ids + '_Dermoscopic_Image/' + img_ids + '.bmp'
        label_path = data_path + img_ids +'/'+ img_ids + '_lesion/' + img_ids + '_lesion.bmp'
        
        self.image_paths = sorted(img_path)
        self.label_paths = sorted(label_path)
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = np.asarray(Image.open(image_path))
        # taking the first channel as they are the same
        mask = np.asarray(Image.open(label_path)) * 1.0
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']
        

class DRIVE(torch.utils.data.Dataset):
    def __init__(self, part, transform, data_path='/dtu/datasets1/02514/DRIVE/', test_size=0.1):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'training')
        image_paths = sorted(glob.glob(data_path + '/images/*.tif'))
        label_paths = sorted(glob.glob(data_path + '/1st_manual/*.gif'))

        n = len(image_paths)
        
        np.random.seed(420)
        train_files = np.random.choice(image_paths, int(n - n*test_size), replace=False)
        np.random.seed(420)
        train_labels = np.random.choice(label_paths, int(n - n*test_size), replace=False)
        
        
        np.random.seed(420)
        val_files = np.random.choice(train_files, int(test_size  * n), replace=False) 
        np.random.seed(420)
        val_labels = np.random.choice(train_labels, int(test_size  * n), replace=False) 

        test_files = [f for f in image_paths if f not in train_files]
        train_files = [f for f in train_files if f not in val_files ]
        
        test_labels = [f for f in label_paths if f not in train_labels]
        train_labels = [f for f in train_labels if f not in val_labels ]


        
        if part == 'train':
            files_to_read = train_files
            labels_to_read = train_labels 
        if part == 'val':
            files_to_read = val_files
            labels_to_read = val_labels 
        if part == 'test':
            files_to_read = test_files
            labels_to_read = test_labels 
           
        self.image_paths = files_to_read
        self.label_paths = labels_to_read 
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = np.asarray(Image.open(image_path))
        # taking the first channel as they are the same
        mask = np.asarray(Image.open(label_path)).astype(int) /255.0
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']