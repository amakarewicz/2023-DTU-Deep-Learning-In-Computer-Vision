
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

        train_files = [f for f in files if f not in val_files ]
        test_files = [f for f in files if f not in list(train_files) + list(val_files) ]
        

        
        if part == 'train':
            files_to_read = train_files
        elif part == 'val':
            files_to_read = val_files
        else:
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
    def __init__(self, train, transform, data_path='/dtu/datasets1/02514/DRIVE/'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'training' if train else 'test')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.tif'))
        self.label_paths = sorted(glob.glob(data_path + '/mask/*.gif'))

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y