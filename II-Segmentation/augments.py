import torchvision.transforms as transforms
import torch
from config.config_baseline import IMAGE_SIZE
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random
import torchvision.transforms as T

aug_none = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE), antialias=True),
                               transforms.ToTensor()
                              ])
aug_no_resize = transforms.Compose([
                               transforms.ToTensor()
                              ])

aug_auto = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE), antialias=True),
                               transforms.AutoAugment(),
                               transforms.ToTensor()
                              ])

aug_minor = transforms.Compose([
                                   transforms.Resize((IMAGE_SIZE,IMAGE_SIZE), antialias=True),
#                                   transforms.Normalize(0, 1),
                                   transforms.RandomRotation(degrees = 30),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomVerticalFlip(p=0.5),
#                                   transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,0.9), antialias=False),
#                                   transforms.ColorJitter(brightness=.3, hue=.3),
#                                   transforms.ElasticTransform(alpha=100.0),
#                                   transforms.RandomCrop((224,224), padding=30, padding_mode='reflect'),
#                                   transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
#                                    transforms.RandomEqualize(p=0.3),
#                                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                    transforms.ToTensor()
                                   ])

aug_kuba_train = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Rotate(limit=90, p=1, border_mode=cv2.BORDER_CONSTANT),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.5),
        A.Flip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
        
    ]
)

aug_kuba_test = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), 
                            A.Normalize(),
                            ToTensorV2()])