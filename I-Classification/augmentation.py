import torchvision.transforms as transforms
import torch
from config.baseline_config import IMG_SIZE

transforms_0 = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((IMG_SIZE,IMG_SIZE), antialias=True),
                                   transforms.Normalize(0, 1)
                                  ])
# # augmentation 1
# transforms_1 = transforms.Compose([transforms.ToPILImage(),
#                                    transforms.RandomRotation(degrees = 30),
#                                    transforms.CenterCrop((96,96)),
#                                    transforms.RandomHorizontalFlip(p=0.3),
#                                    transforms.RandomVerticalFlip(p=0.2),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize((0),(1)),
#                                    ])

# # augmentation 2
# transforms_2 = transforms.Compose([transforms.ToPILImage(),
#                                    transforms.RandomCrop((128,128), padding=30, padding_mode='reflect'),
#                                    transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
#                                    transforms.RandomEqualize(p=0.3),
#                                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
#                                    transforms.ToTensor()
#                                    ])

# TRANSFORMS = [transforms_0, transforms_1, transforms_2]