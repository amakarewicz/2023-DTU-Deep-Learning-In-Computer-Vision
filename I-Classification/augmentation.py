import torchvision.transforms as transforms
import torch
from config.baseline_config import IMG_SIZE

# transforms_0 = transforms.Compose([transforms.ToTensor(),
#                                    transforms.Resize((IMG_SIZE,IMG_SIZE), antialias=True),
#                                    transforms.Normalize(0, 1)
#                                   ])

transforms_0 = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((IMG_SIZE,IMG_SIZE), antialias=True),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # For the Imagenet Transfer learning
                                  ])


# transforms_1 = transforms.Compose([transforms.ToTensor(),
#                                    transforms.Resize((IMG_SIZE,IMG_SIZE), antialias=True),
# #                                    transforms.Normalize(0, 1),
#                                    transforms.RandomRotation(degrees = 30),
# #                                    transforms.CenterCrop((128, 128)),
#                                    transforms.RandomHorizontalFlip(p=0.5),
#                                    transforms.RandomVerticalFlip(p=0.5),
# #                                    transforms.ToTensor(),
#                                    transforms.Normalize(0, 1),
#                                    ])
# transforms_1_test = transforms.Compose([transforms.ToTensor(),
#                                    transforms.Resize((IMG_SIZE,IMG_SIZE), antialias=True),
#                                                                            transforms.Normalize(0, 1)])

# augmentation 1
transforms_1 = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((IMG_SIZE,IMG_SIZE), antialias=True),
                                   transforms.Normalize(0, 1),
                                   transforms.RandomRotation(degrees = 30),
                                   transforms.RandomHorizontalFlip(p=0.5),
#                                    transforms.RandomVerticalFlip(p=0.5),
                                   transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,0.9), antialias=False),
                                   transforms.ColorJitter(brightness=.3, hue=.3),
#                                    transforms.ElasticTransform(alpha=100.0),
#                                    transforms.RandomCrop((224,224), padding=30, padding_mode='reflect'),
                                   transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
#                                    transforms.RandomEqualize(p=0.3),
#                                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
#                                    transforms.ToTensor()
                                   ])

transforms_1_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((IMG_SIZE,IMG_SIZE), antialias=True),
                                        transforms.Normalize(0, 1)
                                       ])

transforms_2 = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE), antialias=True),
                                   transforms.AutoAugment(),
                                   transforms.ToTensor()
                                  ])

transforms_2_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((IMG_SIZE,IMG_SIZE), antialias=True)
                                       ])

TRANSFORMS = [transforms_0, transforms_1, transforms_2]
TEST_TRANSFORMS = [transforms_0, transforms_1_test, transforms_2_test]