import torchvision.transforms as transforms
import torch
from config.config_baseline import IMAGE_SIZE

aug_none = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE), antialias=True),
                               transforms.ToTensor()
                              ])

aug_auto = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE), antialias=True),
                               transforms.AutoAugment(),
                               transforms.ToTensor()
                              ])

aug_auto = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE), antialias=True),
                               transforms.AutoAugment(),
                               transforms.ToTensor()
                              ])