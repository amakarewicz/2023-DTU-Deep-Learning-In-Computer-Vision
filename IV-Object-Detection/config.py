import torch.nn as nn

NUM_IMAGES = 1500
BATCH_SIZE = 16
IMG_SIZE = 256
EPOCHS = 100
LR = 1e-4
DROP_OUT_RATE = 0.5
LOSS_FN = nn.BCELoss()

data_path = '/dtu/datasets1/02514/data_wastedetection/'
anns_file_path = data_path + '/' + 'annotations.json'


