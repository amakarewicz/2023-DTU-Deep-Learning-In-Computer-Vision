import torch.nn as nn

NUM_IMAGES = 1500
IMG_SIZE = 256
BATCH_SIZE = 2
DROP_OUT_RATE = 0.5
HEAD_LEARNING_RATE = 0.001
NUM_EPOCHS = 5
DROP_OUT_RATE = 0.5
LOSS_FN = nn.BCELoss()

data_path = '/dtu/datasets1/02514/data_wastedetection/'
anns_file_path = data_path + '/' + 'annotations.json'


