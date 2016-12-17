import random
from train import *

batch_size = 1024
nb_epoch = 80
data_augmentation = True
num_folds = 2

img_channels = 3
img_size = 32
random.seed(1729)

if __name__ == '__main__':
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)
