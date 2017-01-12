import random
from train import *

random.seed(1729)

if __name__ == '__main__':
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)
