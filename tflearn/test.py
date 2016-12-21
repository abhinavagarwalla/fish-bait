import tflearn
import scipy.io as io
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import sklearn.preprocessing as pre
import pandas as pd

FISH_CLASSES = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
# Data loading and preprocessing
X = io.loadmat('input.mat')['data']
Y = io.loadmat('input.mat')['label']
Y = np.transpose(Y,(1,0))
Y = pre.LabelBinarizer().fit_transform(Y)


# Building convolutional network
network = input_data(shape=[None, 128, 72, 3], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")	
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 128, 3, activation='relu', regularizer="L2")
network = conv_2d(network, 128, 3, activation='relu', regularizer="L2")	
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 8, activation='softmax')

network = regression(network, optimizer='adagrad', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=1, validation_set = 0.95,
           snapshot_step=100, show_metric=True, batch_size =20, run_id = 'anything')
model.save('model')


test_data = io.loadmat('test.mat')['output_data']
test_preds = model.predict(test_data)
test_files = [im for im in os.listdir('test/')]

submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)
submission.insert(0, 'image', test_files)
submission.head()

