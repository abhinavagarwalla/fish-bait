"==========loading libraies========="

# print "loading libraries"
import numpy as np
import cv2
import os
import scipy.io as io
import tensorflow as tf
import math
from sklearn.metrics import log_loss

"========== Constants ============="

FISH_CLASSES = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
CHANNELS =3
HEIGHT = 720
WIDTH = 1280
TRAIN_DIR = 'train/'
TEST_DIR = 'test/'
ROWS = WIDTH/10
COLS = HEIGHT/10
conv1_channels = 32
conv2_channels = 64
conv3_channels = 128
KERNEL_SIZE = 3
NUM_CLASSES = len(FISH_CLASSES)
fc1_units = 256
fc2_units = 64
batch_size = 10

"=========== Network ============="


def inference(images, conv1_channels, conv2_channels, fc1_units, fc2_units):
    
    # Conv 1
    with tf.variable_scope('h_conv_1') as scope:
        weights = tf.get_variable('weights', shape=[KERNEL_SIZE+2, KERNEL_SIZE+2, CHANNELS, conv1_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[conv1_channels], initializer=tf.constant_initializer(0.05))
        z = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='VALID')
        h_conv1 = tf.nn.relu(z+biases, name=scope.name)
        # print h_conv1
    
    # # Maxpool 1
    # h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')
    
    # Conv2
    with tf.variable_scope('h_conv_2') as scope:
        weights = tf.get_variable('weights', shape=[KERNEL_SIZE, KERNEL_SIZE, conv1_channels, conv1_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[conv1_channels], initializer=tf.constant_initializer(0.05))
        z = tf.nn.conv2d(h_conv1, weights, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = tf.nn.relu(z+biases, name=scope.name)
        # print h_conv2

    # Maxpool 2
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
    # print (h_pool2)
    
    # Conv 3
    with tf.variable_scope('h_conv_3') as scope:
        weights = tf.get_variable('weights', shape=[KERNEL_SIZE, KERNEL_SIZE, conv1_channels, conv2_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[conv2_channels], initializer=tf.constant_initializer(0.05))
        z = tf.nn.conv2d(h_pool2, weights, strides=[1, 1, 1, 1], padding='VALID')
        h_conv3 = tf.nn.relu(z+biases, name=scope.name)
        # print h_conv3

    # Conv4
    with tf.variable_scope('h_conv_4') as scope:
        weights = tf.get_variable('weights', shape=[KERNEL_SIZE, KERNEL_SIZE, conv2_channels, conv2_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[conv2_channels], initializer=tf.constant_initializer(0.05))
        z = tf.nn.conv2d(h_conv3, weights, strides=[1, 1, 1, 1], padding='SAME')
        h_conv4 = tf.nn.relu(z+biases, name=scope.name)
        # print h_conv4

    # Maxpool 4
    h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
    # print h_pool4
    # h_pool4_flat = tf.reshape(h_pool4, [-1])
    # size_after_flatten = int(h_pool4_flat.get_shape()[0])
    h_pool4_flat = tf.contrib.layers.flatten(h_pool4)
    dim = h_pool4_flat.get_shape()[1].value
    # print h_pool4_flat

    # FC 1
    with tf.variable_scope('h_FC1') as scope:
        weights = tf.Variable(tf.truncated_normal([dim, fc1_units], stddev=1.0 / math.sqrt(float(dim))), name='weights')
        biases = tf.Variable(tf.zeros([fc1_units]), name='biases')
        h_FC1 = tf.nn.relu(tf.matmul(h_pool4_flat, weights) + biases, name=scope.name)
        # print h_FC1
        
    # FC 2
    with tf.variable_scope('h_FC2'):
        weights = tf.Variable(tf.truncated_normal([fc1_units, fc2_units], stddev=1.0 / math.sqrt(float(fc1_units))), name='weights')
        biases = tf.Variable(tf.zeros([fc2_units]), name='biases')
        h_FC2 = tf.nn.relu(tf.matmul(h_FC1, weights) + biases, name=scope.name)
        # print h_FC2
    
    # Linear
    with tf.variable_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([fc2_units, NUM_CLASSES], stddev=1.0 / math.sqrt(float(fc2_units))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(h_FC2, weights) + biases
        # logits = tf.nn.softmax(logits, name='Softmax')
    	# print logits
    
    return logits




"========= loss function  training  evaluation =========="

def loss(logits, labels):

	labels = tf.to_float(labels)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits( logits, labels, name='xentropy')
	loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	return loss


def training(loss, learning_rate):	

    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# def evaluation(logits, labels):
#     correct = tf.nn.in_top_k(logits, labels, 1)
#     return tf.reduce_sum(tf.cast(correct, tf.int32))

# img = io.loadmat('input.mat')['data'][0]
# img = np.reshape(img,[1,128,72,3])
# inference(img, conv1_channels, conv2_channels, fc1_units, fc2_units)