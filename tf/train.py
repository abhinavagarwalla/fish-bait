"==========loading libraies========="

print "loading libraries"
import numpy as np
import cv2
import os
import scipy.io as io
import tensorflow as tf
import math
import network
import Read_dataset as input_data
import time
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
learning_rate = 0.01
max_steps = 4000

"================ run training ============="

def placeholder_inputs(batch_size):

	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, ROWS, COLS, CHANNELS))
	labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size,NUM_CLASSES))
	return images_placeholder, labels_placeholder


def fill_feed_dict( data_set, images_pl, labels_pl):
	images_feed, labels_feed = data_set.next_batch(batch_size)
	feed_dict = {
		images_pl: images_feed,
		labels_pl: labels_feed,
		}
	return feed_dict



def run_training():
    """Concatenating all the training and test mat files"""
    Training_data = io.loadmat('input.mat')
    Training_data = input_data.read_data_sets(('input.mat'))

    # Test_data = io.loadmat('test.mat')
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = network.inference(images_placeholder,
                                conv1_channels,
                                conv2_channels,
                                fc1_units,
                                fc2_units,
                                )

        # Add to the Graph the Ops for loss calculation.
        loss = network.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = network.training(loss, learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        # eval_correct = network.evaluation(logits, labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
#         summary_op = tf.merge_all_summaries()

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
#         summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in xrange(max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(Training_data,
                                     images_placeholder,
                                     labels_placeholder)

            _, loss_value = sess.run([train_op, loss],
                                   feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            # if step % 50 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
#                 summary_str = sess.run(summary_op, feed_dict=feed_dict)
#                 summary_writer.add_summary(summary_str, step)
#                 summary_writer.flush()


            # Evaluate against the training set.
            if step % 50 == 0 :
	            print('Training Data Eval:')
	            saver.save(sess, 'models/model-depth-CNN.ckpt', global_step=step)
	            # do_eval(sess,
	            #         eval_correct,
	            #         images_placeholder,
	            #         labels_placeholder,
	            #         Training_data)
            # print('Test Data Eval:')
            # do_eval(sess,
            #         eval_correct,
            #         images_placeholder,
            #         labels_placeholder,
            #         Test_data)
# 
# run_training()


def predictor(img):

	with tf.Graph().as_default():
		# Generate placeholders for the images and labels.
		images_placeholder, labels_placeholder = placeholder_inputs(1)

		# Build a Graph that computes predictions from the inference model.
		logits = network.inference(images_placeholder,
		                        conv1_channels,
		                        conv2_channels,
		                        fc1_units,
		                        fc2_units,
		                        )

		sm = tf.nn.softmax(logits)

		saver = tf.train.Saver()

		sess = tf.Session()

		saver.restore(sess,'models/model-depth-CNN.ckpt-50')

		prediction = sess.run(sm,feed_dict={images_placeholder: img})

	return prediction


"============load test data ================="

test_data = io.loadmat('test.mat')['output_data']
# print test_data[0].shape
data = np.reshape((test_data[50]),[1,128,72,3])
# print data.shape
print predictor(data)

# submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)
# submission.insert(0, 'image', test_files)
# submission.head()



