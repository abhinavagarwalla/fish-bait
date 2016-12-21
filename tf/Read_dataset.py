
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import scipy.io as io
import sklearn.preprocessing as pre

# In[2]:

class DataSet(object):

    def __init__(self, images, labels, dtype=tf.float32):

        """Construct a DataSet.
        FIXME: fake_data options
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        #COnvert the shape from [num_exmaple,channels, height, width]
        #to [num_exmaple, height, width, channels]
        #labels[:] = [i - 1 for i in labels]
        # print (labels.shape)
        labels = np.transpose(labels,(1,0))
        labels = pre.LabelBinarizer().fit_transform(labels)
        
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                          dtype)


        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns*depth] 
#         if dtype == tf.float32:
#             # Convert from [0, 255] -> [0.0, 1.0].
#             images = images.astype(numpy.float32)
#             images = numpy.multiply(images, 1.0 / 255.0)
            
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


# In[3]:

def read_data_sets(directory, dtype=tf.float32):

    images = io.loadmat(directory)["data"]
    labels = io.loadmat(directory)['label']
    # labels = np.transpose(labels_temp,(1,))
    data_sets = DataSet(images, labels, dtype=dtype)

    return data_sets


# In[4]:


# In[ ]:



