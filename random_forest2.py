""" Random Forest.
Implement Random Forest algorithm with TensorFlow, and apply it to classify 
handwritten digit images. This example is using the MNIST database of 
handwritten digits as training samples (http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import pandas as pd

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

COLUMNS = ['station', 'rentMonth', 'rentHour', 'rentWeekday', 'temperature',
           'humidity', 'windspeed', 'rainfall', 'changeOfRentable']
FIELD_FEFAULTS = [[0], [0], [0], [0], [0.0],
                  [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    fields = tf.decode_csv(line, record_defaults=FIELD_FEFAULTS)

    features = dict(zip(COLUMNS, fields))

    label = features.pop('changeOfRentable')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


# data = pd.read_csv('m1000.csv')
# input_x = data.iloc[:, 0:-1].values
# input_y = data.iloc[:, -1].values
data = pd.read_csv('merged_data.csv')
data_size = len(data.iloc[:, -1])
# input_x = data.iloc[:, 0:-1].values
# input_y = data.iloc[:, -1].values



# Parameters
num_steps = 500  # Total steps to train
batch_size = 1024  # The number of samples per batch
num_classes = 100  # The 10 digits
num_features = 8  # Each image is 28x28 pixels
num_trees = 10
max_nodes = 1000

# dataset = tf.data.TextLineDataset('merged_data.csv').skip(1)
# dataset = dataset.map(_parse_line)
# dataset = dataset.shuffle(1000).repeat().batch(batch_size)
# it = dataset.make_one_shot_iterator().get_next()
# feature = tf.contrib.data.CsvDataset(filenames,
#                                      feature_default,
#                                      select_cols=[0, 1, 2, 3, 4, 5, 6, 7])
# label = tf.contrib.data.CsvDataset(filenames,
#                                    label_default,
#                                    select_cols=[8])

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars)

# Training
for i in range(1, num_steps + 1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    # batch_x, batch_y = csv_input_fn('output.csv', batch_size)
    mul = data_size // batch_size
    fi = 0
    ti = mul
    while ti < data_size+1:
        input_x = data.iloc[fi:ti, 0:-1].values
        input_y = data.iloc[fi:ti, -1].values
        _, l = sess.run([train_op, loss_op], feed_dict={X: input_x, Y: input_y})
        fi = ti
        if ti + mul < data_size:
            ti = ti + mul
        else:
            ti = data_size+1
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: input_x, Y: input_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

# Test Model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_x, input_y, test_size=0.25, random_state = 0)
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: x_test, Y: y_test}))
