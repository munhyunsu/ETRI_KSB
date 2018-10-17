from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def get_files(path, ext='', recursive=False):
    path_list = [path]
    file_list = list()

    while len(path_list) > 0:
        cpath = path_list.pop()
        with os.scandir(cpath) as it:
            for entry in it:
                if not entry.name.startswith('.') and entry.is_file():
                    if entry.name.endswith(ext):
                        file_list.append(entry.path)
                    else:
                        if recursive:
                            path_list.append(entry.path)
    return file_list


def main():
    # Parameters
    num_steps = 10  # Total steps to train
    batch_size = 1024  # The number of samples per batch
    num_classes = 100  # The 10 digits
    num_features = 8  # Each image is 28x28 pixels
    num_trees = 10
    max_nodes = 1000

    X = tf.placeholder(tf.float32, shape=[None, num_features])
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

    saver = tf.train.Saver()

    # Start TensorFlow session
    sess = tf.Session()

    # Run the initializer
    sess.run(init_vars)

    if os.path.exists('./model/model.ckpt'):
        saver.restore(sess, './model/model.ckpt')

    # Training
    for i in range(1, num_steps + 1):
        # Prepare Data
        for file in get_files('output', ext='csv'):
            data = pd.read_csv(file)
            input_x = data.iloc[:, 0:-1].values
            input_y = data.iloc[:, -1].values
            _, l = sess.run([train_op, loss_op], feed_dict={X: input_x, Y: input_y})
        # file = random.choice(get_files('output', ext='csv'))
        # data = pd.read_csv(file)
        # input_x = data.iloc[:, 0:-1].values
        # input_y = data.iloc[:, -1].values
        # if i % 50 == 0 or i == 1:
        #     acc = sess.run(accuracy_op, feed_dict={X: input_x, Y: input_y})
        #     print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

    # Test Model
    accuracy = 0
    loop = 0
    for file in get_files('output', ext='csv'):
        data = pd.read_csv(file)
        input_x = data.iloc[:, 0:-1].values
        input_y = data.iloc[:, -1].values
        x_train, x_test, y_train, y_test = train_test_split(input_x, input_y, test_size=0.25)
        accuracy = accuracy + sess.run(accuracy_op, feed_dict={X: x_test, Y: y_test})
        loop = loop + 1
    print("Test Accuracy:", accuracy/loop)

    os.makedirs('./model', exist_ok=True)
    print('saved path: ', saver.save(sess, './model/model.ckpt'))


if __name__ == '__main__':
    main()
