from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import pandas as pd
from sklearn.model_selection import train_test_split

import os
# import pathresolver package
import pathResolver as pathRes

os.environ["CUDA_VISIBLE_DEVICES"] = ""

FLAGS = None


def get_files(path, ext='', recursive=False):
    path_list = [path]
    result = list()

    while len(path_list) > 0:
        cpath = path_list.pop()
        for entry in os.scandir(cpath):
            if not entry.name.startswith('.') and entry.is_file():
                if entry.name.endswith(ext):
                    result.append(entry.path)
                else:
                    if recursive:
                        path_list.append(entry.path)

    return result


def main():

    # Initiate pathresolver
    pathresolver = pathRes.PathResolver(FLAGS.input, FLAGS.modelPath, FLAGS.model, FLAGS.output)

    # Get paths from pathresolver
    local_model_base_path, local_model_path, local_checkpoint_file, output_file_path = \
        pathresolver.get_paths()
    data_path = pathresolver.get_input_path()
    print("Resolved input_file_path:", data_path)

    # Parameters
    num_steps = 50
    num_classes = 100
    num_features = 8
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

    # if os.path.exists('./model/model.ckpt'):
    #     saver.restore(sess, './model/model.ckpt')

    # Training
    for i in range(1, num_steps + 1):
        # Prepare Data
        for file in get_files(data_path, ext='csv'):
            data = pd.read_csv(file)
            input_x = data.iloc[:, 0:-1].values
            input_y = data.iloc[:, -1].values
            _, l = sess.run([train_op, loss_op], feed_dict={X: input_x, Y: input_y})
            break

    # Test Model
#    accuracy_sum = 0
#    accuracy_max = -1
#    accuracy_min = 1
#    loop = 0
#    al = list()
#    for file in get_files('history', ext='csv'):
#        data = pd.read_csv(file)
#        input_x = data.iloc[:, 0:-1].values
#        input_y = data.iloc[:, -1].values
#        x_train, x_test, y_train, y_test = train_test_split(input_x, input_y, test_size=0.25)
#        accuracy = sess.run(accuracy_op, feed_dict={X: x_test, Y: y_test})
#        accuracy_sum = accuracy_sum + accuracy
#        if accuracy_max < accuracy:
#            accuracy_max = accuracy
#        if accuracy_min > accuracy:
#            accuracy_min = accuracy
#        loop = loop + 1
#        al.append(accuracy)
#    print('Test Accuracy - Avg: {0:0.4f}, Max: {1:0.4f}, Min: {2:0.4f}'.format(accuracy_sum / loop, accuracy_max,
#                                                                               accuracy_min))
#    print(al)


#    os.makedirs('./model', exist_ok=True)
    print('saved path: ', saver.save(sess, local_checkpoint_file))
#    tf.train.write_graph(sess.graph_def, './model/my_model', 'train.pb', as_text=False)
#    tf.train.write_graph(sess.graph_def, './model/my_model', 'train.pbtxt')

    # export SavedModel
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'x': X}, outputs={'y': infer_op})
    builder = tf.saved_model.builder.SavedModelBuilder(local_model_path)
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "predict": signature
        })
    builder.save()

    # Store model to target file system
    pathresolver.store_output_model()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train rnn on tensorflow.')
    parser.add_argument('--input', type=str, default="", help='input path')
    parser.add_argument('--output', type=str, default="", help='output path')
    parser.add_argument('--model', type=str, default="", help='model path')
    parser.add_argument('--modelPath', type=str, default="", help='model base path')

    FLAGS, unparsed = parser.parse_known_args()
    main()
