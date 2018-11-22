import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

FLAGS = None


def get_files(path, ext='', recursive=False):
    path_list = [path]

    while len(path_list) > 0:
        cpath = path_list.pop()
        with os.scandir(cpath) as it:
            for entry in it:
                if not entry.name.startswith('.') and entry.is_file():
                    if entry.name.endswith(ext):
                        yield entry.path
                    else:
                        if recursive:
                            path_list.append(entry.path)


def get_model_path():
    model_path = FLAGS.model
    model_num = 1
    if os.path.exists(model_path):
        exist_names = list()
        with os.scandir(FLAGS.model) as it:
            for entry in it:
                if not entry.name.startswith('.') and not entry.is_file():
                    exist_names.append(entry.name)
        while True:
            if not str(model_num) in exist_names:
                break
            model_num = model_num + 1
    if model_path[-1] != '/':
        model_path = model_path + '/'
    model_path = model_path + str(model_num)
    print(model_path)
    return model_path


def main():
    input_path = FLAGS.input
    checkpoint_path = FLAGS.checkpoint
    model_path = get_model_path()

    # Parameters
    num_steps = 50
    num_classes = 100
    num_features = 8
    num_trees = 10
    max_nodes = 1000

    features = tf.placeholder(tf.float32, shape=[None, num_features])
    label = tf.placeholder(tf.int32, shape=[None])

    # Random Forest Parameters
    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                          num_features=num_features,
                                          num_trees=num_trees,
                                          max_nodes=max_nodes).fill()

    # Build the Random Forest
    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    # Get training graph and loss
    train_op = forest_graph.training_graph(features, label)
    loss_op = forest_graph.training_loss(features, label)
    # Measure the accuracy
    infer_op, _, _ = forest_graph.inference_graph(features)

    # Initialize the variables (i.e. assign their default value) and forest resources
    init_vars = tf.group(tf.global_variables_initializer(),
                         resources.initialize_resources(resources.shared_resources()))

    saver = tf.train.Saver()

    # Start TensorFlow session
    sess = tf.Session()

    # Run the initializer
    sess.run(init_vars)

    if os.path.exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)

    # Training
    for i in range(1, num_steps + 1):
        # Prepare Data
        for file in get_files(input_path, ext='csv'):
            data = pd.read_csv(file)
            input_x = data.iloc[:, 0:-1].values
            input_y = data.iloc[:, -1].values
            _, l = sess.run([train_op, loss_op], feed_dict={features: input_x, label: input_y})
            break

    print('saved path: ', saver.save(sess, checkpoint_path))

    # export SavedModel
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'x': features}, outputs={'y': infer_op})
    builder = tf.saved_model.builder.SavedModelBuilder(model_path)
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "predict": signature
        })
    builder.save()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train random forest on tensorflow.')
    parser.add_argument('--input', type=str, required=True, help='input path')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint path')
    parser.add_argument('--model', type=str, required=True, help='model path')

    FLAGS, unparsed = parser.parse_known_args()
    main()
