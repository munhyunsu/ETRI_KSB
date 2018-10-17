import tensorflow as tf
from tensorflow.saved_model import tag_constants, signature_constants
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

import os


def main():
    # Parameters
    num_steps = 50  # Total steps to train
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

    # classification_inputs = tf.saved_model.utils.build_tensor_info(X)
    # classification_outputs_classes = tf.saved_model.utils.build_tensor_info(Y)
    # classification_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={
    #             signature_constants.CLASSIFY_INPUTS:
    #                 classification_inputs
    #         },
    #         outputs={
    #             signature_constants.CLASSIFY_OUTPUT_CLASSES:
    #                 classification_outputs_classes
    #         },
    #         method_name=signature_constants.CLASSIFY_METHOD_NAME))
    # tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
    # tensor_info_y = tf.saved_model.utils.build_tensor_info(Y)
    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={'features': tensor_info_x},
    #         outputs={'rph': tensor_info_y},
    #         method_name=signature_constants.PREDICT_METHOD_NAME))
    # builder = tf.saved_model.builder.SavedModelBuilder('./saved_model')
    # builder.add_meta_graph_and_variables(sess,
    #                                      [tag_constants.SERVING],
    #                                      signature_def_map={'predict_rph': prediction_signature,
    #                                                         signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #                                                                        classification_signature},
    #                                      main_op=tf.tables_initializer(),
    #                                      strip_default_attrs=True
    #                                      )
    # builder.save()


if __name__ == '__main__':
    main()
