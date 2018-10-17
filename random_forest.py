import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

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

    print(dataset)

    # Return the dataset.
    return dataset


feature_columns = [tf.feature_column.numeric_column(name)
                   for name in
                   ['station', 'rentMonth', 'rentHour', 'rentWeekday', 'temperature', 'humidity', 'windspeed',
                    'rainfall']]

est = tf.estimator.LinearClassifier(feature_columns,
                                    n_classes=100)

batch_size = 100
est.train(steps=1000,
          input_fn=lambda: csv_input_fn('output.csv', batch_size))
