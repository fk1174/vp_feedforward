"""Build data reading subgraph for the `moving_mnist` dataset.

Adapted from the [MNIST IO example](1) from the official Tensorflow Tutorial as well as the data input code snippet from [`Tensorflow/models/research/video_prediction`](2).

[1]: `https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py`
[2]: `https://github.com/tensorflow/models/blob/master/research/video_prediction/prediction_input.py`
"""

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

# Original image dimensions
ORIGINAL_WIDTH = 64
ORIGINAL_HEIGHT = 64
COLOR_CHAN = 1


def mnist_tfrecord_input(data_dir,
                         training=True,
                         sequence_length=20,
                         img_size=None,
                         batch_size=1,
                         seed=None):
    """Create input tfrecord tensors and queues.

    TFRecord:
      TFRecord(s) are assumed to be placed at `data_dir`.
      Each sample contains raw uint8 image sequences with key `'img_i'`.
      Training and validation set are pre-splitted and their corresponding
      record file have suffix `_trn.tfrecords` or `_val_tfrecords`.

    Preprocessing:
      Crop each image to a square one with size `min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)`
      and resize (bicubic) to `(IMG_WIDTH, IMG_HEIGHT)`. Normalize pixel value from
      [0, 255] to [0, 1]

    Args:
      data_dir: directory holding TFRecord(s).
      training: whether to use training or validation data.
      sequence_length: length of the video sequence.
      img_size: the (hight, width) of processed img input, if None use original size.
      batch_size: size of data mimi-batches.
      seed: random seed for `shuffle_batch` generator.
    Returns:
      list of tensors corresponding to images. The images
      tensor is 5D, batch x time x height x width x 1.
    Raises:
      RuntimeError: if no files found.
    """
    file_suffix = '*_trn.tfrecords' if training else '*_val.tfrecords'
    filenames = gfile.Glob(os.path.join(data_dir, file_suffix))
    if not filenames:
        raise RuntimeError('No data files found.')

    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq = []

    for i in range(sequence_length):
        # extract image tensor
        image_name = 'img_{}'.format(i)
        features = tf.parse_single_example(
            serialized_example,
            features={image_name: tf.FixedLenFeature([], tf.string)}
        )
        image = tf.decode_raw(features[image_name], tf.uint8)

        image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])

        # preprocessing
        crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
        image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
        image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
        if img_size is None:
            img_size = (ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
        if img_size[0] != img_size[1]:
            raise ValueError('Unequal height and width unsupported')
        image = tf.image.resize_bicubic(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0

        image_seq.append(image)

    image_seq = tf.concat(axis=0, values=image_seq)

    image_batch = tf.train.shuffle_batch(
        tensors=[image_seq],
        batch_size=batch_size,
        capacity=100 * batch_size,
        min_after_dequeue=50 * batch_size,
        num_threads=batch_size,
        seed=seed
    )
    return image_batch

