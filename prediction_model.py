# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model architecture for predictive model, including CDNA, DNA, and STP."""
import hobotrl as hrl
import numpy as np
import tensorflow as tf
import logging
logging.basicConfig(format='[%(asctime)s] (%(filename)s): |%(levelname)s| %(message)s')


import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from lstm_ops import basic_conv_lstm_cell
# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5
ACTION_DIM = 5
l2 = 1e-7

def construct_model_ff(images, # oh 15 FeedFroward
                        # actions=None,
                        # use_action=1,
                        iter_num=-1.0,
                        k=-1,
                        context_frames=2):
    logging.warning("-----images8:%s", images)
    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]

    # Generated robot states and images.
    gen_images = []

    if k == -1:
         feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        num_ground_truth = tf.to_int32(
                    tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(iter_num / k)))))
        feedself = False

    for image in images[:-1]: # images[0,1,2,...,8] , no last images[9]    32, 64, 64, 3->9times
        # Reuse variables after the first timestep.
        # tf.reset_default_graph()
        logging.warning("-----np.shape(image5):%s", np.shape(image)) # 32, 64, 64, 3
        reuse = bool(gen_images)
        logging.warning("-----reuse:%s", reuse)

        # reuse = True

        done_warm_start = len(gen_images) > context_frames - 1
        logging.warning("-----len(gen_images):%s", len(gen_images))
        logging.warning("-----np.shape(gen_images):%s", np.shape(gen_images))

        with tf.variable_scope("scope1", reuse=reuse):
                    if feedself and done_warm_start:
                                # Feed in generated image.
                                prev_image = gen_images[-1]
                    elif done_warm_start:
                                # Scheduled sampling
                                prev_image = scheduled_sample(image, gen_images[-1], batch_size, num_ground_truth)
                    else:
                                # Always feed in ground_truth
                                prev_image = image

                    conv_se = hrl.utils.Network.conv2ds(prev_image,
                                                        shape=[(64, 8, 2), (128, 6, 2), (128, 6, 2), (128, 4, 2)],
                                                        out_flatten=True,
                                                        activation=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="conv_se")

                    feature_fc = hrl.network.Utils.layer_fcs(conv_se, [2048], 2048,
                                                                     activation_out=tf.nn.relu,
                                                                     l2=l2,
                                                                     var_scope="feature_fc")

                    concat = hrl.network.Utils.layer_fcs(feature_fc, [2048], 2048,
                                                                    activation_out=tf.nn.relu,
                                                                    l2=l2,
                                                                    var_scope="concat")

                    # 2048 => 128*8*8
                    concat_deconv = tf.reshape(concat, [32, 4, 4, 128])
                    logging.warning("-----np.shape(concat_deconv):%s", np.shape(concat_deconv))

                    output = hrl.utils.Network.conv2ds_transpose(concat_deconv,
                                                                shape=[(128, 4, 2), (128, 6, 2), (128, 6, 2), (3, 8, 2)],
                                                                activation=tf.nn.relu,
                                                                l2=l2,
                                                                var_scope="deconv")

                    logging.warning("-----np.shape(output):%s", np.shape(output))


                    gen_images.append(output)

    return gen_images


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):

    idx = tf.random_shuffle(tf.range(int(batch_size)))
    ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
    generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

    ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
    generated_examps = tf.gather(generated_x, generated_idx)

    logging.warning("-----ground_truth_idx:%s", ground_truth_idx)
    logging.warning("-----generated_idx:%s", generated_idx)
    logging.warning("-----ground_truth_examps:%s", ground_truth_examps)
    logging.warning("-----generated_examps:%s", generated_examps) # (?, 2048, 128, 3) ==> True: (?, 128, 128, 3)

    return tf.dynamic_stitch([ground_truth_idx, generated_idx], [ground_truth_examps, generated_examps])
