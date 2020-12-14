# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks
(also known as ResNet v2).

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math

import os
import tensorflow as tf

# if type(tf.contrib) != type(tf): tf.contrib._warning = None
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARNING)
# tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(1)
# import logging
# logging.disable(logging.WARNING)


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_BIAS_EPOCHS = 2


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_parallel_calls=1):
    """Given a Dataset with raw records, parse each record into images and labels,
    and return an iterator over the records.
    Args:
      dataset: A Dataset representing raw records
      is_training: A boolean denoting whether the input is for training.
      batch_size: The number of samples per batch.
      shuffle_buffer: The buffer size to use when shuffling records. A larger
        value results in better randomness, but smaller values reduce startup
        time and use less memory.
      parse_record_fn: A function that takes a raw record and returns the
        corresponding (image, label) pair.
      num_epochs: The number of epochs to repeat the dataset.
      num_parallel_calls: The number of records that are processed in parallel.
        This can be optimized per data set but for generally homogeneous data
        sets, should be approximately the number of available CPU cores.

    Returns:
      Dataset of (image, label) pairs ready for iteration.
    """
    # We prefetch a batch at a time, This can help smooth out the time taken to
    # load input files as we go through shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffle the records. Note that we shuffle before repeating to ensure
        # that the shuffling respects epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # If we are training over multiple epochs before evaluating, repeat the
    # dataset for the appropriate number of epochs.
    dataset = dataset.repeat(num_epochs)

    # Parse the raw records into images and labels
    dataset = dataset.map(lambda image, label: parse_record_fn(image, label, is_training),
                          num_parallel_calls=num_parallel_calls)

    dataset = dataset.batch(batch_size)

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path.
    dataset = dataset.prefetch(1)

    return dataset


################################################################################
# Functions building the ResNet model.
################################################################################
def batch_norm_relu(inputs, training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def building_block(inputs, filters, training, projection_shortcut, strides,
                   data_format):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def bottleneck_block(inputs, filters, training, projection_shortcut,
                     strides, data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note
      that the third and final convolution will use 4 times as many filters.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, training, name,
                data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format)

  return tf.identity(inputs, name)


class Model(object):
  """Base class for building the Resnet v2 Model.
  """

  def __init__(self, resnet_size, num_classes, num_filters, kernel_size,
               conv_stride, first_pool_size, first_pool_stride,
               second_pool_size, second_pool_stride, block_fn, block_sizes,
               block_strides, final_size, data_format=None):
    """Creates a model for classifying an image.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      num_classes: The number of classes used as labels.
      num_filters: The number of filters to use for the first block layer
        of the model. This number is then doubled for each subsequent block
        layer.
      kernel_size: The kernel size to use for convolution.
      conv_stride: stride size for the initial convolutional layer
      first_pool_size: Pool size to be used for the first pooling layer.
        If none, the first pooling layer is skipped.
      first_pool_stride: stride size for the first pooling layer. Not used
        if first_pool_size is None.
      second_pool_size: Pool size to be used for the second pooling layer.
      second_pool_stride: stride size for the final pooling layer
      block_fn: Which block layer function should be used? Pass in one of
        the two functions defined above: building_block or bottleneck_block
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      block_strides: List of integers representing the desired stride size for
        each of the sets of block layers. Should be same length as block_sizes.
      final_size: The expected size of the model after the second pooling.
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
    """
    self.resnet_size = resnet_size

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.second_pool_size = second_pool_size
    self.second_pool_stride = second_pool_stride
    self.block_fn = block_fn
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.final_size = final_size

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    if self.data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
        strides=self.conv_stride, data_format=self.data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    if self.first_pool_size:
      inputs = tf.layers.max_pooling2d(
          inputs=inputs, pool_size=self.first_pool_size,
          strides=self.first_pool_stride, padding='SAME',
          data_format=self.data_format)
      inputs = tf.identity(inputs, 'initial_max_pool')

    for i, num_blocks in enumerate(self.block_sizes):
      num_filters = self.num_filters * (2**i)
      inputs = block_layer(
          inputs=inputs, filters=num_filters, block_fn=self.block_fn,
          blocks=num_blocks, strides=self.block_strides[i],
          training=training, name='block_layer{}'.format(i + 1),
          data_format=self.data_format)

    inputs = batch_norm_relu(inputs, training, self.data_format)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=self.second_pool_size,
        strides=self.second_pool_stride, padding='VALID',
        data_format=self.data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')

    inputs = tf.reshape(inputs, [-1, self.final_size])
    inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
    inputs = tf.identity(inputs, 'final_dense')
     
    return inputs


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. Should be the same length as
      boundary_epochs.

  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = 0.1 * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  return learning_rate_fn


def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, itera, nb_groups, nb_cl, restore_model_dir, flag_bias, loss_filter_fn=None):
  """Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVAL`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  # Generate a summary node for the images
  tf.summary.image('images', features, max_outputs=6)

  model = model_class(resnet_size, data_format)
  resnet_features, logits, variables_graph, dis_logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # for PREDICT
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'features': resnet_features 
    }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  # for EVALUATE

  if mode == tf.estimator.ModeKeys.EVAL:
    if itera == 0:
      all_cl            = range((itera+1)*nb_cl)
      label_all_classes = tf.stack([labels[:,i] for i in all_cl],axis=1)
      pred_all_classes  = tf.stack([logits[:,i] for i in all_cl],axis=1)
      # pred_all_classes  = tf.stack([bias_logits[:,i] for i in all_cl],axis=1)
      predictions = {
          'classes': tf.argmax(pred_all_classes, axis=1),
      }
      ### top-1 accuracy
      # accuracy = tf.metrics.accuracy(
      #     tf.argmax(label_all_classes, axis=1), predictions['classes'])
      ### top-5 for imagenet 
      accuracy = tf.metrics.mean(tf.nn.in_top_k(predictions=pred_all_classes, targets=tf.argmax(label_all_classes, axis=1), k=5))
      metrics = {'accuracy': accuracy}
      cross_entropy = tf.losses.softmax_cross_entropy(
          logits=pred_all_classes, onehot_labels=label_all_classes)
      return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, eval_metric_ops=metrics)
    else:
      all_cl                = range((itera+1)*nb_cl)
      initial_cl            = range(itera*nb_cl)
      new_cl                = range(itera*nb_cl, (itera+1)*nb_cl)
      label_all_classes     = tf.stack([labels[:,i] for i in all_cl],axis=1)
      pred_initial_classes  = tf.stack([logits[:,i] for i in initial_cl],axis=1)
      pred_new_classes      = tf.stack([logits[:,i] for i in new_cl],axis=1)

      print (restore_model_dir)
      bias_restore_model_dir = restore_model_dir.replace(str(itera-1)+'_', str(itera)+'_')
      bias_restore_model_dir = bias_restore_model_dir.replace("classifier", "bias") 
      print(bias_restore_model_dir)
      beta = tf.train.load_variable(bias_restore_model_dir, 'bias_opt/beta')
      gamma = tf.train.load_variable(bias_restore_model_dir, 'bias_opt/gamma')

      pred_all_classes = tf.concat([pred_initial_classes, beta * pred_new_classes + gamma],axis=1)
      pred_all_classes_before = tf.concat([pred_initial_classes, pred_new_classes],axis=1)
      predictions = {
          'classes': tf.argmax(pred_all_classes, axis=1),
          'classes_before': tf.argmax(pred_all_classes_before, axis=1),
      }

      ### top-1 accuracy
      # accuracy = tf.metrics.accuracy(
      #     tf.argmax(label_all_classes, axis=1), predictions['classes'])
      # accuracy_before = tf.metrics.accuracy(
      #     tf.argmax(label_all_classes, axis=1), predictions['classes_before'])
      ### top-5 accuracy
      accuracy = tf.metrics.mean(tf.nn.in_top_k(predictions=pred_all_classes, targets=tf.argmax(label_all_classes, axis=1), k=5))
      accuracy_before = tf.metrics.mean(tf.nn.in_top_k(predictions=pred_all_classes_before, targets=tf.argmax(label_all_classes, axis=1), k=5))

      metrics = {'accuracy': accuracy, 
                 'accuracy_before' : accuracy_before}

      cross_entropy = tf.losses.softmax_cross_entropy(
          logits=pred_all_classes, onehot_labels=label_all_classes)
      return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, eval_metric_ops=metrics)

  ## for TRAIN
  print (flag_bias, not flag_bias, itera == 0, )
  if itera == 0 and (not flag_bias):
    print ("enter classifier training for the first increment")
    all_cl            = range((itera+1)*nb_cl)
    label_all_classes = tf.stack([labels[:,i] for i in all_cl],axis=1)
    pred_all_classes  = tf.stack([logits[:,i] for i in all_cl],axis=1)
  elif not flag_bias:
    print ("enter classifier training")
    initial_cl            = range(itera*nb_cl)
    all_cl                = range((itera+1)*nb_cl)
    # for classification
    label_initial_classes = tf.stack([labels[:,i] for i in initial_cl],axis=1)
    pred_initial_classes  = tf.stack([logits[:,i] for i in initial_cl],axis=1)
    label_all_classes     = tf.stack([labels[:,i] for i in all_cl],axis=1)
    pred_all_classes      = tf.stack([logits[:,i] for i in all_cl],axis=1)
    # for distilling
    if itera == 1: 
      # the second increment does not need the bias correction in training the classifier
      distill_pred_initial_classes  = tf.stack([dis_logits[:,i] for i in initial_cl],axis=1)
    else:
      # apply bias corrected classifier for distilling loss
      assert(itera >= 2)
      pre_initial_cl                   = range((itera-1)*nb_cl)
      pre_new_cl                       = range((itera-1)*nb_cl, (itera)*nb_cl)
      distill_pred_pre_initial_classes = tf.stack([dis_logits[:,i] for i in pre_initial_cl],axis=1)
      distill_pred_pre_new_classes     = tf.stack([dis_logits[:,i] for i in pre_new_cl],axis=1)

      print (restore_model_dir)
      bias_restore_model_dir = restore_model_dir.replace("classifier", "bias") 
      print (bias_restore_model_dir)

      beta = tf.train.load_variable(bias_restore_model_dir, 'bias_opt/beta')
      gamma = tf.train.load_variable(bias_restore_model_dir, 'bias_opt/gamma')
      print (beta, gamma)
      distill_pred_initial_classes     = tf.concat([distill_pred_pre_initial_classes, beta * distill_pred_pre_new_classes + gamma],axis=1)
  else:
    print ("enter bias correction optimization")
    initial_cl            = range(itera*nb_cl)
    new_cl                = range(itera*nb_cl, (itera+1)*nb_cl)
    all_cl                = range((itera+1)*nb_cl)
    pred_initial_classes  = tf.stack([logits[:,i] for i in initial_cl],axis=1)
    pred_new_classes      = tf.stack([logits[:,i] for i in new_cl],axis=1)
    label_all_classes     = tf.stack([labels[:,i] for i in all_cl],axis=1)
    with tf.variable_scope('bias_opt'):
      beta_variable  = tf.get_variable("beta", initializer = 1.0)
      gamma_variable = tf.get_variable("gamma", initializer = 0.0)
      scope = tf.get_variable_scope()
      scope.reuse_variables()
    variables_bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='bias_opt')

    pred_all_classes = tf.concat([pred_initial_classes, beta_variable * pred_new_classes + gamma_variable],axis=1)
    # pred_all_classes = tf.concat([pred_initial_classes, beta_variable * pred_new_classes],axis=1)

  predictions = {
      'classes': tf.argmax(pred_all_classes, axis=1),
      'probabilities': tf.nn.softmax(pred_all_classes, name='softmax_tensor'),
  }

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=pred_all_classes, onehot_labels=label_all_classes)

  if itera > 0 and not flag_bias:
    T = 2
    dis_logits_soft = tf.nn.softmax(distill_pred_initial_classes/T, name='dis_logits_softmax')
    loss_distill = tf.losses.softmax_cross_entropy(logits=pred_initial_classes/T, onehot_labels=dis_logits_soft)


  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  if not loss_filter_fn:
    def loss_filter_fn(name):
      return 'batch_normalization' not in name

  # Add weight decay to the loss.
  if itera == 0:
    loss = cross_entropy + weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in variables_graph
         if loss_filter_fn(v.name)])
  elif not flag_bias:
    print ("enter training with distilling")
    lambda_ = 1.0 * itera / (itera+1)
    loss = lambda_ * loss_distill + (1 - lambda_) * cross_entropy + weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in variables_graph
         if loss_filter_fn(v.name)])
  else:
    assert(flag_bias == True)
    ## apply L2 regularization to gamma, leave beta unconstrained
    gamma_l2_loss = tf.nn.l2_loss(gamma_variable)
    loss   = cross_entropy + gamma_l2_loss * 0.1
    
  if mode == tf.estimator.ModeKeys.TRAIN:
    if flag_bias:
      tf.train.init_from_checkpoint(restore_model_dir, {'resnet/' : 'resnet/'})
    elif itera > 0:
      # restore from previous model
      tf.train.init_from_checkpoint(restore_model_dir, {'resnet/' : 'resnet/'})
      tf.train.init_from_checkpoint(restore_model_dir, {'resnet/' : 'store_resnet/'})

    global_step = tf.train.get_or_create_global_step()
    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum)

    # Batch norm requires update ops to be added as a dependency to train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      if flag_bias:
        train_op = optimizer.minimize(loss, global_step, var_list=variables_bias)
      else:
        train_op = optimizer.minimize(loss, global_step, var_list=variables_graph)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(label_all_classes, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])
    

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def resnet_main(flags, model_function, input_function, x_train, y_train, x_val, y_val, x_test, y_test, itera, nb_groups, nb_cl, beta, gamma):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  model_dir_current_itera = os.path.join(flags.model_dir, str(itera)) 
  if itera == 0:
    model_dir_previous_iter = os.path.join(flags.model_dir, str(itera)) 
  else:
    model_dir_previous_iter = os.path.join(flags.model_dir, str(itera - 1))
    
  num_train_images = len(x_train)
  # Set up a RunConfig to only save checkpoints once per training cycle.
  model_dir_current_itera_classifier       = model_dir_current_itera + '_classifier'
  model_dir_current_itera_bias             = model_dir_current_itera + '_bias'
  model_dir_previous_iter_itera_classifier = model_dir_previous_iter + '_classifier'

  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  classifier = tf.estimator.Estimator(
      model_fn=model_function, model_dir=model_dir_current_itera_classifier, config=run_config,
      params={
          'resnet_size': flags.resnet_size,
          'data_format': flags.data_format,
          'batch_size': flags.batch_size,
          'itera': itera,
          'nb_groups': nb_groups,
          'nb_cl': nb_cl,
          'restore_model_dir': model_dir_previous_iter_itera_classifier,
          'num_train_images': num_train_images,
          'flag_bias': False
      })

  for _ in range(flags.train_epochs // flags.epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    print('Starting a training cycle.')

    def input_fn_train():
      return input_function(True, x_train, y_train, flags.batch_size,
                            flags.epochs_per_eval, flags.num_parallel_calls)
       
    
    ## This condition is applied when your training is interepted by unexpected power outage,
    ## which happens in my lab sometimes and you would like to resume the training
    # if itera > 6:
    if itera > -1:
      classifier.train(input_fn=input_fn_train, hooks=[logging_hook])
    
    if itera > 0:
      print('Starting bias optimization on validation.')
      epoch_val_times = _BIAS_EPOCHS
      def input_fn_eval():
        return input_function(False, x_val, y_val, flags.batch_size,
                              flags.epochs_per_eval*epoch_val_times, flags.num_parallel_calls)

      num_val_images = len(x_val)
      print (num_val_images)
      run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
      # bias correction optimization using validation set
      model_dir_current_itera_bias = model_dir_current_itera + '_bias' 
      classifier_bias = tf.estimator.Estimator(
        model_fn=model_function, model_dir=model_dir_current_itera_bias, config=run_config,
        params={
            'resnet_size': flags.resnet_size,
            'data_format': flags.data_format,
            'batch_size': flags.batch_size,
            'itera': itera,
            'nb_groups': nb_groups,
            'nb_cl': nb_cl,
            'restore_model_dir': model_dir_current_itera_classifier,
            'num_train_images': num_val_images,
            'flag_bias': True
        })

      ## This condition is applied when your training is interepted by unexpected power outage,
      ## which happens in my lab sometimes and you would like to resume the training
      # if itera > 6:
      if itera > -1:
        classifier_bias.train(input_fn=input_fn_eval, hooks=[logging_hook])

      ## results from validation set
      def input_fn_eval_test():
        return input_function(False, x_val, y_val, flags.batch_size,
                              1, flags.num_parallel_calls)

      test_eval_results = classifier.evaluate(input_fn=input_fn_eval_test)
      print(test_eval_results)

      ## results from test set
      def input_fn_test():
        return input_function(False, x_test, y_test, flags.batch_size,
                              1, flags.num_parallel_calls)

      test_eval_results = classifier.evaluate(input_fn=input_fn_test)
      print(test_eval_results)

    else:
      # Evaluate the model on test and print results
      def input_fn_test():
        return input_function(False, x_test, y_test, flags.batch_size,
                              1, flags.num_parallel_calls)

      test_eval_results = classifier.evaluate(input_fn=input_fn_test)
      print(test_eval_results)

    # log beta and final results
    if itera == 0:
      selected_beta  = 1.0 
      selected_gamma = 0.0
    else:
      selected_beta  = tf.train.load_variable(model_dir_current_itera_bias, 'bias_opt/beta') 
      selected_gamma = tf.train.load_variable(model_dir_current_itera_bias, 'bias_opt/gamma') 
      # selected_gamma = 0.0
      print (selected_beta, selected_gamma)
       
    test_final_accuracy = test_eval_results['accuracy']

    # extract features for the training data and return to select exemplars. 
    def input_fn_feature_extraction():
      return input_function(False, x_train, y_train, flags.batch_size,
                            1, flags.num_parallel_calls)

    resnet_features = classifier.predict(input_fn = input_fn_feature_extraction, predict_keys=['features'])
 
    return list(resnet_features), selected_beta, selected_gamma, test_final_accuracy


class ResnetArgParser(argparse.ArgumentParser):
  """Arguments for configuring and running a Resnet Model.
  """

  def __init__(self, resnet_size_choices=None):
    super(ResnetArgParser, self).__init__()
    self.add_argument(
        '--data_dir', type=str, default='/tmp/resnet_data',
        help='The directory where the input data is stored.')

    self.add_argument(
        '--num_parallel_calls', type=int, default=5,
        help='The number of records that are processed in parallel '
        'during input processing. This can be optimized per data set but '
        'for generally homogeneous data sets, should be approximately the '
        'number of available CPU cores.')

    self.add_argument(
        '--model_dir', type=str, default='/tmp/resnet_model',
        help='The directory where the model will be stored.')

    self.add_argument(
        '--resnet_size', type=int, default=50,
        choices=resnet_size_choices,
        help='The size of the ResNet model to use.')

    self.add_argument(
        '--train_epochs', type=int, default=100,
        help='The number of epochs to use for training.')

    self.add_argument(
        '--epochs_per_eval', type=int, default=1,
        help='The number of training epochs to run between evaluations.')

    self.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training and evaluation.')

    self.add_argument(
        '--data_format', type=str, default=None,
        choices=['channels_first', 'channels_last'],
        help='A flag to override the data format used in the model. '
             'channels_first provides a performance boost on GPU but '
             'is not always compatible with CPU. If left unspecified, '
             'the data format will be chosen automatically based on '
             'whether TensorFlow was built for CPU or GPU.')
