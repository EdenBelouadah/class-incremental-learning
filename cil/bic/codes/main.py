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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import os, math
import tensorflow as tf
# if type(tf.contrib) != type(tf): tf.contrib._warning = None
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARNING)
# tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(1)
import numpy as np
# import tensorflow.contrib.eager as tfe
# Enable eager execution
# tfe.enable_eager_execution()
# import logging
# logging.disable(logging.WARNING)
import resnet
import vgg_preprocessing
import utils_imagenet
from configparser import ConfigParser
from Utils import DataUtils
utils = DataUtils()

##################################################

if len(sys.argv) != 2:  # We have to give 1 arg
    print('Arguments: config')
    sys.exit(-1)

cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]
#######################################################
_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_BIAS_EPOCHS = 2
_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 1500

######### Modifiable Settings ##########
nb_val     = float(cp['nb_val'])
nb_cl     = int(cp['nb_cl'])
nb_groups     = int(cp['nb_groups'])
nb_proto     = int(cp['nb_proto'])
train_path     = cp['train_path']
val_path     = cp['val_path']
models_save_path     = cp['models_save_path']
normalization_dataset_name     = cp['normalization_dataset_name']
datasets_mean_std_file_path     = cp['datasets_mean_std_file_path']
images_mean, _ = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)
images_mean = [e * 255 for e in images_mean]


if not os.path.exists(models_save_path):
    os.makedirs(models_save_path)

np.random.seed(1993)            # Fix the random seed, for class order, same with iCaRL
########################################


_NUM_CLASSES = nb_cl * nb_groups

print("Mixing the classes and putting them in batches of classes...")
order  = np.arange(_NUM_CLASSES)
# np.random.shuffle(order)


### Initialization of some variables ###
x_train_protoset =[]
y_train_protoset =[]
for _ in range(nb_groups*nb_cl):
    x_train_protoset.append([])
    y_train_protoset.append([])

# preload all data and parse into groups
print("Loading all data")
data_dir = ''
data_path        = data_dir
x_train, y_train = utils_imagenet.load_data(train_path, order)
x_test, y_test   = utils_imagenet.load_data(val_path, order)
print (len(x_train), len(y_train), len(x_test), len(y_test))
_NUM_IMAGES = {
    'train': len(x_train),
    'validation': len(x_test),
}

print("Creating a validation set and generating groups...")
max_val = int(nb_val * nb_proto / nb_cl)

print ("max_val:" , max_val)
x_train, y_train, x_val, y_val, x_test, y_test = utils_imagenet.prepare_validation(x_train, y_train, x_test, y_test, nb_groups, nb_cl, max_val)

###############################################################################
# Data processing
###############################################################################

def parse_record(filename, label, is_training):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
"""
  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  # Results in a 3-D int8 Tensor. This will be converted to a float later,
  # during resizing.
  image_encoded = tf.read_file(tf.reduce_join([data_dir, '/', filename]))
  image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
  image = tf.image.decode_jpeg(image_encoded, channels=_NUM_CHANNELS)

  image = vgg_preprocessing.preprocess_image(
      image=image,
      images_mean = images_mean,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      is_training=is_training)

  label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)
  label = tf.one_hot(label, _NUM_CLASSES)

  return image, label


def input_fn(is_training, data, labels, batch_size, num_epochs=1,
             num_parallel_calls=1, multi_gpu=False):
  """Input function which provides batches for train or eval.
  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.
    multi_gpu: Whether this is run multi-GPU. Note that this is only required
      currently to handle the batch leftovers, and can be removed
      when that is handled directly by Estimator.

  Returns:
    A dataset that can be used for iteration.
  """
  dataset = tf.data.Dataset.from_tensor_slices((data, labels))

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)
  num_images = len(labels)
  return resnet.process_record_dataset(dataset, is_training, batch_size,
      num_images, parse_record, num_epochs, num_parallel_calls)

###############################################################################
# Running the model
###############################################################################
class ImagenetModel(resnet.Model):

  def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      block_fn = resnet.building_block
      final_size = 512
    else:
      block_fn = resnet.bottleneck_block
      final_size = 2048

    super(ImagenetModel, self).__init__(
        resnet_size=resnet_size,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        second_pool_size=7,
        second_pool_stride=1,
        block_fn=block_fn,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        data_format=data_format)

  def __call__(self, images, training):
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
      images = tf.transpose(images, [0, 3, 1, 2])

    # representation learning and classifier network
    with tf.variable_scope('resnet'):
      inputs = resnet.conv2d_fixed_padding(
          inputs=images, filters=self.num_filters, kernel_size=self.kernel_size,
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
        inputs = resnet.block_layer(
            inputs=inputs, filters=num_filters, block_fn=self.block_fn,
            blocks=num_blocks, strides=self.block_strides[i],
            training=training, name='block_layer{}'.format(i + 1),
            data_format=self.data_format)

      inputs = resnet.batch_norm_relu(inputs, training, self.data_format)
      inputs = tf.layers.average_pooling2d(
          inputs=inputs, pool_size=self.second_pool_size,
          strides=self.second_pool_stride, padding='VALID',
          data_format=self.data_format)
      inputs = tf.identity(inputs, 'final_avg_pool')
      inputs = tf.reshape(inputs, [-1, self.final_size])
      features = tf.identity(inputs, 'features')
      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
      inputs = tf.identity(inputs, 'final_dense')
      scope = tf.get_variable_scope()
      scope.reuse_variables()

    variables_graph = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet')


    # distilling network
    with tf.variable_scope('store_resnet'):
      dis_inputs = resnet.conv2d_fixed_padding(
          inputs=images, filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format)
      dis_inputs = tf.identity(dis_inputs, 'initial_conv')

      if self.first_pool_size:
        dis_inputs = tf.layers.max_pooling2d(
            inputs=dis_inputs, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        dis_inputs = tf.identity(dis_inputs, 'initial_max_pool')

      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2**i)
        dis_inputs = resnet.block_layer(
            inputs=dis_inputs, filters=num_filters, block_fn=self.block_fn,
            blocks=num_blocks, strides=self.block_strides[i],
            training=False, name='block_layer{}'.format(i + 1),
            data_format=self.data_format)

      dis_inputs = resnet.batch_norm_relu(dis_inputs, False, self.data_format)
      dis_inputs = tf.layers.average_pooling2d(
          inputs=dis_inputs, pool_size=self.second_pool_size,
          strides=self.second_pool_stride, padding='VALID',
          data_format=self.data_format)
      dis_inputs = tf.identity(dis_inputs, 'final_avg_pool')
      dis_inputs = tf.reshape(dis_inputs, [-1, self.final_size])
      dis_features = tf.identity(dis_inputs, 'features')
      dis_inputs = tf.layers.dense(inputs=dis_inputs, units=self.num_classes)
      dis_inputs = tf.identity(dis_inputs, 'final_dense')
      scope = tf.get_variable_scope()
      scope.reuse_variables()

    store_variables_graph = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='store_resnet')

    return features, inputs, variables_graph, dis_inputs




def _get_block_sizes(resnet_size):
  """The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)


def imagenet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""

  if params['flag_bias'] == True:
    ## lr for bias correction learning
    learning_rate_fn = resnet.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=256,
        num_images=params['num_train_images'],
        boundary_epochs=[30*_BIAS_EPOCHS, 60*_BIAS_EPOCHS, 80*_BIAS_EPOCHS, 90*_BIAS_EPOCHS],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])
  else:
    ## normal train lr
    learning_rate_fn = resnet.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=256,
        num_images=params['num_train_images'], boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

  ## increase the weight decay in initial stages
  ## This line of code is here and left unchanged for a long time. 
  weight_decay = 1e-4 * nb_groups / (params['itera'] + 1)
  return resnet.resnet_model_fn(features, labels, mode, ImagenetModel,
                                resnet_size=params['resnet_size'],
                                weight_decay=weight_decay,
                                learning_rate_fn=learning_rate_fn,
                                momentum=0.9,
                                data_format=params['data_format'],
                                itera=params['itera'],
                                nb_groups=params['nb_groups'],
                                nb_cl=params['nb_cl'],
                                restore_model_dir=params['restore_model_dir'],
                                flag_bias=params['flag_bias'],
                                loss_filter_fn=None)


def main(unused_argv):
  beta_all = np.zeros((nb_groups, 1))
  gamma_all = np.zeros((nb_groups, 1))
  test_accuracy_all = np.zeros((nb_groups, 1))
  for itera in range(nb_groups):
    print('Batch of classes number {0} arrives ...'.format(itera+1))
    # Adding the stored exemplars to the training set
    if itera == 0:
      # train data
      x_train_from_cl = x_train[itera][:]
      y_train_from_cl = y_train[itera][:]
      # val data
      x_val_from_cl = []
      y_val_from_cl = []
      for i in range((itera+1)*nb_cl):
        x_val_from_cl += x_val[i][:]
        y_val_from_cl += y_val[i][:]
      # test data
      x_test_from_cl = x_test[itera][:]
      y_test_from_cl = y_test[itera][:]
    else:
      nb_val_itera  = int(np.ceil(nb_val * nb_proto /  (nb_cl * itera)))
      nb_protos_cl  = int(math.ceil(nb_proto /  (nb_cl * itera))) - nb_val_itera # Reducing number of exemplars of the previous classes

      # train data
      x_train_from_cl = x_train[itera][:]
      y_train_from_cl = y_train[itera][:]
      for i in range(itera*nb_cl):
        x_tmp_var = x_train_protoset[i]
        y_tmp_var = y_train_protoset[i]
        x_train_from_cl += x_tmp_var[0:min(len(x_tmp_var),nb_protos_cl)]
        y_train_from_cl += y_tmp_var[0:min(len(y_tmp_var),nb_protos_cl)]
      # val data
      x_val_from_cl = []
      y_val_from_cl = []
      for i in range((itera+1)*nb_cl):
        x_val_from_cl += x_val[i][0:nb_val_itera]
        y_val_from_cl += y_val[i][0:nb_val_itera]

      # test data
      x_test_from_cl = x_test[itera][:]
      y_test_from_cl = y_test[itera][:]
      for i in range(itera):
        x_test_from_cl += x_test[i]
        y_test_from_cl += y_test[i]

    if itera == 0:
      print ("first itera, #training: {}, #val: {}, #test: {}".format(len(x_train_from_cl), len(x_val_from_cl), len(x_test_from_cl)))
      print (len(y_train_from_cl))
      x_train_resnet_features, beta, gamma, test_accuracy = resnet.resnet_main(FLAGS, imagenet_model_fn, input_fn, x_train_from_cl, y_train_from_cl, x_val_from_cl, y_val_from_cl, x_test_from_cl, y_test_from_cl, itera, nb_groups, nb_cl, 1.0, 0.0)
    else:
      print ("incremental iteras, #training: {}, #val: {}, #test: {}".format(len(x_train_from_cl), len(x_val_from_cl), len(x_test_from_cl)))
      x_train_resnet_features, beta, gamma, test_accuracy = resnet.resnet_main(FLAGS, imagenet_model_fn, input_fn, x_train_from_cl, y_train_from_cl, x_val_from_cl, y_val_from_cl, x_test_from_cl, y_test_from_cl, itera, nb_groups, nb_cl, beta_all[itera-1], gamma_all[itera-1])

    beta_all[itera] = beta
    gamma_all[itera] = gamma
    test_accuracy_all[itera] = test_accuracy
    beta_result = ""
    gamma_result = ""
    accuracy_result = ""
    for i in range(itera+1):
      beta_result = "{} {}".format(beta_result, beta_all[i])
      gamma_result = "{} {}".format(gamma_result, gamma_all[i])
      accuracy_result = "{} {}".format(accuracy_result, test_accuracy_all[i])
    print ("beta    : {}".format(beta_result))
    print ("gamma   : {}".format(gamma_result))
    print ("accuracy: {}".format(accuracy_result))

    ## for last increment, do not need to select the exemplars
    if itera == nb_groups - 1:
      break
    ## Exemplars management part  ##
    nb_val_next_itera = int(math.ceil(nb_val * nb_proto / (nb_cl * (itera + 1))))
    nb_protos_cl  = int(math.ceil(nb_proto / (nb_cl * (itera + 1)))) - nb_val_next_itera


    print ("val for next iter:",  nb_val_next_itera )
    print ("exemplars next time for train", nb_protos_cl )

    resnet_features_ = []
    for resnet_feature in x_train_resnet_features:
      resnet_features_.append(resnet_feature['features'])
    print('Exemplars selection starting ...')

    for iter_dico in range((itera+1)* nb_cl):
        x_train_protoset[iter_dico] = []
        y_train_protoset[iter_dico] = []
        y_train_from_cl_ = np.asarray(y_train_from_cl, np.int16) 
        ## np.int8 only works for less than 100 classes
        # y_train_from_cl_ = np.asarray(y_train_from_cl, np.int8)
        ind_cl        = np.where(y_train_from_cl_ == iter_dico)[0]
        D             = np.asarray(resnet_features_, np.float32)[ind_cl]
        mu            = np.mean(D,axis=0)

        selected  = []
        selected_feat = []

        # select nb_protos_cl samples
        for k in range(nb_protos_cl):
          # 512 is the dimension of features in resnet
          sum_others = np.zeros(512)
          for j in selected_feat:
            sum_others += j/(k+1)


          dist_min = np.inf
          assert (len(ind_cl) > 0)
          for item in ind_cl:
            if item not in selected:
              feat = resnet_features_[item]
              dist = np.linalg.norm(mu - feat/(k+1) - sum_others)
              if dist < dist_min:
                dist_min  = dist
                newone = item
                newonefeat = feat
          selected_feat.append(newonefeat)
          selected.append(newone)

          x_train_protoset[iter_dico].append(x_train_from_cl[newone])
          error_message = str(y_train_from_cl[newone]) + " " + str(iter_dico)
          # label should be the same with iter_dico
          assert(y_train_from_cl[newone] == iter_dico), error_message 
          y_train_protoset[iter_dico].append(y_train_from_cl[newone])




if __name__ == '__main__':
  # tf.logging.set_verbosity(tf.logging.INFO)

  parser = resnet.ResnetArgParser(
      resnet_size_choices=[18, 34, 50, 101, 152, 200])
  parser.set_defaults(resnet_size=18,
                    model_dir=models_save_path,
                    train_epochs=100,
                    epochs_per_eval=100,
                    batch_size=256)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
