from __future__ import division
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
import scipy
import os, socket
import math
import copy
from scipy.spatial.distance import cdist
import scipy.io
import sys
try:
    import cPickle
except:
    import _pickle as cPickle

# Syspath for the folder with the utils files
#sys.path.insert(0, "/data/sylvestre")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_dataset_mean(normalization_dataset_name, datasets_mean_std_file_path):
    import re
    datasets_mean_std_file = open(datasets_mean_std_file_path, 'r').readlines()
    for line in datasets_mean_std_file:
        line = line.strip().split(':')
        dataset_name = line[0]
        dataset_stat = line[1]
        if dataset_name == normalization_dataset_name:
            dataset_stat = dataset_stat.split(';')
            dataset_mean = [float (e) for e in re.findall(r'\d+\.\d+',dataset_stat[0])]
            return dataset_mean
    print('Invalid normalization dataset name')
    sys.exit(-1)
    

        

import utils_resnet
import utils_icarl
import utils_data

batch_size = 1            # Batch size
is_cumul   = 'cumul'        # Evaluate on the cumul of classes if 'cumul', otherwise on the first classes
datasets_mean_std_file_path = '/scratch_global/eden/images_list_files/datasets_mean_std.txt'

######### Modifiable Settings ##########
if len(sys.argv) < 8 :
    print('Arguments : S, P, dataset, save_path, output_dir, first_batch_number, last_batch_number ')
    sys.exit(-1)

S  = int(sys.argv[1])            # Classes per group
P  = int(sys.argv[2])     
dataset  = sys.argv[3]
device = '/gpu:0'
save_path = sys.argv[4]
destination_dir = sys.argv[5]
first_batch_number = int(sys.argv[6])
last_batch_number = int(sys.argv[7])


if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# if dataset == 'ilsvrc' :
#     normalization_dataset_name = 'imagenet_full'
# else:
normalization_dataset_name = dataset #+'_batch1'

#####################################################################################################
#Printing parameters
print('dataset_name = '+str(normalization_dataset_name))
print('save_path = '+str(save_path))
print('P = '+str(P))
print('S = '+str(S))
print("Running on " + str(socket.gethostname()) + " | " + str(os.environ["CUDA_VISIBLE_DEVICES"]))

images_mean = get_dataset_mean(normalization_dataset_name, datasets_mean_std_file_path)

### Initialization of some variables ###
print('images_mean = '+str(images_mean))
images_mean = [e*255 for e in images_mean]


str_settings_resnet = str(P)+'settings_resnet.pickle'
with open(os.path.join(save_path,str_settings_resnet),'rb') as fp:
     order       = cPickle.load(fp)
     files_valid = cPickle.load(fp)
     files_train = cPickle.load(fp)


for itera in range(first_batch_number - 1, last_batch_number):
    print('*' * 20)
    print('BATCH '+str(itera+1))
    
    #get data:
    if is_cumul == 'cumul':
        eval_groups = np.array(range(itera + 1))
    else:
        eval_groups = [0]

    # Load the evaluation files
    extract_files = []
    for i in eval_groups:
        extract_files.extend(files_valid[i])

    destination_path = os.path.join(destination_dir, 'b' + str(itera+1) + '_weight_bias.tf')
    print('Saving stats in: ' + destination_path)

    inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(images_mean, extract_files, device, itera, batch_size, S, P, save_path)

    with tf.Session(config=config) as sess:
        # Launch the prefetch system
        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(inits)


        weights =  [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('ResNet18/fc/W:0')]
        bias  = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('ResNet18/fc/b:0')]


        np_weights = sess.run(weights)[0][0][0][:, :(itera+1)*P]
        np_bias = sess.run(bias)[0][0][0][:, :(itera+1)*P]


        with open(destination_path, 'wb') as fp:
            cPickle.dump(np_weights, fp)
            cPickle.dump(np_bias, fp)

        #To load data , use:
        # with open(destination_path, 'rb') as fp:
        #     weights = cPickle.load(fp)
        #     bias = cPickle.load(fp)

        coord.request_stop()
        coord.join(threads)

    # Reset the graph to compute the numbers ater the next increment
    tf.reset_default_graph()