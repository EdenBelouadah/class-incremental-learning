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
    
def get_exemplars(old_classes_number, num_exemplars, files_protoset):
    exemplars = []
    for c in range(old_classes_number):
        exemplars.extend(files_protoset[c][:num_exemplars])
    # print(len(exemplars))
    return exemplars
        

import utils_resnet
import utils_icarl
import utils_data

is_cumul   = 'cumul'        # Evaluate on the cumul of classes if 'cumul', otherwise on the first classes
datasets_mean_std_file_path = '/scratch_global/eden/images_list_files/datasets_mean_std.txt'

######### Modifiable Settings ##########
if len(sys.argv) < 12 :
    print('Arguments : S, P, K, batch_size, dataset, save_path, output_dir, train_or_val, feat_or_scores, first_batch_number, last_batch_number ')
    sys.exit(-1)

S  = int(sys.argv[1])            # Classes per group
P  = int(sys.argv[2])     
K  = int(sys.argv[3])
batch_size = int(sys.argv[4])
dataset  = sys.argv[5]
device = '/gpu:0'
save_path = sys.argv[6]
scores_output_dir = sys.argv[7]
train_or_val = sys.argv[8]
feat_or_scores = sys.argv[9]
first_batch_number = int(sys.argv[10])
last_batch_number = int(sys.argv[11])

assert(train_or_val in ['train', 'val'])
assert(feat_or_scores in ['features', 'scores'])

if not os.path.exists(scores_output_dir):
    os.makedirs(scores_output_dir)

scores_output_dir = os.path.join(scores_output_dir, train_or_val)
if not os.path.exists(scores_output_dir):
    os.mkdir(scores_output_dir)

# if dataset == 'ilsvrc' :
#     normalization_dataset_name = 'imagenet_full'
# else:
normalization_dataset_name = dataset# +'_batch1'

#####################################################################################################
#Printing parameters
print('dataset_name = '+str(normalization_dataset_name))
print('save_path = '+str(save_path))
print('batch_size = '+str(batch_size))
print('P = '+str(P))
print('S = '+str(S))
print("Running on " + str(socket.gethostname()) + " | " + str(os.environ["CUDA_VISIBLE_DEVICES"]))
print('is_cumul = '+str(is_cumul))

images_mean = get_dataset_mean(normalization_dataset_name, datasets_mean_std_file_path)



### Initialization of some variables ###
print('images_mean = '+str(images_mean))
images_mean = [e*255 for e in images_mean]


str_settings_resnet = str(P)+'settings_resnet.pickle'
with open(os.path.join(save_path,str_settings_resnet),'rb') as fp:
     order       = cPickle.load(fp)
     files_valid = cPickle.load(fp)
     files_train = cPickle.load(fp)

str_protoset = os.path.join(save_path,str(P)+'files_protoset.pickle')
with open(str_protoset, 'rb') as fp:
      files_protoset = cPickle.load(fp)


# print(len(files_valid))
# print(len(files_valid[0]))
# print(len(files_valid[1]))

# Load class means
str_class_means = os.path.join(save_path,str(P)+'class_means.pickle')
with open(str_class_means,'rb') as fp:
      class_means = cPickle.load(fp)


# Initialization
acc_list = np.zeros((S,3))

accuracies = []
extract_files = copy.deepcopy(files_train)

for itera in range(first_batch_number - 1, last_batch_number):
    print('*'*20)
    print('BATCH '+str(itera+1))
    print('*' * 20)
    
    #get data:
    if is_cumul == 'cumul':
        eval_groups = np.array(range(itera + 1))
    else:
        eval_groups = [0]

    print("Evaluation on batches {} \t".format(eval_groups))
    # Load the evaluation files

    data_extraction_size = 0
    if train_or_val == 'train':
        if itera > 0 :
            old_classes_number = itera * P
            num_exemplars  = int(math.ceil(K / old_classes_number))
            exemplars = get_exemplars(old_classes_number, num_exemplars, files_protoset)
            extract_files[itera].extend(exemplars)

        data_extraction_size = len(extract_files[itera])
    else:
        extract_files = []
        for i in eval_groups:
            extract_files.extend(files_valid[i])

        data_extraction_size = len(extract_files)

    print('Data extraction size = ' + str(data_extraction_size))



    batch_scores_output_dir = os.path.join(scores_output_dir, 'batch'+str(itera+1))
    if not os.path .exists(batch_scores_output_dir):
        os.mkdir(batch_scores_output_dir)


    full_scores = None
    full_paths = None
    f_out_scores = open(os.path.join(batch_scores_output_dir, feat_or_scores+'.raw'), 'w')
    # f_out_paths = open(os.path.join(batch_scores_output_dir, 'paths.lst'), 'w')
    f_out_paths = open(os.path.join(batch_scores_output_dir, 'paths_'+feat_or_scores+'.lst'), 'w')

    if train_or_val == 'val':
        inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(images_mean, extract_files, device, itera, batch_size, S, P, save_path)
    else:
        inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(images_mean, extract_files[itera], device, itera, batch_size, S, P, save_path)

    with tf.Session(config=config) as sess:
        # Launch the prefetch system
        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(inits)

        if train_or_val == 'train':
            data_size = int(np.ceil(len(extract_files[itera])/batch_size))
        else:
            data_size = int(np.ceil(len(extract_files)/batch_size))

        for i in range(data_size):

            sc, l , loss,files_tmp,feat_map_tmp = sess.run([scores, label_batch,loss_class,file_string_batch,op_feature_map])

            # print(feat_map_tmp.shape)
            # print(feat_map_tmp[:, 0, 0].shape)
            # # print(sc.shape)
            # #
            # sys.exit(-1)
            if full_paths is None :
                full_paths = files_tmp.reshape(-1, 1)
                if feat_or_scores == 'scores':
                    full_scores = sc[:, :(itera+1)*P]
                else:
                    full_scores = feat_map_tmp[:, 0, 0]

            else:
                full_paths = np.vstack((full_paths, files_tmp.reshape(-1, 1)))
                if feat_or_scores == 'scores':
                    full_scores = np.vstack((full_scores, sc[:, :(itera + 1) * P]))
                else:
                    full_scores = np.vstack((full_scores, feat_map_tmp[:, 0, 0]))

        coord.request_stop()
        coord.join(threads)

    # Reset the graph to compute the numbers ater the next increment
    tf.reset_default_graph()

    full_scores = full_scores[:data_extraction_size]
    full_paths = full_paths[:data_extraction_size]

    print('Final scores shape = ' + str(full_scores.shape)+', paths shape = '+str(full_paths.shape))
    assert(len(full_scores) == len(full_paths))

    for i in range(len(full_scores)):
        f_out_scores.write(' '.join(map(str, list(full_scores[i])))+'\n')
        f_out_paths.write(list(full_paths[i])[0]+'\n')
        # f_out_paths.write(str(full_paths[i])+'\n')
    f_out_scores.close()
    f_out_paths.close()
