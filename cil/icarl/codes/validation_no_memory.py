import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
import os
from scipy.spatial.distance import cdist
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


import utils_icarl

batch_size = 128            # Batch size
is_cumul   = 'cumul'        # Evaluate on the cumul of classes if 'cumul', otherwise on the first classes
datasets_mean_std_file_path = '/scratch_global/eden/images_list_files/datasets_mean_std.txt'

######### Modifiable Settings ##########
if len(sys.argv) < 7 :
    print('Arguments : S, P, dataset, top, save_path, output_file_path')
    sys.exit(-1)

nb_groups = S  = int(sys.argv[1])            # Classes per group
nb_cl = P  = int(sys.argv[2])             # Number of groups
dataset  = sys.argv[3]          #
top        = int(sys.argv[4])              # Choose to evaluate the top X accuracy
device = '/gpu:0'
save_path = sys.argv[5]
output_file_path = sys.argv[6]


normalization_dataset_name = dataset

#####################################################################################################
#Printing parameters
output_file = open(output_file_path, 'w')
print('dataset_name = '+str(normalization_dataset_name))
output_file.write('dataset_name = '+str(normalization_dataset_name)+'\n')
print('save_path = '+str(save_path))
output_file.write('save_path = '+str(save_path)+'\n')
print('output_file_path = '+str(output_file_path))
output_file.write('output_file_path = '+str(output_file_path)+'\n')
print('batch_size = '+str(batch_size))
output_file.write('batch_size = '+str(batch_size)+'\n')
print('nb_cl = '+str(nb_cl))
output_file.write('nb_cl = '+str(nb_cl)+'\n')
print('nb_groups = '+str(nb_groups))
output_file.write('nb_groups = '+str(nb_groups)+'\n')
print('device = '+str(os.environ["CUDA_VISIBLE_DEVICES"]))
output_file.write('device = '+str(os.environ["CUDA_VISIBLE_DEVICES"])+'\n')
print('is_cumul = '+str(is_cumul))
output_file.write('top = '+str(top)+'\n')

images_mean = get_dataset_mean(normalization_dataset_name, datasets_mean_std_file_path)


output_file.flush()

### Initialization of some variables ###
print('images_mean = '+str(images_mean))
output_file.write('images_mean = '+str(images_mean)+'\n')
images_mean = [e*255 for e in images_mean]


str_settings_resnet = str(nb_cl) + 'settings_resnet.pickle'
with open(save_path + str_settings_resnet, 'rb') as fp:
    order = cPickle.load(fp)
    files_valid = cPickle.load(fp)
    files_train = cPickle.load(fp)

# Load class means
str_class_means = save_path + str(nb_cl) + 'class_means.pickle'
with open(str_class_means, 'rb') as fp:
    class_means = cPickle.load(fp)

# Loading the labels
# labels_dic, label_names, validation_ground_truth = utils_data.parse_devkit_meta(devkit_path)

# Initialization
acc_list = np.zeros((nb_groups, 3))
incremental_accuracies = []
hb1_accuracies = []

for itera in range(nb_groups):
    print("Processing network after {} increments\t".format(itera))
    output_file.write("Processing network after {} increments\n".format(itera))
    # Evaluation on cumul of classes or original classes
    if is_cumul == 'cumul':
        eval_groups = np.array(range(itera + 1))
    else:
        eval_groups = [0]

    print("Evaluation on batches {} \t".format(eval_groups))
    output_file.write("Evaluation on batches {} \n".format(eval_groups))
    # Load the evaluation files
    files_from_cl = []
    for i in eval_groups:
        files_from_cl.extend(files_valid[i])

    inits, scores, label_batch, loss_class, file_string_batch, op_feature_map = utils_icarl.reading_data_and_preparing_network(images_mean, files_from_cl, device, itera, batch_size, nb_groups, nb_cl, save_path)

    with tf.Session(config=config) as sess:
        # Launch the prefetch system
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(inits)

        # Evaluation routine
        stat_hb1 = []
        stat_icarl = []
        stat_ncm = []

        for i in range(int(np.ceil(len(files_from_cl) / batch_size))):
            sc, l, loss, files_tmp, feat_map_tmp = sess.run([scores, label_batch, loss_class, file_string_batch, op_feature_map])
            mapped_prototypes = feat_map_tmp[:, 0, 0, :]
            pred_inter = (mapped_prototypes.T) / np.linalg.norm(mapped_prototypes.T, axis=0)
            sqd_icarl = -cdist(class_means[:, :, 0, itera].T, pred_inter.T, 'sqeuclidean').T
            sqd_ncm = -cdist(class_means[:, :, 1, itera].T, pred_inter.T, 'sqeuclidean').T
            stat_hb1 += ([ll in best for ll, best in zip(l, np.argsort(sc, axis=1)[:, -top:])])
            stat_icarl += ([ll in best for ll, best in zip(l, np.argsort(sqd_icarl, axis=1)[:, -top:])])
            stat_ncm += ([ll in best for ll, best in zip(l, np.argsort(sqd_ncm, axis=1)[:, -top:])])

        coord.request_stop()
        coord.join(threads)

    print('Increment: %i' % itera)
    output_file.write('Increment: %i\n' % itera)
    print('Hybrid 1 top ' + str(top) + ' accuracy: %f' % np.average(stat_hb1))
    output_file.write('Hybrid 1 top ' + str(top) + ' accuracy: %f\n' % np.average(stat_hb1))

    if itera != 0 :
        incremental_accuracies.append(float(np.average(stat_hb1)))
    else:
        first_batch_accuracy = float(np.average(stat_hb1))

    print('iCaRL top ' + str(top) + ' accuracy: %f' % np.average(stat_icarl))
    output_file.write('iCaRL top ' + str(top) + ' accuracy: %f\n' % np.average(stat_icarl))


    print('NCM top ' + str(top) + ' accuracy: %f' % np.average(stat_ncm))
    output_file.write('NCM top ' + str(top) + ' accuracy: %f\n' % np.average(stat_ncm))
    output_file.flush()
    acc_list[itera, 0] = np.average(stat_icarl)
    acc_list[itera, 1] = np.average(stat_hb1)
    acc_list[itera, 2] = np.average(stat_ncm)

    # Reset the graph to compute the numbers ater the next increment
    tf.reset_default_graph()

np.save(save_path + 'results_top' + str(top) + '_acc_' + is_cumul + '_cl' + str(nb_cl), acc_list)
output_file.close()

incremental_accuracies = [float(str(e)[:6]) for e in incremental_accuracies]
print('first batch accuracy =' + str(first_batch_accuracy))
print('incremental accuracies = ')
print(incremental_accuracies)
print('mean accuracy = '+str(np.mean(np.array(incremental_accuracies))))