# Most of this code is taken from iCaRL (Sylvestre-Alvise Rebuffi et al.)

from __future__ import division
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
import scipy
import os, socket
import scipy.io
import sys

try:
    import cPickle
except:
    import _pickle as cPickle
# Syspath for the folder with the utils files
# sys.path.insert(0, "/media/data/srebuffi")

import utils_resnet
import utils_icarl
import utils_data
from configparser import ConfigParser
from Utils import DataUtils
utils = DataUtils()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if len(sys.argv) != 2:  # We have to give 1 arg
    print('Arguments: config')
    sys.exit(-1)


# loading configuration file
cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]



######### Modifiable Settings ##########
batch_size = int(cp['batch_size'])    # Batch size
nb_groups = int(cp['nb_groups'])  # Number of groups
nb_cl = int(cp['nb_cl'])  # Number of groups
lr_old =  float(cp['lr_old'])    # Initial learning rate
lr_strat = utils.from_str_to_list(cp['lr_strat'], 'int')  # Epochs where learning rate gets decreased
lr_factor =  float(cp['lr_factor'])    # Learning rate decrease factor
wght_decay =  float(cp['wght_decay'])   # Weight Decay
print_each = 10
########################################

normalization_dataset_name = cp['normalization_dataset_name']
save_path   = cp['save_path']
train_images_path = cp['train_images_path']
val_images_path = cp['val_images_path']
output_file_path = cp['output_file_path']
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']
device = '/gpu:0'
epochs = int(cp['epochs'])   # Total number of epochs

if not os.path.exists(save_path):
    os.mkdir(save_path)


#####################################################################################################
#Printing parameters
output_file = open(output_file_path, 'w')


print('dataset_name = '+str(normalization_dataset_name))
output_file.write('dataset_name = '+str(normalization_dataset_name)+'\n')
print('save_path = '+str(save_path))
output_file.write('save_path = '+str(save_path)+'\n')
print('train_images_path = '+str(train_images_path))
output_file.write('train_images_path = '+str(train_images_path)+'\n')
print('val_images_path = '+str(val_images_path))
output_file.write('val_images_path = '+str(val_images_path)+'\n')
print('output_file_path = '+str(output_file_path))
output_file.write('output_file_path = '+str(output_file_path)+'\n')
print('batch_size = '+str(batch_size))
output_file.write('batch_size = '+str(batch_size)+'\n')
print('nb_cl = '+str(nb_cl))
output_file.write('nb_cl = '+str(nb_cl)+'\n')
print('nb_groups = '+str(nb_groups))
output_file.write('nb_groups = '+str(nb_groups)+'\n')
print('epochs = '+str(epochs))
output_file.write('epochs = '+str(epochs)+'\n')
print('lr_old = '+str(lr_old))
output_file.write('lr_old = '+str(lr_old)+'\n')
print('lr_strat = '+str(lr_strat))
output_file.write('lr_strat = '+str(lr_strat)+'\n')
print('lr_factor = '+str(lr_factor))
output_file.write('lr_factor = '+str(lr_factor)+'\n')
print('wght_decay = '+str(wght_decay))
output_file.write('wght_decay = '+str(wght_decay)+'\n')
print("Running on " + str(socket.gethostname()) + " | " + str(os.environ["CUDA_VISIBLE_DEVICES"]))
output_file.write("Running on " + str(socket.gethostname()) + " | " + str(os.environ["CUDA_VISIBLE_DEVICES"])+'\n')


images_mean, _ = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)

output_file.flush()

### Initialization of some variables ###
print('images_mean = '+str(images_mean))
output_file.write('images_mean = '+str(images_mean)+'\n')
images_mean = [e*255 for e in images_mean]
class_means = np.zeros((512, nb_groups * nb_cl, 2, nb_groups))
loss_batch = []
files_protoset = []
for _ in range(nb_groups * nb_cl):
    files_protoset.append([])

### Preparing the files for the training/validation ###

# Random mixing
print("Preparing data...")
output_file.write("Preparing data...\n")
output_file.flush()
np.random.seed(1993)
order = np.arange(nb_groups * nb_cl)
# mixing = np.arange(nb_groups * nb_cl)
# np.random.shuffle(mixing)

# Loading the labels
# labels_dic, label_names, validation_ground_truth = utils_data.parse_devkit_meta(devkit_path)
# Or you can just do like this
# define_class = ['apple', 'banana', 'cat', 'dog', 'elephant', 'forg']
# labels_dic = {k: v for v, k in enumerate(define_class)}

# Preparing the files per group of classes
files_train, files_valid = utils_data.prepare_files(train_images_path, val_images_path, nb_groups, nb_cl)

# Pickle order and files lists and mixing
# with open(str(nb_cl)+'mixing.pickle','wb') as fp:
#     cPickle.dump(mixing,fp)

with open(save_path + str(nb_cl) + 'settings_resnet.pickle', 'wb') as fp:
    cPickle.dump(order, fp)
    cPickle.dump(files_valid, fp)
    cPickle.dump(files_train, fp)

### Start of the main algorithm ###

for itera in range(nb_groups):

    # Files to load : training samples + protoset
    print('Batch of classes number {0} arrives ...'.format(itera + 1))
    output_file.write('Batch of classes number {0} arrives ...\n'.format(itera + 1))
    output_file.flush()
    # Adding the stored exemplars to the training set
    files_from_cl = files_train[itera]


    ## Import the data reader ##
    image_train, label_train = utils_data.read_data(files_from_cl=files_from_cl)
    image_batch, label_batch_0 = tf.train.batch([image_train, label_train], batch_size=batch_size, num_threads=8)
    label_batch = tf.one_hot(label_batch_0, nb_groups * nb_cl)

    ## Define the objective for the neural network ##
    if itera == 0:
        # No distillation
        variables_graph, variables_graph2, scores, scores_stored = utils_icarl.prepare_networks(images_mean, device, image_batch,
                                                                                                nb_cl, nb_groups)

        # Define the objective for the neural network: 1 vs all cross_entropy
        with tf.device(device):
            scores = tf.concat(scores, 0)
            l2_reg = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet18'))
            loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=scores))
            loss = loss_class + l2_reg
            learning_rate = tf.placeholder(tf.float32, shape=[])
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
            train_step = opt.minimize(loss, var_list=variables_graph)

    if itera > 0:
        # Distillation
        variables_graph, variables_graph2, scores, scores_stored = utils_icarl.prepare_networks(images_mean, device, image_batch,
                                                                                                nb_cl, nb_groups)

        # Copying the network to use its predictions as ground truth labels
        op_assign = [(variables_graph2[i]).assign(variables_graph[i]) for i in range(len(variables_graph))]

        # Define the objective for the neural network : 1 vs all cross_entropy + distillation
        with tf.device(device):
            scores = tf.concat(scores, 0)
            scores_stored = tf.concat(scores_stored, 0)
            old_cl = (order[range(itera * nb_cl)]).astype(np.int32)
            new_cl = (order[range(itera * nb_cl, nb_groups * nb_cl)]).astype(np.int32)
            label_old_classes = tf.sigmoid(tf.stack([scores_stored[:, i] for i in old_cl], axis=1))
            label_new_classes = tf.stack([label_batch[:, i] for i in new_cl], axis=1)
            pred_old_classes = tf.stack([scores[:, i] for i in old_cl], axis=1)
            pred_new_classes = tf.stack([scores[:, i] for i in new_cl], axis=1)
            l2_reg = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet18'))
            loss_class = tf.reduce_mean(tf.concat(
                [tf.nn.sigmoid_cross_entropy_with_logits(labels=label_old_classes, logits=pred_old_classes),
                 tf.nn.sigmoid_cross_entropy_with_logits(labels=label_new_classes, logits=pred_new_classes)], 1))
            loss = loss_class + l2_reg
            learning_rate = tf.placeholder(tf.float32, shape=[])
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
            train_step = opt.minimize(loss, var_list=variables_graph)

    ## Run the learning phase ##
    with tf.Session(config=config) as sess:
        # Launch the data reader
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(tf.global_variables_initializer())
        lr = lr_old

        # Run the loading of the weights for the learning network and the copy network
        if itera > 0:
            void0 = sess.run([(variables_graph[i]).assign(save_weights[i]) for i in range(len(variables_graph))])
            void1 = sess.run(op_assign)

        for epoch in range(epochs):
            print("Batch of classes {} out of {} batches".format(itera + 1, nb_groups))
            output_file.write("Batch of classes {} out of {} batches\n".format(itera + 1, nb_groups))
            print('Epoch %i' % epoch)
            output_file.write('Epoch %i\n' % epoch)
            output_file.flush()
            for i in range(int(np.ceil(len(files_from_cl) / batch_size))):
                loss_class_val, _, sc, lab = sess.run([loss_class, train_step, scores, label_batch_0],
                                                      feed_dict={learning_rate: lr})
                loss_batch.append(loss_class_val)

                # Plot the training error every 10 batches
                if len(loss_batch) == 10:
                    # print(np.mean(loss_batch))
                    loss_batch = []

                # Plot the training top 1 accuracy every 80 batches
                if (i + 1) % 80 == 0:
                    stat = []
                    stat += ([ll in best for ll, best in zip(lab, np.argsort(sc, axis=1)[:, -1:])])
                    stat = np.average(stat)
                    print('Training accuracy %f' % stat)
                    output_file.write('Training accuracy %f\n' % stat)
                    output_file.flush()

            # Decrease the learning by 5 every 10 epoch after 20 epochs at the first learning rate
            if epoch in lr_strat:
                lr /= lr_factor

        coord.request_stop()
        coord.join(threads)

        # copy weights to store network
        save_weights = sess.run([variables_graph[i] for i in range(len(variables_graph))])
        utils_resnet.save_model(save_path + 'model-iteration' + str(nb_cl) + '-%i.pickle' % itera, scope='ResNet18',
                                sess=sess)

    # Reset the graph
    tf.reset_default_graph()

    ## Exemplars management part  ##
    # nb_protos_cl = int(
    #     np.ceil(nb_proto * nb_groups * 1. / (itera + 1)))  # Reducing number of exemplars for the previous classes
    # files_from_cl = files_train[itera]
    # inits, scores, label_batch, loss_class, file_string_batch, op_feature_map = utils_icarl.reading_data_and_preparing_network(
    #     images_mean, files_from_cl, device, itera, batch_size, nb_groups, nb_cl, save_path)
    #
    # with tf.Session(config=config) as sess:
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     void3 = sess.run(inits)
    #
    #     # Load the training samples of the current batch of classes in the feature space to apply the herding algorithm
    #     Dtot, processed_files, label_dico = utils_icarl.load_class_in_feature_space(files_from_cl, batch_size, scores,
    #                                                                                 label_batch, loss_class,
    #                                                                                 file_string_batch, op_feature_map,
    #                                                                                 sess)
    #     processed_files = np.array([x.decode() for x in processed_files])
    #
    #     # Herding procedure : ranking of the potential exemplars
    #     print('Exemplars selection starting ...')
    #     output_file.write('Exemplars selection starting ...\n')
    #     output_file.flush()
    #     for iter_dico in range(nb_cl):
    #         ind_cl = np.where(label_dico == order[iter_dico + itera * nb_cl])[0]
    #         D = Dtot[:, ind_cl]
    #         files_iter = processed_files[ind_cl]
    #         mu = np.mean(D, axis=1)
    #         w_t = mu
    #         step_t = 0
    #         while not (len(files_protoset[itera * nb_cl + iter_dico]) == nb_protos_cl) and step_t < 1.1 * nb_protos_cl:
    #             tmp_t = np.dot(w_t, D)
    #             ind_max = np.argmax(tmp_t)
    #             w_t = w_t + mu - D[:, ind_max]
    #             step_t += 1
    #             if files_iter[ind_max] not in files_protoset[itera * nb_cl + iter_dico]:
    #                 files_protoset[itera * nb_cl + iter_dico].append(files_iter[ind_max])
    #
    #     coord.request_stop()
    #     coord.join(threads)

    # Reset the graph
    # tf.reset_default_graph()

    # Class means for iCaRL and NCM
    # print('Computing theoretical class means for NCM and mean-of-exemplars for iCaRL ...')
    # output_file.write('Computing theoretical class means for NCM and mean-of-exemplars for iCaRL ...\n')
    # output_file.flush()
    # for iteration2 in range(itera + 1):
    #     files_from_cl = files_train[iteration2]
    #     inits, scores, label_batch, loss_class, file_string_batch, op_feature_map = utils_icarl.reading_data_and_preparing_network(
    #         images_mean, files_from_cl, device, itera, batch_size, nb_groups, nb_cl, save_path)
    #
    #     with tf.Session(config=config) as sess:
    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(coord=coord)
    #         void2 = sess.run(inits)
    #
    #         Dtot, processed_files, label_dico = utils_icarl.load_class_in_feature_space(files_from_cl, batch_size,
    #                                                                                     scores, label_batch, loss_class,
    #                                                                                     file_string_batch,
    #                                                                                     op_feature_map, sess)
    #         processed_files = np.array([x.decode() for x in processed_files])
    #
    #         for iter_dico in range(nb_cl):
    #             ind_cl = np.where(label_dico == order[iter_dico + iteration2 * nb_cl])[0]
    #             D = Dtot[:, ind_cl]
    #             files_iter = processed_files[ind_cl]
    #             current_cl = order[range(iteration2 * nb_cl, (iteration2 + 1) * nb_cl)]
    #
    #             # Normal NCM mean
    #             class_means[:, order[iteration2 * nb_cl + iter_dico], 1, itera] = np.mean(D, axis=1)
    #             class_means[:, order[iteration2 * nb_cl + iter_dico], 1, itera] /= np.linalg.norm(
    #                 class_means[:, order[iteration2 * nb_cl + iter_dico], 1, itera])
    #
    #             # iCaRL approximated mean (mean-of-exemplars)
    #             # use only the first exemplars of the old classes: nb_protos_cl controls the number of exemplars per class
    #             ind_herding = np.array(
    #                 [np.where(files_iter == files_protoset[iteration2 * nb_cl + iter_dico][i])[0][0] for i in
    #                  range(min(nb_protos_cl, len(files_protoset[iteration2 * nb_cl + iter_dico])))])
    #             D_tmp = D[:, ind_herding]
    #             class_means[:, order[iteration2 * nb_cl + iter_dico], 0, itera] = np.mean(D_tmp, axis=1)
    #             class_means[:, order[iteration2 * nb_cl + iter_dico], 0, itera] /= np.linalg.norm(
    #                 class_means[:, order[iteration2 * nb_cl + iter_dico], 0, itera])
    #
    #         coord.request_stop()
    #         coord.join(threads)
    #
    #     # Reset the graph
    #     tf.reset_default_graph()

    # Pickle class means and protoset
    with open(save_path + str(nb_cl) + 'class_means.pickle', 'wb') as fp:
        cPickle.dump(class_means, fp)
    with open(save_path + str(nb_cl) + 'files_protoset.pickle', 'wb') as fp:
        cPickle.dump(files_protoset, fp)

output_file.close()
