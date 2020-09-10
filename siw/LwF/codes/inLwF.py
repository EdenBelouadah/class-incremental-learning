from __future__ import division
import numpy as np
import torch as th
import AverageMeter as AverageMeter
import sys, os
from sklearn import preprocessing
from Utils import DataUtils
import copy
try:
    import cPickle
except:
    import _pickle as cPickle


if len(sys.argv) != 7:
    print('Arguments: ft_feat_scores_path, ft_weights_path, K, P, S, Dataset_name [vgg_faces|ilsvrc|google_landmarks]')
    sys.exit(-1)
#Parameters###############################
batch_size = 256
scores_path =sys.argv[1]
weights_path =sys.argv[2]
memory_size = sys.argv[3]
P = int(sys.argv[4])
S = int(sys.argv[5])
dataset = sys.argv[6]

#Parameters#############################################
print('Dataset name = '+dataset)



utils = DataUtils()

top1_accuracies = []
rectified_top1_accuracies = []
top5_accuracies = []
rectified_top5_accuracies = []


batch_initial_weight_matrix = {}
batch_initial_bias_vector = {}

#get first batch weights and bias
first_model_weights_path = os.path.join(weights_path, dataset + '/S~' + str(S) + '/K~' + memory_size + '/b1_weight_bias.tf')

with open(first_model_weights_path, 'rb') as fp:
    np_first_model_weights = cPickle.load(fp)
    np_first_model_bias = cPickle.load(fp)

# print(np_first_model_weights.shape)
# print(np_first_model_bias.shape)


batch_initial_weight_matrix[1] = np_first_model_weights
batch_initial_bias_vector[1] = np_first_model_bias


for batch_number in range(1, S + 1):
    print('*****************************************************')
    # print('BATCH : '+str(batch_number))
    val_images_paths_file = os.path.join(scores_path,dataset+'/S~'+str(S)+'/K~'+memory_size+'/val/batch'+str(batch_number)+'/paths_features.lst')
    val_images_features_file = os.path.join(scores_path,dataset+'/S~'+str(S)+'/K~'+memory_size+'/val/batch'+str(batch_number)+'/features.raw')

    batch_weights_path = os.path.join(weights_path, dataset+'/S~'+str(S)+'/K~'+memory_size+'/b'+str(batch_number)+'_weight_bias.tf')

    with open(batch_weights_path, 'rb') as fp:
        np_model_weights = cPickle.load(fp)
        np_model_bias = cPickle.load(fp)


    rectified_np_model_weights = copy.deepcopy(np_model_weights)
    rectified_np_model_bias = copy.deepcopy(np_model_bias)

    # print(np_model_weights.shape)
    # print(np_model_bias.shape)



    # insert in the dict
    batch_initial_weight_matrix[batch_number] = np_model_weights[:, (batch_number - 1) * P: batch_number * P]
    batch_initial_bias_vector[batch_number] = np_model_bias[:, (batch_number - 1) * P: batch_number * P]

    # print(batch_initial_weight_matrix[batch_number].shape)
    # print(batch_initial_bias_vector[batch_number].shape)

    # sys.exit(-1)
    # erase current weights with initial batch weights:
    for b2 in range(1, batch_number):
        rectified_np_model_weights[:, (b2 - 1) * P: b2 * P] = batch_initial_weight_matrix[b2]
        rectified_np_model_bias[:, (b2 - 1) * P: b2 * P] = batch_initial_bias_vector[b2]



    ###############################################################################################

    #statistics calculations

    val_images_paths_file = open(val_images_paths_file, 'r').readlines()
    # val_images_scores_file = open(val_images_scores_file, 'r').readlines()
    val_images_features_file = open(val_images_features_file, 'r').readlines()

    assert(len(val_images_paths_file) == len(val_images_features_file) )

    top1 = AverageMeter.AverageMeter()
    rectified_top1 = AverageMeter.AverageMeter()
    top5 = AverageMeter.AverageMeter()
    rectified_top5 = AverageMeter.AverageMeter()

    ###############################################################################################
    ##############################################################################
    #OLD classes : Calibration and saving both calibrated / non-calibrated scores

    full_np_scores = None
    rectified_full_np_scores = None
    full_labels = []
    examples_counter = 0


    for (val_path_line, val_feat_line) in zip(val_images_paths_file, val_images_features_file):
        val_path_line = val_path_line.strip().split()
        val_feat_line = val_feat_line.strip().split()
        val_image_path = val_path_line[0]
        val_image_class = int(val_path_line[1])
        val_np_feat = np.array(val_feat_line, dtype=np.float).reshape(1, -1)
        val_np_scores = val_np_feat.dot(np_model_weights) + np_model_bias


        rectified_val_np_scores = val_np_feat.dot(rectified_np_model_weights) + rectified_np_model_bias

        predicted_class = np.argmax(val_np_scores)
        rectified_predicted_class = np.argmax(rectified_val_np_scores)


        full_labels.append(val_image_class)
        if full_np_scores is None:
            full_np_scores = val_np_scores
            rectified_full_np_scores = rectified_val_np_scores
        else:
            full_np_scores = np.vstack((full_np_scores, val_np_scores))
            rectified_full_np_scores = np.vstack((rectified_full_np_scores, rectified_val_np_scores))

        examples_counter += 1

        if examples_counter == batch_size:
            full_labels = th.from_numpy(np.array(full_labels, dtype=int))
            full_np_scores = th.from_numpy(full_np_scores)
            rectified_full_np_scores = th.from_numpy(rectified_full_np_scores)
            #compute accuracy
            prec1, prec5 = utils.accuracy(full_np_scores, full_labels, topk=(1, min(5, batch_number * P)))
            rectified_prec1, rectified_prec5 = utils.accuracy(rectified_full_np_scores, full_labels, topk=(1, min(5, batch_number * P)))
            top1.update(prec1.item(), examples_counter)
            top5.update(prec5.item(), examples_counter)
            rectified_top1.update(rectified_prec1.item(), examples_counter)
            rectified_top5.update(rectified_prec5.item(), examples_counter)
            #re-init
            full_np_scores = None
            rectified_full_np_scores = None
            full_labels = []
            examples_counter = 0

    ##############################################################################
    ###########################################"
    if full_labels != []: #still missing some examples
        full_labels = th.from_numpy(np.array(full_labels, dtype=int))
        full_np_scores = th.from_numpy(full_np_scores)
        rectified_full_np_scores = th.from_numpy(rectified_full_np_scores)
        prec1, prec5 = utils.accuracy(full_np_scores, full_labels, topk=(1, min(5, batch_number * P)))
        rectified_prec1, rectified_prec5 = utils.accuracy(rectified_full_np_scores, full_labels, topk=(1, min(5, batch_number * P)))
        top1.update(prec1.item(), examples_counter)
        rectified_top1.update(rectified_prec1.item(), examples_counter)
        top5.update(prec5.item(), examples_counter)
        rectified_top5.update(rectified_prec5.item(), examples_counter)


    #Accuracy
    print('[b{}] | before | Val : acc@1 = {:.1f}% ; acc@5 = {:.1f}%'.format(batch_number, top1.avg, top5.avg))
    print('[b{}] | after  | Val : acc@1 = {:.1f}% ; acc@5 = {:.1f}%'.format(batch_number, rectified_top1.avg, rectified_top5.avg))


    top1_accuracies.append(float(str(top1.avg*0.01)[:6]))
    rectified_top1_accuracies.append(float(str(rectified_top1.avg*0.01)[:6]))
    top5_accuracies.append(float(str(top5.avg*0.01)[:6]))
    rectified_top5_accuracies.append(float(str(rectified_top5.avg*0.01)[:6]))


print('*****************************************************')
print('TOP 1 Before calibration:')
print(top1_accuracies)
print('TOP 1 After calibration:')
print(rectified_top1_accuracies)
print('TOP 5 Before calibration:')
print(top5_accuracies)
print('TOP 5 After calibration:')
print(rectified_top5_accuracies)

print('*********Before rectification**********')
print('TOP1 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(top1_accuracies[1:]))*100))
print('TOP5 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(top5_accuracies[1:]))*100))
print('*********After rectification**********')
print('TOP1 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(rectified_top1_accuracies[1:]))*100))
print('TOP5 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(rectified_top5_accuracies[1:]))*100))
