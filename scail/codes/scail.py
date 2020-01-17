#!/usr/bin/nohup python
# -*- coding: utf-8 -*-

__author__ = "Eden Belouadah & Adrian Popescu"

import sys, copy
import numpy as np
import torch as th
import AverageMeter as AverageMeter
from Utils import DataUtils

utils = DataUtils()
from sklearn import preprocessing

dataset = sys.argv[1]  # PARAM - name of the dataset
list_root_dir = sys.argv[2]  #  root dir for the list of images
local_root_dir = sys.argv[3]  # root dir on node for the features and the classification masks
Z = int(sys.argv[4])  #  PARAM - number of incremental Z for the dataset
B = int(sys.argv[5])  #  PARAM - size of the allowed B for past classes
P = int(sys.argv[6])  #  PARAM - size of each incremental batch
last_batch_number = int(sys.argv[7])  # PARAM - max number of incremental Z
top_rewinded = int(sys.argv[8])  #  PARAM - number of past classes used for rewinding

batch_size = 256

''' USAGE

######### CIFAR-100 ######### S=10 : K={1000,500,250} | S=20, S=50 : K=250
nohup python ./codes/scail.py cifar100 ./data/images_list_files /path/to/scail/data 10 1000 10 10 10   2>&1 | tee /path/to/your/logs/scail/cifar100/s10/k1000/scail_cifar100_s10_1k.log &
nohup python ./codes/scail.py cifar100 ./data/images_list_files /path/to/scail/data 10 500 10 10 10    2>&1 | tee /path/to/your/logs/scail/cifar100/s10/k500/scail_cifar100_s10_0,5k.log &
nohup python ./codes/scail.py cifar100 ./data/images_list_files /path/to/scail/data 10 250 10 10 10    2>&1 | tee /path/to/your/logs/scail/cifar100/s10/k250/scail_cifar100_s10_0,25k.log &
nohup python ./codes/scail.py cifar100 ./data/images_list_files /path/to/scail/data 20 250 5 20 10    2>&1 | tee /path/to/your/logs/scail/cifar100/s20/k250/scail_cifar100_s20_0,25k.log &
nohup python ./codes/scail.py cifar100 ./data/images_list_files /path/to/scail/data 50 250 2 50 10    2>&1 | tee /path/to/your/logs/scail/cifar100/s50/k250/scail_cifar100_s50_0,25k.log &

######### ILSVRC ######### S=10 : K={5000,10000,20000} | S=20, S=50 : K=5000
nohup python ./codes/scail.py ilsvrc ./data/images_list_files /path/to/scail/data 10 20000 100 10 10  2>&1 | tee /path/to/your/logs/scail/ilsvrc/s10/k20000/scail_ilsvrc_s10_20k.log &
nohup python ./codes/scail.py ilsvrc ./data/images_list_files /path/to/scail/data 10 10000 100 10 10  2>&1 | tee /path/to/your/logs/scail/ilsvrc/s10/k10000/scail_ilsvrc_s10_10k.log &
nohup python ./codes/scail.py ilsvrc ./data/images_list_files /path/to/scail/data 10 5000 100 10 10  2>&1 | tee /path/to/your/logs/scail/ilsvrc/s10/k5000/scail_ilsvrc_s10_5k.log &
nohup python ./codes/scail.py ilsvrc ./data/images_list_files /path/to/scail/data 20 5000 50 20 10  2>&1 | tee /path/to/your/logs/scail/ilsvrc/s20/k5000/scail_ilsvrc_s20_5k.log &
nohup python ./codes/scail.py ilsvrc ./data/images_list_files /path/to/scail/data 50 5000 20 50 10  2>&1 | tee /path/to/your/logs/scail/ilsvrc/s50/k5000/scail_ilsvrc_s50_5k.log &

######### VGG FACES ######### S=10 : K={10000,5000,2500} | S=20, S=50 : K=2500
nohup python ./codes/scail.py vgg_faces ./data/images_list_files /path/to/scail/data 10 10000 100 10 10  2>&1 | tee /path/to/your/logs/scail/vgg_faces/s10/k10000/scail_vgg_faces_s10_10k.log &
nohup python ./codes/scail.py vgg_faces ./data/images_list_files /path/to/scail/data 10 5000 100 10 10   2>&1 | tee /path/to/your/logs/scail/vgg_faces/s10/k5000/scail_vgg_faces_s10_5k.log &
nohup python ./codes/scail.py vgg_faces ./data/images_list_files /path/to/scail/data 10 2500 100 10 10   2>&1 | tee /path/to/your/logs/scail/vgg_faces/s10/k2500/scail_vgg_faces_s10_2,5k.log &
nohup python ./codes/scail.py vgg_faces ./data/images_list_files /path/to/scail/data 20 2500 50 20 10   2>&1 | tee /path/to/your/logs/scail/vgg_faces/s20/k2500/scail_vgg_faces_s50_2,5k.log &
nohup python ./codes/scail.py vgg_faces ./data/images_list_files /path/to/scail/data 50 2500 20 50 10   2>&1 | tee /path/to/your/logs/scail/vgg_faces/s50/k2500/scail_vgg_faces_s50_2,5k.log &

######### LANDMARKS ######### S=10 : K={8000,4000,2000} | S=20, S=50 : K=2000
nohup python ./codes/scail.py google_landmarks ./data/images_list_files /path/to/scail/data 10 8000 100 10 10   2>&1 | tee /path/to/your/logs/scail/google_landmarks/s10/k8000/scail_google_landmarks_s10_8k.log &
nohup python ./codes/scail.py google_landmarks ./data/images_list_files /path/to/scail/data 10 4000 100 10 10   2>&1 | tee /path/to/your/logs/scail/google_landmarks/s10/k4000/scail_google_landmarks_s10_4k.log  &
nohup python ./codes/scail.py google_landmarks ./data/images_list_files /path/to/scail/data 10 2000 100 10 10   2>&1 | tee /path/to/your/logs/scail/google_landmarks/s10/k2000/scail_google_landmarks_s10_2k.log &
nohup python ./codes/scail.py google_landmarks ./data/images_list_files /path/to/scail/data 20 2000 50 20 10   2>&1 | tee /path/to/your/logs/scail/google_landmarks/s20/k2000/scail_google_landmarks_s20_2k.log  &
nohup python ./codes/scail.py google_landmarks ./data/images_list_files /path/to/scail/data 50 2000 20 50 10   2>&1 | tee /path/to/your/logs/scail/google_landmarks/s50/k2000/scail_google_landmarks_s50_2k.log  &
'''

'''
implementation of scail function which performs the rewinding by sorting dimensions
of the classification masks in the current and initial state of a class and
matching them for multiplication with current features
'''

classif_masks_dir = local_root_dir + "/" + dataset + "/S~" + str(Z) + "/K~" + str(B) + "/classification_masks"
val_feat_dir = local_root_dir + "/" + dataset + "/S~" + str(Z) + "/K~" + str(B) + "/validation_features"

val_list = open(list_root_dir + "/" + dataset + "/S~" + str(Z) + "/accumulated/val/batch1", 'r').readlines()

top5_acc_ft = []
top5_acc_scail = []

for b in range(2, last_batch_number + 1):
    # get the boundaries of old and new classes
    max_old_label = (b - 1) * P
    max_new_label = b * P
    # open the weights of the current state and store them in a numpy array
    crt_classif_masks = th.load(classif_masks_dir + "/batch_" + str(b))
    np_crt_weights = crt_classif_masks[0].detach().numpy()
    np_crt_bias = crt_classif_masks[1].detach().numpy()

    #####################################################################################
    # compute the average values of new class biases in abs values
    sum_new_bias = np.sum(
        preprocessing.normalize(np.abs(np_crt_bias).reshape(-1, 1), norm='l2'))  # ? why not only new classes

    mean_dim_new = np.mean(-np.sort(-np.abs(np_crt_weights[max_old_label:max_new_label, :])), axis=0)

    # numpy matrix to store the rescaled weights using the scail formula
    scaled_np_weights = copy.deepcopy(np_crt_weights)
    scaled_np_bias = copy.deepcopy(np_crt_bias)

    # go through the old Z to get the initial stats of each class
    for b2 in range(1, b):
        # load masks matrices for old batches
        classif_masks_file = classif_masks_dir + "/batch_" + str(b2)
        x = th.load(classif_masks_file)
        np_weights = x[0].detach().numpy()
        np_bias = x[1].detach().numpy()
        min_label = (b2 - 1) * P
        max_label = b2 * P

        # BIAS RECTIFICATION
        #  compute the correction factor for biases which is the ratio between mean bias of new classes and mean bias of initial classes
        sum_init_bias = np.sum(preprocessing.normalize(np.abs(np_bias).reshape(-1, 1), norm='l2'))
        bias_factor = sum_new_bias / sum_init_bias

        scaled_np_bias[min_label:max_label] = np_bias[min_label:max_label] * bias_factor

        # WEIGHT RECTIFICATION
        # compute the mean values of maximum positive and negative activations of old classes
        mean_dim_old = np.mean(-np.sort(-np.abs(np_weights[min_label:max_label, :])), axis=0)
        weights_factor = mean_dim_new / mean_dim_old

        #  Rectification
        for label in range(min_label, max_label):
            argsrt = np.argsort(-np_weights[label, :])
            for dim in range(np_weights.shape[1]):
                scaled_np_weights[label][dim] = np_weights[label][dim] * weights_factor[argsrt[dim]]

    ##################################################################################
    ### read the initial scores and features and apply the rewinding to the top "top_rewinded" old classes
    #  create a list for scores
    scores_batch = val_feat_dir + "/batch_" + str(b) + "/scores"
    scores_list = open(scores_batch, 'r').readlines()
    feats_batch = val_feat_dir + "/batch_" + str(b) + "/features"
    feats_list = open(feats_batch, 'r').readlines()

    val_list = open(list_root_dir + "/" + dataset + "/S~" + str(Z) + "/accumulated/val/batch" + str(b),
                    'r').readlines()


    # Rectification & Testing
    top5_ft = AverageMeter.AverageMeter()
    top5_scail = AverageMeter.AverageMeter()

    full_np_scores = None
    full_np_scail_scores = None
    full_labels = []
    examples_counter = 0
    assert (len(feats_list) == len(val_list))
    for fline, pline in zip(feats_list, val_list):
        val_image_class = int(pline.strip().split()[-1])
        crt_feats = fline.rstrip().split(" ")
        np_crt_feats = np.asarray(crt_feats, dtype=float)
        test_np_scores = np_crt_feats.dot(np_crt_weights.T) + np_crt_bias
        test_scail_np_scores = np_crt_feats.dot(scaled_np_weights.T) + scaled_np_bias
        # create lists for top old and new classes
        top_old_labels = np.argsort(-test_np_scores[:max_old_label])[:top_rewinded]

        for o in range(max_old_label):
            if o in top_old_labels:
                test_scail_np_scores[o] = test_scail_np_scores[o]
            else:
                test_scail_np_scores[o] = -np.Inf

        full_labels.append(val_image_class)
        if full_np_scail_scores is None:
            full_np_scores = test_np_scores
            full_np_scail_scores = test_scail_np_scores
        else:
            full_np_scores = np.vstack((full_np_scores, test_np_scores))
            full_np_scail_scores = np.vstack((full_np_scail_scores, test_scail_np_scores))

        examples_counter += 1

        # update accuracy measures for each batch of images
        if examples_counter == batch_size:
            full_labels = th.from_numpy(np.array(full_labels, dtype=int))
            full_np_scores = th.from_numpy(full_np_scores)
            full_np_scail_scores = th.from_numpy(full_np_scail_scores)
            # compute accuracy
            prec5 = utils.accuracy(full_np_scores, full_labels, topk=[min(5, b * P)])
            prec5_scail = utils.accuracy(full_np_scail_scores, full_labels,
                                             topk=[min(5, b * P)])
            top5_ft.update(prec5[0].item(), examples_counter)
            top5_scail.update(prec5_scail[0].item(), examples_counter)
            # reinitialize the scores arrays
            full_np_scores = None
            full_np_scail_scores = None
            full_labels = []
            examples_counter = 0

    # if there are some data left at the end, run a last update of the accuracy measures
    if full_labels != []:
        full_labels = th.from_numpy(np.array(full_labels, dtype=int))
        full_np_scores = th.from_numpy(full_np_scores)
        full_np_scail_scores = th.from_numpy(full_np_scail_scores)
        prec5 = utils.accuracy(full_np_scores, full_labels, topk=[min(5, b * P)])
        prec5_scail = utils.accuracy(full_np_scail_scores, full_labels, topk=[min(5, b * P)])
        top5_ft.update(prec5[0].item(), examples_counter)
        top5_scail.update(prec5_scail[0].item(), examples_counter)

    # print accuracy values for each incremental state
    print('[b{}] FT     | Val : acc@5 = {:.4f}%'.format(b, top5_ft.avg))
    print('[b{}] Scail  | Val : acc@5 = {:.4f}%'.format(b, top5_scail.avg))
    print('***********************************************************************')

    top5_acc_ft.append(float(str(top5_ft.avg)[:6]))
    top5_acc_scail.append(float(str(top5_scail.avg)[:6]))

print('*********************************FT************************************')
print('Top5 Acc = ' + str(top5_acc_ft))
# following Castro's "End-to-End Incremental Learning" methodology, the mean accuracy is computed only over incremental states
print('Mean inc Acc | acc@5 = {:.2f}'.format(np.mean(np.array(top5_acc_ft))))

print('***********************************************************************')
print('*********************************Scail**********************************')
print('Top5 Acc = ' + str(top5_acc_scail))
# following Castro's "End-to-End Incremental Learning" methodology, the mean accuracy is computed only over incremental states
print('Mean inc Acc | acc@5 = {:.2f}'.format(np.mean(np.array(top5_acc_scail))))
