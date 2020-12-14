from __future__ import division
import numpy as np
import torch as th
import AverageMeter as AverageMeter
import sys, os, copy
from Utils import DataUtils
utils = DataUtils()

if len(sys.argv) != 9:
    print('Arguments: images_list_files_path, exemplars_files_path, scores_path, b1_scores_path, K, P, S, dataset')
    sys.exit(-1)
#Parameters###############################
batch_size = 256
images_list_files_path = sys.argv[1]
exemplars_files_path = sys.argv[2]
scores_path = sys.argv[3]
b1_scores_path = sys.argv[4]
K = int(sys.argv[5]) #size of the memory for past class exemplars
P = int(sys.argv[6]) #number of classes per state
S = int(sys.argv[7]) #number of states, including the first non-incremental one
dataset = sys.argv[8]



###########################################
print('Dataset name = '+dataset)

top1_acc_ft = []
top1_acc_mc = []
top5_acc_ft = []
top5_acc_mc = []


#dictionaries to store statistics for past classes as learned in their initial states
init_classes_means = {k : 0 for k in range(S * P)}
init_classes_counts = {k : 0 * S for k in range(S * P)}

#dictionaries to store statistics for the classes as learned in the current incremental state
current_classes_means = {k : 0 for k in range(S * P)}
current_classes_counts = {k : 0 * S for k in range(S * P)}

#dictionaries to store statistics for model confidence in different states (i.e. average top-1 prediction scores per state)
models_confidence = {k:0 for k in range(1, S + 1)}
models_counts = {k:0 for k in range(1, S + 1)}

#go through the list of states from first to last
for b in range(2, S + 1):
    old_classes_number = (b - 1) * P
    #computation of class means for the first non-incremental state

    current_classes_means = {k : 0 for k in range(P * (b - 1))}
    current_classes_counts = {k : 0 for k in range(P * (b - 1))}

    old_classes_mean = 0
    new_classes_mean = 0
    old_classes_count = 0
    new_classes_count = 0

    ################################
    #read data for old and new classes for the current incremental state

    old_train_images_paths_file = os.path.join(exemplars_files_path, dataset + '/s' + str(S) + '/k' + str(K) + '/' + dataset + '_s' + str(S) + '_k' + str(K)+'_protoset_'+str(b - 2)+'.lst')
    new_train_images_paths_file = images_list_files_path + dataset + '/S~' + str(S) + '/separated/train/batch' + str(b)

    old_train_images_scores_file = os.path.join(scores_path, dataset + '/s' + str(S) + '/k' + str(K) + '/train/batch' + str(b) + '_old/scores')
    new_train_images_scores_file = os.path.join(scores_path, dataset + '/s' + str(S) + '/k' + str(K) + '/train/batch' + str(b) + '_new/scores')

    old_train_images_paths = open(old_train_images_paths_file, 'r').readlines()
    new_train_images_paths = open(new_train_images_paths_file, 'r').readlines()
    old_train_images_scores = open(old_train_images_scores_file, 'r').readlines()
    new_train_images_scores = open(new_train_images_scores_file, 'r').readlines()


    assert(len(old_train_images_paths) == len(old_train_images_scores))
    assert(len(new_train_images_paths) == len(new_train_images_scores))



    val_images_paths_file = os.path.join(images_list_files_path, dataset + '/S~' + str(S) + '/accumulated/val/batch' + str(b))
    val_images_scores_file = os.path.join(scores_path,  dataset + '/s' + str(S) + '/k' + str(K) + '/val/batch' + str(b) + '/scores')

    #computation of class means for new classes of the current state.
    #will be used to rectify scores in subsequent states, when these classes become old ones
    #the current state's model confidence is also computed
    for (path_line, score_line) in zip(new_train_images_paths, new_train_images_scores):
        path_line = path_line.strip().split()
        score_line = score_line.strip().split()
        image_path = path_line[0]
        image_class = int(path_line[1])
        np_scores = np.array(score_line, dtype=np.float)

        new_classes_mean += np_scores[image_class]
        new_classes_count += 1

    new_classes_mean /= new_classes_count

    #computation of class means for past classes of the current state.
    for (path_line, score_line) in zip(old_train_images_paths, old_train_images_scores):
        path_line = path_line.strip().split()
        score_line = score_line.strip().split()
        image_path = path_line[0]
        image_class = int(path_line[1])
        np_scores = np.array(score_line, dtype=np.float)

        old_classes_mean += np_scores[image_class]
        old_classes_count += 1

    old_classes_mean /= old_classes_count


    ################################################
    #Validation and scores rectification using mc.
    ################################################

    val_images_paths_file = open(val_images_paths_file, 'r').readlines()
    val_images_scores_file = open(val_images_scores_file, 'r').readlines()

    assert (len(val_images_paths_file) == len(val_images_scores_file))

    top1_ft = AverageMeter.AverageMeter()
    top5_ft = AverageMeter.AverageMeter()

    top1_mc = AverageMeter.AverageMeter()
    top5_mc = AverageMeter.AverageMeter()

    full_np_scores = None
    full_np_rectified_scores = None
    full_labels = []
    examples_counter = 0


    for (val_path_line, val_score_line) in zip(val_images_paths_file, val_images_scores_file):
        val_path_line = val_path_line.strip().split()
        val_score_line = val_score_line.strip().split()
        val_image_path = val_path_line[0]
        val_image_class = int(val_path_line[1])
        val_np_scores = np.array(val_score_line, dtype=np.float)
        predicted_class = np.argmax(val_np_scores)

        val_rectified_np_scores = copy.deepcopy(val_np_scores)

        for o in range(old_classes_number):
            val_rectified_np_scores[o] *= new_classes_mean / old_classes_mean

        #####################################

        full_labels.append(val_image_class)
        if full_np_rectified_scores is None:
            full_np_scores = val_np_scores
            full_np_rectified_scores = val_rectified_np_scores
        else:
            full_np_scores = np.vstack((full_np_scores, val_np_scores))
            full_np_rectified_scores = np.vstack((full_np_rectified_scores, val_rectified_np_scores))

        examples_counter += 1
        #update accuracy measures for each batch of images
        if examples_counter == batch_size:
            full_labels = th.from_numpy(np.array(full_labels, dtype=int))
            full_np_scores = th.from_numpy(full_np_scores)
            full_np_rectified_scores = th.from_numpy(full_np_rectified_scores)
            # compute top-1 and top-5 accuracy
            prec1, prec5 = utils.accuracy(full_np_scores, full_labels, topk=(1, min(5, b * P)))
            prec1_rectified, prec5_rectified = utils.accuracy(full_np_rectified_scores, full_labels, topk=(1, min(5, b * P)))
            top1_ft.update(prec1.item(), examples_counter)
            top5_ft.update(prec5.item(), examples_counter)
            top1_mc.update(prec1_rectified.item(), examples_counter)
            top5_mc.update(prec5_rectified.item(), examples_counter)
            # reinitialize the scores arrays
            full_np_scores = None
            full_np_rectified_scores = None
            full_labels = []
            examples_counter = 0

    #if there are some data left at the end, run a last update of the accuracy measures
    if full_labels != []:
        full_labels = th.from_numpy(np.array(full_labels, dtype=int))
        full_np_scores = th.from_numpy(full_np_scores)
        full_np_rectified_scores = th.from_numpy(full_np_rectified_scores)
        prec1, prec5 = utils.accuracy(full_np_scores, full_labels, topk=(1, min(5, b * P)))
        prec1_rectified, prec5_rectified = utils.accuracy(full_np_rectified_scores, full_labels, topk=(1, min(5, b * P)))
        top1_ft.update(prec1.item(), examples_counter)
        top5_ft.update(prec5.item(), examples_counter)
        top1_mc.update(prec1_rectified.item(), examples_counter)
        top5_mc.update(prec5_rectified.item(), examples_counter)

    #print accuracy values for each incremental state
    print('[b{}] LUCIR       | Val : acc@1 = {:.4f}% ; acc@5 = {:.4f}%'.format(b, top1_ft.avg, top5_ft.avg))
    print('[b{}] LUCIR+MC    | Val : acc@1 = {:.4f}% ; acc@5 = {:.4f}%'.format(b, top1_mc.avg, top5_mc.avg))
    print('***********************************************************************')

    top1_acc_ft.append(float(str(top1_ft.avg * 0.01)[:6]))
    top5_acc_ft.append(float(str(top5_ft.avg * 0.01)[:6]))
    top1_acc_mc.append(float(str(top1_mc.avg * 0.01)[:6]))
    top5_acc_mc.append(float(str(top5_mc.avg * 0.01)[:6]))

print('*********************************LUCIR************************************')
print('Top1 Acc = '+str(top1_acc_ft))
print('Top5 Acc = '+str(top5_acc_ft))
#following Castro's "End-to-End Incremental Learning" methodology, the mean accuracy is computed only over incremental states
print('Mean inc Acc | acc@1 = {:.4f} | acc@5 = {:.4f}'.format(np.mean(np.array(top1_acc_ft)), np.mean(np.array(top5_acc_ft))))

print('***********************************************************************')
print('*********************************LUCIR+MC**********************************')
print('Top1 Acc = '+str(top1_acc_mc))
print('Top5 Acc = '+str(top5_acc_mc))
#following Castro's "End-to-End Incremental Learning" methodology, the mean accuracy is computed only over incremental states
print('Mean inc Acc | acc@1 = {:.4f} | acc@5 = {:.4f}'.format(np.mean(np.array(top1_acc_mc)), np.mean(np.array(top5_acc_mc))))
