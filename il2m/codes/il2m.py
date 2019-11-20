from __future__ import division
import numpy as np
import torch as th
import AverageMeter as AverageMeter
import sys, os, copy
from Utils import DataUtils
utils = DataUtils()

if len(sys.argv) != 8:
    print('Arguments: images_list_files_path, scores_path, b1_scores_path, dataset, S, P, K')
    sys.exit(-1)
#Parameters###############################
batch_size = 256
images_list_files_path = sys.argv[1]
scores_path = sys.argv[2]
b1_scores_path = sys.argv[3]
dataset = sys.argv[4]
S = int(sys.argv[5]) #number of states, including the first non-incremental one
P = int(sys.argv[6]) #number of classes per state
K = int(sys.argv[7]) #size of the memory for past class exemplars


###########################################
print('Dataset name = '+dataset)

top1_acc_ft = []
top1_acc_il2m = []
top5_acc_ft = []
top5_acc_il2m = []


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
for b in range(1, S + 1):
    old_classes_number = (b - 1) * P
    #computation of class means for the first non-incremental state
    if b == 1 :
        b1_train_list_path = os.path.join(images_list_files_path, dataset + '/S~' + str(S) + '/batch1/train.lst')
        b1_val_list_path = os.path.join(images_list_files_path, dataset + '/S~' + str(S) + '/batch1/val.lst')
        #read the raw prediction scores for training and validation data	
        b1_train_scores_path = os.path.join(b1_scores_path,  dataset + '/S~' + str(S) +'/train/scores')
        b1_val_scores_path = os.path.join(b1_scores_path,  dataset + '/S~' + str(S) +'/val/scores')
        b1_train_scores = open(b1_train_scores_path, 'r').readlines()
        b1_train_list = open(b1_train_list_path, 'r').readlines()
        val_images_paths_file = b1_val_list_path
        val_images_scores_file = b1_val_scores_path
        #compute the mean prediction scores for the classes initially learned in the first batch
        for (path_line, score_line) in zip(b1_train_list, b1_train_scores):
            path_line = path_line.strip().split()
            score_line = score_line.strip().split()
            image_path = path_line[0]
            image_class = int(path_line[1])
            np_scores = np.array(score_line, dtype=np.float)
            init_classes_means[image_class] += np_scores[image_class]
            init_classes_counts[image_class] += 1

            models_confidence[b] += np.max(np_scores)
            models_counts[b] += 1

        for c in range((b - 1) * P, b  * P):
            init_classes_means[c] = init_classes_means[c] / init_classes_counts[c]

        models_confidence[b] /= models_counts[b]
    #computation of class means for the incremental states (2 to S)    
    else:
        current_classes_means = {k : 0 for k in range(P * (b - 1))}
        current_classes_counts = {k : 0 for k in range(P * (b - 1))}


        ################################
        #read data for old and new classes for the current incremental state
        old_train_images_paths_file = os.path.join(images_list_files_path, dataset + '/S~' + str(S) + '/unbalanced/train'+'/K~' + str(K) + '/' + str(b) + '_old')
        new_train_images_paths_file = os.path.join(images_list_files_path, dataset + '/S~' + str(S) + '/unbalanced/train'+'/K~' + str(K) + '/' + str(b) + '_new')
        old_train_images_scores_file = os.path.join(scores_path, dataset + '/S~' + str(S) + '/K~' + str(K) + '/train/batch' + str(b) + '_old/scores')
        new_train_images_scores_file = os.path.join(scores_path, dataset + '/S~' + str(S) + '/K~' + str(K) + '/train/batch' + str(b) + '_new/scores')

        old_train_images_paths = open(old_train_images_paths_file, 'r').readlines()
        new_train_images_paths = open(new_train_images_paths_file, 'r').readlines()
        old_train_images_scores = open(old_train_images_scores_file, 'r').readlines()
        new_train_images_scores = open(new_train_images_scores_file, 'r').readlines()

        val_images_paths_file = os.path.join(images_list_files_path, dataset + '/S~' + str(S) + '/accumulated/val/batch' + str(b))
        val_images_scores_file = os.path.join(scores_path,  dataset + '/S~' + str(S) + '/K~' + str(K) + '/val/batch' + str(b) + '/scores')

        #computation of class means for new classes of the current state.
        #will be used to rectify scores in subsequent states, when these classes become old ones
        #the current state's model confidence is also computed
        for (path_line, score_line) in zip(new_train_images_paths, new_train_images_scores):
            path_line = path_line.strip().split()
            score_line = score_line.strip().split()
            image_path = path_line[0]
            image_class = int(path_line[1])
            np_scores = np.array(score_line, dtype=np.float)
            init_classes_means[image_class] += np_scores[image_class]
            init_classes_counts[image_class] += 1

            models_confidence[b] += np.max(np_scores)
            models_counts[b] += 1

        for c in range((b - 1) * P, b  * P):
            init_classes_means[c] = init_classes_means[c] / init_classes_counts[c]

        models_confidence[b] /= models_counts[b]

        #computation of class means for past classes of the current state.
        for (path_line, score_line) in zip(old_train_images_paths, old_train_images_scores):
            path_line = path_line.strip().split()
            score_line = score_line.strip().split()
            image_path = path_line[0]
            image_class = int(path_line[1])
            np_scores = np.array(score_line, dtype=np.float)
            current_classes_means[image_class] += np_scores[image_class]
            current_classes_counts[image_class] += 1

        for c in current_classes_means.keys():
            current_classes_means[c] = current_classes_means[c] / current_classes_counts[c]


    ################################################
    #Validation and scores rectification using IL2M.
    ################################################

    val_images_paths_file = open(val_images_paths_file, 'r').readlines()
    val_images_scores_file = open(val_images_scores_file, 'r').readlines()

    assert (len(val_images_paths_file) == len(val_images_scores_file))

    top1_ft = AverageMeter.AverageMeter()
    top5_ft = AverageMeter.AverageMeter()

    top1_il2m = AverageMeter.AverageMeter()
    top5_il2m = AverageMeter.AverageMeter()

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

        #rectify predicted scores only if the top-1 class predicted by the network is a new one
        #otherwise, rectification is not done because an old class, which has fewer examples, is directly predicted
        if int(predicted_class) >= old_classes_number:
            for o in range(old_classes_number):
                val_rectified_np_scores[o] *= init_classes_means[o] / current_classes_means[o] * models_confidence[b] / models_confidence[int(o / P) + 1]

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
            top1_il2m.update(prec1_rectified.item(), examples_counter)
            top5_il2m.update(prec5_rectified.item(), examples_counter)
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
        top1_il2m.update(prec1_rectified.item(), examples_counter)
        top5_il2m.update(prec5_rectified.item(), examples_counter)

    #print accuracy values for each incremental state
    print('[b{}] FT    | Val : acc@1 = {:.4f}% ; acc@5 = {:.4f}%'.format(b, top1_ft.avg, top5_ft.avg))
    print('[b{}] IL2M  | Val : acc@1 = {:.4f}% ; acc@5 = {:.4f}%'.format(b, top1_il2m.avg, top5_il2m.avg))
    print('***********************************************************************')

    top1_acc_ft.append(float(str(top1_ft.avg * 0.01)[:6]))
    top5_acc_ft.append(float(str(top5_ft.avg * 0.01)[:6]))
    top1_acc_il2m.append(float(str(top1_il2m.avg * 0.01)[:6]))
    top5_acc_il2m.append(float(str(top5_il2m.avg * 0.01)[:6]))

print('*********************************FT************************************')
print('Top1 Acc = '+str(top1_acc_ft))
print('Top5 Acc = '+str(top5_acc_ft))
#following Castro's "End-to-End Incremental Learning" methodology, the mean accuracy is computed only over incremental states
print('Mean inc Acc | acc@1 = {:.4f} | acc@5 = {:.4f}'.format(np.mean(np.array(top1_acc_ft[1:])), np.mean(np.array(top5_acc_ft[1:]))))

print('***********************************************************************')
print('*********************************IL2M**********************************')
print('Top1 Acc = '+str(top1_acc_il2m))
print('Top5 Acc = '+str(top5_acc_il2m))
#following Castro's "End-to-End Incremental Learning" methodology, the mean accuracy is computed only over incremental states
print('Mean inc Acc | acc@1 = {:.4f} | acc@5 = {:.4f}'.format(np.mean(np.array(top1_acc_il2m[1:])), np.mean(np.array(top5_acc_il2m[1:]))))
