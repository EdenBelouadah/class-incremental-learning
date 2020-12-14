from __future__ import division
import numpy as np
import torch as th
# import AverageMeter as AverageMeter
import sys, os
from Utils import DataUtils

if len(sys.argv) != 8:
    print('Arguments: fine_tuning_scores_path, K, P, S, Dataset_name [vgg_faces|ilsvrc|google_landmarks], first_batch_number [2-, last_batch_number')
    sys.exit(-1)
#Parameters###############################
batch_size = 256
scores_path =sys.argv[1]
memory_size = sys.argv[2]
P = int(sys.argv[3])
S = int(sys.argv[4])
dataset = sys.argv[5]
first_batch = int(sys.argv[6])
last_batch = int(sys.argv[7])


utils = DataUtils()
#Parameters###############################
images_list_files_path = '/home/eden/images_list_files/'
###########################################
print('Dataset name = '+dataset)
print('scores_path = '+scores_path)
print('K = '+memory_size)
print('P = '+str(P))
print('S = '+str(S))
print('first_batch = '+str(first_batch))
print('last_batch= '+str(last_batch))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



top1_accuracies = []
top1_accuracies_rectified = []
top5_accuracies = []
top5_accuracies_rectified = []


for batch_number in range(first_batch, last_batch + 1):
    # print('BATCH : '+str(batch_number))
    print('*****************************************************')
    #train
    old_train_images_paths_file = images_list_files_path+dataset+'/S~'+str(S)+'/unbalanced/train/K~'+memory_size+'/'+str(batch_number)+'_old'
    new_train_images_paths_file = images_list_files_path+dataset+'/S~'+str(S)+'/unbalanced/train/K~'+memory_size+'/'+str(batch_number)+'_new'
    #Val
    val_images_paths_file = images_list_files_path+dataset+'/S~'+str(S)+'/accumulated/val/batch'+str(batch_number)

    #train
    old_train_images_scores_file = os.path.join(scores_path,dataset+'/S~'+str(S)+'/K~'+memory_size+'/train/batch'+str(batch_number)+'_old/scores')
    new_train_images_scores_file = os.path.join(scores_path,dataset+'/S~'+str(S)+'/K~'+memory_size+'/train/batch'+str(batch_number)+'_new/scores')
    #Val
    val_images_scores_file = os.path.join(scores_path,dataset+'/S~'+str(S)+'/K~'+memory_size+'/val/batch'+str(batch_number)+'/scores')

    ###############################################################################################

    #statistics calculations
    old_train_images_paths_file = open(old_train_images_paths_file, 'r').readlines()
    new_train_images_paths_file = open(new_train_images_paths_file, 'r').readlines()

    old_train_images_scores_file = open(old_train_images_scores_file, 'r').readlines()
    new_train_images_scores_file = open(new_train_images_scores_file, 'r').readlines()

    val_images_paths_file = open(val_images_paths_file, 'r').readlines()
    val_images_scores_file = open(val_images_scores_file, 'r').readlines()

    assert(len(old_train_images_paths_file) == len(old_train_images_scores_file))
    assert(len(new_train_images_paths_file) == len(new_train_images_scores_file))
    assert(len(val_images_paths_file) == len(val_images_scores_file))

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    top1_rectified = utils.AverageMeter()
    top5_rectified = utils.AverageMeter()

    #############################################################################
    #compute total train images + train images per class

    total_nb_train_imgs = 0
    nb_train_imgs_per_class = {}

    for (old_train_path_line) in old_train_images_paths_file:
        old_train_path_line = old_train_path_line.strip().split()
        old_train_image_class = int(old_train_path_line[1])
        if old_train_image_class in nb_train_imgs_per_class.keys():
            nb_train_imgs_per_class[old_train_image_class] += 1
        else:
            nb_train_imgs_per_class[old_train_image_class] = 1

        total_nb_train_imgs += 1


    for (new_train_path_line) in new_train_images_paths_file:
        new_train_path_line = new_train_path_line.strip().split()
        new_train_image_class = int(new_train_path_line[1])
        if new_train_image_class in nb_train_imgs_per_class.keys():
            nb_train_imgs_per_class[new_train_image_class] += 1
        else:
            nb_train_imgs_per_class[new_train_image_class] = 1

        total_nb_train_imgs += 1

    ##############################################################################
    #OLD classes : Calibration and saving both calibrated / non-calibrated scores

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
        _, predicted_class = th.max(th.from_numpy(val_np_scores),0)
        val_rectified_np_scores = np.array(val_score_line, dtype=np.float)

        #apply softmax
        val_rectified_np_scores = softmax(val_rectified_np_scores)

        # for l in range(0,(batch_number-1)*P): #rectify old classes
        for l in range(val_rectified_np_scores.shape[0]):
            val_rectified_np_scores[l] /= (nb_train_imgs_per_class[l] / total_nb_train_imgs)

        _, rectified_predicted_class = th.max(th.from_numpy(val_rectified_np_scores), 0)


        full_labels.append(val_image_class)
        if full_np_rectified_scores is None:
            full_np_scores = val_np_scores
            full_np_rectified_scores = val_rectified_np_scores
        else:
            full_np_scores = np.vstack((full_np_scores, val_np_scores))
            full_np_rectified_scores = np.vstack((full_np_rectified_scores, val_rectified_np_scores))

        examples_counter += 1

        if examples_counter == batch_size:
            full_labels = th.from_numpy(np.array(full_labels, dtype=int))
            full_np_scores = th.from_numpy(full_np_scores)
            full_np_rectified_scores = th.from_numpy(full_np_rectified_scores)
            #compute accuracy
            prec1, prec5 = utils.accuracy(full_np_scores, full_labels, topk=(1, min(5, P * batch_number)))
            prec1_rectified, prec5_rectified = utils.accuracy(full_np_rectified_scores, full_labels, topk=(1, min(5, P * batch_number)))
            top1.update(prec1.item(), examples_counter)
            top5.update(prec5.item(), examples_counter)
            top1_rectified.update(prec1_rectified.item(), examples_counter)
            top5_rectified.update(prec5_rectified.item(), examples_counter)
            #re-init
            full_np_scores = None
            full_np_rectified_scores = None
            full_labels = []
            examples_counter = 0

    ##############################################################################
    ###########################################"
    if full_labels != []: #still missing some examples
        full_labels = th.from_numpy(np.array(full_labels, dtype=int))
        full_np_scores = th.from_numpy(full_np_scores)
        full_np_rectified_scores = th.from_numpy(full_np_rectified_scores)
        prec1, prec5 = utils.accuracy(full_np_scores, full_labels, topk=(1, min(5, P * batch_number)))
        prec1_rectified, prec5_rectified = utils.accuracy(full_np_rectified_scores, full_labels, topk=(1, min(5, P * batch_number)))
        top1.update(prec1.item(), examples_counter)
        top5.update(prec5.item(), examples_counter)
        top1_rectified.update(prec1_rectified.item(), examples_counter)
        top5_rectified.update(prec5_rectified.item(), examples_counter)


    #Accuracy

    print('[batch {}] Before Calibration | Val : acc@1 = {}% ; acc@5 = {}%'.format(batch_number, top1.avg, top5.avg))
    print('[batch {}] After Calibration  | Val : acc@1 = {}% ; acc@5 = {}%'.format(batch_number, top1_rectified.avg, top5_rectified.avg))
    # print('***********************************************************************')


    top1_accuracies.append(float(str(top1.avg*0.01)[:6]))
    top5_accuracies.append(float(str(top5.avg*0.01)[:6]))
    top1_accuracies_rectified.append(float(str(top1_rectified.avg*0.01)[:6]))
    top5_accuracies_rectified.append(float(str(top5_rectified.avg*0.01)[:6]))


print('*********Before rectification**********')
print('TOP 1 Before calibration:')
print(top1_accuracies)

print('TOP 5 Before calibration:')
print(top5_accuracies)

print('*********After rectification**********')
print('TOP 1 After calibration:')
print(top1_accuracies_rectified)

print('TOP 5 After calibration:')
print(top5_accuracies_rectified)

print('************************************')
print('BEFORE | TOP1 mean incremental accuracy = ' + str(np.mean(np.array(top1_accuracies))))
print('BEFORE | TOP5 mean incremental accuracy = ' + str(np.mean(np.array(top5_accuracies))))
print('************************************')
print('AFTER | TOP1 mean incremental accuracy = ' + str(np.mean(np.array(top1_accuracies_rectified))))
print('AFTER | TOP5 mean incremental accuracy = ' + str(np.mean(np.array(top5_accuracies_rectified))))




