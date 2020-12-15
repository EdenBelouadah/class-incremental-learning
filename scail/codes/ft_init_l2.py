from __future__ import division
import numpy as np
import torch as th
import AverageMeter as AverageMeter
import sys, os
from sklearn import preprocessing

if len(sys.argv) != 10:
    print('Arguments: ft_feat_scores_path, ft_weights_dir, first_model_weights_path, K, P, S, Dataset_name [vgg_faces|ilsvrc|google_landmarks], first_batch_number, last_batch_number')
    sys.exit(-1)
#Parameters###############################
batch_size = 256
scores_path =sys.argv[1]
ft_weights_dir =sys.argv[2]
first_model_weights_path =sys.argv[3]
memory_size = sys.argv[4]
P = int(sys.argv[5])
S = int(sys.argv[6])
dataset = sys.argv[7]
first_batch = int(sys.argv[8])
last_batch = int(sys.argv[9])

#Parameters###############################
images_list_files_path = '/scratch_global/eden/images_list_files/'
###########################################
print('Dataset name = '+dataset)

global_mean_old_classes = 0
global_mean_new_classes = 0


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


top1_accuracies = []
top1_accuracies_rectified = []
top5_accuracies = []
top5_accuracies_rectified = []
old_classes_mean_scores = []
old_classes_std_scores = []
new_classes_mean_scores = []
new_classes_std_scores = []

batch_initial_weight_matrix = {}
batch_initial_bias_vector = {}

#get first batch weights and bias

first_model_classif_param = th.load(first_model_weights_path)
np_first_model_weights = first_model_classif_param[0].detach().numpy()
np_first_model_bias = first_model_classif_param[1].detach().numpy()


batch_initial_weight_matrix[1] = np_first_model_weights
batch_initial_bias_vector[1] = np_first_model_bias


for batch_number in range(first_batch, last_batch + 1):
    # print('BATCH : '+str(batch_number))
    print('*****************************************************')

    #Val
    val_images_paths_file = images_list_files_path+dataset+'/S~'+str(S)+'/accumulated/val/batch'+str(batch_number)

    #Val
    val_images_scores_file = os.path.join(scores_path,dataset+'/S~'+str(S)+'/K~'+memory_size+'/val/batch'+str(batch_number)+'/scores')
    val_images_features_file = os.path.join(scores_path,dataset+'/S~'+str(S)+'/K~'+memory_size+'/val/batch'+str(batch_number)+'/features')

    batch_ft_weights_path = os.path.join(ft_weights_dir, 'b'+str(batch_number)+'_weight_bias.pt')
    model_classif_param = th.load(batch_ft_weights_path)
    np_model_weights = model_classif_param[0].detach().numpy()
    np_model_bias = model_classif_param[1].detach().numpy()


    #insert in the dict
    batch_initial_weight_matrix[batch_number] = np_model_weights[(batch_number - 1) * P : batch_number*P]
    batch_initial_bias_vector[batch_number] = np_model_bias[(batch_number - 1) * P : batch_number*P]


    #erase current weights with initial batch weights:
    for b2 in range(1, batch_number):
        np_model_weights[(b2 - 1) * P : b2*P] = batch_initial_weight_matrix[b2]
        np_model_bias[(b2 - 1) * P : b2*P] = batch_initial_bias_vector[b2]



    np_model_weights = preprocessing.normalize(np_model_weights, axis=1, norm='l2')
    np_model_bias = preprocessing.normalize(np_model_bias.reshape(1,-1), norm='l2')

###############################################################################################

    val_images_paths_file = open(val_images_paths_file, 'r').readlines()
    val_images_scores_file = open(val_images_scores_file, 'r').readlines()
    val_images_features_file = open(val_images_features_file, 'r').readlines()

    assert(len(val_images_paths_file) == len(val_images_scores_file) ==len(val_images_features_file) )

    top1 = AverageMeter.AverageMeter()
    top5 = AverageMeter.AverageMeter()

    top1_rectified = AverageMeter.AverageMeter()
    top5_rectified = AverageMeter.AverageMeter()
    ###############################################################################################
    ##############################################################################

    full_np_scores = None
    full_np_rectified_scores = None
    full_labels = []
    examples_counter = 0


    for (val_path_line, val_score_line, val_feat_line) in zip(val_images_paths_file, val_images_scores_file, val_images_features_file):
        val_path_line = val_path_line.strip().split()
        val_score_line = val_score_line.strip().split()
        val_feat_line = val_feat_line.strip().split()
        val_image_path = val_path_line[0]
        val_image_class = int(val_path_line[1])
        val_np_scores = np.array(val_score_line, dtype=np.float)
        val_np_feat = np.array(val_feat_line, dtype=np.float).reshape(-1, 1)
        predicted_class = np.argmax(val_np_scores)



        val_rectified_np_scores = val_np_feat.T.dot(np_model_weights.T) + np_model_bias

        rectified_predicted_class = np.argmax(val_rectified_np_scores)


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
            prec1, prec5 = accuracy(full_np_scores, full_labels, topk=(1, min(5, batch_number * P)))
            prec1_rectified, prec5_rectified = accuracy(full_np_rectified_scores, full_labels, topk=(1, min(5, batch_number * P)))
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
        prec1, prec5 = accuracy(full_np_scores, full_labels, topk=(1, min(5, batch_number * P)))
        prec1_rectified, prec5_rectified = accuracy(full_np_rectified_scores, full_labels, topk=(1, min(5, batch_number * P)))
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


print('TOP 1 Before calibration:')
print(top1_accuracies)
print('TOP 1 After calibration:')
print(top1_accuracies_rectified)
print('TOP 5 Before calibration:')
print(top5_accuracies)
print('TOP 5 After calibration:')
print(top5_accuracies_rectified)

print('TOP1 mean incremental accuracy = ' + str(np.mean(np.array(top1_accuracies))))
print('TOP5 mean incremental accuracy = ' + str(np.mean(np.array(top5_accuracies))))

print('*********After rectification**********')
print('TOP1 mean incremental accuracy = ' + str(np.mean(np.array(top1_accuracies_rectified))))
print('TOP5 mean incremental accuracy = ' + str(np.mean(np.array(top5_accuracies_rectified))))
