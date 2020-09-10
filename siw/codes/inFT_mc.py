from __future__ import division
import numpy as np
import torch as th
import AverageMeter as AverageMeter
import sys, os
from sklearn import preprocessing
from Utils import DataUtils
utils = DataUtils()


if len(sys.argv) != 8:
    print('Arguments: images_list_files_path, ft_feat_scores_path, ft_weights_dir, K, P, S, dataset')
    sys.exit(-1)

#Parameters###############################
batch_size = 256
images_list_files_path =sys.argv[1]
scores_path =sys.argv[2]
ft_weights_dir =sys.argv[3]
memory_size = sys.argv[4]
P = int(sys.argv[5])
S = int(sys.argv[6])
dataset = sys.argv[7]
print('*****************************************************')
print('python '+ ' '.join(sys.argv))
###########################################

top1_accuracies = []
top1_accuracies_rectified = []
top5_accuracies = []
top5_accuracies_rectified = []

batch_initial_weight_matrix = {}
batch_initial_bias_vector = {}
batch_train_mean_score = {k : 0 for k in range(S + 1)}
batch_train_counter =    {k : 0 for k in range(S + 1)}

ft_weights_dir = os.path.join(ft_weights_dir, dataset, 's'+str(S), 'k'+str(memory_size))
#get first batch weights and bias
first_model_classif_param = th.load(os.path.join(ft_weights_dir, 'b1_weight_bias.pt'))
np_first_model_weights = first_model_classif_param[0].detach().numpy()
np_first_model_bias = first_model_classif_param[1].detach().numpy()


batch_initial_weight_matrix[1] = np_first_model_weights
batch_initial_bias_vector[1] = np_first_model_bias

#compute train mean score per state:
for batch_number in range(1, S+1):
    # train
    train_images_paths_file = images_list_files_path + dataset + '/s' + str(S) + '/separated/train/batch' + str(batch_number)

    # train
    train_images_scores_file = os.path.join(scores_path, dataset + '/s' + str(S) + '/k' + memory_size + '/train/batch' + str(batch_number) + '/scores')
    train_images_features_file = os.path.join(scores_path, dataset + '/s' + str(S) + '/k' + memory_size + '/train/batch' + str(batch_number) + '/features')

    # train
    train_images_paths_file = open(train_images_paths_file, 'r').readlines()
    train_images_scores_file = open(train_images_scores_file, 'r').readlines()
    train_images_features_file = open(train_images_features_file, 'r').readlines()

    assert (len(train_images_paths_file) == len(train_images_scores_file) == len(train_images_features_file))

    ######################## compute mean train score:
    for (train_path_line, train_score_line, train_feat_line) in zip(train_images_paths_file, train_images_scores_file, train_images_features_file):
        train_path_line = train_path_line.strip().split()
        train_score_line = train_score_line.strip().split()
        train_feat_line = train_feat_line.strip().split()
        train_image_path = train_path_line[0]
        train_image_class = int(train_path_line[1])
        train_np_scores = np.array(train_score_line, dtype=np.float)
        if train_image_class >= (batch_number - 1) * P and train_image_class < batch_number * P :
            batch_train_counter[batch_number] += 1
            batch_train_mean_score[batch_number] += train_np_scores[train_image_class]


for batch_number in range(1, S+1):
    batch_train_mean_score[batch_number] /= batch_train_counter[batch_number]

for batch_number in range(2, S + 1):
    print('*****************************************************')

    #test 
    test_images_paths_file = images_list_files_path+dataset+'/s'+str(S)+'/accumulated/test/batch'+str(batch_number)

    #test
    test_images_scores_file = os.path.join(scores_path,dataset+'/s'+str(S)+'/k'+memory_size+'/test/batch'+str(batch_number)+'/scores')
    test_images_features_file = os.path.join(scores_path,dataset+'/s'+str(S)+'/k'+memory_size+'/test/batch'+str(batch_number)+'/features')


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


    ###############################################################################################
    #test
    test_images_paths_file = open(test_images_paths_file, 'r').readlines()
    test_images_scores_file = open(test_images_scores_file, 'r').readlines()
    test_images_features_file = open(test_images_features_file, 'r').readlines()

    assert(len(test_images_paths_file) == len(test_images_scores_file) ==len(test_images_features_file) )
    

    #print(len(test_images_paths_file))
    top1 = AverageMeter.AverageMeter()
    top5 = AverageMeter.AverageMeter()

    top1_rectified = AverageMeter.AverageMeter()
    top5_rectified = AverageMeter.AverageMeter()
    ################################################################################

    full_np_scores = None
    full_np_rectified_scores = None
    full_labels = []
    examples_counter = 0


    for (test_path_line, test_score_line, test_feat_line) in zip(test_images_paths_file, test_images_scores_file, test_images_features_file):
        test_path_line = test_path_line.strip().split()
        test_score_line = test_score_line.strip().split()
        test_feat_line = test_feat_line.strip().split()
        test_image_path = test_path_line[0]
        test_image_class = int(test_path_line[1])
        test_np_scores = np.array(test_score_line, dtype=np.float)
        test_np_feat = np.array(test_feat_line, dtype=np.float).reshape(-1, 1)


        predicted_class = np.argmax(test_np_scores)


        test_rectified_np_scores = test_np_feat.T.dot(np_model_weights.T) + np_model_bias

        for o in range((batch_number - 1) * P): #for all past classes
            test_rectified_np_scores[0][o] *=  batch_train_mean_score[batch_number] / batch_train_mean_score[int(o / P) + 1]

        rectified_predicted_class = np.argmax(test_rectified_np_scores)

        full_labels.append(test_image_class)
        if full_np_rectified_scores is None:
            full_np_scores = test_np_scores
            full_np_rectified_scores = test_rectified_np_scores
        else:
            full_np_scores = np.vstack((full_np_scores, test_np_scores))
            full_np_rectified_scores = np.vstack((full_np_rectified_scores, test_rectified_np_scores))

        examples_counter += 1

        if examples_counter == batch_size:
            full_labels = th.from_numpy(np.array(full_labels, dtype=int))
            full_np_scores = th.from_numpy(full_np_scores)
            full_np_rectified_scores = th.from_numpy(full_np_rectified_scores)
            #compute accuracy
            prec1, prec5 = utils.accuracy(full_np_scores, full_labels, topk=(1, min(5, batch_number * P)))
            prec1_rectified, prec5_rectified = utils.accuracy(full_np_rectified_scores, full_labels, topk=(1, min(5, batch_number * P)))
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
    if full_labels != []: #still missing some examples
        full_labels = th.from_numpy(np.array(full_labels, dtype=int))
        full_np_scores = th.from_numpy(full_np_scores)
        full_np_rectified_scores = th.from_numpy(full_np_rectified_scores)
        prec1, prec5 = utils.accuracy(full_np_scores, full_labels, topk=(1, min(5, batch_number * P)))
        prec1_rectified, prec5_rectified = utils.accuracy(full_np_rectified_scores, full_labels, topk=(1, min(5, batch_number * P)))
        top1.update(prec1.item(), examples_counter)
        top5.update(prec5.item(), examples_counter)
        top1_rectified.update(prec1_rectified.item(), examples_counter)
        top5_rectified.update(prec5_rectified.item(), examples_counter)


    #Accuracy
    print('[batch {}] Before Calibration | test : acc@1 = {:.1f}% ; acc@5 = {:.1f}%'.format(batch_number, top1.avg, top5.avg))
    print('[batch {}] After Calibration  | test : acc@1 = {:.1f}% ; acc@5 = {:.1f}%'.format(batch_number, top1_rectified.avg, top5_rectified.avg))
    # print('***********************************************************************')


    top1_accuracies.append(float(str(top1.avg*0.01)[:6]))
    top5_accuracies.append(float(str(top5.avg*0.01)[:6]))
    top1_accuracies_rectified.append(float(str(top1_rectified.avg*0.01)[:6]))
    top5_accuracies_rectified.append(float(str(top5_rectified.avg*0.01)[:6]))


print('*****************************************************')
print('TOP 1 Before calibration:')
print(top1_accuracies)
print('TOP 1 After calibration:')
print(top1_accuracies_rectified)
print('TOP 5 Before calibration:')
print(top5_accuracies)
print('TOP 5 After calibration:')
print(top5_accuracies_rectified)
print('TOP1 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(top1_accuracies))*100))
print('TOP5 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(top5_accuracies))*100))
print('*********After rectification**********')
print('TOP1 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(top1_accuracies_rectified))*100))
print('TOP5 mean incremental accuracy = {:.1f} '.format(np.mean(np.array(top5_accuracies_rectified))*100))