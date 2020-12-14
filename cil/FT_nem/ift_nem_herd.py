from __future__ import division
import numpy as np
import torch as th
import AverageMeter as AverageMeter
import sys
from sklearn import preprocessing
from torch.autograd import Variable
import torch, os

if len(sys.argv) != 9:
    print('Arguments: nem_features_path, exemplars_files_path, S, P, K, Dataset_name [vgg_faces|ilsvrc|google_landmarks|places365], first_b, last_b')
    sys.exit(-1)
#Parameters###############################
batch_size = 256
features_path =sys.argv[1]
exemplars_files_path =sys.argv[2]
S = sys.argv[3]
P = int(sys.argv[4])
K = sys.argv[5]
dataset = sys.argv[6]
first_batch = int(sys.argv[7])
last_batch = int(sys.argv[8])


#Parameters#####################################################################
print('Dataset name = '+dataset)
images_list_files_path = '/home/eden/images_list_files/'

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # batch_size = 1
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


top5_accuracies = []
top1_accuracies = []

for b in range(first_batch, last_batch + 1):
    top5 = AverageMeter.AverageMeter()
    top1 = AverageMeter.AverageMeter()
    #train
    reduced_train_images_paths_file = os.path.join(exemplars_files_path,dataset+'/S~'+S+'/K~'+K+'/'+str(b+1)+'_old')
    #Val
    val_images_paths_file = os.path.join(images_list_files_path,dataset+'/S~'+S+'/accumulated/val/batch'+str(b))

    #train
    reduced_train_images_features_file = os.path.join(features_path,dataset+'/S~'+S+'/K~'+K+'/train/batch'+str(b+1)+'_old/features')
    #Val
    val_images_features_file = os.path.join(features_path,dataset+'/S~'+S+'/K~'+K+'/val/batch'+str(b)+'/features')

    ###############################################################################################

    #loading files
    reduced_train_images_paths_file = open(reduced_train_images_paths_file, 'r').readlines()

    reduced_train_images_features_file = open(reduced_train_images_features_file, 'r').readlines()

    val_images_paths_file = open(val_images_paths_file, 'r').readlines()
    val_images_features_file = open(val_images_features_file, 'r').readlines()

    assert(len(reduced_train_images_paths_file) == len(reduced_train_images_features_file))
    assert(len(val_images_paths_file) == len(val_images_features_file))


    ###############################################################################################
    # computing classes means
    reduced_classes_features = {}

    for (path_line, feature_line) in zip(reduced_train_images_paths_file, reduced_train_images_features_file):
        path_line = path_line.strip().split()
        feature_line = feature_line.strip().split()
        image_class = int(path_line[1])
        feature = np.array(feature_line, dtype=np.float).reshape(1, -1)
        feature = preprocessing.normalize(feature, norm='l2')
        if image_class not in reduced_classes_features.keys():
            reduced_classes_features[image_class] = feature
        else:
            reduced_classes_features[image_class] = np.vstack((reduced_classes_features[image_class], feature))


    assert(len(reduced_classes_features.keys()) == b * P)
    #computing means
    reduced_classes_means = {}
    for key in reduced_classes_features.keys():
        class_mean = np.mean(reduced_classes_features[key], axis = 0).reshape(1, -1)
        reduced_classes_means[key] = preprocessing.normalize(class_mean, norm='l2')

    assert(len(reduced_classes_means.keys()) == b * P)

    #validation
    for (path_line, feature_line) in zip(val_images_paths_file, val_images_features_file):
        path_line = path_line.strip().split()
        feature_line = feature_line.strip().split()
        image_class = int(path_line[1])
        feature = np.array(feature_line, dtype=np.float).reshape(1, -1)
        feature = preprocessing.normalize(feature, norm='l2')

        #compute nearest mean
        outputs = []
        for i in range(b*P):
            outputs.append(- np.linalg.norm(feature - reduced_classes_means[i]))

        label = torch.tensor([image_class])
        outputs = np.array(outputs).reshape(1, -1)
        outputs = Variable(torch.from_numpy(outputs))

        # Accuracy
        prec1, prec5 = accuracy(outputs.data, label, topk=(1, min(5, b * P)))
        top5.update(prec5.item(), 1)
        top1.update(prec1.item(), 1)

    top5_accuracies.append(top5.avg)
    top1_accuracies.append(top1.avg)
    print('Batch '+ str(b)+ ' | Val : acc@1 = {}% ; acc@5 = {}%'.format(top1.avg, top5.avg))


print('TOP1 validation accuracies = '+str([float(str(e)[:6]) for e in top1_accuracies]))
print('TOP5 validation accuracies = '+str([float(str(e)[:6]) for e in top5_accuracies]))
print('**********')
print('TOP1 mean incremental accuracy = '+str(np.mean(np.array(top1_accuracies[1:]))))
print('TOP5 mean incremental accuracy = '+str(np.mean(np.array(top5_accuracies[1:]))))

