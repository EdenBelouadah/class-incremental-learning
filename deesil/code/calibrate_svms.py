#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Adrian Popescu"
__version__ = "1.0.1"
__maintainer__ = "Adrian Popescu"
__email__ = "adrian.popescu@cea.fr"
__status__ = "Prototype"

"""
Script that calibrates the linear SVM models for DeeSIL.
It takes as input training and validation features for the first batch of the dataset.
and runs a grid search for the optimal regularization parameter of the SVM.
This parameter is then used for all new classes that are included in DeeSIL.  
"""

import sys
import os
import unicodedata
import re
import math
import time
# import sklearn
from os import listdir
from os.path import isfile, join
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np

""" list of arguments for the script """
syn_list = sys.argv[1] # INPUT list of ordered icarl synsets
feat_dir = sys.argv[2] # INPUT directory with L2-normalized positive features
neg_feats = sys.argv[3] # INPUT - file with the negatives selected for the first batch 
neg_list = sys.argv[4] # INPUT - list of class labels associated to the negatives 
tmp_dir = sys.argv[5] # INTERM dir for temporary files created during learning
regul = float(sys.argv[6]) # PARAM - current regularization parameter
root_out_dir = sys.argv[7] # OUTPUT root for the output directory in which the models are stored for each tested regularization parameter
val_feats = sys.argv[8] # INPUT - validation features
val_list = sys.argv[9] # INPUT - validation list with labels

""" USAGE
python ./code/calibrate_svms.py ./data/batch1_classes.lst ./output/features_batch1/features_L2 ./output/random_negatives/batch_1/features ./output/random_negatives/batch_1/list ./output/tmp 100 ./output/models_calibration ./output/features_batch1/features_L2/val.txt ./data/lists_batch1/val.txt

"""

def create_calibration_models(syn_list, feat_dir, neg_feats, neg_list, tmp_dir, regul, root_out_dir):
   ''' function that creates linear models for ImageNet synsets with the set of parameters given to the script'''
   # create the root dir if not existing 
   if not os.path.isdir(root_out_dir):
      os.mkdir(root_out_dir)
   # create the dir for the current regularization parameter if not existing
   out_dir = root_out_dir+"/C_"+str(regul)
   if not os.path.isdir(out_dir):
      os.mkdir(out_dir)
   # create the tmp dir is not existing
   if not os.path.isdir(tmp_dir):
      os.mkdir(tmp_dir)
   # set up the solver and its parameters
   clf = LinearSVC(penalty='l2', dual=True, tol=0.0001, C=regul, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=123)
   batch_size = 100
   syn_cnt = 0
   syns = []
   f_list_syn = open(syn_list)
   for syn in f_list_syn:
      syn = syn.rstrip()
      syns.append(syn)
   f_list_syn.close()
   # open the filtered confusion matrix to create the models
   for syn in syns:
      # only create the model if not already created
      target_model = out_dir+"/"+syn+".model"
      if not os.path.exists(target_model):
         print "creating model file:",target_model
         f_target = open(target_model, "w")
         # create a tmp file for the features associated to the current target concept
         target_tmp = tmp_dir+"/"+syn+"_"+str(regul)
         f_tmp = open(target_tmp, "w")
         # write the positive examples to the temporary file
         pos_cnt = 0 # get the number of positive examples - necessary for the selection of negatives              
         pos_file = feat_dir+"/"+syn;
         f_pos = open(pos_file, "r")
         for pos_line in f_pos:
            f_tmp.write(pos_line)
            pos_cnt = pos_cnt + 1
         f_pos.close()        
         # open the list of negatives to store the synsets associated to each position
         tmp_position = []
         f_neg_list = open(neg_list)
         for nline in f_neg_list:
            nline = nline.rstrip()
            tmp_position.append(int(nline))
         f_neg_list.close()
         # case when negatives are not adapted and simply taken from a large negative file
         neg_cnt = 0 # counter for negatives
         same = 0
         f_neg = open(neg_feats, "r")               
         for neg_line in f_neg:                 
            if not tmp_position[neg_cnt] == int(syn):   
               f_tmp.write(neg_line)
            else:
               same = same + 1
            neg_cnt = neg_cnt+1
         f_neg.close()            
         f_tmp.close()
         print "removed same class:",same
         # load training data to a sparse matrix and perform the training
         X, y = load_svmlight_file(target_tmp)
         clf.fit(X,y)
         model_weights = clf.coef_
         bias = clf.intercept_
         out_weights = ""
         for it in range(0, model_weights.size):
            #print model_weights.item(it)            
            out_weights = out_weights+" "+str(model_weights.item(it))
         out_weights = out_weights.lstrip()
         out_bias = str(bias.item(0))
         f_target.write(out_weights+"\n") 
         f_target.write(out_bias+"\n")
         # before continuing, remove the temporary file
         os.remove(target_tmp)
         f_target.close()
      else:
         print "model exists for class:",syn
      syn_cnt = syn_cnt + 1

def compute_accuracy(syn_list, tmp_dir, regul, root_out_dir, val_feats, val_list):
   ''' function that reads the models created for the current regularization parameter
       and computes the classification accuracy using a validation set     
   '''
   # read the labels of the validation set and put them in a list
   val_labels = []
   f_val_list = open(val_list)
   for line in f_val_list:
      line = line.rstrip()
      # case when the label is on the second column of the list (Caffe style list)
      if ' ' in line:
         parts = line.split(" ")
         val_labels.append(int(parts[1]))
      else:
         val_labels.append(int(line))
   f_val_list.close()
   print "labels:",len(val_labels)
   # open the directory that contains the models for the tested regularization param
   regul_dir = root_out_dir+"/C_"+str(regul)
   syns = []
   class_dict = {} # dictionary to store class positions and names
   cls_pos = 0
   f_list_syn = open(syn_list)
   for syn in f_list_syn:
      syn = syn.rstrip()
      syns.append(syn)
      class_dict[cls_pos] = int(syn)
      cls_pos = cls_pos + 1 
   f_list_syn.close()
   # open the models for the classes and put store weights and biases in dedicated lists
   weights_list = []  
   biases_list = []
   for syn in syns:
      line_cnt = 0 # counter to get the weights and bias lines
      target_model = regul_dir+"/"+syn+".model"
      f_model = open(target_model)
      for line in f_model:
         line = line.rstrip()
         # get the weights line
         if line_cnt == 0:
             parts = line.split(" ")
             parts_float = [] # tmp list to store the weights
             for pp in parts:
                parts_float.append(float(pp))
                #print pp
             # add the weights to the dedicated list
             weights_list.append(parts_float)
         elif line_cnt == 1:
            biases_list.append(float(line))
         line_cnt = line_cnt + 1
      f_model.close() 
   print "list sizes - weights:",len(weights_list),"; biases:",len(biases_list)
   # open the validation features and compute their prediction scores 
   # for the models obtained with the current regularizer value
   val_cnt = 0 # counter for validation features - useful to get the ground truth label
   f_val_feat = open(val_feats)
   # counters for correct and total number of predictions 
   correct = 0
   total = 0
   for vline in f_val_feat:
      vparts = vline.split(" ")
      # create a tmp list for the current feature vector
      crt_feat = []
      # start at 1 to discard the libsvm label)
      for dim in range(1,len(vparts)):
         dim_parts = vparts[dim].split(":")
         crt_feat.append(float(dim_parts[1]))
      # compute the prediction scores for the models and retain the one with the highest score
      max_class = -1 # dummy initial value for the class 
      max_score = -1000 # dummy initial value for the max prediction score
      for cls_cnt in range(0, len(weights_list)):
         cls_score = np.dot(crt_feat, weights_list[cls_cnt]) + biases_list[cls_cnt]
         #cls_score = biases_list[cls_cnt]
         if cls_score > max_score:
            max_score = cls_score
            max_class = class_dict[cls_cnt]
      # check if the predicted class is the same as the ground truth class 
      if max_class == val_labels[val_cnt]:
         correct = correct+1
      total = total + 1
      #print correct,total
      val_cnt = val_cnt + 1
   f_val_feat.close()
   # compute the accuracy for the current regularization parameter
   accur = float(correct)/float(total)
   print "C = "+str(regul)+"; accuracy = "+str(accur)
   
  
""" MAIN """
if __name__ == '__main__':
   # create the models for the current regularization parameter
   create_calibration_models(syn_list, feat_dir, neg_feats, neg_list, tmp_dir, regul, root_out_dir)
   #compute the accuracy for the current regularization parameter
   compute_accuracy(syn_list, tmp_dir, regul, root_out_dir, val_feats, val_list)
   
   
