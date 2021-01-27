#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Adrian Popescu"
__version__ = "1.0.1"
__maintainer__ = "Adrian Popescu"
__email__ = "adrian.popescu@cea.fr"
__status__ = "Prototype"

"""
Script that computes DeeSIL model predictions.
It takes a dir with the models and the test set features as entries
and produces sorted predictions for each test set image.  
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
model_dir = sys.argv[2] # INPUT directory in which the DeeSIL models are stored
test_feats = sys.argv[3] # INPUT - validation features file
pred_file = sys.argv[4] # OUTPUT - file with the DeeSIL predictions
                        # all predictions are stored in a single file and the split in batches 
                        # is done in the evaluation file


output_dir = os.path.dirname(pred_file)
if not os.path.exists(output_dir):
   os.mkdir(output_dir)


""" USAGE
python ./code/compute_predictions.py ./data/train_classes.lst ./output/models_full_C_1 ./output/features_val/features_L2/val_10.lst ./output/results/top_1000.txt

"""

def compute_predictions(syn_list, model_dir, test_feats, pred_file):
   ''' function that reads the models created for the current regularization parameter
       and computes the classification accuracy using a test set.
       the sorted prediction results are stored in an output file which
       is then given to an evaluation script
   '''
   if not os.path.exists(pred_file) or 1 == 1: # TODO 
      f_pred = open(pred_file, "w")
      # open the directory that contains the models for the tested regularization param
      syns = []
      f_list_syn = open(syn_list)
      for syn in f_list_syn:
         syn = syn.rstrip()
         syns.append(syn)
      f_list_syn.close()
      print "synsets:",len(syns)
      # open the models for the classes and put store weights and biases in dedicated lists
      weights_list = []  
      biases_list = []
      for syn in range(0,len(syns)):
         #print syn 
         line_cnt = 0 # counter to get the weights and bias lines
         target_model = model_dir+"/"+str(syn)+".model"
         f_model = open(target_model)
         for line in f_model:
            line = line.rstrip()
            # get the weights line
            if line_cnt == 0:
               parts = line.split(" ")
               parts_float = [] # tmp list to store the weights
               for pp in parts:
                  parts_float.append(float(pp))
               # add the weights to the dedicated list
               weights_list.append(parts_float)
            elif line_cnt == 1:
               biases_list.append(float(line))
            line_cnt = line_cnt + 1
         f_model.close() 
      print "list sizes - weights:",len(weights_list),"; biases:",len(biases_list)
      # open the test features and compute their prediction scores 
      # for the models obtained with the current regularizer value
      f_test_feat = open(test_feats)
      # counters for correct and total number of predictions 
      correct = 0
      total = 0
      for vline in f_test_feat:
         vparts = vline.split(" ")
         # create a tmp list for the current feature vector
         crt_feat = []
         # start at 1 to discard the libsvm label)
         for dim in range(1,len(vparts)):
            dim_parts = vparts[dim].split(":")
            crt_feat.append(float(dim_parts[1]))
         # put the predicition results into a dictionary which will be then sorted
         pred_dict = {}
         for cls_cnt in range(0, len(weights_list)):
            cls_score = np.dot(crt_feat, weights_list[cls_cnt]) + biases_list[cls_cnt]
            #cls_score = biases_list[cls_cnt]
            #print cls_cnt,syns[cls_cnt]
            pred_dict[cls_cnt] = cls_score
         pred_line = ""
         for crt_cls, crt_score in sorted(pred_dict.iteritems(), key=lambda (k,v): (v,k), reverse=True):
            pred_line = pred_line+" "+str(crt_cls)+":"+str(crt_score) 
         pred_line = pred_line.lstrip()
         f_pred.write(pred_line+"\n")
      f_test_feat.close()
      f_pred.close()   
   else:
      print "exists predictions file:",pred_file
  
""" MAIN """
if __name__ == '__main__':
   compute_predictions(syn_list, model_dir, test_feats, pred_file)
   
   

