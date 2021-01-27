#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Adrian Popescu"
__version__ = "1.0.1"
__maintainer__ = "Adrian Popescu"
__email__ = "adrian.popescu@cea.fr"
__status__ = "Prototype"

"""
Script that computes linear SVM models for icarl concepts modeled in Semfeat.
The features, normalized L2 in libsvm format are readily available to this program.
The learning is done using negatives that are selected randomly and in a balanced manner
from the concepts that are already known to the system (including the current batch)
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

""" list of arguments for the script """
batch_size = int(sys.argv[1])
syn_list = sys.argv[2] # INPUT list of ordered icarl synsets
feat_dir = sys.argv[3] # INPUT directory with L2-normalized cnn features
regul = int(sys.argv[4]) # INPUT ratio for the number of negatives vs positives for each synset
root_neg_dir = sys.argv[5] # INPUT root dir in which the negative features and lists for each batch are stored
out_dir = sys.argv[6]+"_C_"+str(regul) # OUTPUT stub for the output directory in which the models are stored
min_pos = int(sys.argv[7]) # min position in the list to be considered - useful for parallelization
max_pos = int(sys.argv[8]) # max position in the list to be considered - useful for parallelization



""" USAGE
python ./code/train_svms.py ./data/train_classes.lst ./output/features_train/features_L2 1 ./output/random_negatives_full ./output/tmp ./output/models_full 0 2

python ./code/train_svms_deesil.py 
./data/train_classes.lst 
./output/features_train/features_L2 
1 
./output/random_negatives_full 
./output/tmp 
./output/models_full 
0 
2

"""

def create_model(syn_list, feat_dir, regul, root_neg_dir, out_dir, min_pos, max_pos):
   ''' function that creates linear models for ImageNet synsets with the set of parameters given to the script'''

   if not os.path.isdir(out_dir):
      os.mkdir(out_dir)

   # create the tmp dir is not existing
   tmp_dir = out_dir+"/tmp_"+str(min_pos)
   if not os.path.isdir(tmp_dir):
      os.mkdir(tmp_dir)

   # set up the solver and its parameters
   clf = LinearSVC(penalty='l2', dual=True, tol=0.0001, C=regul, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=123)
   syn_cnt = 0
   #syns = [f for f in listdir(feat_dir) if isfile(join(feat_dir, f))]
   #syns =sorted(syns)
   syns = []
   f_list_syn = open(syn_list)
   for syn in f_list_syn:
      syn = syn.rstrip()
      syns.append(syn)
   f_list_syn.close()
   # open the filtered confusion matrix to create the models
   for syn in syns:
      # perform the learning only for a part of the synset list
      if syn_cnt >= min_pos and syn_cnt < max_pos:
         # only create the model if not already created
         target_model = out_dir+"/"+syn+".model"
         if not os.path.exists(target_model):
            f_target = open(target_model, "w")
            # create a tmp file for the features associated to the current target concept
            target_tmp = tmp_dir+"/"+syn+"_"+str(regul)
            f_tmp = open(target_tmp, "w")
            #print "tmp file:",target_tmp
            # write the positive examples to the temporary file
            pos_cnt = 0 # get the number of positive examples - necessary for the selection of negatives              
            pos_file = feat_dir+"/"+syn;
            f_pos = open(pos_file, "r")
            for pos_line in f_pos:
              f_tmp.write(pos_line)
              pos_cnt = pos_cnt + 1
            f_pos.close()
            #print line_parts[0]," positives ",pos_cnt         
            # open the list of negatives to store the synsets associated to each position
            tmp_position = []
            # get the suffix of the current synset among the 10 icarl batch
            crt_suffix = (int(syn_cnt/batch_size)+1)
            print "syn cnt:",syn_cnt,"; crt suffix:",crt_suffix
            neg_list = root_neg_dir+"/batch_"+str(crt_suffix)+"/list"
            f_neg_list = open(neg_list)
            for nline in f_neg_list:
               nline = nline.rstrip()
               tmp_position.append(nline)
            f_neg_list.close()
            # case when negatives are not adapted and simply taken from a large negative file
            neg_cnt = 0 # counter for negatives
            same = 0
            big_neg = root_neg_dir+"/batch_"+str(crt_suffix)+"/features"
            f_neg = open(big_neg, "r")               
            for neg_line in f_neg:
               if not tmp_position[neg_cnt] == syn:   
                  f_tmp.write(neg_line)
               else:
                  #print "same syn: ",same, syn
                  same = same + 1

               neg_cnt = neg_cnt+1
            f_neg.close()            
            f_tmp.close()
            print "negative counts:",neg_cnt,"; excluded:",same
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
            #print "WEIGHTS: " + ' '.join(str(f) for f in model_weights)
            #print "BIAS: " + str(bias)
            # before continuing, remove the temporary file
            os.remove(target_tmp) # TO DO - remove comment here
            f_target.close()
         else:
             print "exists model:",target_model
      syn_cnt = syn_cnt + 1

""" MAIN """
if __name__ == '__main__':


   # create the SVM models for the current configuration
   create_model(syn_list, feat_dir, regul, root_neg_dir, out_dir, min_pos, max_pos)

