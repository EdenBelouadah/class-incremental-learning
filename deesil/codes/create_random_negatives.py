#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Adrian Popescu"
__version__ = "1.0.1"
__maintainer__ = "Adrian Popescu"
__email__ = "adrian.popescu@cea.fr"
__status__ = "Prototype"

""" 
Script for creating negatives for the batches of the incremental algorithm.
A pseudo-random selection is performed in which features are selected using a fixed step
within the list of features
"""

import sys
import os
import unicodedata
import re
from os import listdir
from os.path import isfile, join
from glob import glob

syn_list = sys.argv[1] # INPUT list of synsets in icarl order
feat_dir = sys.argv[2] # INPUT dir with features per class
list_dir = sys.argv[3] # INPUT dir with lists of images per class
mem_size = int(sys.argv[4]) # PARAM - size of the memory that is allowed in DeeSIL
batch_size = int(sys.argv[5]) # PARAM - size of the batches used in evaluation
max_batch = int(sys.argv[6]) # PARAM - maximum batch number for the current dataset
neg_dir = sys.argv[7]+"_mem_"+str(mem_size) # OUTPUT - directory that will include the negative features and associated classes

""" USAGE:
python ./code/create_random_negatives.py ./data/batch1_classes.lst ./output/features_batch1/features_L2 ./data/lists_batch1 20000 100 1 ./output/random_negatives 
"""

def create_random_list_per_batch(syn_list, feat_dir, list_dir, mem_size, batch_size, max_batch, neg_dir):
   '''
   Creates lists of negatives selected randomly from all the classes that already exist in the recognition system.
   Lists includes negatives from the current batch of classes. 
   Negatives of each learned class itself will be filtered just before training the SVM models
   '''
   # create the negative directory if not existent
   if not os.path.isdir(neg_dir):
      os.mkdir(neg_dir)
   # go through the batches associated to the current dataset
   for cb in range(1,max_batch+1):
      crt_batch = cb * batch_size
      # create a subdirectory for the negative features and list of the current batch
      batch_subdir = neg_dir+"/batch_"+str(cb)
      if not os.path.exists(batch_subdir):
         os.mkdir(batch_subdir)
      per_class = int(mem_size/crt_batch)+1 # number of images to keep per class
      crt_rand_feat = batch_subdir+"/features"
      crt_rand_list = batch_subdir+"/list"
      if not os.path.exists(crt_rand_feat) or os.path.exists(crt_rand_list):    
         out_feat = open(crt_rand_feat, "w")
         out_list = open(crt_rand_list, "w")
         tmp_feat = [] # tmp list for storing features
         tmp_list = [] # tmp list to store syn corresponding to features
         pos = 0
         f_syn = open(syn_list)
         for syn in f_syn:
            syn = syn.rstrip()
            # condition that ensures that only classes up to the current batch are selected
            if pos < crt_batch:
               syn_feat  = feat_dir+"/"+syn
               feat_list = [] # list for tmp storage of features
               f_feat = open(syn_feat)
               for line in f_feat:
                  line = line.rstrip()
                  feat_list.append(line) 
               f_feat.close()
               # compute the step for the pseudo-random selection of features
               step = int(len(feat_list)/per_class)
               #print pos,syn,len(feat_list),step
               for sel in range(0,per_class):
                  crt_pos = sel * step
                  #print crt_pos
                  tmp_feat.append(feat_list[crt_pos])
                  tmp_list.append(syn)
            pos = pos + 1
         f_syn.close() 
         print len(feat_list),len(tmp_feat),len(tmp_list)
         # mix the negative features in order to have 1 per class iteratively
         total_kept = 0
         for cs in range(0,per_class):
            for pc in range(0, crt_batch):
               index = cs + pc*per_class
               #print index, cs, pc, per_class,len(tmp_feat)
               # keep features only as long as the memory size is not exceeed
               if total_kept < mem_size  :
                  #replace the leading "+" of the features with a "-"
                  tmp_feat[index] = tmp_feat[index].replace('+','-')
                  out_feat.write(tmp_feat[index]+"\n")
                  out_list.write(tmp_list[index]+"\n")
                  total_kept = total_kept + 1      
         out_feat.close()
         out_list.close()
         print "batch: ",crt_batch,"; kept:",total_kept
      else:
         print "rand feat and list exist:\n  ",crt_rand_feat,"\n  ",crt_rand_list


""" MAIN """
if __name__ == '__main__':
  create_random_list_per_batch(syn_list, feat_dir, list_dir, mem_size, batch_size, max_batch, neg_dir)
  
