#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Adrian Popescu"
__version__ = "1.0.1"
__maintainer__ = "Adrian Popescu"
__email__ = "adrian.popescu@cea.fr"
__status__ = "Prototype"

""" 
Script that gets a flat representation of features as input and computes
their L2-normalized versions in libsvm format as output

"""

import sys
import os
import re
from os import listdir
from os.path import isfile, isdir, join
import math
import time
import numpy as np

''' USAGE:

python ./code/feat_L2_normalization.py ./output/features_batch1/features +1 ./output/features_batch1/features_L2

'''

# ARGUMENTS
init_feats = sys.argv[1] # INPUT - dir with the initial features
label = sys.argv[2] # PARAM - label for the libsvm normalized features. Default to +1
norm_feats = sys.argv[3] # OUTPUT - L2 normalized version of the features in libsvm format

def normalize(init_feats, label, norm_feats):
   if not os.path.isdir(norm_feats):
      os.mkdir(norm_feats)
   # get the list of initial feature files
   init_list =[ f for f in listdir(init_feats) if isfile(join(init_feats,f)) ]
   for inf in init_list:
      # check if the normalized file exists in order not to create it multiple times
      norm_file = norm_feats+"/"+inf
      if not os.path.exists(norm_file) or 1 == 1: # TODO
         print "creating L2-normalized file:",norm_file
         f_norm = open(norm_file, "w")
         init_file = init_feats+"/"+inf
         f_init = open(init_file)
         for fline in f_init:
            crt_norm = 0
            parts = fline.rstrip().split(" ")
            for pp in parts:
               crt_norm =  crt_norm + float(pp) * float(pp)
            crt_norm = math.sqrt(crt_norm)
            # create the L2-normalized feature in libsvm format
            crt_feat = label
            dim_cnt = 1 # counter for libsvm dimensions - needs to start at 1
            for pp in parts:
               crt_weight = float(pp)/crt_norm
               crt_feat =crt_feat+" "+str(dim_cnt)+":"+str(crt_weight)
               dim_cnt = dim_cnt+1
            f_norm.write(str(crt_feat)+"\n")            
         f_init.close()
         f_norm.close()
      else:
         print "exists norm file:",norm_file


""" MAIN """
if __name__ == '__main__':
   normalize(init_feats, label, norm_feats)
   
