#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Adrian Popescu"
__version__ = "1.0.1"
__maintainer__ = "Adrian Popescu"
__email__ = "adrian.popescu@cea.fr"
__status__ = "Prototype"

""" 
DeeSIL evaluation script that computes accuracy@1 and accuracy@5 for the test seet.
It takes the synsets as ordered in the training list, the validation list and 
the results files and inputs and produces the eval scores
"""

import sys
import os
import unicodedata
import re
from os import listdir
from os.path import isfile, join
from glob import glob

syn_list = sys.argv[1] # INPUT list of classes as they appear in the incremental batches
test_list = sys.argv[2] # INPUT validation list in the order of extraction
batch_size = int(sys.argv[3]) # INPUT size of the incremental batches
res_file = sys.argv[4] # INPUT results file for SVM based classification

"""
Usage:
python ./code/eval.py ./data/train_classes.lst ./data/list_val/val.lst 100 ./output/results/top_1000.txt
"""


def compute_accuracy(syn_list, test_list, batch_size, res_file):
   """
   compute accuracy for batches of synsets whose size increases from 100 to 1000 by step of 100
   for each batch, we evaluate only the examples of images that belong to already seen synsets
   """
   # get the total number of synsets
   cnt_syn = 0
   f_rand = open(syn_list)
   for syn in f_rand:
      cnt_syn = cnt_syn + 1
   f_rand.close()
   max_batch = cnt_syn + batch_size
   sum_accu1 = 0 # var to sum the accuracy@1 over all incremental batches
   sum_accu5 = 0 # var to sum the accuracy@1 over all incremental batches
   increm_batches = 0 # counter for the number of incremental batches
   for step in range(batch_size,max_batch,batch_size):
      sum1 = 0
      sum5 = 0
      # create a dictionary that links the syn positions from the randomized lists to the syn positions from the original test file
      rand_dict = {}
      pos = 0
      f_rand = open(syn_list)
      for syn in f_rand:
         if pos < step:
            syn = syn.rstrip()
            rand_dict[syn] = syn
            #rand_dict.append(syn)
         pos = pos + 1
      f_rand.close()
      #open the test list and select elements that are in the current batches
      # to do this, check if their associated class is in the class dictionary
      gt_dict = {} # save the position of the image and its class number in the randomized set
      f_test_list = open(test_list)
      test_pos = 0
      for line in f_test_list:
         # case when the class name is on the second column
         if ' ' in line:
            tmp_parts = line.rstrip().split(" ")
            gt_syn = tmp_parts[1]
         # case when only the class name is written on the line          
         else:
            gt_syn = line.rstrip()
         if gt_syn in rand_dict:
            gt_dict[test_pos] = int(rand_dict[gt_syn])
         test_pos = test_pos + 1
      f_test_list.close()
      res_pos = 0
      bingo = 0
      #print "step:",step,"; concepts:",len(rand_dict),"; test images:",len(gt_dict)
      f_res = open(res_file)
      for crt_res in f_res:
         if res_pos in gt_dict:
            parts = crt_res.split(" ")
            #print res_pos
            # keep only the top 5 results that are relevant for the current step
            top_list = []
            cnt = 0
            while cnt < len(parts) and len(top_list) < 5:
               con_parts = parts[cnt].split(":")
               # test if the predicted class is one of the known batches to keep it
               if int(con_parts[0]) < step:
                  top_list.append(int(con_parts[0]))
                  #print " ",con_parts
               cnt = cnt + 1
            bingo = bingo + 1
            #print gt_dict[res_pos],top_list,top_list[0]
            if gt_dict[res_pos] == top_list[0]:
               sum1 = sum1 + 1
            for tl in top_list:
               if gt_dict[res_pos] == tl:
                  sum5 = sum5+1
         res_pos = res_pos + 1
      #print "valid set: ",len(gt_dict),"; found:",bingo
      accu1 = float(sum1)/bingo
      accu5 = float(sum5)/bingo
      # compute the average accuracy starting with the second batch 
      # i.e the first incremental one as defined in "End-to-end incremental learning"
      if step > batch_size: # TODO remove the first batch from the computation since non-incremental
         sum_accu1 = sum_accu1 + accu1
         sum_accu5 = sum_accu5 + accu5
         increm_batches = increm_batches+1
      print step,"  accu@1 = ",accu1,";  acc@5 = ",accu5
      #print "     raw:",sum1,sum5,bingo
      f_res.close()
   avg_accu1 = sum_accu1/increm_batches
   avg_accu5 = sum_accu5/increm_batches
   print "\naveraged accuracies for incremental batches" 
   print "   avg accu@1:",avg_accu1
   print "   avg accu@5:",avg_accu5

""" MAIN """
if __name__ == '__main__':
   compute_accuracy(syn_list, test_list, batch_size, res_file)

