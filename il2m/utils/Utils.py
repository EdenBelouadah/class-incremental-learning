import sys, numpy, random, re, os
# from MyStaticSamplers import *
import numpy as np
from tqdm import tqdm


class DataUtils():
    def __init__(self):
        return
        
    def accuracy(self, output, target, topk=(1,)):
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

    def get_dataset_mean_std(self, normalization_dataset_name, datasets_mean_std_file_path):
        datasets_mean_std_file = open(datasets_mean_std_file_path, 'r').readlines()
        for line in datasets_mean_std_file:
            line = line.strip().split(':')
            dataset_name, dataset_stat  = line[0], line[1]
            if dataset_name == normalization_dataset_name:
                dataset_stat = dataset_stat.split(';')
                dataset_mean = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[0])]
                dataset_std = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[1])]
                return dataset_mean, dataset_std
        print('Invalid normalization dataset name')
        sys.exit(-1)


    def from_str_to_list(self, string, type):
        list = []
        params = string.split(',')
        for p in params:
            if type == 'int':
                list.append(int(p.strip()))
            elif type == 'float':
                list.append(float(p.strip()))
            elif type == 'str':
                list.append(str(p.strip()))
        return list







