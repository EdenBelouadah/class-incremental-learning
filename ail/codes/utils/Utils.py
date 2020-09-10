import numpy as np, random

random.seed(0)
np.random.seed(0)

import sys, re

class DataUtils():
    def __init__(self):
        return


    def print_parameters(self, cp):
        for key in cp.keys():
            print(key + ' = ' + str(cp[key]))


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

