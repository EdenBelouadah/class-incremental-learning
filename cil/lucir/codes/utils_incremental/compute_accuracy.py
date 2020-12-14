#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from utils_pytorch import *
import AverageMeter as AverageMeter
from Utils import DataUtils
utils = DataUtils()



def compute_accuracy(tg_model, tg_feature_model, class_means, evalloader, scale=None, print_info=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()

    num_classes = tg_model.fc.out_features

    correct = 0
    correct_icarl = 0
    correct_ncm = 0

    top1_correct = AverageMeter.AverageMeter()
    top5_correct = AverageMeter.AverageMeter()
    top1_correct_icarl = AverageMeter.AverageMeter()
    top5_correct_icarl = AverageMeter.AverageMeter()
    top1_correct_ncm = AverageMeter.AverageMeter()
    top5_correct_ncm = AverageMeter.AverageMeter()


    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, min(5, num_classes)))
            top1_correct.update(prec1.item(), inputs.size(0))
            top5_correct.update(prec5.item(), inputs.size(0))


            outputs_feature = np.squeeze(tg_feature_model(inputs))

            sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature.cpu(), 'sqeuclidean')
            score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()

            prec1, prec5 = utils.accuracy(score_icarl.data, targets, topk=(1, min(5, num_classes)))
            top1_correct_icarl.update(prec1.item(), inputs.size(0))
            top5_correct_icarl.update(prec5.item(), inputs.size(0))


            # Compute score for NCM
            sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature.cpu(), 'sqeuclidean')
            score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()

            prec1, prec5 = utils.accuracy(score_ncm.data, targets, topk=(1, min(5, num_classes)))
            top1_correct_ncm.update(prec1.item(), inputs.size(0))
            top5_correct_ncm.update(prec5.item(), inputs.size(0))


    if print_info:
        print("LUCIR-CNN  | acc@1 = {:.2f}%\tacc@5 = {:.2f}%".format(top1_correct.avg, top5_correct.avg))
        print("LUCIR-NCM  | acc@1 = {:.2f}%\tacc@5 = {:.2f}%".format(top1_correct_ncm.avg, top5_correct_ncm.avg))
        print("iCaRL      | acc@1 = {:.2f}%\tacc@5 = {:.2f}%".format(top1_correct_icarl.avg, top5_correct_icarl.avg))


    top1_cnn_acc, top5_cnn_acc = top1_correct.avg, top5_correct.avg
    top1_icarl_acc, top5_icarl_acc = top1_correct_icarl.avg, top5_correct_icarl.avg
    top1_ncm_acc, top5_ncm_acc = top1_correct_ncm.avg, top5_correct_ncm.avg

    return [top1_cnn_acc, top5_cnn_acc, top1_icarl_acc, top5_icarl_acc, top1_ncm_acc, top5_ncm_acc]



def compute_accuracy_lwf(tg_model, tg_feature_model, evalloader, scale=None, print_info=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()

    num_classes = tg_model.fc.out_features

    correct = 0

    top1_correct = AverageMeter.AverageMeter()
    top5_correct = AverageMeter.AverageMeter()


    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, min(5, num_classes)))
            top1_correct.update(prec1.item(), inputs.size(0))
            top5_correct.update(prec5.item(), inputs.size(0))

    if print_info:
        print("LUCIR-CNN  | acc@1 = {:.2f}%\tacc@5 = {:.2f}%".format(top1_correct.avg, top5_correct.avg))


    top1_cnn_acc, top5_cnn_acc = top1_correct.avg, top5_correct.avg

    return [top1_cnn_acc, top5_cnn_acc]
