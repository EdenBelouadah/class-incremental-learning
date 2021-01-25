#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from utils_pytorch import *
from Utils import DataUtils
import AverageMeter as AverageMeter
utils = DataUtils()

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs

def train_eval_MS(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iteration, \
            lw_ms, \
            fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    top = min(5, tg_model.fc.out_features)


    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        handle_old_scores_bs = tg_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
        handle_new_scores_bs = tg_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)
    for epoch in range(epochs):
        #train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        tg_lr_scheduler.step()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)
            if iteration == start_iteration:
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                ref_outputs = ref_model(inputs)
                ref_scores = ref_outputs.detach() / ref_model.fc.sigma.detach()
                loss1 = nn.MSELoss()(old_scores, ref_scores.detach()) * lw_ms * num_old_classes 
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss = loss1 + loss2
            loss.backward()
            tg_optimizer.step()

        # eval
        top1 = AverageMeter()
        top5 = AverageMeter()
        tg_model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, top))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

        print('{:03}/{:03} | Test ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(
            epoch + 1, epochs, len(testloader), top1.avg, top, top5.avg))


    if iteration > start_iteration:
        print("Removing register_forward_hook")
        handle_old_scores_bs.remove()
        handle_new_scores_bs.remove()
    return tg_model
