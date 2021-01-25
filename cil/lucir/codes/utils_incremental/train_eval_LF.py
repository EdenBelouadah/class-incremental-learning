#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from utils_pytorch import *
from Utils import DataUtils
import AverageMeter as AverageMeter
utils = DataUtils()


cur_features = []
ref_features = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def train_eval_LF(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iteration, \
            lamda, \
            fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    top = min(5, tg_model.fc.out_features)


    if iteration > start_iteration:
        ref_model.eval()
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
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
                loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
                    torch.ones(inputs.shape[0]).to(device)) * lamda
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
        handle_ref_features.remove()
        handle_cur_features.remove() 
    return tg_model
