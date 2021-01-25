#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_pytorch import *
from Utils import DataUtils
import AverageMeter as AverageMeter
utils = DataUtils()

#code for baseline1 : FT+standard distillation
def train_eval(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iteration, \
            T, beta, \
            fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features

    top = min(5, tg_model.fc.out_features)

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
            else: #cross entropy + distillation
                ref_outputs = ref_model(inputs)
                loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:,:num_old_classes]/T, dim=1), \
                    F.softmax(ref_outputs.detach()/T, dim=1)) * T * T * beta * num_old_classes
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss = loss1 + loss2
            loss.backward()
            tg_optimizer.step()

        #eval
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

    return tg_model



def train_eval_lwf(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iteration, \
            T, beta, \
            fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        print('old = ' + str(num_old_classes))
    top = min(5, tg_model.fc.out_features)

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
            # print(outputs.shape)
            # print(targets.shape)
            if iteration == start_iteration:
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else: #cross entropy + distillation
                ref_outputs = ref_model(inputs)
                loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:,:num_old_classes]/T, dim=1), \
                    F.softmax(ref_outputs.detach()/T, dim=1)) * T * T * beta * num_old_classes
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss = loss1 + loss2
            loss.backward()
            tg_optimizer.step()

        #eval
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

    return tg_model



def train_eval_lwf2(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, iter_size,  \
            iteration, start_iteration, \
            T, beta, \
            fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        print('old = ' + str(num_old_classes))
    top = min(5, tg_model.fc.out_features)

    for epoch in range(epochs):
        #train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        # tg_lr_scheduler.step()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)
            # print(outputs.shape)
            # print(targets.shape)
            if iteration == start_iteration:
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else: #cross entropy + distillation
                ref_outputs = ref_model(inputs)
                loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:,:num_old_classes]/T, dim=1), \
                    F.softmax(ref_outputs.detach()/T, dim=1)) * T * T * beta * num_old_classes
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss = loss1 + loss2

            loss.data /= iter_size
            loss.backward()
            if (batch_idx + 1) % iter_size == 0:
                tg_optimizer.step()

        tg_lr_scheduler.step(loss.cpu().data.numpy())


        #eval
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

    return tg_model




def train_eval_lwf3(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, iter_size,  \
            iteration, start_iteration, \
            T, beta, \
            fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        print('old = ' + str(num_old_classes))
    top = min(5, tg_model.fc.out_features)

    for epoch in range(epochs):
        #train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        if iteration == start_iteration:
            tg_lr_scheduler.step()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)
            # print(outputs.shape)
            # print(targets.shape)
            if iteration == start_iteration:
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else: #cross entropy + distillation
                ref_outputs = ref_model(inputs)
                loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:,:num_old_classes]/T, dim=1), \
                    F.softmax(ref_outputs.detach()/T, dim=1)) * T * T * beta * num_old_classes
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss = loss1 + loss2

            loss.data /= iter_size
            loss.backward()
            if (batch_idx + 1) % iter_size == 0:
                tg_optimizer.step()

        if iteration > start_iteration:
            tg_lr_scheduler.step(loss.cpu().data.numpy())


        #eval
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

    return tg_model

