from __future__ import division

import copy
import os
import torch as th
import socket
import sys
import time
import warnings
from datetime import timedelta
import numpy as np
import torch.cuda as tc
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import AverageMeter as AverageMeter
from configparser import ConfigParser
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models
from MyImageFolder import ImagesListFileFolder


def accuracy(output, target, topk=(1,)):
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


def get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path):
    import re
    datasets_mean_std_file = open(datasets_mean_std_file_path, 'r').readlines()
    for line in datasets_mean_std_file:
        line = line.strip().split(':')
        dataset_name = line[0]
        dataset_stat = line[1]
        if dataset_name == normalization_dataset_name:
            dataset_stat = dataset_stat.split(';')
            dataset_mean = [float (e) for e in re.findall(r'\d+\.\d+',dataset_stat[0])]
            dataset_std =  [float (e) for e in re.findall(r'\d+\.\d+',dataset_stat[1])]
            return dataset_mean, dataset_std
    print('Invalid normalization dataset name')
    sys.exit(-1)

def target_transform(target, P, b):
    return target - (P * (b - 1))


if len(sys.argv) != 2:  # We have to give 1 arg
    print('Arguments: general_config')
    sys.exit(-1)

if not os.path.exists(sys.argv[1]):
    print('No configuration file found in the specified path')
    sys.exit(-1)

# loading configuration file
cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]

# reading parameters
algo_name = cp['algo_name']  #full_no_freeze_fine_tuning_ilsvrc
gpu = int(cp['gpu'])
patience = int(cp['patience'])
num_workers = int(cp['num_workers'])
dataset_files_dir = cp['dataset_files_dir']
first_model_load_path = cp['first_model_load_path']
lr_decay = float(cp['lr_decay'])
lr = float(cp['lr'])
momentum = float(cp['momentum'])
weight_decay = float(cp['weight_decay'])
old_batch_size = int(cp['old_batch_size'])
new_batch_size = int(cp['new_batch_size'])
test_batch_size = int(cp['test_batch_size'])
iter_size = int(old_batch_size / new_batch_size)
starting_epoch = int(cp['starting_epoch'])
num_epochs = int(cp['num_epochs'])
normalization_dataset_name = cp['normalization_dataset_name']
first_batch_number = int(cp['first_batch_number'])
last_batch_number = int(cp['last_batch_number'])
models_save_dir = cp['models_save_dir']
K = int(cp['K'])
P = int(cp['P'])

datasets_mean_std_file_path = cp['datasets_mean_std_file_path']
saving_intermediate_models = cp['saving_intermediate_models'] == 'True'


if not os.path.exists(models_save_dir):
    os.makedirs(models_save_dir)

# catching warnings
with warnings.catch_warnings(record=True) as warn_list:
    # Data loading code
    dataset_mean, dataset_std = get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)

    print('normalization dataset name = ' + str(normalization_dataset_name))
    print('dataset mean = ' + str(dataset_mean))
    print('dataset std = ' + str(dataset_std))
    print('first batch number = ' + str(first_batch_number))
    print('last batch number = ' + str(last_batch_number))

    # Data loading code
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    #print parameters
    print("Number of workers = " + str(num_workers))
    print("Old Batch size = " + str(old_batch_size))
    print("New Batch size = " + str(new_batch_size))
    print("test Batch size = " + str(test_batch_size))
    print("Iter size = " + str(iter_size))
    print("Starting epoch = " + str(starting_epoch))
    print("Number of epochs = " + str(num_epochs))
    print("momentum = " + str(momentum))
    print("weight_decay = " + str(weight_decay))
    print("lr = " + str(lr))
    print("lr_decay = " + str(lr_decay))
    print("patience = " + str(patience))
    print("K = " + str(K))
    print("Running on " + str(socket.gethostname()) + " | gpu " + str(gpu))

    top_1_test_accuracies = []
    top_5_test_accuracies = []
    for b in range(first_batch_number, last_batch_number +1):
        print('*' * 110)
        print('*' * 51+'BATCH '+str(b)+' '+'*'*51)
        print('*' * 110)
        if b >= 2 :
            batch_algo_name = algo_name + '_b' + str(b)
            batch_models_save_dir = os.path.join(models_save_dir, batch_algo_name)

            if saving_intermediate_models == True :
                if not os.path.exists(batch_models_save_dir):
                    os.mkdir(batch_models_save_dir)
            new_train_file_path =os.path.join(dataset_files_dir, 'separated/train/batch' + str(b))


            batch_lr = lr / b
            if b == 2:
                model_load_path = first_model_load_path
            else:
                model_load_path = os.path.join(models_save_dir, algo_name+'_b'+str(b-1)+'.pt')


            print('New train data loaded from ' + new_train_file_path)

            batch_models_save_dir = os.path.join(models_save_dir, batch_algo_name)
            if saving_intermediate_models == True:
                if not os.path.exists(batch_models_save_dir):
                    os.mkdir(batch_models_save_dir)


            new_train_dataset = ImagesListFileFolder(
                new_train_file_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize, ]),
                target_transform = lambda x : target_transform(x, P, b)
            )

            train_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=new_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False)

            new_classes_number = len(new_train_dataset.classes)

            print("New classes number = " + str(new_classes_number))
            print("New Training-set size = " + str(len(new_train_dataset)))

            model_ft = models.resnet18(pretrained=False, num_classes=P)

            print('Loading saved model from ' + model_load_path)
            state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
            model_ft.load_state_dict(state['state_dict'])

            for param in model_ft.parameters():
                param.requires_grad = False

            model_ft.fc = nn.Linear(512, new_classes_number)

            top = min(5, new_classes_number)

            if tc.is_available():
                model_ft = model_ft.cuda(gpu)
            else:
                print("GPU not available")
                sys.exit(-1)

            # Define Loss and Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=batch_lr, momentum=momentum, weight_decay=weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=patience, factor=lr_decay)
            # scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=lr_decay)


            # Training
            print("-" * 20)
            print("Training...")
            starting_time = time.time()
            epoch = 0
            for epoch in range(num_epochs):
                model_ft.train()
                # scheduler.step()
                running_loss = 0.0
                nb_batches = 0
                # zero the parameter gradients
                optimizer_ft.zero_grad()
                for i, data in enumerate(train_loader, 0):
                    nb_batches += 1
                    # get the data
                    inputs, labels = data

                    if tc.is_available():
                        inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

                    # wrap it in Variable
                    inputs, labels = Variable(inputs), Variable(labels)

                    # forward + backward + optimize
                    outputs = model_ft(inputs)
                    loss = criterion(outputs, labels)

                    # loss.data[0] /= iter_size
                    # loss.backward()
                    # running_loss += loss.data.cpu().numpy()[0]
                    loss.data /= iter_size
                    loss.backward()
                    running_loss += loss.data.item()
                    if (i+1)%iter_size == 0:
                        optimizer_ft.step()
                        optimizer_ft.zero_grad()

                scheduler.step(loss.cpu().data.numpy())

                # Model evaluation
                model_ft.eval()

                current_elapsed_time = time.time() - starting_time
                print('{:03}/{:03} | {} | Train : loss = {:.4f}'.
                      format(epoch + 1, num_epochs, timedelta(seconds=round(current_elapsed_time)),
                             running_loss / nb_batches))



            #training finished
            print('Saving last model in ' + batch_models_save_dir + '.pt' + '...')
            state = {
                'epoch': epoch,
                'state_dict': model_ft.state_dict(),
                'optimizer': optimizer_ft.state_dict()
            }
            torch.save(state, batch_models_save_dir + '.pt')

        ###########################START VALIDATION

        print("-" * 30)
        print('Testing Started on batch number ' + str(b))
        print("-" * 30)

        top1 = AverageMeter.AverageMeter()
        top5 = AverageMeter.AverageMeter()
        full_outputs = None
        batch_test_loaders = []
        batch_models = []
        print('Loading models and paths...')
        for i in range(1, b + 1):
            # loading test data
            test_file_path = os.path.join(dataset_files_dir, 'separated', 'test', 'batch'+str(i))
            print('*** path ------> '+ test_file_path)

            test_dataset = ImagesListFileFolder(
                test_file_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]))

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False)

            batch_test_loaders.append(test_loader)

            # Model
            model = models.resnet18(pretrained=False, num_classes=P)
            batch_models.append(model)
            if i == 1:
                model_load_path = first_model_load_path
            else:
                model_load_path = os.path.join(models_save_dir, algo_name + '_b' + str(i))+ '.pt'

            print('*** model -----> '+ model_load_path)
            print('-----------------------')

            state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
            batch_models[i - 1].load_state_dict(state['state_dict'])
            batch_models[i - 1] = batch_models[i - 1].cuda(gpu)
            batch_models[i - 1].eval()
            del (test_loader)
            del (test_dataset)
            del (model)

        print(str(len(batch_test_loaders)) + ' test datasets loaded, ' + str(
            len(batch_models)) + ' models loaded')

        # validation
        for i in range(1, b + 1):
            for data in batch_test_loaders[i - 1]:
                inputs, labels = data
                if tc.is_available():
                    inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
                inputs = Variable(inputs)
                long_outputs = None
                for j in range(1, b + 1):
                    short_outputs = batch_models[j - 1](inputs)
                    if long_outputs is None:
                        long_outputs = short_outputs
                    else:
                        long_outputs = th.cat((long_outputs, short_outputs), 1)

                prec1, prec5 = accuracy(long_outputs.data, labels, topk=(1, min(5, P * b)))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

        print('Test : acc@1 = {}% ; acc@{} = {}%'.format(top1.avg, min(5, P * b), top5.avg))
        top_1_test_accuracies.append(top1.avg)
        top_5_test_accuracies.append(top5.avg)
        print('-----------------------')
        print('top1 acc so far : ' + str(top_1_test_accuracies))
        print('top5 acc so far : ' + str(top_5_test_accuracies))

        ###########################END VALIDATION

    print('TOP1 test accuracies = '+str([float(str(e)[:6]) for e in top_1_test_accuracies]))
    print('TOP1 mean incremental accuracy = '+str(np.mean(np.array(top_1_test_accuracies)[1:])))
    print('***************')
    print('TOP5 test accuracies = '+str([float(str(e)[:6]) for e in top_5_test_accuracies]))
    print('TOP5 mean incremental accuracy = '+str(np.mean(np.array(top_5_test_accuracies)[1:])))


# Print warnings (Possibly corrupt EXIF files):
if len(warn_list) > 0:
    print("\n" + str(len(warn_list)) + " Warnings\n")
    # for i in range(len(warn_list)):
    #     print("warning " + str(i) + ":")
    #     print(str(i)+":"+ str(warn_list[i].category) + ":\n     " + str(warn_list[i].message))
else:
    print('No warnings.')