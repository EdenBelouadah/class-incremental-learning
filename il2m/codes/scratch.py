from __future__ import division
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.cuda as tc
import torch.utils.data.distributed
from configparser import ConfigParser
import sys, os, warnings, time
import numpy as np
from datetime import timedelta
import AverageMeter as AverageMeter
from MyImageFolder import ImagesListFileFolder
import socket
from Utils import DataUtils


if len(sys.argv) != 2:
    print('You must specify the configuration file path')
    sys.exit(-1)

if not os.path.exists(sys.argv[1]):
    print('No configuration file found in the specified path')
    sys.exit(-1)

# loading configuration file
cp = ConfigParser()
cp.read(sys.argv[1])
cp = cp[os.path.basename(__file__)]

# reading parameters
num_workers = int(cp['num_workers'])
num_epochs = int(cp['num_epochs'])
gpu = int(cp['gpu'])
train_file_path = cp['train_file_path']
val_file_path = cp['val_file_path']
saving_new_model = cp['saving_new_model'] == 'True'
lr_decay = float(cp['lr_decay'])
lr = float(cp['lr'])
momentum = float(cp['momentum'])
weight_decay = float(cp['weight_decay'])
patience = int(cp['patience'])
normalization_dataset_name = cp['normalization_dataset_name']
old_batch_size = int(cp['old_batch_size'])
new_batch_size = int(cp['new_batch_size'])
val_batch_size = int(cp['val_batch_size'])
iter_size = int(old_batch_size / new_batch_size)
starting_epoch = int(cp['starting_epoch'])
algo_name = cp['algo_name']
intermediate_models_save_dir = os.path.join(cp['intermediate_models_save_dir'], algo_name)
saving_intermediate_models = cp['saving_intermediate_models']  == 'True'
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']

if saving_intermediate_models:
    if not os.path.exists(intermediate_models_save_dir):
        os.makedirs(intermediate_models_save_dir)

print('Loading train images from '+train_file_path)
print('Loading val images from '+val_file_path)
print('Dataset name for normalization = '+normalization_dataset_name)


#catching warnings
with warnings.catch_warnings(record=True) as warn_list:
    utils = DataUtils()
    dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)

    print('dataset mean = '+str(dataset_mean))
    print('dataset std = '+str(dataset_std))

    # Data loading code
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)


    train_dataset = ImagesListFileFolder(
        train_file_path,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))



    val_dataset = ImagesListFileFolder(
        val_file_path, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    # print different parameters
    num_classes = len(train_dataset.classes)
    print("Number of workers = " + str(num_workers))
    print("Old Batch size = " + str(old_batch_size))
    print("New Batch size = " + str(new_batch_size))
    print("Val Batch size = " + str(val_batch_size))
    print("Iter size = " + str(iter_size))
    print("Starting epoch = " + str(starting_epoch))
    print("Number of epochs = " + str(num_epochs))
    print("lr = " + str(lr))
    print("momentum = " + str(momentum))
    print("weight_decay = " + str(weight_decay))
    print("lr_decay = " + str(lr_decay))
    print("patience = " + str(patience))
    print("-" * 20)
    print("Number of classes = " + str(num_classes))
    print("Training-set size = " + str(len(train_dataset)))
    print("Validation-set size = " + str(len(val_dataset)))


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=new_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    print("Number of batches in Training-set = " + str(len(train_loader)))
    print("Number of batches in Validation-set = " + str(len(val_loader)))


    #Creating model
    print('Creating ResNet-18 model...')
    model = models.resnet18(pretrained=False, num_classes=num_classes)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)


    if tc.is_available():
        print("Running on " +str(socket.gethostname())+" | gpu "+str(gpu))
        model = model.cuda(gpu)
    else:
        print("GPU not available")
        sys.exit(-1)



    # Training
    print("-" * 20)
    print("Training...")
    starting_time = time.time()


    try:
        for epoch in range(starting_epoch, starting_epoch + num_epochs):
            top1 = AverageMeter.AverageMeter()
            top5 = AverageMeter.AverageMeter()
            # scheduler.step()
            model.train()
            running_loss = 0.0
            nb_batches = 0
            # zero the parameter gradients
            optimizer.zero_grad()
            for i, data in enumerate(train_loader, 0):
                nb_batches += 1
                # get the data
                inputs, labels = data

                if tc.is_available():
                    inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

                # wrap it in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.data /= iter_size
                loss.backward()
                running_loss += loss.data.item()
                if (i+1)%iter_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler.step(loss.cpu().data.numpy())

            #Model evaluation
            model.eval()

            for data in val_loader:
                inputs, labels = data
                if tc.is_available():
                    inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
                outputs = model(Variable(inputs))
                prec1, prec5 = utils.accuracy(outputs.data, labels, topk=(1, 5))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

            current_elapsed_time = time.time() - starting_time
            print('{:03}/{:03} | {} | Train : loss = {:.4f} | Val : acc@1 = {}% ; acc@5 = {}%'.
                  format(epoch + 1, starting_epoch + num_epochs, timedelta(seconds=round(current_elapsed_time)), running_loss / nb_batches, top1.avg , top5.avg))

            # Saving model
            if saving_intermediate_models:
                state = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, intermediate_models_save_dir + '/' + str(epoch) +'.pt')


    except KeyboardInterrupt:
        print('Keyboard Interruption')

    finally:
        print('Finished Training, elapsed training time : {}'.format(timedelta(seconds=round(time.time() - starting_time))))
        if saving_new_model :
            models_save_dir = os.path.join(cp['models_save_dir'], algo_name)
            if not os.path.exists(models_save_dir):
                os.makedirs(models_save_dir)

            print('Saving model in '+models_save_dir+'.pt'+'...')
            state ={
                'state_dict' : model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
            }
            torch.save(state,models_save_dir+'.pt')


        #Print warnings
        if len(warn_list) > 0:
            print("\n"+str(len(warn_list))+" Warnings\n")
            # for i in range(len(warn_list)):
            #     print("warning " + str(i) + ":")
            #     print(str(i)+":"+ str(warn_list[i].category) + ":\n     " + str(warn_list[i].message))
        else:
            print('No warnings.')
