from __future__ import division
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.cuda as tc
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from configparser import ConfigParser
import sys, os, warnings, time
from datetime import timedelta
import AverageMeter as AverageMeter
import socket
from MyImageFolder import ImagesListFileFolder
import math
import numpy as np
import copy
from sklearn import preprocessing
from numpy.linalg import norm
from Utils import DataUtils
from Herding import StaticHerding

#####################################################################################

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
exemplars_dir = cp['exemplars_dir']
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']
dataset_files_path = cp['dataset_files_path']
first_model_load_path = cp['first_model_load_path']
saving_intermediate_models = cp['saving_intermediate_models'] == 'True'
step_size = int(cp['step_size'])
lr_decay = float(cp['lr_decay'])
lr = float(cp['lr'])
momentum = float(cp['momentum'])
weight_decay = float(cp['weight_decay'])
old_batch_size = int(cp['old_batch_size'])
new_batch_size = int(cp['new_batch_size'])
val_batch_size = int(cp['val_batch_size'])
exemplars_batch_size = int(cp['exemplars_batch_size'])
iter_size = int(old_batch_size / new_batch_size)
starting_epoch = int(cp['starting_epoch'])
num_epochs = int(cp['num_epochs'])
normalization_dataset_name = cp['normalization_dataset_name']
used_model = cp['used_model']
first_batch_number = int(cp['first_batch_number'])
last_batch_number = int(cp['last_batch_number'])
models_save_dir = cp['models_save_dir']
features_destination_dir = cp['features_destination_dir']
file_names_suffix = cp['file_names_suffix']
K = int(cp['K'])
P = int(cp['P'])

if not os.path.exists(models_save_dir):
    os.makedirs(models_save_dir)

# catching warnings
with warnings.catch_warnings(record=True) as warn_list:
    utils = DataUtils()
    herding = StaticHerding()

    # Data loading code
    dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    print('normalization dataset name = ' + str(normalization_dataset_name))
    print('dataset mean = ' + str(dataset_mean))
    print('dataset std = ' + str(dataset_std))

    #print parameters
    print("Number of workers = " + str(num_workers))
    print("Old Batch size = " + str(old_batch_size))
    print("New Batch size = " + str(new_batch_size))
    print("Val Batch size = " + str(val_batch_size))
    print("Iter size = " + str(iter_size))
    print("Starting epoch = " + str(starting_epoch))
    print("Number of epochs = " + str(num_epochs))
    print("momentum = " + str(momentum))
    print("weight_decay = " + str(weight_decay))
    print("Step size = " + str(step_size))
    print("lr_decay = " + str(lr_decay))
    print("patience = " + str(patience))
    print("K = " + str(K))
    print("First batch number = " + str(first_batch_number))
    print("Last batch number = " + str(last_batch_number))
    print("Running on " + str(socket.gethostname()) + " | gpu " + str(gpu))

    top1_val_accuracies = []
    top5_val_accuracies = []

    for b in range(first_batch_number, last_batch_number +1):
        starting_time = time.time()
        print('*' * 110)
        print('*' * 51 + 'BATCH ' + str(b) + ' ' + '*' * 51)
        print('*' * 110)

        if b == 1: #first non incremental batch
            val_file_path = os.path.join(dataset_files_path, 'accumulated/val/batch' + str(b))
            model_load_path = first_model_load_path
            new_train_file_path = os.path.join(dataset_files_path, 'batch1/train.lst')
            print('Train data loaded from ' + new_train_file_path)
            print('Val data loaded from ' + val_file_path)


            val_dataset = ImagesListFileFolder(
                val_file_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]))

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=val_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False)

            old_classes_number = 0
            new_classes_number = len(val_dataset.classes)

            print("Classes number = " + str(new_classes_number))
            print("Validation-set size = " + str(len(val_dataset)))
            print("Number of batches in Validation-set = " + str(len(val_loader)))

            if used_model == 'resnet18':
                model = models.resnet18(pretrained=False, num_classes=new_classes_number)
            else:
                print('wrong model')
                sys.exit(-1)

            print('Loading saved model from ' + model_load_path)
            state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state['state_dict'])

            if tc.is_available():
                model = model.cuda(gpu)
            else:
                print("GPU not available")
                sys.exit(-1)

            print('Validation on Batch 1...')
            model.eval()
            top1 = AverageMeter.AverageMeter()
            top5 = AverageMeter.AverageMeter()
            # Validation on both old and new data
            for data in val_loader:
                inputs, labels = data
                if tc.is_available():
                    inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
                outputs = model(Variable(inputs))
                prec1, prec5 = utils.accuracy(outputs.data, labels, topk=(1, min(5, P * b)))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
            # -------------------------------------------
            print('BATCH 1 | Val : acc@1 = {}% ; acc@5 = {}%'.format(top1.avg, top5.avg))
            top1_val_accuracies.append(top1.avg)
            top5_val_accuracies.append(top5.avg)
            print('top1 accuracies so far : ' + str(top1_val_accuracies))
            print('top5 accuracies so far : ' + str(top5_val_accuracies))


        else:
            batch_algo_name = algo_name + '_b' + str(b)
            batch_models_save_dir = os.path.join(models_save_dir, batch_algo_name)
            old_train_file_path = os.path.join(exemplars_dir, str(b) + '_old'+file_names_suffix)
            new_train_file_path = os.path.join(dataset_files_path, 'separated/train/batch' + str(b))
            old_val_file_path = os.path.join(dataset_files_path, 'accumulated/val/batch' + str(b - 1))
            new_val_file_path = os.path.join(dataset_files_path, 'separated/val/batch' + str(b))

            batch_lr = lr / b
            if b == 2:
                model_load_path = first_model_load_path
            else:
                model_load_path = os.path.join(models_save_dir, algo_name+'_b'+str(b-1)+'.pt')


            print('New train data loaded from ' + new_train_file_path)
            print('Old train data loaded from ' + old_train_file_path)
            print('New val data loaded from ' + new_val_file_path)
            print('Old val data loaded from ' + old_val_file_path)

            if saving_intermediate_models:
                if not os.path.exists(batch_models_save_dir):
                    os.mkdir(batch_models_save_dir)


            old_train_dataset = ImagesListFileFolder(
                old_train_file_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]), return_path=True)

            new_train_dataset = ImagesListFileFolder(
                new_train_file_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize, ]), return_path=True
                )

            new_and_old_train_datasets = torch.utils.data.dataset.ConcatDataset(
                (old_train_dataset, new_train_dataset))

            old_val_dataset = ImagesListFileFolder(
                old_val_file_path,
                transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,]))

            new_val_dataset = ImagesListFileFolder(
                new_val_file_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]))


            val_dataset = torch.utils.data.dataset.ConcatDataset((
                old_val_dataset, new_val_dataset
            ))

            train_loader = torch.utils.data.DataLoader(
                new_and_old_train_datasets, batch_size=new_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False)

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=val_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False)


            old_classes_number = len(old_train_dataset.classes)
            new_classes_number = len(new_train_dataset.classes)
            classes_number = old_classes_number + new_classes_number

            print("lr = " + str(batch_lr))
            print("Old classes number = " + str(old_classes_number))
            print("New classes number = " + str(new_classes_number))
            print("Training-set size = " + str(len(new_and_old_train_datasets)))
            print("Validation-set size = " + str(len(val_dataset)))
            print("Number of batches in Training-set = " + str(len(train_loader)))
            print("Number of batches in Validation-set = " + str(len(val_loader)))

            if used_model == 'resnet18':
                model_ft = models.resnet18(pretrained=False, num_classes=old_classes_number)
            elif used_model == 'resnet50':
                model_ft = models.resnet50(pretrained=False, num_classes=old_classes_number)
            else: #default model
                model_ft = models.resnet50(pretrained=False, num_classes=old_classes_number)

            print('Loading saved model from ' + model_load_path)
            state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
            model_ft.load_state_dict(state['state_dict'])

            model_ft.fc = nn.Linear(512, old_classes_number + new_classes_number)

            if tc.is_available():
                model_ft = model_ft.cuda(gpu)
            else:
                print("GPU not available")
                sys.exit(-1)

            # Define Loss and Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=batch_lr, momentum=momentum, weight_decay=weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=patience, factor=lr_decay)
            # scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=lr_decay)


            # Training
            print("-" * 20)
            print("Training...")
            epoch = 0
            top1_v_acc = 0
            top5_v_acc = 0
            for epoch in range(num_epochs):
                top1 = AverageMeter.AverageMeter()
                top5 = AverageMeter.AverageMeter()
                model_ft.train()
                # scheduler.step()
                running_loss = 0.0
                nb_batches = 0
                # zero the parameter gradients
                optimizer_ft.zero_grad()
                for i, data in enumerate(train_loader, 0):
                    nb_batches += 1
                    # get the data
                    (inputs, labels), paths = data

                    if tc.is_available():
                        inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

                    # wrap it in Variable
                    inputs, labels = Variable(inputs), Variable(labels)


                    # forward + backward + optimize
                    outputs = model_ft(inputs)
                    loss = criterion(outputs, labels)

                    loss.data /= iter_size
                    loss.backward()
                    running_loss += loss.data.item()
                    if (i+1)%iter_size == 0:
                        optimizer_ft.step()
                        optimizer_ft.zero_grad()

                scheduler.step(loss.cpu().data.numpy())

                # Model evaluation
                model_ft.eval()

                #Validation on both old and new data
                for data in val_loader:
                    inputs, labels = data
                    if tc.is_available():
                        inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
                    outputs = model_ft(Variable(inputs))
                    prec1, prec5 = utils.accuracy(outputs.data, labels, topk=(1, min(5, P * b)))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))


                # -------------------------------------------
                current_elapsed_time = time.time() - starting_time
                print('{:03}/{:03} | {} | Train : loss = {:.4f}  | Val : acc@1 = {}% ; acc@5 = {}%'.
                      format(epoch + 1, num_epochs, timedelta(seconds=round(current_elapsed_time)),
                             running_loss / nb_batches, top1.avg , top5.avg))


                top1_v_acc = top1.avg
                top5_v_acc = top5.avg

                # Saving model
                if saving_intermediate_models and (epoch+1) % 10 == 0:
                    state = {
                        'state_dict': model_ft.state_dict(),
                        'optimizer': optimizer_ft.state_dict(),
                    }

                    torch.save(state, batch_models_save_dir +'/'+ str(epoch) + '.pt')


            print('Finished Training, elapsed training time : {}'.format(
                timedelta(seconds=round(time.time() - starting_time))))

            #Saving model
            print('Saving model in ' + batch_models_save_dir + '.pt' + '...')
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer_ft.state_dict()
            }
            torch.save(state, batch_models_save_dir + '.pt')

            top1_val_accuracies.append(top1_v_acc)
            top5_val_accuracies.append(top5_v_acc)


            print('top1 accuracies so far : ' + str(top1_val_accuracies))
            print('top5 accuracies so far : ' + str(top5_val_accuracies))

        #######################################################################
        # Do for all batches, including the first non incremental batch
        new_train_dataset = ImagesListFileFolder(
            new_train_file_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize, ]), return_path = True)

        #computing number of exemplars
        m = int(math.ceil(K / (old_classes_number + new_classes_number)))

        new_train_loader = torch.utils.data.DataLoader(
            new_train_dataset, batch_size=exemplars_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False)


        features_extractor = nn.Sequential(*list(model.children())[:-1])
        features_extractor.eval()

        if tc.is_available():
            features_extractor = features_extractor.cuda(gpu)
        else:
            print("GPU not available")

        #####################

        print('New Training features extraction')
        features_names = None  # np.empty([512, ])
        scores_names = None  # np.empty([512, ])
        file_names = None  # np.empty([1, ])

        i = 0  # beginning

        for data in new_train_loader:
            (inputs, labels), paths = data
            if tc.is_available():
                inputs = inputs.cuda(gpu)
            # wrap it in Variable
            inputs = Variable(inputs)
            features = features_extractor(inputs)
            outputs = model(inputs)
            np_outputs = outputs.data.cpu().numpy()  # variable -> numpy
            np_paths = np.array(paths)  # tuple -> numpy
            np_features = features.data.cpu().numpy()  # variable -> numpy
            np_features = np_features.reshape(np_features.shape[0], np_features.shape[1])

            if i == 0:
                file_names = np_paths
                features_names = np_features
                scores_names = np_outputs
                i = 1
            else:
                file_names = np.append(file_names, np_paths)
                features_names = np.vstack((features_names, np_features))
                scores_names = np.vstack((scores_names, np_outputs))

        print('features final shape = ' + str(features_names.shape))
        print('file names final shape =' + str(len(file_names)))

        ## creating dict
        features_dict = {}
        scores_dict = {}

        for i in range(len(file_names)):
            if file_names[i] in features_dict:  # voir s'il y a une image repetee deux fois dans le fichier d'entree
                print(str(file_names[i]) + ' exists as ' + str(features_dict[file_names[i]]))
            features_dict[file_names[i]] = features_names[i]
            scores_dict[file_names[i]] = scores_names[i]

        print('len features dict = ' + str(len(features_dict.keys())))
        print('len scores dict = ' + str(len(scores_dict.keys())))

        #########################################################

        print('saving features')
        images_files = open(new_train_file_path, 'r').readlines()
        print('image file len = ' + str(len(images_files)))
        batch_features_destination_dir = os.path.join(features_destination_dir, 'batch' + str(b))
        if not os.path.exists(batch_features_destination_dir):
            os.makedirs(batch_features_destination_dir)
        features_out_file = os.path.join(batch_features_destination_dir, 'features')
        scores_out_file = os.path.join(batch_features_destination_dir, 'scores')

        features_out = open(features_out_file, 'w')
        scores_out = open(scores_out_file, 'w')
        for image_file in images_files:
            image_file = image_file.strip('\n')
            image_file = image_file.split()[0]
            if '.jpg' in image_file or '.jpeg' in image_file or '.JPEG' in image_file or '.png' in image_file:
                features_out.write(str(' '.join([str(e) for e in list(features_dict[image_file])])) + '\n')
                scores_out.write(str(' '.join([str(e) for e in list(scores_dict[image_file])])) + '\n')
            else:
                print('image file = ' + str(image_file))
        features_out.close()
        scores_out.close()

        print('Constructing exemplars,  Exemplars number per class = ' + str(m)+'...')

        herding.compute_rebuffi_herding_faster(exemplars_dir, new_train_file_path, features_out_file, m,
                                               str(b + 1) + '_old')

        #reduce exemplars for past classes
        herding.reduce_exemplars(exemplars_dir, old_classes_number, m, b, file_names_suffix)


        print('Total elapsed time for current batch: {}'.format(
            timedelta(seconds=round(time.time() - starting_time))))

    print('TOP1 validation accuracies = '+str([float(str(e)[:6]) for e in top1_val_accuracies[1:]]))
    print('TOP1 mean incremental accuracy = '+str(np.mean(np.array(top1_val_accuracies[1:]))))
    print('***************')
    print('TOP5 validation accuracies = '+str([float(str(e)[:6]) for e in top5_val_accuracies[1:]]))
    print('TOP5 mean incremental accuracy = '+str(np.mean(np.array(top5_val_accuracies[1:]))))

# Print warnings (Possibly corrupt EXIF files):
if len(warn_list) > 0:
    print("\n" + str(len(warn_list)) + " Warnings\n")
    # for i in range(len(warn_list)):
    #     print("warning " + str(i) + ":")
    #     print(str(i)+":"+ str(warn_list[i].category) + ":\n     " + str(warn_list[i].message))
else:
    print('No warnings.')