from __future__ import division
import torch, random, numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from MyImageFolder import ImagesListFileFolder, ImagesListFolder
from Utils import DataUtils, AverageMeter
from Herding_AL import StaticHerding
from torchvision import models
import os, socket, sys, time, warnings, math
from datetime import timedelta
import torch.cuda as tc, torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from configparser import ConfigParser
from torch.autograd import Variable
from torch.optim import lr_scheduler
from data_utils import min_max, create_dist_matrix
from classical_AFs import oracle_annotation
from balancing_driven_AFs import *
from sklearn import preprocessing

try:
    import cPickle
except:
    import _pickle as cPickle

import torch.nn.functional as F

use_cuda = tc.is_available()

if not use_cuda:
    print("GPU not available")
    sys.exit(-1)


def th_calibration(scores, N, n):
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            if n[j] > 0:
                scores[i][j] *= N / n[j]
    return scores


def get_dataset_N_n(dsets, model_num_classes):
    n = {}
    for class_name in range(model_num_classes):
        for dset in dsets:
            if class_name in dset.targets:
                n[class_name] = len([x for x in dset.targets if x == class_name])

        if class_name not in n.keys():  # undetected classes
            n[class_name] = 0

    N = sum([len(dset) for dset in dsets])
    return N, n


def active_learning(rerun, sess, batch_size, batch_number, model, N, n, B, next_new_data_file_path,
                    sess_new_data_paths, run_data_output_dir, undetected_classes):
    """Annotate a new batch of images
       Return the oracle (class oracle_annotation)"""

    new_data_file_fpath = os.path.join(run_data_output_dir, str(batch_number) + '_new')
    new_data_feat_file_fpath = os.path.join(run_data_output_dir, str(batch_number) + '_feat_new')

    features_extractor = nn.Sequential(*list(model.children())[:-1])
    features_extractor.eval()
    features_extractor = features_extractor.cuda(gpu)

    model.eval()
    model = model.cuda(gpu)
    print('\n\n********* PREPARING NEW DATA FOR THIS BATCH ********* ')
    print('------> Features extraction of new data S{}* using model M{}'.format(batch_number, batch_number - 1))
    print('data partially loaded from : ' + next_new_data_file_path)

    dataset = ImagesListFolder(
        sess_new_data_paths, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]), return_path=True)

    new_data_size = len(dataset)
    budget = int((new_data_size * B) / 100)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    full_features = None
    full_paths = None
    full_scores = None
    full_classes = np.array([dataset.samples[i][1] for i in range(len(dataset.samples))])

    for i, data in enumerate(dataset_loader):
        (inputs, labels), paths = data
        inputs = inputs.cuda(gpu)
        features = features_extractor(Variable(inputs))
        scores = model(Variable(inputs))
        # scores[:, undetected_classes] = -np.Inf
        scores = F.softmax(scores, dim=1)

        if apply_th_train or apply_th_val_al:
            scores = th_calibration(scores, N, n)

        np_paths = np.array(paths)
        np_features = features.data.cpu().numpy()
        np_features = np_features.reshape(np_features.shape[0], np_features.shape[1])
        np_scores = scores.data.cpu().numpy()

        if i == 0:
            full_paths = np_paths
            full_features = np_features
            full_scores = np_scores
        else:
            full_paths = np.append(full_paths, np_paths)
            full_features = np.vstack((full_features, np_features))
            full_scores = np.vstack((full_scores, np_scores))

    probas_path = os.path.join(run_data_output_dir, str(batch_number) + '_proba.pkl')
    with open(probas_path, 'wb') as f:
        cPickle.dump(full_scores, f)

    print("")

    # Features normalization
    full_features = min_max(full_features)

    dist_matrix = None
    if ('kcenters' in classical_AF and sess == 0) or 'kcenters' in balancing_AF: dist_matrix = create_dist_matrix(full_features)

    # Available indexes for oracle
    # todo: modify entropy code to allow working on avail indexes
    avail_indexes = np.arange(full_features.shape[0])

    print('------> Manual annotation')
    print('images number -> ' + str(len(full_paths)))
    print('budget of oracle -> ' + str(budget))
    print('')

    oracle = oracle_annotation(budget, full_features, full_paths, dist_matrix, avail_indexes, full_classes, probas_path)

    if sess == 0:
        # Annotation of budget by oracle + Balancing
        oracle.run(classical_AF)

        # Data annotated manually by the oracle
        # annotated_img_idx = oracle.annotated_img_idx
        annotated_classes = oracle.annotated_classes
        annotated_paths = oracle.annotated_paths
        annotated_features = oracle.annotated_features

    else:

        if balancing_AF == 'poor':
            annotated_classes, annotated_paths, annotated_features = poor(
                full_features, full_classes,
                full_paths, budget,
                new_data_file_fpath,
                new_data_feat_file_fpath)


        elif balancing_AF == 'bcore':
            annotated_classes, annotated_paths, annotated_features = bcore(dist_matrix, full_features, full_classes,
                                                                                           full_paths, budget,
                                                                                           new_data_file_fpath,
                                                                                           new_data_feat_file_fpath)

        else:
            print('invalid balancing type')
            sys.exit(-1)

    classes = list(set(annotated_classes))
    print('Classes detected by the oracle = ' + str(sorted(classes)))

    ###################

    # list of new data after oracle : to be saved in a file
    new_data_list = []
    for (path, class_) in zip(annotated_paths, annotated_classes):
        new_data_list.append(path + ' ' + str(class_))


    # Writing new data lists file:
    print('New data S{}+ saved in {} '.format(batch_number, new_data_file_fpath))
    if sess == 0:
        new_data_file = open(new_data_file_fpath, 'w')
    else:
        new_data_file = open(new_data_file_fpath, 'a')


    for line in new_data_list:
        new_data_file.write(line + '\n')
    new_data_file.close()

    ############################### SAVING  annotated features
    if balancing_AF == 'poor':
        new_data_features = None
        for feat in annotated_features:
            if new_data_features is None:
                new_data_features = feat
            else:
                new_data_features = np.vstack((new_data_features, feat))

        if sess == 0:
            new_data_feat_file = open(new_data_feat_file_fpath, 'w')
        else:
            new_data_feat_file = open(new_data_feat_file_fpath, 'a')

        for feat in new_data_features:
            new_data_feat_file.write(str(' '.join([str(e) for e in list(feat)])) + '\n')
        new_data_feat_file.close()
    ##########################################

    oracle_annotated_paths = []
    for (path, class_) in zip(annotated_paths, annotated_classes):
        oracle_annotated_paths.append(path + ' ' + str(class_) + '\n')

    return oracle_annotated_paths


def main(I):
    if __name__ == '__main__':
        if not os.path.exists(data_output_dir):
            os.makedirs(data_output_dir)

        if not os.path.exists(models_save_dir):
            os.makedirs(models_save_dir)

        # catching warnings
        with warnings.catch_warnings(record=True) as warn_list:
            herding = StaticHerding()

            runs_top1_acc = []
            runs_topx_acc = []

            first_run_starting_time = time.time()
            for r in range(1, num_runs + 1):
                run_data_output_dir = os.path.join(data_output_dir, 'run_' + str(r))
                if not os.path.exists(run_data_output_dir):
                    os.makedirs(run_data_output_dir)

                run_models_save_dir = os.path.join(models_save_dir, 'run_' + str(r))
                if not os.path.exists(run_models_save_dir):
                    os.makedirs(run_models_save_dir)

                run_features_destination_dir = os.path.join(run_data_output_dir, 'features')
                if not os.path.exists(run_features_destination_dir):
                    os.mkdir(run_features_destination_dir)

                top1_val_accuracies = []
                topx_val_accuracies = []
                previous_model = None

                run_starting_time = time.time()
                batch_oracle_annotated_paths = {}
                undetected_classes = []
                for b in range(1, T + 1):
                    print('*' * 110)
                    print('*' * 46 + ' Run {}/{} | BATCH {} '.format(r, num_runs, b) + '*' * 45)
                    print('*' * 110 + '\n')

                    if b == 1:
                        model_load_path = first_model_load_path
                        new_train_file_path = path_train_batch1
                        val_file_path = path_val_batch1
                        print('Train data loaded from ' + new_train_file_path)
                        print('Val data loaded from ' + val_file_path)

                        new_train_dataset = ImagesListFileFolder(
                            new_train_file_path,
                            transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize]), return_path=True
                        )

                        model_dsets = [new_train_dataset]

                        val_dataset = ImagesListFileFolder(
                            val_file_path,
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize]), return_path=True)

                        val_loader = torch.utils.data.DataLoader(
                            val_dataset, batch_size=val_batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=False)

                        old_classes_number = 0
                        new_classes_number = len(val_dataset.classes)
                        print("Classes number = " + str(new_classes_number))
                        print("Validation-set size = " + str(len(val_dataset)))

                        model = models.resnet18(pretrained=False, num_classes=base)

                        print('\nLoading model from ' + model_load_path)
                        state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(state['state_dict'])
                        model = model.cuda(gpu)

                        print('\n\n********* VALIDATION ********* ')
                        model.eval()
                        top1 = AverageMeter()
                        topx = AverageMeter()
                        top = min(5, new_classes_number)
                        N, n = get_dataset_N_n(model_dsets, model.fc.out_features)
                        # Validation on batch 1
                        for data in val_loader:
                            (inputs, labels), paths = data
                            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
                            scores = model(Variable(inputs))

                            if apply_th_train or apply_th_val_al:
                                scores = th_calibration(F.softmax(scores, dim=1), N, n)

                            prec1, prec5 = utils.accuracy(scores.data, labels, topk=(1, top))

                            top1.update(prec1.item(), inputs.size(0))
                            topx.update(prec5.item(), inputs.size(0))
                        # -------------------------------------------
                        print('BATCH 1 | Val : acc@1 = {:.2f}% ; acc@{} = {:.2f}%'.format(top1.avg, top, topx.avg))
                        top1_val_accuracies.append(top1.avg)
                        topx_val_accuracies.append(topx.avg)

                        oracle_annotated_paths = open(new_train_file_path, 'r').readlines()
                        batch_oracle_annotated_paths[b] = oracle_annotated_paths

                    else:

                        batch_algo_name = algo_name + '_b' + str(b)

                        old_train_file_path = os.path.join(run_data_output_dir, str(b) + '_old')
                        new_val_file_path = os.path.join(dataset_files_dir, 'separated/val/batch' + str(b))
                        if b == 2:
                            old_val_file_path = path_val_batch1
                        else:
                            old_val_file_path = os.path.join(dataset_files_dir, 'accumulated/val/batch' + str(b - 1))

                        if mode == "il":  # supervised :
                            I = 1
                            new_train_file_path = os.path.join(train_files_dir, 'batch' + str(b))
                            oracle_annotated_paths = open(new_train_file_path, 'r').readlines()
                            batch_oracle_annotated_paths[b] = oracle_annotated_paths

                        print('Old train data loaded from ' + old_train_file_path)
                        print('New val data loaded from ' + new_val_file_path)
                        print('Old val data loaded from ' + old_val_file_path)

                        # Data loaders for training
                        old_train_dataset = ImagesListFileFolder(
                            old_train_file_path,
                            transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize]), return_path=True)

                        old_val_dataset = ImagesListFileFolder(
                            old_val_file_path,
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize]), return_path=True)

                        new_val_dataset = ImagesListFileFolder(
                            new_val_file_path,
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize]), return_path=True)

                        val_datasets = torch.utils.data.dataset.ConcatDataset((
                            old_val_dataset, new_val_dataset
                        ))

                        val_loader = torch.utils.data.DataLoader(
                            val_datasets, batch_size=val_batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=False)

                        old_classes_number = len(old_train_dataset.classes)

                        # Loading the model
                        if b == 2:
                            model_load_path = first_model_load_path
                        else:
                            model_load_path = os.path.join(run_models_save_dir, algo_name + '_b' + str(b - 1) + '.pt')

                        model = models.resnet18(pretrained=False, num_classes= base + P * (b - 2))

                        print('\nLoading saved model from ' + model_load_path)
                        state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(state['state_dict'])

                        model.fc = nn.Linear(model.fc.in_features, base + P * (b - 1))
                        model = model.cuda(gpu)

                        # Define Loss and Optimizer
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
                        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)

                        print("\nlr = {:.4f}".format(lr))
                        print("Old classes number = " + str(old_classes_number))
                        print("Old Training-set size = " + str(len(old_train_dataset)))
                        print("Validation-set size = " + str(len(val_datasets)) + '\n')
                        ##############################
                        # Active learning : update batch_oracle_annotated_paths / Semi-supervised labelisation step
                        batch_oracle_annotated_paths[b] = []
                        next_new_train_file_path = os.path.join(train_files_dir, 'batch' + str(b))

                        for sess in range(I):

                            if sess == 0:
                                al_model = previous_model
                            else:
                                al_model = model

                            sess_epochs = int(num_epochs / I) #todo modify

                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr

                            if mode == "il" or I == 1:
                                sess_budget = B
                            else:
                                if sess == 0:  # take 40% of budget
                                    sess_budget = math.ceil(int(B * 40 / 100))
                                else:
                                    sess_budget = math.ceil(int(B * 20 / 100))

                            next_new_train_paths_list = open(next_new_train_file_path, 'r').readlines()
                            assert (sorted(list(set(next_new_train_paths_list))) == sorted(next_new_train_paths_list))
                            assert (sorted(list(set(batch_oracle_annotated_paths[b]))) == sorted(
                                batch_oracle_annotated_paths[b]))

                            sess_new_train_paths = list(
                                set(next_new_train_paths_list) - set(batch_oracle_annotated_paths[b]))
                            oracle_annotated_paths = active_learning(rerun, sess, new_batch_size, b, al_model, N, n,
                                                                     sess_budget,
                                                                     next_new_train_file_path, sess_new_train_paths,
                                                                     run_data_output_dir,
                                                                     undetected_classes)

                            batch_oracle_annotated_paths[b].extend(oracle_annotated_paths)

                            new_train_file_path = os.path.join(run_data_output_dir, str(b) + '_new')

                            print('New train data loaded from ' + new_train_file_path)

                            new_train_dataset = ImagesListFileFolder(
                                new_train_file_path,
                                transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize]), return_path=True
                            )

                            model_dsets = [old_train_dataset, new_train_dataset]

                            new_and_old_train_datasets = torch.utils.data.dataset.ConcatDataset(
                                (old_train_dataset, new_train_dataset))

                            train_loader = torch.utils.data.DataLoader(
                                new_and_old_train_datasets, shuffle=True, batch_size=new_batch_size,
                                num_workers=num_workers, pin_memory=False)

                            new_classes_number = len(new_train_dataset.classes)
                            undetected_classes.extend(
                                list(set(range(base + P * (b - 2), base + P * (b - 1))) - set(new_train_dataset.classes)))
                            undetected_classes = sorted(list(set(undetected_classes)))
                            print('undetected_classes = ' + str(undetected_classes))

                            print("New classes number = " + str(new_classes_number))
                            print("New Training-set size = " + str(len(new_train_dataset)))
                            print("Training-set size = " + str(len(new_and_old_train_datasets)))

                            N, n = get_dataset_N_n(model_dsets, model.fc.out_features)

                            # Training
                            print("-" * 20)
                            print('\n\n********* TRAINING ********* ')
                            starting_time = time.time()

                            for epoch in range(sess_epochs):
                                top1 = AverageMeter()
                                topx = AverageMeter()
                                model.train()
                                running_loss = 0.0
                                nb_batches = 0
                                optimizer.zero_grad()
                                for i, data in enumerate(train_loader, 0):
                                    nb_batches += 1
                                    (inputs, labels), paths = data
                                    inputs, labels = Variable(inputs.cuda(gpu)), Variable(labels.cuda(gpu))
                                    scores = model(inputs)
                                    # scores[:, undetected_classes] = -np.Inf
                                    loss = criterion(scores, labels)
                                    loss.data /= iter_size
                                    loss.backward()
                                    running_loss += loss.data.item()
                                    if (i + 1) % iter_size == 0:
                                        optimizer.step()
                                        optimizer.zero_grad()
                                scheduler.step(loss.cpu().data.numpy())

                                # Model evaluation
                                model.eval()
                                top = min(5, old_classes_number + new_classes_number)
                                for data in val_loader:
                                    (inputs, labels), paths = data
                                    inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

                                    scores = model(Variable(inputs))
                                    # scores[:, undetected_classes] = -np.Inf

                                    if apply_th_train or apply_th_val_al:
                                        scores = th_calibration(F.softmax(scores, dim=1), N, n)

                                    prec1, prec5 = utils.accuracy(scores.data, labels, topk=(1, top))
                                    top1.update(prec1.item(), inputs.size(0))
                                    topx.update(prec5.item(), inputs.size(0))

                                current_elapsed_time = time.time() - starting_time
                                print(
                                    '{}/{} | lr={:.5f} |{:03}/{:03} | {} | Train : loss = {:.4f}  | Val : acc@1 = {:.2f}% ; acc@{}= {:.2f}%'.
                                    format(sess, I, optimizer.param_groups[0]['lr'], epoch + 1,
                                           num_epochs, timedelta(seconds=round(current_elapsed_time)),
                                           running_loss / nb_batches, top1.avg, top, topx.avg))

                        # Training finished
                        print('Saving model in ' + batch_algo_name + '.pt' + '...')
                        state = {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                        }
                        torch.save(state, os.path.join(run_models_save_dir, batch_algo_name) + '.pt')

                        top1_val_accuracies.append(top1.avg)
                        topx_val_accuracies.append(topx.avg)

                        print("")
                        print('TOP1 val acc = ' + str([float(str(e)[:6]) for e in top1_val_accuracies]))
                        print('TOP{} val acc = '.format(top) + str([float(str(e)[:6]) for e in topx_val_accuracies]))

                    previous_model = model

                    ########## Herding
                    new_train_dataset = ImagesListFileFolder(
                        new_train_file_path,
                        transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize, ]), return_path=True)

                    # computing number of exemplars
                    m = int(math.ceil(K / (old_classes_number + new_classes_number)))

                    new_train_loader = torch.utils.data.DataLoader(
                        new_train_dataset, batch_size=new_batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=False)

                    features_extractor = nn.Sequential(*list(model.children())[:-1])
                    features_extractor.eval()
                    features_extractor = features_extractor.cuda(gpu)

                    print('\n\n********* PREPARING OLD DATA FOR THE NEXT BATCH ********* ')
                    if b == 1:
                        print('------> Features extraction of new data S{}* using model M{}'.format(b, b))
                    else:
                        print('------> Features extraction of new data S{}+ using model M{}'.format(b, b))

                    print('data loaded from : ' + new_train_file_path)

                    full_features = None
                    full_paths = None

                    for data in new_train_loader:
                        (inputs, labels), paths = data
                        inputs = inputs.cuda(gpu)
                        features = features_extractor(Variable(inputs))
                        np_paths = np.array(paths)
                        np_features = features.data.cpu().numpy()
                        np_features = np_features.reshape(np_features.shape[0], np_features.shape[1])
                        if full_features is None:
                            full_paths = np_paths
                            full_features = np_features
                        else:
                            full_paths = np.append(full_paths, np_paths)
                            full_features = np.vstack((full_features, np_features))

                    features_dict = {}
                    for i in range(len(full_paths)):
                        if full_paths[i] in features_dict:
                            print(str(full_paths[i]) + ' is redundant ')
                        features_dict[full_paths[i]] = full_features[i]

                    #########################################################

                    images_files = open(new_train_file_path, 'r').readlines()
                    batch_features_destination_dir = os.path.join(run_features_destination_dir, 'batch' + str(b))
                    if not os.path.exists(batch_features_destination_dir):
                        os.makedirs(batch_features_destination_dir)
                    features_out_file = os.path.join(batch_features_destination_dir, 'features')

                    features_out = open(features_out_file, 'w')
                    for image_file in images_files:
                        image_file = image_file.strip('\n')
                        image_file = image_file.split()[0]
                        if '.jpg' in image_file or '.jpeg' in image_file or '.JPEG' in image_file or '.png' in image_file:
                            features_out.write(str(' '.join([str(e) for e in list(features_dict[image_file])])) + '\n')
                        else:
                            print('image file = ' + str(image_file))
                    features_out.close()

                    print('Exemplars number per class = ' + str(m))
                    print('Choosing exemplars for new classes...')

                    herding.compute_rebuffi_herding_faster(run_data_output_dir, new_train_file_path, features_out_file,
                                                           batch_oracle_annotated_paths[b], m, str(b + 1) + '_old')

                    if b != 1:
                        print('Reducing exemplars for old classes...')
                        herding.reduce_exemplars(run_data_output_dir, old_train_dataset.classes, m, b,
                                                 full_paths_suffix)

                    print('Old data for batch {} saved in {} '.format(b + 1, os.path.join(run_data_output_dir,
                                                                                          str(b + 1) + '_old')))

                print('Current run elapsed time : {}'.format(
                    timedelta(seconds=round(time.time() - run_starting_time))))

                mean_top1 = np.mean(np.array(top1_val_accuracies)[1:]) if len(top1_val_accuracies) > 1 else 0.0
                mean_topx = np.mean(np.array(topx_val_accuracies)[1:]) if len(topx_val_accuracies) > 1 else 0.0
                print("")
                print('TOP1 validation accuracies = ' + str([float(str(e)[:6]) for e in top1_val_accuracies]))
                print('TOP1 mean incremental accuracy = ' + str(mean_top1)[:6])
                print('***************')
                print('TOP{} validation accuracies = '.format(top) + str(
                    [float(str(e)[:6]) for e in topx_val_accuracies]))
                print('TOP{} mean incremental accuracy = '.format(top) + str(mean_topx)[:6])

                runs_top1_acc.append(mean_top1)
                runs_topx_acc.append(mean_topx)

        runs_mean_top1_acc = np.mean(np.array(runs_top1_acc))
        runs_mean_topx_acc = np.mean(np.array(runs_topx_acc))
        runs_std_top1_acc = np.std(np.array(runs_top1_acc))
        runs_std_topx_acc = np.std(np.array(runs_topx_acc))

        print('*' * 110)
        print('*' * 110)
        print('Total elapsed time : {}'.format(
            timedelta(seconds=round(time.time() - first_run_starting_time))))
        print('****************************************************************')
        print('Average runs scores')
        print('****************************************************************')
        print('TOP1 mean incremental accuracy = {:.3f}      [+/- {:.2f}]'.format(runs_mean_top1_acc, runs_std_top1_acc))
        print('TOP{} mean incremental accuracy = {:.3f}      [+/- {:.2f}]'.format(top, runs_mean_topx_acc,
                                                                                  runs_std_topx_acc))

        # Print warnings (Possibly corrupt EXIF files):
        if len(warn_list) > 0:
            print("\n" + str(len(warn_list)) + " Warnings\n")
            # for i in range(len(warn_list)):
            #     print("warning " + str(i) + ":")
            #     print(str(i)+":"+ str(warn_list[i].category) + ":\n     " + str(warn_list[i].message))
        else:
            print('No warnings.')


###########################################################

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

# experiments mode
mode = cp['mode']

# reading parameters
algo_name = cp['algo_name']
balancing_AF = cp['balancing_AF']
gpu = int(cp['gpu'])
patience = int(cp['patience'])
num_workers = int(cp['num_workers'])
lr_decay = float(cp['lr_decay'])
lr = float(cp['lr'])
momentum = float(cp['momentum'])
weight_decay = float(cp['weight_decay'])
old_batch_size = int(cp['old_batch_size'])
new_batch_size = int(cp['new_batch_size'])
val_batch_size = int(cp['val_batch_size'])
iter_size = int(old_batch_size / new_batch_size)
num_epochs = int(cp['num_epochs'])
T = int(cp['T'])
I = int(cp['I'])
models_save_dir = cp['models_save_dir']
K = int(cp['K'])
P = int(cp['P'])
base = int(cp['base'])
num_runs = int(cp['num_runs'])
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']

# First batch parameters
first_model_load_path = cp['first_model_load_path']

# Incremental batches dataset
normalization_dataset_name = cp['normalization_dataset_name']
dataset_files_dir = cp['dataset_files_dir']
data_output_dir = cp['data_output_dir']

# Semi-supervised labelisation settings
B = float(cp['B'])
classical_AF = cp['classical_AF']
rerun = cp['rerun'] == "True"
apply_th_train = cp['apply_th_train'] == "True"
apply_th_val_al = cp['apply_th_val_al'] == "True"
train_files_dir = os.path.join(dataset_files_dir, 'separated/train')
path_val_batch1 = os.path.join(dataset_files_dir, 'batch1/val.lst')
path_train_batch1 = os.path.join(dataset_files_dir, 'batch1/train.lst')
full_paths_suffix = ''
################ Global variables
utils = DataUtils()
dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)
normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

print("Running on " + str(socket.gethostname()) + " | gpu " + str(gpu))
utils.print_parameters(cp)
assert (mode in ['il', 'il_al'])
# start the main program
main(I)
