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
import utils.AverageMeter as AverageMeter
import socket
from utils.MyImageFolder import ImagesListFileFolder
import math
import numpy as np
import copy
from sklearn import preprocessing
from numpy.linalg import norm

####################################################################################

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

def load_class_paths(class_number, batch_paths):
    batch_paths = open(batch_paths, 'r').readlines()
    class_paths = []
    for path_line in batch_paths:
        image_path = path_line.strip()
        path_line = path_line.strip().split()
        image_class = int(path_line[1])
        if image_class == class_number:
            class_paths.append(image_path)
    return class_paths

def compute_rebuffi_herding_faster(img_list, img_feats, exem_dir, m, file_name):
    per_class = m
    if not os.path.isdir(exem_dir):
        os.mkdir(exem_dir)
    rebuffi_exem = exem_dir + "/" + file_name
    f_rebuffi = open(rebuffi_exem, "w")
    # open the list of images and put the labels and associated paths in a list
    pos_list = []
    line_cnt = 0
    f_list = open(img_list)
    for line in f_list:
        pos_list.append(line.rstrip())
    f_list.close()
    # open the features file and store the features and the image paths in dictionaries with class labels as keys
    feat_dict = {}
    path_dict = {}
    ordered_labels = [] # list to store the labels in the original order from the initial list
    feat_cnt = 0
    f_feats = open(img_feats)
    for feat_line in f_feats:
        if 1 == 1: # TO DO put all features back here
            #print feat_cnt
            feat_line = feat_line.rstrip()
            # get the path and the label
            crt_parts = pos_list[feat_cnt].split(" ")
            label = crt_parts[1]
            path = crt_parts[0]
            if not label in feat_dict:
                feat_dict[label] = feat_line
                path_dict[label] = path
                ordered_labels.append(label)
            else:
                feat_dict[label] = feat_dict[label]+"\t"+feat_line
                path_dict[label] = path_dict[label]+"\t"+path
        feat_cnt = feat_cnt + 1
    f_feats.close()
    for ll in range(0, len(ordered_labels)): # TO DO run for all labels here
        label = ordered_labels[ll]
        # get the features and images associated to the current label
        feat_list = feat_dict[label].split("\t")
        img_list = path_dict[label].split("\t")
        # print "rebuffi started class:",ll,", class images:",len(img_list)
        # compute the mean representation of the class. normalize L2 each feature before mean computation
        sum_feat = [] # list for the sum of feature dimensions
        feat_size = feat_list[0].split(" ")
        # initialize the sum list
        for fs in range(0,len(feat_size)):
            sum_feat.append(0)
        feat_list_norm = []
        for feat in feat_list:
            np_feat = np.fromstring(feat,dtype=float,sep=' ')
            l2_norm = norm(np_feat)
            crt_feat_norm = ""
            local_feat = []
            for dd in range(0,np_feat.shape[0]):
                norm_dim = np_feat[dd]/l2_norm
                #crt_feat_norm = crt_feat_norm+" "+str(norm_dim)
                local_feat.append(norm_dim)
                sum_feat[dd] = sum_feat[dd] + norm_dim
            #crt_feat_norm.append(local_feat)
            feat_list_norm.append(local_feat)
        # compute the mean feature for the class
        mean_interm = []
        mean_norm = 0
        for dd in range(0, len(sum_feat)):
            dim_mean = sum_feat[dd]/len(feat_list)
            mean_interm.append(dim_mean)
            mean_norm = mean_norm+dim_mean * dim_mean
        mean_norm = math.sqrt(mean_norm)
        mean_final = [] # list to store the final form of the mean which is L2-normalized
        for dd in range(0, len(mean_interm)):
            l2_dim_mean = mean_interm[dd]/mean_norm
            mean_final.append(l2_dim_mean)
        #print mean_interm,"\n",mean_final
        # sort the features by their aggregate distance to the true mean of the class.
        # in each step, all remaining features are tested and the one that minimized the current distance of means is kept
        ranked_exemplars = {} # dict for exemplars ranked by their distance to the mean of the class
        aggregate_sum = [] # list to store the aggregate mean of exemplars that were already chosen
        for fs in range(0,len(feat_size)):
            aggregate_sum.append(0)
        time_start = time.time()
        #cumul_diff_dist= 0
        #cumul_diff_prep = 0
        #cumul_finalize = 0
        #cumul_diff_min = 0
        #cumul_compute = 0
        # print "started reranking"
        # next two lines to ensure that the number of exemplars is never bigger than the total number of images per class
        if per_class > len(feat_list_norm):
            per_class = len(feat_list_norm)
        while(len(ranked_exemplars) < per_class): # TO DO put the real number of exemplars here
            #print len(ranked_exemplars),per_class
            feat_cnt = 0
            min_dist = 10000000.0 # large initial value for the min distance
            crt_candidate = "" # path to the current image candidate
            feat_candidate = []	# tmp list to store the features of the current candidate
            for crt_feat in feat_list_norm:
                crt_img = img_list[feat_cnt]
                # check if the current image is in the dictionary of selected exemplars
                if not crt_img in ranked_exemplars:
                    #time_start_compute = time.time()
                    l2_dist = 0
                    norm_crt_feats = []
                    count_zeros = 0
                    #time_prep_start = time.time()
                    tmp_norm = 0
                    tmp_interm = []
                    crt_exemplars = float(len(ranked_exemplars)+1)
                    for dd in range(0,len(crt_feat)):
                        dim_sum = aggregate_sum[dd] + crt_feat[dd]
                        tmp_interm.append(dim_sum)
                        tmp_norm = tmp_norm + dim_sum * dim_sum
                    tmp_norm = math.sqrt(tmp_norm/crt_exemplars)
                    #time_prep_stop = time.time()
                    #time_finalize_start = time.time()
                    tmp_final = []
                    # compute the distance between the current aggregate mean and the mean of the class
                    feat_dist = 0
                    for dd in range(0, len(tmp_interm)):
                        l2_dim_mean = tmp_interm[dd]/tmp_norm
                        diff_dim = l2_dim_mean - mean_final[dd]
                        feat_dist = feat_dist + diff_dim * diff_dim
                        #tmp_final.append(l2_dim_mean)
                    #time_finalize_stop = time.time()
                    #cumul_finalize = cumul_finalize + time_finalize_stop - time_finalize_start
                    #cumul_diff_prep = cumul_diff_prep + time_prep_stop - time_prep_start
                    #time_start_min = time.time()
                    if feat_dist < min_dist:
                        min_dist = feat_dist
                        crt_candidate = crt_img
                        #print min_dist,crt_candidate
                        feat_candidate = [] # reinitialize the feats for the new candidate
                        # update the feats for the candidate
                        for nd in crt_feat:
                            feat_candidate.append(nd)
                    #time_stop_min = time.time()
                    #cumul_diff_min = cumul_diff_min + time_stop_min - time_start_min
                    #time_stop_compute = time.time()
                    #cumul_compute = cumul_compute + time_stop_compute - time_start_compute
                feat_cnt = feat_cnt+1
            # update the dictionary of exemplars
            ranked_exemplars[crt_candidate] = len(ranked_exemplars)
            # update the aggregate sum list with the features of the current candidate
            for nd in range(0, len(feat_candidate)):
                aggregate_sum[nd] = aggregate_sum[nd]+feat_candidate[nd]
        for img_path, dist in sorted(ranked_exemplars.iteritems(), key=lambda (k,v): (v,k), reverse=False):
            to_out = img_path+" "+ordered_labels[ll]
            f_rebuffi.write(to_out+"\n")
        time_stop = time.time()
        time_diff = time_stop - time_start
        #print "execution time:",time_diff,", preparation:",cumul_diff_prep,", dist computation:",cumul_finalize,", total compute time:",cumul_compute
        # print "execution time:",time_diff

    f_rebuffi.close()



def reduce_exemplars(old_classes_number, new_classes_number, m_nxt, balanced_file, exemplars_file, exem_dir, file_name):
    if not os.path.isdir(exem_dir):
        os.mkdir(exem_dir)
    rebuffi_exem = exem_dir + "/" + file_name
    f_rebuffi = open(rebuffi_exem, "w")

    for label in range(old_classes_number): #old class
        class_paths = load_class_paths(label, exemplars_file)
        for img_path in class_paths[:m_nxt]:
            f_rebuffi.write(img_path + "\n")


    for label in range(old_classes_number, old_classes_number+new_classes_number): #new class
        class_paths = load_class_paths(label, balanced_file)
        for img_path in class_paths[:m_nxt]:
            f_rebuffi.write(img_path + "\n")

    f_rebuffi.close()

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
# reading parameters
algo_name = cp['algo_name']  #full_no_freeze_fine_tuning_ilsvrc
gpu = int(cp['gpu'])
patience = int(cp['patience'])
num_workers = int(cp['num_workers'])
first_model_load_path = cp['first_model_load_path']
step_size = int(cp['step_size'])
lr_decay = float(cp['lr_decay'])
lr = float(cp['lr'])
momentum = float(cp['momentum'])
weight_decay = float(cp['weight_decay'])
old_batch_size = int(cp['old_batch_size'])
new_batch_size = int(cp['new_batch_size'])
val_batch_size = int(cp['val_batch_size'])
iter_size = int(old_batch_size / new_batch_size)
starting_epoch = int(cp['starting_epoch'])
normalization_dataset_name = cp['normalization_dataset_name']
used_model = cp['used_model']
first_batch_number = int(cp['first_batch_number'])
last_batch_number = int(cp['last_batch_number'])
models_save_dir = cp['models_save_dir']
K = int(cp['K'])
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']
saving_intermediate_models = cp['saving_intermediate_models'] == 'True'
unbalanced_num_epochs = int(cp['unbalanced_num_epochs'])
balanced_num_epochs = int(cp['balanced_num_epochs'])
dataset_files_path = cp['dataset_files_path']
exemplars_batch_size = int(cp['exemplars_batch_size'])
features_destination_dir = cp['features_destination_dir']
balanced_output_dir = cp['balanced_output_dir']
exemplars_output_dir = cp['exemplars_output_dir']



if not os.path.exists(exemplars_output_dir):
    os.makedirs(exemplars_output_dir)

if not os.path.exists(balanced_output_dir):
    os.makedirs(balanced_output_dir)

if not os.path.exists(models_save_dir):
    os.makedirs(models_save_dir)

# catching warnings
with warnings.catch_warnings(record=True) as warn_list:
    # Data loading code
    dataset_mean, dataset_std = get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)
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
    print("Balanced Number of epochs = " + str(balanced_num_epochs))
    print("Unbalanced Number of epochs = " + str(unbalanced_num_epochs))
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
                best_model = models.resnet18(pretrained=False, num_classes=new_classes_number)
            else:
                print('wrong model')
                sys.exit(-1)

            print('Loading saved model from ' + model_load_path)
            state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
            best_model.load_state_dict(state['state_dict'])

            if tc.is_available():
                best_model = best_model.cuda(gpu)
            else:
                print("GPU not available")
                sys.exit(-1)

            print('Validation on Batch 1...')
            best_model.eval()
            top1 = AverageMeter.AverageMeter()
            top5 = AverageMeter.AverageMeter()
            # Validation on both old and new data
            for data in val_loader:
                inputs, labels = data
                if tc.is_available():
                    inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
                outputs = best_model(Variable(inputs))
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
            # -------------------------------------------
            print('BATCH 1 | Val : acc@1 = {}% ; acc@5 = {}%'.format(top1.avg, top5.avg))
            top1_val_accuracies.append(top1.avg)
            top5_val_accuracies.append(top5.avg)

        else:
            batch_algo_name = algo_name + '_b' + str(b)
            batch_models_save_dir = os.path.join(models_save_dir, batch_algo_name)
            old_train_file_path = os.path.join(exemplars_output_dir, str(b) + '_old')
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
            best_top1_v_acc = -1
            best_top5_v_acc = -1
            best_optimizer_ft = None
            best_model = None
            for epoch in range(unbalanced_num_epochs):
#############################
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
                    prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))

                if top1.avg > best_top1_v_acc:
                    best_top1_v_acc = top1.avg
                    best_top5_v_acc = top5.avg
                    best_model = copy.deepcopy(model_ft)
                    best_optimizer_ft = copy.deepcopy(optimizer_ft)
                    best_epoch = epoch

                # -------------------------------------------
                current_elapsed_time = time.time() - starting_time
                print('{:03}/{:03} | {} | Train : loss = {:.4f}  | Val : acc@1 = {}% ; acc@5 = {}%'.
                      format(epoch + 1, balanced_num_epochs + unbalanced_num_epochs, timedelta(seconds=round(current_elapsed_time)),
                             running_loss / nb_batches, top1.avg , top5.avg))



                # Saving model
                if saving_intermediate_models and (epoch+1) % 10 == 0:
                    state = {
                        'epoch': epoch,
                        'state_dict': model_ft.state_dict(),
                        'optimizer': optimizer_ft.state_dict(),
                        'best_v_acc': top5.avg
                    }

                    torch.save(state, batch_models_save_dir +'/'+ str(epoch) + '.pt')

        #######################################################################
        # Do for all batches, including the first non incremental batch

        #computing number of exemplars
        m_nxt = int(math.ceil(K / (old_classes_number + new_classes_number)))


        new_train_dataset = ImagesListFileFolder(
            new_train_file_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize, ]), return_path = True)

        new_train_loader = torch.utils.data.DataLoader(
            new_train_dataset, batch_size=exemplars_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False)


        features_extractor = nn.Sequential(*list(best_model.children())[:-1])
        features_extractor.eval()

        if tc.is_available():
            features_extractor = features_extractor.cuda(gpu)
        else:
            print("GPU not available")

        #####################

        print('Training features extraction')
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
            outputs = best_model(inputs)
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


        if b == 1 :
            print('Constructing exemplar set,  Exemplars number per class = ' + str(m_nxt) + '...')
            compute_rebuffi_herding_faster(new_train_file_path, features_out_file, exemplars_output_dir, m_nxt, str(b+1)+'_old')

        else: #perform bft
            m_crt = int(math.ceil(K / old_classes_number))

            print('Constructing balanced set,  Exemplars number per class = ' + str(m_crt) + '...')


            compute_rebuffi_herding_faster(new_train_file_path, features_out_file, balanced_output_dir, m_crt, str(b)+'_new')

            # new_train_file_path = os.path.join(balanced_train_files_dir, 'K~' + str(K) + '/' + str(b) + '_new')
            new_train_file_path = os.path.join(balanced_output_dir, str(b) + '_new')
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

            train_loader = torch.utils.data.DataLoader(
                new_and_old_train_datasets, batch_size=new_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False)

            print('Switching to balanced fine tuning...')
            print('New train data loaded from ' + new_train_file_path)
            print("Old Training-set size = " + str(len(old_train_dataset)))
            print("New Training-set size = " + str(len(new_train_dataset)))
            print("Training-set size = " + str(len(new_and_old_train_datasets)))
            print("Validation-set size = " + str(len(val_dataset)))

            print('Adjusting lr..')
            for g in optimizer_ft.param_groups:
                g['lr'] = batch_lr  / 10.0


            for epoch in range(unbalanced_num_epochs, unbalanced_num_epochs+ balanced_num_epochs):
                #############################
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
                    if (i + 1) % iter_size == 0:
                        optimizer_ft.step()
                        optimizer_ft.zero_grad()

                scheduler.step(loss.cpu().data.numpy())

                # Model evaluation
                model_ft.eval()

                # Validation on both old and new data
                for data in val_loader:
                    inputs, labels = data
                    if tc.is_available():
                        inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
                    outputs = model_ft(Variable(inputs))
                    prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))

                if top1.avg > best_top1_v_acc:
                    best_top1_v_acc = top1.avg
                    best_top5_v_acc = top5.avg
                    best_model = copy.deepcopy(model_ft)
                    best_optimizer_ft = copy.deepcopy(optimizer_ft)
                    best_epoch = epoch

                # -------------------------------------------
                current_elapsed_time = time.time() - starting_time
                print('{:03}/{:03} | {} | Train : loss = {:.4f}  | Val : acc@1 = {}% ; acc@5 = {}%'.
                      format(epoch + 1, balanced_num_epochs + unbalanced_num_epochs,
                             timedelta(seconds=round(current_elapsed_time)),
                             running_loss / nb_batches, top1.avg, top5.avg))

                # Saving model
                if saving_intermediate_models and (epoch + 1) % 10 == 0:
                    state = {
                        'epoch': epoch,
                        'state_dict': model_ft.state_dict(),
                        'optimizer': optimizer_ft.state_dict(),
                        'best_v_acc': top5.avg
                    }

                    torch.save(state, batch_models_save_dir + '/' + str(epoch) + '.pt')

            print('Reducing exemplars...')
            reduce_exemplars(old_classes_number, new_classes_number, m_nxt, new_train_file_path, old_train_file_path, exemplars_output_dir, str(b+1)+'_old')



            print('Finished Training, elapsed training time : {}'.format(
                timedelta(seconds=round(time.time() - starting_time))))

            # Saving model
            print('Saving best model in ' + batch_models_save_dir + '.pt' + '...')
            state = {
                'epoch': epoch,
                'state_dict': best_model.state_dict(),
                'optimizer': best_optimizer_ft.state_dict()
            }
            torch.save(state, batch_models_save_dir + '.pt')
            print('Best top1 val accuracy = ' + str(best_top1_v_acc))
            print('Best top5 val accuracy = ' + str(best_top5_v_acc))

            top1_val_accuracies.append(best_top1_v_acc)
            top5_val_accuracies.append(best_top5_v_acc)


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
