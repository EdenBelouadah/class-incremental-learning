from __future__ import division
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.cuda as tc
import torch.utils.data.distributed
from configparser import ConfigParser
import sys, os, warnings
import numpy as np
from MyImageFolder import ImagesListFileFolder
from Utils import DataUtils

if len(sys.argv) != 2:
    print('Arguments : general_config')
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
batch_size = int(cp['batch_size'])
model_load_path = cp['model_load_path']
gpu = int(cp['gpu'])
features_size = int(cp['features_size'])
used_model_num_classes = int(cp['used_model_num_classes'])
images_lists_folder = cp['images_lists_folder']
destination_dir = cp['destination_dir']
normalization_dataset_name = cp['normalization_dataset_name']
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']


if not os.path.exists(model_load_path):
    print('No model found in the specified path')
    sys.exit(-1)


#Catching warnings
with warnings.catch_warnings(record=True) as warn_list:

    #Data loading
    utils = DataUtils()
    dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name,
                                                           datasets_mean_std_file_path)

    print('normalization dataset name = ' + str(normalization_dataset_name))
    print('dataset mean = ' + str(dataset_mean))
    print('dataset std = ' + str(dataset_std))

    # Data loading code
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    print("Number of workers = " + str(num_workers))
    print("Batch size = " + str(batch_size))

    model = models.resnet18(pretrained=False, num_classes=used_model_num_classes)

    print('Loading model from ' + model_load_path)
    state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])

    features_extractor = nn.Sequential(*list(model.children())[:-1])
    features_extractor.eval()

    if tc.is_available():
        print("Running on gpu " + str(gpu))
        features_extractor = features_extractor.cuda(gpu)
    else:
        print("GPU not available")


    print('------------------------------------------------')

    images_lists = os.listdir(images_lists_folder)
    for images_list in images_lists:
        images_list_path = os.path.join(images_lists_folder,images_list)
        print('File : '+images_list_path)

        train_dataset = ImagesListFileFolder(
            images_list_path, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize, ]), return_path=True)


        print("Input-set size = " + str(len(train_dataset)))


        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)

        print("Number of batches in Input-set = " + str(len(train_loader)))



        print('Features extraction')

        full_features = None
        full_file_paths = None

        nb_batches = 0
        for data in train_loader:
            nb_batches += 1
            if nb_batches%1000 == 0:
                print('batch n ' + str(nb_batches) +'/'+str(len(train_loader)))
            (inputs, labels), paths = data
            if tc.is_available():
                inputs = inputs.cuda(gpu)
            # wrap it in Variable
            inputs = Variable(inputs)
            features = features_extractor(inputs)
            assert (features.shape[1] == features_size)
            np_paths = np.array(paths) #tuple -> numpy
            np_features = features.data.cpu().numpy() #variable -> numpy
            np_features = np_features.reshape(np_features.shape[0], np_features.shape[1])

            if full_features is None :
                full_file_paths = np_paths
                full_features =  np_features
            else:
                full_file_paths = np.append(full_file_paths, np_paths)
                full_features = np.vstack((full_features, np_features))

        print('full features shape = ' + str(full_features.shape))
        print('full file paths shape =' + str(len(full_file_paths)))

        ## creating dict
        features_dict = {}
        for i in range(len(full_file_paths)):
            full_file_paths[i] = full_file_paths[i].split('/')[-1]
            if full_file_paths[i] in features_dict: #check if there's redundancy in input file
                print(str(full_file_paths[i]) + ' exists as '+str(features_dict[full_file_paths[i]]))
            features_dict[full_file_paths[i]] = full_features[i]

        print('features dict length = '+ str(len(features_dict.keys())))

    #########################################################

        images_file = open(images_list_path, 'r').readlines()
        print('images file length = ' + str(len(images_file)))
        file_destination_dir = os.path.join(destination_dir ,'features')
        if not os.path.exists(file_destination_dir):
            os.mkdir(file_destination_dir)
        features_out_path = os.path.join(file_destination_dir,os.path.basename(images_list_path) )
        features_out = open(features_out_path, 'w')
        print('saving features in : '+features_out_path)

        for image_file in images_file:
            image_file = image_file.strip('\n')
            image_file = image_file.split()[0].split('/')[-1]
            if '.jpg' in image_file or '.jpeg' in image_file  or '.JPEG' in image_file :
                features_out.write(str(' '.join([str(e) for e in list(features_dict[image_file])])) +'\n')
            else:
                print('image file = ' + str(image_file))
        features_out.close()
        print('------------------------------------------------')
