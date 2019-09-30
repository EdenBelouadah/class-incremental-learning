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
images_list = cp['images_list']
destination_dir = cp['destination_dir']
normalization_dataset_name = cp['normalization_dataset_name']
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']

if not os.path.exists(model_load_path):
    print('No model found in the specified path')
    sys.exit(-1)

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

#Catching warnings
with warnings.catch_warnings(record=True) as warn_list:
    utils = DataUtils()
    dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)

    print('normalization dataset name = ' + str(normalization_dataset_name))
    print('dataset mean = ' + str(dataset_mean))
    print('dataset std = ' + str(dataset_std))

    # Data loading code
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    train_dataset = ImagesListFileFolder(
        images_list, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize, ]), return_path=True)

    print("Number of workers = " + str(num_workers))
    print("Batch size = " + str(batch_size))
    print("Training-set size = " + str(len(train_dataset)))


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    print("Number of batches in Training-set = " + str(len(train_loader)))


    model = models.resnet18(pretrained=False, num_classes=used_model_num_classes)
    # model.fc = nn.Linear(512, 100)

    print('Loading saved model from ' + model_load_path)
    print('Loading list file from ' + images_list)
    print('Destination directory ' + destination_dir)
    state = torch.load(model_load_path, map_location = lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])


    features_extractor = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    features_extractor.eval()


    if tc.is_available():
        print("Running on gpu " + str(gpu))
        model = model.cuda(gpu)
        features_extractor = features_extractor.cuda(gpu)
    else:
        print("GPU not available")

    print('Features/scores extraction')

    features_names = None #np.empty([512, ])
    scores_names = None #np.empty([512, ])
    file_names = None #np.empty([1, ])

    i = 0 #beginning

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
        outputs = model(inputs)
        assert (features.shape[1] == features_size)
        assert (outputs.shape[1] == used_model_num_classes)
        np_paths = np.array(paths) #tuple -> numpy
        np_features = features.data.cpu().numpy() #variable -> numpy
        np_features = np_features.reshape(np_features.shape[0], np_features.shape[1])
        np_outputs = outputs.data.cpu().numpy()  # variable -> numpy


        if i == 0 :
            file_names = np_paths
            features_names =  np_features
            scores_names = np_outputs
            i = 1
        else:
            file_names = np.append(file_names, np_paths)
            features_names = np.vstack((features_names, np_features))
            scores_names = np.vstack((scores_names, np_outputs))

    print('features final shape = ' + str(features_names.shape))
    print('scores final shape = ' + str(scores_names.shape))
    print('file names final shape =' + str(len(file_names)))

    ## creating dict
    features_dict = {}
    scores_dict = {}

    for i in range(len(file_names)):
        if file_names[i] in features_dict: #voir s'il y a une image repetee deux fois dans le fichier d'entree
            print(str(file_names[i]) + ' exists as '+str(features_dict[file_names[i]]))
        features_dict[file_names[i]] = features_names[i]
        scores_dict[file_names[i]] = scores_names[i]

    print('len features dict = '+ str(len(features_dict.keys())))
    print('len scores dict = '+ str(len(scores_dict.keys())))

#########################################################

    print('saving features/scores')


    images_files = open(images_list, 'r').readlines()
    print('image file len = ' + str(len(images_files)))
    features_out = open(os.path.join(destination_dir ,'features'), 'w')
    scores_out = open(os.path.join(destination_dir ,'scores'), 'w')
    for image_file in images_files:
        image_file = image_file.strip('\n')
        image_file = image_file.split()[0]
        if '.jpg' in image_file or '.jpeg' in image_file  or '.JPEG' in image_file or '.png' in image_file or '.PNG' in image_file:
            features_out.write(str(' '.join([str(e) for e in list(features_dict[image_file])])) +'\n')
            scores_out.write(str(' '.join([str(e) for e in list(scores_dict[image_file])])) +'\n')
        else:
            print('image file = ' + str(image_file))
    features_out.close()
    scores_out.close()


