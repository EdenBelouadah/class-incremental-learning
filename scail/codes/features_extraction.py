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

utils = DataUtils()


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
models_load_path_prefix = cp['models_load_path_prefix']
gpu = int(cp['gpu'])
val_images_list_dir = cp['val_images_list_dir']
destination_dir = cp['destination_dir']
normalization_dataset_name = cp['normalization_dataset_name']
first_batch_number = int(cp['first_batch_number'])
last_batch_number = int(cp['last_batch_number'])
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']
first_model_load_path = cp['first_model_load_path']


#Catching warnings
with warnings.catch_warnings(record=True) as warn_list:

    dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)

    print('normalization dataset name = ' + str(normalization_dataset_name))
    print('dataset mean = ' + str(dataset_mean))
    print('dataset std = ' + str(dataset_std))
    print('first batch number = ' + str(first_batch_number))
    print('last batch number = ' + str(last_batch_number))

    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    print("Number of workers = " + str(num_workers))
    print("Batch size = " + str(batch_size))
    print("Running on gpu " + str(gpu))

    for b in range (first_batch_number, last_batch_number+1):
        print('*'*50)
        print('BATCH '+str(b))
        print('*'*50)

        print('-------> Val data')
        images_list = os.path.join(val_images_list_dir, 'batch' + str(b))
        batch_destination_dir = os.path.join(destination_dir, 'batch_'+str(b))
        if not os.path.exists(batch_destination_dir):
            os.makedirs(batch_destination_dir)

        val_dataset = ImagesListFileFolder(
            images_list, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize, ]), return_path=True)

        print("Val-set size = " + str(len(val_dataset)))


        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)


        print('Loading list file from ' + images_list)
        print('Destination directory ' + batch_destination_dir)
        print('Features/scores extraction')

        num_classes = len(val_dataset.classes)
        model = models.resnet18(pretrained=False, num_classes=num_classes)

        if b == 1 :
            model_load_path = first_model_load_path
        else: model_load_path = models_load_path_prefix + str(b) + '.pt'

        if not os.path.exists(model_load_path):
            print('No model found in ' + model_load_path)
            continue

        print('Loading saved model from ' + model_load_path)

        state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state['state_dict'])

        features_extractor = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        features_extractor.eval()

        if tc.is_available():
            model = model.cuda(gpu)
            features_extractor = features_extractor.cuda(gpu)
        else:
            print("GPU not available")
            sys.exit(-1)


        full_features = None 
        full_scores = None
        full_file_paths = None 
        
        i = 0 #beginning

        for data in val_loader:
            (inputs, labels), paths = data
            if tc.is_available():
                inputs = inputs.cuda(gpu)
            # wrap it in Variable
            inputs = Variable(inputs)
            features = features_extractor(inputs)
            outputs = model(inputs)
            np_paths = np.array(paths) #tuple -> numpy
            np_features = features.data.cpu().numpy() #variable -> numpy
            np_features = np_features.reshape(np_features.shape[0], np_features.shape[1])
            np_outputs = outputs.data.cpu().numpy()  # variable -> numpy

            if i == 0 :
                full_file_paths = np_paths
                full_features =  np_features
                full_scores = np_outputs
                i = 1
            else:
                full_file_paths = np.append(full_file_paths, np_paths)
                full_features = np.vstack((full_features, np_features))
                full_scores = np.vstack((full_scores, np_outputs))

        print('features final shape = ' + str(full_features.shape))
        print('scores final shape = ' + str(full_scores.shape))
        print('file names final shape =' + str(len(full_file_paths)))

        ## creating dict
        features_dict = {}
        scores_dict = {}

        for i in range(len(full_file_paths)):
            if full_file_paths[i] in features_dict:
                print(str(full_file_paths[i]) + ' exists as '+str(features_dict[full_file_paths[i]]))
            features_dict[full_file_paths[i]] = full_features[i]
            scores_dict[full_file_paths[i]] = full_scores[i]

        print('len features dict = '+ str(len(features_dict.keys())))
        print('len scores dict = '+ str(len(scores_dict.keys())))

    #########################################################

        print('saving features/scores')
        images_files = open(images_list, 'r').readlines()
        print('image file len = ' + str(len(images_files)))
        features_out = open(os.path.join(batch_destination_dir, 'features'), 'w')
        scores_out = open(os.path.join(batch_destination_dir, 'scores'), 'w')
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
