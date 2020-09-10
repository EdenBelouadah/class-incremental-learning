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
inc_models_load_path_prefix = cp['inc_models_load_path_prefix']
first_batch_model_load_path = cp['first_batch_model_load_path']
gpu = int(cp['gpu'])
images_list_dir = cp['images_list_dir']
destination_dir = cp['destination_dir']
normalization_dataset_name = cp['normalization_dataset_name']
P = int(cp['P'])
S = int(cp['S'])
datasets_mean_std_file_path = cp['datasets_mean_std_file_path']
fc_params_destination_path = cp['fc_params_destination_path']



#Catching warnings
with warnings.catch_warnings(record=True) as warn_list:

    dataset_mean, dataset_std = get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)

    print('normalization dataset name = ' + str(normalization_dataset_name))
    print('dataset mean = ' + str(dataset_mean))
    print('dataset std = ' + str(dataset_std))

    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    print("Number of workers = " + str(num_workers))
    print("Batch size = " + str(batch_size))
    print("Running on gpu " + str(gpu))

    data_types = ['train', 'test']

    for data_type in data_types:
        print('Features extraction for '+data_type+' dataset...')
        data_type_destination_dir = os.path.join(destination_dir, data_type)

        if not os.path.exists(data_type_destination_dir):
            os.makedirs(data_type_destination_dir)

        for b in range(1, S + 1):
            print('*' * 50)
            print('BATCH ' + str(b))
            print('*' * 50)

            if data_type == 'train':
                images_list = os.path.join(images_list_dir, 'separated', data_type, 'batch' + str(b))
            else:
                images_list = os.path.join(images_list_dir, 'accumulated', data_type, 'batch' + str(b))


            num_classes = P * b
            model = models.resnet18(pretrained=False, num_classes=num_classes)

            if b == 1 :
                model_load_path = first_batch_model_load_path
            else:
                model_load_path = inc_models_load_path_prefix + str(b) + '.pt'

            if not os.path.exists(model_load_path):
                print('No model found in ' + model_load_path)
                sys.exit(-1)

            print('Loading saved model from ' + model_load_path)

            state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state['state_dict'])

            features_extractor = nn.Sequential(*list(model.children())[:-1])
            model.eval()
            features_extractor.eval()


            # saving last layer weights
            if data_type == 'train':
                batch_fc_dest = os.path.join(fc_params_destination_path, 'b' + str(b) + '_weight_bias.pt')
                if not os.path.exists(fc_params_destination_path):
                    os.makedirs(fc_params_destination_path)
                print('Saving fc params in: ' + batch_fc_dest)
                # parameters
                parameters = [e.cpu() for e in list(model.fc.parameters())]
                torch.save(parameters, batch_fc_dest)


            if tc.is_available():
                model = model.cuda(gpu)
                features_extractor = features_extractor.cuda(gpu)
            else:
                print("GPU not available")
                sys.exit(-1)


            batch_destination_dir = os.path.join(data_type_destination_dir, 'batch' + str(b))

            if not os.path.exists(batch_destination_dir):
                os.makedirs(batch_destination_dir)


            dataset = ImagesListFileFolder(
                images_list, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]), return_path=True)

            print(data_type+"-set size = " + str(len(dataset)))

            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=False)

            print('Loading list file from ' + images_list)
            print('Destination directory ' + batch_destination_dir)
            print('Features/scores extraction')

            features_names = None  # np.empty([512, ])
            scores_names = None  # np.empty([512, ])
            file_names = None  # np.empty([1, ])

            i = 0  # beginning

            for data in loader:
                (inputs, labels), paths = data
                if tc.is_available():
                    inputs = inputs.cuda(gpu)
                # wrap it in Variable
                inputs = Variable(inputs)
                features = features_extractor(inputs)
                outputs = model(inputs)
                assert (outputs.shape[1] == num_classes)
                np_paths = np.array(paths)  # tuple -> numpy
                np_features = features.data.cpu().numpy()  # variable -> numpy
                np_features = np_features.reshape(np_features.shape[0], np_features.shape[1])
                np_outputs = outputs.data.cpu().numpy()  # variable -> numpy

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
            print('scores final shape = ' + str(scores_names.shape))
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