from __future__ import division
from torchvision import models
import torch.utils.data.distributed
import os, sys

if len(sys.argv) != 4:
    print('Arguments : models_load_path_prefix, S , destination_dir')
    sys.exit(-1)

models_load_path_prefix = sys.argv[1]
S = int(sys.argv[2])
destination_dir = sys.argv[3]


if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)


for b in range(2, S+1):
    print('*' * 20)
    print('BATCH '+str(b))
    model_load_path = models_load_path_prefix + str(b - 1) + '.pth'

    if not os.path.exists(model_load_path):
        print('No model found in ' + model_load_path)
        continue

    print('Loading saved model from ' + model_load_path)

    model = torch.load(model_load_path)
    model.eval()

    destination_path = os.path.join(destination_dir, 'batch_'+str(b))

    print(model.fc.out_features)
    print('Saving stats in: '+ destination_path)
    # parameters
    parameters = [e.cpu() for e in list(model.fc.parameters())]


    # print(parameters)
    #
    # # print(model.fc.weight.shape)
    # # print(model.fc.bias.shape)
    # for parameter in model.fc.parameters():
    #     print(parameter)

    # print(list(model.fc.parameters()))
    np_crt_bias = parameters[0].detach().numpy()
    np_crt_weights = parameters[1].detach().numpy()
    # print(np_crt_bias.shape)
    # print(np_crt_weights.shape)
    # sys.exit(-1)


    torch.save(parameters, destination_path)