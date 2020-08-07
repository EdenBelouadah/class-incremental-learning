from __future__ import division
from torchvision import models
import torch.utils.data.distributed
import os, sys

if len(sys.argv) != 5:
    print('Arguments : models_load_path_prefix, S , P, destination_dir')
    sys.exit(-1)

models_load_path_prefix = sys.argv[1]
S = int(sys.argv[2])
P = int(sys.argv[3])
destination_dir = sys.argv[4]


if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)


for b in range(2, S+1):
    print('*' * 20)
    print('BATCH '+str(b))
    num_classes = b * P
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    model_load_path = models_load_path_prefix + str(b) + '.pt'
    print('Loading model from:' + model_load_path)
    state = torch.load(model_load_path, map_location = lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    model.eval()

    destination_path = os.path.join(destination_dir, 'batch_'+str(b))

    print('Saving stats in: '+ destination_path)
    # parameters
    parameters = [e.cpu() for e in list(model.fc.parameters())]

    torch.save(parameters, destination_path)