from __future__ import division
from torchvision import models
import torch.utils.data.distributed
import os, sys

if len(sys.argv) != 5:
    print('Arguments : model_load_path, used_model_num_classes, batch_number , destination_dir')
    sys.exit(-1)

model_load_path = sys.argv[1]
used_model_num_classes = int(sys.argv[2])
batch_number = sys.argv[3]
destination_dir = sys.argv[4]


if not os.path.exists(model_load_path):
    print('No model found in the specified path')
    sys.exit(-1)



model = models.resnet18(pretrained=False, num_classes=used_model_num_classes)

print('Loading saved model from ' + model_load_path)
state = torch.load(model_load_path, map_location = lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])

model.eval()



print('Saving statistics...')
# parameters
parameters = [e.cpu() for e in list(model.fc.parameters())]


if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

torch.save(parameters, os.path.join(destination_dir, 'batch_'+batch_number))