from __future__ import division
import re, shutil, os, sys
import numpy as np
import random
import math
random.seed(2017)

if len(sys.argv) != 5:
    print('Arguments : train_separated_batches_dir, K (memory_size), P (class_batch_size), destination_dir')
    sys.exit(-1)

train_batches_dir = sys.argv[1]
K = int(sys.argv[2])
P = int(sys.argv[3])
destination_dir = sys.argv[4]
destination_dir = os.path.join(destination_dir, 'K~'+str(K))
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)


def from_list_to_dict(images_list, shuffle=False):

    images_dict = {}
    for line in images_list:
        line = line.strip('\n').split()
        image_path, class_number = line[0], int(line[1])

        if str(class_number) not in images_dict.keys():
            images_dict[str(class_number)] = [image_path+' '+str(class_number)]
        else:
            images_dict[str(class_number)].append(image_path+' '+str(class_number))

    if shuffle:
        for key in images_dict.keys():
            random.shuffle(images_dict[key])

    return images_dict


#creating images dict
batches_files = [os.path.basename(e) for e in os.listdir(train_batches_dir)]
images_dict = {}
for batch_file in batches_files:
    train_batch_path = os.path.join(train_batches_dir, batch_file)
    train_batch = open(train_batch_path, 'r').readlines()
    train_batch_dict = from_list_to_dict(train_batch, shuffle=True)
    for key in train_batch_dict.keys():
        images_dict[key] = train_batch_dict[key]


total_classes_number = len(images_dict.keys())
print('total classes number = '+str(total_classes_number))
batches_number  = int(total_classes_number / P)
first_batch_P = P + total_classes_number % P
print('first batch classes number P = '+str(first_batch_P))


taken_images = []

#first batch images:
first_batch_taken_images = []
imgs_nbr_per_class = int(math.ceil(K / (first_batch_P + P * (2 - 2))))
for class_number in range(first_batch_P):
    first_batch_taken_images.extend(images_dict[str(class_number)][:imgs_nbr_per_class])
taken_images.append(from_list_to_dict(first_batch_taken_images, shuffle=False))


for b in range(2, batches_number+1):
    old_data = []
    new_data = []
    current_batch_taken_images = []

    # new data
    imgs_nbr_per_class = int(math.ceil(K / (first_batch_P + P * (b - 2))))
    print('Images number per class = '+str(imgs_nbr_per_class))
    for new_class_number in range(first_batch_P + (b - 2) * P, first_batch_P + (b - 1) * P):
        new_data.extend(images_dict[str(new_class_number)])
        current_batch_taken_images.extend(images_dict[str(new_class_number)])

    taken_images.append(from_list_to_dict(new_data, shuffle=False))

    new_data_size = len(new_data)
    print('new data size for batch '+str(b)+' : '+str(new_data_size))
    new_data_output_path = os.path.join(destination_dir, str(b)+'_new')
    new_data_output_file = open(new_data_output_path, 'w')
    for data in new_data:
        new_data_output_file.write(data+'\n')
    new_data_output_file.close()


    #old data
    for s in range(1, b):
        for key in taken_images[s-1].keys():
            taken_images[s-1][key] = taken_images[s-1][key][:imgs_nbr_per_class]
            old_data.extend(taken_images[s-1][key])


    old_data_size = len(old_data)
    print('old data size for batch ' + str(b) + ' : ' + str(old_data_size))
    print('full data size for batch ' + str(b) + ' : ' + str(old_data_size+new_data_size))
    print('-----------------------')
    old_data_output_path = os.path.join(destination_dir, str(b) + '_old')
    old_data_output_file = open(old_data_output_path, 'w')
    for data in old_data:
        old_data_output_file.write(data+'\n')
    old_data_output_file.close()

    #old and new
    all_data_output_path = os.path.join(destination_dir, str(b) + '_full')
    all_data_output_file = open(all_data_output_path, 'w')
    for data in old_data:
        all_data_output_file.write(data + '\n')

    for data in new_data:
        all_data_output_file.write(data + '\n')

    all_data_output_file.close()
