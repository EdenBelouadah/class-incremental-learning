import tensorflow as tf
import numpy as np
import os
import scipy.io
import sys
try:
    import cPickle
except:
    import _pickle as cPickle


def read_data(files_from_cl):
    image_list = np.array([e.split()[0] for e in files_from_cl])
    labels_list = np.array([e.split()[1] for e in files_from_cl])
    
    assert(len(image_list) == len(labels_list))
    images             = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels             = tf.convert_to_tensor(labels_list, dtype=tf.int32)
    input_queue        = tf.train.slice_input_producer([images, labels], shuffle=True, capacity=2000)
    image_file_content = tf.read_file(input_queue[0])
    label              = input_queue[1]
    image              = tf.image.resize_images(tf.image.decode_jpeg(image_file_content, channels=3), [256, 256])
    image              = tf.random_crop(image, [224, 224, 3])
    image              = tf.image.random_flip_left_right(image)
    
    return image, label

def read_data_test(files_from_cl):
    image_list = np.array([e.split()[0] for e in files_from_cl])
    labels_list = np.array([e.split()[1] for e in files_from_cl])
    files_list = files_from_cl  # or put only the base name?
    
    assert(len(image_list) == len(labels_list))
    images              = tf.convert_to_tensor(image_list, dtype=tf.string)
    files               = tf.convert_to_tensor(files_list, dtype=tf.string)
    labels              = tf.convert_to_tensor(labels_list, dtype=tf.int32)
    input_queue         = tf.train.slice_input_producer([images, labels,files], shuffle=False, capacity=2000)
    image_file_content  = tf.read_file(input_queue[0])
    label               = input_queue[1]
    file_string         = input_queue[2]
    image               = tf.image.resize_images(tf.image.decode_jpeg(image_file_content, channels=3), [224, 224])
    
    return image, label,file_string

def prepare_files(train_images_path, val_images_path, nb_groups, nb_cl):
    files_train = []
    files_valid = []

    train_list = open(train_images_path, 'r').readlines()
    val_list = open(val_images_path, 'r').readlines()

    total_train_images_number = len(train_list)
    total_val_images_number = len(val_list)

    train_dict = {}
    for line in train_list:
        line = line.strip()
        img_path_class = line.split()
        image_path, image_class = img_path_class[0], int(img_path_class[1])
        if image_class not in train_dict.keys():
            train_dict[image_class] = [line]
        else:
            train_dict[image_class].append(line)

    for num in range(nb_groups):
        group_images = []
        for key in train_dict.keys():
            if int(key) >= num * nb_cl and int(key) < (num + 1) * nb_cl:
                group_images.extend(train_dict[int(key)])
        files_train.append(group_images)

    assert (len(files_train) == nb_groups)
    train_images_number = 0
    for group_images in files_train:
        train_images_number += len(group_images)

    assert (train_images_number == total_train_images_number)

    # Same for validation
    val_dict = {}
    for line in val_list:
        line = line.strip()
        img_path_class = line.split()
        image_path, image_class = img_path_class[0], int(img_path_class[1])
        if image_class not in val_dict.keys():
            val_dict[image_class] = [line]
        else:
            val_dict[image_class].append(line)

    for num in range(nb_groups):
        group_images = []
        for key in val_dict.keys():
            if int(key) >= num * nb_cl and int(key) < (num + 1) * nb_cl:
                group_images.extend(val_dict[int(key)])
        files_valid.append(group_images)

    assert (len(files_valid) == nb_groups)
    val_images_number = 0
    for group_images in files_valid:
        val_images_number += len(group_images)

    assert (val_images_number == total_val_images_number)

    return files_train, files_valid
