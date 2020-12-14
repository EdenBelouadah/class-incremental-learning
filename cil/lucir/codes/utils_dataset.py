import argparse
import os
import shutil
import time
import numpy as np

#split trainset.imgs
def split_images_labels(imgs):

    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def split_images_labels_paths(imgs):
    images = []
    labels = []
    paths = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])
        paths.append(item[2])

    return np.array(images), np.array(labels), paths

#concatenate the paths with the true classes
def merge_images_labels(images, labels):
    images = list(images)
    labels = list(labels)
    assert(len(images)==len(labels))
    imgs = []
    for i in range(len(images)):
        item = (images[i], labels[i])
        imgs.append(item)
    
    return imgs


def save_protosets(current_eval_set, b, output_dir):
    ckp_name = os.path.join(output_dir, 'protoset_{}.lst'.format(b))
    protoset_file = open(ckp_name, 'w')
    for protoset in current_eval_set:
        path, class_ = protoset[0], int(protoset[1])
        protoset_file.write(path +' '+str(class_)+'\n')
    protoset_file.close()



