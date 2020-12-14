import numpy as np 
import os
import tensorflow as tf
# if type(tf.contrib) != type(tf): tf.contrib._warning = None
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARNING)
# tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(1)
# import logging
# logging.disable(logging.WARNING)



###################### Load the data #######################


def load_data(fpath):
    lines = open(fpath)
    data = []
    label = []
    for line in lines:
      arr = line.strip().split()
      data.append(arr[0])
      label.append(arr[1])
    # labels = np.asarray(d[label_key], np.int8)
    return data, label

def load_data(fpath, order):
    lines = open(fpath)
    data = []
    label = []


    for line in lines:
      arr = line.strip().split()
      data.append(arr[0])
      label.append(arr[1])

    # labels = np.asarray(d[label_key], np.int8)
    ## map to new labels
    mapping = {}
    for i, j in enumerate(order):
      mapping[j] = i
    labels = [mapping[int(label_)] for label_ in label]
    
    return data, labels
    # return data, label

def prepare_validation(x_train, y_train, x_test, y_test, nb_groups, nb_cl, nb_val):
    x_train_new = []
    y_train_new = []
    x_val_new = []
    y_val_new = []
    x_test_new = []
    y_test_new = []

    for _ in range(nb_groups):
      x_train_new.append([])
      y_train_new.append([])
      x_test_new.append([])
      y_test_new.append([])
    
    for _ in range(nb_groups*nb_cl):
      x_val_new.append([])
      y_val_new.append([])
    # get max val, the results for the first item
    y_train = np.asarray(y_train, np.int16)
    y_test  = np.asarray(y_test, np.int16)
    for i in range(nb_groups):
      for j in range(nb_cl):
        tmp_ind=np.where(y_train == nb_cl * i + j)[0]
        # print (len(tmp_ind))
        np.random.shuffle(tmp_ind)
        # print (tmp_ind[0:len(tmp_ind)-nb_val])
        
        # print ([x_train[k] for k in tmp_ind[0:len(tmp_ind)-nb_val]] )
        x_train_new[i].extend( [x_train[k] for k in tmp_ind[0:len(tmp_ind)-nb_val]])
        y_train_new[i].extend(y_train[tmp_ind[0:len(tmp_ind)-nb_val]].tolist())

        # x_val_new[i*nb_cl+j].extend(x_train[tmp_ind[len(tmp_ind)-nb_val:]])
        x_val_new[i*nb_cl+j].extend([x_train[k] for k in tmp_ind[len(tmp_ind)-nb_val:]])
        y_val_new[i*nb_cl+j].extend(y_train[tmp_ind[len(tmp_ind)-nb_val:]].tolist())

        tmp_ind = np.where(y_test == nb_cl * i + j)[0]
        # x_test_new[i].extend(x_test[tmp_ind])
        x_test_new[i].extend([x_test[k] for k in tmp_ind])
        y_test_new[i].extend(y_test[tmp_ind].tolist())

    return x_train_new, y_train_new, x_val_new, y_val_new, x_test_new, y_test_new

