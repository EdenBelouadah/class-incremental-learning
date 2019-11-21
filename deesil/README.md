# DeeSIL : Deep-Shallow-Incremental-Learning
## Abstract
Incremental Learning (IL) is an interesting AI problem when the algorithm is assumed to work on a budget. This is especially true when IL is modeled using a deep learning approach, where two complex challenges arise due to limited memory, which induces catastrophic forgetting and delays related to the retraining needed in order to incorporate new classes. 
Here we introduce DeeSIL, an adaptation of a known transfer learning scheme that combines a fixed deep representation used as feature extractor and learning independent shallow classifiers to increase recognition capacity. This scheme tackles the two aforementioned challenges since it works well with a limited memory budget and each new concept can be added within a minute. Moreover, since no deep retraining is needed when the model is incremented, DeeSIL can integrate larger amounts of initial data that provide more transferable features. 
Performance is evaluated on ImageNet LSVRC 2012 against three state of the art algorithms. Results show that, at scale, DeeSIL performance is 23 and 33 points higher than the best baseline when using the same and more initial data respectively.   

## Paper
Link to the related paper: [here](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Belouadah_DeeSIL_Deep-Shallow_Incremental_Learning._ECCVW_2018_paper.pdf)  
    

To cite this work:  


    @article{deesil-eccv2018,
       title = {{DeeSIL:} Deep-Shallow Incremental Learning},
       author={Belouadah, Eden and Popescu, Adrian},,
       journal={TaskCV Workshop @ ECCV 2018.},
       year={2018}
    }


## Configuration files

This is the explanation of the different parameters:

### train_dfe.py

* normalization_dataset_name : dataset name used for images normalization (here it's ilsvrc_batch1)
* datasets_mean_std_file_path : path to the file containing the mean/std of dataset images (cf. data/datasets_mean_std.txt)
* algo_name : name that you give to the current execution, a folder will be created using this name to store intermediate models.
* num_workers : number of CPU cores used to load batches
* gpu : number of the used gpu, set to 0 if you only have one gpu in your machine
* old_batch_size : batch size for training examples  
* new_batch_size : mini batch size, useful for accumulating batches before doing the backpropagation.  
* val_batch_size : batch size for validation examples
* num_epochs : total number of epochs
* lr : learning rate
* lr_decay : learning rate decay
* patience : number of epochs needed before adapting the learning rate when the error plateaus  
* train_file_path : path to training images list
* val_file_path : path to validation images list
* model_load_path : path to the model used to initialize the DFE training, (here it's None)
* models_save_dir : path to the directory where we save the final model 
* saving_intermediate_models : True if we want to save the intermediate models, False otherwise. 
* intermediate_models_save_dir : path to the directory where we save the intermediate models, useful only if 'saving_new_model = True'.  

### features_extraction.py

* features_size : features vector size, used only for sanity check
* used_model_num_classes : the number of classes used to train the DFE
* images_lists_folder : path of the directory containing one images list file per class
* destination_dir : path of the output features directory

## Minimal working example : 
1. The toy example is provided for a subset of 10 classes from ILSVRC, the first 5 classes are used to train the DFE and the other 5 to perform incremental learning. The memory size here is = 200. The example can be applied on any other dataset with the same way if you respect the data format. 
2. If your dataset used to train the DFE is different from the first batch of ILSVRC, you need to compute the images mean/std used for normalization of your dataset using the traing images and add it to the [dfe_training.py] code.
3. Feel free to send an email to  {eden.belouadah, adrian.popescu}@cea.fr if there is any issue with the code.

### 0/ Requierements and data
* Python 2.7
* Pytorch 1.0.0
* Numpy 1.13.0
* SkLearn 0.19.1

* DFE data: two text files for the first batch of classes, one containing paths to training images and one containing paths to validation images (cf. './data/train.lst' and './data/val.lst').

* Data for features extraction: a directory containing one images paths file per class (cf. './data/lists_batch1/', './data/lists_train/', './data/list_val/')

* Classes lists in order (cf. './data/train_classes.lst' and './data/batch1_classes.lst'). 


### 1/ Train the Deep Feature Extractor (DFE)
After filling the [dfe_training.py] section in './config.cf', run:
```
python ./code/dfe_training.py ./config.cf
```

You should save the model after training the DFE.

### 2/ Features extraction for the first batch then for all classes
After filling the [features_extraction.py] section in config.cf, run:
```
python ./code/features_extraction.py ./config.cf
```

For the first batch of classes, precise the following paths:

* images_lists_folder = ./data/lists_batch1
* destination_dir = ./output/features_batch1

For all classes:

* images_lists_folder = ./data/lists_train
* destination_dir = ./output/features_train


### 3/ Features L2-normalization and SVMLib format for the two batch of classes
For the first batch:
```
python ./code/feat_L2_normalization.py ./output/features_batch1/features +1 ./output/features_batch1/features_L2
```

For all batches:
```
python ./code/feat_L2_normalization.py ./output/features_train/features +1 ./output/features_train/features_L2
```

### 4/ Create negative features and associated list for all batches:
```
python ./code/create_random_negatives.py ./data/train_classes.lst ./output/features_train/features_L2 ./data/lists_train 200 5 2 ./output/random_negatives
```

In this example, the codes creates negatives for 2 batches of classes each containing 5 classes.


### 5/ Calibrate the SVMs for the first batch - i.e. get the optimal regularization parameter

Try with REGUL = {0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000}

```
python ./code/calibrate_svms.py ./data/batch1_classes.lst ./output/features_batch1/features_L2 ./output/random_negatives_mem_200/batch_1/features ./output/random_negatives_mem_200/batch_1/list ./output/tmp 10 ./output/models_calibration ./output/features_batch1/features_L2/val.txt ./data/lists_batch1/val.txt
```

Save the optimal REGUL and use it for the incremental batch.

### 6/ Use the models with the optimal REGUL

```
python ./code/train_svms.py 5 ./data/train_classes.lst ./output/features_train/features_L2 10 ./output/random_negatives_mem_200 ./output/models_full 0 10
```


### 7/ Extract the validation features
Use the following paths :

* images_lists_folder = ./data/list_val
* destination_dir = ./output/features_val

and run:

```
python ./code/features_extraction.py ./config.cf
```

### 8/ Normalize the validation features
```
python ./code/feat_L2_normalization.py ./output/features_val/features/ +1 ./output/features_val/features_L2
```

### 9/ Compute the DeeSIL predictions when using only the memory
```
python ./code/compute_predictions.py ./data/train_classes.lst ./output/models_full_C_10 ./output/features_val/features_L2/val.lst ./output/results/top_10.txt
```
### 10/ Evaluate accuracy @1 and @5 for the test set
```
python ./code/eval.py ./data/train_classes.lst ./data/list_val/val.lst 5 ./output/results/top_10.txt
```
