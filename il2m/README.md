# IL2M-Class-Incremental-Learning-with-Dual-Memory
## Abstract
This paper presents a class incremental learning (IL) method which exploits fine tuning and a dual memory to reduce the negative effect of catastrophic forgetting in image recognition. 

First, we simplify the current fine tuning based approaches which use a combination of classification and distillation losses to compensate for the limited availability of past data. We find that the distillation term actually hurts performance when a memory is allowed. Then, we modify the usual class IL memory component. 

Similar to existing works, a first memory stores exemplar images of past classes.
A second memory is introduced here to store past class statistics obtained when they were initially learned. 
The intuition here is that classes are best modeled when all their data are available and that their initial statistics are useful across different incremental states. 

A prediction bias towards newly learned classes appears during inference because the dataset is imbalanced in their favor.
The challenge is to make predictions of new and past classes more comparable. To do this, scores of past classes are rectified by leveraging contents from both memories.

The method has negligible added cost, both in terms of memory and of inference complexity.
Experiments with three large public datasets show that the proposed approach is more effective than a range of competitive state-of-the-art methods. 
## Paper
Link to the related paper:

(todo)

To cite this work:

(todo)

## Data
Data used in the experiments is available here : https://drive.google.com/drive/folders/1kgoB0Oxb9Wv2wSWFT5Yf7IoKXR3gAL_3?usp=sharing

## Configuration files

This is the explanation of the different parameters:

0. #### scratch.py
   * old_train_file_path: path to file containing old train image paths.
   * new_tr

1. #### ft.py
   * number: batch number [2 - 10]. Useful to update the learning rate.
   * old_train_file_path: path to file containing old train image paths.
   * new_train_file_path: path to file containing new train image paths.
   * old_val_file_path: path to file containing old val data image paths.
   * new_val_file_path: path to file containing new val data image paths.

2. #### scores_extraction.py
   * images_list: path to file containing list of images that we want to extract scores for.

3. #### fixed_representation_validation.py
   * models_load_path: path to the directory containing the models learned with fixed representation.
   * classes_batch_number: here we use 5 classes in each batch. In the paper we use 100.
   * separated_val_files_dir: path to the directory containing one file per batch, each file containing the list of validation data of this batch only.

## Minimal working example

1. ### Training from scratch

* #### ![Full](https://latex.codecogs.com/gif.latex?Full)  
Use the code 'dfe_training.py' and the configuration file 'config.cf' from this repository: https://github.com/EdenBelouadah/DeeSIL-Deep-Shallow-Incremental-Learning

```
python /home/eden/work/codes/DeeSIL-Deep-Shallow-Incremental-Learning/code/dfe_training.py /home/eden/work/codes/DeeSIL-Deep-Shallow-Incremental-Learning/config.cf
```
More details are given in the 'README.md' file of the other repository.

2. ### IL with fixed_size memory
* #### ![DeeSIL](https://latex.codecogs.com/gif.latex?DeeSIL)  
See the 'README.md' file of the same given repository for Full.

* #### ![iFT](https://latex.codecogs.com/gif.latex?iFT) & ![bFT](https://latex.codecogs.com/gif.latex?bFT) :
After filling the [fine_tuning.py] section in the configuration file 'fine_tuning/config.cf', run:

```
python /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fine_tuning/fine_tuning.py /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fine_tuning/config.cf
```

You should provide one configuration file for each incremental state. The given example in the configuration file is for unbalanced situation, you only need to change the data paths if you want to run for balanced situation.

* #### ![iFT_mc](https://latex.codecogs.com/gif.latex?iFT_%7Bmc%7D)
Once you finish with iFT and save the 9 incremental models, you need to extract features of both train and validation data and this for old and new data. To do this, fill the [scores_extraction.py] section in the 'fine_tuning/config.cf' file and run : 
```
python /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fine_tuning/scores_extraction.py /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fine_tuning/config.cf
```
For validation data, you should provide the accumulated validation images list file. For training data, you separate the old and new data in two files. In other words, you need to precise the following parameters in the configuration file:

   * Old training data:
       * images_list = /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/data/unbalanced/K~200/2_old
       * destination_dir = /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/output/train/batch2_old


   * New training data:
       * images_list = /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/data/unbalanced/K~200/2_new
       * destination_dir = /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/output/train/batch2_new
  
   * Validation data:
     * images_list = /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/data/accumulated/val/batch2
     * destination_dir = /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/output/val/batch2

To finally run the calibration , run:
```
python /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fine_tuning/mean_calibration.py /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/data/unbalanced/train/K~200/ /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/data/accumulated/val/ /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/output/ 2 2 5
```

* #### ![fFR](https://latex.codecogs.com/gif.latex?fFR)   & ![bFR](https://latex.codecogs.com/gif.latex?bFR) : 
For training, fill the [fixed_representation_training.py] section in 'fixed_representation/config.cf' file and run:
```
python /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fixed_representation/fixed_representation_training.py /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fixed_representation/config.cf
```
The example given in the configuration file is for fFR, you only need to change the data paths for bFR. 

You should provide one configuration file for each incremental state. Once you finish with fixed_representation and save the 10 models, name them 'b1.pt' -> 'b10.pt' and put them in the same directory. Fill the [fixed_representation_validation.py] section in 'fixed_representation/config.cf' file and run:

```
python /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fixed_representation/fixed_representation_validation.py /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fixed_representation/config.cf
```

* #### ![iFT_iso](https://latex.codecogs.com/gif.latex?iFT_%7Biso%7D) : 

3. ### IL without memory:

* #### ![iFT](https://latex.codecogs.com/gif.latex?iFT)
After filling the [fine_tuning.py] section in the configuration file 'fine_tuning/config.cf', run:

```
python /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fine_tuning/no_memory_fine_tuning.py /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fine_tuning/config.cf
```

You should provide one configuration file for each incremental state.
* #### ![DeeSIL](https://latex.codecogs.com/gif.latex?DeeSIL)  
* #### ![fFR](https://latex.codecogs.com/gif.latex?fFR)  

For training, fill the [no_memory_fixed_representation_training.py] section in 'fixed_representation/config.cf' file and run:
```
python /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fixed_representation/no_memory_fixed_representation_training.py /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fixed_representation/config.cf
```

You should provide one configuration file for each incremental state. Once you finish with fixed_representation and save the 10 models, name them 'b1.pt' -> 'b10.pt' and put them in the same directory. Fill the [no_memory_fixed_representation_validation.py] section in 'fixed_representation/config.cf' file and run:

```
python /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fixed_representation/no_memory_fixed_representation_validation.py /home/eden/work/codes/In-Defense-Of-Simple-But-Strong-Class-Incremental-Algorithms/fixed_representation/config.cf
```


### 4. IL with prior knowledge:
* #### ![DeeSIL](https://latex.codecogs.com/gif.latex?DeeSIL)  
The same steps as for DeeSIL without memory, you only need to provide the prior knowledge model as the used model to extract features instead of the first batch model, the rest of steps is the same.
* #### ![fFR](https://latex.codecogs.com/gif.latex?fFR)  
The same steps as for fFR without memory, you only need to provide the prior knowledge model as the used model instead of the first batch model, and this for all (10) batches.



### Remarks. 
1. The examples given are for a subset of 10 classes from ILSVRC, the first 5 classes are used as basic classes and the second 5 as an incremental batch. The example can be applied on any other dataset with the same way if you respect the data format. 
2. Don't forget to compute the images mean/std used for normalization of your training dataset (if different from ILSVRC, VGG-Face2 and Google Landmarks) and add it to the codes.
3. The models provided in this minimal working example are all suboptimal and provided only to make the code work.
4. Parameters that don't figure here are explained in DeeSIL repository.
5. If you find that some parts are not well explained or something is missing, don't hesitate to send an email to {eden.belouadah, adrian.popescu}@cea.fr

