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
[[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf)-[[supp]](http://openaccess.thecvf.com/content_ICCV_2019/supplemental/Belouadah_IL2M_Class_Incremental_ICCV_2019_supplemental.pdf)

To cite this work:

```
@InProceedings{Belouadah_2019_ICCV,
author = {Belouadah, Eden and Popescu, Adrian},
title = {IL2M: Class Incremental Learning With Dual Memory},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
} 
```

## Data
Data needed to reproduce the experiments from the paper is available [here](https://drive.google.com/drive/folders/1lSxH3BRnuDjQBYG46wcw6HptUrkSfhS9?usp=sharing)

## Requierements
* Python 2.7
* Pytorch 1.0.0
* Numpy 1.13.0
* SkLearn 0.19.1


## How to run

1. ### Training the first batch of classes from scratch (better on GPU)

```
python codes/scratch.py configs/scratch.cf
```

2. ### IL with Fine tuning (better on GPU)

```
python codes/ft.py configs/ft.cf
```
3. ### Scores extraction (better on GPU)

```
python codes/features_extraction_b1.py configs/features_extraction_b1_train.cf
python codes/features_extraction_b1.py configs/features_extraction_b1_val.cf
python codes/features_extraction_ft.py configs/features_extraction_ft.cf
```
4. ### IL2M (requires CPU only)
You should provide the following parameters to the program: images_list_files_path, scores_path, b1_scores_path, dataset_name, S, P, K

For example, for IL2M on ILSVRC1000 with 9 incremental states (10 states in total, each one having a number of classes = 100), with memory size 20000:
```
python codes/il2m.py  data/images_list_files/ /set/here/your/path/feat_scores_extract_for_ft/ /set/here/your/path/feat_scores_extract_for_first_batch/ ilsvrc 10 100 20000 2>&1 | tee /set/here/your/path/logs/il2m/ilsvrc/S~10/il2m_ilsvrc_s10_20k.log
```


### Remarks. 
1. If your dataset is different from those tested in our paper, you need to use [this code](https://github.com/EdenBelouadah/class-incremental-learning/blob/master/deesil/codes/utils/compute_images_mean_std.py) to compute the images mean/std of your dataset. The input parameters of the code should be the list of training images of the first batch of classes. Add the computed vectors to the file 'data/datasets_mean_std.txt', in order to use them later for image normalization.
2. Please delete all the comments from the configuration files, to avoid compilation errors. 
3. Feel free to send an email to eden.belouadah@cea.fr if there is any issue with the code.

