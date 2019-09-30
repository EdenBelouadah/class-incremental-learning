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
Data used in the experiments is available here : https://drive.google.com/open?id=1kgoB0Oxb9Wv2wSWFT5Yf7IoKXR3gAL_3

## Requierements
* Python 2.7
* Pytorch 1.0.0
* Numpy 1.13.0


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
python codes/il2m.py  images_list_files/ /scratch/eden/feat_scores_extract_for_ft/ /scratch/eden/feat_scores_extract_for_first_batch/ ilsvrc 10 100 20000 2>&1 | tee /home/eden/logs/il2m/ilsvrc/S~10/il2m_ilsvrc_s10_20k.log
```


### Remarks. 
1. You need to compute the images mean/std used for normalization of your dataset using the traing images of the first batch of classes (if different from ILSVRC, VGG-Face2 and Google Landmarks) and add it to the file 'data/datasets_mean_std.txt'.
2. Please delete all the comments from the configuration files, to avoid compilation errors. 
3. Feel free to send an email to {eden.belouadah, adrian.popescu}@cea.fr if there is any issue with the code.

