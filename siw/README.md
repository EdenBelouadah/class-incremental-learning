# Initial-Classifier-Weights-Replay-for-Memoryless-Class-Incremental-Learning
## Abstract
Incremental Learning (IL) is useful when artificial systems need to deal with streams of data and do not have access to all data at all times.
The most challenging setting requires a constant complexity of the deep model and an incremental model update without access to a bounded memory of past data.
Then, the representations of past classes are strongly affected by catastrophic forgetting.
To mitigate its negative effect, an adapted fine tuning which includes knowledge distillation is usually deployed.

We propose a different approach based on a vanilla fine tuning backbone.
It leverages initial classifier weights which provide a strong representation of past classes because they are trained with all class data.
However, the magnitude of classifiers learned in different states varies and normalization is needed for a fair handling of all classes.
Normalization is performed by standardizing the initial classifier weights, which are assumed to be normally distributed.

In addition, a calibration of prediction scores is done by using state level statistics to further improve classification fairness.
We conduct a thorough evaluation with four public datasets in a memoryless incremental learning setting. 

Results show that our method outperforms existing techniques by a large margin for large-scale datasets. 

## Paper
The paper is accepted in BMVC2020. Pre-print link : https://arxiv.org/pdf/2008.13710.pdf

## Data
Data and code needed to reproduce the experiments from the paper will be soon available

## Requierements
* Python 2.7 or python 3
* Pytorch 1.0.0
* Numpy 1.13.0
* SkLearn 0.19.1


## How to run

1. ### Training the first batch of classes from scratch

```
python codes/scratch.py configs/scratch.cf
```

2. ### Fine tuning without memory

```
python codes/no_mem_ft.py configs/no_mem_ft.cf
```
3. ### Features + Last layer parameters (weight and bias) extraction 

```
python codes/features_extraction.py configs/features_extraction.cf
```


4. ### Standardization of Initial Weights (SIW)
You should provide the following parameters:
* images_list_files_path : folder containing data lists files, in this case ./data/images_list_files
* ft_feat_scores_path : folder containing test features extracted after training fine tuning
* ft_weights_dir : folder containing classification layer parameters
* K : memory size, always equal to 0.
* P : number of classes per incremental state
* S : total number of states
* dataset : name of the dataset - ilsvrc, vgg_faces, google_landmarks or cifar100

For example, for ![iFT](https://latex.codecogs.com/svg.latex?inFT_{siw}^{mc} on CIFAR100 with 20 states, each one containing 5 classes:
```
python ./codes/inFT_siw_mc.py ./data/images_list_files ./data/feat_scores_extract_for_no_mem_ft/ ./data/weights_bias_for_no_mem_ft/ 0 5 20 cifar100
```

for the other post processing methods (todo), just change the path to the code file, the parameters are the same. 

### Remarks. 
1. If your dataset is different from ILSVRC, VGG-Face2, Google Landmarks and CIFAR-100, you need to compute the images mean/std used for normalization of your dataset using the training images of the first batch of classes and add it to the file 'data/datasets_mean_std.txt'.
2. Please delete all the comments from the configuration files, to avoid compilation errors. 
3. Feel free to send an email to eden.belouadah@cea.fr if there is any issue with the code.
