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
Data needed to reproduce the experiments are available [here](https://drive.google.com/drive/folders/1nh_uMDFPIGeY4ZcO1lw59_4lZ74W2EDI?usp=sharing)

## How to run

### I. FINE-TUNING (FT)

#### Requierements
* Python 2.7 or python 3
* Pytorch 1.0.0
* Numpy 1.13.0
* SkLearn 0.19.1


1. ### Training the first batch of classes from scratch

```
python ./FT/codes/scratch.py ./FT/configs/scratch.cf
```

2. ### Fine tuning without memory

```
python ./FT/codes/no_mem_ft.py ./FT/configs/no_mem_ft.cf
```
3. ### Features + Last layer parameters (weight and bias) extraction 

```
python ./FT/codes/features_extraction.py ./FT/configs/features_extraction.cf
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

For example, for ![inFT_siw_mc](https://latex.codecogs.com/svg.latex?inFT_{siw}^{mc}) on CIFAR100 with 20 states, each one containing 5 classes:
```
python ./FT/codes/inFT_siw_mc.py ./data/images_list_files ./data/feat_scores_extract_for_no_mem_ft/ ./data/weights_bias_for_no_mem_ft/ 0 5 20 cifar100
```

for the other baselines: ![inFT_siw](https://latex.codecogs.com/svg.latex?inFT_{siw}), ![inFT_mc_l2](https://latex.codecogs.com/svg.latex?inFT_{L2}^{mc}), ![inFT_mc](https://latex.codecogs.com/svg.latex?inFT^{mc}) , ![inFT_l2](https://latex.codecogs.com/svg.latex?inFT_{L2}) and ![inFT](https://latex.codecogs.com/svg.latex?inFT), just change the path to the code file, the parameters are the same. 


### II. Learning-without-Forgetting (LwF)

#### Requierements
* Python 2.7.9
* Tensorflow-gpu 1.4.0


1. ### Training the model

```
python ./LwF/codes/lwf.py ./LwF/configs/lwf.cf
```
3. ### Features  extraction 
Example for CIFAR100 with 20 states.

```
python ./LwF/codes/features_extraction.py  20, 5, 0, 128, cifar100, save_path, output_dir, train_or_val, feat_or_scores, 1, 20 
```

3. ### Last layer parameters (weight and bias)  extraction 

python ./LwF/codes/last_layer_params_extraction.py  20, 5, cifar100, save_path, output_dir, 1, 20


4. ### Standardization of Initial Weights (SIW)
You should provide the following parameters:
* ft_feat_scores_path : folder containing test features extracted after training LwF
* ft_weights_dir : folder containing classification layer parameters
* K : memory size, always equal to 0.
* P : number of classes per incremental state
* S : total number of states
* dataset : name of the dataset - ilsvrc, vgg_faces, google_landmarks or cifar100

For example, for ![inLwF_siw_mc](https://latex.codecogs.com/svg.latex?inLwF_{siw}^{mc}) on CIFAR100 with 20 states, each one containing 5 classes:
```
python ./codes/inLwF_siw_mc.py ./data/feat_scores_extract_for_lwf/ ./data/weights_bias_for_lwf/ 0 5 20 cifar100
```

for the other baselines: ![inLwF_siw](https://latex.codecogs.com/svg.latex?inLwF_{siw}), ![inLwF](https://latex.codecogs.com/svg.latex?inLwF), just change the path to the code file, the parameters are the same. 


### Remarks. 
1. If your dataset is different from those tested in our paper, you need to use [this code](https://github.com/EdenBelouadah/class-incremental-learning/blob/master/deesil/codes/utils/compute_images_mean_std.py) to compute the images mean/std of your dataset. The input parameter should be the list of training images of the first batch of classes. Add the computed vectors to the file 'data/datasets_mean_std.txt' to use them later for image normalization.
2. Please delete all the comments from the configuration files, to avoid compilation errors. 
3. Feel free to report any issue with the code.
