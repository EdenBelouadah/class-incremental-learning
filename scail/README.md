# ScaIL-Classifier-Weights-Scaling-for-Class-Incremental-Learning
## Abstract
Incremental learning is useful if an AI agent needs to integrate data from a stream. The problem is non trivial if the agent runs on a limited computational budget and has a bounded memory of past data.

In a deep learning approach, the constant computational budget requires the use of a fixed architecture for all incremental states.
The bounded memory generates imbalance in favor of new classes and a prediction bias toward them appears. This bias is commonly countered by introducing a data balancing step in addition to the basic network training.

We depart from this approach and propose simple but efficient scaling of past classifiers' weights to make them more comparable to those of new classes. Scaling exploits incremental state statistics and is applied to the classifiers learned in the initial state of classes to profit from all their available data. 

We also question the utility of the widely used distillation loss component of incremental learning algorithms by comparing it to  vanilla fine tuning in presence of a bounded memory. 

Evaluation is done against competitive baselines using four public datasets. Results show that the classifier weights scaling and the removal of the distillation are both beneficial.

## Paper
[[pdf]](https://arxiv.org/pdf/2001.05755.pdf)


## Data
Data needed to reproduce the experiments from the paper is available [here](https://drive.google.com/open?id=1kgoB0Oxb9Wv2wSWFT5Yf7IoKXR3gAL_3)

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

2. ### IL with Fine tuning (with and without herding) (better on GPU)

```
python codes/ft.py configs/ft.cf
python codes/ft_herd.py configs/ft_herd.cf
```
3. ### validation features extraction (better on GPU)

```
python codes/features_extraction.py configs/features_extraction.cf
```

4. ### Last layer parameters extraction (weight and bias)

```
python codes/extract_last_layer_weights_for_first_batch.py path/to/first/batch/model.pt model_num_classes 1 path/to/destination/dir
python codes/extract_last_layer_weights_for_ft.py path/to/ft/models_prefix number_of_states number_of_classes_per_state path/to/destination/dir
```


5. ### ScaIL (requires CPU only) todo
You should provide the following parameters:
* dataset : name of the dataset : ilsvrc, vgg_faces, google_landmarks or cifar100
* list_root_dir : folder containing data lists files, in this case ./data/images_list_files
* local_root_dir : folder containing classification layer parameters + validation features extracted after training fine tuning
* Z : total number of states, including the first non-incremental one
* B : memory size
* P : number of classes per incremental state
* last_batch_number : to stop the algorithm at this incremental state - useful for debugging
* top_rewnded : set it to 10 (see article for more explanations)

For example, for Scail on ILSVRC1000 with 9 incremental states (10 states in total, each one having a number of classes = 100), with memory size 20000:
```
python ./codes/scail.py ilsvrc ./data/images_list_files /path/to/scail/data 10 20000 100 10 10
```

Detailed execution instructions are in 'codes/scail.py'.

### Remarks. 
1. If your dataset is different from ILSVRC, VGG-Face2, Google Landmarks and CIFAR-100, you need to compute the images mean/std used for normalization of your dataset using the training images of the first batch of classes and add it to the file 'data/datasets_mean_std.txt'.
2. Please delete all the comments from the configuration files, to avoid compilation errors. 
3. Feel free to send an email to eden.belouadah@cea.fr if there is any issue with the code.

