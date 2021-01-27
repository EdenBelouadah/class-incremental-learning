# Active-Class-Incremental-Learning-for-Imbalanced-Datasets
## Abstract
Incremental Learning (IL) allows AI systems to adapt to streamed data.
Most existing algorithms make two strong hypotheses which reduce the realism of the incremental scenario: (1) new data are assumed to be readily annotated when streamed and (2) tests are run with balanced datasets while most real-life datasets are imbalanced.
These hypotheses are discarded and the resulting challenges are tackled with a combination of active and imbalanced learning.

We introduce sample acquisition functions which tackle imbalance and are compatible with IL constraints.
We also consider IL as an imbalanced learning problem instead of the established usage of knowledge distillation against catastrophic forgetting.
Here, imbalance effects are reduced during inference through class prediction scaling.

Evaluation is done with four visual datasets and compares existing and proposed sample acquisition functions.
Results indicate that the proposed contributions have a positive effect and reduce the gap between active and standard IL performance.


## Paper
The paper is accepted in IPCV workshop from ECCV2020. Pre-print link : https://arxiv.org/pdf/2008.10968.pdf

## Data

Data needed to reproduce the experiments are available [here](https://drive.google.com/drive/folders/1HDbXAsvqRtZqwryXo9YveFsrE16lj_xv?usp=sharing)


## How to run

### Requierements
* Python 2.7
* Pytorch 1.0.0
* Numpy 1.13.0
* SkLearn 0.19.1


1. ### Training the first batch of classes from scratch

```
python ./codes/scratch.py ./configs/scratch.cf
```


2. ### Active Incremental Learning

```
python ./codes/main.py ./configs/config.cf
```


### Remarks. 
1. If your dataset is different from those tested in our paper, you need to use [this code](https://github.com/EdenBelouadah/class-incremental-learning/blob/master/deesil/codes/utils/compute_images_mean_std.py) to compute the images mean/std of your dataset. The input parameter should be the list of training images of the first batch of classes. Add the computed vectors to the file 'data/datasets_mean_std.txt' to use them later for image normalization.
2. Please delete all the comments from the configuration files, to avoid compilation errors. 
3. Feel free to send an email to eden.belouadah@cea.fr if there is any issue with the code.
