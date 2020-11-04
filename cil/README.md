# Active-Class-Incremental-Learning-for-Imbalanced-Datasets
## Abstract
The ability of artificial agents to increment their capabilities when confronted with new data is an open challenge in artificial intelligence. The main challenge faced in such cases is catastrophic forgetting, i.e., the tendency of neural networks to underfit past data when new ones are ingested. A first group of approaches tackles catastrophic forgetting by increasing deep model capacity to accommodate new knowledge. A second type of approaches fix the deep model size and introduce a mechanism whose objective is to ensure a good compromise between stability and plasticity of the model. While the first type of algorithms were compared thoroughly, this is not the case for methods which exploit a fixed size model.
Here, we focus on the latter, place them in a common conceptual and experimental framework and propose the following contributions: (1) define six desirable properties of incremental learning algorithms and analyze them according to these properties, (2) introduce a unified formalization of the class-incremental learning problem, (3) propose a common evaluation framework which is more thorough than existing ones in terms of number of datasets, size of datasets, size of bounded memory and number of incremental states, (4) investigate the usefulness of herding for past exemplars selection, (5) provide experimental evidence that it is possible to obtain competitive performance without the use of knowledge distillation to tackle catastrophic forgetting, and (6) facilitate reproducibility by integrating all tested methods in a common open-source repository. The main experimental finding is that none of the existing algorithms achieves the best results in all evaluated settings. Important differences arise notably if a bounded memory of past classes is allowed or not. 


## Paper
The paper is under review at the Elsevier's Neural Networks journal. Pre-print link : https://arxiv.org/pdf/2011.01844.pdf

## Code and Data

Code and data will be soon available.
