# Dataset Knowledge Transfer for Class-Incremental Learning without Memory
## Abstract
Incremental learning enables artificial agents to learn from sequential data. While important progress was made by exploiting deep neural networks, incremental learning remains very challenging. This is particularly the case when no memory of past data is allowed and catastrophic forgetting has a strong negative effect. We tackle class-incremental learning without memory by adapting prediction bias correction, a method which makes predictions of past and new classes more comparable. It was proposed when a memory is allowed and cannot be directly used without memory, since samples of past classes are required. We introduce a two-step learning process which allows the transfer of bias correction parameters between reference and target datasets. Bias correction is first optimized offline on reference datasets which have an associated validation memory. The obtained correction parameters are then transferred to target datasets, for which no memory is available. The second contribution is to introduce a finer modeling of bias correction by learning its parameters per incremental state instead of the usual past vs. new class modeling. The proposed dataset knowledge transfer is applicable to any incremental method which works without memory. We test its effectiveness by applying it to four existing methods. Evaluation with four target datasets and different configurations shows consistent improvement, with practically no computational and memory overhead. 

## Paper
The paper is accepted for publication in the WACV 2022 conference. Pre-print link : https://arxiv.org/pdf/2110.08421.pdf

To cite this work:

```

@inproceedings{slim2022_transil,
    author    = {Slim, Habib and Belouadah, Eden and Popescu, Adrian and Onchis, Darian},
    title     = {Dataset Knowledge Transfer for Class-Incremental Learning Without Memory},
    booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
    year      = {2022}
}

```

## Code 
All the codes are available in this GitHub Repository: https://github.com/HabibSlim/DKT-for-CIL/
