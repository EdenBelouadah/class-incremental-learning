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

## Code
The code will be available soon. 

The paper is accepted in BMVC2020

