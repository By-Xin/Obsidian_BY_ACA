#DeepLearning 

## 1. Introduction

Three key problems for lifelong learning:
- Knowledge Retention
  - But NOT Intransigence (i.e. we need it both not to forget the old ones and to be flexible to learn new ones)
  - An observation of **Catastrophic Forgetting** is that: if we train the model first and then fine-tune it on new tasks, the model will forget the old tasks. But if we pool the data from all tasks and train the model from scratch, the model will not forget the old tasks.
  - We call the pool-based training **Multi-Task Learning**. Usually it is considered as the upper bound of the performance of lifelong learning.
- Knowledge Transfer
- Model Expansion


## 2. Evaluation of Lifelong Learning

First of all, we need a sequence of tasks. We can evaluate the performance of lifelong learning by the following metrics:
| | Test Task 1 | Test Task 2 | $\cdots$ | Test Task $T$ |
|---|---|---|---|---|
|Rand Init.| $R_{0,1}$ | $R_{0,2}$ | $\cdots$ | $R_{0,T}$ |
|Aft.Train Task 1| $R_{1,1}$ | $R_{1,2}$ | $\cdots$ | $R_{1,T}$ |
|Aft.Train Task 2| $R_{2,1}$ | $R_{2,2}$ | $\cdots$ | $R_{2,T}$ |
|$\cdots$| $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ |
|Aft.Train Task $T$| $R_{T,1}$ | $R_{T,2}$ | $\cdots$ | $R_{T,T}$ |

$R_{i,j}$ is the performance of the model after **training on task $i$ and testing on task $j$**.
- If $i>j$, then $R_{i,j}$ is the performance of the model on the old task $j$.
- If $i<j$, then $R_{i,j}$ is the performance of the model on the unseen task $j$. (i.e. Transfer Learning)

The performance of lifelong learning can be evaluated by:
-  $\text{Accuraciy} = \frac{1}{T} \sum_{i=1}^T R_{T,i}$
-  $\text{Backward Transfer} = \frac{1}{T-1} \sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})$
-  $\text{Forward Transfer} = \frac{1}{T-1} \sum_{i=2}^{T} (R_{i-1,i} - R_{0,i})$


## 3. Several Methods for Lifelong Learning

### Selective Synaptic Plasticity (Regularization-based)

**Basic Idea**: Some parameters are more important than others w.r.t. the previous tasks.  We should keep the important parameters unchanged and only update the less important ones.

Denote $\theta_i$ as the parameters of the model to be learned. $\theta_{i}^b$ is the parameters of the model already learned. $b_i$ is the importance of the parameters w.r.t. the previous tasks. The loss function is:
$$
\mathcal{L} '( \theta) = \mathcal{L}(\theta) + \lambda \sum_{i=1} b_i (\theta_i - \theta_i^b)^2
$$
- If $b_i$ is large, then to minimize the loss function, we should keep $\theta_i$ basically the same as $\theta_i^b$.
- If $b_i$ is small, we can update $\theta_i$ more freely.
- If $b_i=0$, then it is the case of catastrophic forgetting.
- If $b_i=\infty$, then it is the case of intransigence (i.e. the model will not be updated on any new tasks).

***How to calculate $b_i$?***

The setting of $b_i$ is manually designed. Intuitively, after we have learned the parameters $\theta_i^b$, then we can calculate the change of the loss function w.r.t. the parameters $\theta_i$, i.e. $\frac{\partial \mathcal{L}(\theta)}{\partial \theta_i}$. If the change is large, then we should keep the parameter $\theta_i$ unchanged, i.e. the $b_i$ should be large. If the change is small, then we can update the parameter $\theta_i$ more freely, i.e. the $b_i$ should be small.

> *Several references:*
> - [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796)
> - [Synaptic Intelligence](https://arxiv.org/abs/1703.04200)
> - [Memory Aware Synapses](https://arxiv.org/abs/1711.09601)
> - [RWalk](https://arxiv.org/abs/1802.10112)
> - [Sliced Cramer Preservation](https://openreview.net/forum:id=BJge3TNKwH)
> - [Gradient Episodic Memory](https://arxiv.org/abs/1706.08840)
>   - **Basic Idea**: Genearlly it will calculate the gradient of the loss function w.r.t. the first task $g^b$. Then for the new task with the new gradient $g$, if they point to the opposite direction (i.e. $<g^b, g> < 0$), then we should modify the direction of the gradient $g$ to be the same as $g^b$, namely $g'$ where $g'\cdot g^b \ge 0$, but also need to guarantee that the new gradient $g'$ is close to the original gradient $g$. 

### Additional Neural Resources Allocation

-  [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)
   - For Progressive Neural Networks, it will add a new neural network for each new task, and will not update the old neural networks. 

- [PackNet](https://arxiv.org/abs/1711.05769)
  - For PackNet, it will first allocate a large neural network for all tasks. Then it will allocate a small neural network for each task. Each neural network will not affect each other.

- [Compacting, Picking and Growing](https://arxiv.org/abs/1910.06562)

### Memory Replay

#### Generative Replay

**Basic Idea**: This method will train a generative model generating pseudo-data of the old tasks. Then the model will be trained on the pseudo-data and the new data. 

> *Several references:*
> - https://arxiv.org/abs/1705.08690
> - https://arxiv.org/abs/1711.10563
> - https://arxiv.org/abs/1909.03329

#### Adding new classes

- [Learning without Forgetting](https://arxiv.org/abs/1606.09282)
- [iCaRL](https://arxiv.org/abs/1611.07725)