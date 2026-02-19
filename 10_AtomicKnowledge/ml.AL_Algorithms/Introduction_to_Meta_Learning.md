#MetaLearning

## Introduction

- Machine learning: Find a function $f(\cdot)$ that maps input data $\mathcal{X}$ to output data $\mathcal{Y}$.
- Meta-learning: Find a function $F(\cdot)$ that is able to *find a function $f(\cdot)$ that maps input data $\mathcal{X}$ to output data $\mathcal{Y}$*. (i.e. $F(\mathcal{Data}) = f$)

## Basic Steps of Meta-Learning

- ***1. Define a set of learning algorithms***
    ![20250104122400](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104122400.png)
    - Some of the steps (e.g. hyperparameter tuning) are framed by red boxes, which are the steps that are usually done by humans.
    - The goal of meta-learning is to automate these steps, denoting as $\phi$, e.g.:
      - Neural Architecture Search (NAS)
      - Initial Parameterization
      - Learning Rate Scheduling ...

- ***2. Defining the goodness of a function $F$, i.e. the loss function $\mathcal{L}(\phi)$ for the meta-learning algorithm $F_{\phi}$.***
  - Use the meta-learning algorithm $F_\phi$ on Task $\mathcal{T}^{(i)}$ with training examples (*Support set*) to get a algorithm $f_\theta^{(i)}$, and accordingly get the loss $l^{(i)} = \mathcal{Loss}(f_\theta^{(i)})$ on the testing examples (*Query Set*). Then we can define the goodness of $F$ as the average loss over all tasks, i.e. $\mathcal{L}(\phi) = \frac{1}{N}\sum_{i=1}^{N}l^{(i)}$.
    - Actually, for a meta-learning algorithm, we should provide a triplet of data: $\{(\mathcal{T}_i, \mathcal{D_{\text{train}}}, \mathcal{D_{\text{test}}})\}_{i=1}^{N}$.
    - The training for meta-learning sometimes is called *Across-task Training*, and that of machine learning is called *Within-task Training*. So in one episode of across-task training, it will include within-task training and within-task testing.  The across-task training sometimes is called *outer loop*, and the within-task training is called *inner loop*.

- ***3. Optimize the meta-learning algorithm $F_{\phi^*}$***
  $$\phi^* = \arg\min_{\phi}\mathcal{L}(\phi) \quad \text{(Ideally)}$$
  - If $\phi$ is gradient-based, we can use gradient descent to optimize $\phi$.
  - If $\mathcal{L}(\phi)$ is non-differentiable, **we can use reinforcement learning / evolutionary algorithms to optimize $\phi$**.



> ***Few-shot Learning & Meta-Learning***
> - The goal of few-shot learning is to learn a model that can generalize well to new tasks with only a few examples.
> - To achieve this, we can use meta-learning to learn a model that can quickly adapt to new tasks.

## Some Useful Concepts in Meta-Learning


### OmniGlot Dataset

- OmniGlot dataset is a commonly used dataset for few-shot learning. It contains 1623 different handwritten characters, each of which is drawn by 20 different people. 
- We should define a $N$-way $K$-shot task, i.e. in each training-and-testing task, we have
  - $N$ classes
  - $K$ samples per class
    ![20250104124000](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104124000.png)


###  Meta Learning v.s. ML
Something happens in ML that also happens in Meta Learning:
- Overfitting on the training tasks
- Task augmentation
- hyperparameter for the meta-learning algorithm
- Development Task

## What is Learnable in Meta-Learning? (Simple Literature Review) 

### Parameter Initialization: MAML

- [Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400)

- [How to train your MAML? (ICLR, 2019)](https://arxiv.org/abs/1810.09502)


  
- [Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML (ICLR, 2020)](https://arxiv.org/abs/1909.09157)
  - ANIL (Almost no inner loop) algorithm

- Reptile [Reptile: A Scalable Meta-Learning Algorithm (ICML, 2018)](https://arxiv.org/abs/1803.02999)



### Optimizer

- [Learning to learn by gradient descent by gradient descent (NIPS, 2016)](https://arxiv.org/abs/1606.04474)

### Neural Architecture Search (NAS)

A simple goal of NAS is given a determined structure of the neural network, to find the best hyperparameters for the neural network, i.e. the hyperparameter tuning.

Typical methods for hyperparameter optimization:
- Gird Search
- Random Search 
  - Sample some hyperparameters randomly from the grids or distributions
  - **Theoratical assumptions:**
    - The relavent assumption is that: top $K$ results are good enough to approximate the global optimum. 
    - Then if there are totally $N$ hyperparameters(or its combinations), the probability of finding the best hyperparameters in the top $K$ results is $K/N$.
    - Thus if we sample $x$ times, the probability of finding the (almost) best hyperparameters is $1-(1-K/N)^x$.
    - Further control the probability of finding the best hyperparameters greater than $1-\epsilon$. For example $1-\epsilon=0.90$ and $N=1000, K=10$, then $x=230$.

There are also other things can be designed: e.g. the activation function, the methods for optimization, etc.

More complex in NAS, the goal is to find the best neural network structure $\phi$, which is non-differentiable. 

- **RL for NAS**:
  - [Neural Architecture Search with Reinforcement Learning (ICLR, 2017)](https://arxiv.org/abs/1611.01578)
    -   Here, we regard $\phi$ as the agent's parameter, and the output of the agent is the neural network structure(e.g. the number of layers, the number of neurons in each layer, the activation function, etc.). The reward is $-\mathcal{L}(\phi)$.
     ![20250104135603](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104135603.png) 
  - [Learning Transferable Architectures for Scalable Image Recognition (CVPR, 2018)](https://arxiv.org/abs/1707.07012)
  - [Efficient Neural Architecture Search via Parameter Sharing (ICML, 2018)](https://arxiv.org/abs/1802.03268)



- EA for NAS:
  - [Large-Scale Evolution of Image Classifiers (ICML, 2017)](https://arxiv.org/abs/1703.01041)
  - [Regularized Evolution for Image Classifier Architecture Search (AAAI, 2019)](https://arxiv.org/abs/1802.01548)
  - [Hierarchical Representations for Efficient Architecture Search (ICLR, 2018)](https://arxiv.org/abs/1711.00436)

Sometimes there are some other methods to make it differentiable, e.g.:
- [Differentiable Architecture Search (ICLR, 2019)](https://arxiv.org/abs/1806.09055)
  ![20250104140126](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104140126.png)

- [DARTS: Differentiable Architecture Search (ICLR, 2019)](https://arxiv.org/abs/1806.09055)
### Data Augmentation

- [DADA: Differentiable Automatic Data Augmentation (ECCV, 2020)](https://arxiv.org/abs/2003.03780)
    ![20250104140315](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104140315.png)
- [Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules (ICML, 2019)](https://arxiv.org/abs/1905.05393)
- [AutoAugment: Learning Augmentation Policies from Data (CVPR, 2019)](https://arxiv.org/abs/1805.09501)

### Sample Reweighting

- [Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting (CVPR, 2019)](https://arxiv.org/abs/1902.07379)

### Metric Based Meta-Learning  (Learning to Compare)


## Relations to Other Fields




## Model-Agnostic Meta-Learning (MAML)
> [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (ICML, 2017)](https://arxiv.org/abs/1703.03400)

### Introduction to MAML

The key of MAML is to learn a best initialization of parameters.

Here, define the loss function as:
$$\mathcal{L}(\phi) = \sum_{i=1}^{N} l^{(i)}(\hat\theta^{(i)})$$ 
where $\phi$ is the initialization of the parameters, $\hat\theta^{(i)}$ is the model learned from the $i$-th task which is dependent on $\phi$ (the final post-trained parameter starting from $\phi$), and $l^{(i)}$ is the loss function of the $i$-th task.
- Note that, for MAML, each task is facing the same initialization $\phi$, thus the structure for each task is the same.

To minimize $\mathcal{L}(\phi)$, we can use the gradient descent method (somehow):
$$\phi^* := \phi - \eta \nabla_{\phi}\mathcal{L}(\phi)$$
where $\alpha$ is the learning rate.

### Methodology of MAML

- In practice, we assume that in the training process, the parameter will only be updated for once, i.e. if we initialize the parameter as $\phi$, then the final parameter on task $n$ will be: 
  $$\begin{equation}\hat\theta^{(n)} = \phi - \varepsilon \nabla_{\theta}\ell^{(n)}(\phi)\end{equation}$$ 
  where $\theta$ is the parameter of the model.

  - Several reasons for this assumption:
    1. The computation is expensive.
    2. It is actually a very good model if we can really achieve the goal that the model can be well-trained in one step.
    3. Actually we can really update many times on testing. 
    4. For few-shot learning, the data itself is limited. 

- Also recall that the loss function of the Meta-Learning algorithm is defined as:
  $$\begin{equation}\mathcal{L}(\phi) = \sum_{n=1}^{N} \ell^{(n)}(\hat\theta^{(n)})\end{equation}$$
  where the object parameter is $\phi$, by gradient descent, we can update $\phi$ as:
  $$\begin{equation}
    \phi^* = \phi - \eta \nabla_{\phi}\mathcal{L}(\phi)
  \end{equation}$$

- Further see the gradient of $\mathcal{L}(\phi)$:
  $$\begin{equation}
    \nabla_{\phi}\mathcal{L}(\phi) = \nabla_{\phi}\sum_{n=1}^{N}\ell^{(n)}(\hat\theta^{(n)}) = \sum_{n=1}^{N}\nabla_{\phi}\ell^{(n)}(\hat\theta^{(n)}) 
  \end{equation}$$
  - Note that $\ell^{(n)}(\hat\theta^{(n)})$ is the loss function of the $n$-th task, and $\hat\theta^{(n)}$ is the final parameter of the model on the $n$-th task. But it is actually a vector, i.e. in practice, $\phi \in \mathbb{R}^d,\theta \in \mathbb{R}^d$, and $\ell^{(n)}(\hat\theta^{(n)}) \in \mathbb{R}$, thus:
    $$ \begin{equation}
      \nabla_{\phi}\ell(\hat\theta) = [\cdots, \frac{\partial \ell(\hat\theta)}{\partial \phi_i}, \cdots]^\top
    \end{equation}$$
    where the relationship goes as:
      ![20250104192953](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104192953.png)
    
    Thus by the chain rule, we can get:
    $$\begin{equation}
      \frac{\partial \ell}{\partial \phi_i} = \sum_j \frac{\partial \ell(\hat\theta)}{\partial \hat\theta_j}\frac{\partial \hat\theta_j}{\partial \phi_i}
    \end{equation}$$
    - $\frac{\partial \ell(\hat\theta)}{\partial \hat\theta_j}$ is easy to obtain according to the definition of the loss function
    - $\frac{\partial \hat\theta_j}{\partial \phi_i}$ can be obtained by the equation (1): $\hat\theta_j = \phi_j - \varepsilon \frac{\partial \ell(\phi)}{\partial \phi_j}$, thus:
      $$\begin{aligned}
        \frac{\partial \hat\theta_j}{\partial \phi_i} &= \begin{cases}
          -\varepsilon \frac{\partial^2 \ell(\phi)}{\partial \phi_i \partial \phi_j} \approx 0 & i \neq j \\
          1 - \varepsilon \frac{\partial^2 \ell(\phi)}{\partial \phi_i \partial \phi_i} \approx 1 & i = j
        \end{cases}
      \end{aligned}$$ 
      Here the approximation is by neglecting the second-order term. Thus we can get:
      $$\begin{equation}
        \frac{\partial \ell}{\partial \phi_i} = \sum_j \frac{\partial \ell(\hat\theta)}{\partial \hat\theta_j}\frac{\partial \hat\theta_j}{\partial \phi_i} \approx \frac{\partial \ell(\hat\theta)}{\partial \hat\theta_i}
      \end{equation}$$
  - Then equation $(5)$ i.e. the gradient of the loss function can be approximated as:
    $$\begin{equation}
       \nabla_{\phi}\ell(\hat\theta) = [\frac{\partial \ell(\hat\theta)}{\partial \hat\theta_1}, \cdots, \frac{\partial \ell(\hat\theta)}{\partial \hat\theta_d}]^\top = \nabla_{\hat\theta}\ell(\hat\theta)
    \end{equation}$$

### Real Implementation of MAML

![20250104194500](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104194500.png)


### MAML v.s. Model Pre-training

#### From Loss function Aspect

***For MAML,***
$$
\mathcal{L}(\phi) \approx \sum_{i=1}^{N} l^{(i)}(\hat\theta^{(i)})
$$
where $\hat\theta^{(i)}$ is the model learned from the $i$-th task starting from $\phi$.


We do not care the performance of $\phi$ on the training tasks. We care the final parameter $\hat\theta^{(i)}$ after the training from $\phi$.

![20250104143215](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104143215.png)


***For Model Pre-training,***
$$
\mathcal{L}(\phi) = \sum_{i=1}^{N} l^{(i)}(\phi)
$$


Model pre-training (e.g. BERT) is to pre-train the model on a large dataset, and then fine-tune the model on the target task. It is similar to MAML that we can regard the result of the pre-training as the initialization of the model.

However, we do not care the performance of after training. We take $\phi$ as a *pre-trained* parameter and trying to make it as good as possible on the training tasks. We only focus on the present performance of $\phi$ on the training tasks.

![20250104143654](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104143654.png)

(Though we are not trying to infer that MAML is better than model pre-training)

#### From the Gradient Aspect

![20250104194610](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104194610.png)

---

### Toy Example of MAML (Sine Wave Task)

> *Ref 1 (Blog): [Paper repro: Deep Metalearning using “MAML” and “Reptile”](https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0)*
> 
> *Ref 2 (Notebook): [Jupyter Notebook](https://github.com/AdrienLE/ANIML/blob/master/ANIML.ipynb)*

#### Task Description

Consider the following toy example, for each task:
- Given a target function $y = a \sin (x + b)$. For each task, we may randomly generate $a$ and $b$.
- Sample $K$ points from the target function as the training data.
- Use these samples to estimate the target function.


![20250104184932](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104184932.png)

Actually if we try to see the case with $1000$ tasks, we can see that the data points are overlapped with each other.


![20250104185610](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104185610.png)


#### Solving the Task with Pre-trained Model

![20250104190513](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104190513.png)

Since there are multiple possible values for each x across multiple tasks, if we train a single neural net to deal with multiple tasks at the same time, its best bet will simply be to return the average y value across all tasks for each x. So the average is basically 0, which means a neural network trained on a lot of tasks would simply return 0 everywhere! Basically it looks like our transfer model learns a constant function and that it is really hard to fine tune it to something better. It’s not even clear that our transfer learning is any better than random initialization…

#### Solving the Task with MAML

![20250104190606](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104190606.png)

The idea of MAML is to learn a good initialization of the parameters that can be fine-tuned to any task. So we can see that the model can be well-trained on the task.

![20250104190723](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104190723.png)


## Reptile

> [Reptile: A Scalable Meta-Learning Algorithm (ICML, 2018)](https://arxiv.org/abs/1803.02999)

![20250104194828](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104194828.png)

![20250104195051](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104195051.png)

Performance comparison from the paper:

![20250104195126](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104195126.png)


## Meta-Learning as RNN / LSTM

> - [Optimization as a Model for Few-Shot Learning (ICLR, 2017)](https://arxiv.org/abs/1606.04474)
>
> - [Learning to learn by gradient descent by gradient descent (NIPS, 2016)](https://arxiv.org/abs/1606.04474)

**In some sense, we can regard the meta-learning as a RNN / LSTM model.**

![20250104122400](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250104122400.png)

### RNN & LSTM (Review) 

The structure of RNN and LSTM:

![20250106153030](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250106153030.png)

Particularly, for LSTM:

![20250106155918](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250106155918.png)

- $c_t$ change slowly, $c_t := z^f \odot c_{t-1} + z^i \odot z$ and thus can store the information for a long time.
- $h_t$ change quickly, $h_t := z^o \odot \tanh(c_t)$ and thus can store the information for a short time.
- $y_t$ is the output of the LSTM, $y_t := \sigma(W' h_t)$.

### Gradient Descent as LSTM

Recall that for gradient descent:
$$
\theta_{t} := \theta_{t-1} - \eta \nabla_{\theta} \ell(\theta_{t-1})
$$

Compare ths structure with LSTM:

$$
c_t := z^f \odot c_{t-1} + z^i \odot z
$$

Then $\theta_t$ in gradient descent can be regarded as the hidden state $c_t$ in LSTM. Thus, the LSTM can be reformulated by defining:
$$\begin{aligned}
  c_t &:= \theta_t \\
  [h_{t-1},x_t] &:= - \nabla_{\theta} \ell \\
  W &:= I  \Rightarrow Z = -\nabla_{\theta} \ell\\
  z^f &:= 1_{d \times 1} \\
  z^i &:= \eta 1_{d \times 1} \\
\end{aligned}$$

Then the LSTM can be reformulated from:
$$\begin{aligned}
  c_t &:= z^f \odot c_{t-1} + z^i \odot z \\
  \end{aligned}$$
to:
$$\begin{aligned}
  \theta_t &:= 1_{d \times 1} \odot \theta_{t-1} + \eta 1_{d \times 1} \odot (-\nabla_{\theta} \ell) = \theta_{t-1} - \eta \nabla_{\theta} \ell
  \end{aligned}$$
which is exactly the gradient descent. 

![20250107133922](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250107133922.png)

Moreover, this is the hard coding of the LSTM, and we can also use the LSTM to learn the gradient descent, paticularly the *input gate* $z^i$  and the *forget gate* $z^f$.
- If $z^i$ can be learned, then the learning rate $\eta$ can be learned (*dynamic learning rate*).
- If $z^f$ can be learned, then it can be used as a *regularization* term.

#### Gradient Descent as LSTM in Meta-Learning

![20250107134834](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250107134834.png)

Note that, one problem of this model is that, the cell $\theta_t$ will affect the gradient of the loss function, and thus there is an arrow from $\theta_t$ to $-\nabla_{\theta} \ell$, which does not appear in the original LSTM model. To be rigorous, such link will affect the back-propagation of the gradient. However, in practice, we can ignore this link.

#### Real Implementation

Another problem is that, the parameter to be gradient descent may be a high-dimensional vector, and thus the LSTM may be too large to be trained.  (As for each parameter to be gradient descent, we need a LSTM processing unit.)

In practice, to deal with this problem, we can use the *shared LSTM* model, i.e. for each parameter to be gradient descent, we use the same LSTM model (the same $z^f, z^i$).

---

## Metric Based Meta-Learning

A more bold idea is to use the metric-based method to do the meta-learning. For metic-based method, we:
- Input: Training data with labels & Testing data without labels
- Output: A model that can predict the labels of the testing data.


### Siamese Network

The Siamese Network is a classic model for face verification. The structure of the Siamese Network is as follows:
![20250107143828](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250107143828.png) 


The essence of the Siamese Network is to train a CNN that maps the input image to a feature vector, and try to make the feature vectors of the same person close to each other, and those of different persons far from each other.

#### Application: Face Verification
First clearify the concept:
- *Face Verification*: Given a face image, the model should judge whether the face is the same as the target face (0-1 classification).

Face Verification is a classic *few shot learning* problem, since we only have a few images for each person.

### N-way Few/One-shot Learning



#### Application: Face Identification
- *Face Identification*: Given a face image, the model should judge which person the face belongs to (multi-class classification).

A classical solution is: **Prototypical Network**.

## Relation to Other Fields

### Meta-Learning &  Self-Supervised Learning

> [Meta Learning for Natural Language Processing: A Survey](https://arxiv.org/abs/2205.01500)

- MAML learns the initialization parameter $\phi$ by gradient descent. However, such training process of $\phi$ also needs a initialization. We can use self-supervised learning (e.g. Bert) to learn the initialization of $\phi$.
- Bert, on the other hand, as the pre-training objective is different from the final ones, there is a "learning gap" for such pre-training. Thus we can use Bert as a pre-training model, and then use MAML to locate the starting point of the fine-tuning.

> [Low-Resource Domain Adaptation for Compostional Task-Oriented Semantic Parsing (EMNLP, 2020)](https://arxiv.org/abs/2010.03546)
>
> ![20250107152326](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250107152326.png)

> [Investigating Meta-Learning Algorithms for Low-Resource Natural Language Understanding Tasks (EMNLP, 2019) ](https://arxiv.org/abs/1908.10423)
>
> ![20250107152518](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250107152518.png)

### Meta-Learning &  Domain Generalization

- 


### Meta-Learning &  Knowledge Distillation

- Knowledge Distillation is to first train a large model (teacher model) and then train a small model (student model) to mimic the teacher model (i.e. to make minimize the cross-entropy loss between the output of the teacher model and the student model).



### Meta-Learning &  Life-long Learning