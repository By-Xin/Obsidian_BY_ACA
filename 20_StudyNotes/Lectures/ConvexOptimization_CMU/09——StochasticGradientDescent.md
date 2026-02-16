# Stochastic Gradient Descent

> [!info] References
> - Lecture: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/
> - Reading: 最优化: 建模、算法与理论, 刘浩洋等, 2.7 小节.

## Stochastic Algorithm

为方便讨论, 这里给出一个随机优化在有监督学习中的典型应用. 

- 有输入特征 $X \in \mathbb{R}^d$ 和输出标签 $Y \in \mathbb{R}$, 且 $(X, Y) \sim \mathcal{P}$. 目标是学习一个函数 $\hat{\phi}: \mathbb{R}^d \to \mathbb{R}$ 使得 $\hat{\phi}(X)$ 能够很好地预测 $Y$. 此外, 往往对 $\phi$ 的假设空间 $\mathcal{H}$ 进行限制以缩小搜索范围, 参数化为 $\phi(\cdot; \theta)$, 其中 $\theta \in \mathbb{R}^p$ 是模型参数. 通过引入一个损失函数 $L: \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ 来衡量预测误差, 以及正则项 $h: \mathbb{R}^d \to \mathbb{R}$ 来保证解的某些性质, 可以将学习问题表述为如下优化问题:
  
    $$
    \min_{\theta \in \mathbb{R}^p} \mathbb{E}_{(X, Y) \sim \mathcal{P}}[L(\phi(X; \theta), Y)]+ h(\theta).
    $$

- 在实践中, 我们通常只能获得一个有限的训练数据集 $\{(x_i, y_i)\}_{i=1}^N$ 来近似 $\mathcal{P}$.  因此, 优化问题可以表述为:
  
$$
\min_{\theta \in \mathbb{R}^p} \frac{1}{N} \sum_{i=1}^N L(\phi(x_i; \theta), y_i)+ h(\theta):= f(\theta).
$$

在下面的讨论中, 为表述习惯, 我们将优化问题重新表述为如下形式, 并且暂时假设 $f_i$ 为可微且凸的函数.
$$
\min_{x\in \mathbb{R}^n} f(x):= \frac{1}{N} \sum_{i=1}^N f_i(x),
$$

## Stochastic Gradient Descent

### SGD and Mini-batch SGD

考虑原始的梯度下降算法:
$$
x_{k+1} = x_k - \alpha_k \nabla f(x_k) = x_k - \alpha_k \frac{1}{N} \sum_{i=1}^N \nabla f_i(x_k).
$$
- 由于经验风险为 $f(x) = \frac{1}{N} \sum_{i=1}^N f_i(x)$, 则全梯度就是 $\nabla f(x) = \nabla \frac{1}{N} \sum_{i=1}^N f_i(x) = \frac{1}{N} \sum_{i=1}^N \nabla f_i(x)$. 计算全梯度需要遍历整个数据集, 当 $N$ 很大时, 计算成本非常高.

SGD 随机梯度法则通过在每次迭代中随机选择一个样本 $i_k$ 来近似梯度:
$$x_{k+1} = x_k - \alpha_k \nabla f_{i_k}(x_k).$$
- 这里, $i_k$ 是从 $\{1, 2, \ldots, N\}$ 中随机等可能抽样得到的索引. 用单个样本的梯度近似整个数据集的梯度, 大大降低了每次迭代的计算成本.
- 该操作的合理性在于, 当给定 $x_k$ 时, $\nabla f_{i_k}(x_k)$ 是 $\nabla f(x_k)$ 的无偏估计, 即 $\mathbb{E}_{i_k}[\nabla f_{i_k}(x_k) \mid x_k] = \nabla f(x_k)$.
    - *Proof*: 由于 $i_k$ 是从 $\{1, 2, \ldots, N\}$ 中随机等可能抽样得到的索引, 则有: 
        $$
        \begin{aligned}
        \mathbb{E}_{i_k}[\nabla f_{i_k}(x_k) \mid x_k] &= \sum_{i=1}^N P(i_k = i) \nabla f_i(x_k) \\
        &= \frac{1}{N} \sum_{i=1}^N \nabla f_i(x_k) \\
        &= \nabla f(x_k).
        \end{aligned}
        $$

不过只选取一个样本的梯度会引入较大的方差, 导致 SGD 的收敛速度较慢. 为了平衡计算效率和收敛速度, 一般使用 mini-batch SGD, 即在每次迭代中随机选择一个小批量的样本 $\mathcal{I}_k \subset \{1, 2, \ldots, N\}$ 来近似梯度:
$$x_{k+1} = x_k - \alpha_k \frac{1}{|\mathcal{I}_k|} \sum_{i \in \mathcal{I}_k} \nabla f_i(x_k).$$

此外, 对于不可微的优化问题, 也可以考虑使用随机次梯度法则:
$$x_{k+1} = x_k - \alpha_k g_{i_k}(x_k),$$
其中 $g_{i_k}(x_k)$ 是 $f_{i_k}$ 在 $x_k$ 处的一个次梯度.

### Variants of SGD

SGD 方法有一系列的变体, 旨在提高收敛速度和稳定性. 

#### Momentum SGD

传统的 SGD 和 GD 类似, 在较为病态的优化问题中可能会出现收敛缓慢的情况. Momentum 方法通过引入一个动量项来加速收敛, 并在高曲率或噪声场景中提供更为有效的更新. 