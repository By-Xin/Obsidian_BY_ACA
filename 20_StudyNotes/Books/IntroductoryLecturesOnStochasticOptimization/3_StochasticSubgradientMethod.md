# Introduction and Stochastic Subgradient Methods

## Introduction to Stochastic Optimization


Stochastic optimization 关注的核心问题如下: 
- 给定一个随机数据生成过程 $\mathcal{P}$, 其为样本空间 $\mathcal{S}$ 上的一个概率分布, 从中可以抽取样本 $S \sim \mathcal{P}$. 有决策向量 $\mathbf{x} \in \mathbb{R}^n$, 以及对每个决策向量和每个具体的样本观测 $s$, 都有一个损失函数 $F(\mathbf{x}; s)$ (这里认为其关于 $\mathbf{x}$ 是凸的). 由于随机性的存在, 我们希望最小化的是总体期望损失函数:
    $$
    f(\mathbf{x}) = \mathbb{E}_{S \sim \mathcal{P}}[F(\mathbf{x}; S)] = \int_{\mathcal{S}} F(\mathbf{x}; s) \,\mathrm{d}\mathcal{P}(s)
    $$
    其中 $\mathcal{X} \subseteq \mathbb{R}^n$ 是决策向量的可行域.

- 一阶的随机优化方法 (stochastic first-order methods) 是解决这类问题的主要工具, 尽管其计算速度通常比 Newton 方法更慢, 但其对噪声更为稳健. 另外在大规模问题中, 由于其并不需要遍历所有样本, 而是通过随机抽样来近似梯度, 因此在处理大数据集时具有显著优势.

在随机优化中, 我们还会关心泛化性能. 
- 简单来说, 当前的决策向量 $\mathbf{x}$ 是在给定的有限样本观测 $\{S_i\}_{i=1}^m$ 上得到的一个估计, 即 
    $$
    \hat{\mathbf{x}}(S_1, \ldots, S_m) = \arg\min_{\mathbf{x} \in \mathcal{X}} \frac{1}{m} \sum_{i=1}^m F(\mathbf{x}; S_i) := \hat{f}_m(\hat{\mathbf{x}}).
    $$
    然而我们还关注的是 $\hat{\mathbf{x}}$ 在总体分布 $\mathcal{P}$ 下的表现.

- 因此, 对于随机算法, 要同时关注 Optimization Guarantee 和 Generalization Guarantee. 

最后的部分还会讨论 lower bound / minimax optimality 的问题. 具体的问题分析将在后续章节中展开. 

## Stochastic Subgradient Method

优化中的一阶算法需要得到目标的 $f(\mathbf{x})$ 的 subgradient $\partial f(\mathbf{x})$, 然而真实的总体梯度是较为困难获得的, 因此引入 stochastic subgradient method.

给定函数 $f: \mathbb{R}^n \to \mathbb{R}\cup \{+\infty\}$, stochastic subgradient oracle 相当于一个黑盒, 其输入是一个点 $\mathbf{x} \in \text{dom}(f)$, 输出是一个随机向量 $\mathbf{g}$, 其是在 $\mathbf{x}$ 处的 subgradient 的无偏估计 $\mathbb{E}[\mathbf{g}] \in \partial f(\mathbf{x})$, 即
$$
f(\mathbf{y}) \geq f(\mathbf{x}) + \langle \mathbb{E}[\mathbf{g}], \mathbf{y} - \mathbf{x} \rangle, \quad \forall \mathbf{y} \in \text{dom}(f).
$$


***Definition* (Stochastic subgradient oracle)**: 随机向量 $\mathbf{g}$ 若为 stochastic subgradient oracle, 则其相当于一个三元组 $(\mathbf{g}, \mathcal{S}, \mathcal{P})$, 其中 $\mathcal{S}$ 是样本空间, $\mathcal{P}$ 是样本空间上的概率分布, $\mathbf{g}$ 是一个随机向量, 其满足
$$
\mathbb{E}_{\mathcal{P}}[\mathbf{g}(\mathbf{x}; S)] =\int_{\mathcal{S}} \mathbf{g}(\mathbf{x}; s) \,\mathrm{d}\mathcal{P}(s) \in \partial f(\mathbf{x}), \quad \forall \mathbf{x} \in \text{dom}(f).
$$