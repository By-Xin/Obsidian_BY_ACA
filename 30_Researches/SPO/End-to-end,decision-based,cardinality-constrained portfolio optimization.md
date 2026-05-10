# End-to-end decision-based cardinality-constrained portfolio optimization

Hassan T. Anis, Roy H. Kwon | EJOR

## Introduction

- 传统投资组合优化中, 许多经典问题虽然优化模型本身看起来很确定, 但是本身的输入参数就是不确定的. 例如 $\min_{\mathbf{w}} \mathbf{w}^T \Sigma \mathbf{w}$ 要对 $\Sigma$ 进行估计. 
  - 传统的做法, 比如 predict-then-optimize, 先对参数进行尽量精确的估计, 然后再进行优化. 这是在寻找 in-sample 的最优解, 但是在 out-of-sample 的时候的表现如何并没有直接的保证.
  - 金融中收益建模的常见做法是 factor models, 通过引入一些因子来进行资产建模. 后续可以看到, 其可以整体写成
    $$
    r_{it} = \alpha_i + \sum_{p= 1}^P \beta_{ip} f_{pt} + \epsilon_{it}
    $$
    然而不论如何, 其总体终归是一个以 accuracy 为目标的预测模型, 例如最小化经验风险等. 不论如何, 在预测时, 只关心是否能够预测的准, 但是对于决策, 或者说优化时候的表现其并不关注. 


- Decision-focused learning (DFL) 是一个新的 paradigm, 其目标是直接优化决策的表现, 而不是预测的表现. 
  - 例如 Elmachtoub and Grigas (2021) 提出的 Smart "Predict, then Optimize" (SPO) loss 就是典型工作.  不过其有一个问题是在引入了 SPO loss 之后, 其优化问题就变得非凸, 非常复杂了.  故在 SPO 的工作中, 他们又提出了 SPO+ 作为一个 convex surrogate loss, 以便于优化.


## Background

### Cardinality-constrained portfolio optimization

假设市场共有 $N$ 个资产, 投资者最多可以购买 $k \ll N$ 个资产. 则可以定义对应的 long-only 权重:
$$
\mathcal{W}_k =\left\{\mathbf{w} \in \mathbb{R}_+^N: \boldsymbol{1}^\top \mathbf{w} = 1, \|\mathbf{w}\|_0 \leq k\right\}
$$

- 一个传统的 Big-M 表示方法是引入 binary variable $z_i = \boldsymbol{1}\{w_i > 0\}$, 表示是否购买了第 $i$ 个资产 , 则可以写成
  $$
  W^M_k = \left\{\mathbf{w} \in \mathbb{R}_+^N: \mathbf{z} \in \{0,1\}^N, \boldsymbol{1}^\top \mathbf{w} = 1, \boldsymbol{1}^\top \mathbf{z}  = k, w_i \leq z_i, \forall i \right\}
  $$
  - 该方法本身较难直接求解, 往往通过松弛化为 $\mathbf{z} \in [0,1]^N$ 来求解, 但是其解的质量并没有直接的保证. 一个观察是, relaxation 越松弛, 其产生的"假解"就越多, 就需要更多的后续步骤来进行修正; 反过来, relaxation 越严格, 其求解的难度就越大, 但本身问题就越接近原始整数问题. 

- 为了得到更紧的 formulation, 考虑 complementary formulation:
  $$
  \mathcal{W}_k^C = \left\{\mathbf{w} \in \mathbb{R}_+^N: \mathbf{z} \in \{0,1\}^N, \boldsymbol{1}^\top \mathbf{w} = 1, \boldsymbol{1}^\top \mathbf{z}  = k, w_i(1-z_i) = 0, \forall i \right\}
  $$
  - 该 formulation 的直觉为: 若 $z_i = 0$ 则 $w_i$ 必须为 0, 即对于不购买的资产, 其权重会被强制为 0. 反过来, 若 $z_i = 1$ 则 $w_i$ 可以大于 0, 即对于购买的资产, 其权重可以大于 0.
  - Complementary formulation 比 Big-M 的约束更紧, 然而这也使得优化问题更难以求解. 其强制 feasible region 在一些 $w_i(1-z_i) = 0$ 的边界上, 这往往并不是一个光滑便于优化的区域.


- 为后续处理方便, 这里另外定义 $z_i^c = 1 - z_i$, 表示是否不购买第 $i$ 个资产, 则可以将 complementary formulation 写成
  $$
  \mathcal{W}_k^C = \left\{\mathbf{w} \in \mathbb{R}_+^N: \mathbf{z} \in \{0,1\}^N, \boldsymbol{1}^\top \mathbf{w} = 1, \boldsymbol{1}^\top \mathbf{z}^c  = N-k, w_i z_i^c = 0, \forall i \right\}
  $$


下给出一个一般的优化问题的形式:
$$
\pi^*(\boldsymbol{\xi}) := \min_{\boldsymbol{\omega} \in \Omega} c(\boldsymbol{\omega}, \boldsymbol{\xi})
$$
- $\boldsymbol{\omega} \in \Omega \subseteq \mathbb{R}^n$ 是决策变量. $\boldsymbol{\xi} \in \mathbb{R}^{d_\xi}$ 是问题的相关参数, 例如上文中的 $\Sigma$ 等. $c(\cdot, \cdot) : \mathbb{R}^n \times \mathbb{R}^{d_\xi} \to \mathbb{R}$ 是 cost function. 

在当前文章中, 主要关注最小化投资组合方差的优化问题, 其可以写成
$$
\pi^*(\Sigma) := \min_{\mathbf{w} \in \mathcal{W}_k} \mathbf{w}^\top \Sigma \mathbf{w}
$$
- $\Sigma \in \mathbb{R}^{N \times N}$ 是资产的协方差矩阵, $\mathbf{w} \in \mathbb{R}^N$ 是投资组合的权重, $\mathcal{W}_k$ 是前文定义的 long-only 权重集合.


### Factor models

在金融领域中, 对于 $N$ 只资产, 直接对协方差矩阵 $\Sigma$ 进行估计是非常困难的, 例如
- 数据窗口有限, 估计的噪声大;
- 资产之间的共同波动等因素不容易被捕捉到.

一个常用的建模策略是 factor models, 其假设资产的收益可以由一些共同的且少量的因子来解释. 其建模如下. 
- 最抽象地, 考虑一个抽象的数据结构 $\mathcal{D} = \{ (X^{(j)}, Y^{(j)}) \}_{j=1}^J$, 其中共有 $J$ 条数据示例. 这里引入 $j$ 这个下标主要是考虑到可能有滚动窗口等情况. 
- 具体地, 对于 $i \in \{1, \ldots, N\}$ 只资产, 在时间 $t \in \{1, \ldots, T\}$ 的收益为 $r_{it}$, 通过引入 $P \ll N$ 个因子 $f_{pt}$ 来建模, 则可以写成
  $$
  r_{it} = \alpha_i + \sum_{p= 1}^P \beta_{ip} f_{pt} + \epsilon_{it}
  $$
  - 这里的因子就是比较常见的 Fama-French 五因子等, 比如 `Mkr-RF` (市场超额收益), `SMB` (小盘股因子) 等. 这些数据是我们的输入数据, 在每个时间点 $t$ 对于每个公司 $i$, 其全部的历史的因子数据 $f_{pt}$ 都是已知的. 

- 可以整理成矩阵形式如下. 记 $\mathbf{r}_t \in \mathbb{R}^{N}$ 为时间 $t$ 的全部资产的收益, $\mathbb{f}_t \in \mathbb{R}^P$ 为时间 $t$ 的全部因子的收益, $\boldsymbol{\alpha} \in \mathbb{R}^N$ 为资产的 alpha, $\boldsymbol{\epsilon}_t \in \mathbb{R}^N$ 为时间 $t$ 的误差项, 则可以写成
  $$
  \mathbf{r}_t = \boldsymbol{\alpha} + \mathbf{B} \mathbb{f}_t + \boldsymbol{\epsilon}_t
  $$
  - 其中 $\mathbf{B} \in \mathbb{R}^{N \times P}$ 是资产的 factor loading matrix:
    $$
    \mathbf{B} = \begin{bmatrix}
    \beta_{11} & \beta_{12} & \cdots & \beta_{1P} \\
    \beta_{21} & \beta_{22} & \cdots & \beta_{2P} \\
    \vdots & \vdots & \ddots & \vdots \\
    \beta_{N1} & \beta_{N2} & \cdots & \beta_{NP}
    \end{bmatrix}
    $$
    其第 $i$ 行 $\beta_{i1}, \ldots, \beta_{iP}$ 表示第 $i$ 只资产对于 $P$ 个因子的敏感程度 (或者说暴露程度). 

- 若进一步, $\mathbf{F}^{(j)} \in \mathbb{R}^{T\times P}$ 表示第 $j$ 条 instance 下, 全部时间的全部因子的 factor 矩阵, 对应的收益矩阵 $\mathbf{R}^{(j)} \in \mathbb{R}^{T \times N}$, 则可以写成
  $$
  \mathbf{R}^{(j)} = \mathbf{1}_T \boldsymbol{\alpha}^\top + \mathbf{F}^{(j)} \mathbf{B}^\top + \boldsymbol{\Epsilon}^{(j)} \quad (1)
  $$
  - 其中 $\mathbf{1}_T \in \mathbb{R}^T$ 是全 1 向量, $\boldsymbol{\Epsilon}^{(j)} \in \mathbb{R}^{T \times N}$ 是误差矩阵.

- 对 $(1)$ 求解其协方差矩阵, 则有关系:
  $$
  \Sigma (\mathbf{B}, \boldsymbol{\Psi}; \Sigma_f) = \mathbf{B} \Sigma_f \mathbf{B}^\top + \boldsymbol{\Psi}
  $$
  - 其中 $\Sigma_f \in \mathbb{R}^{P \times P}$ 是因子收益的协方差矩阵, $\boldsymbol{\Psi} \in \mathbb{R}^{N \times N}$ 是误差项的协方差矩阵 (即 $\boldsymbol{\Psi} = \Sigma_\Epsilon$), 并且往往会假设 $\boldsymbol{\Psi}$ 是对角矩阵, 即误差项之间是相互独立的. 
  - 有时为方便起见, 也将 $\boldsymbol{\theta} := \{\mathbf{B}, \boldsymbol{\Psi}\}$ 作为 factor model 的完整参数集合. 注意, 这里的 $\boldsymbol{\theta}$ 是不包含 $\Sigma_f$ 的, 因为 $\Sigma_f$ 是可以通过历史数据直接估计的. 


### Decoupled Optimization

Decoupled optimization 是一种常见的优化策略, 不过也是本文的一个反例. 其思想核心思想即为先预测后决策, 二者分离. 
- 首先, 求解一个预测模型 $\phi: \mathbf{F}^{(j)} \to \mathbf{R}^{(j)}$:
  $$
  \boldsymbol{\theta} = \argmin \mathbb{E}_{(\mathbf{F}, \mathbf{R}) \sim \mathcal{D}} \left[ \ell_a(\phi(\mathbf{F}), \mathbf{R}) \right]
  $$
  从而得到对于 $\mathbf{B}$ 和 $\boldsymbol{\Psi}$ 的估计. 从而根据 $(1)$ 可以得到对于收益率的协方差矩阵的估计 $\hat{\Sigma}_\theta$.

- 其次, 将 $\hat{\Sigma}_\theta$ 作为输入, 求解优化问题:
  $$
  \mathbf{w} := \argmin_{\mathbf{w} \in \mathcal{W}_k} \mathbf{w}^\top \hat{\Sigma}_\theta \mathbf{w}
  $$


这种方法的缺点如下:
- SPO 的文章中曾指出, 预测模型 $\phi$ 的训练目标是 $\ell_a$, 其并不直接关注优化问题的表现. 其是各向同性的, 也就是对于损失函数, 其并不区分不同的预测错误对于优化问题的影响. 然而在优化问题中, 有些资产的预测错误可能会对优化问题的表现产生更大的影响. 
- 此外, 预测步骤本身也是非常复杂的. 例如模型本身的 specification 是否合理, 数据的时间窗口选择等等, 都是非常复杂且敏感的. 然而下层的决策步骤又直接依赖于预测步骤的输出, 这就使得整个 pipeline 的表现非常不稳定. 


## End-to-end Decision-based Learning

E2E 的学习本质在于更换了监督信号, 其直接使用优化问题的表现作为监督信号, 从而直接优化决策的表现. 具体地, 其流程如下:
- 首先根据已有数据 $\mathbf{F}^{(j)}$ 得到 $\Sigma_f$, 并根据 factor model 构造:
  $$
  \hat{\Sigma}_\theta = \Sigma(\mathbf{B}, \boldsymbol{\Psi}; \Sigma_f) = \mathbf{B} \Sigma_f \mathbf{B}^\top + \boldsymbol{\Psi}
  $$
  - 其中这里的参数可以是随机初始化的, 也可以是通过 decoupled optimization 得到的.

- 接着直接求解优化问题:
  $$
  \mathbf{w}^{*(j)} := \argmin_{\mathbf{w} \in \mathcal{W}_k} \mathbf{w}^\top \hat{\Sigma}_\theta \mathbf{w}
  $$

- 最特别的, E2E 会定义一个新的针对优化问题的 loss $\ell_d$, 得到
  $$
  \ell_d(\mathbf{w}^{*(j)}, R^{(j)}) 
  $$
  相当于直接评估我们现在得到的投资组合 $\mathbf{w}^{*(j)}$ 在实际收益 $R^{(j)}$ 上的表现. 


- 最后, somehow 通过反向传播来更新 $\mathbf{B}$ 和 $\boldsymbol{\Psi}$, 从而直接优化 $\ell_d$. 


故总体的 E2E 的流程可以看作是一个三层的 neural network:
$$
\begin{aligned}
\text{Input} &\xrightarrow{F} \text{Covariance Construction Layer}\\& \xrightarrow{\Sigma_\theta} \text{Portfolio Optimization Layer} \xrightarrow{\mathbf{w}^*} \\ &\text{Decision-based Loss Layer} \xrightarrow{\ell_d} \text{Output}
\end{aligned}
$$
- 其中最特殊的就是 Portfolio Optimization Layer, 其本质上是一个 optimization problem, 其输入是 $\hat{\Sigma}_\theta$, 输出是 $\mathbf{w}^*$. 由于 optimization problem 本身是一个 implicit function, 因此如何处理这里的反向传播是一个非常关键的问题. 

总的而言, E2E 的训练目标为:
$$
\begin{aligned}
  \min_{\mathbf{B},\boldsymbol{\Psi}} \quad & \mathbb E[\ell_d(\mathbf{w}^*(\mathbf{B},\boldsymbol{\Psi},\mathbf{F}),\mathbf{R})] \\
  &\text{s.t.} \quad \mathbf{w}^*(\mathbf{B},\boldsymbol{\Psi},\mathbf{F}) \in \argmin_{\mathbf{w} \in \mathcal{W}_k} w\mathbf{w}^\top \Sigma(\mathbf{B},\boldsymbol{\Psi}; \Sigma_f) \mathbf{w}
\end{aligned}
$$
在具体训练过程中, 每次会有一个 batch 的数据 $\mathcal{B}^{(m)}$ 被送入模型 (这里 $\mathcal{B}^{(m)}$ 表示该 batch 的 index), 最终的优化目标为:
$$
\begin{aligned}
  \min_{\mathbf{B},\boldsymbol{\Psi}} \quad & \frac{1}{|\mathcal{B}^{(m)}|} \sum_{j \in \mathcal{B}^{(m)}} \ell_d(\mathbf{w}^*(\mathbf{B},\boldsymbol{\Psi},\mathbf{F}^{(j)}),\mathbf{R}^{(j)}) \\
  &\text{s.t.} \quad \mathbf{w}^*(\mathbf{B},\boldsymbol{\Psi},\mathbf{F}^{(j)}) \in \argmin_{\mathbf{w} \in \mathcal{W}_k} \mathbf{w}^\top \Sigma(\mathbf{B},\boldsymbol{\Psi}; \Sigma_f) \mathbf{w}, \forall j \in \mathcal{B}^{(m)}
\end{aligned}
$$
- 有时约束条件也会 generally  地写作
  $$
  \mathbf{w}^{*(j)} (\boldsymbol{\theta}, \mathbf{F}^{(j)}) = \argmin_{\mathbf{w} \in \mathcal{W}} P_{mV} 