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
  $$

Anyways, 不论如何, 这里最大的一个问题就在于, 如果想要端到端的训练的话, 我们就需要拿到梯度:
$$
\frac{\partial \mathbf{w}^*}{\partial \boldsymbol{\theta}} 
$$
然而由于 $\mathbf{w}^*$ 本身是通过一个 MIP (混合整数规划) 来求解的, 其本身是离散不连续的, 因此无法直接进行反向传播. 故需要将原问题替换成要一个连续的, 凸的, 并且尽量接近原问题的 surrogate problem, 从而进行求解.

### Alternative to Cutting-plane method

现有的关于 E2E 的方法, 往往是通过 cutting-plane method 来求解的, 即通过例如 Big-M 等方法先进行松弛, 然后不断通过额外约束进行修正. 然而本文试图通过另外一种完全不同的技术路径进行求解. 

首先, 考虑 Second-order cone programming (SOCP). 

-  回顾一下, 原问题的 formulation 为
    $$
    \min_{\mathbf{w} \in \mathcal{W}_k} \mathbf{w}^\top \Sigma \mathbf{w} \text{, where  } \mathcal{W}_k = \left\{\mathbf{w} \in \mathbb{R}_+^N, \mathbf{z} \in \{0,1\}^N  : \boldsymbol{1}^\top \mathbf{w} = 1, \|\mathbf{w}\|_0 \leq k\right\}
    $$
    且根据因子模型, $\Sigma$ 可以写成 
    $$
    \Sigma = \mathbf{B} \Sigma_f \mathbf{B}^\top + \boldsymbol{\Psi}
    $$
    因此, 总的优化问题可以写成
    $$
    \min_{\mathbf{w} \in \mathcal{W}_k} \mathbf{w}^\top \mathbf{B} \Sigma_f \mathbf{B}^\top \mathbf{w} + \mathbf{w}^\top \boldsymbol{\Psi} \mathbf{w}
    $$
    - 其中 $\mathbf{z} \in \{0,1\}^N$ 仍然是 binary variable. 
    - 并且, 这里, 前一个部分 $\mathbf{w}^\top \mathbf{B} \Sigma_f \mathbf{B}^\top \mathbf{w}$ 称为是 systematic risk, 是表示共同因子带来的总体系统性风险, 而后一个部分 $\mathbf{w}^\top \boldsymbol{\Psi} \mathbf{w}$ 称为是 idiosyncratic risk, 是表示每个资产自身的特有风险.  

- SOCP 的改进方式是, 虽然我们还是要将 $\mathbf{z}$ 放松到 $[0,1]^N$ 的连续区间, 但是通过引入一个新的辅助变量 $\boldsymbol{\delta} \in \mathbb{R}_+^N$ 进行约束, 并且将 idiosyncratic risk 的部分改写为
  $$
  \operatorname{diag}(\boldsymbol{\Psi})^\top \boldsymbol{\delta} = \sum_{i=1}^N \psi_i^2 \delta_i
  $$
    以及一个额外约束:
    $$
    w_i^2 \leq z_i \delta_i, \forall i
    $$
  - 其中记 $\operatorname{diag}(\boldsymbol{\Psi}) = [\psi_1^2, \ldots, \psi_N^2]^\top$, $\psi_i^2$ 是 $\boldsymbol{\Psi}$ 的第 $i$ 个对角元素.

- 通过上述改写, 可以得到 SOCP 的完整表达式:
  $$\boxed{
  \begin{aligned}
  \min_{\mathbf{w}, \boldsymbol{\delta}, \mathbf{z}} \quad & \mathbf{w}^\top \mathbf{B} \Sigma_f \mathbf{B}^\top \mathbf{w} + \operatorname{diag}(\boldsymbol{\Psi})^\top \boldsymbol{\delta} \\
  \text{s.t.} \quad & \boldsymbol{1}^\top \mathbf{w} = 1, \\
  & \boldsymbol{1}^\top \mathbf{z} \leq k, \\
  &\mathbf{w} \leq \mathbf{z}, \\
  & w_i^2 \leq z_i \delta_i, \forall i, \\
  & \mathbf{z} \leq \mathbf{1}, \\
  & \mathbf{w}, \boldsymbol{\delta}, \mathbf{z} \geq \mathbf{0}. 
  \end{aligned}}
  $$
  - 具体分析这里的约束:
    - $\boldsymbol{1}^\top \mathbf{w} = 1$ 是投资组合权重的约束, 表示所有资产的权重之和必须为 1.
    - $\boldsymbol{1}^\top \mathbf{z} \leq k$ 是原先的 cardinality constraint 的松弛, 但是注意到这里由于 $\mathbf{z}$ 是连续的, 这个约束本身变得不够严格, 这也是 SOCP 要引入额外约束的原因之一.
    - $\mathbf{w} \leq \mathbf{z}$: 回顾, $z_i$ 是表示是否购买第 $i$ 个资产的 binary variable (在 SOCP 中被放松为连续变量), $w_i$ 是第 $i$ 个资产的权重. 该约束的直觉是, 如果 $z_i = 0$ (即不购买第 $i$ 个资产), 则 $w_i$ 必须为 0; 如果 $z_i = 1$ (即购买第 $i$ 个资产), 则 $w_i$ 可以大于 0, 具体配置多少额度则由优化问题来决定. 
    - $w_i^2 \leq z_i \delta_i$: 是为了强化 cardinality constraint 的约束. 该表达式等价于 $\delta_i \geq \frac{w_i^2}{z_i}$, 而另一方面 $\delta_i$ 又出现在最小化的目标函数中, 因此优化问题会倾向于让 $\delta_i$ 尽可能小, 从而对应 $\frac{w_i^2}{z_i}$ 也尽可能小. 这就意味着, 当资产持有变量 $z_i$ 接近于 0 时, 配置权重变量 $w_i$ 也必须接近于 0.


再进一步考虑 Semidefinite programming (SDP).
- 从原始问题的 complementary formulation 出发, 考虑不被选中的资产 $z_i^c = 1 - z_i$, 则原始优化问题可以写成:
  $$
  \begin{aligned}
  \min_{\mathbf{w}, \mathbf{z}} \quad & \mathbf{w}^\top \Sigma \mathbf{w} \\
  \text{s.t.} \quad & \boldsymbol{1}^\top \mathbf{w} = 1, \\
  & \boldsymbol{1}^\top \mathbf{z}^c = N - k, \\
  & w_i z_i^c = 0, \forall i, \\
  & z_i^c = 1 - z_i, \forall i, \\
  & z_i^c \in \{0,1\}, \forall i, 
  \end{aligned}
  $$


- 通过引入一个新的矩阵变量 $\mathbf{W} = \mathbf{w} \mathbf{w}^\top \in [0,1]^{N \times N}$, 可以将目标函数改写为
  $$
  \operatorname{Tr}(\Sigma \mathbf{W}) = \langle \Sigma, \mathbf{W} \rangle = \mathbf{w}^\top \Sigma \mathbf{w}
  $$
  这样就通过引入了一个新矩阵, 将原先的二次项改写成了一个线性项, 从而与 SDP 的目标函数形式相匹配. 不过同时, 矛盾也转移到 $\mathbf{W}$ 上了, 因为 $\mathbf{W}$ 的这个形式注定其是一个 rank-1 的矩阵, 而这样的 rank-1 约束本身是非凸的, 因此, 最终的 relaxation 将不保留其等式, 而是 relax 为
    $$
    \mathbf{W} \succeq \mathbf{w} \mathbf{w}^\top
    $$


- 为了配合 SDP 的形式, 我们同样要对 $z$ 进行松弛话为连续变量, 并类似地对松弛后的 $\mathbf{z}^c$ 进行改写, 引入一个新的矩阵变量 $\mathbf{Z}^c = \mathbf{z}^c (\mathbf{z}^c)^\top \in [0,1]^{N \times N}$. 以及对应地, $\mathbf{Q} = \mathbf{w} (\mathbf{z}^c)^\top \in [0,1]^{N \times N}$, 并且可以通过 $\operatorname{diag}(\mathbf{Q}) = 0$ 来表示 $w_i z_i^c = 0$ 的约束.


- 综上, 我们可以得到 SDP 的完整表达式:
  $$
  \boxed{
  \begin{aligned}
  \min_{\mathbf{U}} \quad & \langle \Sigma, \mathbf{W} \rangle \\
  \text{s.t.} \quad & \boldsymbol{1}^\top \mathbf{w} = 1, \\
  & \boldsymbol{1}^\top \mathbf{z}^c = N - k, \\
  & \operatorname{diag}(\mathbf{Z}^c) = \mathbf{z}^c, \\
  & \operatorname{diag}(\mathbf{Q}) = 0, \\
  & \mathbf{w}, \mathbf{z}^c \geq \mathbf{0}, \\
  & \mathbf{U} = \begin{bmatrix} 1 & \mathbf{w}^\top & (\mathbf{z}^c)^\top \\ \mathbf{w} & \mathbf{W} & \mathbf{Q} \\ \mathbf{z}^c & \mathbf{Q}^\top & \mathbf{Z}^c \end{bmatrix} \succeq 0
  \end{aligned}}
  $$
  - 其中 $\mathbf{U}$ 是一个新的矩阵变量, 其包含了 $\mathbf{w}$, $\mathbf{z}^c$, $\mathbf{W}$, $\mathbf{Z}^c$, $\mathbf{Q}$ 等所有的变量. 优化目标函数的本质还是在说通过改变 $\mathbf{w}, \mathbf{z}^c$ 来优化目标函数.
  - $\langle \Sigma, \mathbf{W} \rangle$ 是 SDP 中的内积表示, 其等价于 $\operatorname{Tr}(\Sigma \mathbf{W})$, 也等价于 $\mathbf{w}^\top \Sigma \mathbf{w}$ (当然这里由于 $\mathbf{W}$ 的定义, 其本质上是 $\mathbf{w}^\top \Sigma \mathbf{w}$ 的一个 relaxation, 因此严谨的讲应当是 $\langle \Sigma, \mathbf{W} \rangle \approx \mathbf{w}^\top \Sigma \mathbf{w}$).
  - $\boldsymbol{1}^\top \mathbf{w} = 1$, $\boldsymbol{1}^\top \mathbf{z}^c = N - k$, $\mathbf{w}, \mathbf{z}^c \geq \mathbf{0}$ 都是和之前相同的约束, 只不过这里的 $\mathbf{z}^c$ 是被松弛为连续变量的.
  - $\operatorname{diag}(\mathbf{Z}^c) = \mathbf{z}^c$ 相当于是之前 $\mathbf{Z}^c = \mathbf{z}^c (\mathbf{z}^c)^\top$ 的一个 relaxation. 观察到, 如果是对于一个 binary variable, 则 $\mathbf{Z}^c = \mathbf{z}^c (\mathbf{z}^c)^\top$ 自然推出 $\operatorname{diag}(\mathbf{Z}^c) = \mathbf{z}^c$. 然而这个关系本身是非凸的. 因此, 这个约束被 relax 成了对于对角线的约束. 这个约束的合理性会最终展现在 $\mathbf{U} \succeq 0$ 的约束中, 稍后会统一分析. 这里可以将这个约束就看做是一个定义式, 其定义了 $\mathbf{Z}^c$ 的对角线元素必须等于 $\mathbf{z}^c$ 的元素, 从而在一定程度上保证了 $\mathbf{Z}^c$ 的结构.
  - $\operatorname{diag}(\mathbf{Q}) = 0$ 是为了保证 $w_i z_i^c = 0$ 的约束. 即这个资产若被选中 (即 $z_i^c = 0$), 则其权重 $w_i$ 可以大于 0; 反过来, 若这个资产未被选中 (即 $z_i^c = 1$), 则其权重 $w_i$ 必须为 0. 不过需要指出这个约束同样也是一个 relaxation, 因为 $\mathbf{Q} = \mathbf{w} (\mathbf{z}^c)^\top$ 同样也是非凸的. 其效力同样会在 $\mathbf{U} \succeq 0$ 的约束中得到体现.
  - $\mathbf{U} \succeq 0$ 是一个半正定约束, 其是一个总的 relaxation. 因为在理想情况下, 应当有
    $$
    \begin{bmatrix} 1 & \mathbf{w}^\top & (\mathbf{z}^c)^\top \\ \mathbf{w} & \mathbf{w} \mathbf{w}^\top & \mathbf{w} (\mathbf{z}^c)^\top \\ \mathbf{z}^c & (\mathbf{w} (\mathbf{z}^c)^\top)^\top & \mathbf{z}^c (\mathbf{z}^c)^\top \end{bmatrix} = \begin{bmatrix} 1 & \mathbf{w}^\top & (\mathbf{z}^c)^\top \\ \mathbf{w} & \mathbf{W} & \mathbf{Q} \\ \mathbf{z}^c & \mathbf{Q}^\top & \mathbf{Z}^c \end{bmatrix}
    $$
    然而 LHS 的形式是非凸的. 因此我们通过 $\mathbf{U} \succeq 0$ 来进行 relaxation. 该约束的合理性在于, 通过 Schur complement 的分析, 可以得到 $\mathbf{U} \succeq 0$ 的约束隐含了 
    $$
    \begin{bmatrix} \mathbf{W} - \mathbf{w} \mathbf{w}^\top & \mathbf{Q} - \mathbf{w} (\mathbf{z}^c)^\top \\ (\mathbf{Q} - \mathbf{w} (\mathbf{z}^c)^\top)^\top & \mathbf{Z}^c - \mathbf{z}^c (\mathbf{z}^c)^\top \end{bmatrix} \succeq 0
    $$
    相当于将矩阵的等式约束放松成了半正定约束. 换言之, 考虑 $\mathbf{U}$ 的子矩阵
    $$
    \mathbf{S} := \begin{bmatrix} \mathbf{W} & \mathbf{Q} \\ \mathbf{Q}^\top & \mathbf{Z}^c \end{bmatrix}
    $$
    则 $\mathbf{U} \succeq 0$ 的约束隐含了 $\mathbf{S} \succeq  0$, 就表示存在要给 PSD slack, 使得
    $$
    \begin{aligned}
    \mathbf{W} &= \mathbf{w} \mathbf{w}^\top + \mathbf{S}_{11} \\
    \mathbf{Q} &= \mathbf{w} (\mathbf{z}^c)^\top + \mathbf{S}_{12} \\
    \mathbf{Z}^c &= \mathbf{z}^c (\mathbf{z}^c)^\top + \mathbf{S}_{22}
    \end{aligned}
    $$
    并且
    $$
    \begin{bmatrix} \mathbf{S}_{11} & \mathbf{S}_{12} \\ \mathbf{S}_{12}^\top & \mathbf{S}_{22} \end{bmatrix} \succeq 0
    $$
    所以总的而言, 我们会有 $\mathbf{W} \approx \mathbf{w} \mathbf{w}^\top$, $\mathbf{Q} \approx \mathbf{w} (\mathbf{z}^c)^\top$, $\mathbf{Z}^c \approx \mathbf{z}^c (\mathbf{z}^c)^\top$. 这就使得 SDP 的 relaxation 更加合理. 还有一种理解方式是, 精确的条件是 $\mathbf{U} \succeq 0$ 且 $\operatorname{rank}(\mathbf{U}) = 1$, 但是由于 rank-1 约束是非凸的, 因此只保留必要条件 $\mathbf{U} \succeq 0$ 来进行 relaxation.

### Framework Architecture

这一小节的主要任务是将上述的 SOCP 和 SDP 的 relaxation 方式对 CvxPyLayers 进行适配, 从而实现 end-to-end 的训练. 


#### Forward pass

CvxPyLayers 是一个基于 CvxPy 的库, 其核心功能是将一个 convex optimization problem 转换成一个 differentiable layer, 从而可以被 end-to-end 的训练. 其核心的一个规则是: Disciplined Parameterized Programming (DPP). 因此, 这部分的主要工作就是在展示如何将上述的 SOCP 和 SDP 的 formulation 转换成 DPP 的形式.

***SOCP***

对于上述的 SOCP formulation, 令 $\mathbf{v} := \mathbf{B}^\top \mathbf{w}$, 以及 $\tilde{\mathbf{v}} := \Sigma_f^{1/2} \mathbf{v}$, 则 
$$
\begin{aligned}
\tilde{\mathbf{v}}^\top \tilde{\mathbf{v}} &= \mathbf{v}^\top \Sigma_f \mathbf{v} = \mathbf{w}^\top \mathbf{B} \Sigma_f \mathbf{B}^\top \mathbf{w}
\end{aligned}
$$

因此, SOCP 的 formulation 可以改写成
$$
\boxed{
\begin{aligned}
\min_{\mathbf{w}, \boldsymbol{\delta}, \mathbf{z}} \quad & \tilde{\mathbf{v}}^\top \tilde{\mathbf{v}} + \operatorname{diag}(\boldsymbol{\Psi})^\top \boldsymbol{\delta} \\
\text{s.t.} \quad & \boldsymbol{1}^\top \mathbf{w} = 1, \\
& \boldsymbol{1}^\top \mathbf{z} \leq k, \\
&\mathbf{w} \leq \mathbf{z}, \\
& w_i^2 \leq z_i \delta_i, \forall i, \\
& \mathbf{v} = \mathbf{B}^\top \mathbf{w}, \\
& \tilde{\mathbf{v}} = \Sigma_f^{1/2} \
& \mathbf{z} \leq \mathbf{1}, \\
& \mathbf{w}, \boldsymbol{\delta}, \mathbf{z} \geq \mathbf{0}. 
\end{aligned}}
$$


***SDP***

回顾, 对于 SDP, 其本身的目标函数设计为:
$$
\min_{\mathbf{U}} \quad \langle \Sigma, \mathbf{W} \rangle
$$
而其中 $\Sigma$ 是通过 factor model 来构造的, 其表达式为
$$
\Sigma = \mathbf{B} \Sigma_f \mathbf{B}^\top + \boldsymbol{\Psi}
$$
因此, 将目标函数展开, 可以得到
$$
\begin{aligned}
\langle \Sigma, \mathbf{W} \rangle &= \langle \mathbf{B} \Sigma_f \mathbf{B}^\top + \boldsymbol{\Psi}, \mathbf{W} \rangle \\
&= \langle \mathbf{B} \Sigma_f \mathbf{B}^\top, \mathbf{W} \rangle + \langle \boldsymbol{\Psi}, \mathbf{W} \rangle \\
&= \langle \Sigma_f, \mathbf{B}^\top \mathbf{W} \mathbf{B} \rangle + \langle \boldsymbol{\Psi}, \mathbf{W} \rangle
\end{aligned}
$$
故类似地, 引入 $\mathbf{\tilde{V}}:= \mathbf{W} \mathbf{B}$, 与 $\mathbf{V} := \mathbf{B}^\top \mathbf{\tilde{V}}= \mathbf{B}^\top \mathbf{W} \mathbf{B}$, 则
$$
\begin{aligned}
\langle \Sigma, \mathbf{W} \rangle &= \langle \Sigma_f, \mathbf{V} \rangle + \langle \boldsymbol{\Psi}, \mathbf{W} \rangle
\end{aligned}
$$
故最终的 SDP formulation 可以写成
$$
\boxed{
\begin{aligned}
\min_{\mathbf{U}, \mathbf{V}, \mathbf{\tilde{V}}} \quad & \langle \Sigma_f, \mathbf{V} \rangle + \langle \boldsymbol{\Psi}, \mathbf{W} \rangle \\
\text{s.t.} \quad & \boldsymbol{1}^\top \mathbf{w} = 1, \\
& \boldsymbol{1}^\top \mathbf{z}^c = N - k, \\
& \operatorname{diag}(\mathbf{Z}^c) = \mathbf{z}^c, \\
& \operatorname{diag}(\mathbf{Q}) = 0, \\
& \mathbf{w}, \mathbf{z}^c \geq \mathbf{0}, \\
& \mathbf{V} = \mathbf{B}^\top \mathbf{\tilde{V}}, \\
& \mathbf{\tilde{V}} = \mathbf{W} \mathbf{B}, \\
& \mathbf{U} = \begin{bmatrix} 1 & \mathbf{w}^\top & (\mathbf{z}^c)^\top \\ \mathbf{w} & \mathbf{W} & \mathbf{Q} \\ \mathbf{z}^c & \mathbf{Q}^\top & \mathbf{Z}^c \end{bmatrix} \succeq 0
\end{aligned}}
$$

---

因此, 总的而言, 通过上述的改写, 我们总的 E2E 的范式如下. 

- 通过上层神经网络, 我们可以得到 $\mathbf{B}$ 和 $\boldsymbol{\Psi}$ 的估计.
- 给定当前 iteration 下的 $\mathbf{B}, \boldsymbol{\Psi}$, 以及历史数据 $\mathbf{F}^{(j)}$, 我们可以构造 DPP 表示下的优化问题
- 通过 CvxPyLayers 来求解该优化问题, 从而得到 $\mathbf{w}^*$. 并通过和真实收益 $\mathbf{R}^{(j)}$ 的比较来得到 loss $\ell_d(\mathbf{w}^*, \mathbf{R}^{(j)})$.
- 最后通过反向传播来更新 $\mathbf{B}$ 和 $\boldsymbol{\Psi}$, 例如 
  $$
  \mathbf{B} \leftarrow \mathbf{B} - \hat{\psi} \frac{\partial \ell_d}{\partial \mathbf{B}}, \quad \boldsymbol{\Psi} \leftarrow \boldsymbol{\Psi} - \hat{\psi} \frac{\partial \ell_d}{\partial \boldsymbol{\Psi}}
  $$


特别地, 这里 $\mathbf{B}$ 是任意的自由参数, 可以正常训练. $\boldsymbol{\Psi}$ 则需要保证其是一个非负对角矩阵(因为其含义表示的是协方差), 这里的处理方法为, 我们不去直接估计这样一个对角矩阵, 而是形式化地首先令
$$
\boldsymbol{\Psi} = \begin{bmatrix} \exp{2\hat{\psi}_1} & 0 & \ldots & 0 \\ 0 & \exp{2\hat{\psi}_2} & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots & \exp{2\hat{\psi}_N} \end{bmatrix}
$$
然后我们只需要去估计 $\hat{\boldsymbol{\psi}} = [\hat{\psi}_1, \ldots, \hat{\psi}_N]^\top$ 这个向量即可. 后面在 inference 的时候, 再通过取指数并拼接在一起的方式来得到一个一定符合要求的对角矩阵 $\boldsymbol{\Psi}$. 

---

这里完整将数据的计算 pipeline 整理如下. 注意, 这个部分的 notation 可能与上方有少许出入, 主要是为了更好地表达数据的计算流程. 不过这个板块自己的 notation 是 self-contained 的. 

假设共有 $N$ 个资产, 每个资产都有共同的 $P$ 个因子, 一共有 $T$ 期交易日. 对于第 $j$ 笔数据 (这里 $j$ 主要是因为比如我们可以通过滑动窗口等方法构造出多个数据点, 不过不重要, $j$ 就是最小的单次训练的最小数据单位):

首先, 我们完整的已知数据就是如下两个:

- 已知的输入特征为 $\mathbf{F}^{(j)} = [\mathbf{f}_1^{(j)}, \ldots, \mathbf{f}_P^{(j)}] \in \mathbb{R}^{T \times P}$, 表示第 $j$ 笔数据中对应的公共 factor 暴露. 注意, 这里的 $\mathbf{f}_P^{(j)} \in \mathbb{R}^T$ 是一个时间序列, 表示第 $P$ 个 factor 随着时间变化数据. 并且, 这里的 factor 是公共的, 是对于所有资产都一样的. 因此这里并没有一个关于资产维度 $N$ 的输入特征, 只有一个关于时间维度 $T$ 和 factor 维度 $P$ 的输入特征.
- $\mathbf{R}^{(j)} \in \mathbb{R}^{T \times N}$, 表示第 $j$ 笔数据中各个资产的收益率. 这个是因资产而异的. 

根据已知输入, 我们可以构建:
- $\boldsymbol{\Sigma}_f^{(j)} := \operatorname{Cov}(\mathbf{F}^{(j)}) \in \mathbb{R}^{P \times P}$, 表示第 $j$ 笔数据中 factor 的协方差矩阵.

还有一些中间过程变量将随着 forward pass 逐步构造. 故下正式开始前向传播. 

1. 首先根据已知数据计算 $\boldsymbol{\Sigma}_f^{(j)}$.

2. 根据上一部迭代(或初始化), 得到参数矩阵 (或称为因子载荷, factor loading) $\mathbf{B} \in \mathbb{R}^{N \times P}$ 和残差的协方差元素 $\boldsymbol{\hat{\psi}} \in \mathbb{R}^N$. 这里, $\boldsymbol{\hat{\psi}}$ 和 $\mathbf{B}$ 是神经网络将要更新的参数, 故统一记为 $\boldsymbol{\Theta} := \{\mathbf{B}, \boldsymbol{\hat{\psi}}\}$. 其各自的含义为:
     - $\mathbf{B}$ 是一个全局共享的参数, 其是与 $j$ 无关的, 相当于是我们用多笔训练数据在一直更新这个 $\mathbf{B}$. 其第 $i$ 行表示第 $i$ 个资产对于各个 factor 的暴露程度, 即权重系数. 因此, 每个资产的 factor model 可以写成
        $$
        R_i =   \alpha_i + \sum_{p=1}^P B_{ip} f_p + \epsilon_i
        $$ 
        或等价地用矩阵的形式写成
        $$
        \mathbf{R}^{(j)}_{T\times N} = \boldsymbol{\alpha}\mathbf{1}^\top + \mathbf{F}^{(j)}_{T\times P} (\mathbf{B}^\top)_{P \times N} + \boldsymbol{\Epsilon}_{T \times N}
        $$
        - 得到协方差关系为:
          $$
          \operatorname{Var}(\mathbf{R}^{(j)}) = \mathbf{B} \operatorname{Var}(\mathbf{F}^{(j)}) \mathbf{B}^\top + \operatorname{Var}(\boldsymbol{\epsilon}) \iff 
          \Sigma_\Theta = \mathbf{B} \Sigma_f \mathbf{B}^\top + \boldsymbol{\Psi} \in \mathbb{R}^{N \times N}
          $$
    - $\boldsymbol{\hat{\psi}} \in \mathbb{R}^N$ 是用来构造 idiosyncratic risk 的元素. 这里, 由于我们假设上述误差 $\epsilon_i$ 是独立的, 因此 idiosyncratic risk 的协方差矩阵 $\operatorname{Var}(\boldsymbol{\varepsilon}) := \boldsymbol{\Psi} \in \mathbb{R}^{N \times N}$ 就是一个对角矩阵: $\boldsymbol{\Psi}  := \operatorname{diag}(\psi_1^2, \ldots, \psi_N^2)$. 不过直接通过神经网络训练这样一个结构特殊的正对角阵是较困难的, 这里的策略就是训练 $N$ 个自由变量 $\boldsymbol{\hat{\psi}} = [\hat{\psi}_1, \ldots, \hat{\psi}_N]^\top$, 然后通过取指数的方式拼接成一个恒为正的对角矩阵:
      $$
      \boldsymbol{\Psi} = \begin{bmatrix} \exp{2\hat{\psi}_1} & 0 & \ldots & 0 \\ 0 & \exp{2\hat{\psi}_2} & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots & \exp{2\hat{\psi}_N} \end{bmatrix}
      $$

3. 将 $\mathbf{B}$, $\boldsymbol{\Psi}$, $\boldsymbol{\Sigma}_f^{(j)}$ 以及 cardinality constraint $k$ 送入 CvxPyLayers 中, 这里以 SOCP-DPP 为例， 则可以通过求解器求解如下优化问题：
    $$
    \begin{aligned}
    & \min_{\mathbf{w}, \mathbf{z}, \mathbf{v}, \tilde{\mathbf{v}}, \boldsymbol{\delta}} \quad &&\tilde{\mathbf{v}}^\top \tilde{\mathbf{v}} + \operatorname{diag}(\boldsymbol{\Psi})^\top \boldsymbol{\delta} \\
    & \text{subject to} && \boldsymbol{1}^\top \mathbf{w} = 1, \\
    &&& \boldsymbol{1}^\top \mathbf{z} \leq k, \\
    &&& \mathbf{w} \leq \mathbf{z}, \\
    &&& w_i^2 \leq z_i \delta_i,  i = 1, \ldots, N, \\
    &&& \mathbf{v} = \mathbf{B}^\top \mathbf{w}, \\
    &&& \tilde{\mathbf{v}} = \Sigma_f^{1/2} \mathbf{v}, \\
    &&& \mathbf{z} \leq \mathbf{1}, \\
    &&& \mathbf{w}, \boldsymbol{\delta}, \mathbf{z} \geq \mathbf{0}.
    \end{aligned}
    $$
    其中
    - $\mathbf{w} \in \mathbb{R}^N$ 是最终求解得到的投资组合权重 (long-only)
    - $\mathbf{z} \in [0,1]^N$ 是松弛后的 binary variable, 其元素 $z_i$ 的值越接近于 1, 则表示第 $i$ 个资产被选中的可能性越大. 
    - $\boldsymbol{1}^\top \mathbf{z} \leq k$ 的约束表示被选中的资产数量不能超过 $k$. 由于 $\mathbf{z}$ 是连续的, 因此这个约束本身是一个 relaxation.
    - $\mathbf{w} \leq \mathbf{z}$ 的约束表示, 如果 $z_i$ 接近于 0 (即不选中第 $i$ 个资产), 则 $w_i$ 必须接近于 0; 如果 $z_i$ 接近于 1 (即选中第 $i$ 个资产), 则 $w_i$ 可以大于 0. 这也是为了强化 cardinality constraint 的约束.
    - $\mathbf{v} = \mathbf{B}^\top \mathbf{w} \in \mathbb{R}^P$ 是一个新的变量, 表示若按照当前 portfolio 的权重配置 $\mathbf{w}$ 来计算, 则每个因子的暴露程度. 
    - $\tilde{\mathbf{v}} = \Sigma_f^{1/2} \mathbf{v} \in \mathbb{R}^P$ 从金融上看相当于是进一步对 $\mathbf{v}$ 进行风险 (标准差) 调整后的暴露程度. 在数学上, 其作用是将 $\tilde{\mathbf{v}}^\top \tilde{\mathbf{v}} \equiv \mathbf{w}^\top \mathbf{B} \Sigma_f \mathbf{B}^\top \mathbf{w}$ 这个二次项改写成了 $\tilde{\mathbf{v}}^\top \tilde{\mathbf{v}}$ 这个形式, 从而满足了 DPP 的要求.
    - $\boldsymbol{\delta} \in \mathbb{R}_+^N$ 是残差风险的一个辅助变量. 回顾, 正常的总 portfolio 中, 以 $\mathbf{w}$ 的权重配置, 其 idiosyncratic risk 的部分是 $\mathbf{w}^\top \boldsymbol{\Psi} \mathbf{w} = \sum_{i=1}^N w_i^2 \psi_i^2$. 但 SOCP 的改进方式是, 通过引入 $\boldsymbol{\delta}$ 来进行约束, 从而将 idiosyncratic risk 的部分改写为 $\operatorname{diag}(\boldsymbol{\Psi})^\top \boldsymbol{\delta} = \sum_{i=1}^N \psi_i^2 \delta_i$, 并且通过 $w_i^2 \leq z_i \delta_i$ 来强化 cardinality constraint 的约束 ($\operatorname{diag}(\boldsymbol{\Psi})^\top \boldsymbol{\delta} = \sum_{i=1}^N \psi_i^2 \delta_i \leq \sum_{i=1}^N \psi_i^2 \frac{w_i^2}{z_i}$, 从而当 $z_i$ 接近于 0 时, $w_i$ 也必须接近于 0).

4. 通过求解器求解上述优化问题, 得到 $\mathbf{w}^*$. 这里的 $\mathbf{w}^*$ 就是当前 iteration 下的投资组合权重. 故通过当前的真实收益 $\mathbf{R}^{(j)}$ 和 $\mathbf{w}^*$ 来计算 portfolio 的真实收益:
    $$
    \mathbf{r}_{\text{portfolio}}^{(j)} = \mathbf{R}^{(j)} \mathbf{w}^* \in \mathbb{R}^T
    $$
    就表示按照当前的投资组合权重 $\mathbf{w}^*$ 来配置资产, 在第 $j$ 笔数据中得到的 portfolio 的收益率 (注意还是同样的这 $T$ 天的回测结果). 
    Anyway, 通过 $\mathbf{r}_{\text{portfolio}}^{(j)}$ 就可以定义一些 loss function 以进行后续的反向传播, 这里先笼统地记为:
    $$
    \mathcal{L}^{(j)} = \ell_d(\mathbf{R}^{(j)}, \mathbf{w}^*) 
    $$


所以这个部分整体的 pytorch 风格伪代码如下:

```python
B = nn.Parameter(...)
psi_hat = nn.Parameter(...)

Sigma_f = covariance(F_j)          # deterministic
Psi = diag(exp(psi_hat)**2)        # uses trainable parameter
w_star = cvxpylayer(Sigma_f, B, Psi)
loss = sharpe_loss(w_star, R_j)
loss.backward()
optimizer.step()
```

#### Loss function design

文中讨论了一些 loss 的设计. 最终其选择的时候 realized Sharpe ratio 作为 loss function. 在我们得到  portfolio 的配置权重 $\mathbf{w}^*$ 之后, 计算在历史 $T$ 天的回测收益率 
$$
\mathbf{r}_{\text{portfolio}}^{(j)} = \mathbf{R}^{(j)} \mathbf{w}^* \in \mathbb{R}^T
$$
然后通过 $\mathbf{r}_{\text{portfolio}}^{(j)}$ 来计算 realized Sharpe ratio, 其定义为
$$
\operatorname{Sharpe}(\mathbf{r}_{\text{portfolio}}^{(j)}):= \frac{\left(1+ \mu(\mathbf{r}_{\text{portfolio}}^{(j)})\right)^{365}}{\sqrt{256} \cdot \sigma(\mathbf{r}_{\text{portfolio}}^{(j)})}
$$
- $\mu(\mathbf{r}_{\text{portfolio}}^{(j)})$ 是 $\mathbf{r}_{\text{portfolio}}^{(j)}$ 的几何平均 (更接近复利的概念):
    $$
  \mu(\mathbf{r}_{\text{portfolio}}^{(j)}) = \left(\prod_{t=1}^T (1 + r_{\text{portfolio}, t}^{(j)})\right)^{\frac{1}{T}} - 1
  $$

- $\sigma(\mathbf{r}_{\text{portfolio}}^{(j)})$ 是 $\mathbf{r}_{\text{portfolio}}^{(j)}$ 的标准差.
    $$
    \sigma(\mathbf{r}_{\text{portfolio}}^{(j)}) = \sqrt{\frac{1}{T-1} \sum_{t=1}^T (r_{\text{portfolio}, t}^{(j)} - \bar{r}_{\text{portfolio}}^{(j)})^2}
    $$
    其中 $\bar{r}_{\text{portfolio}}^{(j)}$ 是 $\mathbf{r}_{\text{portfolio}}^{(j)}$ 的算术平均.

- 365 和 256 是为了将 daily 的 Sharpe ratio annualize. 365 是因为一年有 365 天; 256 是因为一年有大约 256 个交易日 (扣除周末和节假日). 但是这里感觉确实处理的不太一致. 

Sharpe ratio 的定义本身是为了衡量一个投资组合的风险调整后的收益率. 因此, 通过最大化 Sharpe ratio 来训练模型, 就相当于在训练过程中直接优化投资组合的风险调整后的表现. 不过为了和习惯相配合, 我们往往最小化负的 Sharpe ratio, 并定义当前 batch $m$ 的 loss function 为
$$
\mathcal{L}^{(m)} = - \frac{1}{|B|} \sum_{j \in B} \operatorname{Sharpe}(\mathbf{r}_{\text{portfolio}}^{(j)})
$$



#### Backward pass

反向传播的终极目标是计算
$$
\frac{\partial \mathcal{L}^{(j)}}{\partial \boldsymbol{\Theta}} = \left[
\frac{\partial \mathcal{L}^{(j)}}{\partial \boldsymbol{B}}, \quad \frac{\partial \mathcal{L}^{(j)}}{\partial \boldsymbol{\hat{\psi}}}
\right]
$$
以进行梯度更新. 故根据链式法则, 需要计算
$$
\frac{\partial \mathcal{L}^{(j)}}{\partial \mathbf{w}^*} \cdot \frac{\partial \mathbf{w}^*}{\partial \boldsymbol{\Theta}}
$$
前者的计算是比较直接的, 因为 $\mathcal{L}^{(j)}$ 稍后可以看到就是一个关于 $\mathbf{w}^*$ 的函数. 而后者的计算是比较复杂的, 我们要求解当模型的参数 $\boldsymbol{\Theta}  = \{\mathbf{B}, \boldsymbol{\hat{\psi}}\}$ 发生微小变化时, 最优解 $\mathbf{w}^*$ 的变化率. 

这里更为困难的是第二部分的求解, 因为这里的 $\mathbf{w}^*$ 是通过求解一个 optimization problem 得到的:
$$
\mathbf{w}^* = \arg\min_{\mathbf{w}} P^{\text{DPP}}(\boldsymbol{\Theta})
$$
而这一步的反向传播从直观上看, 是通过 CvxPyLayers 来实现的. Generally, 对于这种优化层的反向传播, 其一般原理为: 既然能够求解这样的优化问题, 就一定满足某些最优性条件 (如 KKT 等), 而这样的条件一定是包含了问题参数 $\boldsymbol{\Theta}$ 和最优解 $\mathbf{w}^*$ 的, 故通过最优性条件, 就得到了 $\mathbf{w}^*$ 和 $\boldsymbol{\Theta}$ 之间的关系, 从而可以通过 implicit differentiation 来计算.  

具体而言, 在本文中, 只要我们把内层的优化问题写成一个 DPP 的形式, 那么 CvxPyLayers 就可以将这个形式翻译成 Cone Program 的形式, 从而通过 [Agrawal, A., Barratt, S., Boyd, S., Busseti, E., & Moursi, W. M. (2019). Differentiating through a cone program. Journal of Applied and Numerical Optimization, 1(2), 107–115] 中的方法, 在 forward pass 中求解 $\mathbf{w}^*$ 的同时, 也会在 backward pass 中求解 $\frac{\partial \mathbf{w}^*}{\partial \boldsymbol{\Theta}}$. (虽然说在底层实现上, 并不会真的显式求出 $\frac{\partial \mathbf{w}^*}{\partial \boldsymbol{\Theta}}$, 再乘进 $\frac{\partial \mathcal{L}^{(j)}}{\partial \mathbf{w}^*}$ 来得到 $\frac{\partial \mathcal{L}^{(j)}}{\partial \boldsymbol{\Theta}}$, 而是通过一些线性代数的操作 (如 vector-Jacobian product) 来直接得到 $\frac{\partial \mathcal{L}^{(j)}}{\partial \boldsymbol{\Theta}}$ 的值, 但从数学上讲是等价的).


## Numerical Experiments

### Dataset and Experimental Setup

- Dataset: 
  - 选择了 S&P 500 中 $N=50$ 个成分股. 总体时间跨度为 2010-01-01~2021-12-31. 原始数据为 daily stock price. 
  - 股票代码为: CVX, HES, OXY, SO, BALL, ECL, VMC, FDX, LMT, MMM, RHI, UPS, AMZN, AZO, BBWI, F, HAS, YUM, CPB, EL, MKC, PEP, PM, A, ABC, BIIB, CVS, DGX, JNJ, SYK, AIG, BAC, PGR, SCHW, WFC, AAPL, AKAM, CRM, CTSH, MA, ORCL, DIS, EA, T, AES, CMS, DUK, EQR, PLD, SPG
  - $k \in \{10, 15, 20\}$, 即 cardinality constraint 的值.

  
- 因子选择: Fama-French 5 因子模型, 即 $P=5$. 这 5 个因子分别是: Mkt-RF (市场风险溢价), SMB (规模因子), HML (价值因子), RMW (盈利能力因子), CMA (投资风格因子). 


- Compared models: 
  - E2E Big-M: 直接将 binary variable $z_i$ relax 成 $0 \leq z_i \leq 1$
  - E2E SOCP
  - E2E SDP
  - linReg: 传统的先预测后决策模型. 第一步先用历史数据估计 $\mathbf{B}$ 和 $\boldsymbol{\Psi}$, 并估计 $\Sigma = \mathbf{B} \Sigma_f \mathbf{B}^\top + \boldsymbol{\Psi}$; 第二步将 $\Sigma$ 送入一个传统的 portfolio optimization problem 中求解 $\mathbf{w}$. 这里由于不需要反向传播, 因此可以使用标准的 Big-M MIP / MIQP 直接通过调用 Gurobi 等求解器来求解. 
  - Nominal: 不学习 $\mathbf{B}$ 和 $\boldsymbol{\Psi}$, 直接用历史数据来估计 $\Sigma$, 然后送入 portfolio optimization problem 中求解 $\mathbf{w}$. 


- 训练流程:
  - 在训练过程中, 每次会用五年的 weekly return (例如第一次为 2010 Q1 ~ 2014 Q4) 来训练参数 $\boldsymbol{\Theta} = \{\mathbf{B}, \boldsymbol{\hat{\psi}}\}$. 稍后会提到, 细节上这里还会用 bootstrap 的方式来构造多个训练数据点. 不过总而言之, 我们是首先站在 2015 Q1 的时间点, 首先选择了 2010 Q1 ~ 2014 Q4 的数据得到了一个关于 $\boldsymbol{\Theta}$ 的估计 $\mathbf{B}^*, \boldsymbol{\hat{\psi}}^*$. 
  - 接着, 同样计算 2010 Q1 ~ 2014 Q4 的数据的 factor covariance $\Sigma_f$, 以及通过 $\mathbf{B}^*, \boldsymbol{\hat{\psi}}^*$ 来构造 $\Sigma_{\Theta^*} = \mathbf{B}^* \Sigma_f (\mathbf{B}^*)^\top + \boldsymbol{\Psi}^*$, 其中 $\boldsymbol{\Psi}^* = \operatorname{diag}(\exp(2\hat{\psi}_1^*), \ldots, \exp(2\hat{\psi}_N^*))$. 
  - 最终, 所有模型将各自的 $\Sigma$ 送入标准的 portfolio optimization problem 通过 Gurobi 进行标准化求解, 从而得到 $\mathbf{w}^*$. 这里的 $\mathbf{w}^*$ 就是当前时间点 (2015 Q1) 下的投资组合权重:
      $$
      \mathbf{w}^* = \arg\min_{\mathbf{w}} \mathbf{w}^\top \Sigma \mathbf{w}, \quad \text{s.t.} \quad \boldsymbol{1}^\top \mathbf{w} = 1, \quad \mathbf{w} \geq \mathbf{0}, \quad \|\mathbf{w}\|_0 \leq k
      $$

- 测试流程:
  - 在测试过程中, 在得到当前模型的持仓权重 $\mathbf{w}^*$ 之后, 将各自买入并持有一个季度, 并计算这期间的 daily return 作为当前季度的回测结果: $\mathbf{r}_{\text{portfolio}} = [r_1, \ldots, r_T]^\top$. 这里的 $T$ 是当前季度的交易日数量. 并且假设初始财富 $W = 1000000$.  最终文中关注如下几个指标:
    - Average Return: $\mu(\mathbf{r}_{\text{portfolio}}) = \left(\prod_{t=1}^T (1 + r_t)\right)^{\frac{1}{T}} - 1$
    - Annualized Return: $\left(1 + \mu(\mathbf{r}_{\text{portfolio}})\right)^{365}$ (并且假设 risk free rate = 0)
    - Annualized Volatility: $\sqrt{256} \cdot \sigma(\mathbf{r}_{\text{portfolio}})$
    - Sharpe Ratio: $\operatorname{Sharpe}(\mathbf{r}_{\text{portfolio}}) = \frac{\left(1 + \mu(\mathbf{r}_{\text{portfolio}})\right)^{365}}{\sqrt{256} \cdot \sigma(\mathbf{r}_{\text{portfolio}})}$
    - Max Drawdown: $\operatorname{MDD}(\mathbf{r}_{\text{portfolio}}) = \max_{t \in [T]} \frac{\max_{s \in [t]} W_s - W_t}{\max_{s \in [t]} W_s}$, 其中 $W_t$ 是按照 $\mathbf{w}^*$ 来配置资产, 在第 $t$ 天的财富水平. 这里的 MaxDD 是一个衡量投资组合在回测期间最大回撤程度的指标, 数值越小越好.
    - VaR: $\operatorname{VaR}(\mathbf{r}_{\text{portfolio}}) = -\inf \{x \in \mathbb{R} : P(\mathbf{r}_{\text{portfolio}} \leq x) > 0.05\}$, 这里的 VaR 是一个衡量投资组合在回测期间潜在损失风险的指标, 数值越小越好.
    - CVaR: $\operatorname{CVaR}(\mathbf{r}_{\text{portfolio}}) = -\frac{1}{0.05} \int_{-\infty}^{\operatorname{VaR}(\mathbf{r}_{\text{portfolio}})} x \cdot f_{\mathbf{r}_{\text{portfolio}}}(x) dx$, 这里的 CVaR 是一个衡量投资组合在回测期间潜在损失风险的指标, 数值越小越好. 这里的 CVaR 可以看作是 VaR 的一个改进版本, 因为它不仅考虑了 VaR 的损失水平, 还考虑了在 VaR 水平以下的损失分布情况, 从而提供了一个更全面的风险评估.
    - LPM2: $\operatorname{LPM2}(\mathbf{r}_{\text{portfolio}}) = \frac{1}{T} \sum_{t=1}^T \max(0, r_{\text{target}} - r_t)^2$, 这里的 LPM2 是一个衡量投资组合在回测期间潜在损失风险的指标, 数值越小越好. 这里的 LPM2 可以看作是 VaR 和 CVaR 的一个改进版本, 因为它不仅考虑了 VaR 的损失水平, 还考虑了在 VaR 水平以下的损失分布情况, 从而提供了一个更全面的风险评估. 这里的 $r_{\text{target}}$ 是一个预设的目标收益率水平, 可以根据实际情况进行调整.
    - Turnover: $\operatorname{Turnover}(\mathbf{w}^*) = \sum_{i=1}^N |w_i^* - w_i^{\text{prev}}|$, 这里的 Turnover 是一个衡量投资组合在回测期间交易频率和成本的指标, 数值越小越好. 这里的 $w_i^{\text{prev}}$ 是上一季度的投资组合权重, 可以通过滚动窗口的方式来计算.
  - 最终, 该窗口将每次向前滚动一个季度, 例如下一次为 2010 Q2 ~ 2015 Q1 的数据来训练 $\boldsymbol{\Theta}$, 得到 $\mathbf{w}^*$, 并在 2015 Q2 上进行回测. 


- Circular Block Bootstrap:
  - 对于传统的 Baseline Method, 我们往往只需要用一次 2010 Q1 ~ 2014 Q4 的数据来估计 $\Sigma$, 从而求解 $\mathbf{w}^*$. 但是对于 E2E Method, 由于其训练的方式是通过不断地更新 $\boldsymbol{\Theta}$ 来优化 loss function, 因此我们需要构造多个训练数据点来进行训练. 这里的构造方法为 Circular Block Bootstrap.
  - CBB 是一种对于时间序列进行 bootstrap 的方法. 例如, 考虑抽象意义上我们有时间序列 $X_{1:T}$, 其长度为 $T$. CBB 的方法是, 对整个序列以长度 $b$ 的 block 来进行划分, 从而得到 $T/b$ 个 block:
      $$
      X_{1:T} = [X_{1:b}, X_{b+1:2b}, \ldots, X_{T-b+1:T}] := [B_1, B_2, \ldots, B_{T/b}]
      $$
      接着, 对于这些 block, 进行有放回的抽样, 从而得到一个新的 bootstrap 样本:
      $$
      \tilde{X}_{1:T}^{(j)} = [B_{i_1}, B_{i_2}, \ldots, B_{i_{T/b}}]
      $$
  - 其本身具有两个特点: 1. 由于是 block 的方式进行抽样, 因此可以在一定程度上保留时间序列的因果先后特征, 不过确实需要承认在 block 之间的连接处会存在一些不连续性; 2. 由于是有放回的抽样, 因此这不是一个 shuffle 的过程, 而是一个真正意义上的 bootstrap, 从而可以构造出多个训练数据点来进行训练, 使得训练出的 $\Sigma$ 具有变异性. 并且 bootstrap 保证了新样本的长度和原始样本的长度是一样的, 从而可以直接送入模型进行训练.
  - 在本文中, 一共生成了 $J = 2000$ 条序列, block 的长度 $b=20$. 

- 其余训练配置与实验细节
  - epoch = 4
  - 初始 Learning rate = 0.01, 最后一个 epoch 为 0.001, optimizer 就是 SGD:
      $$
      \boldsymbol{\Theta} \leftarrow \boldsymbol{\Theta} - \gamma\nabla_{\boldsymbol{\Theta}} \mathcal{L}^{(m)}
      $$
      其中 $\gamma$ 是 learning rate, $\mathcal{L}^{(m)}$ 是当前 batch 的 average loss.

  - E2E 的初始化会选择 linReg 的结果来进行初始化, 以加速训练. 


### Results

#### In-Sample

- E2E 确实改变了 $\Sigma$ 的结构:
  ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260513144334.png)


- E2E 确实观察到了 loss (negative Sharpe ratio) 的下降, 并且由于 warm-start 的缘故, 基本 4 个 epoch 就已经收敛了.
  ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260513144424.png)


- 不同的模型的训练耗时差异巨大. Big-M 基本 15 秒一个 epoch, SOCP 基本 1 min 一个 epoch, SDP 需要超过 6 h 一个 epoch. 
  ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260513144559.png)


- 下面是比较的主表格, 分为上中下三个 panel, 分别比较了 variance, average return 和 sharpe ratio. 左中右三列表示不同 $k$ setting 下的结果. 对于每个基本子表, 其第 $i$ 行 $j$ 列处的数值形如 row-wins \ column-wins, 其中 row-wins 表示当前行的模型在滚动的 24 期回测中, 有多少期的表现优于当前列的模型; column-wins 则反之. 
  ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260513150129.png)
  该表格的 takeaway 结论如下.
    - E2E 的方法全面优于 linReg
    - E2E Big-M 的方法往往 return 和 Sharpe 更好, 但方差更高

### Out-of-Sample

在样本外测试结果如下:
- E2E 三种方法整体在 linreg 上方, 多数情况下也优于 nominal
  ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260513150918.png)
- 整体的风险和收益的原始指标中:
  - E2E 全面优于 linreg;
  - E2E 相比于 nominal, 其 nominal return 普遍更高, 且 volatility 普遍更小; 然而一般 nominal 的 VaR 和 CVaR 更好, E2E 的 LPM2 和 MaxDD 更好. 
  - E2E 内部, 仍然是 Big-M 的 return 和 Sharpe 更好, 但方差往往更高. 
- 对于风险调整后指标, 例如 Sharpe 等, 
  - E2E 全面显著优于 linreg, 并且也优于 nominal. 
  - 这里还引入了 information ratio, 是相当于以 nominal 为 benchmark (因为其不适用 factor model, 直接用历史数据来估计 $\Sigma$), 来衡量 E2E 相对于 nominal 的风险调整后超额收益. 结果显示, E2E 的 information ratio 也普遍是正的, 而 linreg 的 information ratio 则普遍是负的.

