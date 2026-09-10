# A Barrier-Metric First-Order Method for Linearly Constrained Bilevel Optimization

## Introduction

考虑如下双层优化问题:
$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^{d_\mathbf{x}}} & \quad F(\mathbf{x}) := f(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})) \qquad && \text{(Upper)} \\
\text{s.t.} & \quad \mathbf{y}^\star(\mathbf{x}) \in \arg\min_{\mathbf{y} \in \mathcal{Y}} g(\mathbf{x}, \mathbf{y}) \qquad &&\text{(Lower)} \\
\end{aligned}
$$

这里主要研究下层问题由于一些线性约束而导致的非光滑性问题. 因此假设上层问题是无约束的, 且最优值是有限的, 即 $F^\star := \min_{\mathbf{x} \in \mathbb{R}^{d_\mathbf{x}}} F(\mathbf{x}) > -\infty$. 考虑下层问题的约束集 $\mathcal{Y}$ 是一个凸多面体, 即
$$
\mathcal{Y} := \{\mathbf{y} \in \mathbb{R}^{d_\mathbf{y}}: \mathbf{A}\mathbf{y} \le \mathbf{b}\} = \{\mathbf{y} \in \mathbb{R}^{d_\mathbf{y}}: \mathbf{a}_i^\top \mathbf{y} \le b_i, i = 1, \ldots, m\}.
$$
其中 $\mathbf{A} \in \mathbb{R}^{m \times d_\mathbf{y}}$ 且 $\mathbf{b} \in \mathbb{R}^{m}$.

这里会考虑如下四个正则性假设, 其确保了类似光滑强凸等性质, 在不同阶段保证了我们处理的函数本身不至于过于复杂.

***Assumption 1* (Objective Regularity).**
1. $f(\mathbf{x}, \mathbf{y})$ 和 $g(\mathbf{x}, \mathbf{y})$ 是关于 $(\mathbf{x}, \mathbf{y})$ jointly smooth 的:


## Problem Formulation and Algorithm

### 2.1 Barrier Smoothing

目前这样的 Bilevel 问题的一个主要挑战在于: 给定 $\mathbf{x}$, 下层问题的最优解 $\mathbf{y}^\star(\mathbf{x})\in \arg\min_{\mathbf{y} \in \mathcal{Y}} g(\mathbf{x}, \mathbf{y})$ 这一映射 $\mathbf{x} \mapsto \mathbf{y}^\star(\mathbf{x})$ 本身可能是不可导的.
- *说明*. 对于下层问题, 其 active set $\mathcal{I}_A(\mathbf{x}) = \{i : \mathbf{a}_i^\top \mathbf{y}^\star(\mathbf{x}) = b_i\}$ 可能会随着 $\mathbf{x}$ 的变化而发生变化, 从而导致 $\mathbf{y}^\star(\mathbf{x})$ 的不连续性.
  - 例如, 假设给定 $\mathbf{x}$, 记无约束条件下的最优解为 $\mathbf{\bar{y}}^\star(\mathbf{x})$. 则若 $\mathbf{\bar{y}}^\star(\mathbf{x}) \in \operatorname{int}(\mathcal{Y})$, 则 $\mathcal{I}_A(\mathbf{x}) = \varnothing$ (因为最优值在约束集的内部, 没有任何边界约束被激活); 若反之, $\mathbf{\bar{y}}^\star(\mathbf{x})$ 在边界或者外部, 则约束的最优就可能落在约束的边界或角落上, 有一些约束被激活. 因此, 根据不同的 $\mathbf{x}$, 下层问题的最优解的 active set 是变化的.
  - 在最本质上, 考虑下层问题的 KKT 条件, 本质上就是由 active set 对应的分块决定的, 即:
    $$
    \begin{aligned}
    0 &= \nabla_\mathbf{y} g(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})) + \mathbf{A}_\mathcal{I_A(\mathbf{x})}^\top \boldsymbol{\lambda}^\star(\mathbf{x}) &&\qquad \text{(Stationarity)} \\
    \mathbf{b}_i &= \mathbf{A}_{\mathcal{I_A(\mathbf{x})}} \mathbf{y}^\star(\mathbf{x}) &&\qquad \text{(Complementarity)} \\
    \end{aligned}
    $$
    因此若同时对上述两式关于 $\mathbf{x}$ 求导, 则得到方程组:
    $$
    \begin{bmatrix}
    \nabla_{\mathbf{y}, \mathbf{y}}^2 g(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})) & \mathbf{A}_\mathcal{I_A(\mathbf{x})}^\top \\
    \mathbf{A}_{\mathcal{I_A(\mathbf{x})}} & \boldsymbol{0}
    \end{bmatrix}
    \begin{bmatrix}
    \frac{\partial \mathbf{y}^\star}{\partial \mathbf{x}} \\
    \frac{\partial \boldsymbol{\lambda}^\star}{\partial \mathbf{x}}
    \end{bmatrix} =
    \begin{bmatrix} -\nabla_\mathbf{x} g(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})) \\
    0
    \end{bmatrix}
    $$
    此时可以注意到, 导数 $\frac{\partial \mathbf{y}^\star}{\partial \mathbf{x}}$ 的存在性依赖于矩阵 $\mathbf{A}_{\mathcal{I_A(\mathbf{x})}}$. 故当 active set 发生变化时, 得到的结果可能会不连续, 从而导致 $\mathbf{y}^\star(\mathbf{x})$ 的不可导性.

为解决下层的不可导性问题, 考虑使用 log barrier.
- 定义标量函数:
    $$
    \phi(\mathbf{y}) := -\sum_{i=1}^m \log(b_i - \mathbf{a}_i^\top \mathbf{y}),
    $$
- 作为 barrier function, 故对应的光滑后的下层目标函数为:
    $$
    \psi_\mu(\mathbf{x}, \mathbf{y}) := g(\mathbf{x}, \mathbf{y}) + \mu \phi(\mathbf{y}), \qquad \mathbf{y} \in \operatorname{int}(\mathcal{Y}), \mu > 0.
    $$
    对应下层问题的 minimizer 为:
    $$
    \mathbf{y}_\mu^\star(\mathbf{x}) := \arg\min_{\mathbf{y} \in \operatorname{int}(\mathcal{Y})} \psi_\mu(\mathbf{x}, \mathbf{y}).
    $$
- 对应的光滑后的上层目标函数为:
    $$
    F_\mu(\mathbf{x}) := f(\mathbf{x}, \mathbf{y}_\mu^\star(\mathbf{x})).
    $$


***Theorem 1* (Barrier Smoothing).** 在 *Assumption 1* 下, 对于任意 $\mathbf{x} \in \mathbb{R}^{d_\mathbf{x}}$, 有如下性质成立.
1. $\mathbf{y}_\mu^\star(\mathbf{x})$ 是 $\psi_\mu(\mathbf{x}, \cdot)$ 在 $\operatorname{int}(\mathcal{Y})$ 上的唯一 minimizer.
2. 映射 $\mathbf{x} \mapsto \mathbf{y}_\mu^\star(\mathbf{x})$ 是可微的.
3. 外层函数 $F_\mu(\mathbf{x}) = f(\mathbf{x}, \mathbf{y}_\mu^\star(\mathbf{x}))$ 是可微的, 且其梯度为:
    $$
    \nabla F_\mu(\mathbf{x}) = \nabla_\mathbf{x} f(\mathbf{x}, \mathbf{y}_\mu^\star(\mathbf{x}))  - \nabla_{\mathbf{x}, \mathbf{y}}^2 \psi_\mu(\mathbf{x}, \mathbf{y}_\mu^\star(\mathbf{x}))^\top \left(\nabla_{\mathbf{y}, \mathbf{y}}^2 \psi_\mu(\mathbf{x}, \mathbf{y}_\mu^\star(\mathbf{x}))\right)^{-1} \nabla_\mathbf{y} f(\mathbf{x}, \mathbf{y}_\mu^\star(\mathbf{x})).
    $$
4. 存在常数 $C_g, C_y, C_F > 0$, 使得对于任意 $\mathbf{x} \in \mathbb{R}^{d_\mathbf{x}}$ 和 $\mu > 0$, 有
    $$
    0 \leq g(\mathbf{x}, \mathbf{y}_\mu^\star(\mathbf{x})) - g(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})) \leq C_g \sqrt\mu, \quad
    \|\mathbf{y}_\mu^\star(\mathbf{x}) - \mathbf{y}^\star(\mathbf{x})\| \leq C_y \sqrt\mu, \quad
    |F_\mu(\mathbf{x}) - F(\mathbf{x})| \leq C_F \sqrt\mu.
    $$

### 2.2 Barrier-Metric First-Order Method

- 回顾, 对于原问题 $\min_{\mathbf{x} \in \mathbb{R}^{d_\mathbf{x}}} F(\mathbf{x}) = f(\mathbf{x}, \mathbf{y}^\star(\mathbf{x}))$, 其中 $\mathbf{y}^\star(\mathbf{x}) \in \arg\min_{\mathbf{y} \in \mathcal{Y}} g(\mathbf{x}, \mathbf{y})$. 为应对其不可导性, 考虑使用 log barrier 方法, 将其转化为光滑问题 $\min_{\mathbf{x} \in \mathbb{R}^{d_\mathbf{x}}} F_\mu(\mathbf{x}) = f(\mathbf{x}, \mathbf{y}_\mu^\star(\mathbf{x}))$, 其中 $\mathbf{y}_\mu^\star(\mathbf{x}) \in \arg\min_{\mathbf{y} \in \operatorname{int}(\mathcal{Y})} \psi_\mu(\mathbf{x}, \mathbf{y}) = g(\mathbf{x}, \mathbf{y}) + \mu \phi(\mathbf{y})$.

- 故此时, 对于 smooth surrogate function $F_\mu(\mathbf{x})$, 可以使用一阶方法进行优化. 然而由 Theorem 1 (3) 可知, 其梯度 $\nabla F_\mu(\mathbf{x})$ 的计算仍然涉及到 Hessian 和 Hessian 的逆, 这仍然是很昂贵的, 因此还需要进一步进行优化.

- 定义 $\psi_\mu^\star(\mathbf{x}) := \min_{\mathbf{y} \in \operatorname{int}(\mathcal{Y})} \psi_\mu(\mathbf{x}, \mathbf{y})$, 考虑如下新的 lower problem:
    $$
    L_{\lambda, \mu} (\mathbf{x}, \mathbf{y}) := f(\mathbf{x}, \mathbf{y}) + \lambda \left[\psi_\mu(\mathbf{x}, \mathbf{y}) - \psi_\mu^\star(\mathbf{x})\right], \qquad \lambda > 0, \mu > 0.
    $$
    对应最优值为
    $$
    \mathbf{y}_{\lambda, \mu}^\star(\mathbf{x}) := \arg\min_{\mathbf{y} \in \operatorname{int}(\mathcal{Y})} L_{\lambda, \mu} (\mathbf{x}, \mathbf{y}).
    $$
    可以证明, 这个新的下层问题的最小值 $\min_{\mathbf{y} \in \operatorname{int}(\mathcal{Y})} L_{\lambda, \mu} (\mathbf{x}, \mathbf{y}) =: C^\star_{\lambda, \mu}(\mathbf{x})$ 具有如下良好性质:
    $$
    \begin{aligned}
    \nabla C^\star_{\lambda, \mu}(\mathbf{x}) &= \nabla_\mathbf{x} f(\mathbf{x}, \mathbf{y}_{\lambda, \mu}^\star(\mathbf{x})) + \lambda \left[\nabla_\mathbf{x} \psi_\mu(\mathbf{x}, \mathbf{y}_{\lambda, \mu}^\star(\mathbf{x})) -  \nabla_\mathbf{x} \psi_\mu(\mathbf{x}, \mathbf{y}_\mu^\star(\mathbf{x}))\right]
    , \\
    &=\nabla F_\mu(\mathbf{x}) + \mathcal{O}\left(\frac{1}{\lambda}\right).
    \end{aligned}
    $$
    - 一方面, 作者证明可以使用 $\nabla C^\star_{\lambda, \mu}(\mathbf{x})$ 来近似 $\nabla F_\mu(\mathbf{x})$, 误差不超过 $\mathcal{O}\left(\frac{1}{\lambda}\right)$.
    - 另一方面, 观察 $\nabla C^\star_{\lambda, \mu}(\mathbf{x})$ 的计算不再涉及 Hessian 的逆, 因此可以使用一阶方法来进行优化, 以求解表达式中的 $\mathbf{y}_{\lambda, \mu}^\star(\mathbf{x})$ 和 $\mathbf{y}_\mu^\star(\mathbf{x})$. 故记第 $k$ 次迭代的 $\mathbf{x}$ 为 $\mathbf{x}_{k}$, 并记
    $$
    q^{\mathbf{x}}_k(\mathbf{y}) := \nabla C^\star_{\lambda_k, \mu}(\mathbf{x}_{k}).
    $$

- 因此在迭代中 (如 Algorithm 1 所示), 引入两个变量 $\mathbf{z}_k \approx \mathbf{y}^\star_{\mu} (\mathbf{x}_{k})$ 和 $\mathbf{y}_k \approx \mathbf{y}^\star_{\lambda_k, \mu}(\mathbf{x}_{k})$ 分别通过梯度下降追踪对应的最优解以进行近似.
  - 不过注意算法当中, 在具体进行 GD 的时候, 还会引入一个 pre-conditioner $\nabla^2 \phi(\mathbf{z}_k)^{-1}$ 或 $\nabla^2 \phi(\mathbf{y}_k)^{-1}$ 来加速收敛 (并且注意角标, 对于内层迭代循环 $t = 0, 1, \ldots, T-1$, 其对应的 $\mathbf{z}_k^t$ 和 $\mathbf{y}_k^t$ 是固定的).
  - 之所以选择 $\nabla^2 \phi(\cdot)$ 作为 pre-conditioner, 其中一个理解角度是因为对于原始的 $F_\mu(\mathbf{x})$, 其梯度 $\nabla F_\mu(\mathbf{x})$ 中涉及到的 $\nabla_{\mathbf{y}, \mathbf{y}}^2 \psi_\mu = \nabla_{\mathbf{y}, \mathbf{y}}^2 g + \mu \nabla^2 \phi$, 而由于 $\phi$ 是 log barrier, 其 Hessian $\nabla^2 \phi = \sum_{i=1}^m \frac{\mathbf{a}_i \mathbf{a}_i^\top}{(b_i - \mathbf{a}_i^\top \mathbf{y})^2}$ 在 $b_i - \mathbf{a}_i^\top \mathbf{y}$ 较小时会变得很大带来巨大 curvature, 因此使用 $\nabla^2 \phi$ 作为 pre-conditioner 可以在一定程度上缓解这个问题. 并且另一方面确实其计算相对也是方便的.  另外还有一个在几何上的便利, 将在 section 3 详细介绍.

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260821170828259.png)