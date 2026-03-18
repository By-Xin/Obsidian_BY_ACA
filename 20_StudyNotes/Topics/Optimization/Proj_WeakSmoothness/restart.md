# Note on Restart: Solving LP using Bundle

## Introduction

首先考虑一般的 LP 问题. 

- 对于如下的线性规划问题：
    $$
    \begin{aligned}
    \min_{\mathbf{x} \in \mathbb{R}^n} & \quad \mathbf{c}^\top \mathbf{x} \\
    \text{s.t.} & \quad \mathbf{A} \mathbf{x} = \mathbf{b}\\
    & \quad \mathbf{x} \geq 0
    \end{aligned}
    $$

- 其对偶问题为：
    $$
    \begin{aligned}
    \max_{\mathbf{y} \in \mathbb{R}^m} & \quad \mathbf{b}^\top \mathbf{y} \\
    \text{s.t.} & \quad \mathbf{A}^\top \mathbf{y} \leq \mathbf{c}
    \end{aligned}
    $$

    其中 $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$, $\mathbf{c} \in \mathbb{R}^n$.


将上述问题统一表示为一个优化问题.

- 引入变量 $\mathbf{z} = \begin{bmatrix} \mathbf{x} \\ \mathbf{y} \end{bmatrix} \in \mathbb{R}^{n+m}$, 以及对应的约束矩阵 $\mathbf{H}$ 和向量 $\mathbf{e}$:
    $$
    \begin{aligned}
    \mathbf{H} = \begin{bmatrix} 
    \mathbf{A} & \mathbf{0}_{m\times m} \\ 
    -\mathbf{A} & \mathbf{0}_{m\times m} \\
    \mathbf{0}_{n\times n} & \mathbf{A}^\top \\
    \mathbf{c}^\top & -\mathbf{b}^\top \\
    -\mathbf{I}_{n} & \mathbf{0}_{n\times m}
    \end{bmatrix} \in \mathbb{R}^{(2m+2n+1) \times (n+m)},
     \quad
    \mathbf{e} = \begin{bmatrix}
    \mathbf{b} \\
    -\mathbf{b} \\
    \mathbf{c} \\
    0 \\
    \mathbf{0}
    \end{bmatrix}
    \end{aligned} \in \mathbb{R}^{2m+2n+1}
    $$

- 下面始终假设 primal 和 dual 问题都有最优解, 因此最优解集的集合 $\mathcal{Z}^\star \subset \mathbb{R}^{n+m}$ 是非空的. 并且记 $\text{dist}(\mathbf{z}, \mathcal{Z}^\star) = \inf_{\mathbf{u} \in \mathcal{Z}^\star} \|\mathbf{z} - \mathbf{u}\|$ 表示 $\mathbf{z}$ 到最优解集的距离.

- 根据上面的定义, 断言 $\mathcal{Z}^\star = \{\mathbf{z} \in \mathbb{R}^{n+m} : \mathbf{H} \mathbf{z} \leq \mathbf{e}\}$, 即 $\mathcal{Z}^\star$ 就是满足 $\mathbf{H} \mathbf{z} \leq \mathbf{e}$ 的所有 $\mathbf{z}$ 的集合.  这是由于,  $\mathbf{H}$ 的第四行, 其要求 $\mathbf{c}^\top \mathbf{x} - \mathbf{b}^\top \mathbf{y} \leq 0$, 再加之对于 primal 和 dual 问题的可行性约束自动保证了弱对偶性的成立, 因此二者合在一起自动给出了 duality gap 为 0 的条件, 从而保证了 $\mathbf{z}$ 是 primal 和 dual 问题的最优解.

另外, 上述表示还可以进一步简化.

- 有时, 为了简洁起见, 我们考虑将非负约束 $\mathbf{x} \geq 0$ 作为一个简单约束直接限制在定义域中而不作为一个单独的约束来处理. 若无特别说明, 下面的分析都基于这种简化后的表示:
    $$
    \begin{aligned}
    \mathbf{H} = \begin{bmatrix} 
    \mathbf{A} & \mathbf{0}_{m\times m} \\ 
    -\mathbf{A} & \mathbf{0}_{m\times m} \\
    \mathbf{0}_{n\times n} & \mathbf{A}^\top \\
    \mathbf{c}^\top & -\mathbf{b}^\top 
    \end{bmatrix} \in \mathbb{R}^{(2m+n+1) \times (n+m)},
     \quad
    \mathbf{e} = \begin{bmatrix}
    \mathbf{b} \\
    -\mathbf{b} \\
    \mathbf{c} \\
    0 
    \end{bmatrix}\in \mathbb{R}^{2m+n+1}
    \end{aligned} 
    $$

- 此时, 引入符号 $[\cdot]_{+}$ 表示向量的非负部分 (即 relu 函数 $[x]_{+} = \max\{x, 0\}$), 并规定此符号对于向量是逐元素应用的. 
  - 取 $[\mathbf{H} \mathbf{z} - \mathbf{e}]_{+} \in \mathbb{R}^{2m+n+1}$ 表示 $\mathbf{H} \mathbf{z} - \mathbf{e}$ 中的每个元素如果为负, 则说明该分量满足约束, 其 "error" 为 0; 如果为正, 则说明该分量违反了约束, 其 "error" 就是该分量的值. 
  - 因此, 用 $\|[\mathbf{H} \mathbf{z} - \mathbf{e}]_{+}\|$ 可以衡量 $\mathbf{z}$ 违反约束的程度. 
  <!-- 特别地, 指出有如下恒等式:
    $$
    \|[\mathbf{H} \mathbf{z} - \mathbf{e}]_{+}\|^2 =
    \|\mathbf{A} \mathbf{x} - \mathbf{b}\|^2 + 
    \|[\mathbf{A}^\top \mathbf{y} - \mathbf{c}]_{+}\|^2 +
    [\mathbf{c}^\top \mathbf{x} - \mathbf{b}^\top \mathbf{y}]_{+}^2 
    $$ -->



## Hoeffman Bound (Sharpness)

由于 $\mathcal{Z}^\star$ 是一个非空凸多面体, 因此由 Hoeffman bound 可知, 存在一个常数 $\alpha_H > 0$, 使得对于任意 $\mathbf{z} \in \text{dom}(z) = \mathbb{R}^{n}_+\times \mathbb{R}^m$, 都有如下的关系成立 (其中默认 $\|\cdot\|$ 是欧几里得范数):
$$
\text{dist}(\mathbf{z}, \mathcal{Z}^\star) \leq \alpha_H \|[\mathbf{H} \mathbf{z} - \mathbf{e}]_{+}\| := \alpha_H \Delta(\mathbf{z})
$$

- 其中定义 residual 为
    $$
    \Delta(\mathbf{z}) := \|[\mathbf{H} \mathbf{z} - \mathbf{e}]_{+}\| 
    $$

- 注意到, 该 residual 的平方可以化简整理如下:
    $$
    \begin{aligned}
    \Delta(\mathbf{z})^2 & = \|[\mathbf{H} \mathbf{z} - \mathbf{e}]_{+}\|^2 \\
    & = \|\mathbf{A} \mathbf{x} - \mathbf{b}\|^2 + 
    \|[\mathbf{A}^\top \mathbf{y} - \mathbf{c}]_{+}\|^2 +
    [\mathbf{c}^\top \mathbf{x} - \mathbf{b}^\top \mathbf{y}]_{+}^2 
    \end{aligned}
    $$

因此, 根据 Hoeffman bound 的形式, 我们可以考虑如下两种 residual 的优化目标:

- Non-smooth 型:
    $$
    f_{(1)}(\mathbf{z}) := \|\mathbf{A} \mathbf{x} - \mathbf{b}\| + 
    \|[\mathbf{A}^\top \mathbf{y} - \mathbf{c}]_{+}\| +
    [\mathbf{c}^\top \mathbf{x} - \mathbf{b}^\top \mathbf{y}]_{+} , \quad \forall z
    $$
- Smooth 型:
    $$
    f_{(2)}(\mathbf{z}) := \|\mathbf{A} \mathbf{x} - \mathbf{b}\|^2 + 
    \|[\mathbf{A}^\top \mathbf{y} - \mathbf{c}]_{+}\|^2 +
    [\mathbf{c}^\top \mathbf{x} - \mathbf{b}^\top \mathbf{y}]_{+}^2  = \Delta(\mathbf{z})^2, \quad \forall z
    $$