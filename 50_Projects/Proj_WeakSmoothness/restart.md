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

由于 $\mathcal{Z}^\star$ 是一个非空凸多面体, 因此由 Hoeffman bound 可知, 存在一个常数 $\alpha_H > 0$, 使得对于任意 $\mathbf{z} \in \text{dom}(z) := \mathcal{Z} = \mathbb{R}^{n}_+\times \mathbb{R}^m$, 都有如下的关系成立 (其中默认 $\|\cdot\|$ 是欧几里得范数):
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
    L_1(\mathbf{z}) := \|\mathbf{A} \mathbf{x} - \mathbf{b}\| + 
    \|[\mathbf{A}^\top \mathbf{y} - \mathbf{c}]_{+}\| +
    [\mathbf{c}^\top \mathbf{x} - \mathbf{b}^\top \mathbf{y}]_{+} , \quad \forall z \in \mathcal{Z}
    $$
    - 注意到, $\Delta(\mathbf{z}) \leq L_1(\mathbf{z})$ 恒成立.
    - 对应 Hoffman bound, 有
        $$
        \text{dist}(\mathbf{z}, \mathcal{Z}^\star) \leq \alpha_H \Delta(\mathbf{z}) \leq \alpha_H L_1(\mathbf{z}), \quad \forall z \in \mathcal{Z}
        $$
- Smooth 型:
    $$
    L_2(\mathbf{z}) := \|\mathbf{A} \mathbf{x} - \mathbf{b}\|^2 + 
    \|[\mathbf{A}^\top \mathbf{y} - \mathbf{c}]_{+}\|^2 +
    [\mathbf{c}^\top \mathbf{x} - \mathbf{b}^\top \mathbf{y}]_{+}^2  = \Delta(\mathbf{z})^2, \quad \forall z \in \mathcal{Z}
    $$
    - 注意到, $L_2(\mathbf{z}) = \Delta(\mathbf{z})^2$ 恒成立.
    - 对应 Hoffman bound, 有
        $$
        \begin{aligned}
        \text{dist}(\mathbf{z}, \mathcal{Z}^\star) & \leq \alpha_H \Delta(\mathbf{z}) = \alpha_H \sqrt{L_2(\mathbf{z})}
        \end{aligned}
        $$
  

此外, 对于上述两种 residual 的优化目标, 不难验证,
$$
\min_{\mathbf{z} \in \mathcal{Z}} L_1(\mathbf{z}) = \min_{\mathbf{z} \in \mathcal{Z}} L_2(\mathbf{z}) = 0
$$

即 $L^\star := \min_{\mathbf{z} \in \mathcal{Z}} L_1(\mathbf{z}) = \min_{\mathbf{z} \in \mathcal{Z}} L_2(\mathbf{z}) = 0$.

## Restart Method

Restart 策略的核心思想是, 在优化过程中, 当某个阶段的优化算法达到一定的迭代次数或者满足某个条件时, 就 "重启" 优化算法, 即重新初始化优化算法的状态 (例如重新设置学习率、重新计算梯度等), 从而使得优化算法能够更快地收敛到最优解.

上述方法使得整体的优化会分为内外两层循环. 内层循环负责优化目标函数 (例如 $L_1$ 或 $L_2$), 外层循环负责监控内层循环的进展并决定何时重启.

在这里, 我们记外层的迭代为 $t = 0, 1, 2, \ldots$ 对应内层的迭代共 $K$ 次为 $k = 0, 1, 2, \ldots, K$. 因此总得而言, restart 的算法如下:
1. 给定初值 $\mathbf{z}_{0}^{(0)} \in \mathcal{Z}$, 以及内层循环的最大迭代次数 $K$.
2. 外层循环, 对于 $t = 0, 1, 2 \ldots$:
    1. 内层循环, 对于 $k = 0, 1, 2, \ldots, K$:
        - 迭代更新 $\mathbf{z}_{t}^{(k+1)} \leftarrow \text{Update }(\mathbf{z}_{t}^{(k)})$.
    2. 检索收敛条件, 如果满足则退出循环.
    3. 否则重启, 更新 $\mathbf{z}_{t+1}^{(0)} \leftarrow \text{Restart }(\mathbf{z}_{t}^{(K)})$.

该策略的核心思想是, 如果在某个阶段能够满足一定的收敛条件, 如对于某个指标 $\Psi$, 有
$$
\Psi(\mathbf{z}_{t}) \leq \beta \Psi(\mathbf{z}_{t-1})
$$
其中 $\beta \in (0, 1)$ 是一个预设的收敛阈值, $\mathbf{z}_{t}:= \mathbf{z}_{t}^{(K)}$ 是内层循环结束后的结果, 则经过外层 $T$ 次迭代后, 就可以保证 $\Psi(\mathbf{z}_{T}) \leq \beta^T \Psi(\mathbf{z}_{0})$, 从而实现指数级的收敛.   

## Optimization of $L_1$ with Restart

如果考虑优化 $L_1$, 由于其是一个非光滑的目标函数, 使用 restart, 在当前第 $t$ 次迭代时, 在内层考虑使用标准 non-smooth 的优化方法 (例如 sub-gradient method). 对于该种标准优化方法, 在进行 $K$ 次迭代后, 有结论:
$$
L_1(\mathbf{z}_{t+1}^{(0)}) \lesssim \frac{M \text{dist}(\mathbf{z}_{t}^{(0)}, \mathcal{Z}^\star)}{\sqrt{K}}
$$

进一步代入 Hoeffman bound 的结论 $\text{dist}(\mathbf{z}, \mathcal{Z}^\star) \leq \alpha_H L_1(\mathbf{z})$, 可以得到如下的收敛率:
$$
L_1(\mathbf{z}_{t+1}^{(0)}) \lesssim \frac{M \alpha_H }{\sqrt{K}} L_1(\mathbf{z}_{t}^{(0)}) := \beta_1 L_1(\mathbf{z}_{t}^{(0)})
$$

因此只要保证每次的优化幅度
$$
\beta_1 = \frac{M \alpha_H }{\sqrt{K}} < 1
$$
就可以保证上述 restart 中描述的几何收敛(线性), 从而在第 $T$ 次外层迭代后, 有
$$
L_1(\mathbf{z}_{T}^{(0)}) \lesssim \beta_1^T L_1(\mathbf{z}_{0}^{(0)})
$$
因此若要求总的误差为 $L_1(\mathbf{z}_{T}^{(0)}) \leq \epsilon$, 则只需
$$
T \gtrsim \frac{\log(L_1(\mathbf{z}_{0}^{(0)})/\epsilon)}{\log(1/\beta_1)} 
$$

进而, 由$\beta$ 的选择反推 $K \asymp \frac{M^2 \alpha_H^2}{\beta_1^2}$ , 故总的迭代次数为 
$$
T \cdot K \asymp \frac{M^2 \alpha_H^2}{\beta_1^2 \log(1/\beta_1)} \log(L_1(\mathbf{z}_{0}^{(0)})/\epsilon).
$$

在特别地, 若强制 $\beta_1 = 1/2$, 则 $K \asymp 4 M^2 \alpha_H^2$, 从而总的迭代次数为 $\mathcal{O}(M^2 \alpha_H^2 \log(L_1(\mathbf{z}_{0}^{(0)})/\epsilon))$.

> [!note]
>
> 这里给出优化中的更完整叙述. 对于凸目标函数 $l:\mathcal{Z} \to \mathbb{R}$, 以及其最优解集 $\mathcal{Z}^\star := \arg\min_{\mathbf{z} \in \mathcal{Z}} l(\mathbf{z})\neq \emptyset$. 此外, 假设 $f$ 在 $\mathcal{Z}$ 上是 $M$-Lipschitz 的, 即对于任意 $g \in \partial l(\mathbf{z})$, 都有 $\|g\| \leq M$. 则此时, 对于标准的 sub-gradient method, 在进行 $K$ 次迭代后, 有如下 $\mathcal{O}(\frac{1}{\sqrt{K}})$ 的收敛率:
> $$
> l(\mathbf{z}_K) - L^\star \leq \frac{\|z_0 - z^\star\|^2}{2 \sum_{k=1}^K \eta_k} + \frac{M^2 \sum_{k=1}^K \eta_k^2}{2 \sum_{k=1}^K \eta_k}
> $$
> - 其中 $\mathbf{z}_0$ 是初始点, $\mathbf{z}^\star$ 是最优解, $\eta_k$ 是第 $k$ 次迭代的学习率. 
>
> 如果选择合适的步长, 例如 $\eta_k \asymp \frac{\|z_0 - z^\star\|}{M \sqrt{K}}$, 则可以得到如下的收敛率:
> $$
> l(\bar{\mathbf{z}}_K) - L^\star \lesssim \frac{M \|z_0 - z^\star\|}{\sqrt{K}}
> $$
> - 其中 $\bar{\mathbf{z}}_K := \frac{1}{K} \sum_{k=1}^K \mathbf{z}_k$ 是迭代点的平均值.
>
> 再进一步对 $\|z_0 - z^\star\|$ 取最小化, 可以得到如下的收敛率:
> $$
> l(\bar{\mathbf{z}}_K) - L^\star \lesssim \frac{M \text{dist}(\mathbf{z}_0, \mathcal{Z}^\star)}{\sqrt{K}}
> $$