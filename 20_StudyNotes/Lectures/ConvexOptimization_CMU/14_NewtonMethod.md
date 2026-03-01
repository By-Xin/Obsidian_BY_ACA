# Newton Method

>[!quote]
>
> - Lecture Reference: <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>

## Newton's Method Interpretation

### Introduction

在一阶的 GD 方法中, 其核心思想是在当前点 $x$ 处, 考虑附近 $x^+$ 处的一阶泰勒展开:
$$
f(x^+) \approx f(x) + \nabla f(x)^\top (x^+-x) + \frac{1}{2t} \|x^+-x\|_2^2
$$

- 该近似成立可以由凸性+Lipschitz 连续性得到:
  - 由凸性可知, 对于任意 $x,y \in \text{dom}(f)$, 有:
    $$
    f(y) \geq f(x) + \nabla f(x)^\top (y-x)
    $$
  - 由 Lipschitz 连续性可知, 对于任意 $x,y \in \text{dom}(f)$, 有:
    $$
    f(y) \leq f(x) + \nabla f(x)^\top (y-x) + \frac{L}{2} \|y-x\|_2^2
    $$

- 若尝试通过寻找 $x^+$ 使得 RHS 最小 $\min_{x^+} f(x) + \nabla f(x)^\top (x^+-x) + \frac{1}{2t} \|x^+-x\|_2^2$, 则有:
  $$
  x^+ = x - \frac{1}{t} \nabla f(x)
  $$

因此, 进一步我们可以通过二阶泰勒展开来得到更精确的近似:
$$
f(x^+) \approx f(x) + \nabla f(x)^\top (x^+-x) + \frac{1}{2} (x^+-x)^\top \nabla^2 f(x) (x^+-x)
$$

- 用同样的方法对 RHS 进行最小化, 则有:
    $$
    x^+ = x - \nabla^2 f(x)^{-1} \nabla f(x)
    $$

因此, 我们可以得到 Newton 方法的更新规则:
$$
x_{t+1} = x_t - \nabla^2 f(x_t)^{-1} \nabla f(x_t)
$$

### Affine Invariance of Newton's Method

Newton's Method 具有 Affine Invariance 性质. 

- *Proof*:

  - 对于目标函数 $f$, 以及可逆矩阵 $A\in \mathbb{R}^{n\times n}$, 考虑迭代 $x_{t+1} = x_t - \nabla^2 f(x_t)^{-1} \nabla f(x_t)$. 
  
  - 若对其进行 Affine Transformation $y := Ax$, 则 Newton's Method 更新规则为:
    $$
    y_{t+1} = y_t - [\nabla^2 f(y_t)]^{-1} \nabla f(y_t) = A x_t - [\nabla^2 f(Ax_t)]^{-1} \nabla f(Ax_t)
    $$

  - 则有:
    $$
    \tilde{x}_{t+1} := A^{-1} y_{t+1} = x_t  - A^{-1}[\nabla^2 f(Ax_t)]^{-1} \nabla f(Ax_t) 
    $$

  - 另一方面, 考虑 $\phi(x) := f(Ax)$, 则有:
    $$
    \begin{aligned}
      x_{t+1} &= x_t - [\nabla^2 \phi(x_t)]^{-1} \nabla \phi(x_t) \\
      &= x_t - [A^\top \nabla^2 f(Ax_t) A]^{-1} A^\top \nabla f(Ax_t) \\
      &= x_t - A^{-1}[\nabla^2 f(Ax_t)]^{-1} \nabla f(Ax_t) \\
      &= \tilde{x}_{t+1}
    \end{aligned}
    $$
  
  $\square$


### Newton Decrement

对于目标函数 $f$ 及当前点 $x$, Newton Decrement 定义为:
$$
\lambda(x) = \left[\nabla f(x)^\top (\nabla^2 f(x))^{-1} \nabla f(x)\right]^{1/2}
$$

这个量可以有一下几个理解的角度:

- 首先, 其刻画了更新量经过 Hessian 矩阵校正后的长度. 若记 $\Delta x= x_{t+1} - x_t = - [\nabla^2 f(x)]^{-1} \nabla f(x)$, 则有:
  $$
  \lambda(x) = [\Delta x^\top (\nabla^2 f(x)) \Delta x]^{1/2} = \|\Delta x\|_{\nabla^2 f(x)}
  $$
  - 其中 $\|\cdot\|_{P}$ 表示在 $P$-范数下的长度, 即 $\|v\|_{P} = (v^\top P v)^{1/2}$.

- 其次, 其衡量了在当前点处距离二阶泰勒展开的最优解之间的距离. 即:
  $$
  \begin{aligned}
    \frac{1}{2} \lambda(x)^2 &= f(x) - \min_{y} \left[f(x) + \nabla f(x)^\top (y-x) + \frac{1}{2} (y-x)^\top \nabla^2 f(x) (y-x)\right] \\
    &= f(x) - \left[f(x) - \frac{1}{2} \nabla f(x)^\top (\nabla^2 f(x))^{-1} \nabla f(x)\right] 
  \end{aligned}
  $$
  - 对于牛顿法, 其在最优点附近处, 可以近似代替作为当前点距离最优解的距离. 

- 此外指出, Newton Decrement 也是 Affine Invariance 的.


### Damped Newton's Method (Backtracking Line Search)

在 pure Newton's Method 中，Newton 方向为 $v_t = - \nabla^2 f(x_t)^{-1}\nabla f(x_t)$. Damped Newton's Method 用 Backtracking Line Search 选择步长 $s$.

***Armijo Condition***

对给定 $s>0$，接受该步长当且仅当
$$
f(x_t + s v_t) \le f(x_t) + \alpha s \nabla f(x_t)^\top v_t,
$$
其中 $\alpha \in (0,1/2]$，$\beta \in (0,1)$.

> [!note]
> 若 $\nabla^2 f(x_t)$ 正定，则 $\nabla f(x_t)^\top v_t < 0$，因此右侧是对 $f(x_t)$ 的“下降”要求.

***Algorithm***

- 选初始步长 $s_0>0$ 以及超参数 $\alpha\in(0,1/2]$、$\beta\in(0,1)$.
- 在第 $t$ 次外层迭代：
  - 计算 Newton 方向：$v_t = -\nabla^2 f(x_t)^{-1}\nabla f(x_t)$.
  - 令 $s \leftarrow s_0$.
  - 重复（线搜索回溯）直到 Armijo 条件成立：
    - 若 $f(x_t + s v_t) > f(x_t) + \alpha s \nabla f(x_t)^\top v_t$，则令 $s \leftarrow \beta s$；
    - 否则接受该步长并停止回溯。
  - 更新：$x_{t+1} = x_t + s v_t$.


## Convergence Analysis

### Pure Newton's Method

假设 $f$ 是二阶连续可微的函数 ($f \in C^2(\mathbb{R}^n)$), 且假设其 Hessian 矩阵在最优解 $x^\star$ 的一个 $\delta$-邻域内是 Lipschitz 连续的, 即存在常数 $L>0$ 使得对于任意 $x,y \in N_{\delta}(x^\star)$ 都有:
$$
\|\nabla^2 f(x) - \nabla^2 f(y)\|_2 \leq L \|x - y\|_2
$$

如果函数 $f(x)$ 在 $x^\star$ 处满足 $\nabla f(x^\star) = 0$ 且 $\nabla^2 f(x^\star) \succ 0$, 则对于上述 pure Newton's Method 有如下系列结论:

1. 如果初始点距离 $x^\star$ 的足够近, 则牛顿法产生的迭代点列会收敛到 $x^\star$.

2. $\{x_k\}$ 的收敛速度为 Q-quadratic 的. 
   - $\|x_{k+1} - x^\star\|_2 \leq L \|\nabla^2 f(x_k)^{-1}\|_2 \|x_k - x^\star\|_2 := C_1 \|x_k - x^\star\|_2^2$.
   - 换言之, 若初始点 $x_0$ 满足 $\|x_0 - x^\star\|_2 \leq \min\{\delta, r, 1/2L \| \nabla^2 f(x_0)^{-1}\|_2\} := \hat{\delta}$, 则可保证点列一直处于 $N_{\hat{\delta}}(x^\star)$ 内, 从而保证点列收敛到 $x^\star$. 其中 $r$ 是一个局部邻域半径保证在 $x^\star$ 附近其 Hessian 具有连续, 非退化等性质.

3. $\{\|\nabla f(x_k)\|_2\}$ 以 Q-quadratic 的速率收敛到零. 具体地:
    $$
    \|\nabla f(x_{k+1})\|_2 \leq 2L \|\nabla^2 f(x^\star)^{-1}\|_2^2\cdot \|\nabla f(x_k)\|_2^2 := C_2 \|\nabla f(x_k)\|_2^2
    $$
    

由此可见, Newton's Method 的收敛速度非常快, 其收敛速度为 Q-quadratic 的. 但其同时也有代价:

- 初始点必须足够接近最优解, 牛顿法只具有局部收敛性.
- Hessian $\nabla^2 f(x^{\star})$ 需要为正定矩阵. 若是其是奇异的非正定, 则收敛速度可能只有 Q-linear 的.
- 尽管条件数不会直接影响收敛速度, 但对于病态问题, 牛顿法的收敛域可能会变小, 故对初值的选取有了更大的要求.

### Damped Newton's Method with Strong Convexity

假设 $f$ 是 $m$-强凸函数, $\nabla f$ 是 $L$-Lipschitz 连续的, $\nabla^2 f$ 是 $M$-Lipschitz 连续的, 则对于上述 damped Newton's Method 会有如下 2-stage 的收敛结构:

- 第一阶段 (Damped Phase): 当 $\|\nabla f(x_k)\| \geq \eta$ 远离最优解时, 为线性收敛速率:
  $$
  f(x_{k+1}) - f(x_k) \leq - \gamma
  $$

- 第二阶段 (Pure Newton Phase): 当 $\|\nabla f(x_k)\| \leq \eta$ 接近最优解时, 有二次收敛速率:
  $$
  \|\nabla f(x_{k+1})\|_2 \leq C \|\nabla f(x_k)\|_2^2
  $$

正是由于前面的强凸性, 全局 Lipschitz 连续以及 Armijo Condition 的共同作用, 使得在远离最优解时保证算法依然不会发散, 函数值保证下降, 梯度范数保证下降.

> [!note] 回顾$m$-强凸性:
>
> ***Definition* (Strong Convexity)**: 称 $f$ 是 $m$-强凸函数, 若存在常数 $m>0$ 使得对于任意 $x,y \in \mathbb{R}^n$ 都有:
>
> $$
> f(y) \geq f(x) + \nabla f(x)^\top (y-x) + \frac{m}{2}\|y-x\|_2^2
> $$
>
> 其中 $m$ 称为强凸常数.
>
> ***Definition* (Lipschitz Continuity)**: 称 $f$ 是 $L$-Lipschitz 连续的, 若存在常数 $L>0$ 使得对于任意 $x,y \in \mathbb{R}^n$ 都有:
>
> 其具有如下性质:
>
> - 若 $f \in C^2$, 则 $\nabla^2 f(x) \succeq m I$. 这说明曲率的下界被 $m$ 所控制, 不会退化, 所有的特征值均大于 $m$.
>
> - 函数值的变化率被 $m$ 控制: $f(x) - f(x^\star) \geq \frac{m}{2}\|x-x^\star\|_2^2$.
>
> - 梯度变化被 $m$ 控制: $\|\nabla f(x)\| \geq m \|x-x^\star\|$.
>
> - 梯度变化是强单调的: $(\nabla f(x) - \nabla f(x^\star))^\top (x-x^\star) \geq m \|x-x^\star\|_2^2$.
>
> - 在强凸+Lipschitz 连续的假设下, 如下三者在常数意义下等价: $\|x-x^\star\|^2\sim f(x) - f(x^\star) \sim \|\nabla f(x)\|_2^2$.


### Convergence under Self-concordance

***Definition* (Self-concordance)**: 以一元函数为例. 称 $f$ 是 $\kappa$-self-concordant 的, 若存在常数 $\kappa>0$ 使得对于任意 $x \in \mathbb{R}$ 都有:
$$
\left|\frac{\mathrm{d}^3}{\mathrm{d}x^3} f(x)\right| \leq \kappa \left|\frac{\mathrm{d}^2}{\mathrm{d}x^2}f(x)\right|^{3/2}
$$

- 默认 $\kappa  = 2$. 对于其他的取值, 事实上也可以通过缩放 $f(x) = \kappa^2 g(x) / 4$ 来得到.

若目标函数 $f$ 是 self-concordant 的, 则不需要上述强凸+光滑的假设, 亦能保证上述的线性+二次的收敛结构.


## Discussion

### Comparison with First-order Methods


| Method | Gradient Descent | Newton's Method | 
|--------|------------------|-----------------|
| 内存复杂度 | $\mathcal{O}(n)$ | $\mathcal{O}(n^2)$ |
| 计算复杂度 | $\mathcal{O}(n)$ | $\mathcal{O}(n^3)$ (对于 Dense 的 Hessian 矩阵) |
| Backtracking 成本 | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ |
| 条件数影响 | 敏感 | 局部不敏感 |
| 稳健性 | 强 | 弱, 受到数值稳定性, 奇异性等问题的影响 |


### Sparse, Structured Problems

Hessian 的求解是 Newton's Method 的瓶颈. 对于一些结构化问题, 例如: sparse, banned (只有主对角线附近有非零元素), 块对角, Toeplitz / Kronecker 结构, Low-rank 等, 可以利用其结构特性来加速 Hessian 的求解.


### Equality Constrained Newton's Method

对于等式约束优化问题:
$$
\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \quad Ax = b
$$

一个比较直观的思路是在 $x$ 的切线空间中进行优化. 记优化的更新方向为 $v$, 即 $x^+ = x + v$. 则我们只需保证 $Av = 0$ 即可保证 $Ax^+ = A(x + v) = Ax + Av = b$.

还原到 Newton 法最开始推导的二阶展开表达式, 我们的最小化任务为:
$$
\begin{aligned}
\min_{Av = 0} \left[ \nabla f(x)^\top v + \frac{1}{2} v^\top \nabla^2 f(x) v\right]
\end{aligned}
$$

对应 Lagrangian 形式为:
$$
L(v, \lambda) = \nabla f(x)^\top v + \frac{1}{2} v^\top \nabla^2 f(x) v + \lambda^\top Av
$$

Stationarity Condition 为:
$$
\begin{cases}
\nabla f(x) + \nabla^2 f(x) v + A^\top \lambda = 0\\
Av = 0
\end{cases} 
\iff
\begin{bmatrix}
\nabla^2 f(x) & A^\top \\
A & 0
\end{bmatrix}
\begin{bmatrix}
v \\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
-\nabla f(x) \\
0
\end{bmatrix}
$$

由此, 在计算出 $v$ 后, 更新:
$$
x_{t+1} = x_t + v
$$

