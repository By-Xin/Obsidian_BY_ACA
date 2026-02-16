# Gradient Descent

> Lecture Reference: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/

## Unconstrained Minimization Problems

- *Book Reference: Convex Optimization by Boyd & Vandenberghe, Section 9.1*

给定无约束优化问题:

$$
\min_x f(x)
$$

其中 $f: \mathbb{R}^n \to \mathbb{R}$ 为二次可微函数. 

假设该问题存在最优解 $x^\star$ 且唯一, 并记为最优值 $p^\star := f(x^\star) = \inf_x f(x)$.

在一般的条件下, 我们需要通过数值方法来求解该无约束优化问题, 即计算一系列迭代点 $x^{(0)}, x^{(1)}, x^{(2)}, \ldots \in \text{dom}(f)$, 使得 $k \to \infty$ 时, $f(x^{(k)}) \to p^\star$. 给定容许的误差 $\epsilon > 0$, 当 $f(x^{(k)}) - p^\star < \epsilon$ 时, 迭代过程停止.

此外, 对于初始点 $x^{(0)} \in \text{dom}(f)$, 要求其下水平集 (sublevel set) 为:

$$
S = \{x \in \text{dom}(f) ~|~ f(x) \leq f(x^{(0)})\}.
$$

是闭集 (即点列的极限点仍在该集合内) 以确保迭代的点列不会收敛到定义域外.

### Strong Convexity

在后续分析中, 我们通常假设目标函数 $f$ 在其定义域上是强凸的 (strong convexity), 即存在常数 $m > 0$, 使得对任意 $x \in S$, 有:

$$
\nabla^2 f(x) \succeq m I
$$

强凸性具有一些良好性质, 其一是满足如下不等式:

$$
\boxed{f(y) \geq f(x) + \nabla f(x)^\top (y-x) + \frac{m}{2}\|y-x\|_2^2, \quad \forall x,y \in S \quad (\star)}
$$
- *Proof.*
  - 对于强凸函数 $f(x)$ 进行二阶 Taylor 展开, 可得:

    $$
    f(y)=f(x)+\nabla f(x)^\top (y-x)+\frac{1}{2}(y-x)^\top \nabla^2 f(z)(y-x)
    $$

    其中 $z$ 在 $x$ 和 $y$ 之间
  - 由于 $\nabla^2 f(z) \succeq m I$, 因此 $(y-x)^\top \nabla^2 f(z)(y-x) \geq m \|y-x\|_2^2$, 代入上式即得所需不等式.
    $\square$

- 当 $m=0$ 时, 上述不等式即退化为一般的凸函数定义. 当 $m>0$ 时, 该不等式提供了一个更为强的下界. 

利用该不等式, 我们可以有效的分析梯度的大小对于优化值与最优值差距 $f(x)-p^\star$ 的影响, 分析如下. 

- 将上述不等式的 RHS 看作是关于 $y$ 的凸二次函数

  $$
  q(y):= f(x) + \nabla f(x)^\top (y-x) + \frac{m}{2}\|y-x\|_2^2,
  $$

  故该性质说明在任意 $y\in S$, 都有 $f(y) \geq q(y)$.
- 进一步求 $q(y)$ 的最小值, 记为 $q(\tilde{y})$:
  - 令 $\nabla_y q(y) = 0$, 可得最优解 $\tilde{y} = x - \frac{1}{m} \nabla f(x)$.
  - 代入可得最小值 $q(\tilde{y}) = f(x) - \frac{1}{2m} \|\nabla f(x)\|_2^2.$
  - 因此, 对任意 $y \in S$, 有:

    $$
    f(y) \geq q(y) \geq q(\tilde{y}) = f(x) - \frac{1}{2m} \|\nabla f(x)\|_2^2.
    $$
- 由于 $f(y) \geq f(x) - \frac{1}{2m} \|\nabla f(x)\|_2^2$ 对任意 $y \in S$ 成立, 故取 $y = x^\star$ 可得:

$$
p^\star = f(x^\star) \geq f(x) - \frac{1}{2m} \|\nabla f(x)\|_2^2
$$
- 整理可得重要不等式:

$$
\boxed{f(x) - p^\star \leq \frac{1}{2m} \|\nabla f(x)\|_2^2}
$$
    - 其直观含义很简单: 当点 $x$ 处的梯度 $\nabla f(x)$ 越小, 则该点的函数值 $f(x)$ 越接近最优值 $p^\star$. 
    - 这里相当于在强凸假设下给出了收敛的速度估计. 根据收敛准则 $f(x^{(k)}) - p^\star < \epsilon$, 故令 $\frac{1}{2m} \|\nabla f(x^{(k)})\|_2^2 < \epsilon$, 可得梯度范数的**次优性条件**:

  $$
  \boxed{\|\nabla f(x^{(k)})\|_2 < \sqrt{2m\epsilon}}
  $$

除了在函数值层面的分析, 还可以在任意点 $x \in S$ 上分析其与最优解 $x^\star$ 之间的距离, 具体如下.
- 根据强凸性性质, 特令 $(\star)$ 中的 $y = x^\star$, 可得:

  $$
  f(x^\star) \geq f(x) + \nabla f(x)^\top (x^\star - x) + \frac{m}{2}\|x^\star - x\|_2^2.
  $$
- 注意到 $\nabla f(x)^\top (x^\star - x)$ 作为内积, 可由 Cauchy-Schwarz 不等式得到下界:

  $$
  \nabla f(x)^\top (x^\star - x) \geq -\|\nabla f(x)\|_2\cdot \|x^\star - x\|_2.
  $$

  故代回上式可得:

  $$
  f(x^\star) \geq f(x) - \|\nabla f(x)\|_2 \cdot  \|x^\star - x\|_2 + \frac{m}{2}\|x^\star - x\|_2^2.
  $$
- 由于 $f(x^\star) - f(x) \leq 0$, 故上式移项可得:

  $$
  -\|\nabla f(x)\|_2 \cdot  \|x^\star - x\|_2 + \frac{m}{2}\|x^\star - x\|_2^2 \leq f(x^\star) - f(x) \leq 0.
  $$
- 记 $r := \|x^\star - x\|_2 \geq 0$, 则上式可化为关于 $r$ 的不等式:

  $$
  -\|\nabla f(x)\|_2 \cdot r + \frac{m}{2} r^2  = r\left(\frac{m}{2}r - \|\nabla f(x)\|_2\right) \leq 0.
  $$
- 由于 $r \geq 0$, 故上式成立当且仅当 $\frac{m}{2}r - \|\nabla f(x)\|_2 \leq 0$, 整理可得:

  $$
  \boxed{\|x - x^\star\|_2 \leq \frac{2}{m} \|\nabla f(x)\|_2}
  $$
- 该不等式说明, 当点 $x$ 处的梯度 $\nabla f(x)$ 越小, 则该点与最优解 $x^\star$ 之间的距离越近.

### Smoothness

同时, 由于 $\nabla^2 f(x)$ 在 $S$ 上连续, 且 $S$ 是紧集 (compact set), 故还能确认存在常数 $M > 0$, 使得对任意 $x \in S$, 有:

$$
\nabla^2 f(x) \preceq M I
$$
- 该条件说明, 在集合 $S$ 上, 函数 $f$ 的曲率 (curvature) 被上界 $M$ 所控制. 这也称为函数 $f$ 在集合 $S$ 上是 $M$-Smooth 的, 或者说函数 $f$ 在集合 $S$ 上具有 $M$-Lipschitz 连续的梯度:

  $$
  \|\nabla f(x) - \nabla f(y)\|_2 \leq M \|x-y\|_2, \quad \forall x,y \in S.
  $$
- 据此, 由 $M-$Smooth 提供的关于 Hessian 的上界, 可得如下重要不等式:

$$
\boxed{f(y) \leq f(x) + \nabla f(x)^\top (y-x) + \frac{M}{2}\|y-x\|_2^2, \quad \forall x,y \in S \quad (\star\star)}
$$

一般而言, 强凸性和光滑性可以通过如下的矩阵不等式来统一表达:

$$
m I \preceq \nabla^2 f(x) \preceq M I
$$

进一步定义 $\kappa := \frac{M}{m} \geq 1$, 称为函数 $f$ 在集合 $S$ 上的条件数 (condition number). 该条件数反映了函数在该集合上的曲率变化情况, 其值越接近 1, 则说明函数在该集合上越接近于二次函数.

> **关于 Lowner 序的补充说明**: 
> - 对于两个对称矩阵 $A,B \in \mathbb{S}^n$, 若 $A-B$ 为半正定矩阵, 则称 $A \succeq B$ (或等价地 $B \preceq A$). 该关系称为 Lowner 序 (Lowner order).  在数学上, 可以表达为, 若 $A \succeq B$, 则对任意非零向量 $v \in \mathbb{R}^n$, 有 $v^\top (A-B) v \geq 0$, 或等价地
>
>   $$
>   v^\top A v \geq v^\top B v, \quad \forall v \in \mathbb{R}^n\setminus \{0\}.
>   $$
> - 故上述的 $m I \preceq \nabla^2 f(x) \preceq M I$ 可等价于
>
>   $$
>   m \|v\|_2^2 \leq v^\top \nabla^2 f(x) v \leq M \|v\|_2^2, \quad \forall v \in \mathbb{R}^n\setminus \{0\}.
>   $$
> - 注意到, 对于上述的 Hessian $H:=\nabla^2 f(x)$, 其是一个对称矩阵, 故有谱分解 $H = Q \Lambda Q^\top$, 其中 $Q$ 为正交矩阵, $\Lambda = \text{diag}(\lambda_1,\ldots,\lambda_n)$ 为对角矩阵, 且 $\lambda_1,\ldots,\lambda_n$ 为 $H$ 的特征值. 且有事实:
>
>   $$
>   \min_{\|v\|_2=1} v^\top H v = \lambda_{\min}(H), \quad \max_{\|v\|_2=1} v^\top H v = \lambda_{\max}(H).
>   $$
> - 因此对于 $m I \preceq \nabla^2 f(x) \preceq M I$, 取 $\tilde v$ 为单位向量, 可知 $m \leq \tilde{v}^\top \nabla^2 f(x) \tilde{v} \leq M$, 进而有: $ m = \min m \leq \min \tilde{v}^\top \nabla^2 f(x) \tilde{v} = \lambda_{\min}(\nabla^2 f(x))$, 同理 $M \geq \lambda_{\max}(\nabla^2 f(x))$. 综上, 可知:
> $$
> m \leq \lambda_{\min}(\nabla^2 f(x)), \quad \lambda_{\max}(\nabla^2 f(x)) \leq M.
> $$
> - 该结果说明, $m I \preceq \nabla^2 f(x) \preceq M I$ 等价于 Hessian 矩阵的特征值被界定在区间 $[m,M]$ 上.


## Descent Methods

- *Book Reference: Convex Optimization by Boyd & Vandenberghe, Section 9.2*

### General Descent Methods

总的而言, 下降方法将产生一个优化点列 $x^{(k)}, k=0,1,2,\ldots$, 其中:

$$
x^{(k+1)} = x^{(k)} + t^{(k)} \Delta x^{(k)}
$$

或简记为:

$$
x^+ = x + t \Delta x,
$$

其中 $\Delta x$ 为搜索方向 (search direction), $t$ 为步长 (step size). 如何选择搜索方向 $\Delta x$ 和步长 $t$ 是下降方法的核心问题.

总的而言, 下降方法的基本算法为:
- 确定初始点 $x^{(0)} \in \text{dom}(f)$, 以及容许的误差 $\epsilon > 0$.
- 重复迭代:
  - 计算搜索方向 $\Delta x^{(k)}$.
  - 计算步长 $t^{(k)}$.
  - 更新 $x^{(k+1)} = x^{(k)} + t^{(k)} \Delta x^{(k)}$.
- 若 $f(x^{(k+1)}) - p^\star < \epsilon$, 则停止迭代.

---

下降方法要求除了最优点 $x^{(k)}$ 以外的任何搜索点 $x^+ = x^{(k)} + t \Delta x$ 都满足:

$$
f(x^{(k+1)} )< f(x^{(k)}), \quad \forall t > 0.
$$

而这一要求直接可以推出:

$$
\nabla f(x^{(k)})^\top \Delta x < 0.
$$
- 直观从几何意义上, 这说明搜索方向 $\Delta x$ 必须与负梯度 $-\nabla f(x^{(k)})$ 形成锐角, 即沿最陡下降的某个方向进行搜索.
- *Proof*.
  - 根据凸性的 supporting hyperplane 定理, 对于任意 $x, x^+$, 有 $f(x^+) - f(x) \leq \nabla f(x)^\top (x^+ - x) = t \nabla f(x)^\top \Delta x$. 由于 Descent Method 要求 $f(x^+) < f(x)$ 对任意 $t > 0$ 成立, 故 $\nabla f(x)^\top \Delta x < 0$.
    $\square$ 

### Line Search

首先对步长的选择进行讨论. Line Search 是一种常用的步长选择方法. 其包括两种常见的策略:

***Exact Line Search***

既然要求更新后的函数值 $f(x+t\Delta x)<f(x)$ 对任意 $t > 0$ 成立, 则可以通过求解如下的一维优化问题来选择最优的步长 $t$:

$$
t^* = \arg\min_{t \ge 0} f(x + t \Delta x)
$$


***Backtracking Line Search***

实践中常常使用 Backtracking Line Search 来选择步长, 其算法如下:
- 选择参数 $\alpha \in (0,0.5)$ 和 $\beta \in (0,1)$.
- 初始化 $t := 1$.
- 当 $f(x + t \Delta x) > f(x) + \alpha t \nabla f(x)^\top \Delta x$ 时, 更新 $t := \beta t$.
- 否则返回 $t$.

该算法的核心思想是, 从一个较大的初始步长 $t=1$ 开始, 不断缩小步长 $t$ 直到满足 Armijo 条件 (Armijo condition):

$$
f(x + t \Delta x) \leq f(x) + \alpha t \nabla f(x)^\top \Delta x.
$$

- 考虑 $f(t+\Delta x)$ 的 Taylor 展开:

  $$
  f(x + t \Delta x) \approx f(x) + t \nabla f(x)^\top \Delta x < \underbrace{f(x) + \alpha t \nabla f(x)^\top \Delta x}_{\text{Armijo Cond.}}.
  $$

  (由于 $\alpha \in (0,0.5), \nabla f(x)^\top \Delta x < 0$, 故 $\alpha t \nabla f(x)^\top \Delta x$ 是一个更小的负数). 
  因此只要 $t$ 能够被不断缩小, 就能满足 Armijo 条件.

经验上, $\alpha \in (0.01, 0.3)$, 表示可接受的 $f$ 的减少量是线性外推的 $1\%$ 到 $30\%$ 之间; $\beta \in (0.1, 0.8)$, 表示每次缩小步长的比例, 其越小表示每次缩小的幅度越大, 搜索越粗糙. 


## Gradient Descent

对于搜索方向 $\Delta x$ 的选择, 最自然的选择是负梯度方向, 即 $\Delta x = -\nabla f(x)$. 该方法被称为 Gradient Descent (GD). 其更新规则为:

$$
x^{(k+1)} = x^{(k)} - t^{(k)} \nabla f(x^{(k)}).
$$




### Convergence Analysis for Gradient Descent

假设 $f$ 在集合 $S$ 上满足 $m I \preceq \nabla^2 f(x) \preceq M I$, 且只考虑满足 $x-t \nabla f(x) \in S$ 的步长 $t$. 为方便书写, 还引入或重申如下符号:
- $x^+ := x - t \nabla f(x)$, 表示 GD 更新后的点.
- $\tilde f(t) := f(x - t \nabla f(x))$, 强调以步长 $t$ 作为自变量进行 GD 更新后的函数值. 其等价于 $\tilde f(t) = f(x^+)$.
- $p^\star := f(x^\star)$, 表示最优值.

根据 $M$-Smooth 推得到的 $(\star\star)$ 可得:

$$
\begin{aligned}
\tilde f(t) = f(x^+) &\leq f(x) + \nabla f(x)^\top (x^+ - x) + \frac{M}{2}\|x^+ - x\|_2^2 \\
&= f(x) - t \|\nabla f(x)\|_2^2 + \frac{M}{2} t^2 \|\nabla f(x)\|_2^2. \quad (\dagger)
\end{aligned}
$$
这一不等式将在后续分析中被反复使用. 

#### Convergence with Strong Convexity and Smoothness

首先讨论在强凸性和光滑性的条件下, GD 的收敛率.

***Convergence of GD with Exact Line Search***

- 回顾, 对于 Exact Line Search, 其步长 $t$ 的选择满足: $t^* = \arg\min_{t \ge 0} \tilde f(t)$.  故对 $\dagger$ 中左右两侧同取最小值, 可得:

  $$
  \begin{aligned}
  f(x^+) = \min_{t \ge 0} \tilde f(t) &\leq \min_{t \ge 0} \left\{f(x) - t \|\nabla f(x)\|_2^2 + \frac{M}{2} t^2 \|\nabla f(x)\|_2^2\right\} \\
  &= f(x) - \frac{1}{2M} \|\nabla f(x)\|_2^2.
  \end{aligned}
  $$
  - RHS 就是当作为关于 $t$ 的二次函数即可正常求得. 

- 再进一步对上述不等式左右两侧同时减去最优值 $p^\star$, 可得:

  $$
  \begin{aligned}
  f(x^+) - p^\star &\leq f(x) - p^\star - \frac{1}{2M} \|\nabla f(x)\|_2^2
  \end{aligned}
  $$
- 由强凸性得到的 $f(x) - p^\star \leq \frac{1}{2m} \|\nabla f(x)\|_2^2$ 仍然成立, 故可得:

$$
\begin{aligned}
f(x^+) - p^\star &\leq f(x) - p^\star - \frac{1}{2M} \|\nabla f(x)\|_2^2 \\
&\leq f(x) - p^\star - \frac{m}{M} (f(x) - p^\star) \\
&= \left(1 - \frac{m}{M}\right) (f(x) - p^\star) \\
&:=c\cdot (f(x) - p^\star).
\end{aligned}
$$
- 若从 $k=0$ 开始迭代, 则可得:

$$
\boxed{f(x^{(k)}) - p^\star \leq c^k (f(x^{(0)}) - p^\star)}
$$
  - 其中 $c = 1 - \frac{m}{M} \in (0,1)$. 
  - 这说明 $f(x^{(k)})$ geometrically 收敛到 $p^\star$, 其收敛率由 $c$ 决定. 由于 $c$ 与条件数 $\kappa = \frac{M}{m}$ 有关, 故函数的条件数越小, 则 GD 的收敛率越快. 
  - 这种收敛速度在优化算法中被称为线性收敛 (linear convergence), 其含义是误差 $f(x^{(k)}) - p^\star$ 的减少率在每次迭代中至少是一个常数 $c$ 的倍数.
- 若再进一步结合收敛准则 $f(x^{(k)}) - p^\star < \epsilon$, 即令 $c^k (f(x^{(0)}) - p^\star) \leq \epsilon$, 可得 GD 的迭代次数 $k$ 满足:

  $$
  k \geq \frac{\log\left(\frac{f(x^{(0)}) - p^\star}{\epsilon}\right)}{\log\left(\frac{1}{c}\right)} = \frac{\log\left(\frac{f(x^{(0)}) - p^\star}{\epsilon}\right)}{\log\left(\frac{M}{m}\right)}.
  $$
  - 分子说明迭代的次数依赖于初始点 $x^{(0)}$ 的选择 (反映在初始点与最优点的 gap) 和容许的误差 $\epsilon$ (结束点与最优点的 gap).
  - 分母说明迭代的次数依赖于函数的条件数 $\kappa = \frac{M}{m}$, 其值越大 (即函数越不平坦), 则迭代的次数越多.

***Convergence of GD with Backtracking Line Search***

回顾, 对于 Backtracking Line Search, 其步长 $t$ 的选择满足 Armijo 条件: $f(x + t \Delta x) \leq f(x) + \alpha t \nabla f(x)^\top \Delta x.$ 由于 $\Delta x = -\nabla f(x)$, 故在 GD 中, Armijo 条件可化为:

$$
f(x - t \nabla f(x)) \leq f(x) - \alpha t \|\nabla f(x)\|_2^2.
$$

- 这里首先给出断言, 只要通过算法将步长优化至 $0\leq t \leq 1/M$, 则 Armijo 条件必然满足. 证明如下.
  - *Proof*.
    - 注意到 $\dagger: \tilde f(t) \leq f(x) - t \|\nabla f(x)\|_2^2 + \frac{M}{2} t^2 \|\nabla f(x)\|_2^2 = f(x) - \|\nabla f(x)\|_2^2 \cdot \left(t - \frac{M}{2} t^2\right)$. 其 RHS 的二次函数 $t - \frac{M}{2} t^2$ 在 $t\in [0,1/M]$ 上满足

      $$
      t - \frac{M}{2} t^2 \geq \frac{t}{2}, \quad t \in [0,1/M].
      $$
    - 因此, 当 $t \in [0,1/M]$ 时, 可得

      $$
      \tilde f(t) \leq f(x) - \|\nabla f(x)\|_2^2 \cdot \left(t - \frac{M}{2} t^2\right) \leq f(x) - \frac{t}{2} \|\nabla f(x)\|_2^2.
      $$
      由于 $\alpha \in (0,0.5)$, 故 $\frac{t}{2} \|\nabla f(x)\|_2^2 > \alpha t \|\nabla f(x)\|_2^2$, 进而可得 $\tilde f(t) \leq f(x) - \frac{1}{2} t \|\nabla f(x)\|_2^2 < f(x) - \alpha t \|\nabla f(x)\|_2^2$. 综上, 当 $t \in [0,1/M]$ 时, Armijo 条件必然满足.

接着讨论其收敛率. 

-  Backtracking Line Search 的算法设计保证了, 其步长 $t$ 要么终止在 $t=1$, 要么终止在 $t \geq \beta/M$ (因为当 $t \leq 1/M$ 时, Armijo 条件必然满足, 故不会继续缩小步长, 故 $\beta/M$ 将会是最后一次缩小步长的下界). 因此, 考虑 Backtracking Line Search 的 Armijo 条件, 可得:
  - 当 $t=1$, 则 Armijo 条件为: $f(x - \nabla f(x)) \leq f(x) - \alpha \|\nabla f(x)\|_2^2$.
  - 当 $t \geq \beta/M$, 则 Armijo 条件为: $f(x - t \nabla f(x)) \leq f(x) - \alpha t \|\nabla f(x)\|_2^2 \leq f(x) - \alpha \frac{\beta}{M} \|\nabla f(x)\|_2^2$.
- 综上, Backtracking Line Search 的 Armijo 条件可化为:

  $$
  f(x^+) \leq f(x) - \alpha \min\{1, \frac{\beta}{M}\} \|\nabla f(x)\|_2^2.
  $$
- 进一步对上述不等式左右两侧同时减去最优值 $p^\star$, 可得:

$$
\begin{aligned}
f(x^+) - p^\star &\leq f(x) - p^\star - \alpha \min\{1, \frac{\beta}{M}\} \|\nabla f(x)\|_2^2
\end{aligned}
$$
- 进一步由强凸性(或 PL) 的等价形式 $\|\nabla f(x)\|_2^2 \geq 2m(f(x)-p^\star)$, 乘上负系数 $-\alpha \min\{1, \frac{\beta}{M}\}<0$ 时不等号方向翻转, 从而可得:

$$
\begin{aligned}
f(x^+) - p^\star &\leq f(x) - p^\star - \alpha \min\{1, \frac{\beta}{M}\} \|\nabla f(x)\|_2^2 \\
&\leq f(x) - p^\star - 2m \alpha \min\{1, \frac{\beta}{M}\} (f(x) - p^\star) \\
&= \left(1 - 2m \alpha \min\{1, \frac{\beta}{M}\}\right) (f(x) - p^\star) \\
&:=c\cdot (f(x) - p^\star).
\end{aligned}
$$

综上, 与 Exact Line Search 的收敛率分析类似, Backtracking Line Search 的收敛率也为线性收敛, 其收敛率由 $c = 1 - 2m \alpha \min\{1, \frac{\beta}{M}\}$ 决定. 由于 $\alpha \in (0,0.5), \beta \in (0,1)$, 故 $c$ 的值将会大于 $1 - \frac{m}{M}$, 即 Backtracking Line Search 的收敛率将会慢于 Exact Line Search.

#### Convergence with Convexity

下面讨论只保留 $M$-Smooth 条件和一般的凸性条件下, GD 的收敛率. 此时我们能够沿用的是由 $M$-Smooth 提供的 $\dagger$ 不等式:

$$
\tilde f(t) \leq f(x) - t \|\nabla f(x)\|_2^2 + \frac{M}{2} t^2 \|\nabla f(x)\|_2^2.
$$
并且在 Backtracking Line Search 中, 曾讨论当 $t \in [0,1/M]$ 时, 有

$$
\tilde f(t) = f(x^+) \leq f(x) - \frac{t}{2} \|\nabla f(x)\|_2^2.
$$

下面在此基础上通过一般的凸性条件来分析 GD 的收敛率. 
- 由凸性的 Supporting Hyperplane 定理:

  $$
  f(x^\star) \geq f(x) + \nabla f(x)^\top (x^\star - x), ~\forall x \in S.
  $$

  整理有

  $$
  f(x) - p^\star \leq \nabla f(x)^\top (x - x^\star).
  $$
- 对 $\|x^+ - x^\star\|_2^2$ 进行展开, 可得:

  $$
  \begin{aligned}
  \|x^+ - x^\star\|_2^2 &= \|x - t \nabla f(x) - x^\star\|_2^2 \\
  &= \|x - x^\star\|_2^2 - 2t \nabla f(x)^\top (x - x^\star) + t^2 \|\nabla f(x)\|_2^2.
  \end{aligned}
  $$

  - 进行整理可得

    $$
    2t \nabla f(x)^\top (x - x^\star) = \|x - x^\star\|_2^2 - \|x^+ - x^\star\|_2^2 + t^2 \|\nabla f(x)\|_2^2.
    $$
- 结合上述两式, 可得:

  $$
  \begin{aligned}
  f(x) - p^\star &\leq \nabla f(x)^\top (x - x^\star) \\
  &= \frac{1}{2t} \left(\|x - x^\star\|_2^2 - \|x^+ - x^\star\|_2^2 + t^2 \|\nabla f(x)\|_2^2\right) \\
  &\leq \frac{1}{2t} \left(\|x - x^\star\|_2^2 - \|x^+ - x^\star\|_2^2\right)
  \end{aligned}
  $$
  - 其中最后一步是由于 $t^2 \|\nabla f(x)\|_2^2 \geq 0$.
- 迭代 $k$ 次后, 可得:

  $$
  \begin{aligned}
  f(x^{(k)}) - p^\star &\leq \frac{1}{2t} \left(\|x^{(k-1)} - x^\star\|_2^2 - \|x^{(k)} - x^\star\|_2^2\right)
  \end{aligned}
  $$

  - 为进一步得到该式数量级的估计, 对其进行如下求和放缩:

    $$
    \begin{aligned}
    \sum_{i=1}^k (f(x^{(i)}) - p^\star) &\leq \frac{1}{2t} \sum_{i=1}^k \left(\|x^{(i-1)} - x^\star\|_2^2 - \|x^{(i)} - x^\star\|_2^2\right) \\
    &= \frac{1}{2t} \left(\|x^{(0)} - x^\star\|_2^2 - \|x^{(k)} - x^\star\|_2^2\right) \\
    &\leq \frac{1}{2t} \|x^{(0)} - x^\star\|_2^2.
    \end{aligned}
    $$
  - 又由于 Descent Method 要求 $f(x^{(i)})$ 是单调递减的, 故 $k\cdot (f(x^{(k)}) - p^\star) \leq \sum_{i=1}^k (f(x^{(i)}) - p^\star)$, 进而可得:

$$
\boxed{f(x^{(k)}) - p^\star \leq \frac{1}{k} \cdot \frac{1}{2t} \|x^{(0)} - x^\star\|_2^2}
$$
- 该不等式说明, 在一般的凸性条件下, GD 的收敛率为 sublinear convergence, 即 $f(x^{(k)}) - p^\star = \mathcal{O}(1/k)$, 或 $k = \mathcal{O}(1/\epsilon)$. 相比于强凸性条件下的线性收敛, $k = \mathcal{O}(\log(1/\epsilon))$, 其收敛速度明显变慢.


### Worst-case Lower Bound of First-order Methods

一般地, 一阶方法 (first-order method) 都可以抽象为如下的迭代过程: 对于第 $k$ 次迭代, 其更新点 $x^{(k)}$ 为:
$$\begin{aligned}
x^{(k)} \in x^{(0)} + \text{span}\{\nabla f(x^{(0)}), \nabla f(x^{(1)}), \ldots, \nabla f(x^{(k-1)})\}.
\end{aligned}$$

如下定理说明任意一阶方法在的收敛速率下界为 $\mathcal{\Omega}(1/k^2)$.

对于任意 $k \leq (n-1)/2$ 及任意初始点 $x^{(0)}$, 都能存在一个 $M$-Smooth 的凸函数 $f: \mathbb{R}^n \to \mathbb{R}$, 使得对于任意满足上述迭代过程的一阶方法, 都有:
$$\begin{aligned}
f(x^{(k)}) - p^\star \geq \frac{3M\|x^{(0)} - x^\star\|_2^2}{32(k+1)^2} 
\end{aligned}$$

若进一步放宽 Convexity 的要求, 此时对于非凸优化问题, 我们只能考察其 $\epsilon$-stationary point 的收敛速率 (即 $\|\nabla f(x^{(k)})\|_2 \leq \epsilon$), 有定理如下.

对于固定步长 $t\leq 1/L$, GD 方法有:
$$\begin{aligned}
\min_{i=0,\ldots,k} \|\nabla f(x^{(i)})\|_2 \leq \sqrt{\frac{2(f(x^{(0)}) - p^\star)}{t(k+1)}}.
\end{aligned}$$

这说明在非凸优化问题中, GD 的收敛速率将不超过 $\mathcal{O}(1/\sqrt{k})$.