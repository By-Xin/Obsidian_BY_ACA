# Subgradient

> Lecture Reference: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/
>
> Reading Reference: 最优化: 建模、算法与理论, 刘浩洋等, 2.7 小节.

## Subgradients and Subdifferentials

回顾, 对于可微且凸的函数 $f: \mathbb{R}^n \to \mathbb{R}$, 其在任意点 $x$ 处的梯度 $\nabla f(x)$ 都满足以下不等式:
$$f(y) \geq f(x) + \nabla f(x)^\top (y - x), \quad \forall y \in \mathbb{R}^n.$$
因此, $\nabla f(x)$ 可以被看作是函数 $f$ 在点 $x$ 处的一个**全局下界**. 但是, 当函数 $f$ 不可微时, 其在某些点处可能没有梯度. 这时, 我们可以引入**次梯度**的概念来推广梯度的定义.

***Definition* (Subgradient)**: 对于一个凸函数 $f: \mathbb{R}^n \to \mathbb{R}$, 如果存在一个向量 $g \in \mathbb{R}^n$ 满足以下不等式:
$$f(y) \geq f(x) + g^\top (y - x), \quad \forall y \in \mathbb{R}^n,$$
则称 $g$ 是函数 $f$ 在点 $x$ 处的一个**次梯度**.

***Example 1:*** 考虑函数 $f(x) = |x|$, 其在 $x = 0$ 处不可微. 其在任意 $x\neq 0$ 处的次梯度为 $g = \text{sign}(x)$, 而在 $x = 0$ 处的次梯度为 $g \in [-1, 1]$ 中的任意值.

***Definition* (Subdifferential)**: 对于一个凸函数 $f: \mathbb{R}^n \to \mathbb{R}$, 定义其在点 $x$ 处的**次微分**为所有次梯度的集合, 记为 $\partial f(x)$:
$$\partial f(x) = \{g \in \mathbb{R}^n: f(y) \geq f(x) + g^\top (y - x), \forall y \in \mathbb{R}^n\}.$$

***Theorem* (Existence of Subgradients)**: 次梯度对任意凸函数在定义域中的内点一定存在. 即对于任意凸函数 $f$, 其定义域为 $\text{dom}(f)$. 给定任意 $\mathbf{x}_0\in \text{int}(\text{dom}(f))$, 则 $\partial f(\mathbf{x}_0) \neq \emptyset$.
  - *Proof*: 
    - 令 $C = \text{epi}(f) = \{(\mathbf{x}, t) \in \mathbb{R}^n \times \mathbb{R}: t \geq f(\mathbf{x})\}$. 由于 $f$ 是凸函数, 则 $\text{epi}(f)$ 是一个凸集. 又由于 $\mathbf{x}_0 \in \text{int}(\text{dom}(f))$, 则 $(\mathbf{x}_0, f(\mathbf{x}_0))$ 是 $\text{epi}(f)$ 的一个边界点. 
      - 回顾 Supporting Hyperplane Theorem: 对于一个凸集 $C \subseteq \mathbb{R}^n$ 和边界点 $\mathbf{x}_0\in \partial C$, 存在一个超平面 $H := \{x: \mathbf{g}^\top (\mathbf{x} - \mathbf{x}_0) = 0\}$, 使得 $C$ 完全位于 $H$ 的一侧, 即 $\mathbf{g}^\top (\mathbf{x} - \mathbf{x}_0) \leq 0$ 对任意 $x \in C$ 都成立.
      - 因此, 根据 Supporting Hyperplane Theorem, 存在一个不全为 $0$ 的向量 $\mathbf{g} := \begin{bmatrix} \mathbf{g}_x\\ g_t \end{bmatrix} \in \mathbb{R}^{n+1}$ 使得对任意 $(\mathbf{x}, t) \in \text{epi}(f)$ , 都有:
        $$\begin{bmatrix} \mathbf{g}_x \\ g_t\end{bmatrix}^\top \left(\begin{bmatrix} \mathbf{x}  \\ t \end{bmatrix} - \begin{bmatrix} \mathbf{x}_0 \\ f(\mathbf{x}_0) \end{bmatrix}\right) = \mathbf{g}_x^\top (\mathbf{x} - \mathbf{x}_0) + g_t (t - f(\mathbf{x}_0)) \leq 0.$$
        整理即有:
        $$\mathbf{g}_x^\top (\mathbf{x} - \mathbf{x}_0) \leq g_t (f(\mathbf{x}_0) - t), \quad \forall (\mathbf{x}, t) \in \text{epi}(f).$$
    - 首先判断上述不等式中 $g_t$ 的符号.
      - 要使不等式对于给定的 $\mathbf{x}_0$ 和由此确认的 $\mathbf{g}$ 关于任意 $(\mathbf{x}, t) \in \text{epi}(f)$ 都成立, **首先断言必有 $g_t \leq 0$**. 
        - 否则, 若 $g_t \gt 0$, 则取 $t \to +\infty$ 时, 右侧 $g_t (f(\mathbf{x}_0) - t) \to -\infty$, 而左侧 $\mathbf{g}_x^\top (\mathbf{x} - \mathbf{x}_0)$ 为固定的有限值, 不等式无法成立. 
      - 进一步还可以根据 $\mathbf{x}_0 \in \text{int}(\text{dom}(f))$ 的假设, **断言必有 $g_t < 0$**. 
        - 否则若 $g_t = 0$, 则不等式变为 $\mathbf{g}_x^\top (\mathbf{x} - \mathbf{x}_0) \leq 0$ 对任意 $(\mathbf{x}, t) \in \text{epi}(f)$ 都成立. 特别地, 取 $t = f(\mathbf{x})$ , 同样有 $\mathbf{g}_x^\top (\mathbf{x} - \mathbf{x}_0) \leq 0$ 对任意 $(\mathbf{x}, f(\mathbf{x})) \in \text{epi}(f)$, 即对任意 $\mathbf{x} \in \text{dom}(f)$ 成立. 
        - 此时, 由于内点 $\mathbf{x}_0$ 的性质, 在其小邻域内的点 $\mathbf{x} := \mathbf{x}_0 + \epsilon \mathbf{g}_x$ 也属于 $\text{dom}(f)$, 其中 $\epsilon > 0$ 是一个足够小的常数. 代入上述不等式, 则有 $\mathbf{g}_x^\top (\mathbf{x}_0 + \epsilon \mathbf{g}_x - \mathbf{x}_0) = \epsilon \|\mathbf{g}_x\|^2 \leq 0$, 从而 $\mathbf{g}_x = 0$. 
        - 因此, 若 $g_t = 0$, 则 $\mathbf{g} = \begin{bmatrix} \mathbf{g}_x \\ g_t \end{bmatrix} = \mathbf{0}$, 与 $\mathbf{g}$ 不全为 $0$ 的假设矛盾.
    - 由于上述不等式对于任意 $(\mathbf{x}, t) \in \text{epi}(f)$ 都成立, 特别地, 取 $t = f(\mathbf{x})$ , 则有:
      $$\mathbf{g}_x^\top (\mathbf{x} - \mathbf{x}_0) \leq g_t (f(\mathbf{x}_0) - f(\mathbf{x})), \quad \forall \mathbf{x} \in \text{dom}(f).$$
      - 由于 $g_t < 0$, 上式等价于:
        $$\left(-\frac{\mathbf{g}_x}{g_t}\right)^\top (\mathbf{x} - \mathbf{x}_0) \leq f(\mathbf{x}) - f(\mathbf{x}_0), \quad \forall \mathbf{x} \in \text{dom}(f).$$
        - 因此, 定义 $g := -\frac{\mathbf{g}_x}{g_t}$, 则 $g$ 是函数 $f$ 在点 $\mathbf{x}_0$ 处的一个次梯度, 即 $g \in \partial f(\mathbf{x}_0)$.


***Example 2:*** 考虑函数 $f(x) = \max\{f_1(x), f_2(x)\}$, 其中 $f_1, f_2: \mathbb{R}^n \to \mathbb{R}$ 是两个可微凸函数. 则:
- 对于 $f_1(x) > f_2(x)$ 的点 $x$,  $\partial f(x) = \{\nabla f_1(x)\}$.
- 对于 $f_1(x) < f_2(x)$ 的点 $x$,  $\partial f(x) = \{\nabla f_2(x)\}$.
- 对于 $f_1(x) = f_2(x)$ 的点 $x$, $\partial f(x) = \text{conv}\{\nabla f_1(x), \nabla f_2(x)\} = \{\alpha \nabla f_1(x) + (1-\alpha) \nabla f_2(x): \alpha \in [0, 1]\}$. 即在 $f_1$ 和 $f_2$ 的梯度之间的任意凸组合都是 $f$ 在点 $x$ 处的次梯度.

***Example 3:*** 上一个例子还可以进一步推广到 $n$ 个函数的最大值. 定义 $f(x) = \max_{i=1...n} f_i(x)$, 其中每个 $f_i: \mathbb{R}^n \to \mathbb{R}$ 都是可微凸函数. 定义 active set $\mathcal{A}(x) = \{i: f_i(x) = f(x)\}$, 即在某点 $x$ 处达到最大值的函数索引集合 (若 $\mathcal{A}(x)$ 只有一个元素, 则 $f$ 在点 $x$ 处可微; 否则, $f$ 在点 $x$ 处不可微). 则:
$$\partial f(x) = \text{conv}\left\{\nabla f_i(x): i \in \mathcal{A}(x)\right\} = \left\{\sum_{i \in \mathcal{A}(x)} \alpha_i \nabla f_i(x): \alpha_i \geq 0, \sum_{i \in \mathcal{A}(x)} \alpha_i = 1\right\}.$$

- *Proof*
    - 对于任意 $\alpha_i \geq 0$ 且 $\sum_{i \in \mathcal{A}(x)} \alpha_i = 1$, 定义 $g = \sum_{i \in \mathcal{A}(x)} \alpha_i \nabla f_i(x)$. 
    - 由支撑超平面定理, 对于任意 $y \in \mathbb{R}^n$ 和 $i \in \mathcal{A}(x)$, 都有 $f_i(y) \geq f_i(x) + \nabla f_i(x)^\top (y - x)$.
    - 又根据 $\max$ 的性质, 对于任意 $y \in \mathbb{R}^n$: $f(y) = \max_{i=1...n} f_i(y) \geq \sum_{i \in \mathcal{A}(x)} \alpha_i f_i(y)$.
    - 因此, 对于任意 $y \in \mathbb{R}^n$:
        $$\begin{aligned}
        f(y) &\geq \sum_{i \in \mathcal{A}(x)} \alpha_i f_i(y) \\
        &\geq \sum_{i \in \mathcal{A}(x)} \alpha_i [f_i(x) + \nabla f_i(x)^\top (y - x)] \\
        &= \sum_i \alpha_i f_i(x) + \left(\sum_{i \in \mathcal{A}(x)} \alpha_i \nabla f_i(x)\right)^\top (y - x) \\
        &:= f(x) + g^\top (y - x).
        \end{aligned}$$


***Example 4:*** 特别地, 考虑 indicator function $\delta_C(x)$, 其定义为:
$$\delta_C(x) = \begin{cases}0, & x \in C \\ +\infty, & x \notin C\end{cases}$$
其中 $C \subseteq \mathbb{R}^n$ 是一个凸集. 则 $\delta_C$ 的次微分 $\partial \delta_C(x)$ 恰为 $C$ 在点 $x$ 处的**法向量**集合, 记为 $\mathcal{N}_C(x)$:
$$\partial \delta_C(x) :=  \mathcal{N}_C(x) = \{g \in \mathbb{R}^n: g^\top (y - x) \leq 0, \forall y \in C\}.$$

- *Proof*
    - 由次梯度定义, 对于任意 $g \in \partial \delta_C(x)$ 和 $y \in C$, 有:
        $$\delta_C(y) \geq \delta_C(x) + g^\top (y - x).$$
        其中假设 $x \in C$.
        - 如果 $y \notin C$, 则 $\delta_C(y) = +\infty$. 此时不等式变为 $\infty \geq 0 + g^\top (y - x)$, 恒成立.
        - 如果 $y \in C$, 则 $\delta_C(y) = 0$. 此时不等式变为 $0 \geq 0 + g^\top (y - x)$, 即 $g^\top (y - x) \leq 0$, 即要求 $g^\top (y - x) \leq 0$ 对任意 $y \in C$ 都成立. 记为 $g \in \mathcal{N}_C(x)$.
- 在几何上, $y-x$ 表示从 $x$ 指向集合内任意点 $y$ 的向量, $g^\top (y - x) \leq 0$ 表示 $g$ 与 $y-x$ 之间的夹角大于等于 $90^\circ$, 即 $g$ 是指向集合外部的法向量. 

## Properties of Subgradients

***Property 1***: **次微分在凸函数定义域内为凸闭集, 在定义域内点为非空有界集**. 对于凸函数 $f$: (1) 对任意 $x \in \text{dom}(f)$, 次微分 $\partial f(x)$ 是一个凸且闭的集合, 但可能为空; (2) 对于任意 $y\in \text{int}(\text{dom}(f))$, $\partial f(y)$ 非空且有界. 

***Property 2***: **凸函数若某在某点可微, 则其梯度是唯一的次梯度**. 如果 $f$ 在点 $x\in \text{int}(\text{dom}(f))$ 处可微, 则 $\partial f(x) = \{\nabla f(x)\}$.
- *Proof*: 
    - 首先, 由于 $f$ 在点 $x$ 处可微, 则梯度 $\nabla f(x)$ 满足次梯度定义.
    - 下证明其唯一性. 假设存在另一个次梯度 $g\in \partial f(x)$, 且 $g \neq \nabla f(x)$. 
      - 由次梯度定义, 对任意 $v \in \mathbb{R}^n$, 考虑满足 $y = x + t v \in \text{dom}(f)$, 其中 $t > 0$ 的点, 则有:
        $$f(x + t v) \geq f(x) + g^\top (t v) = f(x) + t g^\top v.$$
      - 继续变形, 有
        $$\frac{f(x + t v) - f(x) - t\nabla f(x)^\top v}{t\|v\|} \geq \frac{tg^\top v - t\nabla f(x)^\top v}{t\|v\|}.$$
      - 取 $v = g - \nabla f(x)\neq 0$, 则上式变为
        $$\frac{f(x + t v) - f(x) - t\nabla f(x)^\top v}{t\|v\|} \geq \frac{t(g - \nabla f(x))^\top v}{t\|v\|} = \frac{t\|v\|^2}{t\|v\|} = \|v\| > 0.$$
      - 取 $t \to 0$, 根据可微性的定义, 上式左侧趋近于 $0$, 与右侧 $\|v\| > 0$ 矛盾. 因此, 不存在另一个次梯度 $g \neq \nabla f(x)$, 即 $\partial f(x) = \{\nabla f(x)\}$.

***Property 3***: **次梯度对于凸函数是“单调递增”的**. 对于任意凸函数 $f$ 和任意 $x, y \in \text{dom}(f)$, 任意 $g_x \in \partial f(x)$ 和 $g_y \in \partial f(y)$ 都满足:
$$(g_y - g_x)^\top (y - x) \geq 0.$$
- 该性质在一元的特殊情况下很好理解. 例如对于 $f(x) = \exp(x)$, 其次梯度为 $\partial f(x) = \{\exp(x)\}$, 则对于任意 $x < y$, 都有 $\exp(y) - \exp(x) > 0$ 和 $y - x > 0$, 从而 $(\exp(y) - \exp(x))(y - x) > 0$.

***Property 4***: **次梯度的图象 $\{(x, g): g \in \partial f(x)\}$ 是闭集.** 对于任意闭凸函数 $f$, 考虑序列 $\{x_k\}$ 且 $x_k \to \bar{x}$, 对应 $g_k \in \partial f(x_k)$ 且 $g_k \to \bar{g}$, 则 $\bar{g} \in \partial f(\bar{x})$.

***Property 5***: **凸函数 $f(x)$ 关于方向 $d$ 的方向导数 $\partial f(x; d)$ 是 $f$ 在 $x$ 出所有次梯度与方向 $d$ 的内积的最大值**. 具体地, 定义 $f$ 在点 $x$ 关于方向 $d$ 的方向导数为:
$$\partial f(x; d) = \lim_{t \to 0^+} \frac{f(x + t d) - f(x)}{t} = \inf_{t > 0} \frac{f(x + t d) - f(x)}{t}.$$
- *Proof Sketch*: 
    - 由方向导数的定义:
        $$\partial f(x; d) = \inf_{t > 0} \frac{f(x + t d) - f(x)}{t}.$$ 
    - 同时又知在内点 $x$ 处, 存在次梯度 $g \in \partial f(x)$, 使得对任意 $t > 0$:
        $$f(x + t d) \geq f(x) + g^\top (t d).$$
    - 因此, 对任意 $t > 0$:
        $$\inf_{t > 0} \frac{f(x + t d) - f(x)}{t} \geq \inf_{t > 0} \frac{f(x) + g^\top (t d) - f(x)}{t} = g^\top d.$$
    - 上述即说明方向导数是任意次梯度与方向 $d$ 的内积的上界. 通过进一步的分析能够证明其为上确界. 过程略. 

关于次梯度, 有如下运算规则 (往往默认在内点以及各函数定义域的交集内等一般情况下):
- **线性组合**: 对于任意 $a, b > 0$ 和凸函数 $f, g: \mathbb{R}^n \to \mathbb{R}$, 有:
    $$\partial (a f + b g)(x) = a \partial f(x) + b \partial g(x).$$
- **Affine 变换**: 若 $A \in \mathbb{R}^{m \times n}$ 是一个矩阵, $b \in \mathbb{R}^m$ 是一个向量, 则:
    $$\partial (f(Ax + b)) = A^\top \partial f(Ax + b) .$$
- **最大值**: 对于 $f(x) = \max_{i=1...n} f_i(x)$, 其中每个 $f_i: \mathbb{R}^n \to \mathbb{R}$ 都是可微凸函数, 定义 active set $\mathcal{A}(x) = \{i: f_i(x) = f(x)\}$, 则:
    $$\partial f(x) = \text{conv}\left\{\cup_{i \in \mathcal{A}(x)} \partial f_i(x)\right\}.$$
  - 更一般地, 考虑 $f(x) = \max_{i \in S} f_i(x)$, 其中 $S$ 是任意集合 (可能是不可列等情况), 同样定义 active set $\mathcal{A}(x) = \{i \in S: f_i(x) = f(x)\}$, 则:
    $$\partial f(x) \supseteq \text{cl}\left(\text{conv}\left\{\cup_{i \in \mathcal{A}(x)} \partial f_i(x)\right\}\right).$$
    - 其中 $\text{cl}$ 表示闭包, 因为在某些 pathological 情况下, $\partial f(x)$ 可能包含一些极限点, 但不包含某些凸组合.
    - 当 $f_s$ 和 $S$ 满足了一些额外的正则化条件时, 上述包含关系可以变为等式.

## Optimality Condition for Subgradients

首先不考虑约束, 仅考虑一个凸函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的最小化问题 $\min_x f(x)$. 

***Theorem* (Optimality Condition for Subgradients)**: 对于一个凸函数 $f: \mathbb{R}^n \to \mathbb{R}$, 如果 $0 \in \partial f(x^*)$, 则 $x^*$ 是 $f$ 的一个全局最小点. 反之, 如果 $x^*$ 是 $f$ 的一个全局最小点, 则 $0 \in \partial f(x^*)$.
- 对于可微函数, 上述定理退化为 $f$ 在 $x^*$ 处的梯度 $\nabla f(x^*)$ 等于 $0$ 是 $f$ 的一个全局最小点的充分必要条件. 但对于不可微函数, 上述定理提供了一个更一般的最优性条件.

进一步考虑任意凸优化问题, 此时由于约束的存在, 其最优性并不一定在全局最小点处达到. 不过仍然可以通过次梯度来刻画其局部最优点的性质. 回顾, 对于一个凸优化问题, 给定 $f$ 是凸且可微的, 则
$$\begin{aligned}\min_x &\quad f(x) \\
\text{s.t.} &\quad x \in C\end{aligned}$$
其在 $x$ 是最优点的一阶充要条件为:
$$
\nabla f(x)^\top (y - x) \geq 0, \quad \forall y \in C.
$$
该条件也可以通过次梯度来分析. 
- 将上述优化问题整理为无约束优化问题的形式:
    $$\min_x f(x) + \delta_C(x),$$
其中 $\delta_C(x)$ 是集合 $C$ 的 indicator function. 
- 此时对于扩展后的目标函数 $f(x) + \delta_C(x)$, 其全局最优条件为 $0 \in \partial (f + \delta_C)(x)$. 
  - 根据次微分的线性组合规则, 上式等价于 $0 \in \partial f(x) + \partial \delta_C(x)$, 即存在 $g_f \in \partial f(x)$ 和 $g_\delta \in \partial \delta_C(x)$ 使得 $g_f + g_\delta = 0$. 
  - 又根据 $\partial \delta_C(x) = \mathcal{N}_C(x)$, 则存在 $g_f \in \partial f(x) = \{\nabla f(x)\}$ 和 $g_\delta \in \mathcal{N}_C(x)$ 使得 $g_f + g_\delta = 0$, 即 $-\nabla f(x) \in \mathcal{N}_C(x)$. 
    - 回顾 $\mathcal{N}_C(x) = \{g \in \mathbb{R}^n: g^\top (y - x) \leq 0, \forall y \in C\}$, 则 $-\nabla f(x) \in \mathcal{N}_C(x)$ 等价于 $\nabla f(x)^\top (y - x) \geq 0, \forall y \in C$, 与之前的最优性条件一致.

因此, 对于任意一个凸优化问题, 我们都可以给出其最优点的一个一般性条件: 
$$0 \in \partial f(x) + \mathcal{N}_C(x).$$


***Example 1:*** 对于 $y\in \mathbb{R}^n, X\in \mathbb{R}^{n\times p}$ 和 $\lambda \geq 0$, 考虑 Lasso 问题:
$$\min_\beta \frac{1}{2}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1.$$
- 该问题的次微分最优性条件为:
    $$\begin{aligned}
    0 &\in \partial \left(\frac{1}{2}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1\right) \\
    & = \partial \left(\frac{1}{2}\|y - X\beta\|_2^2\right) + \partial \left(\lambda \|\beta\|_1\right) \\
    & = -X^\top (y - X\beta) + \lambda \partial \|\beta\|_1.
    \end{aligned}$$
- 故整理有 $X^\top (y - X\beta)  = \lambda g$, 其中  $g \in \partial \|\beta\|_1$, 即对于每个 $i= 1\cdots p$:
    $$g_i = \begin{cases}1, & \beta_i > 0 \\ -1, & \beta_i < 0 \\ [-1, 1], & \beta_i = 0\end{cases}.$$