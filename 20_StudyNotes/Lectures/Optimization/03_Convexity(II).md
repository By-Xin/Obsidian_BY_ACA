# Convexity (II): Convex Optimization Problems

> Lecture Reference: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/

## Optimization Problems

> Ref: Boyd & Vandenberghe, Convex Optimization, Section 4.2

### Problem Formulation

回顾, 一个凸优化问题具有如下形式:
$$\begin{aligned}
\min_{x\in D} \quad & f(x)\\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i=1, \ldots, m \\
& Ax = b
\end{aligned}$$
其中 $f, g_i$ 是凸函数, optimization domain $D = \bigcap_{i=1}^m \text{dom}(g_i) \bigcap \text{dom}(f)$ 是凸集.
- $f$ 称作 criterion 或 objective function.
- $g_i$ 称作 inequality constraint functions.
- 对于满足所有约束条件的 $x$ 称作 feasible point, 否则称作 infeasible point.
- 若存在 $x^*$ 是所有 feasible point 中使 $f(x)$ 取得最小值的点, 则称 $x^*$ 为 optimal point 或该问题的 solution, 最小值 $f(x^*)$ 称作 optimal value.
- 若 $x$ 是 feasible point, 且有 $f(x) \leq f^*(x)+\epsilon$, 则称 $x$ 为 $\epsilon$-suboptimal point.
- 若 $x$ 是 feasible point 且 $g_i(x) = 0$ 对某些 $i$ 成立, 则称该约束条件 $g_i$ 在 $x$ 处为 active constraint, 否则称为 inactive constraint.
- 一个凸的最小化问题可以等价于最大化 $-f(x)$ 的问题, 因此有时也称 $f$ 为 cost function.

同时方便起见, 定义优化问题的最优值 $p^\star$:
$$\begin{aligned}
p^\star = \inf \{f(x) \mid g_i(x) \leq 0, \forall i~; Ax=b\}
\end{aligned}$$
- 这里 $p^\star$ 可能为 $-\infty$, 若存在 feasible point 使得 $f(x)$ 可以任意小, 则称该问题为 unbounded below.

---

### Global and Local Optimality

- 若优化问题有解, 则记所有最优解组成的集合为 $X_{\text{opt}}$:
  $$
  X_{\text{opt}} = \{x^* \in \argmax f(x) \mid g_i(x) \leq 0, \forall i~; Ax=b\}
  $$

- 若优化问题对于 $\epsilon > 0$ 满足
  $$
  f(x) \leq p^\star + \epsilon
  $$
  则称 $x$ 为 $\epsilon$-suboptimal point.

- 若对于某个 feasible point $x_0$, 存在某个 $R>0$, 满足
  $$
  f(x_0)  = \inf \{f(z) \mid g_i(z) \leq 0, \forall i~; Az=b; \|z - x_0\|_2 \leq R\}
  $$
  则称 $x_0$ 为该优化问题的局部最优解.

关于最优解集, 有两条重要性质:
1. 由于 $f$ 和 $g_i$ 均为凸函数, 且约束条件均为凸集, 因此 $X_{\text{opt}}$ 也是凸集.
   - *Proof.*
     - 假设 $x_1, x_2 \in X_{\text{opt}}$, 则对任意 $\theta \in [0,1]$, 有:
       - $g_i(\theta x_1 + (1-\theta)x_2) \leq \theta g_i(x_1) + (1-\theta)g_i(x_2) \leq 0$, 因此 $\theta x_1 + (1-\theta)x_2$ 满足不等式约束条件. 
       - $A(\theta x_1 + (1-\theta)x_2) = \theta Ax_1 + (1-\theta)Ax_2 = \theta b + (1-\theta)b = b$, 因此 $\theta x_1 + (1-\theta)x_2$ 满足等式约束条件.
       - $f(\theta x_1 + (1-\theta)x_2) \leq \theta f(x_1) + (1-\theta)f(x_2) = f^*$, 因此 $\theta x_1 + (1-\theta)x_2$ 也是最优解.
     - 综上, $\theta x_1 + (1-\theta)x_2 \in X_{\text{opt}}$, 因此 $X_{\text{opt}}$ 是凸集.

2. 若 $f$ 严格凸, 则最优解若存在则唯一. 故可立刻推出任意局部最优解也是全局最优解.
    - *Proof.*
      - 假设存在 $x_1, x_2 \in X_{\text{opt}}$, 且 $x_1 \neq x_2$, 则对任意 $\theta \in (0,1)$, 有:
         - $f(\theta x_1 + (1-\theta)x_2) < \theta f(x_1) + (1-\theta)f(x_2) = f^*$, 这与 $f^*$ 为最小值矛盾.
      - 因此, 最优解唯一.

### Different Forms of Convex Optimization Problems

对于上述的优化问题, 我们还可以等价地给出其等价形式:
$$\begin{aligned}
\min_x f(x) \quad \text{s.t.} \quad x\in C
\end{aligned}$$
  - 其中 $C = \{x \mid g_i(x) \leq 0, \forall i~; Ax=b\}$ 是约束条件的可行域, 即所有满足约束条件的 $x$ 的集合.

亦或通过引入 indicator function $\delta_C(x)$, 将问题转化为如下无约束优化问题:
$$\begin{aligned}
\min_x f(x) + \delta_C(x)
\end{aligned}$$
- 其中 $\delta_C(x)$ 是 indicator function, 定义为:
    $$\begin{aligned}
    \delta_C(x) = \begin{cases}
    0 & \text{if } x\in C \\
    \infty & \text{otherwise}
    \end{cases}
    \end{aligned}$$

---

***Example* (LASSO)** 给定 $\boldsymbol{y} \in \mathbb{R}^n$, $\boldsymbol{X} \in \mathbb{R}^{n \times p}$, LASSO 问题定义为:
$$\begin{aligned}
\min_{\boldsymbol{\beta} \in \mathbb{R}^p} \quad & \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\|_2^2 \\
\text{s.t.} \quad & \|\boldsymbol{\beta}\|_1 \leq \lambda
\end{aligned}$$

- 若 $n\ge p$ 且 $\boldsymbol{{X}}$ 是列满秩的, 则该问题的解是唯一的. 
  - *Proof.* 
    - 考虑 $\nabla^2 \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\|_2^2 = 2\boldsymbol{X}^\top\boldsymbol{X}$, 由于 $\boldsymbol{X}$ 是列满秩的, 因此 $\boldsymbol{X}^\top\boldsymbol{X}$ 是正定矩阵, 故 $\|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\|_2^2$ 是严格凸函数, 因此该问题的解唯一.
- 若 $n < p$ 的高维场景, 则该问题可能存在多个解.

***Example* (SVM)** 考虑如下支持向量机(SVM)的优化问题. 对于给定的训练数据集 $\{(\boldsymbol{x}_i, y_i)\}_{i=1}^n$, 其中 $\boldsymbol{x}_i \in \mathbb{R}^p$ 是样本特征, $y_i \in \{-1, 1\}$ 是样本标签, SVM 的优化问题定义为 (其中 $\boldsymbol{\xi} = (\xi_1, \ldots, \xi_n)^\top$ 是松弛变量, 允许一定程度的分类错误):
$$\begin{aligned}
\min_{\boldsymbol{\beta} \in \mathbb{R}^p,\beta_0 \in \mathbb{R}, \boldsymbol{\xi}\in \mathbb{R}^n} \quad & \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 + C \sum_{i=1}^n \xi_i \\
\text{s.t.} \quad \quad& y_i (\boldsymbol{x}_i^\top \boldsymbol{\beta} + \beta_0) \geq 1 - \xi_i,  \quad i=1, \ldots, n\\
& \xi_i \geq 0, \quad i=1, \ldots, n
\end{aligned}$$

---

### First-Order Optimality Conditions

> Ref: Boyd & Vandenberghe, Convex Optimization, Section 4.2

给定可微的凸函数 $f$, 考虑如下优化问题:
$$\begin{aligned}
\min_x f(x) \quad \text{s.t.} \quad x\in C
\end{aligned}$$

则 $x^*$ 是该问题的最优解当且仅当:
$$\begin{aligned}
\nabla f(x^*)^\top (x - x^*) \geq 0, \quad \forall x\in C \quad (\star)
\end{aligned}$$

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202602021150491.png)

- *Proof.*
  - $(\star) \Rightarrow x^*$ 是最优解:
    - 由凸性可知, 对于 $x^*$ 及任意 $y\in \text{dom}(f)$, 有 
      $$f(y) \geq f(x^*) + \nabla f(x^*)^\top (y - x^*)$$
    - 由 $(\star)$ 可知, 对于任意 $y\in C$, 有 $\nabla f(x^*)^\top (y - x^*) \geq 0$. 
    - 又由于 $C \subseteq \text{dom}(f)$, 因此 $f(y) \geq f(x^*)$ 对任意 $y\in C$ 成立, 即 $x^*$ 是最优解.
  - $x^*$ 是最优解 $\Rightarrow (\star)$:
    - 用反证法, 假设 $x^*$ 是最优解但存在 $y\in C$ 使得 $\nabla f(x^*)^\top (y - x^*) < 0$.
    - 此时考虑从 $x^*$ 到 $y$ 的线段上的点 $z(t) = ty + (1-t)x^*$, 其中 $t\in [0,1]$. 并记在该点处的函数值为 $g(t):=f(z(t))$. *下尝试说明, 由反证法假设, 我们可以找到某 $t>0$ 使得 $z(t)\in C$ 且 $g(t) < g(0) = f(x^*)$.*
    - 首先由凸优化问题可知, $z(t)\in C$ 对任意 $t\in [0,1]$ 成立.
    - 接着考虑 $g(t)$ 的导数 $\mathrm{d}g(t)/\mathrm{d}t = \nabla f(z(t))^\top \frac{\mathrm{d}z(t)}{\mathrm{d}t} = \nabla f(z(t))^\top (y - x^*)$. 特别地, 当 $t=0$ 时, 有 $\mathrm{d}g(0)/\mathrm{d}t = \nabla f(x^*)^\top (y - x^*) < 0$.
    - 由导数的定义可知, 存在 $\delta > 0$ 使得当 $0 < t < \delta$ 时, 有 $\mathrm{d}g(t)/\mathrm{d}t < 0$, 即 $g(t)$ 在该区间内是递减的. 故在这一区间内, 一定存在某 $t_0 > 0$ 使得 $g(t_0) < g(0) = f(x^*)$. 故矛盾.
  - 从几何角度看, $\nabla f(x^*)^\top (y - x^*) \geq 0$ 意味着向量 $y - x^*$ 与 $\nabla f(x^*)$ 的夹角锐角, 即从 $x^*$ 指向任意 feasible point 的向量与 $\nabla f(x^*)$ (即最陡上升方向)的夹角锐角, 即从 $x^*$ 出发沿着任意 feasible point 的方向, 函数值都是不减的.

---

***无约束情况***: 

特别地, 当 $C=\mathbb{R}^n$ 时, 上述条件退化为 $\nabla f(x^*) = 0$. 当 $f$ 是严格凸函数时, 该条件为充要条件. 若为一般可微函数, 则该条件为必要条件.
- *Proof.* 下说明退化之方式. 
  - 由于 $f$ 是可微函数, 因此在任意 $x\in \text{dom}(f)$, 都能找到小球 $B(x, \epsilon)$ 使得 $B(x, \epsilon) \subseteq \text{dom}(f)$, 即任意足够靠近 $x$ 的点都是 feasible point.
  - 因此构造 $y = x^* - \epsilon \nabla f(x^*)$, 对于足够小的 $\epsilon > 0$, $y$ 仍是 feasible point.
  - 因此, $\nabla f(x^*)^\top (y - x^*) = \nabla f(x^*)^\top (-\epsilon \nabla f(x^*)) = -\epsilon \|\nabla f(x^*)\|_2^2 \ge 0$. 又由于 $\epsilon > 0$, 因此 $\|\nabla f(x^*)\|_2^2 = 0$. 故 $\nabla f(x^*) = 0$.

---

***仅含等式约束情况***: 

考虑凸优化问题:
$$\begin{aligned}
\min_{\mathbf{x}\in \mathbb{R}^n} f(\mathbf{x}) \quad \text{s.t.} \quad A\mathbf{x} = \mathbf{b}
\end{aligned}$$

其中 $f$ 是可微函数, $A \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$. 

对于仅含等式约束的凸优化问题其一阶条件退化为: 存在 $\mathbf{u}^* \in \mathbb{R}^m$ 使得
$$\begin{aligned}
\nabla f(\mathbf{x}^*) + A^\top \mathbf{u}^* = 0
\end{aligned}$$

其中 $\mathbf{u}^*$ 是拉格朗日乘子, 该条件亦称为 Lagrange Multiplier Optimality Condition.

- *Proof.* 
  - 根据初始的一阶最优条件, $\mathbf{x}^*$ 应当满足 $A\mathbf{x}^* = \mathbf{b}$; 且对所有满足 $A\mathbf{y}=\mathbf{b}$ 的 $\mathbf{y}$, 有
    $$\begin{aligned}
    \nabla f(\mathbf{x}^*)^\top (\mathbf{y} - \mathbf{x}^*) \geq 0.
    \end{aligned}$$
  - 由于 $\mathbf{y},\mathbf{x}^*$ 同时满足等式约束, 故 $A(\mathbf{y} - \mathbf{x}^*) := A\mathbf{u} = 0$. 这说明, 所有的可行位移都应处在 $A$ 的零空间内, 即 $\mathbf{u} \in \text{Nul}(A) = \{\mathbf{u} \mid A^\top \mathbf{u} = 0\}$. 换言之, 所有可行的 $\mathbf{y}$ 都应满足 $\mathbf{y} = \mathbf{x}^* + \mathbf{u}, \mathbf{u} \in \text{Nul}(A)$, 即 $\mathbf{y}$ 和 $\mathbf{u}$ 是一一对应的, 故最优性条件可以改写为:
    $$\begin{aligned}
    \nabla f(\mathbf{x}^*)^\top \mathbf{u} \geq 0, \quad \forall \mathbf{u} \in \text{Nul}(A)
    \end{aligned}$$
  - 而由于 $\mathbf{u}$ 的取值任意性, 必有 $\nabla f(\mathbf{x}^*)^\top \mathbf{u} = 0$. 这是因为定同时有 $\nabla f(\mathbf{x}^*)^\top \mathbf{u} \geq 0$ 和 $\nabla f(\mathbf{x}^*)^\top (-\mathbf{u}) \geq 0$ 成立. 故有
    $$\begin{aligned}
    \nabla f(\mathbf{x}^*)^\top \mathbf{u} = 0, \quad \forall \mathbf{u} \in \text{Nul}(A)
    \end{aligned}$$
  - 而这一表述等价于 $\nabla f(\mathbf{x}^*) \perp \text{Nul}(A)$. 又根据线性代数结论, $A$ 的 Null Space 的正交补空间为 Row Space, 即 $A^\top$ 的 Column Space , 因此 $\nabla f(\mathbf{x}^*) \in \text{Col}(A^\top)$, 即存在 $\mathbf{v} \in \mathbb{R}^{n}$, 使得 $\nabla f(\mathbf{x}^*) = A^\top \mathbf{v}$. 记 $\mathbf{u}^* = -\mathbf{v}$, 则有
    $$\begin{aligned}
    \nabla f(\mathbf{x}^*) + A^\top \mathbf{u}^* = 0. 
    \end{aligned}$$

  $\square$


## Equivalence between Different Forms of Optimization Problems

> Ref: Boyd & Vandenberghe, Convex Optimization, Section 4.1.3

有一些变换可以将不同形式的优化问题相互转化. 

### Transforms of Variables

设 $\phi: \mathbb{R}^n \to \mathbb{R}^n$ 是一个 1-1 映射, 其象能够包含原始问题的定义域 $\mathcal{D}$ (即 $\phi(\text{dom}(\phi)) \supseteq \mathcal{D}$). 则原始问题:
$$\begin{aligned} 
\min ~ f(x)& \\
\text{s.t.} ~ g_i(x) &\leq 0, \quad i=1,\ldots,m \\
h_j(x) &= 0, \quad j=1,\ldots,p
\end{aligned}$$
等价于:
$$\begin{aligned}
\min ~ f(\phi(z)) := \tilde f(z)& \\
\text{s.t.} ~ g_i(\phi(z)) := \tilde g_i(z) & \leq 0, \quad i=1,\ldots,m \\
h_j(\phi(z)) := \tilde h_j(z) &= 0, \quad j=1,\ldots,p
\end{aligned}$$

显然, 如果 $x$ 解决了原始问题, 则 $z = \phi^{-1}(x)$ 解决了变换后的问题, 反之亦然.

### Transforms of Objective and Constraint Functions

对于单调递增函数 $\psi_0: \mathbb{R} \to \mathbb{R}$; 以及函数 $\psi_1, \ldots, \psi_m: \mathbb{R} \to \mathbb{R}$ 满足当且仅当 $t \leq 0$ 时 $\psi_i(t) \leq 0$, 函数 $\psi_{m+1}, \ldots, \psi_{m+p}: \mathbb{R} \to \mathbb{R}$ 满足当且仅当 $t = 0$ 时 $\psi_j(t) = 0$, 则原始问题等价于
$$\begin{aligned}
\min ~ \psi_0(f(x))& := \tilde f(x) \\
\text{s.t.} ~ \psi_i(g_i(x)) & := \tilde g_i(x) \leq 0, \quad i=1,\ldots,m \\
\psi_{m+j}(h_j(x)) &:= \tilde h_j(x) = 0, \quad j=1,\ldots,p
\end{aligned}$$

***Example* (Minimizing Euclidean Norm)** 无约束的 Euclidean 范数最小化问题:
$$\begin{aligned}
\min_x \|Ax - b\|_2
\end{aligned}$$
等价于:
$$\begin{aligned}
\min_x \|Ax - b\|_2^2
\end{aligned}$$

- 虽然这两个问题是等价的, 但并不相同, 二者在定义域上的可微性不同.


### Partial Optimum

> Ref: Boyd & Vandenberghe, Convex Optimization, Section 3.2.5, Section 4.1.3

如如果 $f$ 关于 $(x,y)$ 是凸函数 , 且 $C$ 是非空凸集, 则函数
$$\begin{aligned}
g(x) = \inf_{y\in C} f(x,y)
\end{aligned}$$
也是 $x$ 的凸函数 (只要 $g(x)$ 在其定义域内取值有限). 
- 其中 $\text{dom}(g) = \{x \mid \exists y\in C, \text{ s.t. } (x,y)\in\text{dom}(f)\}$, 即 $\text{dom}(f)$ 在 $x$ 方向上的投影.    

- *Proof.* 
  - 根据 Infimum 的定义, 对于任意 $x_1, x_2 \in \text{dom}(g)$, 以及任意 $\epsilon > 0$, 存在 $y_1, y_2 \in C$ 使得:
    $$\begin{aligned}
    g(x_1) &\geq f(x_1, y_1) - \epsilon \\
    g(x_2) &\geq f(x_2, y_2) - \epsilon
    \end{aligned}$$
  - 根据 $g$ 的定义, 对于对于任意 $\theta \in [0,1]$:
    $$\begin{aligned}
    g(\theta x_1 + (1-\theta)x_2) &= \inf_{y\in C} f(\theta x_1 + (1-\theta)x_2, y) \\
    &\leq f(\theta x_1 + (1-\theta)x_2, \theta y_1 + (1-\theta)y_2) \quad (\text{Infimum})\\
    &\leq \theta f(x_1,y_1) + (1-\theta) f(x_2,y_2) \quad (\text{Jensen Ineq.})\\
    &= \theta g(x_1) + (1-\theta) g(x_2) + \epsilon  \quad (\text{Infimum})
    \end{aligned}$$
  - 由于 $\epsilon > 0$ 是任意的, 故有:
    $$\begin{aligned}
    g(\theta x_1 + (1-\theta)x_2) \leq \theta g(x_1) + (1-\theta) g(x_2)
    \end{aligned}$$

因此, 该性质说明, 我们能够将一个关于多个变量的凸函数, 通过对部分变量取 Infimum 的方式, 得到一个关于剩余变量的凸函数. 


### Eliminating Equality Constraints

考虑如下优化问题:
$$\begin{aligned}
\min_{\mathbf{x}\in \mathbb{R}^n} f(\mathbf{x}) \quad \text{s.t.} \quad A\mathbf{x} = \mathbf{b}, \quad g_j(\mathbf{x}) \leq 0, \quad j=1,\ldots,p
\end{aligned}$$
- 其中 $A \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$. 

则可以通过如下变换将等式约束消除. 对等式约束, 我们可以确定任意一个 particular solution $\mathbf{x}_0$ 满足 $A\mathbf{x}_0 = \mathbf{b}$. 则任意满足等式约束的 $\mathbf{x}$ 都可以表示为:
$$\begin{aligned}
\mathbf{x} = \mathbf{x}_0 + \mathbf{v}, \quad \text{where } \mathbf{v} \in \text{Nul}(A)
\end{aligned}$$
- 这是因为对于任意满足 $A\mathbf{x} = \mathbf{b}$ 的 $\mathbf{x}$, 有 $A(\mathbf{x} - \mathbf{x}_0) = 0$, 因此 $\mathbf{v} := \mathbf{x} - \mathbf{x}_0 \in \text{Nul}(A)$.

对于 $\text{Nul}(A)$, 其维度 $\text{dim}(\text{Nul}(A)) = n - \text{rank}(A) := k$, 故可以找到一组基 $\{\phi_1, \ldots, \phi_k\}$ 使得 $\text{Nul}(A) = \text{span}\{\phi_1, \ldots, \phi_k\}$. 换言之, 对于任意 $\mathbf{v} \in \text{Nul}(A)$, 都存在 $z_1, \ldots, z_k \in \mathbb{R}$ 使得:
$$\begin{aligned}
\mathbf{v} = \sum_{i=1}^k z_i \phi_i:= \Phi\mathbf{z},
\end{aligned}$$
- 其中 $\Phi = [\phi_1, \ldots, \phi_k] \in \mathbb{R}^{n \times k}$, $\mathbf{z} = (z_1, \ldots, z_k)^\top \in \mathbb{R}^k$.

故我们可以将 $\mathbf{x}$ 表示为 $\mathbf{x} = \mathbf{x}_0 + \Phi \mathbf{z}$, 因此原始问题等价于:
$$\begin{aligned}
\min_{\mathbf{z}\in \mathbb{R}^k} f(\mathbf{x}_0 + \Phi \mathbf{z}), \quad \text{s.t.} \quad g_j(\mathbf{x}_0 + \Phi \mathbf{z}) \leq 0, \quad j=1,\ldots,p
\end{aligned}$$

### Slack Variables to Eliminate Inequality Constraints

注意到 $g_i(x) \leq 0$ 等价于 存在 $s_i \geq 0$ 使得 $g_i(x) + s_i = 0$. 因此, 原始问题
$$\begin{aligned}
\min_x \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i=1,\ldots,m; \\
& h_j(x) = 0, \quad j=1,\ldots,p
\end{aligned}$$
等价于:
$$\begin{aligned}
\min_{x,s} \quad & f(x) \\
\text{s.t.} \quad &g_i(x) + s_i = 0, \quad i=1,\ldots,m; \\
&h_j(x) = 0, \quad j=1,\ldots,p; \\ 
&s_i \geq 0, \quad i=1,\ldots,m
\end{aligned}$$

其中 $s_i$ 称作 slack variable. 通过引入 $s_i$, **每个不等式约束都可以转化为一个等式约束加上一个非负约束**. 

然而, 除非 $g_i, i=1,\ldots,m$ 都是 affine 函数, 否则该变换并不保持凸性.

### Relaxation Non-affine Equality Constraints

考虑一般的优化问题:
$$\begin{aligned}
\min_x \quad & f(x) \quad \text{s.t.}\quad  x \in C
\end{aligned}$$

我们总可以找到一个更大的集合 $\tilde C \supseteq C$, 考虑
$$\begin{aligned}
\min_x \quad & f(x) \quad \text{s.t.}\quad
  x \in \tilde C
\end{aligned}$$

则该问题称作原始问题的 relaxation. 显然, relaxation 问题的最优值不大于等于原始问题的最优值.

- 特别地, 对于凸但非 affine 的等式约束 $h_j(x) = 0$, 我们可以考虑将其优化为 $h_j(x) \leq 0$, 以确保凸性.

- 注意, relaxation 问题的解不一定是原始问题的可行解, 其只能作为原始问题的下界估计. 若 relaxation 问题的解恰好是原始问题的可行解, 则该解也是原始问题的最优解.