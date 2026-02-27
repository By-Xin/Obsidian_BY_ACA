# Duality Uses and Correspondents

> [!quote] References
> - Lecture: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/
> - Reading: 
>   1. 最优化: 建模、算法与理论, 刘浩洋等, 2.6 小节 (Conjugate Functions)
>   2. [一文读懂凸优化中的「对偶」概念（二）：可以用一个观点解释所有对偶吗？ - 江南FLY的文章 - 知乎](https://zhuanlan.zhihu.com/p/1994408053886432686)
>   3. [一文读懂凸优化中的「对偶」概念（三）：Fenchel 共轭是什么东西？它有什么用？ - 江南FLY的文章 - 知乎](https://zhuanlan.zhihu.com/p/2006663715471250367)

本章将希望通过讨论对偶这一概念在数学中的各种应用, 来串联深化对对偶概念的理解. 

## Dual Norm

***Definition* (Norm and Semi-Norm)**: 对于向量空间 $V$, 其上的范数 (Norm) 是一个从 $V$ 到实数域 $\mathbb{R}$ 的函数, 记为 $\|\cdot\|: V \to \mathbb{R}$. 满足以下性质:

1. 绝对齐次性: 对于任意 $x \in V$ 和任意实数 $t$, 有 $\|tx\| = |t| \|x\|$.
2. 三角不等式: 对于任意 $x, y \in V$, 有 $\|x + y\| \leq \|x\| + \|y\|$.
3. 非负性: 对于任意 $x \in V$, 有 $\|x\| \geq 0$. 并且当且仅当 $x = \mathbf{0}$ (零向量) 时, 有 $\|x\| = 0$.
   - 若对于某范数只满足绝对齐次性和三角不等式, 但允许在某些 $x\neq \mathbf{0}$ 时 $\|x\| = 0$, 则称为半范数 (Semi-Norm). 

范数有如下性质:
- 对于任意范数 $f(x) = \|x\|$, 其都是凸函数. 
  - *Proof*: 立即由三角不等式和齐次性推出, 对于任意 $x, y \in V$ 和 $t \in [0,1]$, 有:
    $$
    \|tx + (1-t)y\| \leq t\|x\| + (1-t)\|y\|
    $$
    因此, $f(x)$ 是凸函数.

***Definition* (Dual Norm)**: 对于任意范数 $f(x) = \|x\|$, 其对偶范数 $f_*$ 定义为:
$$
f_*(x) := \|x\|_* = \max_{\|z\| \leq 1} z^\top x
$$

对于范数和其对偶范数, 由定义立刻可以推得如下不等关系:
$$
|z^\top x| \leq \|x\| \cdot \|z\|_*
$$

- *Proof*
  - 对任意 $x\neq 0$, 都有:
    $$
    z^\top x = \|x\| z^\top \frac{x}{\|x\|}
    $$
  - 而 $x/\|x\|$ 是单位向量, 因此有:
    $$
    z^\top x = \|x\| z^\top \frac{x}{\|x\|} \leq \|x\| \cdot \max_{\|v\| \leq 1} z^\top v = \|x\| \cdot \|z\|_*
    $$


给定任意一个具体范数, 其对偶范数总是存在的, 例如:
- 对于 $\ell_p$ 范数, 其对偶范数为 $\ell_q$ 范数, 其中 $1/p + 1/q = 1$.
  - $\ell_1$ 范数的对偶范数为 $\ell_\infty$ 范数; $\ell_2$ 范数的对偶范数为 $\ell_2$ 范数
  - *Proof*.
    - 由 Holder's Inequality 可得:
      $$
      |z^\top x| \leq \|z\|_q \cdot \|x\|_p
      $$
    - 若 $\|x\| \leq 1$, 则有:
      $$
      |z^\top x| \leq \|z\|_q \cdot \|x\|_p \leq \|z\|_q \cdot 1 = \|z\|_q
      $$
    - 因此:
      $$
      \max_{\|x\|_p \leq 1} |z^\top x| \leq \max_{\|x\| \leq 1} \|z\|_q \cdot \|x\|_p = \|z\|_q
      $$
    - 并且确能找到一个 $x$ 使得上式等号成立. 
   

- 对于 Trace Norm, 其对偶范数为 Nuclear Norm.
  - Trace Norm 类比 $\ell_1$ 范数, 定义为 $\|X\|_* = \sum_{i=1}^n \sigma_i(X)$, 其中 $\sigma_i(X)$ 是 $X$ 的奇异值.
  - Nuclear Norm 类比 $\ell_\infty$ 范数, 定义为 $\|X\|_{op} = \sigma_{\max}(X)$, 其中 $\sigma_{\max}(X)$ 是 $X$ 的最大奇异值.

>[!note] Young's Inequality and Holder's Inequality
>
> Holder's Inequality 是一个常见的代数不等式, 其证明由 Young's Inequality 推出. 而对偶相当于是给出了该不等式的几何解释.
>
> ***Lemma* (Young's Inequality)**: 设 $p>1$, 且 $q$ 满足 $1/p + 1/q = 1$, 则对于任意 $a,b\geq 0$, 有:
> $$
> ab \leq \frac{a^p}{p} + \frac{b^q}{q}
> $$
>
> ***Theorem* (Holder's Inequality)**: 设 $1\leq p \leq \infty$, 且 $q$ 满足 $1/p + 1/q = 1$, 则对任意 $x,z \in \mathbb{R}^n$, 有:
> $$
> |x^\top z| \leq \|x\|_p\cdot \|z\|_q
> $$


对偶范数还有一个重要性质:
$$
(\|x\|_*)_* = \|x\|
$$

- *Proof*:
  - 考虑如下优化问题:
    $$
    \min_y \|y\|, \quad \text{s.t.}\quad y = x
    $$
    - 显然, 唯一可行解为 $y = x$, 故最优解也是 $p^* = \min y = \|x\|$.
  - 其 Lagrangian 为:
    $$
    L(y, \lambda) = \|y\| + \lambda^\top (x-y) = \|y\| + \lambda^\top x - \lambda^\top y
    $$
  - 其对偶函数为:
    $$
    d(\lambda) = \inf_y L(y, \lambda) = \inf_y (\|y\| + \lambda^\top x - \lambda^\top y) = \inf_y (\|y\| - \lambda^\top y) + \lambda^\top x
    $$
    - 若 $\|\lambda\|_* \leq 1$
      - 根据广义 Holder's Inequality 可得:
        $$
        \lambda^\top y \leq \|\lambda\|_* \cdot \|y\| \leq \|y\|
        $$
      - 故 $\|y\| - \lambda^\top y \geq \|y\| - \|y\|= 0$.
      - 此时 $d(\lambda) = \lambda^\top x$.
    - 若 $\|\lambda\|_* > 1$
      - 根据 Dual Norm 的定义, 有:
        $$
        \|\lambda\|_* = \max_{\|y\| \leq 1} y^\top \lambda\geq 1
        $$
      - 这说明在单位球内, 至少存在一个 $y$ 使得 $y^\top \lambda \geq 1$. 故有关系:
        $$
        y^\top \lambda \geq 1 \geq \|y\|
        $$
      - 此时 $d(\lambda) = \inf_y (\|y\| - \lambda^\top y) = -\infty$.
    - 综上所述, Lagrangian 的对偶函数为:
      $$
      d(\lambda) = \begin{cases}
      \lambda^\top x, & \text{if } \|\lambda\|_* \leq 1 \\
      -\infty, & \text{if } \|\lambda\|_* > 1
      \end{cases}
      $$
  - 而 Lagrange 对偶问题本身即为对偶函数的最优值:
    $$
    \max_{\|\lambda\|_* \leq 1} d(\lambda)=\max_{\|\lambda\|_* \leq 1} \lambda^\top x = (\|x\|_*)_*
    $$
  - 最后可以验证, 最开始构造的 $\min_y \|y\|, \quad \text{s.t.}\quad y = x$ 是一个满足 Slater 条件的凸优化问题, 满足强对偶性, 故原问题和对偶问题的最优值相等, 即:
    $$
    \|x\| = (\|x\|_*)_*
    $$

    $\square$

## Conjugate Functions

***Definition* (Fenchel Conjugate)**: 对于函数 $f: \mathbb{R}^n \to \mathbb{R}$, 其 Fenchel 共轭 (Fenchel Conjugate) $f^*: \mathbb{R}^n \to \mathbb{R}$ 定义为:
$$
f^*(\mathbf{t}) = \sup_{\mathbf{x} \in \mathbb{R}^n} (\mathbf{t}^\top \mathbf{x} - f(\mathbf{x}))
$$

Fenchel 共轭在几何上可以直观理解如下 (此处考虑一元情况).
- 共轭函数将遍历每一个可能的 $t$ 的取值. 对于当前的 $t$ 值:
  - 考虑一个以 $t$ 为斜率的直线, 记作 $l_t = tx + c$. 
  - 保证 $l_t$ 恒在 $f$ 的下方, 即 $l_t \leq f(x)$ 恒成立. 
  - 在此前提下, 通过移动 $c$ 使得 $l_t$ 与 $f$ 相切, 则此时的截距 $c$ 即为 $-f^*(t)$, 即在该 $t$ 值下, 共轭函数 $f^*(t)$ 的值为 $-c$.
- *Proof*:
  - 对于直线 $l_t = tx + c$, 由于其恒在 $f$ 的下方, 故有:
    $$
    tx + c \leq f(x)
    $$
  - 因此:
    $$
    c \leq f(x) - tx
    $$
  - 在相切的极端情况下, 表示 $c$ 已取最大值, 此时有 $\inf_x f(x) - tx = c_{\max}$. 故取相反数, 有 $\sup_x (f(x) - tx) = -c_{\max}$. 即 $f^*(t) = -c_{\max}$.

Fenchel 共轭有如下性质:
1. 不论 $f$ 是否为凸函数, $f^*$ 都是凸函数.
2. 若 $f$ 为凸函数, 则 $f^{**} = f$. 这意味着, 对于凸函数, 共轭函数和凸函数是一一对应的. 给定一个凸函数的共轭函数, 可以唯一确定一个凸函数:
   $$
   f(x) = \sup_{\mathbf{t} \in \mathbb{R}^n} (\mathbf{t}^\top \mathbf{x} - f^*(\mathbf{t}))
   $$
3. 即使对于非凸函数, 也有 $f^{**}$ 也能正常定义, 且有 $f^{**}\leq f$ 恒成立. 此时, $f^{**}$ 可以看作是 $f$ 的一种凸化(convexification)近似. 
4. **Fenchel-Young Inequality**: $\forall \mathbf{t} \in \mathbb{R}^n, \mathbf{x} \in \mathbb{R}^n$, 有: 
    $$
    f(\mathbf{x}) + f^*(\mathbf{t}) \geq \mathbf{t}^\top \mathbf{x}
    $$
5. 若 $f(\mathbf{u},\mathbf{v}) = f_1(\mathbf{u}) + f_2(\mathbf{v})$, 则有 $f^*(\mathbf{u},\mathbf{v}) = f_1^*(\mathbf{u}) + f_2^*(\mathbf{v})$.
6. 如果函数 $f$ 是 $\beta$-strongly convex 的, 则有 $f^{*}$ 是 $\beta^{-1}$-smooth 的. 
7. 对于闭凸函数 $f$ 及任意 $\mathbf{x},\mathbf{y} \in \mathbb{R}^n$, 有:
   $$
   \mathbf{x} \in \partial f(\mathbf{y}) \iff \mathbf{y} \in \partial f^*(\mathbf{x}) \iff \mathbf{x}^\top \mathbf{y} = f(\mathbf{x}) + f^*(\mathbf{y})
   $$

>[!note] Geometric Interpretation and Convexity Revisit
>
> 首先补充一些解析几何的基础知识. 
>
> - 给定一个方向向量 $\mathbf{u} \in \mathbb{R}^n$ 以及截距 $b\in \mathbb{R}$, 则可以由此定义一个超平面: $\mathbf{u}^\top \mathbf{x} = b$.
>    <img src="https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202602270001392.png" width="50%">
>   - 其含义为, 所有在 $\mathbf{u}$ 方向上投影长度为 $b$ 的点的集合, 即为一个超平面.
>   - 其中, $\mathbf{u}$ 为超平面的法向量; $b$ 为超平面距离原点的距离, 并且其正负代表了超平面是沿着法向量的正方向还是负方向.
> - 给定超平面 $\mathbf{u}^\top \mathbf{x} = b$, 其将空间分为两个半空间, 分别记作 $\mathbf{u}^\top \mathbf{x} \leq b$ 和 $\mathbf{u}^\top \mathbf{x} \geq b$. 此处符号本身就代表了是沿着 $\mathbf{u}$ 的箭头方向同侧或异侧.
>
> - 给定凸集 $C \subseteq \mathbb{R}^n$, 固定方向 $\mathbf{u}$, 则定存在一个超平面, 使得 $C$ 中的所有点都在该超平面的同侧, 即支撑超平面 (Supporting Hyperplane) 定理.
>   - 若记这个 $\mathbf{u}$ 方向上的关于 $C$ 的支撑超平面为 $\mathbf{u}^\top \mathbf{x} = h_C(\mathbf{u})$, ($h$ 称为支撑函数 (Support Function)), 则有:
>     $$
>     h_C(\mathbf{u}) = \sup_{\mathbf{x} \in C} \mathbf{u}^\top \mathbf{x}
>     $$
>       - 此处可以理解为: $h_C(\mathbf{u})$ 即为超平面的截距项, 相当于是在给定平面的方向角度后, 不断沿着 $\mathbf{u}$ 指向方向移动, 直到恰好和 $C$ 中的点"相切", 此时得到的最大距离即为 $h_C(\mathbf{u})$.  
>       - 支持函数本身还具有如下性质:
>           - 支持函数本身是凸函数, 且具有齐次性: $h_C(\alpha \mathbf{u}) = \alpha h_C(\mathbf{u})$.
>           - 两个凸集的 Minkowski Sum 的支撑函数等于两个凸集的支撑函数之和: $h_{C_1 + C_2}(\mathbf{u}) = h_{C_1}(\mathbf{u}) + h_{C_2}(\mathbf{u})$. 其中 $C_1 + C_2 = \{ \mathbf{x} + \mathbf{y} \mid \mathbf{x} \in C_1, \mathbf{y} \in C_2 \}$.
>           - 若集合在 $\mathbf{u}$ 方向上的支撑超平面与集合的交点唯一, 则其在该点可微. 
>
> - 任意凸集也可以表示为其所有的支撑超平面的交集.

>[!note] Conjugate Functions and Duality 
>
> 我们可以从 epigraph 的角度来理解 Fenchel 共轭 (参见 Reading References 3). 
>
> - 回忆, 对于凸函数 $f$, 其 epigraph 为这个函数图象及其上方所有点构成的集合, 即:
>   $$
>   \text{epi}(f) = \{(\mathbf{x},t)^\top \in \mathbb{R}^{n+1}: t \geq f(\mathbf{x})\}
>   $$
>
>   并且对于凸函数, 其 epigraph 是凸集. 
>
> - 根据支撑超平面的性质, 可以得到, 对于凸集 $\text{epi}(f)$, 其在 $\mathbf{u}\in \mathbb{R}^{n+1}$ 方向上的支撑超平面为 $\mathbf{u}^\top \mathbf{x} = h_f(\mathbf{u})$, 其中:
>   $$
>   h_f(\mathbf{u}) = \sup_{(\mathbf{x},t)^\top \in \text{epi}(f)} \left\langle \begin{pmatrix} \mathbf{x} \\t \end{pmatrix}, \mathbf{u} \right\rangle
>   $$
>   - 进一步, 将方向 $\mathbf{u}$ 分解为 $\mathbf{u} = (\mathbf{v},\nu)^\top$, 其中 $\mathbf{v} \in \mathbb{R}^n$ 和 $\nu \in \mathbb{R}$, 则有:
>     $$
>     h_f(\mathbf{v},\nu) = \sup_{(\mathbf{x},t)^\top \in \text{epi}(f)} (\nu t + \mathbf{v}^\top \mathbf{x})
>     $$
>   - 注意到, 在最后一个维度 (即 $t$ 所在维度) 上, 恒有 $t \ge f(\mathbf{x})$, 即 $t$ 是有下界而无上界的. 因此, 为保证上式有解 (确定一个有穷的超平面), 必须有 $\nu \lt 0$ (这里为了和后文记号一致, 取 $\nu = -\lambda \lt 0$, 则此刻有:
>     $$
>     \begin{aligned}
>     h_f(\mathbf{v},\nu) &= \sup_{(\mathbf{x},t)^\top \in \text{epi}(f)} (\mathbf{v}^\top \mathbf{x}-\lambda t)\\
>     &\leq \sup_{\mathbf{x} \in \mathbb{R}^n} \left(\mathbf{v}^\top \mathbf{x}-\lambda f(\mathbf{x})\right),\quad \text{since~} t \ge f(\mathbf{x})\\
>     &= \lambda \left(\sup_{\mathbf{x} \in \mathbb{R}^n} \left\langle \frac{\mathbf{v}}{\lambda}, \mathbf{x} \right\rangle -  f(\mathbf{x})\right)\\
>     &= \lambda f^*\left(\frac{\mathbf{v}}{\lambda}\right)
>     \end{aligned}
>     $$
> - 惊! 这里发现, 对于凸函数的 epigraph, 其支撑超平面的截距项 (即支持函数) 竟然就是其 Fenchel 共轭函数! 即:
>   $$
>   h_f(\mathbf{v},\nu) = \lambda f^*\left(\frac{\mathbf{v}}{\lambda}\right)
>   $$
> 
> 反过头来, 我们再从代数角度说明共轭函数和支撑函数的关系. 
> 
> - 考虑一个关于凸集的指示函数, 即:
> 
>   $$
>   f(\mathbf{x}) = I_C(\mathbf{x}) = \begin{cases} 0, & \mathbf{x} \in C \\ +\infty, & \mathbf{x} \notin C \end{cases}
>   $$
>   其中 $C$ 是凸集. 
> 
> - 易知, 对于该函数, 当 $\mathbf{x} \in C$ 时, $f(\mathbf{x}) = 0$; 当 $\mathbf{x} \notin C$ 时, $f(\mathbf{x}) = +\infty$. 因此为取 $\sup$ 时, 只有当 $\mathbf{x} \in C$ 时, 函数有界, 即:
>   $$
>   f^*(\mathbf{y}) = \sup_{\mathbf{x} \in C} (\mathbf{y}^\top \mathbf{x})
>   $$
> 
> - 该共轭函数恰恰就是刚刚介绍的支撑函数! 这也再次从数学上印证了支撑函数和共轭函数是等价的.
> 
> 
> 因此, 支撑函数和共轭函数, 其本质相当于都在描述对于凸集的支持平面的超参数化. 



---

### Examples

#### Conjugate of Simple Quadratic Function

- 考虑 $f(x) = \frac{1}{2} \mathbf{x}^\top Q \mathbf{x}$，其中 $Q \in \mathbb{S}^n_{++}$（即 $Q$ 是 $n$ 阶实对称正定矩阵）。
- 根据共轭函数的定义，有：
  $$
  f^*(\mathbf{y}) = \sup_{\mathbf{x} \in \mathbb{R}^n} (\mathbf{y}^\top \mathbf{x} - \frac{1}{2} \mathbf{x}^\top Q \mathbf{x})
  $$
- 易知，对于该函数，这个表达式是关于 $\mathbf{x}$ 的严格 concave 函数，在 $\mathbf{x} = Q^{-1} \mathbf{y}$ 处取得极值。因此，共轭函数为：
  $$
  f^*(\mathbf{y}) = \frac{1}{2} \mathbf{y}^\top Q^{-1}\mathbf{y}
  $$

#### Conjugate of Norm

任何范数的共轭是其对偶范数单位球的 indicator, 即:
$$
\left(\|x\|_p\right)_{\text{conjugate}} \longleftrightarrow I_{\{\|x\|_q \leq 1\}}, \quad \frac{1}{p} + \frac{1}{q} = 1
$$

>[!warning] 注意区分 Dual Norm 和 Conjugate Norm. 


#### Lasso Dual

我们下面尝试用这个例子将上述的几个理论和性质串联起来.

给定 $\mathbf{y}\in \mathbb{R}^n, X \in \mathbb{R}^{n\times p}$, 考虑如下 Lasso 问题:
$$
\min_{\boldsymbol{\beta}\in \mathbb{R}^p} \frac{1}{2} \|\mathbf{y} - X\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1
$$

- 首先, 根据 Lagrange Duality 的思路, 我们需要将该问题转为一个含约束问题. 引入辅助变量 $\mathbf{z} = X\boldsymbol{\beta}$, 则有:
  $$
  \min_{\boldsymbol{\beta}\in \mathbb{R}^p, \mathbf{z}\in \mathbb{R}^n} \frac{1}{2} \|\mathbf{y} - \mathbf{z}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1, \quad \text{s.t.} \quad \mathbf{z} = X\boldsymbol{\beta}
  $$
  - 则对应的 Lagrangian 为:
    $$
    L(\boldsymbol{\beta}, \mathbf{z}, \boldsymbol{\lambda}) = \frac{1}{2} \|\mathbf{y} - \mathbf{z}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1 + \boldsymbol{\mu}^\top (\mathbf{z} - X\boldsymbol{\beta})
    $$
  - 则对应的 Lagrange Dual Function 为:
    $$
    \begin{aligned}
    d(\boldsymbol{\mu}) &= \inf_{\boldsymbol{\beta}\in \mathbb{R}^p, \mathbf{z}\in \mathbb{R}^n} L(\boldsymbol{\beta}, \mathbf{z}, \boldsymbol{\mu}) = \inf_{\boldsymbol{\beta}\in \mathbb{R}^p, \mathbf{z}\in \mathbb{R}^n} \left( \frac{1}{2} \|\mathbf{y} - \mathbf{z}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1 + \boldsymbol{\mu}^\top (\mathbf{z} - X\boldsymbol{\beta}) \right) \\
    &= \inf_{\mathbf{z}} \left( \frac{1}{2} \|\mathbf{y} - \mathbf{z}\|_2^2 + \boldsymbol{\mu}^\top \mathbf{z} \right) + \inf_{\boldsymbol{\beta}} \left( \lambda \|\boldsymbol{\beta}\|_1 - \boldsymbol{\mu}^\top (X\boldsymbol{\beta}) \right) \\
    &= \frac{1}{2} \|\mathbf{y}\|_2^2 -\frac{1}{2} \|\mathbf{y}-\boldsymbol{\mu}\|_2^2 - \lambda \cdot\underbrace{\sup_{\boldsymbol{\beta}} \left( - \|\boldsymbol{\beta}\|_1 + \left\langle \boldsymbol{\beta}, \frac{X^\top\boldsymbol{\mu}}{\lambda} \right\rangle \right)}_{\text{Fenchel Conjugate of } \|\boldsymbol{\beta}\|_1} \\
    &= \frac{1}{2} \|\mathbf{y}\|_2^2 -\frac{1}{2} \|\mathbf{y}-\boldsymbol{\mu}\|_2^2 - \lambda\cdot I{\{\|X^\top\boldsymbol{\mu}/\lambda\|_\infty \leq 1\}}
    \end{aligned}
    $$
  - 则对应的对偶问题为:
    $$
    \sup_{\boldsymbol{\mu}} d(\boldsymbol{\mu})  = 
    \frac{1}{2} \|\mathbf{y}\|_2^2 -\frac{1}{2} \|\mathbf{y}-\boldsymbol{\mu}\|_2^2,  \quad \text{s.t.} \quad \|X^\top\boldsymbol{\mu}\|_\infty \leq \lambda \\
    \iff \inf_{\boldsymbol{\mu}} \|\mathbf{y}-\boldsymbol{\mu}\|_2^2 \quad \text{subject to} \quad \|X^\top\boldsymbol{\mu}\|_\infty \leq \lambda
    $$
    - 其相当于一个投影问题, 将 $\mathbf{y}$ 投影到一个凸多面体上: $C:= \{\mathbf{u} \in \mathbb{R}^n: \|X^\top \mathbf{u}\|_\infty \leq \lambda\}$. 
  - 此外, 可以验证 Slater 条件成立, 故强对偶性成立, 最优解必满足 KKT 条件. 
    - 其中, 关于 $\mathbf{z}$ 的 Stationary 条件为:
      $$
      \nabla_{\mathbf{z}} L(\boldsymbol{\beta}, \mathbf{z}, \boldsymbol{\mu}) = \mathbf{z} - \mathbf{y} + \boldsymbol{\mu} = 0 \implies \boldsymbol{\mu} = \mathbf{z} - \mathbf{y} = X\boldsymbol{\beta} - \mathbf{y}
      $$
    - 这说明对偶变量 $\boldsymbol{\mu}$ 事实上即为 Lasso 问题的残差, 对偶问题即为在一个 $\ell_\infty$ 的约束下, 最小化残差.
    - 反过来, $X\boldsymbol{\beta} = \mathbf{y} - \boldsymbol{\mu}$, 因此原问题的解同时也相当于对偶问题的残差.


- 上述的 Lasso Dual 问题事实上是 **Fenchel Duality** 的特例. 因此若在不引入辅助变量 $\mathbf{z}$ 的情况下, 也可以直接使用 Fenchel Duality 来求解. 其一般形式为:
  $$
  \inf_{\boldsymbol{\beta}} f(X\boldsymbol{\beta}) + g(\boldsymbol{\beta})
  $$
  给出结论, 其对偶问题为:
  $$
  \sup_{\boldsymbol{\mu}} - f^*(\boldsymbol{\mu}) - g^*(-X^\top \boldsymbol{\mu})
  $$
  - 首先利用该结论求解 Lasso 问题. 将 Lasso 的表达式与 Fenchel Duality 这里的记号对齐, 则有原始形式 $\inf_{\boldsymbol{\beta}} \frac{1}{2} \|\mathbf{y} - X\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1$, 其中 $f(X\boldsymbol{\beta}) = \frac{1}{2} \|\mathbf{y} - X\boldsymbol{\beta}\|_2^2$ 和 $g(\boldsymbol{\beta}) = \lambda \|\boldsymbol{\beta}\|_1$. 
  - 分别计算 $f^*$ 和 $g^*$ 的值. 
    - $f(\mathbf{z}) = \frac{1}{2} \|\mathbf{y} - \mathbf{z}\|_2^2 =\frac{1}{2} \|\mathbf{y}\|_2^2 - \mathbf{y}^\top \mathbf{z} + \frac{1}{2} \|\mathbf{z}\|_2^2$, 因此根据 Fenchel 共轭的定义, 有:
      $$
      \begin{aligned}
      f^*(\boldsymbol{\mu}) &= \sup_{\mathbf{z}} (\boldsymbol{\mu}^\top \mathbf{z} - f(\mathbf{z})) \\
      &= \sup_{\mathbf{z}} (\boldsymbol{\mu}^\top \mathbf{z} - \frac{1}{2} \|\mathbf{y}\|_2^2 + \mathbf{y}^\top \mathbf{z} - \frac{1}{2} \|\mathbf{z}\|_2^2) \\
      &= \mathbf{y}^\top \boldsymbol{\mu} + \frac{1}{2} \|\boldsymbol{\mu}\|_2^2 - \frac{1}{2} \|\mathbf{y}\|_2^2
      \end{aligned}
      $$
    - $g(\boldsymbol{\beta}) = \lambda \|\boldsymbol{\beta}\|_1$, 因此根据 Fenchel 共轭的定义, 有:
      $$
      g^*(\boldsymbol{\mu}) = I{\{\|\boldsymbol{\mu}\|_\infty \leq \lambda\}}
      $$
    
  - 因此, 其对偶问题为:
    $$
    \sup_{\boldsymbol{\mu}}\left[- f^*(\boldsymbol{\mu}) - g^*(-X^\top \boldsymbol{\mu})\right]  = \sup_{\boldsymbol{\mu}}\left[-\mathbf{y}^\top \boldsymbol{\mu} - \frac{1}{2} \|\boldsymbol{\mu}\|_2^2 - \frac{1}{2} \|\mathbf{y}\|_2^2 - I{\{\|X^\top\boldsymbol{\mu}\|_\infty \leq \lambda\}}\right] 
    $$
    其等价于:
    $$
    \begin{aligned}
    \inf_{\|X^\top\boldsymbol{\mu}\|_\infty \leq \lambda} \left[\mathbf{y}^\top \boldsymbol{\mu} + \frac{1}{2} \|\boldsymbol{\mu}\|_2^2 + \frac{1}{2} \|\mathbf{y}\|_2^2 \right] &= \inf_{\|X^\top\boldsymbol{\mu}\|_\infty \leq \lambda} \left[\frac{1}{2} \|\mathbf{y}+\boldsymbol{\mu}\|_2^2 - \frac{1}{2} \|\mathbf{y}\|_2^2 +\frac{1}{2} \|\mathbf{y}\|_2^2 \right] \\
    &= \frac{1}{2} \inf_{\|X^\top\boldsymbol{\mu}\|_\infty \leq \lambda} \left[\|\mathbf{y}+\boldsymbol{\mu}\|_2^2 \right]
    \end{aligned}
    $$
  - 若进行变量等价代换, 令 $\mathbf{u} = \mu+\mathbf{y}$, 则有:
    $$
    \inf_{\|X^\top \mathbf{u}\|_\infty \leq \lambda} \left[\|\mathbf{y} - \mathbf{u}\|_2^2 \right]
    $$
    此处和 Lasso 问题的对偶问题形式一致. 


#### Duality and conjugate

最终再强调一下 Duality 和 Conjugate 的关系作为上述的总结. 

当我们考虑一个一般的无约束优化问题:
$$
\min f(x)+g(x)
$$
我们可以引入辅助变量 $z = x$, 则有:
$$
\min f(x)+g(z), \quad \text{s.t.} \quad z = x
$$
因此其 Lagrange 函数为:
$$
L(x, \mu, z) = f(x) + g(z) + \mu (z - x)
$$
因此其 Lagrange Dual Function 为:
$$
\begin{aligned}
d(\mu) &= \inf_{x,z} L(x, \mu, z) \\
&= \inf_{x} \left[ f(x) + g(z) + \mu (z - x) \right] \\
&= - \sup_{x} \left[ -f(x) + \mu x\right] - \sup_{z} \left[ g(z) - \mu z \right]\\
&= - f^*(\mu) - g^*(-\mu)
\end{aligned}
$$


故: 对于 $\inf_x f(x)+g(x)$ 问题, 其对偶问题为 $\sup_{\mu} - f^*(\mu) - g^*(-\mu)$.  对偶和共轭总是这样成对出现的. 


## Dual Cones

***Definition* (Dual Cone)** 对于一个 Cone $K \subseteq \mathbb{R}^n$ ($K$ 是锥, 即满足 $\forall x \in K, \alpha \geq 0, \alpha x \in K$), 其对偶锥 (Dual Cone) 定义为:
$$
K^* = \{ \mathbf{y} \in \mathbb{R}^n \mid \mathbf{y}^\top \mathbf{x} \geq 0 \text{ for all } \mathbf{x} \in K \}
$$

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202602271236250.png)


对于 Dual Cone, 有如下性质:
- $K^*$ 永远是凸锥, 即使 $K$ 不是凸的.
- $y\in K^*$ 当且仅当 $\{y^\top x \ge 0\}$ 的半空间包含 $K$.
- 对于凸且闭的 $K$, 有 $K^{**} = K$.

我们同样可以用支撑超平面的方法来理解 Dual Cones 的定义.  (参见 Reading Reference 2)

- 首选原始的 $K$, 其作为锥, 天然的就必须以原点为顶点. 而对应的支撑超平面也就必然经过原点
- 而考虑其支撑函数, 有
  $$
  h_K(\mathbf{u}) = \sup_{\mathbf{x} \in K} \mathbf{u}^\top \mathbf{x} = \begin{cases}
    \infty & \text{if } \mathbf{u} \notin K^* \\
    0 & \text{if } \mathbf{u} \in K^*
  \end{cases}
  $$

- 因此, 如果我们从支撑超平面的视角去看，对偶锥是非常平凡的记号, 其表示的就是该凸锥的支撑超平面的所有法向量. 

### Dual Cones and Dual Problems

考虑如下一般形式的优化问题:
$$
\min_x f(x), \quad \text{s.t.} \quad x \in K
$$
其中 $f$ 是凸函数, $K$ 是凸锥. 

其对偶问题为:
$$
\max_\mu -f^*(A^\top \mu) - I^*_K(-\mu)
$$
其中 $I^*_K(y) = \sup_{z \in K} y^\top z$, 是 $K$ 的 support function. 因此其等价于
$$
\max_\mu -f^*(A^\top \mu), \quad \text{s.t.} \quad \mu \in K^*
$$
其中 $K^*$ 是 $K$ 的对偶锥. 

