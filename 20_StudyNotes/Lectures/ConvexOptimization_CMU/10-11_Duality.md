# Duality

> [!info] References
> - Lecture: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/
> - Reading: 最优化: 建模、算法与理论, 刘浩洋等, 5.4 小节.

## Duality and Lagrangian

考虑如下一般的含约束的优化问题 (不要求是凸的):
$$\begin{aligned}
& \min_{\mathbf{x}\in \mathbb{R}^n} && f(x) \\
& \text{subject to} && g_i(\mathbf{x}) \leq 0, i\in \mathcal{I} \\
& && h_j(\mathbf{x}) = 0, j\in \mathcal{E}
\end{aligned}$$

其中 $f, g_i, h_j$ 都是定义在 $\mathbb{R}^n$ 上或其子集上的实值函数. 该问题的可行域为: $\mathcal{X} = \{\mathbf{x}\in \mathbb{R}^n: g_i(\mathbf{x}) \leq 0, i\in \mathcal{I}, h_j(\mathbf{x}) = 0, j\in \mathcal{E}\}$. 记该问题的最优值为 $p^* = \inf_{\mathbf{x}\in \mathcal{X}} f(\mathbf{x}) = f(\mathbf{x}^*)$.

***Definition* (Lagrangian)** 对于上述优化问题, 定义 Lagrangian 函数 $L: \mathbb{R}^n \times \mathbb{R}_+^{|\mathcal{I}|} \times \mathbb{R}^{|\mathcal{E}|} \to \mathbb{R}$ 如下:
$$
L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i\in \mathcal{I}} \lambda_i g_i(\mathbf{x}) + \sum_{j\in \mathcal{E}} \nu_j h_j(\mathbf{x})
$$

- 其中 $\boldsymbol{\lambda} = (\lambda_i)_{i\in \mathcal{I}} \in \mathbb{R}_+^{|\mathcal{I}|}$ 是与不等式约束相关的 Lagrange 乘子, $\boldsymbol{\nu} = (\nu_j)_{j\in \mathcal{E}} \in \mathbb{R}^{|\mathcal{E}|}$ 是与等式约束相关的 Lagrange 乘子.

> 注意, 此处 $\mathbf{x}$ 的有关全部约束已经被放入了 Lagrangian 函数中, 因此 $L$ 是定义在 $\mathbb{R}^n$ 上的一个实值函数, 而不再是定义在可行域 $\mathcal{X}$ 上的函数了.

***Definition* (Lagrange Dual Function)** 定义 Lagrange 函数在给定 $\boldsymbol{\lambda}, \boldsymbol{\nu}$ 下对关于 $\mathbf{x}$ 的下确界为 Lagrange Dual Function $g: \mathbb{R}_+^{|\mathcal{I}|} \times \mathbb{R}^{|\mathcal{E}|} \to [-\infty, +\infty)$:
$$
g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}\in \mathbb{R}^n} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
$$
- 给定 $(\boldsymbol{\lambda}, \boldsymbol{\nu})$, 若 $L$ 关于 $\mathbf{x}$ 是 unbounded below 的, 则 $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = -\infty$; 否则 $g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ 是一个实数.
- 由于 $g$ 是关于 $\boldsymbol{\lambda}, \boldsymbol{\nu}$ 的一族逐点定义的 affine 函数的下确界, 可以证明不论原问题的凹凸性, **Lagrange Dual Function 是一个凹函数**.

***Lemma* (Weak Duality)** 对于上述优化问题, $g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ 是原问题的一个下界, 即对于任意 $\boldsymbol{\lambda} \in \mathbb{R}_+^{|\mathcal{I}|}, \boldsymbol{\nu} \in \mathbb{R}^{|\mathcal{E}|}$, 恒有:
$$
g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq p^*
$$
- *Proof*
    - 对于可行解 $\mathbf{x}\in \mathcal{X}$, 由于 $g_i(\mathbf{x}) \leq 0$ 和 $h_j(\mathbf{x}) = 0$, 可得:
        $$
        L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i\in \mathcal{I}} \lambda_i g_i(\mathbf{x}) + \sum_{j\in \mathcal{E}} \nu_j h_j(\mathbf{x}) \leq f(\mathbf{x})
        $$
    - 因此 $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}\in \mathbb{R}^n} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \leq \inf_{\mathbf{x}\in \mathcal{X}} f(\mathbf{x}) = p^*$.
   
    $\square$

***Definition* (Lagrange Dual Problem)** 定义 Lagrange Dual Problem 如下:
$$
\begin{aligned}
& \max_{\boldsymbol{\lambda} \in \mathbb{R}_+^{|\mathcal{I}|}, \boldsymbol{\nu} \in \mathbb{R}^{|\mathcal{E}|}} && g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \max_{\boldsymbol{\lambda} \in \mathbb{R}_+^{|\mathcal{I}|}, \boldsymbol{\nu} \in \mathbb{R}^{|\mathcal{E}|}} \inf_{\mathbf{x}\in \mathbb{R}^n} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
\end{aligned}
$$
- 由于 $g$ 是一个凹函数, Lagrange Dual Problem 是一个凸优化问题 (即使原问题不是凸的).
  - 下图是一个具体的例子, 其原函数为 $\min f(x) = x^4 - 50 x^2 + 100x$ 是显然非凸的, 但其 Lagrange Dual Function 是凸的.
     ![Ref: CMU Lecture Notes.](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202602202314700.png)

- 当 $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = -\infty$ 时, 该下界没有意义. 因此往往我们只考虑那些使 $g$ 有界的 $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ 作为 Lagrange Dual Function 的定义域: $\text{dom}(g) = \{(\boldsymbol{\lambda}, \boldsymbol{\nu}): \boldsymbol{\lambda} \geq \boldsymbol{0}, g(\boldsymbol{\lambda}, \boldsymbol{\nu}) > -\infty\}$. 对于满足条件的 $(\boldsymbol{\lambda}, \boldsymbol{\nu})\in \text{dom}(g)$, 称为 **dual feasible**.

***Definition* (Strong Duality)** 记 Lagrange Dual Problem 的最优值为 $q^* = \sup_{\boldsymbol{\lambda} \in \mathbb{R}_+^{|\mathcal{I}|}, \boldsymbol{\nu} \in \mathbb{R}^{|\mathcal{E}|}} g(\boldsymbol{\lambda}, \boldsymbol{\nu})$, 原问题和 Lagrange Dual Problem 之间的最优值差距为 **duality gap**: $p^* - q^* \geq 0$. 当 $p^* = q^*$ 时, 称原问题满足 **strong duality**.

> [!example] Univariate Example of Lagrangian and Dual Function
>
> ![Ref: Convex Optimization, Boyd. p.231](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250707170216.png)
>
> 如图左图为 Lagrange 函数一元情况的简化例子. 
> - 黑色实线为原问题的目标函数 $f(x)$, 短横线为不等式约束 $g(x) < 0$, 因此红色框住的区域为原问题的可行域 $\mathcal{X}\approx[-0.46, 0.46]$. 
> - 此时的 Lagrangian 函数为 $L(x, \lambda) = f(x) + \lambda g(x)$, 其中 $\lambda \geq 0$ 是与不等式约束相关的 Lagrange 乘子, 在图中为一族点虚线. 
>   - 根据 $\lambda$ 的取值不同, 该族点虚线的具体形状不同, 但可以看到在可行域内, 该族点虚线都在黑色实线的下方, 即 $L(x, \lambda) \leq f(x)$.
> 
> 右图为对应的 Lagrange Dual Function $g(\lambda) = \inf_{x\in \mathbb{R}} L(x, \lambda)$ 的图像. 
>   - 可以看到 $g(\lambda)$ 是一个关于 $\lambda$ 的凹函数, 对应着左图中一族点虚线的下确界.
>   - 图中水平虚线为原问题的最优值 $p^*$. 可以看到对于任意 $\lambda \geq 0$, 都有 $g(\lambda) \leq p^*$, 即满足弱对偶. 
>   - 由于 $g(\lambda)$ 的最大值 $q^*$  距离水平虚线 $p^*$ 有一个 gap, 因此该例子不满足强对偶.

## Examples of Duality for Classic Canonical Problems

### Duality for Linear Programming

考虑如下的线性规划问题:
$$
\begin{aligned}
& \min_{\mathbf{x}\in \mathbb{R}^n} && \mathbf{c}^\top \mathbf{x} \\
& \text{subject to} && A\mathbf{x} = \mathbf{b} \\
& && \mathbf{x} \geq \mathbf{0}
\end{aligned}
$$

- 其中 $\mathbf{c}\in \mathbb{R}^n$ 是目标函数的系数向量, $A\in \mathbb{R}^{m\times n}$ 是约束矩阵, $\mathbf{b}\in \mathbb{R}^m$ 是约束的常数项.

对于等式约束, 引入 Lagrange 乘子 $\boldsymbol{\nu} \in \mathbb{R}^m$; 对于不等式约束, 引入 Lagrange 乘子 $\mathbf{s} \in \mathbb{R}_+^n$. 则该问题的 Lagrangian 函数为:
$$
L(\mathbf{x}, \boldsymbol{\nu}, \mathbf{s}) = \mathbf{c}^\top \mathbf{x} + \boldsymbol{\nu}^\top (A\mathbf{x} - \mathbf{b}) - \mathbf{s}^\top \mathbf{x} = -\mathbf{b}^\top \boldsymbol{\nu} + (A^\top \boldsymbol{\nu} - \mathbf{s} + \mathbf{c})^\top \mathbf{x}
$$

其 Lagrange Dual Function 为:
$$
g(\boldsymbol{\nu}, \mathbf{s}) = \inf_{\mathbf{x}\in \mathbb{R}^n} L(\mathbf{x}, \boldsymbol{\nu}, \mathbf{s}) = \begin{cases}
-\mathbf{b}^\top \boldsymbol{\nu}, & \text{if } A^\top \boldsymbol{\nu} - \mathbf{s} + \mathbf{c} = \mathbf{0} \\
-\infty, & \text{otherwise}
\end{cases}
$$

因此只考虑使得 $A^\top \boldsymbol{\nu} - \mathbf{s} + \mathbf{c} = \mathbf{0}$ 的可行解 $\boldsymbol{\nu}, \mathbf{s}$ 作为 Lagrange Dual Function 的定义域: $\text{dom}(g) = \{(\boldsymbol{\nu}, \mathbf{s}): A^\top \boldsymbol{\nu} - \mathbf{s} + \mathbf{c} = \mathbf{0}\}$. 对于满足条件的 $(\boldsymbol{\nu}, \mathbf{s})\in \text{dom}(g)$, 称为 **dual feasible**.

Lagrange Dual Problem 为:
$$
\begin{aligned}
& \max_{\boldsymbol{\nu}, \mathbf{s}} && -\mathbf{b}^\top \boldsymbol{\nu} \\
& \text{subject to} && A^\top \boldsymbol{\nu} - \mathbf{s} + \mathbf{c} = \mathbf{0} \\
& && \mathbf{s} \geq \mathbf{0}
\end{aligned}
$$

令 $\mathbf{y} = - \boldsymbol{\nu}$, 则 Lagrange Dual Problem 可以写为更熟悉的标准形式:
$$
\begin{aligned}
& \max_{\mathbf{y}} && \mathbf{b}^\top \mathbf{y} \\
& \text{subject to} && A^\top \mathbf{y}  + \mathbf{s} = \mathbf{c} \\
& && \mathbf{s} \geq \mathbf{0}
\end{aligned}
$$

若再次将该对偶问题视为原问题, 则该对偶问题的对偶问题与原问题等价. 事实上, **LP 问题与其对偶问题互为对偶问题**.

### Duality for Quadratic Programming

考虑如下 QP 问题:
$$
\begin{aligned}
& \min_{\mathbf{x}\in \mathbb{R}^n} && \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} \\
& \text{subject to} && A\mathbf{x} = \mathbf{b} \\
& && \mathbf{x} \geq \mathbf{0}
\end{aligned}
$$

其中 $\mathbf{Q}\in \mathbb{S}_n^{++}$ 是正定矩阵, $A\in \mathbb{R}^{m\times n}$ 是约束矩阵, $\mathbf{b}\in \mathbb{R}^m$ 是约束的常数项.

其 Lagrangian 函数为:
$$
L(\mathbf{x}, \mathbf{u}, \mathbf{v}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} - \mathbf{u}^\top \mathbf{x} + \mathbf{v}^\top (A\mathbf{x} - \mathbf{b})
$$

其 Lagrange Dual Function 为:
$$
g(\mathbf{u}, \mathbf{v}) = \inf_{\mathbf{x}\in \mathbb{R}^n} L(\mathbf{x}, \mathbf{u}, \mathbf{v}) = -\frac{1}{2} (\mathbf{c} - \mathbf{u} + A^\top \mathbf{v})^\top \mathbf{Q}^{-1} (\mathbf{c} - \mathbf{u} + A^\top \mathbf{v}) - \mathbf{b}^\top \mathbf{v}
$$

但是若令 $\mathbf{Q}\in \mathbb{S}_n^+$ 是半正定矩阵, 则 Lagrange Dual Function 为:
$$
g(\mathbf{u}, \mathbf{v}) = \begin{cases}
-\frac{1}{2} (\mathbf{c} - \mathbf{u} + A^\top \mathbf{v})^\top \mathbf{Q}^{-1} (\mathbf{c} - \mathbf{u} + A^\top \mathbf{v}) - \mathbf{b}^\top \mathbf{v}, & \text{if } \mathbf{c} - \mathbf{u} + A^\top \mathbf{v} \perp \text{Nul}(\mathbf{Q}) \\
-\infty, & \text{otherwise}
\end{cases}
$$

> [!example]
>
> ![Ref: CMU Lecture Notes](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202602202243818.png)
>
> 如图所示是一个二元场景下的 QP 问题的原问题和对偶问题. 可见, 对偶问题在任意给定 $\mathbf{u}, \mathbf{v}$ 时, 都是一个凹函数, 且都提供了原问题的一个 lower bound. 

