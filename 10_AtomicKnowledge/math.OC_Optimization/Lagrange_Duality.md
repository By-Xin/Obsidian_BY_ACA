---
aliases: [拉格朗日对偶问题, 拉格朗日对偶, Lagrange Duality, Duality]
tags:
  - "#concept"
  - "#math/optimization"
---

# Lagrange Duality

## Lagrange Dual Function

### The Lagrangian

考虑如下标准问题形式 (不一定是凸的):
$$\begin{aligned}
\min_{\boldsymbol{x}} \quad & f_0(\boldsymbol{x}) \\
\text{s.t.} \quad & f_i(\boldsymbol{x}) \leq 0, \quad i = 1, \ldots, m \\
& h_j(\boldsymbol{x}) = 0, \quad j = 1, \ldots, p
\end{aligned}$$
其中 $\boldsymbol{x} \in \mathbb{R}^n$ 是优化变量, 定义域为 $\mathcal{D} \subseteq \mathbb{R}^n$. 记最优取值为 $p^*$, 即:
$$p^* = \inf_{x \in \mathcal{D}, f_i(x)\leq 0, h_j(x)=0} f_0(x)$$

***Definition*** Lagrangian 函数 $L: \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ 在 $\text{dom} L = \mathcal{D} \times \mathbb{R}^m \times \mathbb{R}^p$ 上定义为:
$$L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f_0(\boldsymbol{x}) + \sum_{i=1}^m \lambda_i f_i(\boldsymbol{x}) + \sum_{j=1}^p \nu_j h_j(\boldsymbol{x})$$
其中 $\boldsymbol{\lambda} \in \mathbb{R}^m$ 是拉格朗日乘子, $\boldsymbol{\nu} \in \mathbb{R}^p$ 是对等式约束的拉格朗日乘子.

### Lagrange Dual Function

***Definition*** 拉格朗日对偶函数 $g: \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ 的定义为:
$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\boldsymbol{x} \in \mathcal{D}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})= \inf_{\boldsymbol{x} \in \mathcal{D}} \left( f_0(\boldsymbol{x}) + \sum_{i=1}^m \lambda_i f_i(\boldsymbol{x}) + \sum_{j=1}^p \nu_j h_j(\boldsymbol{x}) \right)$$

注意到, 拉格朗日对偶函数 $g$ 是一个 concave 函数 (即使原优化问题不是凸的), 即对于任意的 $(\boldsymbol{\lambda}, \boldsymbol{\nu}), (\boldsymbol{\lambda}', \boldsymbol{\nu}') \in \mathbb{R}^m \times \mathbb{R}^p$ 和任意的 $\theta \in [0, 1]$, 都有:
$$g(\theta (\boldsymbol{\lambda}, \boldsymbol{\nu}) + (1 - \theta) (\boldsymbol{\lambda}', \boldsymbol{\nu}')) \geq \theta g(\boldsymbol{\lambda}, \boldsymbol{\nu}) + (1 - \theta) g(\boldsymbol{\lambda}', \boldsymbol{\nu}')$$

该性质是由于 $g$ 是关于 $\boldsymbol{\lambda}$ 和 $\boldsymbol{\nu}$ 的一组 affine 函数的 infimum, 这样的函数被证明是 concave 的.

同时若在一些情况下 $g$ 关于 $\boldsymbol{x}$ 是没有下界的, 则 $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = -\infty$, 负无穷是可能取到的.

### Lower Bound Property

***Theorem (Weak Duality)*** 对于上述 Duality 函数, 若 $\boldsymbol{\lambda} \succeq 0$ (表示 $\boldsymbol{\lambda}$ 的所有分量都非负), 则 $\forall \boldsymbol{\nu} \in \mathbb{R}^p$ 该 Dual Function 一定有:
$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq p^*$$

这个定理表明, 对偶函数的值 $g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ 一定不会超过原问题的最优值 $p^*$. 这个结论并不依赖于问题的凸性, 可导性等条件. 其只依赖于拉格朗日函数的结构.

下面逐步对这个定理进行说明.

考虑原问题的任意一个可行解 $\tilde{\boldsymbol{x}}$ (即满足所有约束条件 $f_i(\tilde{\boldsymbol{x}}) \leq 0$ 和 $h_j(\tilde{\boldsymbol{x}}) = 0$ ,$\forall i, j$). 故只要 $\forall i:\lambda_i \geq 0$, 就一定能有:
$$\sum_{i=1}^m \lambda_i f_i(\tilde{\boldsymbol{x}}) + \sum_{j=1}^p \nu_j h_j(\tilde{\boldsymbol{x}}) \leq 0$$

回顾拉格朗日函数的定义:
$$L(\tilde{\boldsymbol{x}}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f_0(\tilde{\boldsymbol{x}}) + \sum_{i=1}^m \lambda_i f_i(\tilde{\boldsymbol{x}}) + \sum_{j=1}^p \nu_j h_j(\tilde{\boldsymbol{x}})
\leq f_0(\tilde{\boldsymbol{x}})$$

又根据对偶函数的定义, 由于 infimum 是所有可能的 $\boldsymbol{x}$ 的取值中最小的, 所以:
$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\boldsymbol{x} \in \mathcal{D}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \leq L(\tilde{\boldsymbol{x}}, \boldsymbol{\lambda}, \boldsymbol{\nu})$$

因此结合上面两式, 可以得到:
$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq f_0(\tilde{\boldsymbol{x}})$$

而这个式子中 $\tilde{\boldsymbol{x}}$ 是任意的可行解, 所以其中一定包含了最优解 $\boldsymbol{x}^*$, 即:
$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq f_0(\boldsymbol{x}^*) = p^*$$


![Ref: Convex Optimization, Boyd. p.231](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250707170216.png)

如图所示是在二维简化场景下的示意图. 
- 左图实线部分为 $f_0(x)$ 是我们的优化目标函数. 短横线部分为约束条件 $f_1(x)$, 其 $f_1(x) \leq 0$ 的部分为可行域, 对应图中框选的区域,约在 $[-0.46,0.46]$ 之间. 可见在当前可行域上的最优解约取在 $x^* \approx -0.46$ 出, 以圆点标记. 该点对应的 $f_0(x^*) \approx 1.54$.图中的若干条虚线即为给定 $\lambda = 0.1, 0.2, \cdots, 1.0$ 时构造的 Lagrangian 函数 $L(x, \lambda) = f_0(x) + \lambda f_1(x)$ 的图像. 很直观地, 在可行域内, 任意虚线均处在实线之下 ($L(x, \lambda) \leq f_0(x)$), 且每条虚线的最小值都不超过 $p^*$ ($\inf_x L(x, \lambda) = g(\lambda) \leq p^*$).
- 延续左图, 对于不同的 $\lambda$ 值, 我们可以得到不同的 Lagrangian 函数 $L(x, \lambda)$ (左图虚线), 又对应了不同的最小值. 因此可以清晰地看到 $g(\lambda) = \inf_x L(x, \lambda)$ 是一个关于 $\lambda$ 的函数. 将该函数图象绘制出来, 如右图所示. 图中的横虚线表示 $p^*$. 可以看到, 即使对于 $f_0, f_1$ 二者均不是 convex 的, 我们最终得到的 dual function $g(\lambda)$ 仍然是一个 concave 的.


需要说明, 在一些情况下 $g(\boldsymbol{\lambda}, \boldsymbol{\nu})=-\infty$, 显然不等式依然成立, 但为 vacuous 的. 因此若希望得到非平凡的结论, 我们常考虑对偶函数的定义域为:
$$\text{dom }g = \{ (\lambda, \nu) \in \mathbb{R}^m \times \mathbb{R}^p \mid g(\lambda, \nu) > -\infty \}$$
此时称满足 $\boldsymbol{\lambda} \succeq 0, g(\boldsymbol{\lambda}, \boldsymbol{\nu}) > -\infty$ 的 $\boldsymbol{\lambda}, \boldsymbol{\nu}$ 为 *feasible* 的.


### Examples of Lagrangian Duality


下面给出一些具体的例子. 

####  Least-norm Solution of Linear Equations

考虑如下优化问题:
$$\begin{aligned}
\min_{\boldsymbol{x}} \quad & \boldsymbol{x}^\top \boldsymbol{x} \\
\text{s.t.} \quad & A \boldsymbol{x} = \boldsymbol{b}
\end{aligned}$$
其中 $A \in \mathbb{R}^{m \times n}$, $\boldsymbol{b} \in \mathbb{R}^m$. 该问题的目标是求解一个最小二乘问题, 即在满足 $A\boldsymbol{x} = \boldsymbol{b}$ 的条件下, 求 $\boldsymbol{x}$ 的二范数最小值.

首先, 该问题的 Lagrangian 函数为:
$$L(\boldsymbol{x}, \boldsymbol{\nu}) = \boldsymbol{x}^\top \boldsymbol{x} + \boldsymbol{\nu}^\top (A\boldsymbol{x} - \boldsymbol{b})$$
其中 $\boldsymbol{\nu} \in \mathbb{R}^m$ 是拉格朗日乘子.

接下来, 计算对偶函数 $g(\boldsymbol{\nu})$:
$$g(\boldsymbol{\nu}) = \inf_{\boldsymbol{x}} L(\boldsymbol{x}, \boldsymbol{\nu}) = \inf_{\boldsymbol{x}} \left( \boldsymbol{x}^\top \boldsymbol{x} + \boldsymbol{\nu}^\top (A\boldsymbol{x} - \boldsymbol{b}) \right)$$

为求解这个 infimum, 对 $\boldsymbol{x}$ 求导并令其为零:
$$\nabla_{\boldsymbol{x}} L(\boldsymbol{x}, \boldsymbol{\nu}) = 2\boldsymbol{x} + A^\top \boldsymbol{\nu} = 0$$
解得:
$$\boldsymbol{x} = -\frac{1}{2} A^\top \boldsymbol{\nu}$$

将其代入对偶函数中:
$$\begin{aligned}
g(\boldsymbol{\nu}) &= \inf_{\boldsymbol{x}} \left( \boldsymbol{x}^\top \boldsymbol{x} + \boldsymbol{\nu}^\top (A\boldsymbol{x} - \boldsymbol{b}) \right) \\
&= -\frac{1}{4} \boldsymbol{\nu}^\top A A^\top \boldsymbol{\nu} - \boldsymbol{\nu}^\top \boldsymbol{b}
\end{aligned}$$

不难发现 $g(\boldsymbol{\nu})$ 是一个关于 $\boldsymbol{\nu}$ 的 concave 函数. 

下验证弱对偶性, 即对于任意 $\nu \in \mathbb{R}^m$, 都有:
$$=\frac{1}{4} \boldsymbol{\nu}^\top A A^\top \boldsymbol{\nu} - \boldsymbol{\nu}^\top \boldsymbol{b} \leq \inf_{\boldsymbol{x}} \{\boldsymbol{x}^\top \boldsymbol{x} \mid A\boldsymbol{x} = \boldsymbol{b}\}$$


对任意可行点 $\tilde{\boldsymbol{x}} \in \{\boldsymbol{x} \mid A\boldsymbol{x} = \boldsymbol{b}\}$, 需验证 $-\frac{1}{4} \boldsymbol{\nu}^\top A A^\top \boldsymbol{\nu} - \boldsymbol{\nu}^\top \boldsymbol{b} \leq \inf \tilde{\boldsymbol{x}}^\top \tilde{\boldsymbol{x}} ~~(\dagger)$. 而由于对于所有可行点 $\tilde{\boldsymbol{x}}$, 都有 $A\tilde{\boldsymbol{x}} = \boldsymbol{b}$, 故 $(\dagger)$ 可以改写为:
$$ -\frac{1}{4} \boldsymbol{\nu}^\top A A^\top \boldsymbol{\nu} - \boldsymbol{\nu}^\top A\tilde{\boldsymbol{x}} \leq \inf \tilde{\boldsymbol{x}}^\top \tilde{\boldsymbol{x}}$$
而该式显然是对任意 $\tilde{\boldsymbol{x}}$ 均成立的, 因为其等价于 $\|\tilde{\boldsymbol{x}}+\frac{1}{2} A^\top \boldsymbol{\nu}\|^2 \geq 0$.

#### Standard Form Linear Programming

考虑如下标准形式的线性规划问题:
$$\begin{aligned}
\min_{\boldsymbol{x}} \quad & \boldsymbol{c}^\top \boldsymbol{x} \\
\text{s.t.} \quad & A \boldsymbol{x} = \boldsymbol{b} , ~\boldsymbol{x} \succeq 0
\end{aligned}$$

其 Lagrangian 函数为:
$$L(\boldsymbol{x}, \boldsymbol{\nu}, \boldsymbol{\lambda}) = \boldsymbol{c}^\top \boldsymbol{x} + \boldsymbol{\nu}^\top (A\boldsymbol{x} - \boldsymbol{b}) - \boldsymbol{\lambda}^\top \boldsymbol{x}=-\boldsymbol{b}^\top \boldsymbol{\nu}+(\boldsymbol{c}+A^\top \boldsymbol{\nu} - \boldsymbol{\lambda})^\top \boldsymbol{x}$$

考虑其对偶函数:
$$g(\boldsymbol{\nu}, \boldsymbol{\lambda}) = \inf_{\boldsymbol{x} \succeq 0} L(\boldsymbol{x}, \boldsymbol{\nu}, \boldsymbol{\lambda}) = \inf_{\boldsymbol{x} \succeq 0} \left(-\boldsymbol{b}^\top \boldsymbol{\nu}+(\boldsymbol{c}+A^\top \boldsymbol{\nu} - \boldsymbol{\lambda})^\top \boldsymbol{x} \right)$$

由于 $L(\boldsymbol{x}, \boldsymbol{\nu}, \boldsymbol{\lambda})$ 是关于 $\boldsymbol{x}$ 的 affine 函数, 当且仅当 $\partial L / \partial \boldsymbol{x} = \boldsymbol{c}^\top + \boldsymbol{\nu}^\top A - \boldsymbol{\lambda}^\top = 0$ 时, 对偶函数 $g(\boldsymbol{\nu}, \boldsymbol{\lambda}) = - \boldsymbol{\nu}^\top \boldsymbol{b}$. 否则, 对偶函数为 $-\infty$. 

因此当且仅当 $\boldsymbol{c} + A^\top \boldsymbol{\nu} - \boldsymbol{\lambda} = 0$ 且 $\boldsymbol{\lambda} \succeq 0$ 时, $\boldsymbol{b}^\top \boldsymbol{\nu}$ 是上述优化问题的非平凡下界.



#### Two-way Partitioning Problem

考虑如下二分问题(非凸):
$$\begin{aligned}
\min_{\boldsymbol{x}} \quad & \boldsymbol{x}^\top W \boldsymbol{x} \\
\text{s.t.} \quad & x_i^2 = 1, \quad i = 1, \ldots, n \\
\end{aligned}$$
