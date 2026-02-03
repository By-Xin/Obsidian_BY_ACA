# Canonical Problem Forms

## Linear Programs (LPs)

> Ref: Convex Optimization by Boyd & Vandenberghe, Section 4.3

目标函数和约束函数都是 affine 时的优化问题称为线性规划 (Linear Program, LP), 其一般形式为:
$$
\begin{aligned}
& \text{minimize} && \mathbf{c}^\top \mathbf{x} + d \\
& \text{subject to} && G\mathbf{x} \preceq \mathbf{h} \\
& && A\mathbf{x} = \mathbf{b}
\end{aligned}
$$
- 其中 $\mathbf{x},\mathbf{c} \in \mathbb{R}^n$, $d \in \mathbb{R}$, $G \in \mathbb{R}^{m \times n}$, $\mathbf{h} \in \mathbb{R}^m$, $A \in \mathbb{R}^{p \times n}$, $\mathbf{b} \in \mathbb{R}^p$. 

***LP 标准形式与一般形式***. 

对于含等式约束的线性规划, 其标准形式为:
$$\begin{aligned}
& \text{minimize} && \mathbf{c}^\top \mathbf{x} \\
& \text{subject to} && A\mathbf{x} = \mathbf{b} \\
& && \mathbf{x} \succeq \mathbf{0}
\end{aligned}$$

对于不含等式约束的线性规划, 其标准形式为:
$$\begin{aligned}
& \text{minimize} && \mathbf{c}^\top \mathbf{x} \\
& \text{subject to} && A\mathbf{x} \preceq \mathbf{b}
\end{aligned}$$

我们可以通过引入松弛变量和变量替换将一般形式转化为标准形式.
- 对于不等式约束 $G\mathbf{x} \preceq \mathbf{h}$, 引入松弛变量 $\mathbf{s} \succeq \mathbf{0}$, 使得 $G\mathbf{x} + \mathbf{s} = \mathbf{h}$.
- 对于无约束变量 $x_i$, 可以将其表示为两个非负变量之差: $x_i = x_i^+ - x_i^-$, 其中 $x_i^+, x_i^- \succeq 0$.
- 目标函数中的常数项 $d$ 可以忽略, 因为它不会影响最优解.


***Example:* (Basis Pursuit)**. 给定高维稀疏的线性系统 $X\boldsymbol{\beta} = \mathbf{y}$, 其中 $X \in \mathbb{R}^{n \times p}$, 且 $n < p$, 希望找到最稀疏的解 $\boldsymbol{\beta}$. 则原始问题为:
$$\begin{aligned}
& \text{minimize} && \|\boldsymbol{\beta}\|_0 \\
& \text{subject to} && X\boldsymbol{\beta} = \mathbf{y}
\end{aligned}$$
其中 $\|\boldsymbol{\beta}\|_0$ 表示 $\boldsymbol{\beta}$ 中非零元素的个数. 可以通过将 $\ell_0$ 范数替换为 $\ell_1$ 范数来得到一个线性规划近似:
$$\begin{aligned}
& \text{minimize} && \|\boldsymbol{\beta}\|_1 \\
& \text{subject to} && X\boldsymbol{\beta} = \mathbf{y}
\end{aligned}$$
该问题之所以是线性规划, 是因为 $\|\boldsymbol{\beta}\|_1 = \sum_{i=1}^p | \beta_i |$ 可以通过引入辅助变量 $z_i \geq |\beta_i|$ 转化为线性目标函数:
$$\begin{aligned}
& \text{minimize} && \sum_{i=1}^p z_i \\
& \text{subject to} && z_i \geq \beta_i, \quad i=1,\ldots,p \\
& && z_i \geq -\beta_i, \quad i=1,\ldots,p \\
& && X\boldsymbol{\beta} = \mathbf{y}
\end{aligned}$$
- 前两个不等式确保 $z_i \geq |\beta_i|$, 而目标函数最小化 $\sum z_i$ 会推动 $z_i$ 接近 $|\beta_i|$.

***Example:* (Dantzig Selector)**. 对于上述高维稀疏问题, 当允许 $X\boldsymbol{\beta}\approx \mathbf{y}$ 时, 可以使用 Dantzig 选择器:
$$\begin{aligned}
& \text{minimize} && \|\boldsymbol{\beta}\|_1 \\
& \text{subject to} && \|X^\top (X\boldsymbol{\beta} - \mathbf{y})\|_\infty \leq \lambda
\end{aligned}$$
- 其中 $(X\boldsymbol{\beta} - \mathbf{y})$ 是残差 $\mathbf{r}$, 故 $\|X^\top \mathbf{r}\|_\infty$ 表示每个预测变量与残差的相关性, 该约束限制了这种相关性最大值不超过超参数 $\lambda$.

该问题同样可以转化为线性规划:
- 对于 $\|X^\top (X\boldsymbol{\beta} - \mathbf{y})\|_\infty \leq \lambda$, 该约束等价于松弛后逐分量 $-\lambda \mathbf{1} \preceq X^\top (X\boldsymbol{\beta} - \mathbf{y}) \preceq \lambda \mathbf{1}$.
- 对于 $\|\boldsymbol{\beta}\|_1$, 同样引入辅助变量 $z_i \geq |\beta_i|$ 并最小化 $\sum z_i$.
因此, Dantzig 选择器可以表示为线性规划:
$$\begin{aligned}   
& \text{minimize}_{\boldsymbol{\beta}, \mathbf{z}} && \boldsymbol{1}^\top \mathbf{z} \\
& \text{subject to} && \mathbf{z} \succeq \boldsymbol{\beta}, \\
& && \mathbf{z} \succeq -\boldsymbol{\beta}, \\
& && X^\top (X\boldsymbol{\beta} - \mathbf{y}) \preceq \lambda \boldsymbol{1}, \\
& && -X^\top (X\boldsymbol{\beta} - \mathbf{y}) \preceq \lambda \boldsymbol{1}
\end{aligned}$$

## Quadratic Programs (QPs)

> Ref: Convex Optimization by Boyd & Vandenberghe, Section 4.4


## Semidefinite Programs (SDPs)

> Ref: Convex Optimization by Boyd & Vandenberghe, Section 4.6.2