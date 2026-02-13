# Canonical Problem Forms

## Linear Programs (LPs)

> Ref: Convex Optimization by Boyd & Vandenberghe, Section 4.3

***Definition:* (Linear Program, LP)** 目标函数和约束函数都是 affine 时的优化问题称为线性规划 (Linear Program, LP), 其一般形式为:
$$
\begin{aligned}
& \min && \mathbf{c}^\top \mathbf{x} + d \\
& \text{  s.t.} && G\mathbf{x} \leq \mathbf{h} \\
& && A\mathbf{x} = \mathbf{b}
\end{aligned}
$$
- 其中 $\mathbf{x},\mathbf{c} \in \mathbb{R}^n$, $d \in \mathbb{R}$, $G \in \mathbb{R}^{m \times n}$, $\mathbf{h} \in \mathbb{R}^m$, $A \in \mathbb{R}^{p \times n}$, $\mathbf{b} \in \mathbb{R}^p$. 

**LP 标准形式与一般形式**

对于含等式约束的线性规划, 其标准形式为:
$$\begin{aligned}
& \min && \mathbf{c}^\top \mathbf{x} \\
& \text{  s.t.} && A\mathbf{x} = \mathbf{b} \\
& && \mathbf{x} \geq \mathbf{0}
\end{aligned}$$

对于不含等式约束的线性规划, 其标准形式为:
$$\begin{aligned}
& \min && \mathbf{c}^\top \mathbf{x} \\
& \text{  s.t.} && A\mathbf{x} \leq \mathbf{b}
\end{aligned}$$

我们可以通过引入松弛变量和变量替换将一般形式转化为标准形式.
- 对于不等式约束 $G\mathbf{x} \leq \mathbf{h}$, 引入松弛变量 $\mathbf{s} \geq \mathbf{0}$, 使得 $G\mathbf{x} + \mathbf{s} = \mathbf{h}$.
- 对于无约束变量 $x_i$, 可以将其表示为两个非负变量之差: $x_i = x_i^+ - x_i^-$, 其中 $x_i^+, x_i^- \geq 0$.
- 目标函数中的常数项 $d$ 可以忽略, 因为它不会影响最优解.

因此通过上述变换, 一般形式转化为标准形式的结果为:
$$\begin{aligned}
& \text{minimize} && \mathbf{c}^\top \mathbf{x}^+  - \mathbf{c}^\top \mathbf{x}^- \\
& \text{subject to} && G(\mathbf{x}^+ - \mathbf{x}^-) + \mathbf{s} = \mathbf{h} \\
& && A(\mathbf{x}^+ - \mathbf{x}^-) = \mathbf{b} \\
& && \mathbf{x}^+, \mathbf{x}^-, \mathbf{s} \succeq \mathbf{0}
\end{aligned}$$


***Example:* (Basis Pursuit)**. 给定高维稀疏的线性系统 $X\boldsymbol{\beta} = \mathbf{y}$, 其中 $X \in \mathbb{R}^{n \times p}$, 且 $n < p$, 希望找到最稀疏的解 $\boldsymbol{\beta}$. 则原始问题为:
$$\begin{aligned}
& \min && \|\boldsymbol{\beta}\|_0 \\
& \text{  s.t.} && X\boldsymbol{\beta} = \mathbf{y}
\end{aligned}$$
其中 $\|\boldsymbol{\beta}\|_0$ 表示 $\boldsymbol{\beta}$ 中非零元素的个数. 可以通过将 $\ell_0$ 范数替换为 $\ell_1$ 范数来得到一个线性规划近似:
$$\begin{aligned}
& \min && \|\boldsymbol{\beta}\|_1 \\
& \text{  s.t.} && X\boldsymbol{\beta} = \mathbf{y}
\end{aligned}$$
该问题之所以是线性规划, 是因为 $\|\boldsymbol{\beta}\|_1 = \sum_{i=1}^p | \beta_i |$ 可以通过引入辅助变量 $z_i \geq |\beta_i|$ 转化为线性目标函数:
$$\begin{aligned}
& \min && \sum_{i=1}^p z_i \\
& \text{  s.t.} && z_i \geq \beta_i, \quad i=1,\ldots,p \\
& && z_i \geq -\beta_i, \quad i=1,\ldots,p \\
& && X\boldsymbol{\beta} = \mathbf{y}
\end{aligned}$$
- 前两个不等式确保 $z_i \geq |\beta_i|$, 而目标函数最小化 $\sum z_i$ 会推动 $z_i$ 接近 $|\beta_i|$.

***Example:* (Dantzig Selector)**. 对于上述高维稀疏问题, 当允许 $X\boldsymbol{\beta}\approx \mathbf{y}$ 时, 可以使用 Dantzig 选择器:
$$\begin{aligned}
& \min && \|\boldsymbol{\beta}\|_1 \\
& \text{  s.t.} && \|X^\top (X\boldsymbol{\beta} - \mathbf{y})\|_\infty \leq \lambda
\end{aligned}$$
- 其中 $(X\boldsymbol{\beta} - \mathbf{y})$ 是残差 $\mathbf{r}$, 故 $\|X^\top \mathbf{r}\|_\infty$ 表示每个预测变量与残差的相关性, 该约束限制了这种相关性最大值不超过超参数 $\lambda$.

该问题同样可以转化为线性规划:
- 对于 $\|X^\top (X\boldsymbol{\beta} - \mathbf{y})\|_\infty \leq \lambda$, 该约束等价于松弛后逐分量 $-\lambda \mathbf{1} \leq X^\top (X\boldsymbol{\beta} - \mathbf{y}) \leq \lambda \mathbf{1}$.
- 对于 $\|\boldsymbol{\beta}\|_1$, 同样引入辅助变量 $z_i \geq |\beta_i|$ 并最小化 $\sum z_i$.
因此, Dantzig 选择器可以表示为线性规划:
$$\begin{aligned}   
& \min_{\boldsymbol{\beta}, \mathbf{z}} && \boldsymbol{1}^\top \mathbf{z} \\
& \text{  s.t.} && \mathbf{z} \geq \boldsymbol{\beta}, \\
& && \mathbf{z} \geq -\boldsymbol{\beta}, \\
& && X^\top (X\boldsymbol{\beta} - \mathbf{y}) \leq \lambda \boldsymbol{1}, \\
& && -X^\top (X\boldsymbol{\beta} - \mathbf{y}) \leq \lambda \boldsymbol{1}
\end{aligned}$$

## Quadratic Programs (QPs)

> Ref: Convex Optimization by Boyd & Vandenberghe, Section 4.4

***Definition:* (Convex Quadratic Program, CQP)** 凸二次规划（Convex Quadratic Program, CQP）的一般形式为:
$$\begin{aligned}
\min_x &\quad && \frac{1}{2} \mathbf{c}^\top \mathbf{x} + \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} \\
\text{s.t.} &\quad && D\mathbf{x} \leq \mathbf{d} \\
&\quad && A\mathbf{x} = \mathbf{b}
\end{aligned}$$
- 其中 $\mathbf{Q}\geq 0$ 是半正定矩阵, 以确保目标函数是凸的. 

同样其也可以转化为标准形式:
$$\begin{aligned}
\min_{\mathbf{x}} &\quad && \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} \\
\text{s.t.} &\quad && A\mathbf{x} = \mathbf{b} \\
&\quad && \mathbf{x} \geq 0
\end{aligned}$$

***Example:* (Portfolio Optimization)** 给定 $n$ 种资产, 其预期收益率为 $\mu \in \mathbb{R}^n$, 协方差矩阵为 $\Sigma \in \mathbb{R}^{n \times n}$, 投资组合权重为 $\mathbf{x} \in \mathbb{R}^n$. 投资组合优化问题可以表示为:
$$\begin{aligned}
& \max_{\mathbf{x}} && \mu^\top \mathbf{x} - \frac{\gamma}{2} \mathbf{x}^\top \Sigma \mathbf{x} \\
& \text{s.t.} && \mathbf{1}^\top \mathbf{x} = 1 \\
& && \mathbf{x} \geq 0
\end{aligned}$$
- 其中 $\gamma > 0$ 是风险厌恶系数 (risk-aversion coefficient). 

***Example:* (Support Vector Machine)** 支持向量机 (SVM) 的训练问题也可以表示为凸二次规划. 给定训练数据集 $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, 其中 $\mathbf{x}_i \in \mathbb{R}^d$ 是特征向量, $y_i \in \{-1, 1\}$ 是标签. SVM 的优化问题为:
$$\begin{aligned}
& \min_{\mathbf{w}, b, \xi} && \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \\
& \text{s.t.} && y_i (\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad i = 1, \ldots, n \\
& && \xi_i \geq 0, \quad i = 1, \ldots, n
\end{aligned}$$
- 其中 $\mathbf{w}$ 是权重向量, $b$ 是偏置, $\xi_i$ 是松弛变量, $C > 0$ 是惩罚参数.

***Example:* (LASSO)** LASSO (Least Absolute Shrinkage and Selection Operator) 回归问题也可以转化为凸二次规划. 给定数据矩阵 $X \in \mathbb{R}^{m \times n}$ 和响应向量 $\mathbf{y} \in \mathbb{R}^m$, LASSO 的优化问题为:
$$\begin{aligned}
& \min_{\boldsymbol{\beta}} && \frac{1}{2} \|X \boldsymbol{\beta} - \mathbf{y}\|^2_2 \\
& \text{s.t.} && \|\boldsymbol{\beta}\|_1 \leq s
\end{aligned}$$
- 其中 $s > 0$ 是一个超参数, 控制稀疏性水平.
  




## Semidefinite Programs (SDPs)

> Ref: Convex Optimization by Boyd & Vandenberghe, Section 4.6.2

**Motivation**

回顾在 LP 问题中, 我们考虑一般形式:
$$\begin{aligned}
& \min && \mathbf{c}^\top \mathbf{x} \\
& \text{  s.t.} && G\mathbf{x} \leq \mathbf{h} \\
& && A\mathbf{x} = \mathbf{b}
\end{aligned}$$
- 其中不等式约束 $G\mathbf{x} \leq \mathbf{h}$ 是逐分量的. 

如果我们希望将不等式约束推广, 从元素级别的约束扩展到矩阵级别的约束, 则可以引入半正定锥 (positive semidefinite cone) 的概念.


> ***Recap* (Facts about Positive Semidefinite Matrices)**
>
> 在进行半正定规划之前, 我们先回顾一些关于半正定矩阵的基本事实. 
> 
> - 记 $\mathbb{S}^n$ 为所有 $n \times n$ 实对称矩阵的集合(*实对称矩阵的特征值均为实数*). 
> - 记 $\mathbb{S}^n_+ = \{X \in \mathbb{S}^n |  \mathbf{v}^\top X \mathbf{v} \geq 0, \forall \mathbf{v} \in \mathbb{R}^n\}$ 为所有 $n \times n$ 实对称半正定矩阵的集合, 其所有特征值均非负. 
> - 记 $\mathbb{S}^n_{++} = \{X \in \mathbb{S}^n |  \mathbf{v}^\top X \mathbf{v} > 0, \forall \mathbf{v} \in \mathbb{R}^n\backslash \{0\}\}$ 为所有 $n \times n$ 实对称正定矩阵的集合, 其所有特征值均为正. 
>
> 可以定义在 $\mathbb{S}^n$ 上的内积. 对于任意 $X, Y \in \mathbb{S}^n$, 定义:
> $$X\circ Y = \text{tr}(XY) = \sum_{i,j} X_{ij} Y_{ij}$$
>
> 还可以定义矩阵的不等式. 对于 $X, Y \in \mathbb{S}^n$, 记 $X \succeq Y$ 当且仅当 $X - Y \in \mathbb{S}^n_+$; 类似地, 记 $X \succ Y$ 当且仅当 $X - Y \in \mathbb{S}^n_{++}$.

***Definition:* (Semidefinite Program)** 半正定规划 (Semidefinite Program, SDP) 的一般形式为:
$$\begin{aligned}
& \min_{\mathbf{x}} && \mathbf{c}^\top \mathbf{x} \\
& \text{  s.t.} && x_1 F_1 + x_2 F_2 + \ldots + x_n F_n + F_0 \preceq 0 \\
& && A\mathbf{x} = \mathbf{b}
\end{aligned}$$
- 其中 $F_0, F_1, \ldots, F_n \in \mathbb{S}^d$, $A \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$, $\mathbf{c} \in \mathbb{R}^n$.
- 当 $F_0, F_1, \ldots, F_n$ 为对角矩阵时, SDP 退化为线性规划 (LP).

其标准形式为:
$$\begin{aligned}
& \min_{X} && C \circ X \\
& \text{  s.t.} && A_i \circ X = b_i, \quad i = 1, \ldots, m \\
& && X \succeq 0
\end{aligned}$$
- 其中 $X \in \mathbb{S}^n$, $C, A_i \in \mathbb{S}^n$, $b_i \in \mathbb{R}$.


***Example:* (Trace Norm Minimization)** 
- 对于未知量 $\mathbf{x} \in \mathbb{R}^n$, 可以观测到线性测量 $\mathbf{y} = A\mathbf{x}$, 其中 $A \in \mathbb{R}^{m \times n}, m \ll n$ 是已知的测量矩阵, 此时方程有无穷多解. 额外引入稀疏的先验假设, 故优化目标位 $\min \|\mathbf{x}\|_0, \text{ s.t. } A\mathbf{x} = \mathbf{y}$. 该问题是 NP-hard 的, 因此可以使用凸的 $\ell_1$ 范数作为松弛. 
- 进一步扩展到矩阵形式, 对于矩阵未知量 $X \in \mathbb{R}^{m \times n}$, 观测到线性测量 $\mathcal{A}(X) = \mathbf{y}$, 其中 $\mathcal{A}: \mathbb{R}^{m \times n} \to \mathbb{R}^p$ 是线性映射, $p \ll mn$, 其第 $i$ 个分量为 $\mathcal{A}_i(X) = \text{tr}(A_i^\top X), i=1,\ldots,p$.  故优化目标为 $\min \text{rank}(X), \text{ s.t. } \mathcal{A}(X) = \mathbf{y}$. 类比 $\ell_0$ 与 $\ell_1$ 的关系, 可以使用矩阵的迹范数 (trace norm, 矩阵奇异值之和) 作为矩阵秩 (矩阵非零奇异值个数) 的凸松弛, 因此得到的优化问题为:
$$\begin{aligned}
& \min_{X} && \|X\|_* \\
& \text{  s.t.} && \mathcal{A}(X) = \mathbf{y}
\end{aligned}$$
  - 其中 $\|X\|_* = \sum_i \sigma_i(X)$, $\sigma_i(X)$ 是 $X$ 的奇异值. 该问题可通过对偶形式转化为 SDP.

## Conic Programs

> Ref: Convex Optimization by Boyd & Vandenberghe, Section 4.6.1

锥规划 (Conic Program) 是一种更一般的凸优化问题形式, 其目标函数为线性函数, 约束条件为变量属于某个凸锥.

***Definition:* (Conic Program)** 锥规划的一般形式为:
$$\begin{aligned}
& \min_{\mathbf{x}} && \mathbf{c}^\top \mathbf{x} \\
& \text{  s.t.} && A\mathbf{x} = \mathbf{b} \\
& && \mathcal{D}(\mathbf{x}) + d \in K
\end{aligned}$$
- 其中 $\mathbf{x}, \mathbf{c} \in \mathbb{R}^n$, $A \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$. 
- $\mathcal{D}: \mathbb{R}^n \to \mathcal{Y}$ 是线性映射, 其中 $\mathcal{Y}$ 是某个有限维 Euclidean 空间, 例如 $\mathbb{R}^k$ 中 $\mathcal{D}(\mathbf{x}) = F\mathbf{x}$ ($F \in \mathbb{R}^{k \times n}$) 或 $\mathbb{S}^k$ 中 $\mathcal{D}(\mathbf{x}) = \sum_{i=1}^n x_i F_i$ ($F_i \in \mathbb{S}^k$).
- $K$ 是 $\mathcal{Y}$ 中的一个闭凸锥 (closed convex cone), 其满足:
  - Cone: $\forall x \in K,  \alpha \geq 0, \alpha x \in K$.
  - Convex: $\forall x_1, x_2 \in K, \theta \in [0,1], \theta x_1 + (1-\theta)x_2 \in K$.
  - Closed: $K$ 包含其边界点.

在 Boyd 的书中, 考虑 $\mathcal{Y} = \mathbb{R}^p$, 此时的 conic program 为:
$$\begin{aligned}
& \min_{\mathbf{x}} && \mathbf{c}^\top \mathbf{x} \\
& \text{  s.t.} && A\mathbf{x} = \mathbf{b} \\
& && F\mathbf{x} + \mathbf{g} \preceq_K 0
\end{aligned}$$
- 其中 $F \in \mathbb{R}^{p \times n}$, $\mathbf{g} \in \mathbb{R}^p$
- $K \subseteq \mathbb{R}^p$ 是一个闭凸锥, 且不等式 $\preceq_K$ 定义为: $\mathbf{y} \preceq_K \mathbf{z}$ 当且仅当 $\mathbf{z} - \mathbf{y} \in K$.