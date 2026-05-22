# Primal-Dual Methods

>[!quote]
>
> - Lecture Reference: <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>
> - Readings: Boyd & Vandenberghe, Convex Optimization, Chapter 11.7

## Introduction

Primal-Dual 方法是一个和 barrier method 类似的内点方法. 不过相对而言, 其更适合高精度的情况, 因为其能够达到超线性的收敛速度. 另外, 期可以有效处理可行但不严格可行的情况.

同样考虑如下的优化问题
$$
\begin{align*}
\min_{\mathbf{x}} \quad & f_0(\mathbf{x}) \\
\text{s.t.} \quad& f_i(\mathbf{x}) \leq 0, i = 1, \ldots, m \\
& \mathbf{A} \mathbf{x} = \mathbf{b}
\end{align*}
$$
其中 $f_0, f_1, \ldots, f_m$ 都是二阶连续可微的凸函数. $\mathbf{A}$ 是一个 $p \times n$ 行满秩. Slater 条件成立.  此外记
$$
f(\mathbf{x}) = \begin{bmatrix}f_1(\mathbf{x}) \\ \vdots \\ f_m(\mathbf{x})
\end{bmatrix} \in \mathbb{R}^m, \quad Df(\mathbf{x}) = \begin{bmatrix}\nabla f_1(\mathbf{x})^T \\ \vdots \\ \nabla f_m(\mathbf{x})^T
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$
原问题的 KKT 条件为:
$$
\begin{align*}
& \nabla f_0(\mathbf{x}^*) + Df(\mathbf{x}^*)^T \boldsymbol{\lambda}^* + \mathbf{A}^T \boldsymbol{\nu}^* = \nabla f_0(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(\mathbf{x}^*) + \mathbf{A}^T \boldsymbol{\nu}^* = 0 \\
& f_i(\mathbf{x}^*) \leq 0, i = 1, \ldots, m \\
& \mathbf{A} \mathbf{x}^* = \mathbf{b} \\
& \lambda_i^* \geq 0, i = 1, \ldots, m \\
& \lambda_i^* f_i(\mathbf{x}^*) = 0, i = 1, \ldots, m
\end{align*}
$$

回顾对于 barrier method:
$$
\begin{align*}
&  \min_{\mathbf{x}} \quad f_0(\mathbf{x}) - \frac{1}{t} \sum_{i=1}^m \log(-f_i(\mathbf{x})) \\
& \text{s.t.} \quad \mathbf{A} \mathbf{x} = \mathbf{b}
\end{align*}
$$
其 KKT stationarity condition 为
$$
\nabla f_0(\mathbf{x}^*) + \sum_{i=1}^m \frac{1}{t} \frac{1}{-f_i(\mathbf{x}^*)} \nabla f_i(\mathbf{x}^*) + \mathbf{A}^T \boldsymbol{\nu}^* = 0
$$
故原问题和 barrier method 的 stationarity condition 之间的区别在于 $\lambda_i^* = \frac{1}{t} \frac{1}{-f_i(\mathbf{x}^*)}$, 也就是说 $\lambda_i^* f_i(\mathbf{x}^*) = -\frac{1}{t}$, 而非 $0$. 故, 称下面的方程为 Modified KKT condition:
$$
\begin{cases}
& \nabla f_0(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(\mathbf{x}^*) + \mathbf{A}^T \boldsymbol{\nu}^* = 0 \\
& -\lambda_i^* f_i(\mathbf{x}^*) = \frac{1}{t}, i = 1, \ldots, m \\
& \mathbf{A} \mathbf{x}^* = \mathbf{b} \\
\end{cases}
$$
对于 barrier method, 其工作重心在于给定一系列的 $t$ 来求解 modified KKT condition 以得到 $\mathbf{x}^*(t)$, 并且当求出 $\mathbf{x}^*(t)$ 后, 再通过 $\lambda_i^* = \frac{1}{t} \frac{1}{-f_i(\mathbf{x}^*(t))}$ 来恢复 $\lambda_i^*$. 然而, primal-dual method 则是直接通过 modified KKT condition, 在给定 $t$ 的情况下, 同时求解 $\mathbf{x}^*(t), \boldsymbol{\lambda}^*(t), \boldsymbol{\nu}^*(t)$.

## Primal-dual search direction

对于上面的 Modified KKT condition, 可以写成矩阵的形式, 并 accordingly 定义 residual:
$$
\mathbf{r}_t(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \begin{bmatrix}
\nabla f_0(\mathbf{x}) + Df(\mathbf{x})^T \boldsymbol{\lambda} + \mathbf{A}^T \boldsymbol{\nu} \\
-\text{diag}(\boldsymbol{\lambda}) f(\mathbf{x}) - \frac{1}{t} \mathbf{1} \\
\mathbf{A} \mathbf{x} - \mathbf{b}
\end{bmatrix} := \begin{bmatrix}
\mathbf{r}_{\text{dual}}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \\
\mathbf{r}_{\text{perturb}}(\mathbf{x}, \boldsymbol{\lambda}) \\
\mathbf{r}_{\text{primal}}(\mathbf{x})
\end{bmatrix}
$$
其中 $\text{diag}(\boldsymbol{\lambda})$ 是一个 $m \times m$ 的对角矩阵, 其对角线元素为 $\lambda_1, \ldots, \lambda_m$. 可以看出, 当 $\mathbf{r}_t(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = 0$ 时, 就满足了 modified KKT condition. 因此, primal-dual method 的核心就是通过 Newton's method 来求解 $\mathbf{r}_t(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = 0$.