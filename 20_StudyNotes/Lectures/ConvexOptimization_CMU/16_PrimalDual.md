# Primal-Dual Methods

>[!quote]
>
> - Lecture Reference: <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>
> - Readings: 
>   - Boyd & Vandenberghe, Convex Optimization, Chapter 11.7
>   - 刘浩洋, 最优化: 建模、算法与理论, Chapter 7.3.  

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
\end{bmatrix} \in \mathbb{R}^m, \quad Df(\mathbf{x}) = \begin{bmatrix}\nabla f_1(\mathbf{x})^\top \\ \vdots \\ \nabla f_m(\mathbf{x})^\top
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$
原问题的 KKT 条件为:
$$
\begin{align*}
& \nabla f_0(\mathbf{x}^*) + Df(\mathbf{x}^*)^\top \boldsymbol{\lambda}^* + \mathbf{A}^\top \boldsymbol{\nu}^* = \nabla f_0(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(\mathbf{x}^*) + \mathbf{A}^\top \boldsymbol{\nu}^* = 0 \\
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
\nabla f_0(\mathbf{x}^*) + \sum_{i=1}^m \frac{1}{t} \frac{1}{-f_i(\mathbf{x}^*)} \nabla f_i(\mathbf{x}^*) + \mathbf{A}^\top \boldsymbol{\nu}^* = 0
$$
故原问题和 barrier method 的 stationarity condition 之间的区别在于 $\lambda_i^* = \frac{1}{t} \frac{1}{-f_i(\mathbf{x}^*)}$, 也就是说 $\lambda_i^* f_i(\mathbf{x}^*) = -\frac{1}{t}$, 而非 $0$. 故, 称下面的方程为 Modified KKT condition:
$$
\begin{cases}
& \nabla f_0(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(\mathbf{x}^*) + \mathbf{A}^\top \boldsymbol{\nu}^* = 0 \\
& -\lambda_i^* f_i(\mathbf{x}^*) = \frac{1}{t}, i = 1, \ldots, m \\
& \mathbf{A} \mathbf{x}^* = \mathbf{b} \\
\end{cases}
$$
对于 barrier method, 其工作重心在于给定一系列的 $t$ 来求解 modified KKT condition 以得到 $\mathbf{x}^*(t)$, 并且当求出 $\mathbf{x}^*(t)$ 后, 再通过 $\lambda_i^* = \frac{1}{t} \frac{1}{-f_i(\mathbf{x}^*(t))}$ 来恢复 $\lambda_i^*$. 然而, primal-dual method 则是直接通过 modified KKT condition, 在给定 $t$ 的情况下, 同时求解 $\mathbf{x}^*(t), \boldsymbol{\lambda}^*(t), \boldsymbol{\nu}^*(t)$.

## Primal-dual search direction

对于上面的 Modified KKT condition, 可以写成矩阵的形式, 并 accordingly 定义 residual:
$$
\mathbf{r}_t(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \begin{bmatrix}
\nabla f_0(\mathbf{x}) + Df(\mathbf{x})^\top \boldsymbol{\lambda} + \mathbf{A}^\top \boldsymbol{\nu} \\
-\text{diag}(\boldsymbol{\lambda}) f(\mathbf{x}) - \frac{1}{t} \mathbf{1} \\
\mathbf{A} \mathbf{x} - \mathbf{b}
\end{bmatrix} := \begin{bmatrix}
\mathbf{r}_{\text{dual}}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \\
\mathbf{r}_{\text{cent}}(\mathbf{x}, \boldsymbol{\lambda}) \\
\mathbf{r}_{\text{primal}}(\mathbf{x})
\end{bmatrix}
$$
其中 $\text{diag}(\boldsymbol{\lambda})$ 是一个 $m \times m$ 的对角矩阵, 其对角线元素为 $\lambda_1, \ldots, \lambda_m$. 这三个 residual 分别衡量:
- $\mathbf{r}_{\text{primal}}(\mathbf{x})$ 衡量 primal feasibility. 即 equality constraint 的满足程度, 当 $\mathbf{r}_{\text{primal}}(\mathbf{x}) = 0$ 时, 就满足了 primal feasibility.
- $\mathbf{r}_{\text{dual}}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ 衡量 stationarity condition 的满足程度, 当 $\mathbf{r}_{\text{dual}}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = 0$ 时, 原问题的 Lagrangian 关于 $\mathbf{x}$ 的梯度为 $0$, 即满足了 stationarity condition.
- $\mathbf{r}_{\text{cent}}(\mathbf{x}, \boldsymbol{\lambda})$ 衡量 centrality condition 的满足程度, 当 $\mathbf{r}_{\text{cent}}(\mathbf{x}, \boldsymbol{\lambda}) = 0$ 时, 对应第 $i$ 个分量为 $r_t^{(i)} = -\lambda_i f_i(\mathbf{x}) - \frac{1}{t} = 0$, 即 $\lambda_i f_i(\mathbf{x}) = -\frac{1}{t}$, 也就是 modified KKT condition 中的 centrality condition.

可以看出, 当 $\mathbf{r}_t(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = 0$ 时, 就满足了 modified KKT condition. 因此, primal-dual method 的核心就是通过 Newton's method 来求解 $\mathbf{r}_t(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = 0$.

因此用牛顿法的思路来尝试求解. 记 $\mathbf{y} = \begin{bmatrix}\mathbf{x} \\ \boldsymbol{\lambda} \\ \boldsymbol{\nu}\end{bmatrix}$, 对应 $\Delta \mathbf{y} = \begin{bmatrix}\Delta \mathbf{x} \\ \Delta \boldsymbol{\lambda} \\ \Delta \boldsymbol{\nu}\end{bmatrix}$, 考虑对当前点 $\mathbf{y}$ 进行一阶近似:
$$
\mathbf{r}_t(\mathbf{y} + \Delta \mathbf{y}) \approx \mathbf{r}_t(\mathbf{y}) + D\mathbf{r}_t(\mathbf{y}) \Delta \mathbf{y}
$$

因此, 为了尽量让下一步 $\mathbf{r}_t(\mathbf{y} + \Delta \mathbf{y})$ 尽量接近于 $0$, 就需要让 $\mathbf{r}_t(\mathbf{y}) + D\mathbf{r}_t(\mathbf{y}) \Delta \mathbf{y} = 0$, 从而得到 (我们这里会求解线性系统而非直接求逆的):
$$D\mathbf{r}_t(\mathbf{y})\Delta \mathbf{y} = -\mathbf{r}_t(\mathbf{y})$$
下面来计算 $D\mathbf{r}_t(\mathbf{y})$:
$$
\begin{aligned}
    & \nabla_{\mathbf{x}} \mathbf{r}_{\text{dual}}(\mathbf{y}) = \nabla^2 f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i \nabla^2 f_i(\mathbf{x}) ,\quad 
    && \nabla_{\boldsymbol{\lambda}} \mathbf{r}_{\text{dual}}(\mathbf{y})  = Df(\mathbf{x})^\top ,\quad
    &&& \nabla_{\boldsymbol{\nu}} \mathbf{r}_{\text{dual}}(\mathbf{y}) = \mathbf{A}^\top \\
    &\nabla_{\mathbf{x}} \mathbf{r}_{\text{cent}}(\mathbf{y})  = -\text{diag}(\boldsymbol{\lambda}) Df(\mathbf{x}) , \quad
    &&\nabla_{\boldsymbol{\lambda}} \mathbf{r}_{\text{cent}}(\mathbf{y})  = -\text{diag}(f(\mathbf{x})) , \quad
    &&&\nabla_{\boldsymbol{\nu}} \mathbf{r}_{\text{cent}}(\mathbf{y})  = \mathbf{0} \\
    &\nabla_{\mathbf{x}} \mathbf{r}_{\text{primal}}(\mathbf{y})  = \mathbf{A} , \quad
    &&\nabla_{\boldsymbol{\lambda}} \mathbf{r}_{\text{primal}}(\mathbf{y})  = \mathbf{0} , \quad
    &&&\nabla_{\boldsymbol{\nu}} \mathbf{r}_{\text{primal}}(\mathbf{y})  = \mathbf{0}
\end{aligned}
$$
其中:
- $\operatorname{diag}(f(\mathbf{x})) = \operatorname{diag}(f_1(\mathbf{x}), \ldots, f_m(\mathbf{x}))$ 
- $\operatorname{diag}(\boldsymbol{\lambda}) Df(\mathbf{x}) = \begin{bmatrix}\lambda_1 \nabla f_1(\mathbf{x}) & \ldots & \lambda_m \nabla f_m(\mathbf{x})\end{bmatrix}^\top$.


故整理后有:
$$
\begin{bmatrix}
\nabla^2 f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i \nabla^2 f_i(\mathbf{x}) & Df(\mathbf{x})^\top & \mathbf{A}^\top \\
-\text{diag}(\boldsymbol{\lambda}) Df(\mathbf{x}) & -\text{diag}(f(\mathbf{x})) & \mathbf{0} \\
\mathbf{A} & \mathbf{0} & \mathbf{0}
\end{bmatrix} \begin{bmatrix}\Delta \mathbf{x} \\ \Delta \boldsymbol{\lambda} \\ \Delta \boldsymbol{\nu}\end{bmatrix} = -\begin{bmatrix}\mathbf{r}_{\text{dual}}(\mathbf{y}) \\ \mathbf{r}_{\text{cent}}(\mathbf{y}) \\ \mathbf{r}_{\text{primal}}(\mathbf{y})\end{bmatrix}
$$

### Primal-dual 与 barrier method 的比较

为说明 primal-dual method 和 barrier method 的关联, 对于上述的方程组进行进一步整理. 由第二行可以得到:
$$
-\text{diag}(\boldsymbol{\lambda}) Df(\mathbf{x}) \Delta \mathbf{x} - \text{diag}(f(\mathbf{x})) \Delta \boldsymbol{\lambda} = -\mathbf{r}_{\text{cent}}(\mathbf{y}) 
$$
$$
\implies \Delta \boldsymbol{\lambda} = -\text{diag}(f(\mathbf{x}))^{-1} \text{diag}(\boldsymbol{\lambda}) Df(\mathbf{x}) \Delta \mathbf{x} + \text{diag}(f(\mathbf{x}))^{-1} \mathbf{r}_{\text{cent}}(\mathbf{y})
$$

接着代入到第一行中, 整理有:
$$
\begin{aligned}
& \underbrace{\left(\nabla^2 f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i \nabla^2 f_i(\mathbf{x}) - Df(\mathbf{x})^\top \text{diag}(f(\mathbf{x}))^{-1} \text{diag}(\boldsymbol{\lambda}) Df(\mathbf{x})\right) }_{\mathbf{H}_{\text{pd}}}\Delta \mathbf{x} + \mathbf{A}^\top \Delta \boldsymbol{\nu} \\
& = -\mathbf{r}_{\text{dual}}(\mathbf{y}) - Df(\mathbf{x})^\top \text{diag}(f(\mathbf{x}))^{-1} \mathbf{r}_{\text{cent}}(\mathbf{y})
\end{aligned}
$$
记 
$$
\begin{aligned}
\mathbf{H}_{\text{pd}} &= \nabla^2 f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i \nabla^2 f_i(\mathbf{x}) - Df(\mathbf{x})^\top \text{diag}(f(\mathbf{x}))^{-1} \text{diag}(\boldsymbol{\lambda}) Df(\mathbf{x}) \\
& = \nabla^2 f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i \nabla^2 f_i(\mathbf{x}) + \sum_{i=1}^m \frac{\lambda_i}{-f_i(\mathbf{x})} \nabla f_i(\mathbf{x}) \nabla f_i(\mathbf{x})^\top
\end{aligned}
$$ 
并且注意到 RHS 中, 代入 $\mathbf{r}_{\text{cent}}(\mathbf{y}) = -\text{diag}(\boldsymbol{\lambda}) f(\mathbf{x}) - \frac{1}{t} \mathbf{1}$,  以及 $\mathbf{r}_{\text{dual}}(\mathbf{y}) = \nabla f_0(\mathbf{x}) + Df(\mathbf{x})^\top \boldsymbol{\lambda} + \mathbf{A}^\top \boldsymbol{\nu}$, 可以得到:
$$
\begin{aligned}
& -\mathbf{r}_{\text{dual}}(\mathbf{y}) - Df(\mathbf{x})^\top \text{diag}(f(\mathbf{x}))^{-1} \mathbf{r}_{\text{cent}}(\mathbf{y}) \\
& = -\nabla f_0(\mathbf{x}) - Df(\mathbf{x})^\top \boldsymbol{\lambda} - \mathbf{A}^\top \boldsymbol{\nu} + Df(\mathbf{x})^\top \text{diag}(f(\mathbf{x}))^{-1} \text{diag}(\boldsymbol{\lambda}) f(\mathbf{x}) + \frac{1}{t} Df(\mathbf{x})^\top \text{diag}(f(\mathbf{x}))^{-1} \mathbf{1} \\
& = -\nabla f_0(\mathbf{x}) - Df(\mathbf{x})^\top \boldsymbol{\lambda} - \mathbf{A}^\top \boldsymbol{\nu} + Df(\mathbf{x})^\top \boldsymbol{\lambda} - \frac{1}{t} \sum_{i=1}^m \frac{1}{-f_i(\mathbf{x})} \nabla f_i(\mathbf{x}) \\
&= -\left(\nabla f_0(\mathbf{x}) + \frac{1}{t} \sum_{i=1}^m \frac{1}{-f_i(\mathbf{x})} \nabla f_i(\mathbf{x}) + \mathbf{A}^\top \boldsymbol{\nu}\right)
\end{aligned}
$$

因此, primal-dual method 中的第1, 3行方程组 (并对应 $\Delta \mathbf{x}$ 和 $\Delta \boldsymbol{\nu}$两个变量) 可以写成如下精简的形式:
$$
\begin{bmatrix}
\mathbf{H}_{\text{pd}} & \mathbf{A}^\top \\
\mathbf{A} & \mathbf{0}
\end{bmatrix} \begin{bmatrix}\Delta \mathbf{x} \\ \Delta \boldsymbol{\nu}\end{bmatrix} = -\begin{bmatrix}\mathbf{r}_{\text{dual}}(\mathbf{y}) + Df(\mathbf{x})^\top \text{diag}(f(\mathbf{x}))^{-1} \mathbf{r}_{\text{cent}}(\mathbf{y}) \\ \mathbf{r}_{\text{primal}}(\mathbf{y})\end{bmatrix}
$$
代入上面两侧的表达式, 可以得到:
$$
\begin{bmatrix}
\nabla^2 f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i \nabla^2 f_i(\mathbf{x}) + \sum_{i=1}^m \frac{\lambda_i}{-f_i(\mathbf{x})} \nabla f_i(\mathbf{x}) \nabla f_i(\mathbf{x})^\top & \mathbf{A}^\top \\
\mathbf{A} & \mathbf{0}
\end{bmatrix} \begin{bmatrix}\Delta \mathbf{x} \\ \Delta \boldsymbol{\nu}\end{bmatrix} = -\begin{bmatrix}\nabla f_0(\mathbf{x}) + \frac{1}{t} \sum_{i=1}^m \frac{1}{-f_i(\mathbf{x})} \nabla f_i(\mathbf{x}) + \mathbf{A}^\top \boldsymbol{\nu} \\ \mathbf{A} \mathbf{x} - \mathbf{b}\end{bmatrix}
$$

下面我们对照 barrier method 的 Newton step 来看一下两者的区别. 对于 barrier method, 其目标函数为:
$$
\psi(\mathbf{x}) := f_0(\mathbf{x}) - \frac{1}{t} \sum_{i=1}^m \log(-f_i(\mathbf{x})) \implies \nabla \psi = \nabla f_0(\mathbf{x}) + \frac{1}{t} \sum_{i=1}^m \frac{1}{-f_i(\mathbf{x})} \nabla f_i(\mathbf{x}).
$$
因此, barrier method 的 Newton step 可以写成如下的方程组:
$$
\begin{bmatrix}\nabla^2 \psi(\mathbf{x}) & \mathbf{A}^\top \\ \mathbf{A} & \mathbf{0}\end{bmatrix} \begin{bmatrix}\Delta \mathbf{x}^{\text{barrier}} \\ \Delta \boldsymbol{\nu}^{\text{barrier}}\end{bmatrix} = -\begin{bmatrix}\nabla \psi(\mathbf{x}) + \mathbf{A}^\top \boldsymbol{\nu} \\ \mathbf{A} \mathbf{x} - \mathbf{b}\end{bmatrix}
$$
代入具体表达式
$$
\begin{bmatrix}
\nabla^2 f_0(\mathbf{x}) + \frac{1}{t} \sum_{i=1}^m \frac{1}{-f_i(\mathbf{x})} \nabla^2 f_i(\mathbf{x}) + \frac{1}{t} \sum_{i=1}^m \frac{1}{(-f_i(\mathbf{x}))^2} \nabla f_i(\mathbf{x}) \nabla f_i(\mathbf{x})^\top & \mathbf{A}^\top \\
\mathbf{A} & \mathbf{0} 
\end{bmatrix} \begin{bmatrix}\Delta \mathbf{x}^{\text{barrier}} \\ \Delta \boldsymbol{\nu}^{\text{barrier}}\end{bmatrix} = -\begin{bmatrix}\nabla f_0(\mathbf{x}) + \frac{1}{t} \sum_{i=1}^m \frac{1}{-f_i(\mathbf{x})} \nabla f_i(\mathbf{x}) + \mathbf{A}^\top \boldsymbol{\nu} \\ \mathbf{A} \mathbf{x} - \mathbf{b}\end{bmatrix}
$$

因此, 通过对比不难看出, 若令 $\lambda_i = \frac{1}{t} \frac{1}{-f_i(\mathbf{x})}$, 则 $\mathbf{H}_{\text{pd}} = \nabla^2 \psi(\mathbf{x})$. 并且二者的 search direction 的 RHS 也是一样的. 因此, 可以看出, barrier method 的 Newton step 是 primal-dual method 的一个 special case, 其对应 $\lambda_i = \frac{1}{t} \frac{1}{-f_i(\mathbf{x})}$.


## The surrogate duality gap & Primal-dual interior-point method

### Surrogate duality gap

上一个小节展示了在给定 $t$ 的情况下, primal-dual 方法通过 Newton method 来求解 $\mathbf{r}_t(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = 0$ 来得到 $\mathbf{x}^*(t), \boldsymbol{\lambda}^*(t), \boldsymbol{\nu}^*(t)$.  这一小节将讨论 $t$ 的选择, 算法的终止策略等内容. 

首先引入 surrogate gap 的概念. 对于原问题而言, 其 Lagrangian 为:
$$
L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i f_i(\mathbf{x}) + \boldsymbol{\nu}^\top (\mathbf{A} \mathbf{x} - \mathbf{b})
$$
对于一个 primal feasible 的 $\mathbf{\tilde{x}}$, 在 $\boldsymbol{\lambda}\geq 0$ 恒有
$$
\begin{aligned}
L(\mathbf{\tilde{x}}, \boldsymbol{\lambda}, \boldsymbol{\nu}) & = f_0(\mathbf{\tilde{x}}) + \sum_{i=1}^m \lambda_i f_i(\mathbf{\tilde{x}}) + \boldsymbol{\nu}^\top (\mathbf{A} \mathbf{\tilde{x}} - \mathbf{b}) \\
& \leq f_0(\mathbf{\tilde{x}}) + \boldsymbol{\nu}^\top (\mathbf{A} \mathbf{\tilde{x}} - \mathbf{b}) = f_0(\mathbf{\tilde{x}})
\end{aligned}
$$
故其 dual function $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ 满足 $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq L(\mathbf{\tilde{x}}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \leq f_0(\mathbf{\tilde{x}})$. 因此, 真正的 duality gap 可以定义为 $f_0(\mathbf{\tilde{x}}) - g(\boldsymbol{\lambda}, \boldsymbol{\nu})$. 进一步, 在 primal feasible 的基础上, 还有其 dual residual 为 $0$, 即:
$$
\mathbf{r}_{\text{dual}}(\mathbf{\tilde{x}}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \nabla f_0(\mathbf{\tilde{x}}) + Df(\mathbf{\tilde{x}})^\top \boldsymbol{\lambda} + \mathbf{A}^\top \boldsymbol{\nu} = 0
$$
这相当于说, $\mathbf{\tilde{x}}$ 是 dual function $g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ 的一个 minimizer. 则此时, duality gap 就变为:
$$
\begin{aligned}
f_0(\mathbf{\tilde{x}}) - g(\boldsymbol{\lambda}, \boldsymbol{\nu}) 
&= f_0(\mathbf{\tilde{x}}) - [f_0(\mathbf{\tilde{x}}) + \sum_{i=1}^m \lambda_i f_i(\mathbf{\tilde{x}}) + \boldsymbol{\nu}^\top (\mathbf{A} \mathbf{\tilde{x}} - \mathbf{b})] \\
&= -\sum_{i=1}^m \lambda_i f_i(\mathbf{\tilde{x}}) = - \boldsymbol{\lambda}^\top f(\mathbf{\tilde{x}})
\end{aligned}
$$

然而在 primal-dual 方法中, 我们需要求解如下方程:
$$
\begin{bmatrix}
\nabla^2 f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i \nabla^2 f_i(\mathbf{x}) & Df(\mathbf{x})^\top & \mathbf{A}^\top \\
-\text{diag}(\boldsymbol{\lambda}) Df(\mathbf{x}) & -\text{diag}(f(\mathbf{x})) & \mathbf{0} \\
\mathbf{A} & \mathbf{0} & \mathbf{0}
\end{bmatrix} \begin{bmatrix}\Delta \mathbf{x} \\ \Delta \boldsymbol{\lambda} \\ \Delta \boldsymbol{\nu}\end{bmatrix} = -\begin{bmatrix}\mathbf{r}_{\text{dual}}(\mathbf{y}) \\ \mathbf{r}_{\text{cent}}(\mathbf{y}) \\ \mathbf{r}_{\text{primal}}(\mathbf{y})\end{bmatrix}
$$
在第 $k$ 次迭代点 $\mathbf{y}^{(k)} = (\mathbf{x}^{(k)}, \boldsymbol{\lambda}^{(k)}, \boldsymbol{\nu}^{(k)})$ 上, 通常有 $\mathbf{r}_{t, \text{primal}}(\mathbf{y}^{(k)}) = \mathbf{A} \mathbf{x}^{(k)} - \mathbf{b} \neq 0$, $\mathbf{r}_{t, \text{dual}}(\mathbf{y}^{(k)}) = \nabla f_0(\mathbf{x}^{(k)}) + Df(\mathbf{x}^{(k)})^\top \boldsymbol{\lambda}^{(k)} + \mathbf{A}^\top \boldsymbol{\nu}^{(k)} \neq 0$. 即其 primal 和 dual 不一定是 feasible 的. 因此, 只能将最后的结果定义为 surrogate duality gap, 
$$
{\hat\eta^{(k)}} = -\boldsymbol{\lambda}^{(k)\top} f(\mathbf{x}^{(k)})
$$

另一方面, $\hat{\eta}^{(k)}$ 也可以看成是对于互补松弛条件的一个度量. 因为 $\hat{\eta}^{(k)} = -\boldsymbol{\lambda}^{(k)\top} f(\mathbf{x}^{(k)}) = \sum_{i=1}^m -\lambda_i^{(k)} f_i(\mathbf{x}^{(k)})$, 其每个分量 $-\lambda_i^{(k)} f_i(\mathbf{x}^{(k)})$ 都衡量了第 $i$ 个 complementary slackness condition 的 violation, 因此 $\hat{\eta}^{(k)}$ 就是所有 complementary slackness condition violation 的总和.

### Primal-dual interior-point method

下给出 primal-dual interior-point method 的算法. 

> [!algorithm] Primal-dual interior-point method
>
> - **INPUT**: 给定初始点 $\mathbf{x}^{(0)}$ 满足 $f_i(\mathbf{x}^{(0)}) < 0, i = 1, \ldots, m$ (不对 $\mathbf{A} \mathbf{x}^{(0)} = \mathbf{b}$ 做要求). 给定 $\boldsymbol{\lambda}^{(0)} > 0$. 给定参数 $\mu >1$, $\varepsilon_{\text{feas}} > 0$, $\varepsilon_{\text{gap}} > 0$.
> - **REPEAT**: 对于第 $k$ 次迭代, 当前点为 $\mathbf{y}^{(k)} = (\mathbf{x}^{(k)}, \boldsymbol{\lambda}^{(k)}, \boldsymbol{\nu}^{(k)})$.
>     1. **Determine $t^{(k)}$**: 令 $t^{(k)} = \mu \frac{m}{\hat\eta^{(k)}}$.
>     2. **Compute search direction**: 通过求解如下方程组来得到 search direction $\Delta \mathbf{y}_{\text{nt}}^{(k)}$:
>         $$
>         \begin{bmatrix}
>         \nabla^2 f_0(\mathbf{x}^{(k)}) + \sum_{i=1}^m \lambda_i^{(k)} \nabla^2 f_i(\mathbf{x}^{(k)}) & Df(\mathbf{x}^{(k)})^\top & \mathbf{A}^\top \\
>         -\text{diag}(\boldsymbol{\lambda}^{(k)}) Df(\mathbf{x}^{(k)}) & -\text{diag}(f(\mathbf{x}^{(k)})) & \mathbf{0} \\
>         \mathbf{A} & \mathbf{0} & \mathbf{0}
>         \end{bmatrix} \begin{bmatrix}\Delta \mathbf{x}_{\text{nt}}^{(k)} \\ \Delta \boldsymbol{\lambda}_{\text{nt}}^{(k)} \\ \Delta \boldsymbol{\nu}_{\text{nt}}^{(k)}\end{bmatrix} = -\begin{bmatrix}\mathbf{r}_{t, \text{dual}}(\mathbf{y}^{(k)}) \\ \mathbf{r}_{t, \text{cent}}(\mathbf{y}^{(k)}) \\ \mathbf{r}_{t, \text{primal}}(\mathbf{y}^{(k)})\end{bmatrix}
>         $$
>     3. **Line search and update**: 通过 line search 来确定 step      size $s^{(k)}$, 并且更新 $\mathbf{y}^{(k+1)} = \mathbf{y}^{(k)} + s^{(k)} \Delta \mathbf{y}_{\text{nt}}^{(k)}$.
> - **UNTIL**: $\|\mathbf{r}_{t, \text{primal}}(\mathbf{y}^{(k)})\|_2 \leq \varepsilon_{\text{feas}}$, $\|\mathbf{r}_{t, \text{dual}}(\mathbf{y}^{(k)})\|_2 \leq \varepsilon_{\text{feas}}$, $\hat\eta^{(k)} \leq \varepsilon_{\text{gap}}$.

对于这个算法的解读如下. 
- 在 step 1 中, 如果当前点恰好在某个 central path 上, 则必有 $\hat\eta^{(k)} = m/t^{(k)}$, 因此 $t^{(k)} = \mu \frac{m}{\hat\eta^{(k)}}$ 就是 $t^{(k)} = \mu t^{(k)}$, 也就是说, $t$ 的值会在每次迭代中乘以 $\mu$. 因此, 这个 step 的作用就是让 $t$ 随着迭代的进行而逐渐增大, 从而使得 iterates 越来越接近于 central path.
- step 3 中, 我们需要通过 line search 来给三个分量 $\Delta \mathbf{x}_{\text{nt}}^{(k)}, \Delta \boldsymbol{\lambda}_{\text{nt}}^{(k)}, \Delta \boldsymbol{\nu}_{\text{nt}}^{(k)}$ 来确定一个公共的可行步长 $s^{(k)}$. 其中, 步长的选取要考虑保持满足如下条件:
  - $\boldsymbol{\lambda}^{(k)} + s^{(k)} \Delta \boldsymbol{\lambda}_{\text{nt}}^{(k)} > 0$, 因为 $\boldsymbol{\lambda}$ 需要保持非负. 因此最大的安全步长为 $s_{\text{max}}^{\boldsymbol{\lambda}} = \min_{i: \Delta \lambda_{\text{nt}, i}^{(k)} < 0} -\frac{\lambda_i^{(k)}}{\Delta \lambda_{\text{nt}, i}^{(k)}}$.
  - $f_i(\mathbf{x}^{(k)} + s^{(k)} \Delta \mathbf{x}_{\text{nt}}^{(k)}) < 0, i = 1, \ldots, m$, 因为 $\mathbf{x}$ 需要保持 strictly feasible. 不过这个无法给出一个 closed-form 的表达式, 只能通过 backtracking line search 来确定一个合适的 $s^{(k)}$. 如果某一个 $s$ 不满足 $f_i(\mathbf{x}^{(k)} + s \Delta \mathbf{x}_{\text{nt}}^{(k)}) < 0$ 的条件, 则通过类似 $s \leftarrow \beta s, \beta \in (0, 1)$ 的方式来缩小 $s$ 的值, 直到满足条件为止.
  - 此外, 还要求 residual 的 norm 有足够的 decrease, 例如满足 $\|\mathbf{r}_t(\mathbf{y}^{(k)} + s \Delta \mathbf{y}_{\text{nt}}^{(k)})\|_2 \leq (1 - \alpha s) \|\mathbf{r}_t(\mathbf{y}^{(k)})\|_2$, $\alpha \in (0, 1)$.