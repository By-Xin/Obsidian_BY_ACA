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
其中 $\text{diag}(\boldsymbol{\lambda})$ 是一个 $m \times m$ 的对角矩阵, 其对角线元素为 $\lambda_1, \ldots, \lambda_m$. 可以看出, 当 $\mathbf{r}_t(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = 0$ 时, 就满足了 modified KKT condition. 因此, primal-dual method 的核心就是通过 Newton's method 来求解 $\mathbf{r}_t(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = 0$.

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