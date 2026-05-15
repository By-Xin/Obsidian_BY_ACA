# Interior Point Methods: Barrier Method

>[!quote]
>
> - Lecture Reference: <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>
> - Reference Book: Boyd and Vandenberghe, Convex Optimization, Chapter 11

## Inequality Constrained Optimization

考虑如下含约束凸优化问题:
$$
\begin{aligned}
& \min_{\mathbf{x}} && f_0(\mathbf{x}) \\
& \text{subject to } && f_i(\mathbf{x}) \leq 0, \quad i=1, \ldots, m \\
& \quad\quad\quad\quad\quad && \mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned}
$$
- 其中 $f_0, \cdots, f_m: \mathbb{R}^n \to \mathbb{R}$ 是凸函数且二阶导连续可微. $\mathbf{A} \in \mathbb{R}^{p \times n}$ 是一个矩阵, 其行满秩 (即 $\operatorname{rank}(\mathbf{A}) = p<n$). 
  - 许多问题, 例如 LP, QO, QCQP (quadratically constrained quadratic program) 等问题都满足上述条件.
  - 还有一些问题, 例如最小化一组线性函数的最大值 ($\min_{\mathbf{x}} \{\max_{i=1, \ldots, m} \mathbf{a}_i^\top \mathbf{x}\}$) 的问题也可以通过引入一个新的变量 $t$ 转化为上述形式的优化问题.
  - 此外, SOCP, SDP 等问题也可以再更广义的形式下用同样的框架来描述.
- 假设最优解 $\mathbf{x}^\star$ 存在, 并记 $p^\star := f_0(\mathbf{x}^\star)$. 
- 此外假设 Slater 条件成立. 这意味着该凸优化问题是严格可行的, 即存在 $\mathbf{x}$ 使得 $f_i(\mathbf{x}) < 0$ 对所有 $i=1, \ldots, m$ 都成立, 且 $\mathbf{A} \mathbf{x} = \mathbf{b}$. 根据 Slater 定理, 若凸优化问题满足 Slater 条件, 则强对偶成立, 即原问题的最优值等于其对偶问题的最优值.

下给出其对偶问题及 KKT 条件:

- 原问题的拉格朗日函数:
    $$
    L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i f_i(\mathbf{x}) + \boldsymbol{\nu}^\top (\mathbf{A} \mathbf{x} - \mathbf{b}), \quad \lambda_i \geq 0, \quad i=1, \ldots, m
    $$

- 对应的 dual function 为:
    $$
    g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
    $$
- dual problem 为
    $$      
    \begin{aligned}
    & \max_{\boldsymbol{\lambda}, \boldsymbol{\nu}} && g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \\
    & \text{subject to } && \lambda_i \geq 0, \quad i=1, \ldots, m
    \end{aligned}
    $$

- 完整的 KKT 条件为:
    $$
    \begin{aligned}
    & \text{Primal feasibility: } && f_i(\mathbf{x}^\star) \leq 0, \quad i=1, \ldots, m; \quad \mathbf{A} \mathbf{x}^\star = \mathbf{b} \\
    & \text{Dual feasibility: } && \lambda_i^\star \geq 0, \quad i=1, \ldots, m \\
    & \text{Complementary slackness: } && \lambda_i^\star f_i(\mathbf{x}^\star) = 0, \quad i=1, \ldots, m \\
    & \text{Stationarity: } && \nabla f_0(\mathbf{x}^\star) + \sum_{i=1}^m \lambda_i^\star \nabla f_i(\mathbf{x}^\star) + \mathbf{A}^\top \boldsymbol{\nu}^\star = 0
    \end{aligned}
    $$

因此, 由凸优化的 Slater 条件, 求解最优解 $\mathbf{x}^\star$ 就等价于求解满足上述 KKT 条件的 $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$.

## Logarithmic Barrier Function and Central Path

暂时回到原问题来. 含有等式约束的方法比较好处理的, 这在 [[14_NewtonMethod.md]]  中有所体现. 故主要的问题就是如何处理不等式约束. 从一个比较统计学习的角度看, 我们可以通过引入一个罚函数 (penalty function) 来将不等式约束转化为一个无约束优化问题, 即
$$
\begin{aligned}
& \min_{\mathbf{x}} && f_0(\mathbf{x}) + \sum_{i=1}^m I_-(f_i(\mathbf{x})) \\
& \text{subject to } && \mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned}
$$
其中 $I_-(\cdot)$ 是一个指示函数, 定义如下:
$$
I_-(u) = \begin{cases}
0, & u \leq 0 \\
+\infty, & u > 0
\end{cases}
$$

### Logarithmic Barrier Function

由于上述 hard indicator 函数本身较难优化, 这里考虑一个近似的光滑版本, 即 logarithmic barrier function:
$$
\hat{I}_-(u) = -\frac{1}{t} \log(-u), \quad t > 0
$$
其定义域为 $\operatorname{dom}(\hat{I}_-) = \{u: u < 0\}$. 该函数相当于只在 $u < 0$ 的区域内对 $I_-(u)$ 进行近似, g故只允许严格满足约束 $f_i(\mathbf{x}) < 0$ 的解, 亦故名为 interior point method.  

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260515220053.png)
  
定义 log-barrier 函数 $\phi(\mathbf{x}) := -\sum_{i=1}^m \log(-f_i(\mathbf{x}))$, 则上述优化问题的近似版本为:
$$
\begin{aligned}
& \min_{\mathbf{x}} && f_0(\mathbf{x}) + \frac{1}{t} \phi(\mathbf{x}) = f_0(\mathbf{x}) - \frac{1}{t} \sum_{i=1}^m \log(-f_i(\mathbf{x})) \\
& \text{subject to } && \mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned}
$$
此外为分析方便, 给出 log-barrier 函数 $\phi$ 的梯度和 Hessian 的表达式:
$$
\begin{aligned}
\nabla \phi(\mathbf{x}) &= -\sum_{i=1}^m \frac{1}{f_i(\mathbf{x})} \nabla f_i(\mathbf{x}) \\
\nabla^2 \phi(\mathbf{x}) &= \sum_{i=1}^m \frac{1}{f_i(\mathbf{x})^2} \nabla f_i(\mathbf{x}) \nabla f_i(\mathbf{x})^\top - \sum_{i=1}^m \frac{1}{f_i(\mathbf{x})} \nabla^2 f_i(\mathbf{x})
\end{aligned}
$$

- $t$ 相当于这个问题的一个超参数, 其值越大, 近似越好, 但优化问题也越难求解.  因此, 在实践中往往会考虑 $t_1 < t_2 < \cdots$, 从一个较小的 $t_1$ 开始定义一个较为容易求解的优化问题并求解一个较为粗略的解 $\mathbf{x}_1$, 然后以 $\mathbf{x}_1$ 作为初始点来求解 $t_2$ 对应的优化问题, 以此类推, 直到 $t_k$ 足够大时得到一个较为精确的解 $\mathbf{x}_k$.
- 并且, 对于当前的这个转化后的等式约束问题, 其非常适合用 Newton 类二阶方法进行求解. 一方面, 我们拥有其二阶的全面信息, 能够得到更高精度下的更快收敛 (不过确实还是需要承认其二阶信息的处理成本); 并且, 由于 log-barrier 函数的特殊结构, 其 Hessian 矩阵具有一些特殊的结构 (尤其在接近边界时), 因此一阶方法反而可能会遇到一些数值稳定性的问题.


### Central Path

由于 $t$ 的这组 tradeoff 关系的存在, 我们从直觉上考虑通过一组 $t_1 < t_2 < \cdots$ 来进行序列化的求解. Central path 就是对这一过程的一个正式定义. 

***Definition* (Central Path)**: 对于每个 $t > 0$, 设 $\mathbf{x}^\star(t)$ 是如下问题的最优解:
$$
\begin{aligned}
& \min_{\mathbf{x}} && t f_0(\mathbf{x}) + \phi(\mathbf{x}) \\
& \text{subject to } && \mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned}
$$
则 $\{\mathbf{x}^\star(t): t > 0\}$ 就被称为 central path. 其中注意到这里对最优问题进行了一个等价的变换, 但在最优解上是等价的. 


下分析这个 centarl path 的性质.
-  首先, 上述问题的 Lagrangian function 可以写为:
    $$
    L(\mathbf{x}, \boldsymbol{\hat{\nu}}) = t f_0(\mathbf{x}) + \phi(\mathbf{x}) + \boldsymbol{\hat{\nu}}^\top (\mathbf{A} \mathbf{x} - \mathbf{b})
    $$
- 由于 $\mathbf{x}^\star(t)$ 是上述问题的最优解, 则其满足 KKT 条件:
    $$
    \begin{aligned}
    & \text{Primal feasibility: } && \mathbf{A} \mathbf{x}^\star(t) = \mathbf{b}, \quad  f_i(\mathbf{x}^\star(t)) < 0, \quad i=1, \ldots, m \\
    & \text{Stationarity: } && t \nabla f_0(\mathbf{x}^\star(t)) + \nabla \phi(\mathbf{x}^\star(t)) + \mathbf{A}^\top \boldsymbol{\hat{\nu}}^\star(t) = 0
    \end{aligned}
    $$
    再代入 $\nabla \phi(\mathbf{x}^\star(t))$ 的表达式, 可得:
    $$
    t \nabla f_0(\mathbf{x}^\star(t)) - \sum_{i=1}^m \frac{1}{f_i(\mathbf{x}^\star(t))} \nabla f_i(\mathbf{x}^\star(t)) + \mathbf{A}^\top \boldsymbol{\hat{\nu}}^\star(t) = 0
    $$