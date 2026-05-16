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
\end{aligned} \tag{P}
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
    L_P(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i f_i(\mathbf{x}) + \boldsymbol{\nu}^\top (\mathbf{A} \mathbf{x} - \mathbf{b}), \quad \lambda_i \geq 0, \quad i=1, \ldots, m
    $$

- 对应的 dual function 为:
    $$
    g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} L_P(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
    $$
- dual problem 为
    $$      
    \begin{aligned}
    & \max_{\boldsymbol{\lambda}, \boldsymbol{\nu}} && g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \\
    & \text{subject to } && \lambda_i \geq 0, \quad i=1, \ldots, m
    \end{aligned} \tag {D-P}
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
\end{aligned} \tag{CP($t$)}
$$
则 $\{\mathbf{x}^\star(t): t > 0\}$ 就被称为 central path. 其中注意到这里对最优问题进行了一个等价的变换, 但在最优解上是等价的. 

因此, 给定一个原问题 $(\text{P})$, 我们可以构造一个 barrier 的子问题 $(\text{CP}(t))$. 下分析这个 centarl path 的性质.
-  首先, 上述问题的 Lagrangian function 可以写为:
    $$
    L_{CP}(\mathbf{x}, \boldsymbol{\hat{\nu}}) = t f_0(\mathbf{x}) + \phi(\mathbf{x}) + \boldsymbol{\hat{\nu}}^\top (\mathbf{A} \mathbf{x} - \mathbf{b})
    $$
- 由于 $\mathbf{x}^\star(t)$ 是上述问题的最优解, 则其满足 KKT 条件:
    $$
    \begin{aligned}
    & \text{Primal feasibility: } && \mathbf{A} \mathbf{x}^\star(t) = \mathbf{b}, \quad  f_i(\mathbf{x}^\star(t)) < 0, \quad i=1, \ldots, m \\
    & \text{Stationarity: } && t \nabla f_0(\mathbf{x}^\star(t)) - \sum_{i=1}^m \frac{1}{f_i(\mathbf{x}^\star(t))} \nabla f_i(\mathbf{x}^\star(t)) + \mathbf{A}^\top \boldsymbol{\hat{\nu}}^\star(t) = 0
    \end{aligned}
    $$
    其中 stationarity 条件系直接将 log-barrier 函数 $\phi$ 的梯度表达式代入到 KKT 条件中得到的. 


观察 central path 的 KKT 条件, 可以发现其与原问题 $(\text{P})$ 的 KKT 条件之间存在很强的联系.

- **Stationary 条件**: 对这个 central path 的 KKT stationary 进行变形, 左右两侧同时除以 $t$, 则可以得到如下表达式:
    $$
    \nabla f_0(\mathbf{x}^\star(t)) + \sum_{i=1}^m \underbrace{\frac{1}{t} \frac{1}{-f_i(\mathbf{x}^\star(t))}}_{\lambda_i^\star(t)} \nabla f_i(\mathbf{x}^\star(t)) + \mathbf{A}^\top \underbrace{\frac{1}{t} \boldsymbol{\hat{\nu}}^\star(t)}_{\boldsymbol{\nu}^\star(t)} = 0
    $$
    $$\iff
    \nabla f_0(\mathbf{x}^\star(t)) + \sum_{i=1}^m \lambda_i^\star(t) \nabla f_i(\mathbf{x}^\star(t)) + \mathbf{A}^\top \boldsymbol{\nu}^\star(t) = 0 \quad \tag{1}
    $$

- **Primal Feasibility 条件**: central path 的 primal feasibility 和原问题的 primal feasibility 条件是一样的.

- **Dual Feasibility 条件**: 并且, 由于 central path 上的点是严格可行的, 即 $f_i(\mathbf{x}^\star(t)) < 0$ 对所有 $i=1, \ldots, m$ 都成立, 则 $\lambda_i^\star(t) = \frac{1}{t} \frac{1}{-f_i(\mathbf{x}^\star(t))} > 0$ 对所有 $i=1, \ldots, m$ 都成立. 因此, central path 上的点自动满足原问题的 dual feasibility 条件.

- **Complementary Slackness 条件**: 原问题的 complementary slackness 条件要求 $\lambda_i^\star f_i(\mathbf{x}^\star) = 0$. 而 central path 上的点 $\mathbf{x}^\star(t)$ 根据 $\lambda_i^\star(t) = \frac{1}{t} \frac{1}{-f_i(\mathbf{x}^\star(t))}$ 的定义, 则 $\lambda_i^\star(t) f_i(\mathbf{x}^\star(t)) = -\frac{1}{t}$. 因此这个是一个 *perturbed complementary slackness* 条件, 其值为 $-\frac{1}{t}$ 而非 $0$. 不过随着 $t \to +\infty$, 这个值会趋近于 $0$, 从而满足原问题的 complementary slackness 条件. 因此, 从这一角度看, Barrier Method 相当于就是在求解一系列放松了 complementary slackness 条件的优化问题, 随着 $t$ 的增大, 这个放松的程度逐渐减小, 最终趋近于原问题的 KKT 条件.

由此可见, $(\text{CP}(t))$ 的 KKT 条件会诱导出一个 $(\mathbf{x}^\star(t), \boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t))$ 满足原问题 $(\text{P})$ 的 primal feasibility, dual feasibility 和 stationarity 条件, 以及一个 perturbed complementary slackness 条件. 下面说明, 这样诱导出的 
$$
\boldsymbol{\lambda}^\star(t) = \frac{1}{t} \frac{1}{-f_i(\mathbf{x}^\star(t))}, \quad \boldsymbol{\nu}^\star(t) = \frac{1}{t} \boldsymbol{\hat{\nu}}^\star(t)
$$ 
是原问题的对偶问题 $(\text{D-P})$ 的一个可行解.
  
- 回顾, 原问题的对偶问题为:
  $$
  \begin{aligned}
  & \max_{\boldsymbol{\lambda}, \boldsymbol{\nu}} && g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \\
  & \text{subject to } && \lambda_i \geq 0, \quad i=1, \ldots, m
  \end{aligned}
  $$
  其中 
  $$
  g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} L_P(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \{f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i f_i(\mathbf{x}) + \boldsymbol{\nu}^\top (\mathbf{A} \mathbf{x} - \mathbf{b})\}
  $$
  要说明 $(\boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t))$ 是 $(\text{D-P})$ 的一个可行解, 需要说明 $\lambda_i^\star(t) \geq 0$ 对所有 $i=1, \ldots, m$ 都成立 (已证 ), 以及 $g(\boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t)) > -\infty$. 


- 观察 $(1)$ 的表达式, 其恰好是原问题的 Lagrangian, 在固定 $\boldsymbol{\lambda} = \boldsymbol{\lambda}^\star(t), \boldsymbol{\nu} = \boldsymbol{\nu}^\star(t)$ 的情况下, 关于 $\mathbf{x}$ 的梯度 (stationary 条件) 的表达式. 即:
    $$ (1) \iff
    \frac{\partial L_P(\mathbf{x}, \boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t))}{\partial \mathbf{x}} \Big|_{\mathbf{x} = \mathbf{x}^\star(t)} = 0
    $$
    故 $\mathbf{x}^\star(t)$ 是 $\inf_{\mathbf{x}} L_P(\mathbf{x}, \boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t))$ 的一个 stationary point.  又由于 $L_P(\mathbf{x}, \boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t))$ 是一个关于 $\mathbf{x}$ 的凸函数, 则 $\mathbf{x}^\star(t)$ 也是 $\inf_{\mathbf{x}} L_P(\mathbf{x}, \boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t))$ 的一个 global minimizer. 因此, $g(\boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t)) = L_P(\mathbf{x}^\star(t), \boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t)) > -\infty$.


最后考察其对偶间隙 (duality gap). 

- 由于 $\mathbf{x}^\star(t)$ 是原问题的对偶问题的一个可行解, 故
    $$
    \begin{aligned}
    g(\boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t)) &= \inf_{\mathbf{x}} L_P(\mathbf{x}, \boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t)) \\
    &= L_P(\mathbf{x}^\star(t), \boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t)) \\
    &= f_0(\mathbf{x}^\star(t)) + \sum_{i=1}^m \lambda_i^\star(t) f_i(\mathbf{x}^\star(t)) + \boldsymbol{\nu}^\star(t)^\top (\mathbf{A} \mathbf{x}^\star(t) - \mathbf{b}) \\
    \end{aligned}
    $$


- 而根据 definition, $\lambda_i^\star(t) = \frac{1}{t} \frac{1}{-f_i(\mathbf{x}^\star(t))}$, 且注意到 $\mathbf{A} \mathbf{x}^\star(t) = \mathbf{b}$, 则上式可以继续变形为:
    $$
    \begin{aligned}
    g(\boldsymbol{\lambda}^\star(t), \boldsymbol{\nu}^\star(t)) &= f_0(\mathbf{x}^\star(t)) + \sum_{i=1}^m \frac{1}{t} \frac{1}{-f_i(\mathbf{x}^\star(t))} f_i(\mathbf{x}^\star(t)) + \boldsymbol{\nu}^\star(t)^\top (\mathbf{A} \mathbf{x}^\star(t) - \mathbf{b}) \\
    &= f_0(\mathbf{x}^\star(t)) - \frac{m}{t}
    \end{aligned}
    $$

- 因此最终有结论:
    $$
    f_0(\mathbf{x}^\star(t)) - p^\star \leq \frac{m}{t}
    $$
    
    
综上, 对于每次的 barrier subproblem $(\text{CP}(t))$, 其最优解 $\mathbf{x}^\star(t)$ 的目标值与原问题的最优值 $p^\star$ 之间的 gap 不超过 $\frac{m}{t}$. 因此, 随着 $t \to +\infty$, 这个 gap 会趋近于 $0$, 从而 $\mathbf{x}^\star(t)$ 会趋近于原问题的最优解 $\mathbf{x}^\star$.


## Barrier Method Algorithm

基于上述分析, 下正式讨论使用 Barrier Method 来求解原问题 $(\text{P})$ 的算法细节. 由于已知 $f_0(\mathbf{x}^\star(t)) - p^\star \leq \frac{m}{t}$, 则给定优化精度 $\varepsilon > 0$, 只需要选择 $t \geq \frac{m}{\varepsilon}$ 即可保证 $\mathbf{x}^\star(t)$ 是一个 $\varepsilon$-optimal 的解. 故下考虑
$$
\begin{aligned}
& \min_{\mathbf{x}} && \frac{m}{\varepsilon} f_0(\mathbf{x}) + \phi(\mathbf{x}) \\
& \text{subject to } && \mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned}  \tag{CP($\frac{m}{\varepsilon}$)}
$$
的求解. 不过立即发现, 若对精度 $\varepsilon$ 的要求较高, 则 $t = \frac{m}{\varepsilon}$ 会非常大, 从而导致求解 $(\text{CP}(\frac{m}{\varepsilon}))$ 这个 barrier subproblem 变得非常困难. 因此, 在实践中会使用 warm start 的方式来逐步求解一系列的 barrier subproblem. 一个典型的递进方法是, 给定一个初始的 $t_0 > 0$ 和一个增大因子 $\mu > 1$, 则对于 $k=0, 1, \ldots$, 依次求解如下的 barrier subproblem:
$$
\begin{aligned}& \min_{\mathbf{x}} && t_k f_0(\mathbf{x}) + \phi(\mathbf{x}) \\
& \text{subject to } && \mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned} \tag{CP($t_k$)}
$$
其中 $t_k = t_0 \mu^k$. 直到 $t_k \geq \frac{m}{\varepsilon}$ 时停止, 此时 $\mathbf{x}^\star(t_k)$ 就是一个 $\varepsilon$-optimal 的解. 也就是得到一个序列 $\{\mathbf{x}^\star(t_0), \mathbf{x}^\star(\mu t_0), \mathbf{x}^\star(\mu^2 t_0), \ldots\}$, 其中 $\mathbf{x}^\star(t_k)$ 是 $(\text{CP}(t_k))$ 的最优解, 且随着 $k$ 的增大, $\mathbf{x}^\star(t_k)$ 会逐渐趋近于原问题 $(\text{P})$ 的最优解 $\mathbf{x}^\star$.

对应的算法伪代码如下:

- **Input**: 严格可行的初始点 $\mathbf{x}_0$, 初始 barrier 参数 $t_0 > 0$, 增大因子 $\mu > 1$, 精度要求 $\varepsilon > 0$.
- **Repeat**:
    1. 对当前 $t$, 求解 barrier subproblem: $\mathbf{x}^\star(t) = \arg\min_{\mathbf{x}} \{t f_0(\mathbf{x}) + \phi(\mathbf{x}): \mathbf{A} \mathbf{x} = \mathbf{b}\}$. 通常这一步会使用 Newton 类二阶方法来求解.
    2. 更新 $\mathbf{x} \leftarrow \mathbf{x}^\star(t)$为下一个子问题的初始点.
    3. 直到 $t \geq \frac{m}{\varepsilon}$ 时停止.
    4. 若 $t < \frac{m}{\varepsilon}$, 则更新 $t \leftarrow \mu t$.

总的而言, 这里其实会有内外两层迭代结构. 外层迭代是对 $t$ 的更新, 每次定义了一个新的 barrier subproblem; 内层迭代是对当前的 barrier subproblem 的例如通过 Newton 方法来求解. Newton 方法的求解过程总是能够保证每次迭代都在 log-barrier 函数的定义域内, 但我们确实需要 Newton 的解的精度 $\varepsilon_{\text{Newton}} \ll \varepsilon$ 来保证最终的解 $\mathbf{x}^\star(t)$ 的精度 $\varepsilon$.