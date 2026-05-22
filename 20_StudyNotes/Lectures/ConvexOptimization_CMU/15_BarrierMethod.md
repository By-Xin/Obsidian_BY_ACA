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

### Algorithm Details

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

总的而言, 这里其实会有内外两层迭代结构. 外层迭代是对 $t$ 的更新, 每次定义了一个新的 barrier subproblem; 内层迭代是对当前的 barrier subproblem 的例如通过 Newton 方法来求解.

在实现细节上, 有如下讨论.

- 关于内层 centering 的精度: 
  - 在理论推导上, 我们总是假设每次的内层方法都能得到一个收敛的最优解 $\mathbf{x}^\star(t)$.
  - 在实际的算法实现中, 由于数值计算的限制, 我们只能得到一个近似的解 $\hat{\mathbf{x}}(t) \approx \mathbf{x}^\star(t)$. 不过可以证明, 只要每次的求解足够精确, 即使不是完全收敛到 $\mathbf{x}^\star(t)$, 也能在渐近的意义上保证收敛. 
  - 需要承认, 这样的估计会影响我们对于 Stationarity 的判断, 从而会影响 $m/t$ 这个 gap 的估计.
  - 后续也有一些工作会在这样的情况下对 $\boldsymbol{\lambda}, \boldsymbol{\nu}$ 的估计进行一些修正. 

- 关于 $\mu$ 的选择:
  - 过小的 $\mu$ 会导致需要求解更多的 barrier subproblem, 从而增加外层迭代的次数; 过大的 $\mu$ 会导致每次的 barrier subproblem 变得更难求解, 从而增加内层迭代的次数. 因此, $\mu$ 的选择需要在外层迭代的次数和内层迭代的难度之间进行权衡.
  - 不过经验上, $\mu$ 的选择并不敏感, 大概在 $\mu \in [3,100]$ 的范围内都能得到不错的性能. 通常会选择 $\mu = 10, 20$ 等.
  - 若需要在理论上进行 worst case 的一些复杂度相关分析, 则往往会选择 $\mu$ 接近 $1^+$, 从而保证每次的 barrier subproblem 之间比较接近. 


- 关于初始强度 $t_0$ 的选择:
  - 若 $t_0$ 过小, 则初始的 barrier subproblem 会比较容易求解, 但这个 gap $m/t$ 会比较大, 从而需要更多的外层迭代来逐步缩小 gap; 若 $t_0$ 过大, 则初始的 barrier subproblem 就会比较难求解, 从而增加内层迭代的次数. 
  - 若在初始时, 我们有 primal-dual gap 的初始估计, 例如已经有一个 primal feasible 解 $\mathbf{x}_0$ 和一个 dual feasible 解 $(\boldsymbol{\lambda}_0, \boldsymbol{\nu}_0)$, 则可以通过 $t_0 = \frac{m}{f_0(\mathbf{x}_0) - g(\boldsymbol{\lambda}_0, \boldsymbol{\nu}_0)}$ 来选择一个合适的 $t_0$. 
  - 若只有一个 feasible 的初始点 $\mathbf{x}_0$ (即满足 $f_i(\mathbf{x}_0) < 0$ 且 $\mathbf{A} \mathbf{x}_0 = \mathbf{b}$), 但并不知道 $\boldsymbol{\lambda}_0, \boldsymbol{\nu}_0$ 的话, 可以选择一个 $t_0$ 使得当前已知的初始值 $\mathbf{x}_0$ 尽量接近于这一步想要的最优值 $\mathbf{x}^\star(t_0)$, 从而保证内层迭代的效率. 
    - 当然这个条件的判断不是让我们去求解 $\mathbf{x}^\star(t_0)$ 来进行比较. 考虑这个问题的 KKT Stationary 条件. 对于最优解 $\mathbf{x}^\star(t_0)$, 其当然满足 $t_0 \nabla f_0(\mathbf{x}^\star(t_0)) + \nabla \phi(\mathbf{x}^\star(t_0)) + \mathbf{A}^\top \boldsymbol{\hat{\nu}}^\star(t_0) = 0$. 因此, 可以通过求解 $\min_{\boldsymbol{\nu}, t>0} \|t \nabla f_0(\mathbf{x}_0) + \nabla \phi(\mathbf{x}_0) + \mathbf{A}^\top \boldsymbol{\nu}\|$ 来选择一个合适的 $t_0$. 这是一个标准的最小二乘问题. 
    - 还有一个细节是, 这里的范数可以考虑使用这个残差的由 $H_0^{-1}$ (Hessian 诱导的范数) 来进行度量, 其中 $H_0$ 是 $\mathbf{x}_0$ 处的 Hessian 矩阵. 这是因为 $\ell_2$ 的 norm 是各项同性的, 而 Hessian 诱导的 norm 则能够更好地反映出不同维度上残差的相对重要性.

- 关于初始点 $\mathbf{x}_0$ 的选择:
  - 一般而言, 需要选择一个严格可行的初始点 $\mathbf{x}_0$. 不论如何, $f_i(\mathbf{x}_0) < 0$ 的约束是不可以违反的 (至少在当前传统框架下). 不过对于 $\mathbf{A} \mathbf{x}_0 = \mathbf{b}$ 的约束, 在一开始时也可以允许其被违反. 此时, 可以通过 infeasible-start Newton method 来开始优化, 这个方法将逐步调整 $\mathbf{x}$ 来满足等式约束, 同时逐步优化目标函数. 

- 简单的收敛性分析:
  - 根据结论, $f_0(\mathbf{x}^\star(t)) - p^\star \leq \frac{m}{t}$, 若使用 $t_k = t_0 \mu^k$, 则当 $t_k \geq \frac{m}{\varepsilon}$ 时, 就有 $f_0(\mathbf{x}^\star(t_k)) - p^\star \leq \varepsilon$. 因此, 只需要 $k \geq \log_\mu(\frac{m}{t_0 \varepsilon})$ 就能保证 $\mathbf{x}^\star(t_k)$ 是一个 $\varepsilon$-optimal 的解. 这说明外层迭代的次数是 $O(\log(\frac{1}{\varepsilon}))$ 的级别.


## Feasibility and Phase I Problem

### Basic Phase I Problem

上面我们曾提过, 对于一个初始解 $\mathbf{x}_0$ 来说, 其必须是严格可行的, 即满足 $f_i(\mathbf{x}_0) < 0$ 对所有 $i=1, \ldots, m$ 都成立, 且 $\mathbf{A} \mathbf{x}_0 = \mathbf{b}$. 这里, 对于 $\mathbf{A} \mathbf{x}_0 = \mathbf{b}$ 的违反可以通过 infeasible-start Newton method 来逐步调整. 然而若 $f_i(\mathbf{x}_0) < 0$ 的约束被违反, 则根本无法开始优化. 因此, 需要先解决一个 feasibility problem 来找到一个满足 $f_i(\mathbf{x}) < 0$ 对所有 $i=1, \ldots, m$ 都成立的初始点 $\mathbf{x}_0$. 这就是所谓的 Phase I problem, 是 barrier method 的一个预处理步骤.


对于 Phase I 问题, 其目标是通过 barrier method 逐步优化求解如下优化问题:
$$
\begin{aligned}
& \min_{\mathbf{x}, s} && s \\
& \text{subject to } && f_i(\mathbf{x}) \leq s, \quad i=1, \ldots, m \\
& \quad\quad\quad\quad\quad && \mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned} \tag{P-Phase I}
$$
换言之其 centering problem 的形式为:
$$
\begin{aligned}
& \min_{\mathbf{x}, s} && t s - \sum_{i=1}^m \log(s - f_i(\mathbf{x})) \\
& \text{subject to } && \mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned} \tag{CP-Phase I($t$)}
$$
其中 $s$ 是一个 slack variable, 其值越小, 则 $\mathbf{x}$ 就越接近于满足 $f_i(\mathbf{x}) < 0$ 的约束. 

在迭代过程中 ($(s^{(0)}, \mathbf{x}^{(0)}) \to (s^{(1)}, \mathbf{x}^{(1)}) \to \cdots$), 初始时可以选择一个足够大的 $s^{(0)}$ 来保证 $f_i(\mathbf{x}^{(0)}) \leq s^{(0)}$ 对所有 $i=1, \ldots, m$ 都成立, 从而保证初始点的可行性. 随着迭代的进行, $s^{(k)}$ 会逐渐减小, 从而 $\mathbf{x}^{(k)}$ 也会逐渐趋近于满足 $f_i(\mathbf{x}) < 0$ 的约束. 因此只需要在某次迭代过程中有 $s^{(k)} < 0$ 就可以停止, 此时 $\mathbf{x}^{(k)}$ 就是一个满足 $f_i(\mathbf{x}) \leq s < 0$ 的可行解, 从而可以作为后续 barrier method 的初始点 $\mathbf{x}_0$ 来使用.

相应地, 根据这个 Phase I 问题的最优值 $p^\star_{\text{Phase I}}$, 可以得到如下结论:
- 若 $p^\star_{\text{Phase I}} < 0$, 则原问题 $(\text{P})$ 是可行的. 并且我们这个g于辅助的题的求解也不需要太过精确, 只要能够得到一个 $s < 0$ 的解即可.
- 若 $p^\star_{\text{Phase I}} > 0$, 则原问题 $(\text{P})$ 是不可行的. 可以直接停止算法, 输出原问题不可行.
- 若 $p^\star_{\text{Phase I}} = 0$, 则还会分为两种具体情况. 如果 $p^\star_{\text{Phase I}} = 0$ 的最优解 $\mathbf{x}^\star_{\text{Phase I}}$ 是可达的, 说明其不等式组是可行的, 但是没有严格可行解 (没有内点). 若 $p^\star_{\text{Phase I}} = 0$ 的最优解 $\mathbf{x}^\star_{\text{Phase I}}$ 是不可达的, 则说明其不等式组是不可行的. 


### Sum-of-Infeasibilities Phase I Variant

上述的 Phase I 辅助问题还有许多其他的变体, 例如 sum of infeasibilities 的形式:
$$
\begin{aligned}
& \min_{\mathbf{x}, s} && \sum_{i=1}^m s_i \\
& \text{subject to } && f_i(\mathbf{x}) \leq s_i, \quad i=1, \ldots, m \\
& \quad\quad\quad\quad\quad && \mathbf{A} \mathbf{x} = \mathbf{b} \\
& \quad\quad\quad\quad\quad && s_i \geq 0, \quad i=1, \ldots, m
\end{aligned} \tag{P-Phase I-Sum}
$$

相比于 (P-Phase I) 中的单一 slack variable $s$, 这里为每个约束引入了一个 slack variable $s_i$, 相当于优化的更为细致. 让所有的约束共享一个 $s$ 的形式优化, 就很有可能造成 "许多约束都违背一点点" 的情况. 然而, 如果对于每个约束各自引入一个 slack variable, 则可以更好地反映出每个约束的具体违背程度, 从而只让少部分约束违背, 从而到更为稀疏的违背情况. 

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260522174227.png)

如图左右两侧分别是 (P-Phase I) 和 (P-Phase I-Sum) 的违背情况的 histogram. 对于传统方法, 其只满足 100 个约束中的 39 个. 而对于 sum of infeasibilities 的方法, 其满足了 100 个约束中的 79 个.

### Termination near the Phase II central path

上述的 Phase I 过程中, 尽管找到了严格可行点  $\mathbf{x} \in \{f_i(\mathbf{x}) < 0, \mathbf{A} \mathbf{x} = \mathbf{b}\}$, 然而这个点可能离 Phase II 的 central path 比较远, 从而导致后续的 barrier method 需要多步的 Newton 迭代才能逐步调整到 central path 上, 从而增加了整体的计算成本. 因此, 在 Phase I 的过程中, 可以进一步考虑让 $\mathbf{x}$ 不仅满足严格可行, 并且尽量能够接近 Phase II 的 central path. 

考虑回 basic Phase I 的问题, 对其进行进一步改造为:
$$
\begin{aligned}
& \min_{\mathbf{x}, s} && s  \\
& \text{subject to } && f_i(\mathbf{x}) \leq s, \quad i=1, \ldots, m \\
& \quad\quad\quad\quad\quad && \mathbf{A} \mathbf{x} = \mathbf{b} \\
& \quad\quad\quad\quad\quad && f_0(\mathbf{x}) \leq M 
\end{aligned} \tag{P-Phase I-Modified}
$$
其中 $M$ 是一个 sufficiently large 的常数, 其值需要满足
$$ 
M > \max \{f_0(\mathbf{x^{(0)}}), p^\star\}
$$

为说明其作用, 对 (P-Phase I-Modified) 使用 log-barrier 来进行求解, 给定 $\tilde{t}$, 则其 centering problem 的形式为:
$$
\begin{aligned}
& \min_{\mathbf{x}, s} && \tilde{t} s - \sum_{i=1}^m \log(s - f_i(\mathbf{x})) - \log(M - f_0(\mathbf{x})) \\
& \text{subject to } && \mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned} \tag{CP-Phase I-Modified($t$)}
$$
- 求解其 KKT 条件的 Stationary 条件, 可以得到如下表达式:
    $$
    \begin{aligned}
    & \tilde{t} - \sum_{i=1}^m \frac{1}{s - f_i(\mathbf{x})} = 0 \\
    & \sum_{i=1}^m \frac{1}{s - f_i(\mathbf{x})} \nabla f_i(\mathbf{x}) + \frac{1}{M - f_0(\mathbf{x})} \nabla f_0(\mathbf{x}) + \mathbf{A}^\top \boldsymbol{\hat{\nu}} = 0
    \end{aligned}
    $$


- 特别地, 对于这个 Stationary 条件, 其在 $s = 0$ 时, 有
    $$
    \begin{aligned}
    & \tilde{t} - \sum_{i=1}^m \frac{1}{- f_i(\mathbf{x})} = 0 \\
    &  \frac{1}{M - f_0(\mathbf{x})} \nabla f_0(\mathbf{x}) -  \sum_{i=1}^m \frac{1}{f_i(\mathbf{x})} \nabla f_i(\mathbf{x}) + \mathbf{A}^\top \boldsymbol{\hat{\nu}} = 0
    \end{aligned}
    $$
    回顾, 对于原先的 Phase II 的 Barrier 子问题的 Stationary 条件, 其表达式为:
    $$
    t \nabla f_0(\mathbf{x}^\star(t)) - \sum_{i=1}^m \frac{1}{f_i(\mathbf{x}^\star(t))} \nabla f_i(\mathbf{x}^\star(t)) + \mathbf{A}^\top \boldsymbol{\hat{\nu}}^\star(t) = 0
    $$
    故只要令 $t = (M - f_0(\mathbf{x}))^{-1}$, 则上式就和 Phase II 的 Stationary 条件的表达式完全一样了. 


- 综上, 通过在 Phase I 的问题中引入一个额外的约束 $f_0(\mathbf{x}) \leq M$, 就能够让 Phase I 的 central path 上的点在 $s=0$ 时, 就满足 Phase II 的 Stationary 条件. 因此, 通过求解 Phase I 的问题, 就能够直接得到一个满足 Phase II 的 Stationary 条件的点, 从而避免了后续的 barrier method 需要多步的 Newton 迭代来逐步调整到 central path 上的问题.


### Phase I via infeasible start Newton method

与前述不同, 这里不再专门考虑通过求解 Phase I 来找严格可行点, 而是用 infeasible start Newton method, 从一个不满足等是约束的点开始, 逐渐拉回到满足等式约束的 feasible region 上. 考虑如下问题:
$$
\begin{aligned}
    & \min_{\mathbf{x}} && f_0(\mathbf{x}) \\
    & \text{subject to } && f_i(\mathbf{x}) \leq s, \quad i=1, \ldots, m \\
    & \quad\quad\quad\quad\quad && \mathbf{A} \mathbf{x} = \mathbf{b} \\
    & \quad\quad\quad\quad\quad && s_i = 0, \quad i=1, \ldots, m
\end{aligned}
$$

这个设置非常的 tricky, 显然这个优化问题和原始优化问题是等价的. 然而此时即使我们 $f_i(\mathbf{x}) > 0$, 我们违反的仍然是 $s_i = 0$ 的约束, 而非 $f_i(\mathbf{x}) \leq s$ 的约束. 因此, 在实践中, 我们最开始先取 $s > \max_i f_i(\mathbf{x})$, 从而保证 $f_i(\mathbf{x}) \leq s$ 的约束是严格满足的, 从而保证初始点的可行性. 因此我们可以求解这个问题的 barrier subproblem 来得到一个满足 $f_i(\mathbf{x}) \leq s$ 的可行点, 从而作为后续 barrier method 的初始点 $\mathbf{x}_0$ 来使用:
$$
\begin{aligned}
& \min_{\mathbf{x}, s} && \bar{t} s - \sum_{i=1}^m \log(s - f_i(\mathbf{x})) \\
& \text{subject to } && \mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned}
$$
这是一个标准的 barrier subproblem, 可以通过 Newton 方法来求解. 

进一步, 记这个优化问题中所有函数的公共定义域 $\mathcal{D} = \bigcap_{i=0}^m \text{dom}(f_i)$. 若可以确定任意一个 $\mathbf{x} \in \mathcal{D}$, 则可以选择一个足够大的 $s > \max_i f_i(\mathbf{x})$, 从而保证 log barrier $-\log(s - f_i(\mathbf{x}))$ 全部有定义. 因此, 虽然此时 $s = 0$ 的约束被违反了, 但可以通过 infeasible start Newton method 来完成优化. 

不过有时本身, 当约束本身就比较复杂, 能够直接确定一个 $\mathbf{x} \in \mathcal{D}$ 也较为困难时, 可以进一步给每一个函数单独引入一个 shift: $z_0, \ldots, z_m$, 从而考虑如下的优化问题:
$$
\begin{aligned}
& \min_{\mathbf{x}, s, z_0, \ldots, z_m} &&  \bar{t} f_0(\mathbf{x} + z_0) + \bar{t} s - \sum_{i=1}^m \log(s - f_i(\mathbf{x} + z_i)) \\
& \text{subject to } && \mathbf{A} \mathbf{x} = \mathbf{b} \\
& \quad\quad\quad\quad\quad && s_i = 0, \quad i=1, \ldots, m \\
& \quad\quad\quad\quad\quad && z_i = 0, \quad i=0, 1, \ldots, m
\end{aligned}
$$
也就是先不必费力去寻找所有函数的一个公共定义域的点, 而是通过引入 shift variable 来让每个函数都能够有一个单独的定义域, 从而保证初始点的可行性. 当然, 这里的 shift variable 最终也需要被优化到 $0$ 的位置, 从而保证最终求解的 $\mathbf{x}$ 是原问题的一个可行解. 