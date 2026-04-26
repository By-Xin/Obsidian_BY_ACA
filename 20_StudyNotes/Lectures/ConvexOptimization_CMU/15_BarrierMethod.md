# Barrier Method

>[!quote]
>
> - Lecture Reference: <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>

## Introduction

回顾对于无约束问题,
$$
\min_{\mathbf{x}\in\mathbb{R}^n} f(\mathbf{x}),
$$
我们可以直接使用牛顿法来求解:
$$
\mathbf{x}^{(k)}=\mathbf{x}^{(k-1)}-t_k\left[\nabla^2 f(\mathbf{x}^{(k-1)})\right]^{-1}\nabla f(\mathbf{x}^{(k-1)}),
$$
其中 $t_k$ 是步长可以通过 backtracking line search 等方法来确定. 

对于只包含等式约束的问题,
$$
\begin{aligned}
\min_{\mathbf{x}\in\mathbb{R}^n} \quad & f(\mathbf{x}) \\
\text{s.t.} \quad & A\mathbf{x}=\mathbf{b},
\end{aligned}
$$

Newton 方法需要确保每次迭代的点 $\mathbf{x}^{(k)}$ 都满足约束条件 $A\mathbf{x}^{(k)}=\mathbf{b}$, 这可以通过引入拉格朗日乘子来实现, 具体表现为求解以下线性系统:
$$
\begin{bmatrix}
\nabla^2 f(\mathbf{x}^{(k-1)}) & A^\top \\
A & 0
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{x} \\
\Delta \mathbf{y}
\end{bmatrix}=
\begin{bmatrix}
-\nabla f(\mathbf{x}^{(k-1)}) \\
0
\end{bmatrix},
$$
其中 $\Delta \mathbf{x}$ 和 $\Delta \mathbf{y}$ 分别是 primal 和 dual 变量的更新量.

最终的更新规则为:
$$
\mathbf{x}^{(k)}=\mathbf{x}^{(k-1)}+t\Delta \mathbf{x}.
$$
其中 $t$ 是步长, 可以通过 backtracking line search 等方法来确定.


真正比较麻烦的是当存在不等式约束的情况, 这也是 Barrier Method 主要解决的问题.

## Methodology

### Log Barrier Function

考虑如下约束优化问题
$$
\begin{aligned}
\min_{\mathbf{x}\in\mathbb{R}^n} \quad & f(\mathbf{x}) \\
\text{s.t.} \quad & h_i(\mathbf{x})\leq 0, \quad i=1,\ldots,m, \\
& \mathbf{A}\mathbf{x}=\mathbf{b},
\end{aligned}
$$
其中假设 $f$ 和 $h_i$ 都是定义在 $\mathbb{R}^n$ 上的凸函数且二阶可微, $\mathbf{A}\in\mathbb{R}^{p\times n}$, $\mathbf{b}\in\mathbb{R}^p$. 


对于上述问题, 额外定义一个 barrier function $\phi: \mathbb{R}^n\to\mathbb{R}$ 如下:
$$
\phi(\mathbf{x})=-\sum_{i=1}^m \log(-h_i(\mathbf{x})).
$$

- 注意到, 该函数的定义域是所有约束的严格可行域, 即 $\text{dom}(\phi)=\{\mathbf{x}\in\mathbb{R}^n: h_i(\mathbf{x})<0, i=1,\ldots,m\}$. 换言之, 只要有一个约束不满足, 就会导致 $\phi(\mathbf{x}) \to +\infty$. (此时也确保了 strong duality 的条件).
- 因此, 这相当于通过 log 函数设置了一个无穷大的 barrier 来阻止优化算法进入不可行域. 

如果暂时只考虑不包含等式约束的情况, 则可以将原问题转化为如下无约束优化问题:
$$
\begin{aligned}
\min_{\mathbf{x}\in\mathbb{R}^n} \quad & f(\mathbf{x}) - \frac{1}{t}\sum_{i=1}^m \log(-h_i(\mathbf{x})) 
\end{aligned}
$$


- 事实上, 这样的 barrier function 就相当于一个光滑版本的 indicator function $\delta(\mathbf{x})$:
    $$
    \min_{\mathbf{x}\in\mathbb{R}^n} \quad f(\mathbf{x})+\sum_{i=1}^m \delta_{\{h_i(\mathbf{x})\leq 0\}}(\mathbf{x}),
    $$
- 考虑函数 $-\log(-u) / t$. 事实上, 当 $t \to \infty$, 该函数就是上述 indicator function.


为后续分析方便, 首先给出 barrier function 的相关性质:
- 其梯度为
    $$
    \nabla \phi(\mathbf{x} )= - \sum_{i=1}^m \frac{1}{h_i(\mathbf{x})}\nabla h_i(\mathbf{x}),
    $$
- 其 Hessian 为
    $$
    \nabla^2 \phi(\mathbf{x}) = \sum_{i=1}^m \frac{1}{h_i(\mathbf{x})^2}\nabla h_i(\mathbf{x})\nabla h_i(\mathbf{x})^\top - \sum_{i=1}^m \frac{1}{h_i(\mathbf{x})}\nabla^2 h_i(\mathbf{x}).
    $$


### Central Path

给定一个 $t>0$, 考虑如下问题 (这里将 $1/t$ 吸收到了 $f$ 中, 以便后续分析):
$$
\begin{aligned}
\min_{\mathbf{x}\in\mathbb{R}^n} \quad & tf(\mathbf{x}) - \sum_{i=1}^m \log(-h_i(\mathbf{x})) := tf(\mathbf{x}) + \phi(\mathbf{x})  \\
\text{s.t.} \quad & \mathbf{A}\mathbf{x}=\mathbf{b}.
\end{aligned}
$$

对于每一个给定的 $t$, 上述问题都有一个唯一的最优解 $\mathbf{x}^\star(t)$, 因此这也可以看作是一个关于 $t$ 的函数(轨迹), 即 $t \mapsto \mathbf{x}^\star(t)$. 该函数被称为 central path. 
- 后续理论分析可以证明, 当 $t \to \infty$, $\mathbf{x}^\star(t)$ 会收敛到原问题的最优解 $\mathbf{x}^\star$.
- 当 $t$ 较小时, $\phi(\mathbf{x})$ 的影响较大, 因此 $\mathbf{x}^\star(t)$ 会远离约束的边界; 当 $t$ 较大时, $tf(\mathbf{x})$ 的影响较大, 因此 $\mathbf{x}^\star(t)$ 会更接近原问题的最优解.
- 因此, 最终的  central path 的感觉就是, 从较小的 $t$ 开始, 即从一个可行域内部比较中央远离边界的点开始, 随着 $t$ 的增加, 沿着一条轨迹逐渐接近原问题的最优解.
- 特别的, 我们之所以从小的 $t$ 开始而不是一上来就使用较大的 $t$, 是因为当 $t$ 越大就越接近非光滑的 indicator function, 这会导致优化问题变得更加困难. 而如果我们先用一个较小的 $t$ 来较为平稳的求解一个光滑的近似问题, 那么我们就可以以这个解作为下一次迭代的初始点, 从而逐渐增加 $t$ 来更接近原问题的最优解. 这也是 Barrier Method 的核心思想.

#### Central Path 的 KKT 条件和 Duality

对于上述的 barrier problem
$$
\begin{aligned}
\min_{\mathbf{x}\in\mathbb{R}^n} \quad & tf(\mathbf{x}) - \sum_{i=1}^m \log(-h_i(\mathbf{x})) := tf(\mathbf{x}) + \phi(\mathbf{x})  \\
\text{s.t.} \quad & \mathbf{A}\mathbf{x}=\mathbf{b},
\end{aligned}
$$
- 其 Lagrangian function 可以写作
    $$
    \mathcal{L}_t(\mathbf{x}, \mathbf{w}) = tf(\mathbf{x}) + \phi(\mathbf{x}) + \mathbf{w}^\top (\mathbf{A}\mathbf{x}-\mathbf{b}),
    $$
- 其关于 $\mathbf{x}$ 的梯度为:
    $$
    \nabla_{\mathbf{x}} \mathcal{L}_t(\mathbf{x}, \mathbf{w}) = t\nabla f(\mathbf{x}) + \nabla \phi(\mathbf{x}) + \mathbf{A}^\top \mathbf{w} = t\nabla f(\mathbf{x}) - \sum_{i=1}^m \frac{1}{h_i(\mathbf{x})}\nabla h_i(\mathbf{x}) + \mathbf{A}^\top \mathbf{w}.
    $$
- 关于 $\mathbf{w}$ 的梯度为:
    $$
    \nabla_{\mathbf{w}} \mathcal{L}_t(\mathbf{x}, \mathbf{w}) = \mathbf{A}\mathbf{x}-\mathbf{b}.
    $$
- 故在 central path $\mathbf{x}^\star(t)$ 上, 需满足:
    $$
    t\nabla f(\mathbf{x}^\star(t)) - \sum_{i=1}^m \frac{1}{h_i(\mathbf{x}^\star(t))}\nabla h_i(\mathbf{x}^\star(t)) + \mathbf{A}^\top \mathbf{w}^\star(t) = 0,
    $$
    $$
    \mathbf{A}\mathbf{x}^\star(t)-\mathbf{b} = 0.
    $$

但是另一方面, 我们真正关注的并不是这个 barrier problem 的对偶变量 $\mathbf{w}$, 而是原问题的相应变量. 其中
$$
\begin{aligned}
\min_{\mathbf{x}\in\mathbb{R}^n} \quad & f(\mathbf{x}) \\
\text{s.t.} \quad & h_i(\mathbf{x})\leq 0, \quad i=1,\ldots,m, \\
& \mathbf{A}\mathbf{x}=\mathbf{b},
\end{aligned}
$$
对应的 Lagrangian function 
$$
\mathcal{L}(\mathbf{x}, \mathbf{u}, \mathbf{v}) = f(\mathbf{x}) + \sum_{i=1}^m u_i h_i(\mathbf{x}) + \mathbf{v}^\top (\mathbf{A}\mathbf{x}-\mathbf{b}),
$$
中的对偶变量 $\mathbf{u}$ 和 $\mathbf{v}$. 

- 下根据上述的 central path 的 KKT 条件, 可以得到如下的关系:
    $$
    \begin{aligned}
    u_i^\star(t) &= -\frac{1}{t h_i(\mathbf{x}^\star(t))}, \quad i=1,\ldots,m, \\ 
    \mathbf{v}^\star(t) &= \frac{\mathbf{w}^\star(t)}{t}.
    \end{aligned}
    $$
    
    - 理由如下. 对于 Barrier 的 KKT 条件, 可以对其左右两侧同时除以 $t$, 从而得到
        $$
        \nabla f(\mathbf{x}^\star(t)) - \sum_{i=1}^m \frac{1}{t h_i(\mathbf{x}^\star(t))}\nabla h_i(\mathbf{x}^\star(t)) + \mathbf{A}^\top \frac{\mathbf{w}^\star(t)}{t} = 0,
        $$
        对应原问题的 KKT 条件, 可以得到上述的关系.


- 该原始问题对应的 dual function 为:
    $$
    g(\mathbf{u}, \mathbf{v}) = \inf_{\mathbf{x}\in\mathbb{R}^n} \mathcal{L}(\mathbf{x}, \mathbf{u}, \mathbf{v}) 
    $$
    若其为 dual feasible 的, 需要满足 $\mathbf{u}\geq 0$ 和 $g(\mathbf{u}, \mathbf{v}) > -\infty$. 故下尝试证明由上面得到的 $\mathbf{u}^\star(t)$ 和 $\mathbf{v}^\star(t)$ 满足上述条件. 
    - 首先断言 $\mathbf{u}^\star(t) > 0$. 理由如下.
        - 由于 $\mathbf{x}^\star(t)$ 是 central path 上的点是严格可行的, 因此 $h_i(\mathbf{x}^\star(t)) < 0$ 对于所有 $i=1,\ldots,m$ 都成立. 因此, 对于 $t > 0$, 可以得到 $u_i^\star(t) = -(t h_i(\mathbf{x}^\star(t)))^{-1} > 0$.
    - 其次断言 $g(\mathbf{u}^\star(t), \mathbf{v}^\star(t)) > -\infty$. 理由如下.
        - 由于 $\mathbf{x}^\star(t)$ 是 barrier problem 的最优解, 而 Barrier problem 本身和