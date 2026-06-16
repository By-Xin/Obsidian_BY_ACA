# Dual Decomposition

>[!quote]
>
> - Lecture Reference: 
>   - <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>

## Introduction

### Preliminary: Fenchel Conjugate

给定 $f: \mathbb{R}^n \to \mathbb{R}$ 是一个 proper, closed, convex 函数, 其 subgradient 在 $\mathbf{x} \in \mathbb{R}^n$ 处定义为:
$$
\partial f(\mathbf{x}) = \{\mathbf{g} \in \mathbb{R}^n: f(\mathbf{z}) \geq f(\mathbf{x}) + \mathbf{g}^\top (\mathbf{z} - \mathbf{x}), \forall \mathbf{z} \in \mathbb{R}^n\}.
$$

其有性质如下.

***Property 1.*** $\mathbf{y} \in \partial f(\mathbf{x})$ 当且仅当 $\mathbf{x} \in \arg\min_{\mathbf{z} \in \mathbb{R}^n} \left\{f(\mathbf{z}) - \mathbf{y}^\top \mathbf{z}\right\}$.

- *Proof.* 由 $\mathbf{y} \in \partial f(\mathbf{x})$ 可得对任意 $\mathbf{z} \in \mathbb{R}^n$, 有 $f(\mathbf{z}) \geq f(\mathbf{x}) + \mathbf{y}^\top (\mathbf{z} - \mathbf{x})$, 从而 $f(\mathbf{z}) - \mathbf{y}^\top \mathbf{z} \geq f(\mathbf{x}) - \mathbf{y}^\top \mathbf{x}$, 即 $\mathbf{x} \in \arg\min_{\mathbf{z} \in \mathbb{R}^n} \left\{f(\mathbf{z}) - \mathbf{y}^\top \mathbf{z}\right\}$. 

    $\square$

$f$ 的 Fenchel conjugate 定义为:
$$
f^*(\mathbf{y}) = \sup_{\mathbf{x} \in \mathbb{R}^n} \left( \mathbf{y}^\top \mathbf{x} - f(\mathbf{x}) \right)  \iff  -f^*(\mathbf{y}) = \inf_{\mathbf{x} \in \mathbb{R}^n} \left( f(\mathbf{x}) - \mathbf{y}^\top \mathbf{x} \right), \quad  \mathbf{y} \in \mathbb{R}^n .
$$

其有如下性质. 

***Property 2.*** $\mathbf{y} \in \partial f(\mathbf{x})$ 当且仅当 $\mathbf{x} \in \partial f^*(\mathbf{y})$.
- *Proof.* 由 $\mathbf{y} \in \partial f(\mathbf{x})$ 可得 $\mathbf{x} \in \arg\min_{\mathbf{z} \in \mathbb{R}^n} \left\{f(\mathbf{z}) - \mathbf{y}^\top \mathbf{z}\right\}$, 即 $f(\mathbf{x}) - \mathbf{y}^\top \mathbf{x} = \inf_{\mathbf{z} \in \mathbb{R}^n} \left\{f(\mathbf{z}) - \mathbf{y}^\top \mathbf{z}\right\}$. 根据 Fenchel conjugate 的定义, 可得 $f^*(\mathbf{y}) = \mathbf{y}^\top \mathbf{x} - f(\mathbf{x})$. 而对任意 $\mathbf{u} \in \mathbb{R}^m$,  $f^*(\mathbf{u}) = \sup_{\mathbf{z} \in \mathbb{R}^n} \left( \mathbf{u}^\top \mathbf{z} - f(\mathbf{z}) \right) \geq \mathbf{u}^\top \mathbf{x} - f(\mathbf{x})$. 从而将两式相减, 有 $f^*(\mathbf{u}) - f^*(\mathbf{y}) \geq (\mathbf{u} - \mathbf{y})^\top \mathbf{x}$, 即 $\mathbf{x} \in \partial f^*(\mathbf{y})$.

    $\square$

***Property 3*** 若 $f$ 是 closed, proper, convex 函数, 则 $f$ 是 $m$-strongly convex $\iff$ $\nabla f^*$ 是 $1/m$-Lipschitz 的. 

- *Proof.* 
    - $(\Rightarrow)$ 记 $\mathbf{x}_{\mathbf{u}} = \arg\min_{\mathbf{x}} \{ f(\mathbf{x}) - \mathbf{u}^\top \mathbf{x} \}$, $\mathbf{x}_{\mathbf{v}} = \arg\min_{\mathbf{x}} \{ f(\mathbf{x}) - \mathbf{v}^\top \mathbf{x} \}$. 由于 $f$ 是 $m$-强凸的, 故 $f(\mathbf{x}) - \mathbf{u}^\top \mathbf{x}$ 也是 $m$-强凸且唯一最小值取在 $\mathbf{x}_\mathbf{u}$ 的函数 (由强凸可推出 $f^*$ 可微, 故后文取 $\nabla f^*$ 合法). 根据强凸的性质, 对于任意 $\mathbf{y} \in \operatorname{dom}(f)$ 有:
        $$
        f(\mathbf{y})  - \mathbf{u}^\top \mathbf{y} \geq f(\mathbf{x}_{\mathbf{u}}) - \mathbf{u}^\top \mathbf{x}_{\mathbf{u}} + \frac{m}{2}\|\mathbf{y} - \mathbf{x}_{\mathbf{u}}\|_2^2
        $$
        再令 $\mathbf{y} = \mathbf{x}_{\mathbf{v}}$, 则有:
        $$
        f(\mathbf{x}_{\mathbf{v}}) - \mathbf{u}^\top \mathbf{x}_{\mathbf{v}} \geq f(\mathbf{x}_{\mathbf{u}}) - \mathbf{u}^\top \mathbf{x}_{\mathbf{u}} + \frac{m}{2}\|\mathbf{x}_{\mathbf{v}} - \mathbf{x}_{\mathbf{u}}\|_2^2.
        $$
        同理, 对称地以 $\mathbf{v}$ 为最优点、代入 $\mathbf{y} = \mathbf{x}_{\mathbf{u}}$ 可以得到
        $$
        f(\mathbf{x}_{\mathbf{u}}) - \mathbf{v}^\top \mathbf{x}_{\mathbf{u}} \geq f(\mathbf{x}_{\mathbf{v}}) - \mathbf{v}^\top \mathbf{x}_{\mathbf{v}} + \frac{m}{2}\|\mathbf{x}_{\mathbf{u}} - \mathbf{x}_{\mathbf{v}}\|_2^2.
        $$
        从而将两式相加, 化简整理有
        $$
        (\mathbf{u} - \mathbf{v})^\top (\mathbf{x}_{\mathbf{u}} - \mathbf{x}_{\mathbf{v}}) \geq m \|\mathbf{x}_{\mathbf{u}} - \mathbf{x}_{\mathbf{v}}\|_2^2.
        $$
        再根据 Cauchy-Schwarz 不等式, 有:
        $$
        \|\mathbf{u} - \mathbf{v} \| \|\mathbf{x}_\mathbf{u} - \mathbf{x}_{\mathbf{v}}\|_2 \geq m \|\mathbf{x}_{\mathbf{u}} - \mathbf{x}_{\mathbf{v}}\|_2^2 \iff \|\mathbf{x}_{\mathbf{u}} - \mathbf{x}_{\mathbf{v}}\|_2 \leq \frac{1}{m} \|\mathbf{u} - \mathbf{v} \|        
        $$
        最后根据 Fenchel conjugate 的性质 1,2, 有 $\mathbf{x}_{\mathbf{u}} = \nabla f^*(\mathbf{u})$, $\mathbf{x}_{\mathbf{v}} = \nabla f^*(\mathbf{v})$. 故命题成立.

    - $(\Leftarrow)$ 由于 $f^*(\mathbf{z})$ 是 $\frac{1}{m}$-Lipschitz smooth, 故对任意参考点 $\mathbf{x}$, $f^*(\mathbf{z}) - \mathbf{x}^\top \mathbf{z}$ 同样也是 $\frac{1}{m}$-Lipschitz smooth 的. 因此, 对于任意 $\mathbf{u}, \mathbf{v} \in \operatorname{dom}(f^*)$, 有:
        $$
        f^*(\mathbf{u}) - \mathbf{x}^\top \mathbf{u} \leq f^*(\mathbf{v}) - \mathbf{x}^\top \mathbf{v} + \langle \nabla f^*(\mathbf{v}) - \mathbf{x}, \mathbf{u} - \mathbf{v} \rangle + \frac{1}{2m}\|\mathbf{u} - \mathbf{v}\|_2^2.
        $$
        由上式 pointwise 成立, 故对左右两侧同时求关于 $\mathbf{u}$ 的最小值, 有
        $$
        \min\text{RHS} = f^*(\mathbf{v}) - \mathbf{x}^\top \mathbf{v} - \frac{m}{2} \|\nabla f^*(\mathbf{v}) - \mathbf{x}\|_2^2,
        $$
        其最优值就是通过直接求导可得. 以及根据 Fenchel conjugate 的定义, 以及性质 $(f^*)^* = f$, 有:
        $$
        \min \text{LHS} = \min_{u} \{f^*(\mathbf{u}) - \mathbf{x}^\top \mathbf{u}\} = -f(\mathbf{x})
        $$
        故我们能得到不等关系:
        $$
        f(\mathbf{x}) \geq \mathbf{x}^\top \mathbf{v} - f^*(\mathbf{v}) + \frac{m}{2} \|\nabla f^*(\mathbf{v}) - \mathbf{x}\|_2^2.
        $$
        进一步, 取参考点 $\mathbf{y} = \nabla f^*(\mathbf{v})$, 则根据 conjugate 的两个事实: (a) $\mathbf{y} = \nabla f^*(\mathbf{v}) \iff \mathbf{v} = \nabla f(\mathbf{y})$; (b) Fenchel-Young Inequality: $f^*(\mathbf{v}) =\mathbf{y}^\top \nabla f(\mathbf{y}) - f(\mathbf{y})$ (该不等式在 $\nabla f(\mathbf{y}) = \mathbf{v}$ 时取等). 因此将上述两个事实代入不等关系中, 即有:
        $$
        f(\mathbf{x}) \geq (\mathbf{x} - \mathbf{y})^\top \nabla f(\mathbf{y}) + f(\mathbf{y}) + \frac{m}{2} \|\mathbf{x} - \mathbf{y}\|_2^2.
        $$
        即 $f$ 是 $m$-stongly convex 
         
    $\square$  



### Dual First-Order Methods

***Intuition***

考虑等式约束凸优化问题:
$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}) \quad \text{s.t.} \quad \mathbf{A}\mathbf{x} = \mathbf{b}
$$
其中 $f$ 是一个凸函数, $\mathbf{A} \in \mathbb{R}^{m \times n}$ 是一个矩阵, $\mathbf{b} \in \mathbb{R}^m$ 是一个向量.   

- 考虑其 Lagrangian 函数:
    $$
    L(\mathbf{x}, \mathbf{u}) = f(\mathbf{x}) + \mathbf{u}^\top (\mathbf{A}\mathbf{x} - \mathbf{b})
    $$
    其中 $\mathbf{u} \in \mathbb{R}^m$ 是 对偶变量. 
- 对应的对偶函数为:
    $$
    g(\mathbf{u}) = \inf_{\mathbf{x} \in \mathbb{R}^n} L(\mathbf{x}, \mathbf{u}) = \inf_{\mathbf{x} \in \mathbb{R}^n} \left( f(\mathbf{x}) + \mathbf{u}^\top (\mathbf{A}\mathbf{x} - \mathbf{b}) \right) = \inf_{\mathbf{x} \in \mathbb{R}^n} \left\{f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u})^\top \mathbf{x}  \right\} - \mathbf{u}^\top \mathbf{b}.
    $$

***Gradient Information of Dual Function***

对于这类性质良好的凸优化问题 (凸且约束为 affine), 则只要约束本身是可行的, strong duality 就会自动满足. 因此求解原问题的最优解, 等价于求解对偶问题 $\max_{\mathbf{u} \in \mathbb{R}^m} g(\mathbf{u})$ 的最优解. 故很自然地考虑用梯度法等一系列一阶方法来求解对偶问题. 不过由于 $g$ 本身是通过 inf 定义的, 因此需要更仔细分析其具体性质. 

$f$ 的 Fenchel conjugate 定义为:
$$
f^*(\mathbf{y}) = \sup_{\mathbf{x} \in \mathbb{R}^n} \left( \mathbf{y}^\top \mathbf{x} - f(\mathbf{x}) \right)  \iff  -f^*(\mathbf{y}) = \inf_{\mathbf{x} \in \mathbb{R}^n} \left( f(\mathbf{x}) - \mathbf{y}^\top \mathbf{x} \right), \quad  \mathbf{y} \in \mathbb{R}^n .
$$
- 因此, 可以根据 Fenchel conjugate 的定义, 将对偶函数 $g$ 表示为:
    $$
    g(\mathbf{u}) = -f^*(-\mathbf{A}^\top \mathbf{u}) - \mathbf{u}^\top \mathbf{b} = \inf_{\mathbf{x} \in \mathbb{R}^n} \left\{f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u})^\top \mathbf{x}  \right\} - \mathbf{u}^\top \mathbf{b}.
    $$
- 若定义 $\mathbf{x}^\star \in \arg\min_{\mathbf{x} \in \mathbb{R}^n} \{f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u})^\top \mathbf{x}\}$, 则根据 Preliminary 中的两条性质, 立即有 $\mathbf{x}^\star \in \partial f^*(-\mathbf{A}^\top \mathbf{u})$ (注意这里的 subgradient 是针对 $f^*$ 的 input 整体的). 若进一步表示成关于 $\mathbf{u}$ 的 subgradient, 则有 $-\mathbf{A}\mathbf{x}^\star \in \partial_\mathbf{u} f^*(-\mathbf{A}^\top \mathbf{u})$, 或等价地, $\mathbf{A}\mathbf{x}^\star \in \partial_\mathbf{u} f^*(-\mathbf{A}^\top \mathbf{u})$. 又知 $\partial_\mathbf{u} (-\mathbf{u}^\top \mathbf{b}) = \{-\mathbf{b}\}$. 因此, 将两个 subgradient 相加, 即可得到最终结论:
    $$
    \boxed{
    \mathbf{A}\mathbf{x}^\star - \mathbf{b} \in \partial g(\mathbf{u}), \quad \text{where} \quad \mathbf{x}^\star \in \arg\min_{\mathbf{x} \in \mathbb{R}^n} \left\{f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u})^\top \mathbf{x}\right\}}
    $$
    - 该式给我们的启发是: 对于 dual gradient 的计算, 不需要显式进行求导, 而只需要求解由给定  $\mathbf{u}$ 所决定的 primal problem 的最优解. 

***Dual Gradient Ascent***

上式给我们提供了关于对偶函数的梯度信息. 因此全部的一阶方法都可以以此为基础进行设计. 例如最基本的 Dual Gradient Ascent 方法.  假设 $f$ 是 strictly convex 的. 则 (1) $f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u})^\top \mathbf{x}$ 具有唯一解; (2) dual function $g(\mathbf{u})$ 是可微的, 即 $\nabla g(\mathbf{u}) = -\mathbf{A}\mathbf{x}^\star + \mathbf{b}$. 故可以总结 **Dual Gradient Ascent** 方法. 初始化一个对偶变量 $\mathbf{u}^{(0)}$ 后, 则可以依次开始对 $k = 1, 2, \ldots$ 进行迭代:
$$
\begin{aligned}
\mathbf{x}^{(k)} &= \arg\min_{\mathbf{x} \in \mathbb{R}^n} \left\{f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u}^{(k-1)})^\top \mathbf{x}\right\} \\
\mathbf{u}^{(k)} &= \mathbf{u}^{(k-1)} + t_k (\mathbf{A}\mathbf{x}^{(k)} - \mathbf{b})
\end{aligned}
$$
其中 $t_k$ 是 step size, 可以通过正常的 line search 等方法进行选择.
- 第一步 primal update, 就是最小化在当前迭代下原问题 $\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})$ 的 Lagrangian (只相差了一个与优化变量 $\mathbf{x}$ 无关的常数项 $-\mathbf{u}^\top \mathbf{b}$). 通过该步, 就得到了当前迭代下的 dual function. 并且证明出, 最优点 $\mathbf{x}^{(k)}$ 对应的 $\mathbf{A}\mathbf{x}^{(k)} - \mathbf{b}$ 就是当前迭代下 dual function 的梯度.
- 第二步 dual update, 就是根据第一步得到的梯度信息, 对 dual function 进行梯度上升, 使得 dual function 的值不断增加, 从而逐渐逼近其最优值.

在 strong duality 被满足的条件下, 当求得 $\mathbf{u}^\star = \arg\max g(\mathbf{u})$ 时, 由此诱导的
$$
\mathbf{x}^\star(\mathbf{u}) = \arg\min_{\mathbf{x} \in \mathbb{R}^n} \left\{f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u}^\star)^\top \mathbf{x}\right\}
$$
将自动成为原问题的最优解. 



***Convergence Analysis***

Dual gradient method 只是 gradient method 应用在对偶问题上的一个特例. 因此其收敛性符合 gradient method 的正常分析. 例如, 
- 若优化目标 $g$ 是凸且 $L$-smooth 的, 则当 step size 选择为 $t_k = 1/L$ 时, 有 sublinear 的收敛率 $\mathcal{O}(1/\varepsilon)$, $\varepsilon$ 是目标精度. 
- 若目标是 $\mu$-strongly convex 且 $L$-smooth 的, 则 GD 方法是 linear convergence 的, 即当 step size 选择为 $t_k = \frac{2}{\mu+L}$ 时, 有 $\mathcal{O}(\log(1/\varepsilon))$ 的收敛率. 

因此, 根据上述原函数与 conjugate 的关系, 可以得到如下结论:
- 若 $f$ 是 $m$-strongly convex 的, 则 $f^*$ 是 $\frac{1}{m}$-smooth 的, 因此该 dual ascent 是 sublinear 收敛的, 并且最优步长 $t_k = m$.
- 若 $f$ 是 $m$-strongly convex 且 $M$-smooth 的, 则 $f^*$ 是 $\frac{1}{m}$-smooth 且 $\frac{1}{M}$-strongly convex 的 (相当于同时将定理反过来用了一次), 因此该 dual ascent 是 linear 收敛的, 并且最优步长 $t_k = \frac{2}{\frac{1}{m} + \frac{1}{M}}$.

不过更严谨地讲, 这里我们的目标函数事实上为 $g(\mathbf{u}) = -f^*(-\mathbf{A}^\top \mathbf{u}) - \mathbf{u}^\top \mathbf{b}$, 因此需要进一步考虑 $\mathbf{A}$ 的影响. 根据算子范数的定义, 可以得到
$$
\begin{aligned}
\|\nabla g(\mathbf{u}) - \nabla g(\mathbf{v})\|_2 &= \|\mathbf{A}\nabla f^*(-\mathbf{A}^\top \mathbf{u}) - \mathbf{A}\nabla f^*(-\mathbf{A}^\top \mathbf{v})\|_2 \\
&\leq \sigma_{\max}(\mathbf{A}) \|\nabla f^*(-\mathbf{A}^\top \mathbf{u}) - \nabla f^*(-\mathbf{A}^\top \mathbf{v})\|_2 \\
&\leq \frac{\sigma_{\max}(\mathbf{A})}{m} \|-\mathbf{A}^\top \mathbf{u} + \mathbf{A}^\top \mathbf{v}\|_2 \\
&\leq \frac{\sigma_{\max}^2(\mathbf{A})}{m} \|\mathbf{u} - \mathbf{v}\|_2
\end{aligned}
$$
其中 $\sigma_{\max}(\mathbf{A})$ 是 $\mathbf{A}$ 的最大奇异值. 因此各步长等还需要根据具体矩阵的条件数进行调整.

## Dual Decomposition

### Distributed Optimization via Dual Decomposition

***Dual Decomposition for Equality Constraints***

上述的 Dual Gradient Ascent 方法还只是关于对偶的一个最基本的应用. 其更为核心的应用是将一个复杂问题进行分解, 从而实现并行分布式优化. 

考虑如下 **目标得分, 约束耦合** 的优化问题:
$$
\begin{aligned}
\min_{\mathbf{x}} &\quad \sum_{i=1}^B f_i(\mathbf{x}_i) \\
\text{s.t.} &\quad \mathbf{A}\mathbf{x} = \mathbf{b}
\end{aligned}
$$
其中 $\mathbf{x} = (\mathbf{x}_1, \ldots, \mathbf{x}_B) \in \mathbb{R}^n$ 是一个 block vector, 其中每一个 block $\mathbf{x}_i \in \mathbb{R}^{n_i}$ 是一个子向量, $\sum n_i = n$, 并且每个 $f_i$ 仅依赖于 $\mathbf{x}_i$. 然而其约束却是耦合的, 即 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 中 $\mathbf{A}$ 是一个 $m \times n$ 的矩阵, 其每一行都可能同时依赖于多个 block $\mathbf{x}_i$.

因此我们可以 accordingly 地对 $\mathbf{A}$ 进行 block partition, 即 $\mathbf{A} = [\mathbf{A}_1, \ldots, \mathbf{A}_B]$, 其中 $\mathbf{A}_i$ 是一个 $m \times n_i$ 的矩阵. 从而可以将原问题表示为:
$$
\begin{aligned}
\min_{\mathbf{x}} &\quad \sum_{i=1}^B f_i(\mathbf{x}_i) \\
\text{s.t.} &\quad \sum_{i=1}^B \mathbf{A}_i \mathbf{x}_i = \mathbf{b}
\end{aligned}
$$

对该优化问题套用前面的 dual gradient ascent 方法, 则在每一次迭代中, 其 primal update 的步骤为:
$$
\begin{aligned}
\mathbf{x}^{(k)} &= \arg\min_{\mathbf{x} \in \mathbb{R}^n} \left\{f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u}^{(k-1)})^\top \mathbf{x}\right\} \\
&= \arg\min_{\mathbf{x} \in \mathbb{R}^n} \left\{\sum_{i=1}^B \left(f_i(\mathbf{x}_i) + (\mathbf{A}_i^\top \mathbf{u}^{(k-1)})^\top \mathbf{x}_i\right)\right\} \\
\end{aligned}
$$

而这里, 观察到其中的每一项 $f_i(\mathbf{x}_i) + (\mathbf{A}_i^\top \mathbf{u}^{(k-1)})^\top \mathbf{x}_i$ 仅依赖于 $\mathbf{x}_i$, 因此这 $B$ 项独立解耦的最优化问题就可以被独立各自分开求解, 从而实现并行分布式优化. 这就是 Dual Decomposition 的核心思想. 故其 primal update 的步骤可以被改写为:
$$
\begin{aligned}
\mathbf{x}_i^{(k)} &= \arg\min_{\mathbf{x}_i \in \mathbb{R}^{n_i}} \left\{f_i(\mathbf{x}_i) + (\mathbf{A}_i^\top \mathbf{u}^{(k-1)})^\top \mathbf{x}_i\right\}, \quad i = 1, \ldots, B\\
\mathbf{u}^{(k)} &= \mathbf{u}^{(k-1)} + t_k \left(\sum_{i=1}^B \mathbf{A}_i \mathbf{x}_i^{(k)} - \mathbf{b}\right)
\end{aligned}
$$

***Dual Decomposition for Inequality Constraints (Projected Subgradient Approach)***

若进一步将上述等式约束改为不等式约束, 即
$$
\begin{aligned}
\min_{\mathbf{x}} &\quad \sum_{i=1}^B f_i(\mathbf{x}_i) \\
\text{s.t.} &\quad \sum_{i=1}^B \mathbf{A}_i \mathbf{x}_i \leq \mathbf{b}
\end{aligned}
$$

则整个的 Dual Ascent 方法的框架仍然是一样的, 只是需要在每一次迭代中, 对更新后的 $\mathbf{u}^{(k)}$ 进行一个 projection, 即
$$
\begin{aligned}
\mathbf{x}_i^{(k)} &= \arg\min_{\mathbf{x}_i \in \mathbb{R}^{n_i}} \left\{f_i(\mathbf{x}_i) + (\mathbf{A}_i^\top \mathbf{u}^{(k-1)})^\top \mathbf{x}_i\right\}, \quad i = 1, \ldots, B\\
\mathbf{u}^{(k)} &= \Pi_{\mathbb{R}_+^m}\left(\mathbf{u}^{(k-1)} + t_k \left(\sum_{i=1}^B \mathbf{A}_i \mathbf{x}_i^{(k)} - \mathbf{b}\right)\right)
\end{aligned}
$$
其中 $\Pi_{\mathbb{R}_+^m}(\cdot) = \max\{\cdot, \mathbf{0}\}$ 是一个逐元素的 projection operator, 将输入的每一个元素都投影到非负实数上.
- 从算法角度看, 该方法是 projected subgradient method 的一个特例. 在每一次迭代中, 必须要保证对应的 Lagrangian multiplier $\mathbf{u}$ 是非负的, 因此需要在每一次迭代中进行 projection. 
  

***Distribution Optimization***


这是一个 synchronous 的分布式优化算法. 其每一次迭代中, 每一个 block $\mathbf{x}_i$ 都可以分发 (broadcast) 到不同的计算节点上独立求解, 从而实现并行化. 然而在进行梯度迭代时, 需要将所有 block 的结果进行聚合 (gather), 等到收集齐全的 block 结果后才能进行下一步的迭代. 

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/dual_decomposition_master_worker_topology.png)

借此场景进一步讨论一下这种 synchronous 的分布式优化算法.
- 在许多现实场景中, 事实上并不是数据被分发到不同的计算节点上, 而是数据本身本身就天然地分布在不同的计算节点上. 例如以现实场景为例. 一共有 $B$ 个单位, 每个单位 $i$ 都有自己的本地数据, 这些数据或因隐私政策, 或因数据规模等种种原因, 无法进行集中式的存储和处理. 然而与此同时, 整个系统又想在共享约束 (由于总资源 $b$ 的分配等原因, 导致约束耦合) 的条件下, 对各自 local objective $f_i$ 的加总进行优化, 其中 $\mathbf{x}_i$ 是第 $i$ 个单位的本地决策变量. 这时, Dual Decomposition 就提供了一个非常自然的解决方案. 
- 这里梳理整个优化系统的各方角色. 
  - Master (中心): 持有对偶变量 $\mathbf{u} \in \mathbb{R}^m$, 以及全局的约束 $\mathbf{b} \in \mathbb{R}^m$. 其不持有任何具体的 block $\mathbf{x}_i$ 或 local objective $f_i$. 其主要负责在每一次迭代中, 将当前的对偶变量 $\mathbf{u}$ 分发给各个 worker, 等待收集齐全的 block 结果后, 对 $\mathbf{u}$ 进行梯度更新. 
    $$
    \mathbf{u}^{(k)} = \mathbf{u}^{(k-1)} + t_k \left(\sum_{i=1}^B \boxed{\mathbf{A}_i \mathbf{x}_i^{(k)}} - \mathbf{b}\right)
    $$
    其中 $\mathbf{A}_i \mathbf{x}_i^{(k)}$ 是每一个 worker 计算得到的 block 结果, 需要被 master 收集后才能进行下一步的迭代. 并且对于 master 来说这是一个 $m$ 维的黑盒结果, 其并不知道具体的 $\mathbf{x}_i$ 或 $f_i$ 的任何信息.
  - Worker (计算节点): 每一个 worker $i$ 持有一个 block $\mathbf{x}_i$ 和 local objective $f_i$. 其主要负责在每一次迭代中, 根据 master 分发的对偶变量 $\mathbf{u}$, 以及本地的 block $\mathbf{x}_i$ 和 local objective $f_i$, 来求解该 block 的最优化问题, 从而得到 block 结果 $\mathbf{A}_i \mathbf{x}_i^{(k)}$, 并将该结果上传给 master.
      $$
      \mathbf{x}_i^{(k)} = \arg\min_{\mathbf{x}_i \in \mathbb{R}^{n_i}} \left\{f_i(\mathbf{x}_i) + (\mathbf{A}_i^\top\underline{\mathbf{u}^{(k-1)}})^\top \mathbf{x}_i\right\}, \quad i = 1, \ldots, B
      $$

因此可以汇总该种方法的特性. 
- 劣势:
  -  **Stragger effect**. 由于每一个 block 的计算时间可能不一样, 因此在每一次迭代中, 若有节点的子问题特别困难或通行等出现问题, 那么其余的 worker 即使已经完成了计算, 也只能等着该节点完成后才能进行下一步的迭代. 因此 straggler effect 就会导致整体的效率降低.
- 优势: 
  - **通信量小**. 每轮每个 worker 只需要传输 $\mathbf{A}_i \mathbf{x}_i^{(k)}$ 和 $\mathbf{u}^{(k)}$ 两个 $m$ (即约束维度) 维的向量, 其往往远小于 block $\mathbf{x}_i$ 的维度 $n_i$. 因此通信量较小, 适合于通信受限的分布式系统.
  - **隐私保护**. 每个 worker 只需要将 $\mathbf{A}_i \mathbf{x}_i^{(k)}$ 这样的 block 结果上传给 master, 而不需要上传具体的 $\mathbf{x}_i$ 或 local objective $f_i$ 的任何信息. 节点之间同样也不需要直接进行通信. 因此该方法适合于隐私受限的分布式系统.
  - **可扩展**. 中心的计算复杂度不会因节点数的增加而增加. 

***Price Coordination Interpretation** (Vandenberghe)*

此外, 进一步沿用当前的现实例子进一步加深对于对偶问题的理解. 
- 对于每个单位 $i$, 其可以自行决定自己的 local decision $\mathbf{x}_i$, 从而得到 local cost $f_i(\mathbf{x}_i)$. 但是由于整个系统的资源是有限的, 因此每个单位的决策都必须满足一个全局的约束 $\sum_{i=1}^B \mathbf{A}_i \mathbf{x}_i \leq \mathbf{b}$.  其中 $\mathbf{b}\in \mathbb{R}^m$ 中的每一行就代表了一个资源的总量 (例如电力, 总带宽, 总预算等). 
- 对应的对偶变量 $\mathbf{u} \in \mathbb{R}^m_+$, 其就相当于每一项资源的单位价格. 回顾每个 worker 解决的子问题
    $$
    \min_{\mathbf{x}_i \in \mathbb{R}^{n_i}} \left\{f_i(\mathbf{x}_i) + \mathbf{u}^\top \mathbf{A}_i \mathbf{x}_i\right\}
    $$
    其第二项 $\sum_j u_j (\mathbf{A}_i \mathbf{x}_i)_j$ 就相当于每个单位 $i$ 需要为自己使用的各项资源的用量 $(\mathbf{A}_i \mathbf{x}_i)_j$, 以单价 $u_j$ 进行支付的成本. 因此, 每个单位 $i$ 在做决策时, 不仅要考虑自己的 local cost $f_i(\mathbf{x}_i)$, 还要考虑自己使用的资源所带来的成本 $\mathbf{u}^\top \mathbf{A}_i \mathbf{x}_i$. 


- 而 master 的更新步骤
    $$
    \mathbf{u}^{(k)} = \Pi_{\mathbb{R}_+^m}\left(\mathbf{u}^{(k-1)} + t_k \left(\sum_{i=1}^B \mathbf{A}_i \mathbf{x}_i^{(k)} - \mathbf{b}\right)\right)
    $$
    可以理解为一个价格调整的过程. 定义 slack variable $\mathbf{s} =\mathbf{b} -  \sum_{i=1}^B \mathbf{A}_i \mathbf{x}_i$,  其中 $\mathbf{s} \in \mathbb{R}^m$ 中的每一项 $s_j$ 就代表了第 $j$ 个资源的剩余量. 当 $s_j < 0$, 就说明该项资源已经被过度使用了, 因此 master 会以 $t_k$ 的幅度增加该资源的价格 $u_j$, 从而在下一轮迭代中, 促使各个单位减少对该资源的使用. 反之亦然. 此外, 整个系统还会加上一个价格触底的保护机制, 即 $u_j$ 不会被调整到负数 (倒贴钱), 从而保证了价格的合理性.

## Augmented Lagrangian Method (ALM) / Method of Multipliers

ALM 是一类解决有约束问题的经典方法. 其既可以是在 Dual Ascent 的视角出发, 视作是其对 primal 的一种 regularization; 可以看作是对 penalty method 的一种改进, 通过引入对偶变量来避免 penalty method 中的数值不稳定问题. ALM 的细致分析将单独进行展开, 这里现只单纯在 dual decomposition 的基础上, 介绍一下 ALM 的基本思想.

### Intuition of ALM

回忆在前面的 Dual Ascent 中, 若对偶目标函数期望达到线性收敛, 则需要对偶目标函数光滑且强凸, 这对应着原函数需要同时满足强凸和光滑.  因此, 一种理解 ALM 的 motivation 的思路就是, 通过在 primal problem 中引入一个 quadratic penalty term 来增强原函数的强凸性, 从而放松对于原函数 $f$ 的要求.  

对于原始问题
$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} &\quad f(\mathbf{x}) \\
\text{s.t.} &\quad \mathbf{A}\mathbf{x} = \mathbf{b}
\end{aligned}
$$
其对应的 ALM 的 primal 为:
$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} &\quad f(\mathbf{x}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 \\
\text{s.t.} &\quad \mathbf{A}\mathbf{x} = \mathbf{b}
\end{aligned}
$$
其中 $\rho > 0$ 是一个 penalty parameter. 


可以看到, 这两个问题在可行域上是完全等价的. 然而通过额外引入的二次项, 使得 ALM 的 primal objective function 具有更好的性质. 
- 例如, 考虑其 Hessian:
    $$
    \nabla^2 f(\mathbf{x}) + \rho \mathbf{A}^\top \mathbf{A}
    $$
    其中 $\rho \mathbf{A}^\top \mathbf{A} \succ 0$ 只要 $\mathbf{A}$ 是 full column rank 的. 这也就使得 primal objective function 不论原函数 $f$ 的性质如何, 都具有强凸性. 

### ALM as Dual Ascent of Augmented Primal Problem

故对 Augmented primal problem 进行 dual decomposition, 则其 primal update 的步骤为:
$$
\begin{aligned}
\mathbf{x}^{(k)} &= \arg\min_{\mathbf{x} \in \mathbb{R}^n} \left\{f(\mathbf{x}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 + (\mathbf{A}^\top \mathbf{u}^{(k-1)})^\top \mathbf{x}\right\} \\
\mathbf{u}^{(k)} &= \mathbf{u}^{(k-1)} + \rho \cdot(\mathbf{A}\mathbf{x}^{(k)} - \mathbf{b})
\end{aligned}
$$

注意到, 这里的步长 $t_k$ 已经被固定为 $\rho$. 下将证明, 这样的选择会恰好让每次的 dual 迭代点满足原始问题的 Stationarity 最优条件:
- 由于 $\mathbf{x}^{(k)}$ 是 primal update 的最优点, 则其满足该问题的 KKT 条件, 即
    $$
    \begin{aligned}
    \mathbf{0} &\in \partial f(\mathbf{x}^{(k)}) + \rho \mathbf{A}^\top (\mathbf{A}\mathbf{x}^{(k)} - \mathbf{b}) + \mathbf{A}^\top \mathbf{u}^{(k-1)} \\
    & = \partial f(\mathbf{x}^{(k)}) + \mathbf{A}^\top \underbrace{\left(\mathbf{u}^{(k-1)} + \rho (\mathbf{A}\mathbf{x}^{(k)} - \mathbf{b})\right)}_{\text{dual update}} \\
    & = \partial f(\mathbf{x}^{(k)}) + \mathbf{A}^\top \mathbf{u}^{(k)}
    \end{aligned}
    $$
- 由上式可见, 每一次迭代的 dual variable $\mathbf{u}^{(k)}$ 只要按照步长 $\rho$ 进行更新, 就能保证 primal variable $\mathbf{x}^{(k)}$ 满足原始问题的 stationarity 条件. 

同时另一方面, dual update 也起到了对于 primal variable 的 feasibility 的促进作用. 当且仅当 primal variable $\mathbf{x}^{(k)}$ 满足可行性条件 $\mathbf{A}\mathbf{x}^{(k)} = \mathbf{b}$ 时, dual variable $\mathbf{u}^{(k)}$ 才会保持不变. 

总的而言, 考虑原始问题 $\min f(\mathbf{x}) \text{ s.t. } \mathbf{A}\mathbf{x} = \mathbf{b}$ 的 KKT 条件:
- Stationary: $\mathbf{0} \in \partial f(\mathbf{x}) + \mathbf{A}^\top \mathbf{u}$ 这在 ALM 中每一个 primal update 的步骤中都被满足了.
- Feasibility: $\mathbf{A}\mathbf{x} = \mathbf{b}$ 这在 ALM 中通过 dual update 中 asymptotically 满足.

因此通过 ALM 的 primal update 和 dual update 的交替迭代, 就能同时满足原始问题的 stationarity 和 feasibility 条件, 从而最终收敛到原始问题的最优解. 进而, 只要 $f$ 满足一些 mild conditions 等, 就可以满足 strong duality, 从而保证 ALM 的收敛性.  这相比于 Dual Ascent 来说, 其对于原函数 $f$ 的要求被大大放宽. 

然而, ALM 的 primal update 中引入了一个 quadratic penalty term, 考虑前述的分块结构, 可以观察到
$$
\begin{aligned}
\frac{\rho}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 &= \frac{\rho}{2}\left\|\sum_{i=1}^B \mathbf{A}_i \mathbf{x}_i - \mathbf{b}\right\|_2^2 \\
&= \frac{\rho}{2}\left[\sum_{i} \|\mathbf{A}_i \mathbf{x}_i\|_2^2 + 2\sum_{i < j} (\mathbf{A}_i \mathbf{x}_i)^\top (\mathbf{A}_j \mathbf{x}_j) - 2\sum_i (\mathbf{A}_i \mathbf{x}_i)^\top \mathbf{b} + \|\mathbf{b}\|_2^2\right]
\end{aligned}
$$
而这里由于 $\sum_{i<j}\cdot$ 这一项的存在, 导致不同 block 之间的耦合, 从而无法进行分解. 因此 ALM 的代价是破坏了 decomposability, 从而无法进行分布式优化. 这也为后文的 ADMM 的设计提供了一个重要的启发.


## Alternating Direction Method of Multipliers (ADMM)

ADMM 同样也是在有约束优化领域的重要经典方法. 其主体内容也将详细的进行展开分析. 这里同样也只是站在上面提到的 ALM 的基础上, 尝试通过进一步的改进, 整合 ALM 和 Dual Decomposition 的优势.

### Intuition of ADMM

考虑一个两块分块的优化问题:
$$
\begin{aligned}
\min_{\mathbf{x}, \mathbf{z}} &\quad f(\mathbf{x}) + g(\mathbf{z}) \\
\text{s.t.} &\quad \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} = \mathbf{c}
\end{aligned}
$$

这样的问题在很多现实场景中都非常常见, 例如:
- 在 Lasso regression 等中, $f$ 是损失函数, $g$ 是正则项, 约束 $\mathbf{x} = \mathbf{z}$.
- 在复合优化中, $f$ 是 smooth function, $g$ non-smooth 但 proximal-friendly.
- 在分布式优化中, $f$ 是 local objective, $g$ 是 global objective, 约束 $\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} = \mathbf{c}$ 则是耦合约束.

沿用前面 ALM 的思路, 对其进行 Augmentation:
$$
\begin{aligned}
\min_{\mathbf{x}, \mathbf{z}} &\quad f(\mathbf{x}) + g(\mathbf{z}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c}\|_2^2 \\
\text{s.t.} &\quad \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} = \mathbf{c}
\end{aligned}
$$
这个优化问题对应的 Lagrangian function 为:
$$
L_\rho(\mathbf{x}, \mathbf{z}, \mathbf{u}) = f(\mathbf{x}) + g(\mathbf{z}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c}\|_2^2 + \mathbf{u}^\top (\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c})
$$

回顾, 若沿用 ALM 的 primal update 的步骤, 则需要同时对 $\mathbf{x}$ 和 $\mathbf{z}$ 进行联合优化, 即
$$
\begin{aligned}
(\mathbf{x}^{(k)}, \mathbf{z}^{(k)}) &= \arg\min_{\mathbf{x}, \mathbf{z}} L_\rho(\mathbf{x}, \mathbf{z}, \mathbf{u}^{(k-1)}) \\
&= \arg\min_{\mathbf{x}, \mathbf{z}} \left\{f(\mathbf{x}) + g(\mathbf{z}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c}\|_2^2 + (\mathbf{u}^{(k-1)})^\top (\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c})\right\}
\end{aligned}
$$
然而同样地, 由于 $\frac{\rho}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c}\|_2^2$ 这一项的存在, 导致 $\mathbf{x}$ 和 $\mathbf{z}$ 之间的耦合, 从而无法进行分解.

因此 ADMM 的核心思想就是, 不再联合求解 primal Lagrangian 的最小值, 而是通过 alternating optimization 的方式, 交替地对 $\mathbf{x}$ 和 $\mathbf{z}$ 进行优化. 具体而言, ADMM 的 primal update 的步骤为:
$$
\begin{aligned}
\mathbf{x}^{(k)} &= \arg\min_{\mathbf{x}} L_\rho(\mathbf{x}, \mathbf{z}^{(k-1)}, \mathbf{u}^{(k-1)}), \\
\mathbf{z}^{(k)} &= \arg\min_{\mathbf{z}} L_\rho(\mathbf{x}^{(k)}, \mathbf{z}, \mathbf{u}^{(k-1)}), \\
\end{aligned}
$$
然后再进行 dual update:
$$
\mathbf{u}^{(k)} = \mathbf{u}^{(k-1)} + \rho \cdot (\mathbf{A}\mathbf{x}^{(k)} + \mathbf{B}\mathbf{z}^{(k)} - \mathbf{c}).
$$

> 注意, 这里的 decomposition 是通过交替更新实现的, 应当区分其与 Dual Decomposition 中的 parallel decomposition. 

因此, ADMM 对于一般的温和条件
- 不要求 $A, B$ 是 full column rank
- 不要求 $f, g$ 是强凸或可微的
-  只需要是正常凸, 约束有可行解等即可

都能保证收敛到原始问题的最优解, 并且同时和步长 $\rho$ 无关, 其只会影响收敛的速度但不会影响收敛的结果. 并且一般有如下收敛保证:
- Feasibility Convergence: $\mathbf{r}^{(k)} = \mathbf{A}\mathbf{x}^{(k)} + \mathbf{B}\mathbf{z}^{(k)} - \mathbf{c} \to \mathbf{0}$.
- Objective Convergence: $f(\mathbf{x}^{(k)}) + g(\mathbf{z}^{(k)}) \to f^\star + g^\star$.
  - 注意, 这里只保证了 objective value 的收敛, 并不保证 primal variable $\mathbf{x}^{(k)}, \mathbf{z}^{(k)}$ 的收敛. 例如, 在某些 degenerate 的问题中, 可能存在多个 primal optimal solution, 从而导致 primal variable 的震荡, 但其 objective value 却是收敛的. 
- Dual Convergence: $\mathbf{u}^{(k)} \to \mathbf{u}^\star$.

### Scaled Form of ADMM

Scaled Form ADMM 和 ADMM 在本质上完全等价. 其事实上就是在对 $L_\rho(\mathbf{x}, \mathbf{z}, \mathbf{u})$ 进行适当的变量替换进行配方, 以得到一个更为简洁的形式.

通常引入 scaled dual variable $\mathbf{w} := \mathbf{u}/\rho$. 则 Lagrangian 可以写为:
$$
L_\rho(\mathbf{x}, \mathbf{z}, \mathbf{w}) = f(\mathbf{x}) + g(\mathbf{z}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c} + \mathbf{w}\|_2^2 - \frac{\rho}{2}\|\mathbf{w}\|_2^2
$$
因此 ADMM 的更新公式可以用 scaled dual variable 来表示, 从而得到更为简洁的形式:
$$
\begin{aligned}
\mathbf{x}^{(k)} &= \arg\min_{\mathbf{x}} f(\mathbf{x}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z}^{(k-1)} - \mathbf{c} + \mathbf{w}^{(k-1)}\|_2^2, \\
\mathbf{z}^{(k)} &= \arg\min_{\mathbf{z}} g(\mathbf{z}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x}^{(k)} + \mathbf{B}\mathbf{z} - \mathbf{c} + \mathbf{w}^{(k-1)}\|_2^2, \\
\mathbf{w}^{(k)} &= \mathbf{w}^{(k-1)} + (\mathbf{A}\mathbf{x}^{(k)} + \mathbf{B}\mathbf{z}^{(k)} - \mathbf{c})
\end{aligned}
$$

不过确实在进行缩放之后, $\mathbf{w}$ 的起到了 running sum of residuals 的作用, 用来记录历史上所有约束违反的总和.

### Example: Alternating Projections

Alternating Projections 是一种经典的 ADMM 应用场景. 

给定两个凸集 $C, D \subseteq \mathbb{R}^n$, 优化的目标是寻找两几个的交集中的点. 若引入 indicator function:
$$
I_\Omega(\boldsymbol{x}) = 
\begin{cases}
0, & \boldsymbol{x} \in \Omega \\
\infty, & \boldsymbol{x} \notin \Omega
\end{cases}
$$
则可以将寻找交集中的点的问题转化为如下优化问题:
$$
\min_{\boldsymbol{x}} I_C(\boldsymbol{x}) + I_D(\boldsymbol{x}).
$$

尝试使用 ADMM 解决该问题. 对应标准形式, 则有
$$
\min_{\boldsymbol{x}, \boldsymbol{y}} f(\boldsymbol{x}) + g(\boldsymbol{y}) \quad \text{s.t.} \quad \boldsymbol{x} = \boldsymbol{y}.
$$
其中 $f(\boldsymbol{x}) = I_C(\boldsymbol{x}), g(\boldsymbol{y}) = I_D(\boldsymbol{y})$.

代入 ADMM 的更新公式, 则有
$$
\begin{aligned}
\boldsymbol{x}^{(k)} &= \arg\min_{\boldsymbol{x}} I_C(\boldsymbol{x}) + \frac{\rho}{2}\|\boldsymbol{x} - \boldsymbol{y}^{(k-1)} + \boldsymbol{w}^{(k-1)}\|_2^2, \\
\boldsymbol{y}^{(k)} &= \arg\min_{\boldsymbol{y}} I_D(\boldsymbol{y}) + \frac{\rho}{2}\|\boldsymbol{x}^{(k)} - \boldsymbol{y} + \boldsymbol{w}^{(k-1)}\|_2^2, \\
\boldsymbol{w}^{(k)} &= \boldsymbol{w}^{(k-1)} + (\boldsymbol{x}^{(k)} - \boldsymbol{y}^{(k)}).
\end{aligned}
$$

观察这里的更新公式, 可以发现
$$
\arg\min_{\boldsymbol{x}} I_C(\boldsymbol{x}) + \frac{\rho}{2}\|\boldsymbol{x} - \boldsymbol{y}^{(k-1)} + \boldsymbol{w}^{(k-1)}\|_2^2 
\iff
\arg\min_{\boldsymbol{x} \in C} \frac{\rho}{2}\|\boldsymbol{x} - \boldsymbol{y}^{(k-1)} + \boldsymbol{w}^{(k-1)}\|_2^2.
$$
而后者恰恰是将点 $\boldsymbol{y}^{(k-1)} - \boldsymbol{w}^{(k-1)}$ 投影到集合 $C$ 上的数学定义. 对于 $ \boldsymbol{y}^{(k)}$ 同理. 因此我们的更新公式可以写作:
$$
\begin{aligned}
\boldsymbol{x}^{(k)} &= \mathcal{P}_C(\boldsymbol{y}^{(k-1)} - \boldsymbol{w}^{(k-1)}), \\
\boldsymbol{y}^{(k)} &= \mathcal{P}_D(\boldsymbol{x}^{(k)} + \boldsymbol{w}^{(k-1)}), \\
\boldsymbol{w}^{(k)} &= \boldsymbol{w}^{(k-1)} + (\boldsymbol{x}^{(k)} - \boldsymbol{y}^{(k)}).
\end{aligned}
$$

综上, 这样的 ADMM 其实就相当于交替进行集合 $C$ 和集合 $D$ 上的投影, 而投影对许多集合都有简单的闭式解, 因此极大地简化了求解过程.

若再对比对于该问题的经典 von Neumann 算法, 则其迭代公式为
$$
\boldsymbol{x}^{(k)} = \mathcal{P}_C(\boldsymbol{y}^{(k-1)}), \quad 
\boldsymbol{y}^{(k)} = \mathcal{P}_D(\boldsymbol{x}^{(k)}).
$$
其唯一的区别在于 ADMM 引入了一个 dual variable $\boldsymbol{w}$ 作为当前位置的  offset, 这体现了通过历史偏差的累积这一信息, 对当前投影进行了修正.


