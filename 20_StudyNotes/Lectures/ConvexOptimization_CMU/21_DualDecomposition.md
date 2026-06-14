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
因此这是一个 synchronous 的分布式优化算法. 其每一次迭代中, 每一个 block $\mathbf{x}_i$ 都可以分发 (broadcast) 到不同的计算节点上独立求解, 从而实现并行化. 然而在进行梯度迭代时, 需要将所有 block 的结果进行聚合 (gather), 等到收集齐全的 block 结果后才能进行下一步的迭代. 若稍微再仔细讨论一下这种分布式的特性:
- 问题: straggler effect. 由于每一个 block 的计算时间可能不一样, 因此在每一次迭代中, 若有节点的子问题特别困难或通行等出现问题, 那么其余的 worker 即使已经完成了计算, 也只能等着该节点完成后才能进行下一步的迭代. 因此 straggler effect 就会导致整体的效率降低.
- 优势: 
  - 通信量小: 每轮每个 worker 只需要上下传输一个 block 的结果. 并且事实上, 中心只需要给每个 worker 传入 $\mathbf{u} \in \mathbb{R}^m$ 该 $m$ 维 (即约束个数) 的大小, 而 worker 也只需要完整地将 $\mathbf{A}