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
    - $(\Rightarrow)$ 记 $\mathbf{x}_{\mathbf{u}} = \arg\min_{\mathbf{x}} \{ f(\mathbf{x}) + \mathbf{u}^\top \mathbf{x} \}$, $\mathbf{x}_{\mathbf{v}} = \arg\min_{\mathbf{x}} \{ f(\mathbf{x}) + \mathbf{v}^\top \mathbf{x} \}$. 由于 $f$ 是强凸的, 故 $f(\mathbf{x}) - \mathbf{u}^\top \mathbf{x}$ 也是唯一最小值取到 $\mathbf{x}_\mathbf{u}$ 的强凸函数, 根据强凸的性质, 对于任意 $\mathbf{y} \in \operatorname{dom}(f)$ 有:
        $$
        f(\mathbf{y})  - \mathbf{u}^\top \mathbf{y} \geq f(\mathbf{x}_{\mathbf{u}}) - \mathbf{u}^\top \mathbf{x}_{\mathbf{u}} + \frac{m}{2}\|\mathbf{y} - \mathbf{x}_{\mathbf{u}}\|_2^2
        $$
        再令 $\mathbf{y} = \mathbf{x}_{\mathbf{v}}$, 则有:
        $$
        f(\mathbf{x}_{\mathbf{v}}) - \mathbf{v}^\top \mathbf{x}_{\mathbf{v}} \geq f(\mathbf{x}_{\mathbf{u}}) - \mathbf{v}^\top \mathbf{x}_{\mathbf{u}} + \frac{m}{2}\|\mathbf{x}_{\mathbf{v}} - \mathbf{x}_{\mathbf{u}}\|_2^2.
        $$
        同理, 对称地可以得到
        $$
        f(\mathbf{x}_{\mathbf{u}}) - \mathbf{u}^\top \mathbf{x}_{\mathbf{u}} \geq f(\mathbf{x}_{\mathbf{v}}) - \mathbf{v}^\top \mathbf{x}_{\mathbf{v}} + \frac{m}{2}\|\mathbf{x}_{\mathbf{u}} - \mathbf{x}_{\mathbf{v}}\|_2^2.
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


因此, 上述流程可以迭代交替进行. 初始化一个对偶变量 $\mathbf{u}^{(0)}$ 后, 则可以依次开始对 $k = 1, 2, \ldots$ 进行迭代:
- 对于给定 $\mathbf{u}^{(k-1)}$, 求解
    $$
    \mathbf{x}^{(k)} \in \arg\min_{\mathbf{x} \in \mathbb{R}^n} \left\{f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u}^{(k-1)})^\top \mathbf{x}\right\}
    $$
- 因此这时的 $\mathbf{A}\mathbf{x}^{(k)} - \mathbf{b}$ 就是对偶函数的 subgradient, 因此可以用该次梯度来更新对偶变量:
    $$
    \mathbf{u}^{(k)} = \mathbf{u}^{(k-1)} + t_k (\mathbf{A}\mathbf{x}^{(k)} - \mathbf{b})
    $$
    其中 $t_k$ 是 step size, 可以通过 line search 来确定. 


### Dual Gradient Ascent 

下正式将上述思路进行 formulation. 假设 $f$ 是 strictly convex 的. 则 (1) $f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u})^\top \mathbf{x}$ 具有唯一解; (2) dual function $g(\mathbf{u})$ 是可微的, 即 $\nabla g(\mathbf{u}) = -\mathbf{A}\mathbf{x}^\star + \mathbf{b}$. 故可以总结 **Dual Gradient Ascent** 方法: 
$$
\begin{aligned}
\mathbf{x}^{(k)} &= \arg\min_{\mathbf{x} \in \mathbb{R}^n} \left\{f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u}^{(k-1)})^\top \mathbf{x}\right\} \\
\mathbf{u}^{(k)} &= \mathbf{u}^{(k-1)} + t_k (\mathbf{A}\mathbf{x}^{(k)} - \mathbf{b})
\end{aligned}
$$

注意, 这里我们实际在做的是通过 $\mathbf{x}$ 的求解来确定 $g(\mathbf{u})$ 的梯度, 从而指导对偶函数的更新. 并且注意这里的更新为梯度上升, 我们在不断尝试提高对偶函数. 在当前 convex + affine constrain 的情况下, strong duality 将自动满足. 因此当求得 $\mathbf{u}^\star = \arg\max g(\mathbf{u})$ 时, 由此诱导的
$$
\mathbf{x}^\star(\mathbf{u}) = \arg\min_{\mathbf{x} \in \mathbb{R}^n} \left\{f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u}^\star)^\top \mathbf{x}\right\}
$$
将自动成为原问题的最优解. 