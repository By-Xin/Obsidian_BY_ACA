# Coordinate Descent

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

$f$ 的 Fenchel conjugate 定义为:
$$
f^*(\mathbf{y}) = \sup_{\mathbf{x} \in \mathbb{R}^n} \left( \mathbf{y}^\top \mathbf{x} - f(\mathbf{x}) \right)  \iff  -f^*(\mathbf{y}) = \inf_{\mathbf{x} \in \mathbb{R}^n} \left( f(\mathbf{x}) - \mathbf{y}^\top \mathbf{x} \right), \quad  \mathbf{y} \in \mathbb{R}^n .
$$

其有如下性质. 

***Property 2.*** $\mathbf{y} \in \partial f(\mathbf{x})$ 当且仅当 $\mathbf{x} \in \partial f^*(\mathbf{y})$.
- *Proof.* 由 $\mathbf{y} \in \partial f(\mathbf{x})$ 可得 $\mathbf{x} \in \arg\min_{\mathbf{z} \in \mathbb{R}^n} \left\{f(\mathbf{z}) - \mathbf{y}^\top \mathbf{z}\right\}$, 即 $f(\mathbf{x}) - \mathbf{y}^\top \mathbf{x} = \inf_{\mathbf{z} \in \mathbb{R}^n} \left\{f(\mathbf{z}) - \mathbf{y}^\top \mathbf{z}\right\}$. 根据 Fenchel conjugate 的定义, 可得 $f^*(\mathbf{y}) = \mathbf{y}^\top \mathbf{x} - f(\mathbf{x})$. 而对任意 $\mathbf{u} \in \mathbb{R}^m$,  $f^*(\mathbf{u}) = \sup_{\mathbf{z} \in \mathbb{R}^n} \left( \mathbf{u}^\top \mathbf{z} - f(\mathbf{z}) \right) \geq \mathbf{u}^\top \mathbf{x} - f(\mathbf{x})$. 从而将两式相减, 有 $f^*(\mathbf{u}) - f^*(\mathbf{y}) \geq (\mathbf{u} - \mathbf{y})^\top \mathbf{x}$, 即 $\mathbf{x} \in \partial f^*(\mathbf{y})$.


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
    \mathbf{A}\mathbf{x}^\star - \mathbf{b} \in \partial g(\mathbf{u}), \quad \text{where} \quad \mathbf{x}^\star \in \arg\min_{\mathbf{x} \in \mathbb{R}^n} \left\{f(\mathbf{x}) + (\mathbf{A}^\top \mathbf{u})^\top \mathbf{x}\right\}.
    $$