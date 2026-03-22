# Holder Smooth: 从 Canonical 到 Affine 模型

本节考虑一个稍微 general 一些的优化问题, 即在 Canonical 模型的基础上, 将残差 $\mathbf A\mathbf x$ 扩展为 affine 模型 $\mathbf A\mathbf x-\mathbf b$. 

具体地, 令 $\mathbf{A} \in \mathbb{R}^{m\times n}$, $\mathbf{b} \in \mathbb{R}^m$, $1<p \leq 2$, 对应共轭指数 $q = \frac{p}{p-1} \in [2,+\infty)$.

我们研究如下优化问题:
$$
\min_{\mathbf x\in\mathbb R^n} F_{\mathbf b}(\mathbf x)
$$
其中
$$
F_{\mathbf b}(\mathbf x)
:=
\frac1p\|\mathbf A\mathbf x-\mathbf b\|_p^p = \max_{\mathbf y\in\mathbb R^m}\left\{\langle \mathbf A\mathbf x-\mathbf b,\mathbf y\rangle - \frac1q\|\mathbf y\|_q^q\right\}.
$$

余下的部分我们会分为如下两种情况进行讨论:
1. 对于 feasible point, 即 $\mathcal{X}^\star = \{\mathbf x\in\mathbb R^n: \mathbf A\mathbf x=\mathbf b\} \neq \emptyset$. 此时, 对于任意 $\tilde{\mathbf x}\in\mathcal{X}^\star$, 任意 $\mathbf x\in\mathbb R^n$, 都有
    $$
    \mathbf{A}\mathbf{x} - \mathbf{b}  = \mathbf{A}\mathbf{x} - \mathbf{A}\tilde{\mathbf{x}} := \mathbf{A}(\mathbf{z}),
    $$
    即相当于在 feasible point 上的 Canonical 模型. 因此, 该问题的分析与 Canonical 模型几乎完全一致, 可以直接进行推广. 

2. 