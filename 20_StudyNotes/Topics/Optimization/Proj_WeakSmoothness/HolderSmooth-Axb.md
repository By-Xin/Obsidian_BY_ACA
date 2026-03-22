# Holder Smooth: 从 Canonical 到 Affine 模型

本节考虑一个更 general 的模型:
$$
F_{\mathbf b}(\mathbf x)
:=
\frac1p\|\mathbf A\mathbf x-\mathbf b\|_p^p,
\qquad
\mathbf A\in\mathbb R^{m\times n},\ \mathbf b\in\mathbb R^m,\ \mathbf x\in\mathbb R^n,
$$

且有 $p \in(1,2]$, 对应
$$
q = \frac p{p-1} \in [2,+\infty), ~s:= 2 - \frac{2}{p} \in (0,1].
$$

## Fenchel Representation

Generally, 定义
$$
h(\mathbf y) := \frac1q\|\mathbf y\|_q^q, \quad \mathbf{y} \in \mathbb R^m, q\in [2,+\infty).
$$

其 Fenchel conjugate 为
$$
h^*(\mathbf z) = \max_{\mathbf y}\{\langle\mathbf z,\mathbf y\rangle - h(\mathbf y)\} = \frac1p\|\mathbf z\|_p^p,
\quad \mathbf{z} \in \mathbb R^m, p \in (1,2].
$$


因此, 对应本文的具体形式, 有
$$
F_{\mathbf b}(\mathbf x) = h^*(\mathbf A\mathbf x-\mathbf b) 
= \frac{1}{p}\|\mathbf A\mathbf x-\mathbf b\|_p^p
=\max_{\mathbf y \in \mathbb{R}^m}\left\{\langle\mathbf A\mathbf x-\mathbf b,\mathbf y\rangle - \frac{1}{q}\|\mathbf y\|_q^q\right\}.
$$

***命题* (原问题的解)**: 对于 $F_{\mathbf b}(\mathbf x) = \frac{1}{p}\|\mathbf A\mathbf x-\mathbf b\|_p^p$,  其梯度满足
$$
\nabla F_{\mathbf{b}} (\mathbf{x}) = \mathbf{A}^\top \left(\text{sign}(\mathbf{A}\mathbf{x}-\mathbf{b}) \odot |\mathbf{A}\mathbf{x}-\mathbf{b}|^{p-1}\right).
$$
考虑