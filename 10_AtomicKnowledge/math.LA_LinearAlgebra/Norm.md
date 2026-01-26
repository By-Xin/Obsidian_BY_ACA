---
aliases: [范数, Norm, 向量范数, 矩阵范数, Lp Norm]
tags:
  - concept
  - math/linear-algebra
related_concepts:
  - "[[Matrix]]"
  - "[[Linear_Space]]"
  - "[[Regularization]]"
  - "[[Lagrange_Duality]]"
source: "线性代数基础"
---

# 范数 (Norm)

#LinearAlgebra 

## 定义

$l_k$ norm of vector $\mathbf{v}$:
$$
||\mathbf{v}||_k = (|v_1|^k+\cdots+|v_n|^k)^{1/k}
$$

## 特殊情况

- $l_0$ norm means the count of non-zero element of vector $\mathbf{v}$
- $l_\infty$ norm means the absolute value of the largest element of vector $\mathbf{v}$

## 性质

- The large $k$ is, the more that it will emphasis on the large elements and ignore the small ones. 