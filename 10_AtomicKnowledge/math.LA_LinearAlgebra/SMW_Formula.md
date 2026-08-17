---
aliases: [Sherman-Morrison-Woodbury 公式, SMW 公式, Matrix Inversion Lemma, 矩阵求逆引理]
tags:
  - concept
  - math/linear-algebra
  - math/optimization
related_concepts:
  - "[[Matrix]]"
  - "[[Cholesky_Decomposition]]"
  - "[[Rank_and_Eigenvalue]]"
  - "[[Gaussian_Process]]"
  - "[[Bayesian_Linear_Regression]]"
source: "数值线性代数; 统计优化"
---

# Sherman–Morrison–Woodbury Formula

Sherman–Morrison–Woodbury（SMW）公式（又称 *matrix inversion lemma*）用于将“低秩更新后的逆矩阵”写成“原逆矩阵 + 小矩阵求逆”的形式，常用于优化/统计推断中的快速更新。

## Statement

令：
- $A \in \mathbb{R}^{n \times n}$ 可逆
- $U \in \mathbb{R}^{n \times k}$
- $C \in \mathbb{R}^{k \times k}$ 可逆
- $V \in \mathbb{R}^{k \times n}$

则：
$$
(A + UCV)^{-1}
= A^{-1} - A^{-1}U(C^{-1} + V A^{-1} U)^{-1} V A^{-1}.
$$

（很多资料也写成 $A + UCV^\top$ 的形式；只要维度匹配即可。）

## Special case (Sherman–Morrison, rank-1)

当 $k=1$，即 $A + u v^\top$ 的秩 1 更新时：
$$
(A + u v^\top)^{-1}
= A^{-1} - \frac{A^{-1} u v^\top A^{-1}}{1 + v^\top A^{-1} u}.
$$

## Typical uses

- 在线/增量更新：对 Hessian / covariance / normal equations 的低秩修正。
- 大规模计算：把 $n\times n$ 求逆转成 $k\times k$ 求逆（$k \ll n$）。
- 推断与优化：高斯后验更新、牛顿法近似更新、拟牛顿/预条件相关推导。
