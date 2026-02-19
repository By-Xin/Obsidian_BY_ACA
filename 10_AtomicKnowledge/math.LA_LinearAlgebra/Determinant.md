---
aliases: [行列式, Determinant, 矩阵行列式]
tags:
  - concept
  - math/linear-algebra
related_concepts:
  - "[[Matrix]]"
  - "[[Jacobian_Matrix]]"
  - "[[Rank_and_Eigenvalue]]"
  - "[[Linear_Space]]"
  - "[[Cholesky_Decomposition]]"
source: "线性代数基础"
---

# 行列式 (Determinant)

#LinearAlgebra 

## 基本性质

$$ \det(A) = 1/\det(A^{-1})$$

$$ \det(J_{f^{-1}}) = 1/\det(J_f)$$

## 几何意义

- Determinant of matrix $A$ is the volume of the parallelepiped spanned by the row vectors of $A$.
- 行列式的绝对值表示线性变换对"体积"的缩放因子
- 若 $\det(A) = 0$，则矩阵 $A$ 是奇异的（不可逆）

## 相关概念

- [[Jacobian_Matrix]] - 雅可比矩阵的行列式在变量替换中起关键作用
- [[Rank_and_Eigenvalue]] - 行列式等于所有特征值的乘积