---
aliases: [Cholesky分解, Cholesky Decomposition, Cholesky Factorization]
tags:
  - concept
  - math/linear-algebra
  - math/optimization
related_concepts:
  - "[[Matrix]]"
  - "[[Positive_Definite_Matrix]]"
  - "[[Determinant]]"
  - "[[Rank_and_Eigenvalue]]"
  - "[[SMW_Formula]]"
source: "数值线性代数"
---

# Cholesky 分解

对于一个实对称正定矩阵 $A \in \mathbb{R}^{n \times n}$，Cholesky 分解提供了一种将其分解为下三角矩阵和其转置的方式。具体地，存在一个下三角矩阵 $L\in \mathbb{R}^{n \times n}$，使得：
$$
A = LL^\top
$$
Cholesky 分解的主要优点在于其计算效率和数值稳定性。通过将矩阵分解为下三角矩阵的形式，我们可以更高效地求解线性方程组和计算矩阵的逆。

当我们想要计算 $Ax = b$ 的解时，可以通过 Cholesky 分解加速计算:
1. **Cholesky 分解** $A = LL^\top$ ($\mathcal{O}(n^3/3)$):
   - 对于每个 $i = 1, \ldots, n$ (对角线元素)，计算 $L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1} L_{ik}^2}$.
   - 对于每个 $j < i$ (非对角线元素)，计算 $L_{ij} = \frac{1}{L_{jj}}(A_{ij} - \sum_{k=1}^{j-1} L_{ik} L_{jk})$.
2. **Forward Substitution** ($\mathcal{O}(n^2)$):
   - 将方程替换为 $LL^\top x = b$.
   - 令 $y = L^\top x$，则方程变为 $Ly = b$. 这是一个下三角矩阵对应的线性方程.
   - 此时可以通过前向替换求解 $y$: $y_i = \frac{1}{L_{ii}}(b_i - \sum_{j=1}^{i-1} L_{ij} y_j)$.
3. **Backward Substitution** ($\mathcal{O}(n^2)$):
   - 现在我们得到了 $y$，接下来求解 $x$.
   - 我们现在求解 $L^\top x = y$，这是一个上三角矩阵对应的线性方程.
   - 此时可以通过后向替换求解 $x$: $x_i = \frac{1}{L_{ii}}(y_i - \sum_{j=i+1}^{n} L_{ji} x_j)$.