# Multiplying and Factoring Matrices

> [!quote]
>
> - Lecture 2 in the course [Matrix Methods in Data Science](https://ocw.mit.edu/courses/mathematics/18-065-matrix-methods-in-data-science-spring-2019/lecture-videos/lecture-2-multiplying-and-factoring-matrices/) by MIT OpenCourseWare.

在这个 lecture, 我们将主要考虑如下几个矩阵的分解 (factorization):
- LU factorization ($\mathbf{A} = \mathbf{L}\mathbf{U}$)
- QR factorization / Gram-Schmidt process ($\mathbf{A} = \mathbf{Q}\mathbf{R}$)
- SVD factorization ($\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$)
- Eigen-decomposition ($\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}$)
- Spectral decomposition ($\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$)    

## LU Factorization

LU 分解被广泛应用在求解线性方程组, 求逆等问题中. 先笼统地说, 给定 $\mathbf{A}\in \mathbb{R}^{m \times n}$, 如果存在一个下三角矩阵 $\mathbf{L}\in \mathbb{R}^{m \times m}$ 和一个上三角矩阵 (准确地说是 echelon form 的矩阵) $\mathbf{U}\in \mathbb{R}^{m \times n}$ 使得 $\mathbf{A} = \mathbf{L}\mathbf{U}$, 则称 $\mathbf{A}$ 可以被 LU 分解. 其中, $\mathbf{L}$ 的对角线元素通常被设定为 $1$. 其形式上展现为:
$$
\mathbf{A} = \mathbf{L}\mathbf{U} = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ l_{21} & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ l_{m1} & l_{m2} & \cdots & 1 \end{bmatrix} \begin{bmatrix} u_{11} & u_{12} & \cdots & u_{1n} \\ 0 & u_{22} & \cdots & u_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & u_{mn} \end{bmatrix}.
$$
- 其中 echelon form 的矩阵 $\mathbf{U}$ 要求:
    1. $\mathbf{U}$ 中非零行要在全零行的上方.
    2. $\mathbf{U}$ 中每一行的第一个非零元素 (称为 leading entry) 要在其上一行的 leading entry 的右侧.
    3. $\mathbf{U}$ 中每一列的 leading entry 的下方元素要全为 $0$.

先假设我们已经知道 $\mathbf{L}$ 和 $\mathbf{U}$, 那么我们可以通过如下方式来求解线性方程组 $\mathbf{A}\mathbf{x} = \mathbf{b}$:
1. 首先, 将 $\mathbf{A}$ 替换为 $\mathbf{L}\mathbf{U}$, 得到 $\mathbf{L}\mathbf{U}\mathbf{x} = \mathbf{b}$.
2. 令 $\mathbf{y} = \mathbf{U}\mathbf{x}$, 则上式可以重写为 $\mathbf{L}\mathbf{y} = \mathbf{b}$.
3. 求解 $\mathbf{L}\mathbf{y} = \mathbf{b}$: 因为 $\mathbf{L}$ 是一个下三角矩阵, 我们可以通过前向替换 (forward substitution) 来求解 $\mathbf{y}$.
4. 最后, 求解 $\mathbf{U}\mathbf{x} = \mathbf{y}$: 因为 $\mathbf{U}$ 是一个上三角矩阵, 我们可以通过后向替换 (backward substitution) 来求解 $\mathbf{x}$.

形式上, 原先我们的映射关系为:
$$
\mathbf{x} \xrightarrow{\mathbf{A}} \mathbf{b}.
$$
通过 LU 分解, 我们将其分解为两个映射关系:
$$
\mathbf{x} \xrightarrow{\mathbf{U}} \mathbf{y} \xrightarrow{\mathbf{L}} \mathbf{b}.
$$


## Symmetric Eigen-decomposition

> [!quote]
> - Reading: Linear Algebra and Its Applications, Section 5

***Definition 1.* (Eigenvectors and eigenvalues)**  给定一个 $n \times n$ 的矩阵 $\mathbf{A}$, 如果存在一个非零向量 $\mathbf{v}$ 和一个标量 $\lambda$ (可能为 $0$), 使得 $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$, 则称 $\mathbf{v}$ 是 $\mathbf{A}$ 的一个特征向量 (eigenvector), $\lambda$ 是对应的特征值 (eigenvalue).

- 上述定义等价于, 若 $\lambda$ 是 eigenvalue, 则 $(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = \mathbf{0}$ 应该有非零解 $\mathbf{v}$, 这说明 $\mathbf{A} - \lambda \mathbf{I}$ 是一个奇异矩阵 (singular matrix), 即 $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$.
- 亦或者, $\mathbf{v}$ 落在 $\mathbf{A} - \lambda \mathbf{I}$ 的 null space 中. 因此, 称这个子空间为 $\mathbf{A}$ 的 $\lambda$-eigenspace.


> [!quote]
> - Reading: Linear Algebra and Its Applications, Section 7.1


对于一个对称矩阵 $\mathbf{S}$, 其具有如下性质. 

***Theorem 1.* (Eigenvectors of symmetric matrices)**  对于对称矩阵 $\mathbf{S}$, 任意两个不同的特征值 $\lambda_i$ 和 $\lambda_j$ 对应的特征向量 $\mathbf{v}_i$ 和 $\mathbf{v}_j$ 是正交的.

- *Proof*. 
  - 要证明 $\mathbf{v}_i$ 和 $\mathbf{v}_j$ 是正交的, 即 $\mathbf{v}_i^\top \mathbf{v}_j = 0$.
  - 考虑 $\lambda_i \mathbf{v}_i^\top \mathbf{v}_j$ . 根据特征值定义, 我们有 $\lambda_i \mathbf{v}_i^\top = (\mathbf{S} \mathbf{v}_i)^\top = \mathbf{v}_i^\top \mathbf{S}$. 因此, $\lambda_i \mathbf{v}_i^\top \mathbf{v}_j = \mathbf{v}_i^\top \mathbf{S} \mathbf{v}_j$.
  - 同样地, $\lambda_j \mathbf{v}_i^\top \mathbf{v}_j = \mathbf{v}_i^\top \mathbf{S} \mathbf{v}_j$.
  - 因此, $\lambda_i \mathbf{v}_i^\top \mathbf{v}_j = \lambda_j \mathbf{v}_i^\top \mathbf{v}_j$.
  - 因为 $\lambda_i \neq \lambda_j$, 上式只能成立当 $\mathbf{v}_i^\top \mathbf{v}_j = 0$ 时. 这就证明了 $\mathbf{v}_i$ 和 $\mathbf{v}_j$ 是正交的.

  $\square$


***Theorem 2.* (Spectral Decomposition)**  对于一个 $n \times n$ 的矩阵 $\mathbf{A}$, $\mathbf{A}$ 是对称矩阵当且仅当 $\mathbf{A}$ 可以被正交对角化 (orthogonally diagonalized), 即存在一个正交矩阵 $\mathbf{Q}\in \mathbb{R}^{n \times n}$ 和一个对角矩阵 $\mathbf{\Lambda}\in \mathbb{R}^{n \times n}$ 使得 $\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^\top$. 其中, $\mathbf{Q}$ 的每一列对应 $\mathbf{A}$ 的一个特征向量, $\mathbf{\Lambda}$ 的对角线元素对应 $\mathbf{A}$ 的特征值, 即
$$
\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^\top = \begin{bmatrix} \mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n \end{bmatrix} \begin{bmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{bmatrix} \begin{bmatrix} \mathbf{v}_1^\top \\ \mathbf{v}_2^\top \\ \vdots \\ \mathbf{v}_n^\top \end{bmatrix}.
$$

若进一步将上述矩阵分解写成 $\mathbf{A} = \sum_{i=1}^n \lambda_i \mathbf{v}_i \mathbf{v}_i^\top$, 则称 $\mathbf{A}$ 的上述分解为 $\mathbf{A}$ 的谱分解 (spectral decomposition).

事实上, 对于一个对称矩阵 $\mathbf{S}$, 其有如下性质:
- $\mathbf{S}$ 的所有 $n$ 个特征值都是实数.
- 特征空间彼此正交, 因此对应不同特征值的特征向量可以被选取为相互正交的.
- $\mathbf{S}$ 可以被正交对角化.


