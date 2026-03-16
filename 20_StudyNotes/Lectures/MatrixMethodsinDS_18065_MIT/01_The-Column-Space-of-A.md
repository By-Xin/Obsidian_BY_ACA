# Lecture 1: The Column Space of $\mathbf{A}$ Contains All Vectors $\mathbf{Ax}$

> [!quote]
> - Lecture: 18.065 Matrix Methods in Data Science, MIT, by Gilbert Strang
> - Reading: Linear Algebra and Its Applications, Lay, et al., Global Edition.

***Matrix-Vector Multiplication**

对于矩阵 $\mathbf{A} \in \mathbb{R}^{n \times d}$ 和向量 $\mathbf{x} \in \mathbb{R}^d$ 之间的乘法 $\mathbf{Ax}$, 除了最基本的逐元素的思路之外, 总是可以看作是关于 $\mathbf{A}$ 的列向量的线性组合:
$$
\mathbf{Ax}  = x_1 \begin{bmatrix}a_{11} \\ a_{21} \\ \vdots \\ a_{n1}\end{bmatrix} + 
x_2 \begin{bmatrix}a_{12} \\ a_{22} \\ \vdots \\ a_{n2}\end{bmatrix} + \cdots + x_d \begin{bmatrix}a_{1d} \\ a_{2d} \\ \vdots \\ a_{nd}\end{bmatrix}
$$

这事实上如果按照 regression 的思路看是很自然的. 我们一共就固定了 $d$ 个特征, 每个特征对应 $\mathbf{A}$ 的一列, 但是我们可以通过收集新的数据来得到新的预测结果. 

***Definition* (Vector Space)**: 对于一个非空集合 $V$, 并定义该集合上的两个运算: 向量加法 $+$ 和数乘 $\cdot$, 如果满足以下 9 条公理, 则称 $V$ 是一个**向量空间**:
1. **封闭性**: 对于任意 $\mathbf{u}, \mathbf{v} \in V$, $\mathbf{u} + \mathbf{v} \in V$; 对于任意 $\mathbf{v} \in V$ 和标量 $\alpha$, $\alpha \cdot \mathbf{v} \in V$.
2. **交换律**: 对于任意 $\mathbf{u}, \mathbf{v} \in V$, $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$.
3. **结合律**: 对于任意 $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$, $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$.
4. **存在零向量**: 存在一个零向量 $\mathbf{0} \in V$, 使得对于任意 $\mathbf{v} \in V$, $\mathbf{v} + \mathbf{0} = \mathbf{v}$.
5. **存在负向量**: 对于任意 $\mathbf{v} \in V$, 存在一个负向量 $-\mathbf{v} \in V$, 使得 $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$.
6. **数乘分配律**: 对于任意标量 $\alpha, \beta$ 和向量 $\mathbf{v} \in V$, $\alpha \cdot (\beta \cdot \mathbf{v}) = (\alpha \beta) \cdot \mathbf{v}$.
7. **数乘分配律**: 对于任意标量 $\alpha$ 和向量 $\mathbf{u}, \mathbf{v} \in V$, $\alpha \cdot (\mathbf{u} + \mathbf{v}) = \alpha \cdot \mathbf{u} + \alpha \cdot \mathbf{v}$.
8. **数乘分配律**: 对于任意标量 $\alpha, \beta$ 和向量 $\mathbf{v} \in V$, $(\alpha + \beta) \cdot \mathbf{v} = \alpha \cdot \mathbf{v} + \beta \cdot \mathbf{v}$.
9. **存在单位标量**: 存在一个单位标量 $1$, 使得对于任意 $\mathbf{v} \in V$, $1 \cdot \mathbf{v} = \mathbf{v}$.

>[!example]
>
> - **Example**: $\mathbb{R}^n$ 是一个向量空间, 因为它满足上述所有公理.
> - **Example**: 考虑多项式 $\mathbb{P}_n$ 的集合, 其中每个多项式的次数不超过 $n$, $\mathbb{P}_n$ 也是一个向量空间, 因为它满足上述所有公理.


***Definition* (Subspace)**: 对于一个向量空间 $V$, 如果一个非空子集 $H \subseteq V$ 满足以下条件, 则称 $H$ 是 $V$ 的一个**子空间**:
1. 存在零向量: $\mathbf{0} \in H$.
2. **封闭性**: 对于任意 $\mathbf{u}, \mathbf{v} \in H$, $\mathbf{u} + \mathbf{v} \in H$.
3. **封闭性**: 对于任意 $\mathbf{v} \in H$ 和标量 $\alpha$, $\alpha \cdot \mathbf{v} \in H$.

***Definition* (Subspace Spanned by a Set)**: 对于一个向量空间 $V$ 和一个向量集合 $S \subseteq V$, 定义 $S$ 的**生成子空间**为包含 $S$ 中所有向量的最小子空间, 记为 $\text{Span}(S)$. 换言之, $\text{Span}(S)$ 包含 $S$ 中的所有向量以及所有可以通过 $S$ 中的向量进行线性组合得到的向量.


***Definition* (Column Space)**: 对于一个矩阵 $\mathbf{A} \in \mathbb{R}^{n \times d}$, 定义 $\mathbf{A}$ 的**列空间**为 $\mathbf{A}$ 的列向量的生成子空间, 记为 $\text{Col}(\mathbf{A})$. 若 $\mathbf{A} = [\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_d]$, 则 $\text{Col}(\mathbf{A}) = \text{Span}\{\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_d\}$.

- 对于 $\mathbf{A} \in \mathbb{R}^{n \times d}$, $\text{Col}(\mathbf{A})$ 是 $\mathbb{R}^n$ 的一个子空间, 因为它满足子空间的定义.
- Column Space 还可以被看作是 $\mathbf{A}$ 的线性变换的像 (image), 因为 $\mathbf{Ax}$ 的所有可能取值都在 $\text{Col}(\mathbf{A})$ 中. 换言之, 
    $$
    \text{Col}(\mathbf{A}) = \{\mathbf{y} \in \mathbb{R}^n : \mathbf{y} = \mathbf{Ax} \text{ for some } \mathbf{x} \in \mathbb{R}^d\}
    $$
- 对应的, $\mathbf{A}$ 的行空间 (row space) 就是 $\mathbf{A}^\top$ 的列空间, 也就是 $\mathbf{A}$ 的行向量的生成子空间. 行空间也是 $\mathbb{R}^d$ 的一个子空间.

> [!example]
>
> 考虑矩阵 $\mathbf{A} = \begin{bmatrix}2 & 1 & 3\\3 & 1 & 4\\5 & 7 & 12\end{bmatrix}$. 
> - 若考虑其列空间 $\text{Col}(\mathbf{A})$, 可以注意到其 rank 为 2, 因此 $\text{Col}(\mathbf{A})$ 是 $\mathbb{R}^3$ 的一个二维子空间. 具体来说, 不妨选择 $\mathbf{a}_1$ 和 $\mathbf{a}_2$ 作为 $\text{Col}(\mathbf{A})$ 的基, 即 $\text{Col}(\mathbf{A}) = \text{Span}\left\{\begin{bmatrix}2 \\ 3 \\ 5\end{bmatrix}, \begin{bmatrix}1 \\ 1 \\ 7\end{bmatrix}\right\}$. 并记 $C = \begin{bmatrix}2 & 1 \\ 3 & 1 \\ 5 & 7\end{bmatrix}$. 
> - 同理, 对于矩阵, 总有其行秩等于列秩等于矩阵的秩. 因此 $\mathbf{A}$ 的行空间 $\text{Row}(\mathbf{A})$ 也是一个二维子空间. 这里不妨取 $\text{Col}(\mathbf{A}^\top) = \text{Span}\left\{\begin{bmatrix}1 & 0 & 1 \\ 0 & 1 & 1\end{bmatrix}\right\}$ 作为 $\text{Row}(\mathbf{A})$ 的基. 并记 $R = \begin{bmatrix}1 & 0 & 1 \\ 0 & 1 & 1\end{bmatrix}$.
> - 对于任意矩阵, 我们总是可以分解出一个列满秩的矩阵 $C$ 和一个行满秩的矩阵 $R$, 使得 $\mathbf{A} = CR$. 

***Matrix-Matrix Multiplication***

对于矩阵 $\mathbf{A} \in \mathbb{R}^{n \times d}$ 和 $\mathbf{B} \in \mathbb{R}^{d \times m}$, 它们的乘积 $\mathbf{AB} \in \mathbb{R}^{n \times m}$ 可以由如下几个视角来理解:
- 内积: $\mathbf{AB}$ 的第 $i$ 行第 $j$ 列的元素 $(\mathbf{AB})_{ij}$ 可以看作是 $\mathbf{A}$ 的第 $i$ 行和 $\mathbf{B}$ 的第 $j$ 列的内积, 即
    $$
    (\mathbf{AB})_{ij} = \sum_{k=1}^d A_{ik} B_{kj}
    $$

- 列向量的拼接: 可以将 $\mathbf{B}$ 看作是由 $m$ 个列向量的拼接, 即 $\mathbf{B} = [\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_m]$, 则 $\mathbf{AB}$ 可以看作是 $m$ 个矩阵-向量乘积的拼接, 即
    $$
    \mathbf{AB} = [\mathbf{A}\mathbf{b}_1, \mathbf{A}\mathbf{b}_2, \ldots, \mathbf{A}\mathbf{b}_m]
    $$
    