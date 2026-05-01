# Numerical Linear Algebra

>[!quote]
>
> - Lecture Reference: <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>
> - Book Reference: 最优化: 建模算法与理论

## Complexity of Basic Operations

在进行优化过程中, 要进行大量的线性系统的求解运算. 首先介绍相关的数值计算中的运算的基本复杂度单位: Flops (Floating Point Operations). 粗略地说, 一次 flop 是指一次浮点数运算, 例如加法, 减法, 乘法, 除法等. 我们可以将各种计算操作的复杂度用 flop 来衡量.  下面是一些常见的线性代数操作及其复杂度分析.

对于向量乘法, $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$, 及 $c \in \mathbb{R}$, 下面是一些基本的线性代数操作及其复杂度:
- $\mathbf{a} + \mathbf{b}$ 需要 $n$ 次加法, 因此复杂度为 $\mathcal{O}(n)$ flops.
- 标量乘法 $c \mathbf{a}$ 需要 $n$ 次乘法, 因此复杂度为 $\mathcal{O}(n)$ flops.
- 内积 $\mathbf{a}^\top \mathbf{b} = \sum_{i=1}^n a_i b_i$ 需要 $n$ 次乘法和 $n-1$ 次加法, 因此复杂度为 $\mathcal{O}(2n)$ flops.

对于矩阵-向量运算 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 和 $\mathbf{x} \in \mathbb{R}^n$, 下面是一些常见的矩阵运算及其复杂度:
- 矩阵-向量乘法 $\mathbf{A} \mathbf{x}$ 需要 $m$ 行和 $n$ 列的乘法, 相当于 $m$ 个内积, 每个内积是长度为 $n$ 的向量, 因此复杂度为 $\mathcal{O}(2mn)$ flops.
- 特别地, 若 $\mathbf{A}$ 是 $s$-稀疏的, 即整个矩阵中只有 $s$ 个非零元素, 则矩阵-向量乘法只需要处理这 $s$ 个非零元素, 大概的复杂度为 $\mathcal{O}(2s)$ flops.
- 若 $\mathbf{A}$ 是 $k$-banned, 即矩阵集中在对角线附近宽度为 $k$ 的区域内, 则矩阵-向量乘法的复杂度大约为 $\mathcal{O}(2nk)$ flops.
- 若 $\mathbf{A}$ 是一个秩为 $r$ 的 low-rank 矩阵, 其可以拆分为 $\mathbf{A} = \sum_{i=1}^r \mathbf{u}_i \mathbf{v}_i^\top$, 则乘法 $\mathbf{A} \mathbf{x}$ 可以通过计算 $\sum_{i=1}^r \mathbf{u}_i (\mathbf{v}_i^\top \mathbf{x})$ 来实现, 相当于 $r$ 次内积和 $r$ 次标量乘法, 因此复杂度为 $\mathcal{O}(2r(n+m))$ flops.
- 若 $\mathbf{A}$ 是一个 permutation矩阵, 即每行和每列只有一个非零元素, 相当于对向量进行重排, 则没有实际的浮点数运算, 只是相当于在内存中进行数据的重排, 故在 flops 的意义上复杂度为 $\mathcal{O}(0)$ flops.

对于矩阵乘法 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 和 $\mathbf{B} \in \mathbb{R}^{n \times p}$, 一般的矩阵乘法需要 $m$ 行, $p$ 列向量分别进行长度为 $n$ 的内积, 因此复杂度为 $\mathcal{O}(2mnp)$ flops. 但是对于特殊结构的矩阵, 例如 $\mathbf{A}$ 是 $s$-稀疏的, 则矩阵乘法的复杂度大约为 $\mathcal{O}(2s p)$ flops, 

对于矩阵-矩阵-向量运算 $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{B} \in \mathbb{R}^{n \times p}$ 和 $\mathbf{x} \in \mathbb{R}^p$, 计算 $\mathbf{A} \mathbf{B} \mathbf{x}$ 可以通过先计算 $\mathbf{y} = \mathbf{B} \mathbf{x}$, 然后计算 $\mathbf{A} \mathbf{y}$ 来实现. 其总的复杂度为 $\mathcal{O}(2np + 2mn)$ flops. 但若采取错误的计算顺序, 则先计算 $\mathbf{A} \mathbf{B}$ 的复杂度为 $\mathcal{O}(2mnp)$ flops, 然后再计算 $(\mathbf{A} \mathbf{B}) \mathbf{x}$ 的复杂度为 $\mathcal{O}(2mp)$ flops, 总的复杂度为 $\mathcal{O}(2mnp + 2mp)$ flops, 这在 $n$ 很大的情况下会非常昂贵. 因此选择正确的计算顺序对于优化计算效率非常重要.


## Solving Linear Systems

给定义一个非奇异方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 和一个向量 $\mathbf{b} \in \mathbb{R}^n$, 求解线性系统 $\mathbf{A} \mathbf{x} = \mathbf{b}$. 
- 一般而言, 若无特殊结构, 则其通过 Gaussian elimination 等类似方法进行消元的一般复杂度为 $\mathcal{O}(n^3)$ flops.
- 若 $\mathbf{A}$ 是纯对角阵, 例如 $\mathbf{A} = \text{diag}(a_1, a_2, \ldots, a_n)$, 则求解线性系统 $\mathbf{A} \mathbf{x} = \mathbf{b}$ 只需要进行 $n$ 次标量除法: $x_i = b_i / a_i$ for $i=1, 2, \ldots, n$, 因此复杂度为 $\mathcal{O}(n)$ flops.
- 若 $\mathbf{A}$ 是一个下三角矩阵 (上三角矩阵同理), 则其总的复杂度为 $\mathcal{O}(n^2)$ flops. 其 forward substitution (上三角为 backward substitution) 求解流程如下. 给定
    $$
    \begin{aligned}
    A = \begin{bmatrix}
    a_{11} & 0 & 0 & \cdots & 0 \\
    a_{21} & a_{22} & 0 & \cdots & 0 \\
    a_{31} & a_{32} & a_{33} & \cdots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & a_{n3} & \cdots & a_{nn}
    \end{bmatrix}, \quad
    \mathbf{b} = \begin{bmatrix}
    b_1 \\
    b_2 \\
    b_3 \\
    \vdots \\
    b_n
    \end{bmatrix}, \quad
    \mathbf{x} = \begin{bmatrix}
    x_1 \\
    x_2 \\
    x_3 \\
    \vdots \\
    x_n
    \end{bmatrix}
    \end{aligned}
    $$
    则 
    $$
    \begin{aligned}
    x_1 &= b_1 / a_{11} \\
    x_2 &= (b_2 - a_{21} x_1) / a_{22} \\
    x_3 &= (b_3 - a_{31} x_1 - a_{32} x_2) / a_{33} \\
    &\vdots \\
    x_n &= (b_n - a_{n1} x_1 - a_{n2} x_2 - \cdots - a_{n, n-1} x_{n-1}) / a_{nn}
    \end{aligned}
    $$
    其总共的计算量为 $1 + 2 + \cdots + (n-1) = \frac{n(n-1)}{2}$ 次乘法和 $n$ 次除法, 因此复杂度为 $\mathcal{O}(n^2)$ flops.


- 若  $\mathbf{A}$ 是一个稀疏矩阵, 则其求解线性系统的复杂度取决于稀疏矩阵的结构. 有些矩阵在进行消元之后仍然很稀疏, 例如 $k$-banded 矩阵, 其求解线性系统的复杂度大约为 $\mathcal{O}(nk^2)$ flops. 但有些矩阵在进行消元之后会变得非常稠密, 该现象成为 fill-in, 其求解线性系统的复杂度可能会增加到 $\mathcal{O}(n^3)$ flops. 因此, 真正影响复杂度不是原始矩阵的稀疏性,而是在进行分解之后的稀疏性. 这也是为什么在进行矩阵分解时, 需要选择合适的 pivoting strategy 来尽量减少 fill-in 的现象.
- 若 $\mathbf{A}$ 是正交矩阵, 即 $\mathbf{A}^\top \mathbf{A} = \mathbf{I}$, 则其求解线性系统的复杂度为 $\mathcal{O}(n^2)$ flops. 这是因为对于正交矩阵 $\mathbf{A}$, 求解线性系统 $\mathbf{A} \mathbf{x} = \mathbf{b}$ 可以通过计算 $\mathbf{x} = \mathbf{A}^\top \mathbf{b}$ 来实现, 其复杂度为 $\mathcal{O}(n^2)$ flops. 

### Sherman-Morrison-Woodbury Formula

这里额外补充一个求解你矩阵的技巧. 其核心在讨论的问题是, 如果我们已经知道了 $\mathbf{A}^{-1}$ 或已经可以高效求解$\mathbf{A} \mathbf{x} = \mathbf{b}$, 那么当 $\mathbf{A}$ 发生了一个 low-rank 的 perturbation, 例如 $\mathbf{A} + \mathbf{U} \mathbf{C} \mathbf{V}$ 后, 我们可以有更高效的方式来求解更新后的线性系统. 

***Theorem* (Sherman-Morrison-Woodbury Formula)** 给定 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 是一个非奇异矩阵, $\mathbf{U} \in \mathbb{R}^{n \times k}$, $\mathbf{C} \in \mathbb{R}^{k \times k}$ 和 $\mathbf{V} \in \mathbb{R}^{k \times n}$ 是任意的矩阵 ($\mathbf{C}$ 需要是非奇异的), 则
$$
\mathbf{A}_{n \times n} + \mathbf{U}_{n \times k} \mathbf{C}_{k \times k} \mathbf{V}_{k \times n} \text{可逆} \iff \mathbf{C}_{k \times k}^{-1} + \mathbf{V}_{k \times n} \mathbf{A}_{n \times n}^{-1} \mathbf{U}_{n \times k} \text{可逆}
$$
且其逆矩阵可以通过以下公式计算:
$$
(\mathbf{A} + \mathbf{U} \mathbf{C} \mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1} \mathbf{U} (\mathbf{C}^{-1} + \mathbf{V} \mathbf{A}^{-1} \mathbf{U})^{-1} \mathbf{V} \mathbf{A}^{-1}.
$$

特别地, 
- 若 $k = 1$, $C = 1$, 则 $\mathbf{A} + \mathbf{u} \mathbf{v}^\top$ 可逆当且仅当 $1 + \mathbf{v}^\top \mathbf{A}^{-1} \mathbf{u} \neq 0$, 且其逆矩阵为
    $$
    (\mathbf{A} + \mathbf{u} \mathbf{v}^\top)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1} \mathbf{u} \mathbf{v}^\top \mathbf{A}^{-1}}{1 + \mathbf{v}^\top \mathbf{A}^{-1} \mathbf{u}}.
    $$

- 若 $\mathbf{C} = I$, 则 $\mathbf{A} + \mathbf{U} \mathbf{V}$ 可逆当且仅当 $\mathbf{I} + \mathbf{V} \mathbf{A}^{-1} \mathbf{U}$ 可逆, 且其逆矩阵为
    $$
    (\mathbf{A} + \mathbf{U} \mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1} \mathbf{U} (\mathbf{I} + \mathbf{V} \mathbf{A}^{-1} \mathbf{U})^{-1} \mathbf{V} \mathbf{A}^{-1}.
    $$


SMW 分解的核心在于, 对于更新后的系统 $\mathbf{A} + \mathbf{U} \mathbf{C} \mathbf{V}$, 其总的维度仍然是 $n \times n$, 若直接计算其逆矩阵, 则浪费了大量的关于 low-rank perturbation 的结构信息. 通过 SMW 分解, 我们可以将其逆矩阵的计算转化为对 $\mathbf{A}^{-1}$ 的计算 (该计算我们已经知道了), 以及对一个 $k \times k$ 的矩阵 $\mathbf{C}^{-1} + \mathbf{V} \mathbf{A}^{-1} \mathbf{U}$ 的计算 (该矩阵的维度远小于 $n$), 从而大大降低了计算的复杂度 (其中 $k$ 是 perturbation 的秩, 通常远小于 $n$). 

## Matrix Factorizations for Solving Linear Systems

矩阵分解的思路如下. 在进行求解 $\mathbf{A} \mathbf{x} = \mathbf{b}$ 的过程中, 我们几乎从不会直接求解 $\mathbf{A}$ 的逆矩阵 $\mathbf{A}^{-1}$ 来计算. 更常见的做法是先将 $\mathbf{A}$ 分解成一些特殊矩阵的乘积:
$$
\mathbf{A} = \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_k \iff \mathbf{A} \mathbf{x} = \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_k \mathbf{x} = \mathbf{b}
$$
则此时我们可以通过依次求解以下线性系统来得到 $\mathbf{x}$:
$$
\begin{aligned}
\mathbf{A}_1 (\mathbf{A}_2 \cdots \mathbf{A}_k \mathbf{x}) &:= \mathbf{A}_1 \mathbf{y}_1 = \mathbf{b} \quad \text{(求解 $\mathbf{y}_1$)} \\
\mathbf{A}_2 (\mathbf{A}_3 \cdots \mathbf{A}_k \mathbf{x}) &:= \mathbf{A}_2 \mathbf{y}_2 = \mathbf{y}_1 \quad \text{(求解 $\mathbf{y}_2$)} \\
&\vdots \\
\mathbf{A}_k \mathbf{x} &= \mathbf{y}_{k-1} \quad \text{(求解 $\mathbf{x}$)}
\end{aligned}
$$
其中, 每一步的求解虽然形式上仍然要处理 $\mathbf{y}_i = \mathbf{A}_i^{-1} \mathbf{y}_{i-1}$, 但由于 $\mathbf{A}_i$ 的特殊结构, 其求解的复杂度通常会大大降低. 常见的特殊结构就包括前面提到的对角矩阵, 三角矩阵, 稀疏矩阵, 正交矩阵等. 

该效应尤其还在当我们要求解多个线性系统 $\mathbf{A} \mathbf{x}_i = \mathbf{b}_i$ for $i=1, 2, \ldots, m$ 的时候更加明显. 因为在这种情况下, 我们只需要进行一次矩阵分解 $\mathbf{A} = \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_k$ (该复杂度通常为 $\mathcal{O}(n^3)$ flops), 然后对于每个线性系统, 只需要依次求解 $k$ 个特殊结构的线性系统 (每个复杂度通常为 $\mathcal{O}(n^2)$ flops), 因此总的复杂度为 $\mathcal{O}(n^3 + m n^2)$ flops, 而如果直接求解每个线性系统, 则总的复杂度为 $\mathcal{O}(m n^3)$ flops, 当 $m$ 很大的时候, 前者的效率会远远高于后者.

### QR Decomposition

对于任意的矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}, ~ m \geq n$, 其都存在如下标准 QR 分解:
$$
\mathbf{A}_{m \times n} = \mathbf{Q}_{m \times m} \mathbf{R}_{m \times n}
$$
其中 $\mathbf{Q}$ 是一个正交矩阵, $\mathbf{R}$ 是一个上三角矩阵. 然而这里, 当 $m > n$ 的时候, $\mathbf{R}$ 的下半部分是全零的, 因此会浪费大量的存储空间. 故在实际的数值计算中, 我们通常会使用一个更为紧凑的 QR 分解形式, 称为 reduced QR decomposition, 其本质在于去除 $\mathbf{R}$ 中的全零部分, 以及 $\mathbf{Q}$ 中对应的列. 其形式如下:
$$
\mathbf{A}_{m \times n} = \mathbf{Q}_{m \times n} \mathbf{R}_{n \times n}
$$
其中仍有 $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$, 但 $\mathbf{Q}$ 不再是一个方阵; $\mathbf{R}$ 是一个上三角矩阵. 有数学定理可以保证, 只要 $\mathbf{A}$ 是列满秩的, 且额外要求 $\mathbf{R}$ 的对角线元素都为正, 则 reduced QR 分解是唯一的. 

通过 Gram-Schmidt 等方法, 都可以实现 QR 分解. 其总体的复杂度约为 $\mathcal{O}(m n^2)$ flops. 


***Example* (QR Decomposition for OLS)** 给定标签 $\mathbf{y} \in \mathbb{R}^n$ 和特征矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$, 考虑如下最小二乘问题:
$$
\min_{\boldsymbol{\beta} \in \mathbb{R}^d} \|\mathbf{y} - \mathbf{X} \boldsymbol{\beta}\|_2^2
$$
为求解该问题, 我们可以先对 $\mathbf{X}$ 进行 reduced QR 分解, 得到
$$
\mathbf{X}_{n \times d} = \mathbf{Q}_{n \times d} \mathbf{R}_{d \times d}, \quad \text{where } \mathbf{Q}^\top \mathbf{Q} = \mathbf{I}_d.
$$

由于 $\mathbf{Q} \in \mathbb{R}^{n \times d}$ 还不是一个完整的正交矩阵, 故额外引入一个矩阵 $\widetilde{\mathbf{Q}} \in \mathbb{R}^{n \times (n-d)}$, 得到 $\mathbf{P} := [\mathbf{Q} ~ \widetilde{\mathbf{Q}}] \in \mathbb{R}^{n \times n}$ 是一个完整的正交矩阵, 其满足 $\mathbf{P}^\top \mathbf{P} = \mathbf{I}_n$. 此时我们有如下观察:
$$
\begin{aligned}
\|\mathbf{y} - \mathbf{X} \boldsymbol{\beta}\|_2^2 
&= \|\mathbf{P}^\top (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})\|_2^2 \\
&= \left\|\begin{bmatrix}\mathbf{Q}^\top \\ \widetilde{\mathbf{Q}}^\top\end{bmatrix} \left(\mathbf{y} - \mathbf{Q} \mathbf{R} \boldsymbol{\beta}\right)\right\|_2^2 \quad \text{(Since $\mathbf{X} = \mathbf{Q} \mathbf{R}$ and $\mathbf{P} = [\mathbf{Q} ~ \widetilde{\mathbf{Q}}]$) } \\
&= \left\|\mathbf{Q}^\top \mathbf{y} - \mathbf{Q}^\top \mathbf{Q} \mathbf{R} \boldsymbol{\beta}\right\|_2^2 + \left\|\widetilde{\mathbf{Q}}^\top \mathbf{y} - \widetilde{\mathbf{Q}}^\top \mathbf{Q} \mathbf{R} \boldsymbol{\beta}\right\|_2^2 \quad \text{(Since $\|[a; b]\|_2^2 = \|a\|_2^2 + \|b\|_2^2$) } \\
&= \left\|\mathbf{Q}^\top \mathbf{y} - \mathbf{R} \boldsymbol{\beta}\right\|_2^2 + \left\|\widetilde{\mathbf{Q}}^\top \mathbf{y}\right\|_2^2 \quad \text{(Since $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$ and $\widetilde{\mathbf{Q}}^\top \mathbf{Q} = 0$) }
\end{aligned}
$$
故最小化 $\|\mathbf{y} - \mathbf{X} \boldsymbol{\beta}\|_2^2$ 等价于最小化 $\left\|\mathbf{Q}^\top \mathbf{y} - \mathbf{R} \boldsymbol{\beta}\right\|_2^2$ (因为 $\left\|\widetilde{\mathbf{Q}}^\top \mathbf{y}\right\|_2^2$ 与 $\boldsymbol{\beta}$ 无关).  而由于 $\mathbf{R}$ 是一个上三角矩阵其一定是可逆的, 因此 $\left\|\mathbf{Q}^\top \mathbf{y} - \mathbf{R} \boldsymbol{\beta}\right\|_2^2$ 的最小值为 $0$, 其最优解为 $\boldsymbol{\widehat{\beta}} = \mathbf{R}^{-1} \mathbf{Q}^\top \mathbf{y}$. 不过这里指出, 由于 $\mathbf{R}$ 是一个上三角矩阵, 其求解 $\boldsymbol{\widehat{\beta}}$ 的过程可以通过 backward substitution 来实现, 因此不需要显式地计算 $\mathbf{R}^{-1}$ 来得到 $\boldsymbol{\widehat{\beta}}$, 从而大大降低了计算的复杂度. 

其总体复杂度如下:
- 首先对 $\mathbf{X}$ 进行 reduced QR 分解的复杂度为 $\mathcal{O}(2 n d^2 - d^3 / 3)$ flops.
- 计算 $\mathbf{Q}^\top \mathbf{y}$ 的复杂度为 $\mathcal{O}(2 n d)$ flops.
- 通过 backward substitution 求解 $\boldsymbol{\widehat{\beta}}$ 的复杂度为 $\mathcal{O}(d^2)$ flops.

因此总的复杂度约为 $\mathcal{O}(2 n d^2 - d^3 / 3)$ flops. 该复杂度在 $n \gg d$ 的情况下, 主要由对 $\mathbf{X}$ 进行 reduced QR 分解的复杂度主导, 大约为 $\mathcal{O}(2 n d^2)$ flops.

### LU Decomposition & Cholesky Decomposition

**LU Decomposition for General Non-Singular Matrices**

- 对于一般的非奇异矩阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$, 一个通用的 LU 分解为:
    $$
    \mathbf{P} \mathbf{A} = \mathbf{L} \mathbf{U}
    $$
    其中 $\mathbf{P} \in \mathbb{R}^{n \times n}$ 是一个 permutation 矩阵, 用来进行行交换. $\mathbf{L} \in \mathbb{R}^{n \times n}$ 是一个下三角矩阵, 其对角线元素为 $1$, $\mathbf{U} \in \mathbb{R}^{n \times n}$ 是一个上三角矩阵.  $\mathbf{P}$ 的引入是为了保证在进行 Gaussian elimination 的过程中, 可以通过行交换来避免出现零 pivot 的情况, 从而保证分解的稳定性. 

- 在进行 LU 分解之后, 求解线性系统 $\mathbf{A} \mathbf{x} = \mathbf{b}$ 的流程如下:
    $$
    \begin{aligned}
    \mathbf{A} \mathbf{x} = \mathbf{b} &\iff \mathbf{P} \mathbf{A} \mathbf{x} = \mathbf{P} \mathbf{b} \iff \mathbf{L} \mathbf{U} \mathbf{x} = \mathbf{P} \mathbf{b} \\
    \end{aligned}
    $$
    故求解可分两步进行:
    $$
    \begin{aligned}
    \mathbf{L} \mathbf{y} &= \mathbf{P} \mathbf{b} \quad \text{(forward substitution 求解 $\mathbf{y})$} \\
    \mathbf{U} \mathbf{x} &= \mathbf{y} \quad \text{(backward substitution 求解 $\mathbf{x})$}
    \end{aligned}
    $$
    对于分解后的系统, 求解的复杂度约为 $\mathcal{O}(3n^2)$ flops.

- LU 分解本身的一般复杂度为 $\mathcal{O}(\frac{2n^3}{3})$ flops. 

**Cholesky Decomposition for Symmetric Positive Definite Matrices**

- 特别地, 对于 $\mathbf{A}$ 是一个对称正定矩阵的情况 ($\mathbf{A} = \mathbf{A}^\top \succ 0$), 存在一个更为高效的分解方法, 称为 Cholesky 分解, 其形式如下:
    $$
    \mathbf{A} = \mathbf{L} \mathbf{L}^\top
    $$
    其中 $\mathbf{L}$ 是一个下三角矩阵, 其对角线元素为正. 根据待定系数法, 对于一个一般的 $n \times n$ 的矩阵 $\mathbf{A}$, 可以推得其递推公式为:
    $$
    L_{ij} = \begin{cases}
        \left(A_{ij} - \sum_{k=1}^{j-1} L_{ik} L_{jk}\right) / L_{jj}, \quad \text{for } i > j \\
        \sqrt{A_{jj} - \sum_{k=1}^{j-1} L_{jk}^2}, \quad \text{for } i = j \\
    \end{cases}
    $$
    其中其对角线元素需要进行开方运算. 

- 或等价地, 通过引入一个对角矩阵 $\mathbf{D}$, 我们可以得到一个不用进行开方运算的 Cholesky 分解形式, 其形式如下:
    $$
    \mathbf{A} = \mathbf{\tilde{L}} \mathbf{D} \mathbf{\tilde{L}}^\top
    $$
    其中 $\mathbf{\tilde{L}}$ 是一个下三角矩阵, 其对角线元素为 $1$, $\mathbf{D}$ 是一个对角矩阵, 其对角线元素为正.  


- 由于对称性的存在, 我们大概只需要处理一半的信息, 因此 Cholesky 分解的复杂度大约为 $\mathcal{O}(\frac{n^3}{3})$ flops, 大约是 LU 分解的一半. 

- 在进行 Cholesky 分解之后, 求解线性系统 $\mathbf{A} \mathbf{x} = \mathbf{b}$ 的流程如下:
    $$
    \begin{aligned}
    \mathbf{A} \mathbf{x} = \mathbf{b} &\iff \mathbf{L} \mathbf{L}^\top \mathbf{x} = \mathbf{b} \\
    &\iff \mathbf{L} \mathbf{y} = \mathbf{b} \quad \text{(forward substitution 求解 $\mathbf{y}$)} \\
    &\iff \mathbf{L}^\top \mathbf{x} = \mathbf{y} \quad \text{(backward substitution 求解 $\mathbf{x}$)}
    \end{aligned}
    $$
    对于分解之后的线性系统, 求解的复杂度约为 $\mathcal{O}(2n^2)$ flops. 


***Example* (Cholesky Decomposition for OLS)** 给定标签 $\mathbf{y} \in \mathbb{R}^n$ 和特征矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$, 考虑如下最小二乘问题:
$$
\min_{\boldsymbol{\beta} \in \mathbb{R}^d} \|\mathbf{y} - \mathbf{X} \boldsymbol{\beta}\|_2^2
$$
假设 $\mathbf{X}$ 是列满秩的, 则其闭式解为:
$$
\boldsymbol{\widehat{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$
其计算复杂度为:
- 首先计算 $\mathbf{X}^\top \mathbf{X}$ 的复杂度为 $\mathcal{O}(n d^2)$ flops.
- 同时计算 $\mathbf{X}^\top \mathbf{y}$ 的复杂度为 $\mathcal{O}(2 n d)$ flops.
- 对 $\mathbf{X}^\top \mathbf{X}$ 进行 Cholesky 分解的复杂度为 $\mathcal{O}(d^3 / 3)$ flops.
- 通过分解后的系统求解线性系统 $(\mathbf{X}^\top \mathbf{X}) \boldsymbol{\beta} = \mathbf{X}^\top \mathbf{y}$, 其复杂度为 $\mathcal{O}(2 d^2)$ flops.

因此总的复杂度为 $\mathcal{O}(n d^2 + d^3/3)$ flops. 该复杂度在 $n \gg d$ 的情况下, 主要由计算 $\mathbf{X}^\top \mathbf{X}$ 的复杂度主导, 大约为 $\mathcal{O}(n d^2)$ flops.

**Cholesky-like Decomposition for Symmetric Indefinite Matrices** 

- 对于对称但非定的矩阵 $\mathbf{A} = \mathbf{A}^\top$, 其分解的形式如下:
    $$
    \mathbf{P} \mathbf{A} \mathbf{P}^\top = \mathbf{L} \mathbf{D} \mathbf{L}^\top
    $$
    其中 $\mathbf{P}$ 是一个 permutation 矩阵, $\mathbf{L}$ 是一个下三角矩阵, 其对角线元素为 $1$, $\mathbf{D}$ 是一个对角矩阵或分块对角矩阵, 其对角线元素可以是正的, 也可以是负的. 


## Matrix Factorizations for Numerical Stability

除了进行求解线性系统之外, 线性代数中另外一大类关心的问题是数值稳定性. 在正式展开之前, 回顾一下相关的代数知识. 

### Eigenvalue and Eigenvector

***Definition* (Eigenvalue and Eigenvector)** 对于方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$, 若存在一个非零向量 $\mathbf{v} \in \mathbb{R}^n$ 和一个标量 $\lambda \in \mathbb{R}$, 使得 
$$
\mathbf{A} \mathbf{v} = \lambda \mathbf{v}
$$
则称 $\lambda$ 是 $\mathbf{A}$ 的一个特征值 (eigenvalue), $\mathbf{v}$ 是 $\mathbf{A}$ 的一个特征向量 (eigenvector).

- $\lambda$ 是 $\mathbf{A}$ 的一个特征值当且仅当 $\mathbf{A} - \lambda \mathbf{I}$ 是一个奇异矩阵, 即 $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$. 
- $p_A(\lambda) := \det(\mathbf{A} - \lambda \mathbf{I})$ 被称为 $\mathbf{A}$ 的特征多项式 (characteristic polynomial), 其根就是 $\mathbf{A}$ 的特征值. 对于一个 $n$ 阶矩阵 $\mathbf{A}$ 其定有 $n$ 个特征值 (重根算作多个).
- 对于一个对角阵, 其特征值就是其对角线元素, 其特征向量就是标准基向量.
- 对于矩阵 $\mathbf{A}$, 其特征值, 行列式和 trace 之间的关系如下:
    $$
    \begin{aligned}
    \operatorname{det}(\mathbf{A}) &= \prod_{i=1}^n \lambda_i \\
    \operatorname{tr}(\mathbf{A}) &= \sum_{i=1}^n \lambda_i
    \end{aligned}
    $$
    其中特征值按照重数进行计数.

***Definition* (Similarity)** 对于两个矩阵 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{n \times n}$, 若存在一个可逆矩阵 $\mathbf{X} \in \mathbb{R}^{n \times n}$, 使得
$$
\mathbf{B} = \mathbf{X} \mathbf{A} \mathbf{X}^{-1}
$$
则称 $\mathbf{A}$ 和 $\mathbf{B}$ 是相似的 (similar).
- 矩阵的相似关系是一种等价关系. 其表示同一个线性变换在不同基下的矩阵表示. 

- 相似的矩阵具有相同的特征值, 但不一定具有相同的特征向量.
    - $\operatorname{det}(\lambda \mathbf{I} - \mathbf{B}) = \operatorname{det}(\lambda \mathbf{I} - \mathbf{X} \mathbf{A} \mathbf{X}^{-1}) = \operatorname{det}(\mathbf{X} (\lambda \mathbf{I} - \mathbf{A}) \mathbf{X}^{-1}) = \operatorname{det}(\lambda \mathbf{I} - \mathbf{A})$, 故 $\mathbf{A}$ 和 $\mathbf{B}$ 的特征多项式相同, 从而具有相同的特征值. 但由于 $\mathbf{B}$ 的特征向量 $\mathbf{v}$ 满足 $\mathbf{B} \mathbf{v} = \lambda \mathbf{v}$, 则 $\mathbf{A}$ 的特征向量为 $\mathbf{X}^{-1} \mathbf{v}$, 从而不一定与 $\mathbf{B}$ 的特征向量相同.

- 如果存在非奇异矩阵 $\mathbf{X}$ 使得 $\mathbf{X} \mathbf{A} \mathbf{X}^{-1} =: \mathbf{D} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ 是一个对角矩阵, 则称 $\mathbf{A}$ 是**可对角化的 (diagonalizable)**, 其特征值为 $\lambda_1, \lambda_2, \ldots, \lambda_n$, 其特征向量为 $\mathbf{X}^{-1} e_i$ for $i=1, 2, \ldots, n$ (其中 $e_i$ 是标准基向量).
  - 一个矩阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 是可对角化的当且仅当 $\mathbf{A}$ 的全部特征向量 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$ 是线性无关的. 

### Eigenvalue Decomposition & Singular Value Decomposition

事实上, 上述对角化的过程就是我们在数值线性代数中非常重要的一个工具, 称为 eigenvalue decomposition. 其形式如下.

***Theorem* (Eigenvalue Decomposition)** 对于一个可对角化的矩阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$, 其存在 $n$ 个线性无关的特征向量 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$ 和对应的特征值 $\lambda_1, \lambda_2, \ldots, \lambda_n$, 使得
$$
\mathbf{A} = \mathbf{P}  \boldsymbol{\Lambda} \mathbf{P}^{-1} = \begin{bmatrix}| & | & & | \\
\mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n \\
| & | & & |\end{bmatrix} \begin{bmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_n
\end{bmatrix} \begin{bmatrix}| & | & & | \\
\mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n \\
| & | & & |\end{bmatrix}^{-1}
$$
- 注意, 对于任意的可对角化矩阵, 其分解形式为
    $$
    \mathbf{A} = \mathbf{P}  \boldsymbol{\Lambda} \mathbf{P}^{-1}
    $$
    若对于实对称矩阵, 其一定是可以被对角化的, 并且更进一步是可被 orthogonally diagonalizable 的, 其分解形式为
    $$
    \mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^\top
    $$
- 反过来, 如果对于一个方阵 $\mathbf{M} \in \mathbb{R}^{n \times n}$, 我们已经明确的知道了 $\mathbf{M} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^\top$ 的分解形式, 其中 $\mathbf{Q}\in \mathbb{R}^{n \times n}$ 满足 $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$, $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ 是一个对角矩阵, 则我们可以直接得出 $\mathbf{M}$ 的特征值为 $\lambda_1, \lambda_2, \ldots, \lambda_n$, 其特征向量为 $\mathbf{Q}$ 的列向量.
  
然而, 并不是任意矩阵都能进行 eigenvalue decomposition. 其要求矩阵必须是方阵, 可对角化的, 且其特征值必须是实数. 一个更为一般的分解方法是 singular value decomposition, 其形式如下.

***Theorem* (Singular Value Decomposition)** 对于任意的矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$, 其存在一个 $m \times m$ 的正交矩阵 $\mathbf{U}$, 一个 $n \times n$ 的正交矩阵 $\mathbf{V}$, 以及一个 $m \times n$ 的对角型矩阵 $\boldsymbol{\Sigma}$, 使得
$$
\mathbf{A}_{m\times n} = \mathbf{U}_{m\times m} \boldsymbol{\Sigma}_{m\times n} \mathbf{V}_{n\times n}^\top
$$

- 其中 $\mathbf{A}$ 若为方阵, 则 $\boldsymbol{\Sigma}$ 直接为对角矩阵 $\operatorname{diag}(\sigma_1, \sigma_2, \ldots, \sigma_n)$. 若非方阵, 则相当于一个对角矩阵和一个纯零矩阵的拼接. 例如
    $$
    \boldsymbol{\Sigma} = \begin{bmatrix}
    \sigma_1 & 0 & \cdots & 0 \\
    0 & \sigma_2 & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & \sigma_r \\
    0 & 0 & \cdots & 0 \\
    \vdots & \vdots & & \vdots \\
    0 & 0 & \cdots & 0 \\
    \end{bmatrix}_{m \times n} 
    \quad \text{or} \quad
    \boldsymbol{\Sigma} = \begin{bmatrix}
    \sigma_1 & 0 & \cdots & 0 & 0 & \cdots & 0 \\
    0 & \sigma_2 & \cdots & 0 & 0 & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots & \vdots & & \vdots \\
    0 & 0 & \cdots & \sigma_r & 0 & \cdots & 0 \\
    \end{bmatrix}_{m \times n}
    $$

- $\boldsymbol{\Sigma}$ 中的 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ 被称为 $\mathbf{A}$ 的奇异值 (singular value). 事实上, $\mathbf{A}$ 的特征值 $\lambda_i$ 即为 $\mathbf{A}^\top \mathbf{A}$ 的特征值, 其关系为
    $$
    \sigma_i(\mathbf{A}) = \sqrt{\lambda_i(\mathbf{A}^\top \mathbf{A})}
    $$
    - *Proof*. 因为 $\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$, 则 $\mathbf{A}^\top \mathbf{A} = \mathbf{V} \boldsymbol{\Sigma}^\top \mathbf{U}^\top \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top = \mathbf{V} \boldsymbol{\Sigma}^\top \boldsymbol{\Sigma} \mathbf{V}^\top$. 因此 $\mathbf{A}^\top \mathbf{A}$ 的特征值为 $\sigma_i^2$, 从而 $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top \mathbf{A})}$.
    - 非零奇异值的个数事实上就相当于衡量了输入空间在经过线性变换 $\mathbf{A}$ 之后的输出空间的维度. 例如, 如果某一个对应的维度的奇异值为 $0$, 则相当于在该维度的信息会被压缩成一个零向量, 从而导致输出空间的维度降低.
    - 总的而言, 奇异值是分析一个矩阵的奇异性和稳定性的一个非常重要的工具. 其相比于 rank, 能够提供更为细致的关于矩阵的结构信息. 例如, 对于 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{n \times n}$, 而 $\operatorname{rank}(\mathbf{B}) = r < n$ 是一个不满秩的矩阵而 $\operatorname{rank}(\mathbf{A}) = n$ 是一个满秩的矩阵. 则若用 operator norm 来衡量 $\mathbf{A}$ 和 $\mathbf{B}$ 之间的距离, 并且记 $\mathcal{R}_k$ 为所有秩不超过 $k$ 的矩阵的集合, 则我们有如下观察:
        $$
        \operatorname{dist}(\mathbf{A}, \mathcal{R}_r) = \min_{\mathbf{B} \in \mathcal{R}_r} \|\mathbf{A} - \mathbf{B}\|_2 = \sigma_{r+1}(\mathbf{A})
        $$
      - 该观察说明了, 对于一个满秩的矩阵 $\mathbf{A}$ 来说, 其距离所有不满秩的矩阵的距离就是其第 $r+1$ 大的奇异值. 因此, 若 $\sigma_{r+1}(\mathbf{A})$ 非常小, 则说明 $\mathbf{A}$ 非常接近于一个秩为 $r$ 的矩阵, 从而在数值计算中可能会出现一些不稳定的现象.


- 其对应的几何直观如下. SVD 衡量了任何一个线性变换 $\mathbf{A}$ 对于给定向量 $\mathbf{x}$ 的作用. 
    $$
    \mathbf{A} \mathbf{x} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top \mathbf{x}
    $$
  1.  旋转 / 建系 ($\mathbf{V}^\top \mathbf{x}$): Recall, 给定一个正交矩阵 $\mathbf{V}^\top$, 其左乘作用在向量 $\mathbf{x}$ 上的结果就相当于求解向量 $\mathbf{x}$ 在 $\mathbf{V}$ 的空间中的坐标表示. 因此若记 $\mathbf{c} = \mathbf{V}^\top \mathbf{x}$, 则 $\mathbf{c}$ 就是向量 $\mathbf{x}$ 在 $\mathbf{V}$ 的空间中的坐标表示. 
  2.  伸缩 / 缩放 ($\boldsymbol{\Sigma} \mathbf{c}$): 由于 $\boldsymbol{\Sigma}$ 是一个对角型矩阵, 其左乘作用在向量 $\mathbf{c}$ 上的结果就相当于对 $\mathbf{c}$ 的每个坐标进行伸缩. 其中 $\sigma_i$ 就是第 $i$ 个坐标的伸缩因子.
  3. 旋转  / 摆放 ($\mathbf{U} \boldsymbol{\Sigma} \mathbf{c}$): 由于 $\mathbf{U}$ 是一个正交矩阵, 其左乘作用在向量 $\boldsymbol{\Sigma} \mathbf{c}$ 上的结果就相当于对刚刚旋转伸缩后的坐标再翻译回原空间中的坐标表示. 特别地, 若 $\mathbf{U}$ 和 $\mathbf{V}$ 的行列式不同号, 则会发生一个额外的翻转 (flip) 的现象. 


- 奇异值分解还是和矩阵压缩密切相关的. 例如, 对于一个矩阵 $\mathbf{A}$, 其 SVD 分解为 $\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$, 其可以写成:
    $$
    \mathbf{A} = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
    $$
    - 因此, 若只保留前 $k$ 大的奇异值, 则可以得到一个秩为 $k$ 的矩阵 $\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$, 其是 $\mathbf{A}$ 的一个近似. 该近似在 Frobenius 范数意义下是最优的, 即 $\mathbf{A}_k = \arg\min_{\operatorname{rank}(\mathbf{M}) \leq k} \|\mathbf{A} - \mathbf{M}\|_F$. 即为 low-rank approximation 的一个重要工具.


***Definition* (Condition Number)** 对于一个矩阵 $\mathbf{A}$, 其 condition number 定义为:
$$
\kappa (\mathbf{A}) = \frac{\sigma_{\max}(\mathbf{A})}{\sigma_{\min}(\mathbf{A})}
$$
其中 $\sigma_{\max}(\mathbf{A})$ 和 $\sigma_{\min}(\mathbf{A})$ 分别是 $\mathbf{A}$ 的最大和最小(非零)奇异值.

- 若 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 可逆, 则可进一步得到 $\kappa(\mathbf{A}) = \|\mathbf{A}\|_2 \|\mathbf{A}^{-1}\|_2$. 
- 若 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 是一个矩形矩阵但满秩 (例如 OLS 中的标准情况), 则类似可以得到 $\kappa(\mathbf{A}) = \|\mathbf{A}\|_2 \|\mathbf{A}^\dagger\|_2$, 其中 $\mathbf{A}^\dagger$ 是 $\mathbf{A}$ 的 Moore-Penrose pseudo-inverse.

***Example* (Sensitivity of Linear Systems)** 给定一个线性系统 $\mathbf{A} \mathbf{x} = \mathbf{b}$, 记其此时的解为 $\mathbf{x}_0$. 现在, 假设我们对 $\mathbf{A}$ 和 $\mathbf{b}$ 都引入了一个小的扰动, 得到新的线性系统 $(\mathbf{A} + \varepsilon \Delta) \mathbf{x} = \mathbf{b} + \varepsilon \delta$, 其中 $\varepsilon$ 是一个非常小的标量, $\Delta$ 和 $\delta$ 分别是 $\mathbf{A}$ 和 $\mathbf{b}$ 的扰动矩阵. 记新的线性系统的解为 $\mathbf{x}_\varepsilon$. 则有如下关系:
$$
\frac{\|\mathbf{x}_\varepsilon - \mathbf{x}_0\|_2}{\|\mathbf{x}_0\|_2} \leq \kappa(\mathbf{A}) |\varepsilon| \left( \frac{\|\Delta\|_2}{\|\mathbf{A}\|_2} + \frac{\|\delta\|_2}{\|\mathbf{b}\|_2} \right) + o(\varepsilon^2).
$$

***Example* (Sensitivity Analysis of OLS by Cholesky and QR Decompositions)** 考虑同样的最小二乘问题. 这里想要说明, 虽然 Cholesky 分解本身的计算复杂度更快, 但其数值稳定性却不如 QR 分解. 其原因在于, Cholesky 分解需要对 $\mathbf{X}^\top \mathbf{X}$ 进行分解, 而
$$
\kappa(\mathbf{X}^\top \mathbf{X}) = \kappa^2(\mathbf{X})
$$
因此若原先的 $\mathbf{X}$ 已经是一个比较 ill-conditioned 的矩阵了, 则 $\mathbf{X}^\top \mathbf{X}$ 会进一步加剧这种 ill-conditioning 的现象, 从而导致 Cholesky 分解的数值稳定性较差. 反过来, QR 分解直接对 $\mathbf{X}$ 进行分解, 因此其数值稳定性更好. 可以推导证明, 对于 QR 分解, 其最终的误差敏感性大约为 $\kappa(\mathbf{X}) + \|\mathbf{Y} - \mathbf{X} \boldsymbol{\beta}\|_2^2 \cdot \kappa^2(\mathbf{X})$, 而后者在 OLS 能够被很好地拟合的情况下将取值较小, 此时的误差敏感性会明显优于 Cholesky 分解的 $\kappa^2(\mathbf{X})$.


## Indirect Methods for Solving Linear Systems

