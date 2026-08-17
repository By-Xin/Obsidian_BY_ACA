---
aliases: [线性空间, Linear Space, Vector Space, 向量空间, 基, 维数]
tags:
  - concept
  - math/linear-algebra
related_concepts:
  - "[[Matrix]]"
  - "[[Linear_Transformation]]"
  - "[[Rank_and_Eigenvalue]]"
  - "[[Column_Space_of_Matrix]]"
  - "[[Norm]]"
status: incomplete
---

# 线性空间 (Linear Space)

## 1. 数域 (Field)


- **定义 1.1 (数域)**: 设 $\mathbb{K}$ 是复数集 $\mathbb{C}$ 的子集, 且至少有两个不同的元素, 若 $\mathbb{K}$ 中任意两个元素的加法、减法、乘法和除法（除零外）都在 $\mathbb{K}$ 中, 则称 $\mathbb{K}$ 是一个数域.
  - *命题 1.1*: 有理数集, 实数集, 复数集都是数域. 
  - *命题 1.2*: 整数集 $\mathbb{Z}$ 不是数域, 因为它不满足除法的封闭性. 通常称关于加, 减, 乘封闭, 但除法不封闭的集合为环, 故整数集 $\mathbb{Z}$ 是一个数环 (Ring).
  - *例题*: 一些人为构造的数域的例子:
    - 所有形如 $a + b\sqrt{2}$ 的数, 其中 $a, b \in \mathbb{Q}$. 这个集合在加法、减法、乘法和除法（除零外）下都是封闭的. 对于这种扩展数域, 通常记为 $\mathbb{Q}(\sqrt{2})$.
    - 所有形如 $\frac{a_0 + a_1 \pi + \cdots + a_n \pi^n}{b_0 + b_1 \pi + \cdots + b_m \pi^m}$ 的数, 其中 $a_i, b_j \in \mathbb{Q}$ 且 $\exists b_j \neq 0$. 这个集合也是一个数域.
  - *命题 1.3*: 任意一个数域必须包含 $0$ 和 $1$, 且 $0 \neq 1$. 相同元素之差为 $0$, 相同非零元素之商为 $1$. 
  - *命题 1.4 (数域的等价定义)*: 设 $\mathbb{K}$ 是复数集 $\mathbb{C}$ 的子集, 且包含 $0$ 和 $1$, 则 $\mathbb{K}$ 是数域当且仅当 $\mathbb{K}$ 中任意两个元素的加法、减法、乘法和除法（除零外）都在 $\mathbb{K}$ 中.

- **定理 1.1 (数域必含 $\mathbb{Q}$)**: 任意数域 $\mathbb{K}$ 都包含有理数域 $\mathbb{Q}$. 换言之, 有理数域是最小的数域.
  - 证明: 
    - 因为任意数域必包含 $0$ 和 $1$, 故通过连加 $1$ 或连减 $1$ 可以得到任意整数.
    - 任意两两整数的商(分母不为零)构成全体有理数, 故 $\mathbb{K}$ 中也包含所有有理数.

- **命题 1.4**: 实数域 $\mathbb{R}$ 和复数域 $\mathbb{C}$ 间不存在其他任何数域.
  - 证明: 用反证法. 假设确存在某数域 $\mathcal{K}$, 且 $\mathbb{R} \subset \mathcal{K} \subset \mathbb{C}$. 
    - 设 $\mathcal{K} \neq \mathbb{R}$, 则定存在某复数 $a + b \mathrm{i} \in \mathcal{K}$, 其中 $a,b \in \mathbb{R}$ 且 $b \neq 0$.
    - 由于 $\mathcal{K}$ 是数域, 根据减法的封闭性, $(a + b \mathrm{i}) - a = b \mathrm{i} \in \mathcal{K}$.
    - 根据除法的封闭性, $\frac{b \mathrm{i}}{b} = \mathrm{i} \in \mathcal{K}$.
    - 由于 $1, \mathrm{i} \in \mathcal{K}$, 故对于任意 $c+d \mathrm i \in \mathbb{C}, c+d\mathrm i \in \mathcal{K}$. 即 $\mathcal{K} = \mathbb{C}$, 与假设矛盾.

## 2. 行向量与列向量 (Row and Column Vectors)


- **定义 2.1 (行向量)**
  - 列向量
  - 向量的加减法
  - 向量运算规则

## 3. 线性空间 (Linear Space)

- **定义 3.1 (线性空间)**: 设 $\mathbb{K}$ 是数域, $V$ 是非空集合, 在 $V$ 上定义了一个加法 $+$, 在 $V$ 和 $\mathbb{K}$ 上定义了数乘运算. 我们称 $V$ 为 $\mathbb{K}$ 上的线性空间或 $\mathbb{K}$-向量空间, 若满足如下 8 条运算规则:
    1. 加法交换律: $\forall \mathbf{u}, \mathbf{v} \in V, \mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$.
    2. 加法结合律: $\forall \mathbf{u}, \mathbf{v}, \mathbf{w} \in V, (\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$.
    3. 存在零向量: 存在 $\mathbf{0} \in V$, 使得 $\forall \mathbf{v} \in V, \mathbf{v} + \mathbf{0} = \mathbf{v}$.
    4. 存在负向量: 对于任意 $\mathbf{v} \in V$, 存在 $\mathbf{u} \in V$, 使得 $\mathbf{v} + \mathbf{u} = \mathbf{0}$. 将这样的 $\mathbf{u}$ 称为 $\mathbf{v}$ 的负向量, 通常记为 $-\mathbf{v}$.
    5. 存在单位元: 存在 $1 \in \mathbb{K}$, 使得 $\forall \mathbf{v} \in V, 1 \cdot \mathbf{v} = \mathbf{v}$.
    6. 分配律 1: $\forall a, b \in \mathcal{K}, \forall \mathbf{v} \in V, (a + b) \cdot \mathbf{v} = a \cdot \mathbf{v} + b \cdot \mathbf{v}$.
    7. 分配律 2: $\forall a \in \mathcal{K}, \forall \mathbf{u}, \mathbf{v} \in V, a \cdot (\mathbf{u} + \mathbf{v}) = a \cdot \mathbf{u} + a \cdot \mathbf{v}$.
    8. 结合律: $\forall a, b \in \mathcal{K}, \forall \mathbf{v} \in V, (a \cdot b) \cdot \mathbf{v} = a \cdot (b \cdot \mathbf{v})$.


- **命题 3.1**: 线性空间的零向量是唯一的.
  
- **命题 3.2**: 线性空间的负向量是唯一的.、


## 4. 向量的线性关系

- **定义 4.1 (线性组合)**: 设 $V$ 是 $\mathbb{K}$ 上的线性空间, $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n \in V, \mathrm{u}\in V$. 若 存在 $a_1, a_2, \ldots, a_n \in \mathbb{K}$, 使得 $\mathrm{u} = a_1 \mathbf{v}_1 + a_2 \mathbf{v}_2 + \cdots + a_n \mathbf{v}_n$, 则称 $\mathrm{u}$ 是 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$ 的线性组合.
  - 若对于方程组 $A\mathbf{x} = \mathbf{b}$, 则方程组有解当且仅当 $\mathbf{b}$ 是 $A$ 的列向量的线性组合.
  - 若记 $A = [\boldsymbol{\alpha}_1, \boldsymbol{\alpha}_2, \ldots, \boldsymbol{\alpha}_n]$, 则上述方程组还可以写成 $\mathbf{b} = x_1 \boldsymbol{\alpha}_1 + x_2 \boldsymbol{\alpha}_2 + \cdots + x_n \boldsymbol{\alpha}_n$, 其中 $x_i \in \mathbb{K}$ 是未知数.



- **定义 4.2 (线性相关与线性无关)**: 设 $V$ 是 $\mathbb{K}$ 上的线性空间, $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n \in V$. 若存在 $a_1, a_2, \ldots, a_n \in \mathbb{K}$, 使得 $a_1 \mathbf{v}_1 + a_2 \mathbf{v}_2 + \cdots + a_n \mathbf{v}_n = \mathbf{0}$ 且不全为零, 则称 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$ 是线性相关的. 反之则为线性无关的.
  - 线性相关/无关的定义中一定要规定数域 $\mathbb{K}$, 因为不同数域下的线性相关性可能不同. 例如考虑 $1$ 和 $\mathrm i = \sqrt{-1}$, 在 $\mathbb{R}$ 上它们是线性无关的 (找不到不全为零的 $a,b\in \mathbb{R}$ 使得 $a + b\mathrm i = 0$), 但在 $\mathbb{C}$ 上它们是线性相关的 (取 $a = 1, b = \mathrm i $ 即可).

- **例 4.5** 考虑 $\mathbb{K}$ 上的线性空间 $V$. 若向量组 $S$ 只含有一个向量 $\alpha$, 则 $S$ 是线性无关的当且仅当 $\alpha \neq \mathbf{0}$. 若向量组 $S$ 包含多个向量, 只要其中任意一个向量为零, 则 $S$ 必然是线性相关的.

- **定理 4.1 (向量组的子集的相关性)**: 若 $\alpha_1,\cdots, \alpha_n$ 是一组线性相关的向量, 则任意包含 $\alpha_1,\cdots, \alpha_n$ 的向量组都是线性相关的. 反之, 若 $\alpha_1,\cdots, \alpha_n$ 是线性无关的, 则从中任意取出一组向量, 该组向量都是线性无关的.

- **定理 4.2 (线性相关与线性表示)**: 设 $V$ 是 $\mathbb{K}$ 上的线性空间, $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n \in V$. 则以下命题等价:
  1. $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$ 是线性相关的.
  2. 存在 $a_1, a_2, \ldots, a_n \in \mathbb{K}$, 使得 $a_1 \mathbf{v}_1 + a_2 \mathbf{v}_2 + \cdots + a_n \mathbf{v}_n = \mathbf{0}$ 且不全为零.
  3. 存在 $\mathbf{v}_i$ (其中 $1 \leq i \leq n$), 使得 $\mathbf{v}_i$ 可以表示为其他向量的线性组合, 即 $\mathbf{v}_i = a_1 \mathbf{v}_1 + a_2 \mathbf{v}_2 + \cdots + a_{i-1} \mathbf{v}_{i-1} + a_{i+1} \mathbf{v}_{i+1} + \cdots + a_n \mathbf{v}_n$.

- **定理 4.3 (线性表示的唯一性)**: 设 $V$ 是 $\mathbb{K}$ 上的线性空间, $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n, \mathbf{u} \in V$. 若 $\mathbf{u} = a_1 \mathbf{v}_1 + a_2 \mathbf{v}_2 + \cdots + a_n \mathbf{v}_n$, 则该表示唯一 (即 $a_1, a_2, \ldots, a_n$ 唯一确定) 当且仅当 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$ 是线性无关的.


- **定理4.4 (线性表示的传递性)**: 设有三个向量组 $A, B, C$, 满足: $A$ 中的任意向量都可以表示为 $B$ 中向量的线性组合, 且 $B$ 中的任意向量都可以表示为 $C$ 中向量的线性组合. 则 $A$ 中的任意向量也可以表示为 $C$ 中向量的线性组合.

## 5. 向量组的秩

- **定义 5.1 (极大线性无关组)**: 设 $V$ 是 $\mathbb{K}$ 上的线性空间, 向量的集合称为向量族, 其中有限的集合称为向量组. 对于向量族 $S$, 若 $S$ 中存在一组向量 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_r$, 满足:
  1. $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_r$ 是线性无关的.
  2. 对于 $S$ 中的任意向量 $\mathbf{v}$, 都可以表示为 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_r$ 的线性组合.

  则称 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_r$ 是 $S$ 的一个极大线性无关组.
    - 这里的“极大”指的是在 $S$ 中找不到比 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_r$ 更多的线性无关向量. 任意向该组添加一个向量都会使得该组线性相关.

- **命题 5.1**: 设 $S$ 是一个向量组且至少包含一个非零向量, 则 $S$ 中一定存在一个极大线性无关组.

- **例 5.1**: 一般而言, $S$ 的极大线性无关组不唯一. 但包含的向量个数是唯一的.

- **引理 5.1**: 若 $A,B$ 是 $V$ 中的两组向量且 $A$ 中的任意向量都可以表示为 $B$ 中向量的线性组合. 若 $A$ 是线性无关的, 则 $A$ 中的向量个数定不超过 $B$ 中的向量个数. 反之, 若 $A$ 中的元素个数多于 $B$ 中的元素个数, 则 $A$ 中的向量必然是线性相关的.

- **引理 5.2**: 若 $A,B$ 均值线性无关的向量组, 且 $A$ 中的任意向量都可以表示为 $B$ 中向量的线性组合, $B$ 中的任意向量也可以表示为 $A$ 中向量的线性组合, 则 $A$ 和 $B$ 的向量个数相同.

- **定理 5.1**: 若 $A,B$ 均为 $S$ 的极大线性无关组, 则 $A$ 和 $B$ 的向量个数相同. 

- **定义 5.2 (向量组的秩)**: 设 $V$ 是 $\mathbb{K}$ 上的线性空间, 向量组 $S$ 的秩 (Rank) 定义为 $S$ 的任意极大线性无关组的向量个数. 通常记为 $\mathrm{rank}(S)$.

- **定义 5.3 (向量组等价)**: 若向量组 $A$ 和 $B$ 可以互相线性表示, 则称 $A$ 和 $B$ 是等价的.

- **定理 5.2 (向量组的秩与等价关系)**: 等价的向量组具有相同的秩.

- **定义 5.4 (线性空间的基)**: $\mathbb{K}$ 上的线性空间 $V$ 的极大线性无关组称为该线性空间的基 (Basis). 基的个数 $n$ 称为该线性空间的维数 (Dimension), 通常记为 $\mathrm{dim}_{\mathbb{K}}(V) = n$. 如果不能用有限个基向量线性表示 $V$ 中的任意向量, 则称 $V$ 是无限维的.

- **例 5.3**: 将复数域 $\mathbb{C}$ 视为实数域 $\mathbb{R}$ 上的线性空间, 则 $\{1, \mathrm{i}\}$ 是 $\mathbb{C}$ 的一个基, 且 $\mathrm{dim}_{\mathbb{R}}(\mathbb{C}) = 2$.

- **推论 5.1**: $n$ 维线性空间 $V$ 中任意超过 $n$ 个向量的向量组都是线性相关的.

- **定理 5.3**: $n$ 维线性空间中 $e_1, e_2, \ldots, e_n$ 是一组向量. 若这组向量是线性无关的, 或满足 $V$ 中的任意向量都可以表示为 $e_1, e_2, \ldots, e_n$ 的线性组合, 则称这组向量是 $V$ 的一个基.
- **定理 5.4 (基的扩张定理)**: 设 $V$ 是 $n$ 维线性空间, $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ 是 $V$ 中 $k<n$ 个线性无关的向量. 又假设 $e_1, e_2, \ldots, e_n$ 是 $V$ 的一个基, 则在这组基中定存在 $n-k$ 个向量, 使其和 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ 结合在一起, 构成 $V$ 的一个基.

## 6. 矩阵的秩


## 7. 坐标向量


## 8. 基变换与过渡矩阵


## 9. 子空间


## 10. 线性方程组的解