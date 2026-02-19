---
aliases: [矩阵, Matrix, 矩阵运算, 矩阵的秩]
tags:
  - concept
  - math/linear-algebra
related_concepts:
  - "[[Linear_Space]]"
  - "[[Determinant]]"
  - "[[Rank_and_Eigenvalue]]"
  - "[[Column_Space_of_Matrix]]"
  - "[[Cholesky_Decomposition]]"
  - "[[SMW_Formula]]"
source: "线性代数基础"
---

# 矩阵 (Matrix)

## 1. 矩阵的概念

- **定义 1.1 (矩阵)**: 
  - 实矩阵
  - $\boldsymbol{O}$ 零矩阵
  - 对角矩阵
  - $n$ 阶单位矩阵 $\boldsymbol{I}_n$
  - 上/下三角矩阵
  - 矩阵的相等
  - 向量



## 2. 矩阵的运算 


- **定义 2.1 (矩阵的加法)**: 
  - 矩阵的减法
  - 矩阵的加减运算规则: 交换律, 结合律, 零矩阵, 相反矩阵.

- **定义 2.2 (矩阵的数乘)**: 
  - 矩阵的数乘运算规则
    - 分配律1: $k(A + B) = kA + kB$.
    - 分配律2: $(k + m)A = kA + mA$.
    - 结合律: $k(mA) = (km)A$.
    - 单位元: $1A = A$.
    - 零元: $0A = \boldsymbol{O}$.
- **定义 2.3 (矩阵的乘法)**:
  - 矩阵乘法的定义: $C = AB$ 的第 $i$ 行第 $j$ 列元素为 $c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$ (同样符合前行后列的原则).
  - 矩阵乘法的性质:
    - 不满足交换律, 即 $AB \neq BA$.
    - 满足结合律, 即 $(AB)C = A(BC)$.
    - 满足分配律, 即 $A(B + C) = AB + AC$ 和 $(A + B)C = AC + BC$.
    - 单位矩阵: 对任意矩阵 $A$, 有 $A\boldsymbol{I}_n = A$ 和 $\boldsymbol{I}_n A = A$.
    - 方阵的幂次: $A^k = A \cdot A \cdots A$ ($k$ 次).
      - $(A^k)^m = A^{km}$.
      - $A^k A^m = A^{k+m}$.
- **定义 2.4 (矩阵的转置)**: 
  - 矩阵转置的性质:
    - $(A^\top)^\top = A$.
    - $(AB)^\top = B^\top A^\top$.
    - $(A + B)^\top = A^\top + B^\top$.
    - $(kA)^\top = kA^\top$.
- **定义 2.5 (矩阵的共轭)**: 
  - 矩阵的共轭是指对矩阵中的每个元素取复共轭, 记作 $A^*$.

## 3. 方阵的逆矩阵

- **定义 3.1 (方阵的逆矩阵)**: 
  - 如果存在矩阵 $B$ 使得 $AB = BA = \boldsymbol{I}_n$, 则称 $B$ 为 $A$ 的逆矩阵, 记为 $A^{-1}$.
  - *断言*: 如果 $A$ 的某一行/列全部为零, 则 $A$ 不可逆.
  - **性质** (假设均为 $n$ 阶方阵且逆矩阵存在): 
    - $(A^{-1})^{-1} = A$.
    - $(AB)^{-1} = B^{-1}A^{-1}$. 更一般地, 对于 $k$ 个矩阵的乘积, 有 $(A_1 A_2 \cdots A_k)^{-1} = A_k^{-1} A_{k-1}^{-1} \cdots A_1^{-1}$.
    - $(A^\top)^{-1} = (A^{-1})^\top$.
    - $(cA)^{-1} = \frac{1}{c}A^{-1}$, 其中 $c \neq 0$.

下面是利用一个利用伴随矩阵求逆矩阵的方法.

- **定义 3.2 (方阵的伴随矩阵, adjoint matrix)**: 方阵 $A$ 的伴随矩阵 $A^*$ 
  - $A^*$ 的第 $i$ 行第 $j$ 列元素为 $(-1)^{i+j} M_{ji}$, 其中 $M_{ji}$ 是 $A$ 中去掉第 $j$ 行和第 $i$ 列后的余子式.
- **引理 3.1 (伴随矩阵的性质)**: 对于 $n$ 阶方阵, $AA^* = A^*A = |A| \boldsymbol{I}_n$.
- **定理 3.1 (伴随矩阵与逆矩阵)**: 对于 $n$ 阶方阵 $A$, 如果 $|A| \neq 0$, 则 $A$ 可逆, 且其逆矩阵为:
  $$A^{-1} = \frac{1}{|A|} A^*$$
  - **注**: 判断 $A$ 是否可逆的一个方法是计算其行列式 $|A|$ 是否为零.

## 4. 矩阵的初等变换与初等矩阵

### 高斯消去法与矩阵的初等变换

- 高斯消去法解线性方程组
  1. 用矩阵形式表示线性方程组: $A\mathbf{x} = \mathbf{b}$ 并写出增广矩阵 $\tilde{A} = [A | \mathbf{b}]$.
  2. 将 $\tilde{A}$ 的某一行调换到第一行, 使得第一行第一列的元素不为零.
  3. 对第一行进行数乘并加到其他行上, 除了第一行第一列的元素外, 其他行第一列的元素都变为零.
  4. 重复上述步骤, 使得第 $i$ 行第 $i$ 列的元素为 $1$, 且第 $i$ 行之前的所有行第 $i$ 列的元素都为零. (即$A$变成对角矩阵)

- **定义 4.1 (初等变换, elementary transformation)**: 矩阵的初等变换是指对矩阵进行以下三种操作之一:
  1. 互换两行或两列.
  2. 将某一行或某一列乘以一个非零常数.
  3. 将某一行或某一列加上另一行或另一列的某个倍数.

- **定义 4.2 (矩阵相抵, matrix equivalence)**: 两个矩阵 $A$ 和 $B$ 相抵, 如果存在有限个初等变换, 使得 $A$ 变为 $B$ 或 $B$ 变为 $A$. 记为 $A \simeq B$.
  - 矩阵相抵是一个等价关系:
    - 自反性: 对任意矩阵 $A$, 有 $A \simeq A$.
    - 对称性: 如果 $A \simeq B$, 则 $B \simeq A$.
    - 传递性: 如果 $A \simeq B$ 且 $B \simeq C$, 则 $A \simeq C$.
- **定理 4.1 (任意矩阵的相抵形式)**: 任意 $m\times n$ 矩阵相抵于如下矩阵. 若引入初等矩阵概念,则可以描述为: 对于任意矩阵 $A$, 存在初等矩阵 $P_1, P_2, \ldots, P_s, Q_1, Q_2, \ldots, Q_t$ 使得: 
      $$ P_s \cdots P_2 P_1 A  Q_1 \cdots Q_t = 
      \begin{pmatrix}  
        1 &  &  & & &\\
        &  \ddots & & & &\\
        & & 1&  & &\\
        & & & 0 &  &\\
        & & & & \ddots &\\
        & & & & & 0 \end{pmatrix} $$
        其中 $1$ 的个数恰恰是矩阵 $A$ 的秩, $0$ 的个数恰恰是矩阵 $A$ 的零空间的维数.


- **定义 4.3 (矩阵的阶梯点与阶梯矩阵)**

- **定理 4.2 (任意矩阵的阶梯形式)**: 对于任意 $m\times n$ 矩阵 $A$, 经过若干次**行**初等变换后, 可以得到一个阶梯矩阵. 

### 初等矩阵

- **定义 4.4 (初等矩阵, elementary matrix)**: 初等矩阵是指通过对单位矩阵进行一次初等变换得到的矩阵.

- **定理 4.3 (初等矩阵与矩阵的初等变换)**: 对任意矩阵 $A$, 左乘一个初等矩阵相当于对 $A$ 进行行的初等变换; 右乘一个初等矩阵相当于对 $A$ 进行列的初等变换.
- **推论 4.1 (初等矩阵的性质)**: 初等矩阵都是可逆矩阵, 且其逆也是初等矩阵.
- **推论 4.2 (矩阵的可逆性与初等变换)**: 可逆矩阵经过初等变换后仍然是可逆矩阵, 非可逆矩阵经过初等变换后仍然是非可逆矩阵.



## 5. 矩阵乘积的行列式与初等变换法求逆矩阵

### 乘积的行列式

- **引理 5.1 (可逆矩阵与单位阵)**: 对于 $n$ 阶可逆矩阵 $A$, 仅用初等行或仅用初等列变换, 可以将 $A$ 化为单位矩阵 $\boldsymbol{I}_n$.

- **推论 5.1 (可逆矩阵的初等矩阵表示)**: 任意 $n$ 阶可逆矩阵 $A$ 可以表示为有限个初等矩阵的乘积, 即 $A = E_1 E_2 \cdots E_k$, 其中每个 $E_i$ 都是初等矩阵.

- **引理 5.2**: 对于 $n$ 阶方阵 $A$, $n$ 阶初等矩阵 $E$: $|AE|=|EA| = |A|$.
  
- **定理 5.1 (可逆矩阵与行列式)**: 对于 $n$ 阶方阵 $A$ 可逆的充要条件是 $|A| \neq 0$.
  - **证明**: 
    - [必要性]: 由推论 5.1, 若 $A$ 可逆, 则存在初等矩阵 $E$, 使得 $A = E_1 E_2 \cdots E_k$, 则由引理 5.2, 有 $|A| = |E_1| |E_2| \cdots |E_k| \neq 0$.
    - [充分性]: 由定理 3.1 可知.

- **定理 5.2 (矩阵乘积的行列式)**: 对于 $n$ 阶方阵 $A, B$, 有 $|AB| = |A||B|$.
  - **证明**: 
    - 若 $A$ 非奇异, 则定能将 $A$ 表示为初等矩阵的乘积, 再利用引理 5.2 即证.
    - 若 $A$ 奇异, 则 $|A| = 0$, 故只需证 $|AB| = 0$ 即可. 而 $A$ 可以表示为 $PAQ = D$, 其中 $D$ 是相抵标准型, 且由于奇异 $A$, 导致 $D$ 至少最后一行全为零. 故 $DQ^{-1} = PA$ 的第 $n$ 行全为零, 故 $PAB$ 的第 $n$ 行全为零, 即 $|AB| = 0$.

- **推论 5.2 (方阵的乘积与奇异性)**: 奇异矩阵乘以任意矩阵仍然是奇异矩阵; 非奇异矩阵乘以非奇矩阵仍然是非奇异矩阵.

- **推论 5.3 (非奇异矩阵的逆矩阵)**: 对于非奇异矩阵 $A$, 有 $|A^{-1}| = \frac{1}{|A|}$.

- **推论 5.4 (逆矩阵的唯一性)**: 对于 $n$ 阶方阵 $A,B$, 若 $AB = I_n$ 或 $BA = I_n$, 则 $B = A^{-1}$.
  - 证明:
    - 首先由 $|AB| = |A||B| = |I_n| = 1$, 故 $|A| \neq 0$ 且 $|B| \neq 0$, $A$ 存在逆矩阵, 记为 $C$.
    - 则 $B = IB = CAB = C(AB) = C I_n = C$, 即 $B = A^{-1}$.

- **例 5.2**: 计算行列式 $\begin{vmatrix} x & -y & -z & w \\ y & x & -w & z \\ z & w & x & -y \\ w & -z & y & x \end{vmatrix}$.
  - 解:
    - 观察发现, 该行列式对应的矩阵 (记为 $A$) 对称位置的元素互为相反数, 具有特殊结构. 故尝试计算 $AA^\top = \begin{pmatrix} u^2 & 0 & 0 & 0 \\ 0 & u^2 & 0 & 0 \\ 0 & 0 & u^2 & 0 \\ 0 & 0 & 0 & u^2 \end{pmatrix}$, 其中 $u = \sqrt{x^2 + y^2 + z^2 + w^2}$.
    - 由行列式的性质, 有 $|AA^\top| = |A|^2 = |A|^2 = u^8$, 故 $|A| = u^4 = (x^2 + y^2 + z^2 + w^2)^2$.

- *命题 5.1 (不可逆矩阵的条件)*: $A \in \mathbb{R}^{n\times n}$ 是不可逆矩阵的充要条件是: 存在 $B \in \mathbb{R}^{n\times n}, B\neq \boldsymbol{0}$, 使得 $A B = \boldsymbol{0}$.
  - [必要性]: 用反证法, 若 $A$ 可逆, 则存在 $B = A^{-1} \boldsymbol{0} = \boldsymbol{0}$, 与 $B \neq \boldsymbol{0}$ 矛盾.
  - [充分性]: 由于 $A$ 不可逆, 则其不是满秩矩阵, 即存在可逆矩阵 $P,Q$, 使得 $PAQ = \begin{pmatrix}  I_r &  0 \\0 & 0 \end{pmatrix}$. 又令 $C = \begin{pmatrix}  0 & 0 \\0 & I_{n-r} \end{pmatrix}$, 则 $(PAQ)C = \boldsymbol{0}$, 即 $A(QC) = \boldsymbol{0}$, 取 $B = QC \neq \boldsymbol{0}$.


- *命题 5.2 (矩阵可逆的等价条件)*:  对于 $n$-阶矩阵 $A$, 如下六个命题等价:
  1. $A$ 可逆.
  2. $|A| \neq 0$.
  3. $\exists B \in \mathbb{R}^{n\times n}, AB = I_n$.
  4. $\exists B \in \mathbb{R}^{n\times n}, BA = I_n$.
  5. $A \simeq I_n$.
  6. $A$ 可以表示为若干个初等矩阵的乘积.

### 初等变换求逆矩阵

- 初等变换法求逆矩阵的步骤:
  1. 将矩阵 $A$ 与单位矩阵 $\boldsymbol{I}_n$ 拼接成增广矩阵 $\tilde{A} = [A | \boldsymbol{I}_n]$.
  2. 对增广矩阵进行初等行变换, 直到左边的部分变为单位矩阵 $\boldsymbol{I}_n$.
  3. 此时右边的部分即为 $A^{-1}$.
  
## 6. 分块矩阵

- 分块矩阵的定义
- 分块矩阵的运算
  - 加减法
  - 数乘
  - 矩阵乘法
- **例 6.2 (分块对角矩阵的乘法)**: 设两个分块对角矩阵 $A = \text{diag}(A_1, \cdots, A_k)$ 和 $B = \text{diag}(B_1, \dots, B_k)$, 则 $AB = \text{diag}(A_1B_1, \dots, A_kB_k)$.

- **例 6.3 (分块对角矩阵的可逆性)**: 设 $A = \text{diag}(A_1, \cdots, A_k)$ 是一个分块对角矩阵, 且每一块 $A_i$ 都是可逆的, 则 $A$ 可逆, 且 $A^{-1} = \text{diag}(A_1^{-1}, \cdots, A_k^{-1})$.

- *命题 6.2 (分块矩阵的转置)*:
  - 对于分块矩阵 $A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$, 其转置为 $A^\top = \begin{pmatrix} A_{11}^\top & A_{21}^\top \\ A_{12}^\top & A_{22}^\top \end{pmatrix}$. 更高维度的分块矩阵同理.

- **引理 6.1 (分块上/下三角矩阵的行列式)** 设 $A,C$ 分别是 $m,n$ 阶方阵, 则对分块上(下)三角矩阵 $G = \begin{pmatrix} A & B \\ O & C \end{pmatrix}$, $H = \begin{pmatrix} A & O \\ B & C \end{pmatrix}$, 有: $|G| = |A||C|$ 和 $|H| = |A||C|$.


- **定理 6.1 (行列式的降阶公式)**: 设 $A \in \mathbb{R}^{m \times m}$ 且可逆, $D \in \mathbb{R}^{n \times n}$, 则 $\begin{vmatrix} A & B \\ C & D \end{vmatrix} = |A||D - CA^{-1}B|$. 
  - 证明: 构造 $\begin{pmatrix} I_m & 0 \\ -CA^{-1} & I_n \end{pmatrix} \begin{pmatrix} A & B \\ C & D \end{pmatrix} = \begin{pmatrix} A & B \\ 0 & D - CA^{-1}B \end{pmatrix}$, 而行列式不变, 即 $\begin{vmatrix} A & B \\ C & D \end{vmatrix} = |A| |D - CA^{-1}B|$.
  - 类似的, 若 $D$ 可逆, 则 $\begin{vmatrix} A & B \\ C & D \end{vmatrix} = |D||A - BD^{-1}C|$.

- *命题 6.2 (矩阵乘法的行列式)*: 设 $A \in \mathbb{R}^{n \times m}, B\in \mathbb{R}^{m \times n}$, 则 $|I_n - AB| = |I_m - BA|$.
  - 证明: 构造两个上下三角矩阵: $\begin{pmatrix} I_n & \boldsymbol{0} \\ -B & I_m \end{pmatrix} \begin{pmatrix} I_n & A \\ B & I_m \end{pmatrix} = \begin{pmatrix} I_n & A \\ \boldsymbol{0} & I_m - BA \end{pmatrix}$, 和 $\begin{pmatrix} I_n & -A \\ \boldsymbol{0} & I_m \end{pmatrix} \begin{pmatrix} I_n & A \\ B & I_m \end{pmatrix} = \begin{pmatrix} I_n - AB & A \\ \boldsymbol{0} & I_m \end{pmatrix}$. 对两式左右同取行列式, 得到 $|I_n - AB| = |I_m - BA|$.
  - 其本质上揭示了 **$AB$ 和 $BA$ 具有相同的非零特征值**. 几何上看, 二者相当于进行 $x \to Bx \to A(Bx)$ 和 $x \to Ax \to B(Ax)$ 的变换, 这两种变换在特征值的非零部分上是等价的.
  - 尤其在一些计算向量外积的场景中可以简化计算: $|I_n - \mathbf{u}\mathbf{v}^\top| = |1 - \mathbf{v}^\top \mathbf{u}|$, 其中 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$.

- **例 6.5 (分块矩阵的逆)**: 已知 $A,D$ 是可逆矩阵, 求 $\begin{pmatrix} A & B \\ O & D \end{pmatrix}$ 的逆矩阵.
  - 解: 通过构造分块增广矩阵 $\begin{pmatrix} A & B & I & O \\ O & D & O & I \end{pmatrix}$, 进行初等行变换, 最终得到 $\begin{pmatrix} A^{-1} & -A^{-1}B D^{-1} \\ O & D^{-1} \end{pmatrix}$.

## 7. Cauchy-Binet 定理

- **定理 7.1 (Cauchy-Binet 定理)**: 设 $A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times m}$, 则有:
  - 若 $m > n$, 则定有 $|AB| = 0$. (此时 $AB$ 是一个 $m \times m$ 的矩阵, 定不满秩)
  - 若 $m \leq n$, 则有:
    $$|AB| = \sum_{I \subseteq \{1, 2, \ldots, n\}, |I| = m} |A_I||B_I|$$
    其中 $I$ 是从 $\{1, 2, \ldots, n\}$ 中选取 $m$ 个元素的集合, $A_I$ 和 $B_I$ 分别是 $A$ 和 $B$ 在这些列上的子矩阵.
  - 该公式的核心思想在于, 当两个矩阵相乘得到一个方阵时, 这个方阵的行列式可以表示为原矩阵所有可能的 "匹配" 到的子矩阵的行列式之和

- **定理 7.2 (Cauchy-Binet 定理的推广)**: 设 $A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times m}$, 且有正整数 $r\leq m$.
  - 若 $r>n$, 则 $AB$ 的任意 $r$ 阶子式的行列式为零. (当前 $AB$ 最大的可能秩为 $n$, 故 $r>n$ 时子式必为零)
  - 若 $r \leq n$, 则有:
    $$ \det(AB)_r = \sum_{I \subseteq \{1, 2, \ldots, n\}, |I| = r} |A_I||B_I| $$
    即 $AB$ 的任意 $r$ 阶子式的行列式可以表示为 $A$ 和 $B$ 在这些列上的子矩阵的行列式之和.


- **推论 7.1 (Cauchy-Binet 定理的应用)**: 设 $A \in \mathbb{R}^{m \times n}$, 则 $AA^\top$ 的任意主子式均非负.

- **例 7.1 (Lagrange 恒等式)**: 对于 $n\ge 2$, 恒有:
  $$ (\sum_{i=1}^{n} a_i^2)(\sum_{i=1}^{n} b_i^2) - (\sum_{i=1}^{n} a_i b_i)^2 = \sum_{1\le i < j \le n} (a_i b_j - a_j b_i)^2 $$
  - 证明:
    - LHS 等价于 $\begin{vmatrix} \sum a_i^2 & \sum a_i b_i \\ \sum a_i b_i & \sum b_i^2 \end{vmatrix}$, 对应形式 $\det(AA^\top)$, 其中 $A = \begin{pmatrix} a_1 & a_2 & \cdots & a_n \\ b_1 & b_2 & \cdots & b_n \end{pmatrix}$.
    - 由 Cauchy-Binet 定理, $\det(AA^\top) = \sum_{1\le i<j \leq n} \begin{vmatrix} a_i & a_j \\ b_i & b_j \end{vmatrix} \begin{vmatrix} a_i & b_i \\ a_j & b_j \end{vmatrix} = \sum_{1\le i < j \le n} (a_i b_j - a_j b_i)^2$.

- **例 7.2 (Cauchy-Schwarz 不等式)**: 对于任意 $n$ 个实数 $a_1, a_2, \ldots, a_n$ 和 $b_1, b_2, \ldots, b_n$, 恒有:
  $$ (\sum_{i=1}^{n} a_i^2)(\sum_{i=1}^{n} b_i^2) \geq (\sum_{i=1}^{n} a_i b_i)^2 $$
  - 证明: 根据 Lagrange 恒等式, 其 RHS 非负可证.

## The Art of Linear Algebra: 矩阵篇

**一个矩阵 $(m\times n)$ 可以看作是：**
- 1 个矩阵
- $mn$ 个数
- $n$ 个列
- $m$ 个行

$$
A = 
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix} = 
\begin{pmatrix}
\lvert & \lvert &  & \lvert \\
\mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_n\\
\lvert & \lvert &  & \lvert
\end{pmatrix} = 
\begin{pmatrix} -\mathbf{a}^\top_1- \\- \mathbf{a}^\top_2 - \\
\vdots \\ - \mathbf{a}^\top_m -\end{pmatrix}
$$

**一个向量 $\mathrm v_1$ 乘以一个向量 $\mathrm v_2$ (均为 $n$ 维列向量), 可以得到:**
- 一个数 (点积): $\mathrm v_1^\top \mathrm v_2$
  $$\begin{pmatrix} 1 & 2 & 3 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} = 1x_1 + 2x_2 + 3x_3$$
- 一个 $\text{rank}=1$ 的矩阵 (外积): $\mathrm v_1 \mathrm v_2^\top$
  $$\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} \begin{pmatrix} x_1 & x_2 & x_3 \end{pmatrix} = \begin{pmatrix} 1x_1 & 1x_2 & 1x_3 \\ 2x_1 & 2x_2 & 2x_3 \\ 3x_1 & 3x_2 & 3x_3 \end{pmatrix}$$

**一个矩阵右乘以一个向量 ($A\mathbf{x}$), 可以看作是:**
- $A$ 的每一行与向量的点积 (方程视角)
  $$A \mathbf{x} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 1x_1 + 2x_2 \\ 3x_1 + 4x_2 \\ 5x_1 + 6x_2 \end{pmatrix}$$

- $A$ 的每一列的线性组合, 权重由向量的元素给出 (线性变换视角)
  $$A \mathbf{x} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = x_1 \begin{pmatrix} 1 \\ 3 \\ 5 \end{pmatrix} + x_2 \begin{pmatrix} 2 \\ 4 \\ 6 \end{pmatrix}$$
  - 矩阵 $A$ 的列向量的所有线性组合构成了一个子空间, 记为 $\text{Col}(A)$, 称为 $A$ 的列空间.
  - $A\mathbf{x} = 0$ 的解集构成的子空间称为 $A$ 的零空间, 记为 $\text{Nul}(A)$.

**一个矩阵左乘以一个向量 ($\mathbf{y}^\top A$), 可以看作是:**
- $A$ 的每一列与该向量的点积
  $$\mathbf{y}^\top A = \begin{pmatrix} y_1 & y_2 & y_3 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix} = \begin{pmatrix} y_1 + 3y_2 + 5y_3 & 2y_1 + 4y_2 + 6y_3 \end{pmatrix}$$
- $A$ 的每一行的线性组合, 权重由向量的元素给出
  $$\mathbf{y}^\top A = \begin{pmatrix} y_1 & y_2 & y_3 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix} = y_1 \begin{pmatrix} 1 & 2 \end{pmatrix} + y_2 \begin{pmatrix} 3 & 4 \end{pmatrix} + y_3 \begin{pmatrix} 5 & 6 \end{pmatrix}$$
  - 矩阵 $A$ 的行向量的所有线性组合构成了一个子空间, 若沿用 $\text{Col}(A)$ 的记法, 则记为 $\text{Col}(A^\top)$, 称为 $A$ 的行空间.
  - $\mathbf{y}^\top A = 0$ 的解集构成的子空间称为 $A$ 的左零空间, 记为 $\text{Nul}(A^\top)$.

**矩阵 $A$ 的四个基本子空间:**
- 列空间 $\text{Col}(A)$: 由 $A$ 的列向量的所有线性组合构成的子空间.
- 行空间 $\text{Col}(A^\top)$: 由 $A$ 的行向量的所有线性组合构成的子空间.
- 零空间 $\text{Nul}(A)$: 由 $A\mathbf{x} = 0$ 的所有解构成的子空间.
- 左零空间 $\text{Nul}(A^\top)$: 由 $\mathbf{y}^\top A = 0$ 的所有解构成的子空间.

在 $\mathbb{R}^n$ 中的 $\text{Col}(A)$ 和 $\text{Nul}(A)$ 是正交补关系; 在 $\mathbb{R}^m$ 中的 $\text{Col}(A^\top)$ 和 $\text{Nul}(A^\top)$ 也是正交补关系.

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250723174157.png)

**一个矩阵乘以一个矩阵 ($AB$), 可以看作是:**
- $A$ 的每一行与 $B$ 的每一列的点积
  $$AB = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix} \begin{pmatrix} x_1 & y_1 \\ x_2 & y_2 \end{pmatrix} = \begin{pmatrix} 1x_1 + 2x_2 & 1y_1 + 2y_2 \\ 3x_1 + 4x_2 & 3y_1 + 4y_2 \\ 5x_1 + 6x_2 & 5y_1 + 6y_2 \end{pmatrix}$$
- $A$ 与 $B$ 的列向量的组合
  $$AB = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix} \begin{pmatrix} x_1 & y_1 \\ x_2 & y_2 \end{pmatrix} = A \begin{pmatrix} \boldsymbol{x} & \boldsymbol{y} \end{pmatrix} = \begin{pmatrix} A\boldsymbol{x} & A\boldsymbol{y} \end{pmatrix}$$
- $B$ 与 $A$ 的行向量的组合
  $$AB = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix} \begin{pmatrix} x_1 & y_1 \\ x_2 & y_2 \end{pmatrix} = \begin{pmatrix} -\boldsymbol{a}^\top_1 - \\ -\boldsymbol{a}^\top_2 - \\ \vdots \\ -\boldsymbol{a}^\top_5 - \end{pmatrix} B = \begin{pmatrix} -\boldsymbol{a}^\top_1 B - \\ -\boldsymbol{a}^\top_2 B - \\ \vdots \\ -\boldsymbol{a}^\top_5 B - \end{pmatrix}$$
- 一系列 $\text{rank}$-1 矩阵的和:
  $$AB = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix} \begin{pmatrix} x_1 & y_1 \\ x_2 & y_2 \end{pmatrix} = \begin{pmatrix} \mathbf{a}_1 & \mathbf{a}_2 \end{pmatrix} \begin{pmatrix} -\mathbf{b}^\top_1 - \\ -\mathbf{b}^\top_2 - \end{pmatrix} = \begin{pmatrix} \mathbf{a}_1 \mathbf{b}^\top_1 + \mathbf{a}_2 \mathbf{b}^\top_2 \end{pmatrix} = \begin{pmatrix} 1x_1 & 1y_1 \\ 3x_1 & 3y_1 \\ 5x_1 & 5y_1 \end{pmatrix} + \begin{pmatrix} 2x_2 & 2y_2 \\ 4x_2 & 4y_2 \\ 6x_2 & 6y_2 \end{pmatrix}$$