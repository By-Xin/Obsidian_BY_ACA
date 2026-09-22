# Lecture 1: Convex Optimization

> References:
> - [Boyd] Stephen Boyd and Lieven Vandenberghe. *Convex Optimization*. Cambridge University Press,2004. Relevant parts for this lecture: Chapter 2 (convex sets, separation, dual cones), §5.8 (theorems of alternatives), §8.1 (projection).
> - [Nemirovski] Aharon Ben-Tal and Arkadi Nemirovski. *Lectures on Modern Convex Optimization: Analysis, Algorithms, and Engineering Applications*. MPS-SIAM Series on Optimization, SIAM, 2001. Relevant parts for this lecture: Lecture 1 and Appendix B, especially B.1 and B.2.1–B.2.9.
> - [Dimitris] Dimitris Bertsimas and John N. Tsitsiklis. *Introduction to Linear Optimization*. Athena Scientific, 1997. Relevant parts for this lecture: Chapter 2 (geometry of LP) and Chapter 4 §4.6–4.9 (Farkas, separation, cones, extreme rays, resolution theorem).
> - [Ye] David G. Luenberger and Yinyu Ye. *Linear and Nonlinear Programming*, 5th ed. Springer, 2021. Relevant parts for this lecture: Chapter 2 (BFS, fundamental theorem, Farkas), Chapter 6 (convex cones), Appendix B (convex sets and separation)
> - [Nesterov] Nesterov, Yurii. Lectures on Convex Optimization. Vol. 137. Springer Optimization and Its Applications. Springer International Publishing, 2018.
> - [Beck] Beck, Amir. *First-Order Methods in Optimization*. Society for Industrial and Applied Mathematics, 2017. Chapter 2.


<!-- > | Topic | Boyd–Vandenberghe | Ben-Tal–Nemirovski | Bertsimas–Tsitsiklis | Luenberger–Ye |
> |---|---|---|---|---|
> | Affine/convex sets, hull | Ch. 2 §2.1–2.3 | App. B §B.1 | Ch. 2 §2.1 | App. B §B.1 |
> | Carathéodory | background/exercises | App. B §B.2.1 | Ch. 2 / Ch. 4 exercises | Ch. 2 fundamental theorem link |
> | Topology, relative interior | Ch. 2 §2.1 | App. B §B.1.6 | Ch. 4 §4.7 background | App. A/B |
> | Projection | Ch. 8 §8.1 | separation machinery | Ch. 4 §4.7 proof route | implicit in separation appendix |
> | Separation/support | Ch. 2 §2.5 | App. B §B.2.6 | Ch. 4 §4.7 | App. B §B.3 |
> | Farkas/alternatives | Ch. 5 §5.8 | App. B §B.2.5 | Ch. 4 §4.6–4.7 | Ch. 2 §2.6 |
> | Cones/dual cones | Ch. 2 §2.6 | Lecture 1 + App. B §B.2.7 | Ch. 4 §4.8 | Ch. 6 §6.1 |
> | Extreme points/BFS | polyhedra background | App. B §B.2.8 | Ch. 2 §2.2–2.6 | Ch. 2 §2.3–2.5 |
> | Extreme rays/recession | indirect/conic language | App. B §B.2.9 | Ch. 4 §4.8 | Ch. 2 + Ch. 6 |
> | Minkowski–Weyl | convex hull description | App. B §B.2.9 | Ch. 4 §4.9 | LP geometry discussion | -->




## 1. Convex Sets and Basic Geometry

### 1.1 Optimization and Convexity

#### 一般优化模型

考虑 $\min_{\bm{x} \in S} f(\bm{x})$, 其中 $S \subseteq \mathbb{R}^n$ 是可行域，$f: S \to \mathbb{R}$ 是目标函数. 

- 若 $\bm{x}^\star \in S$，则称 $\bm{x}^\star$ 是可行解.
- 令 $B_\varepsilon(\bm{x}^\star) := \{\bm{x} \in \mathbb{R}^n: \|\bm{x}-\bm{x}^\star\|_2 < \varepsilon\}$ 为 $\bm{x}^\star$ 的 $\varepsilon$-邻域. 
  - 若对所有 $\bm{x} \in B_\varepsilon(\bm{x}^\star) \bigcap S$，有 $f(\bm{x}) \geq f(\bm{x}^\star)$，则称 $\bm{x}^\star$ 是 local optimizer. 若对所有 $\bm{x} \neq \bm{x}^\star$，有 $f(\bm{x}) > f(\bm{x}^\star)$，则称 $\bm{x}^\star$ 是 strict local optimizer. 若对所有 $\bm{x} \in S$，$\bm{x}^\star$ 是唯一的局部最优解，则称 $\bm{x}^\star$ 是 isolated / strong local optimizer.
  - 若对所有 $\bm{x} \in S$，有 $f(\bm{x}) \geq f(\bm{x}^\star)$，则称 $\bm{x}^\star$ 是全局最优解.

#### 线性规划与标准型

考虑线性规划问题, 其一般形式为
$$
\min_{\bm{x} \in \mathbb{R}^n} \bm{c}^\top \bm{x}, \quad \text{s.t. } \bm{x} \in P,
$$
其中 $P \subseteq \mathbb{R}^n$ 是一个 polyhedral set. 

最经典的标准形式为:
$$
\min_{\bm{x} \in \mathbb{R}^n} \bm{c}^\top \bm{x}, \quad \text{s.t. } \bm{A}\bm{x} = \bm{b}, \quad \bm{x} \geq 0,
$$ 
其中 $\bm{A} \in \mathbb{R}^{m \times n}$ 且通常假设 $\text{rank}(\bm{A}) = m < n$. 

一般的 LP 问题都可以通过如下基本转化变为标准型:
1. min-max 转化: $\min \bm{c}^\top \bm{x} = -\max (-\bm{c}^\top \bm{x})$.
2. 不等式约束转化为等式约束: 
    - $\bm{a}^\top \bm{x} \geq b \iff \bm{a}^\top \bm{x} - s = b$, $s \geq 0$.
    - $\bm{a}^\top \bm{x} \leq b \iff \bm{a}^\top \bm{x} + s = b$, $s \geq 0$.
3. 自由变量转化为非负变量: $x_i$ 自由 $\iff x_i = x_i^+ - x_i^-$, $x_i^+, x_i^- \geq 0$.


#### 凸优化问题

给定非空凸集 $S \subseteq \mathbb{R}^n$ 和凸函数 $f: S \to \mathbb{R}$，则一个一般的凸优化问题可以写为 $\min_{\bm{x} \in S} f(\bm{x})$. 

凸优化问题的一个重要性质: 每个局部最优解都是全局最优解.


### 1.2 Convex geometry

#### Affine set, convex set, convex combination

- 给定 $\bm{x}_1, \ldots, \bm{x}_k \in \mathbb{R}^n$，若系数满足 $\sum_{i=1}^k \theta_i = 1$，则 $\sum_{i=1}^k \theta_i \bm{x}_i$ 称为 $\bm{x}_1, \ldots, \bm{x}_k$ 的仿射组合. 
- 给定集合 $S$ 的所有 affine combination 构成的集合称为 $S$ 的 affine hull，记为 $\text{aff}(S)$. 

- 集合 $C \subseteq \mathbb{R}^n$ 称为凸集，若对任意 $\bm{x}_1, \bm{x}_2 \in C$ 和 $\theta \in [0,1]$，有 $\theta \bm{x}_1 + (1-\theta) \bm{x}_2 \in C$. 
- 若 $\theta_1, \ldots, \theta_k \geq 0$ 且 $\sum_{i=1}^k \theta_i = 1$，则 $\sum_{i=1}^k \theta_i \bm{x}_i$ 称为 $\bm{x}_1, \ldots, \bm{x}_k$ 的凸组合. 
- 给定集合 $S$ 的所有 convex combination 构成的集合称为 $S$ 的 convex hull，记为:
    $$
    \text{conv}(S) := \left\{\sum_{i=1}^k \theta_i \bm{x}_i: k \in \mathbb{Z}_+, \bm{x}_i \in S, \theta_i \geq 0, \sum_{i=1}^k \theta_i = 1\right\}.
    $$
    - **Proposition**: $\text{conv}(S)$ 是最小的包含 $S$ 的凸集. 等价地, $\text{conv}(S)$ 是所有包含 $S$ 的凸集的交集.
  
#### 常见的凸集

**Hyperplane and halfspace.**

给定法向量 $\bm{a} \in \mathbb{R}^n \setminus \{\bm{0}\}$ 和 $b \in \mathbb{R}$，定义 hyperplane:
$$
H := \{\bm{x} \in \mathbb{R}^n: \bm{a}^\top \bm{x} = b\},
$$
- Hyperplane 是一个仿射集, 也是一个凸集.
- 其几何直观为: 先确定一个方向 $\bm{a}$，然后所有向这个方向投影为定长 $b/\|\bm{a}\|$ 的点的集合就是一个 hyperplane.
  
    <img src="https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260922111438897.png" width="300">

Closed halfspace:
$$
H_- := \{\bm{x} \in \mathbb{R}^n: \bm{a}^\top \bm{x} \leq b\},\qquad H_+ := \{\bm{x} \in \mathbb{R}^n: \bm{a}^\top \bm{x} \geq b\}.
$$
- Closed halfspace 是一个凸集.

**Norm ball, ellipsoid.**

任意范数 $\|\cdot\|$ 定义的以 $\bm{c} \in \mathbb{R}^n$ 为中心、$r > 0$ 为半径的 norm ball 是 convex set:
$$
B(\bm{c}, r) := \{\bm{x} \in \mathbb{R}^n: \|\bm{x}-\bm{c}\| \leq r\}.
$$
- 注意: $0<p<1$ 时 对应的 $\ell_p$-norm 是 quasi-norm, 其不满足严格 norm 要求的三角不等式. 此时对应的 $\ell_p$-ball 一般不是凸集. 

Ellipsoid 是一个凸集, 给定 $\bm{P} \in \mathbb{S}_{++}^n$ 和 $\bm{c} \in \mathbb{R}^n$，定义椭球的标准形式:
$$
E(\bm{c}, \bm{P}) := \{\bm{x} \in \mathbb{R}^n: (\bm{x}-\bm{c})^\top \bm{P}^{-1} (\bm{x}-\bm{c}) \leq 1\} = \{\bm{x} \in \mathbb{R}^n: \|\bm{P}^{-1/2}(\bm{x}-\bm{c})\|_2 \leq 1\}.
$$
- 或等价地, 以 affine 形式定义为 $E = \{\bm{c} + \bm{A}\bm{u}: \|\bm{u}\|_2 \leq 1\}$, 其中 $\bm{A} \in \mathbb{R}^{n \times n}$ 非奇异.

**Polyhedron, polytope.**

有限个 closed halfspace 和 hyperplane 的交集是一个 polyhedron:
$$
P := \{\bm{x} \in \mathbb{R}^n: \bm{A}\bm{x} \leq \bm{b}, \bm{C}\bm{x} = \bm{d}\},
$$

- Polyhedron 和 polytope 都是凸集. 
- 二者的说法有时会混用, 但严格说: polyhedron 可以是无界的, 而 polytope 是有界的. 

**Simplex.**

若 $\bm{v}_0, \ldots, \bm{v}_k \in \mathbb{R}^n$ 是 affinely independent 的点 (即 $\bm{v}_1-\bm{v}_0, \ldots, \bm{v}_k-\bm{v}_0$ 线性无关), 则称
$$
S := \text{conv}\{\bm{v}_0, \ldots, \bm{v}_k\} 
$$
是一个 $k$-simplex.

- Affine independent 可以理解为, 给定一个参考点 $\bm{v}_0$，与其余点构成的向量两两是不共线的. 因此这与其所处的空间维度是有密切联系的. 例如在二维平面中, 3 个点不共线, 给定一个参考点, 其余两个点可以 span 整个平面; 然而如果有四个点, 则必定会有一个点是冗余的. **在 $n$ 维空间中, 最多只能有 $n+1$ 个 affinely independent 的点.** (这些点 span 出了一个 $n$-dimensional affine space)
- Simplex 是一种特殊的 polytope. 其相当于在对应 affine space 中最小 polytope. 其具有**唯一表出**的性质:
  $$
  \bm{x} \in S \implies \bm{x} = \sum_{i=0}^k \theta_i \bm{v}_i,~ \text{ exists unique } \theta_i \geq 0, \sum_{i=0}^k \theta_i = 1.
  $$
  也鉴于这种性质, 可以考虑如下 $d$ 维 simplex 的标准形式:
  $$
  \Delta_d := \left\{\bm{\lambda} \in \mathbb{R}^{d+1}: \sum_{i=0}^d \lambda_i = 1,~ \lambda_i \geq 0,~ i=0,\ldots,d\right\}.
  $$
  其对应的 vertices 为 $\bm{e}_0, \ldots, \bm{e}_d$，其中 $\bm{e}_i$ 是 $(d+1)$ 维单位向量. 因此任意 $d$-dimensional simplex $S$ 都存在一个 affine map $\Delta_d \to S$，因此所有 $d$-dimensional simplex 都是 affine equivalent 的. 后面可以看到, 这种唯一表出的性质就可以让我们用来定义坐标系 (barycentric coordinates). 


**Convex cone.**

一个集合 $K$ 若满足: $\bm{x} \in K, \theta \geq 0 \implies \theta \bm{x} \in K$, 则称 $K$ 是一个 cone. 若 $K$ 还是一个 convex set, 则称 $K$ 是一个 convex cone. Convex cone 满足:
$$
\bm{x}_1, \bm{x}_2 \in K,~ \theta_1, \theta_2 \geq 0 \implies \theta_1 \bm{x}_1 + \theta_2 \bm{x}_2 \in K.
$$

- 考虑平面直角坐标系中的 $x$,$y$ 两个正半轴 $\{(t,0), t \geq 0\} \cup \{(0,t), t \geq 0\}$，则其是一个 cone, 但不是 convex cone. 若考虑整个象限 $\mathbb{R}_+^n := \{\bm{x} \in \mathbb{R}^n: x_i \geq 0, i=1,\ldots,n\}$，则其是一个 convex cone.

- Second-order cone (SOC) 是一个 convex cone, 定义为:
    $$
    Q^{n+1} := \{(t, \bm{x}) \in \mathbb{R}^{n+1}: t \geq \|\bm{x}\|_2\}.
    $$
    - SOC 是一个重要的 convex cone，它在优化问题中经常出现. 考虑 nonlinear 约束:
        $$
        \|\bm{A}\bm{x} + \bm{b}\|_2 \leq \bm{c}^\top \bm{x} + d,
        $$
        其中 $\bm{A} \in \mathbb{R}^{m \times n}$, $\bm{b}, \bm{c} \in \mathbb{R}^m$, $d \in \mathbb{R}$. 这就是一个 SOC 约束.

- Positive semidefinite cone (PSD cone) 是一个 convex cone, 定义为:
    $$
    \mathbb{S}_+^n := \{\bm{X} \in \mathbb{S}^n: \bm{X} \succeq 0\},
    $$
    即对称半正定矩阵本身就是一个 convex cone. 一个典型的 SDP 问题即为:
    $$
    \min_{\bm{X}} \langle \bm{C}, \bm{X} \rangle \quad \text{s.t.} \quad \langle \bm{A}_i, \bm{X} \rangle = b_i,~ i=1,\ldots,m,~ \bm{X} \succeq 0.
    $$
    其中 $\bm{C}, \bm{A}_i \in \mathbb{S}^n$, $i=1,\ldots,m$, $\bm{X} \in \mathbb{S}^n$.

总的而言, conic optimization 是一个重要的优化问题类别, 其形式为:
$$
\min_{\bm{x}} \langle \bm{c}, \bm{x} \rangle \quad \text{s.t.} \quad \bm{A}\bm{x} = \bm{b},~ \bm{x} \in K,
$$
其中 $K$ 是一个 convex cone. 其包含了 LP, SOCP, SDP 等优化问题. 


**Cone 是一种广义的非负.** simplex 主要讨论的是一种位置关系, 而 cone 主要讨论的是一种方向关系. 对应的, simplex 会关心 extreme points, 而 cone 会关心 extreme rays. 

#### 保凸运算

实践中往往不会通过定义来验证给定复杂集合的凸性, 而是通过一些基本凸集出发, 通过一些保凸运算来构造新的凸集. 给定指标集合 $I$，任意 $C, C_i \subseteq \mathbb{R}^n$，$i \in I$ 是凸集, 则有以下保凸运算:
- 任意交集: $\bigcap_{i \in I} C_i$ 是凸集.
- 仿射映射: 若 $\bm{A} \in \mathbb{R}^{m \times n}$ 且 $\bm{b} \in \mathbb{R}^m$，则 $\bm{A}C + \bm{b} := \{\bm{A}\bm{x} + \bm{b}: \bm{x} \in C\}$ 是凸集.
- 仿射的 preimage: 若 $\bm{A} \in \mathbb{R}^{m \times n}$ 且 $\bm{b} \in \mathbb{R}^m$，则 $\{\bm{x} \in \mathbb{R}^n: \bm{A}\bm{x} + \bm{b} \in C\}$ 是凸集.
- Minkowski sum: $C_1 + C_2 := \{\bm{x}_1 + \bm{x}_2: \bm{x}_1 \in C_1, \bm{x}_2 \in C_2\}$ 是凸集.
- Cartesian product: $C_1 \times C_2 := \{(\bm{x}_1, \bm{x}_2): \bm{x}_1 \in C_1, \bm{x}_2 \in C_2\}$ 是凸集.
- Projection: 若 $C \subseteq \mathbb{R}^{n+m}$ 是凸集, 则将 $C$ 投影到 $\bm{x}$ 上得到 $\text{proj}_x C := \{\bm{x} \in \mathbb{R}^n:  \exists \bm{y} \in \mathbb{R}^m, \text{s.t. } (\bm{x}, \bm{y}) \in C\}$ 是凸集.


### 1.3 Carathéodory's theorem

在正式给出 Carathéodory 定理之前, 先明确 **affine space** 的概念. 
- Affine space 可以理解为一个平移后的线性子空间, 或一个不经过(也不关心)原点的线性子空间, 即 $\mathcal{A} = \bm{x}_0 + \mathcal{V}$, 其中 $\mathcal{V}$ 是一个线性子空间, $\bm{x}_0$ 是一个固定点. 
- 或者这么说, 本身 affine-linear 就是一组类似相对应的词汇. affine 是有截距的 linear.
- 所以比如前文在讨论 simplex 时, 说 $\bm{v}_0, \ldots, \bm{v}_k$ 是 affinely independent 的点, 也就是说 $\text{aff}(\{\bm{v}_0, \ldots, \bm{v}_k\})$ 是一个 $k$-dimensional affine space.
- 在后面的认知当中, 也可以区分 affine dimension 和 ambient dimension. Ambient dimension 可以粗略的理解为就是所在客观世界的维度, 其可能是很大的. 而 affine dimension 是指真正我们关注的比如 feasible region $S$ 所在的 affine space 的维度. 若 $S$ 位于一个 $m$-dimensional affine space 中, 则即使整个空间是很高维的, 我们也只需要 $m+1$ 个点就可以表示出任意一个凸组合. 


***Theorem* (Carathéodory)**: 设 $S \subseteq \mathbb{R}^n$. 若 $\bm{x} \in \text{conv}(S)$，则 $\bm{x}$ 可以表示为 $S$ 中至多 $n+1$ 个点的凸组合. $^\dagger$ 也即, 若 $S \subseteq \mathbb{R}^n$，$\text{dim}~\text{aff}(\text{conv}(S)) = m$，那么对任意 $\bm{x} \in \text{conv}(S)$，存在至多 $m+1$ 个点 $\bm{x}_1, \ldots, \bm{x}_{k} \in S$ ($k \leq m+1$) 和系数 $\theta_1, \ldots, \theta_k \geq 0$，使得 $\sum_{i=1}^k \theta_i = 1$ 且 $\bm{x} = \sum_{i=1}^k \theta_i \bm{x}_i$.

- 直观理解, 在 $n$ 维空间中, 最多只需要 $n+1$ 个点就可以表示出任意一个凸组合. 其实这个 theorem 有点类似 simplex 的反向理解. 


*Proof$~^\dagger$* *[Nemirovski, App. B §B.2.1]* :
- 由于 $\bm{x} \in \text{conv}(S)$，一定存在有限 $N$, 使得 $\bm{x} = \sum_{i=1}^N \theta_i \bm{x}_i$，其中 $\bm{x}_i \in S$, $\theta_i \geq 0$, $\sum_{i=1}^N \theta_i = 1$. 并且选取所有可能构造中目前最小的 $N$ (例如去掉一些 $\theta_i = 0$ 的点).
- 下证明, 定有 $N \leq m+1$. 用反证法, 假设 $N > m+1$. 此时由于 $\bm{x}_1, \ldots, \bm{x}_N$ 在 $m$-dimensional affine space 中, 定是 affine dependent 的, 即存在 $\alpha_1, \ldots, \alpha_N$ 不全为零, 且 $\sum_{i=1}^N \alpha_i = 0$, 使得 $\sum_{i=1}^N \alpha_i \bm{x}_i = 0$.
- 故根据 $\sum_{i=1}^N \theta_i = 1$ 和 $\sum_{i=1}^N \alpha_i = 0$，有 $\sum_{i=1}^N (\theta_i + t\alpha_i) = 1$ 对任意 $t \in \mathbb{R}$ 成立. 同时, $\sum_{i=1}^N (\theta_i + t\alpha_i) \bm{x}_i = \sum_{i=1}^N \theta_i \bm{x}_i + t\sum_{i=1}^N \alpha_i \bm{x}_i = \bm{x}$ 对任意 $t \in \mathbb{R}$ 成立.
- 因此我们总可以通过调整 $t$ 来得到一个新的凸组合, 使得至少有一个新的系数为零, 也即我们可以去掉一个点, 这与我们选取的最小 $N$ 矛盾. 因此定有 $N \leq m+1$.


$\square$

### 1.4 Basic Topologies


在优化过程中, 算法总是会产生一系列的迭代点 $\bm{x}_1, \bm{x}_2, \ldots$，我们希望 $\bm{x}_k \to \bm{x}^\star$ 收敛到一个最优解 $\bm{x}^\star$. 因此会有一系列问题: 如果 $\bm{x}_k \in S$ 且 $\bm{x}_k \to \bm{x}^\star$，是否有 $\bm{x}^\star \in S$? 如果 $f(\bm{x}_k) \to \inf f$, 是否有 $\bm{x}^\star \in S$ 且 $f(\bm{x}^\star) = \inf f$? 这些问题都与集合的拓扑性质有关.

给定 norm ball $B(\bm{x}, \varepsilon) := \{\bm{y} \in \mathbb{R}^n: \|\bm{y}-\bm{x}\| \leq \varepsilon\}$.在欧式空间中, 任意 norm 都是等价的.  

首先考虑如下几个基本的集合拓扑概念. 给定 $S \subseteq \mathbb{R}^n$:
- $\text{int}(S) := \{\bm{x} \in S: \exists \varepsilon > 0, B(\bm{x}, \varepsilon) \subseteq S\}$. 
  - $\bm{x}$ 是集合的内点, 说明存在一个小邻域完全包含在集合中.
  - 集合 $S$ 是 open set 当且仅当 $S = \text{int}(S)$, 即集合的任意点都是内点.
- $\text{cl}(S) := \{\bm{x} \in \mathbb{R}^n: \forall \varepsilon > 0, B(\bm{x}, \varepsilon) \cap S \neq \emptyset\}$. 
  - 集合的闭包可以认为是集合+集合的极限点. 
  - $\bm{x} \in \text{cl}(S)$ 当且仅当存在一个序列 $\{\bm{x}_k\} \subseteq S$ 使得 $\bm{x}_k \to \bm{x}$. 所有从 $S$ 中的点构成的极限点都包含在 $\text{cl}(S)$ 中.
  - 集合 $S$ 是 closed set 当且仅当 $S = \text{cl}(S)$, 即集合包含其所有极限点.
- $\partial S := \text{cl}(S) \setminus \text{int}(S)$.
  - 集合的边界是集合的闭包减去集合的内点. 
  - $B(\bm{x}, \varepsilon) \cap S \neq \emptyset$ 且 $B(\bm{x}, \varepsilon) \cap (\mathbb{R}^n \setminus S) \neq \emptyset$ 对任意 $\varepsilon > 0$ 成立, 则 $\bm{x}$ 是集合的边界点.
- $S$ 是 bounded set 当且仅当存在 $M > 0$ 使得 $\|\bm{x}\| \leq M$ 对所有 $\bm{x} \in S$ 成立. 
- 有限维欧式空间中, $S$ 是 compact set 当且仅当 $S$ 是 closed 且 bounded (Heine-Borel theorem). 
  - 在优化视角, **$S$ 是 compact set 当且仅当 $S$ 中任何一个序列都存在一个收敛的子序列, 且其极限点仍然在 $S$ 中.** 这保证了算法迭代点的收敛性. 
    - (Bolzano-Weierstrass theorem 保证任何有界序列都存在收敛子序列. 而 compactness 保证了收敛子序列的极限点仍然在集合中.)
  - Compactness 在优化中避免了如下两种最优解求解失败之场景:
    - 集合不闭: 例如 $\min_{x \in (0,1)} x$，其 infimum 为 $0$，但 $0 \notin (0,1)$.
    - 集合不有界: 例如 $\min_{x \geq 0} \exp(-x)$，其 infimum 为 $0$，但取在 $x \to \infty$.

接着引入 **relative interior** 的概念. 
- 给定集合 $S \subseteq \mathbb{R}^n$，其 affine hull 为 $\text{aff}(S)$. 定义 $S$ 的相对内点为:
    $$
    \text{ri}(S) := \{\bm{x} \in S: \exists \varepsilon > 0, B(\bm{x}, \varepsilon) \cap \text{aff}(S) \subseteq S\}.
    $$
- Intuition: 考虑一个 $\mathbb{R}^3$ 中的二维 simplex $\Delta_2$. 显然, $\text{int}(\Delta_2) = \emptyset$, 因为三维空间内的任意球体都无法完全包含在一个二维的平面中. 但是这显然并不是我们关心的. 我们只需要考虑这个我们关注集合 $S$ 所在的 affine space, 然后考虑 ambient space 中的球体与 affine space 的公共部分, 讨论这个公共部分的内点情况即可. 
- Relative interior 帮助我们能够良好的处理在高维空间内的低维集合之间的关系. 
- 反过来也可以定义 relative boundary, 即 $\partial_{\text{rel}}(S) := \text{cl}(S) \setminus \text{ri}(S)$.

故有关系:
$$
\text{ri}(S) \subseteq S \subseteq \text{cl}(S) \subseteq \text{aff}(S),
$$

- 对于非空 convex set $C \subseteq \mathbb{R}^n$，有 $\text{ri}(C) \neq \emptyset$, $\text{ri}(\text{cl}(C)) = \text{ri}(C)$, $\text{cl}(\text{ri}(C)) = \text{cl}(C)$. 

- 此外, 若 $\bm{x} \in \text{ri}(C)$ 且 $\bm{y} \in \text{cl}(C)$，则 
    $$
    (1-t) \bm{x} + t \bm{y} \in \text{ri}(C),~ \forall t \in [0,1).
    $$
    即从相对内点出发, 向任意闭包点连线, 只要没有到达闭包点, 都会落在相对内点中.

### 1.5 Weierstrass theorem, existence of optimal solution

#### Weierstrass theorem

若 $S \subseteq \mathbb{R}^n$ 是非空的 compact set, 且 $f: S \to \mathbb{R}$ 是连续函数, 则 $f$ 在 $S$ 上有最小和最大值. 特别地, 存在 $\bm{x}^\star \in S$ 使得
$$
f(\bm{x}^\star) = \min_{\bm{x} \in S} f(\bm{x}).
$$

*Proof Sketch*: 令 $\alpha := \inf_{\bm{x} \in S} f(\bm{x})$. 选择一个序列 $\{\bm{x}_k\} \subseteq S$ 使得 $f(\bm{x}_k) \downarrow \alpha$. 因为 $S$ 是 compact set, $\{\bm{x}_k\}$ 有一个收敛的子序列 $\{\bm{x}_{k_j}\} \to \bm{x}^\star \in S$. 由于 $f$ 是连续函数, 则 $f(\bm{x}_{k_j}) \to f(\bm{x}^\star)$. 因此 $f(\bm{x}^\star) = \alpha$, 即 $\bm{x}^\star$ 是最优解.

*Proof* *[Beck, Ch. 2]*:

> *在 Beck 的论述中, 其将 Weierstrass theorem 的条件, 从 $f: S \to \mathbb{R}$ 是连续函数, 放宽为 $f$ 是 proper 且 closed 函数. 这是一个更弱的条件, 但其证明思路是类似的.]*

- **先证 $f$ bounded below**. *在 sketch 中我们直接定义 $\alpha := \inf_{\bm{x} \in S} f(\bm{x})$. 然而这样的 inf 的存在本身也是需要严格证明的.* 即, 要证: 存在 $L \in \mathbb{R}$ 使得 $f(\bm{x}) \geq L$ 对所有 $\bm{x} \in S$ 成立. 
  - 反证法, 假设 $f$ 不存在下界, 则对于每一个 $\bm{x}_k$, 都可以找到 $\bm{x}_k \in S$ 使得 $f(\bm{x}_k) \leq -k$. 于是我们可以构造一个序列 $\{\bm{x}_k\} \subseteq S$ 使得 $f(\bm{x}_k) \to -\infty$. 由于 $S$ 是 compact set, $\{\bm{x}_k\}$ 有一个收敛的子序列 $\{\bm{x}_{k_j}\} \to \bar{\bm{x}} \in S$. 由于 $f$ 是连续函数且 $\bm{x}_{k_j} \to \bar{\bm{x}}$，则 $f(\bm{x}_{k_j}) \to f(\bar{\bm{x}})$ 是一个有限值. 然而 $f(\bm{x}_{k_j}) \to -\infty$, 这与 $f(\bar{\bm{x}})$ 是有限值矛盾. 

- **再证 $f$ 在 $S$ 上有最小值**. 由实数完备性, $f$ 有下界故有下确界, 记 $\alpha := \inf_{\bm{x} \in S} f(\bm{x})$. 下证 $\exists \bm{x}^\star \in S$ 使得 $f(\bm{x}^\star) = \alpha$. 由 inf 的定义, 对于每个 $k \geq 1$, 都存在 $\bm{x}_k \in S$ 使得 $\alpha \leq f(\bm{x}_k) < \alpha + 1/k$. 由 squeeze theorem 可以给出 $f(\bm{x}_k) \to \alpha$. 再次由于 $S$ 是 compact set, $\{\bm{x}_k\}$ 有一个收敛的子序列 $\{\bm{x}_{k_j}\} \to \bm{x}^\star \in S$. 由于 $f$ 是连续函数, 则 $f(\bm{x}_{k_j}) \to f(\bm{x}^\star)$. 因此 $f(\bm{x}^\star) = \alpha$, 即 $\bm{x}^\star$ 是最优解.

$\square$

<!-- 



## 2. Projections, Separation and Certificates

### 2.1 Euclidean projection

### 2.2 Projection variational inequality / normal cone

### 2.3 Farkas lemma

### 2.4 Convex cones, dual cones, polar cones

## 3. Polyhedral Geometry and Linear Programming

### 3.1 H- and V-representations

### 3.2 Extreme points and active constraints

### 3.3 Basic feasible solutions

### 3.4 LP optimization at verticies

### 3.5 Recession directions and extreme rays

### 3.6 Minkowski-Weyl theorem

### -->