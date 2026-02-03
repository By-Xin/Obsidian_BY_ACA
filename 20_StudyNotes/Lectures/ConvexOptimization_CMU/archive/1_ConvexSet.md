# Convexity

> Refs:
> - Nonlinear Programming: Theory and Algorithms, Mokhtar S. Bazaraa, Hanif D. Sherali, C. M. Shetty, 2006.
> - Convex Optimization, Stephen Boyd, Lieven Vandenberghe, 2004.
 
## 1. Definition of Convex Sets

***Definition* (Convex Set)**: 对于任意两个点 \(x_1, x_2 \in C\)，对于任意 \(\lambda \in [0, 1]\), 有:
\[\lambda x_1 + (1 - \lambda) x_2 \in C
\]
则称集合 \(C\) 是凸集.

- 该定义可以推广为: 在凸集内任意多个点的凸组合仍然在该凸集内. 对于任意有限个点 \(x_1, x_2, \ldots, x_k \in C\) 为凸集, 对于任意非负数 \(\lambda_1, \lambda_2, \ldots, \lambda_k\) 满足 \(\sum_{i=1}^k \lambda_i = 1\)，有:
\[\sum_{i=1}^k \lambda_i x_i \in C
\]
  - 这种组合称为 *凸组合* (convex combination).

***Definition* (Convex Hull)**: 给定任意集合 \(S\)，其凸包 (convex hull) 定义为包含 \(S\) 的所有凸集的交集，记为 \(\text{conv}(S)\). 换句话说, \(\text{conv}(S)\) 是包含 \(S\) 的最小凸集.
- 凸包也可以表示为 \(S\) 中所有点的凸组合的集合:
\[\text{conv}(S) = \left\{\sum_{i=1}^k \lambda_i x_i \mid x_i \in S, \lambda_i \geq 0, \sum_{i=1}^k \lambda_i = 1, k \in \mathbb{N}\right\}
\]

## 2. Common Convex Sets

- 超平面 (Hyperplane): \(H = \{x \mid a^\top x = b\}\)，其中 \(a \neq 0\).
- 半空间 (Halfspace): \(H^+ = \{x \mid a^\top x \leq b\}\)，其中 \(a \neq 0\).
- 多面体 (Polyhedron): 由有限个线性不等式和等式定义的集合 \(P = \{x \mid Ax \leq b, Cx = d\}\).
- 球体 (Euclidean Ball): \(B(x_c, r) = \{x \mid \|x - x_c\|_2 \leq r\} = \{x_c+ru \mid \|u\|_2 \leq 1\}\).
- 椭球体 (Ellipsoid): \(E(x_c, P) = \{x \mid (x - x_c)^\top P^{-1} (x - x _c) \leq 1\} = \{x_c + A u \mid \|u\|_2 \leq 1\}\)，其中 \(P = AA^\top \succ 0\).
- 二次锥 (Second-Order Cone): \(C = \{(x, t) \mid \|x\|_2 \leq t\}\).
  - Note: 锥的定义为, 对于任意 \(x \in C\) 和非负数 \(\alpha \geq 0\), 有 \(\alpha x \in C\).
- 半定锥 (Positive Semidefinite Cone): 首先定义 \(S^n\) 为所有 \(n \times n\) 对称矩阵的集合, 则 \(S_+^n = \{X \in S^n \mid X \succeq 0\}\) 为所有 \(n \times n\) 半正定矩阵的集合. 则 \(S^n_{++} = \{X \in S^n \mid X \succ 0\}\) 为所有 \(n \times n\) 正定矩阵的集合.

## 3. Operations that Preserve Convexity

设 \(C_1\) 和 \(C_2\) 为凸集, \(\alpha \in \mathbb{R}\).
1. 交集 (Intersection): \(C = C_1 \cap C_2\) 是凸集.
   - 并集 (Union) 一般不是凸集.
2. Minkowski Sum: \(C = C_1 \pm C_2 = \{x_1 \pm x_2 \mid x_1 \in C_1, x_2 \in C_2\}\) 是凸集.
3. 仿射映射 (Affine Mapping) 是凸集. 
   - 设 $f: \mathbb{R}^n \to \mathbb{R}^m$ 为仿射映射, 即 \(f(x) = Ax + b\)，其中 \(A \in \mathbb{R}^{m \times n}\), \(b \in \mathbb{R}^m\). 则对于凸集 \(C \subseteq \mathbb{R}^n\), 有 \(f(C) = \{f(x) \mid x \in C\}\) , \(f^{-1}(C) = \{x \mid f(x) \in C\}\) 也是凸集.
   - 特殊的仿射映射包括:
     - 平移 (Translation): \(C + a = \{x + a \mid x \in C\}\).
     - 缩放 (Scaling): \(\alpha C = \{\alpha x \mid x \in C\}\).
     - 投影 (Projection): 设 \(C \subseteq \mathbb{R}^{m+n}\), 则其在前 \(m\) 个坐标上的投影为 \(\{x \in \mathbb{R}^m \mid \exists y \in \mathbb{R}^n, \begin{bmatrix} x \\ y \end{bmatrix} \in C\}\).


## 4. Basic Properties of Convex Sets

***Theorem* (Projection Theorem)**: 设 \(C\) 为非空闭凸集, 则对于任意点 \(y \in \mathbb{R}^n, y \notin C\), 则:
1. 存在唯一的点 \(x_0 \in C\), 使得 \(x_0\) 是 \(y\) 到 \(C\) 的最近点, 即:
   \[\|y - x_0\|_2 = \inf_{x \in C} \|y - x \|_2 > 0
   \]
2. 该点 \(x_0\) 满足:
   \[(y - x_0)^\top (x - x_0) \leq 0, \quad \forall x \in C
   \]
   即, 向量 \(y - x_0\) 与从 \(x_0\) 指向 \(C\) 中任意点的向量成钝角.


***Theorem* (Separating Hyperplane Theorem)**: 设 \(C_1\) 和 \(C_2\) 为两个非空凸集, 若对于非零向量 \(a \in \mathbb{R}^n\) 和标量 \(b \in \mathbb{R}\), 有:
\[\alpha^\top x \ge b, \quad \forall x \in C_1, \quad \alpha^\top z \le b, \quad \forall z \in C_2
\]
则称超平面 \(H = \{x \mid \alpha^\top x =   b\}\) 分离 (separates) 了集合 \(C_1\) 和 \(C_2\).

- 若 \(C_1\) 和 \(C_2\) 互不相交, 则存在一个超平面分离它们.

***Theorem* (Supporting Hyperplane Theorem)**: 设 \(C\) 为非空凸集, 对于任意边界点 \(x_0 \in \partial C\), 存在非零向量 \(a \in \mathbb{R}^n\) 使得:
\[\alpha^\top x \le \alpha^\top x_0, \quad \forall x \in C
\]
即, 存在一个超平面 \(H = \{x \mid \alpha^\top x = \alpha^\top x_0\}\) 支持 (supports) 集合 \(C\) 在点 \(x_0\) 处.


## 5. Farkas' Lemma

给定矩阵 \(A \in \mathbb{R}^{m \times n}\) 和向量 \(b \in \mathbb{R}^n\), 以下两种情况恰有其一成立:
1. 存在向量 \(Ax \leq 0\) 且 \(b^\top x > 0\).
2. 存在向量 \(y \geq 0\) 使得 \(A^\top y = c\).

## 6. Convex Functions and Their Epigraphs

***Definition* (Convex Function)**: 函数 \(f: \mathbb{R}^n \to \mathbb{R}\) 是凸函数, 如果其定义域 \(C\) 是凸集, 且对于任意 \(x_1, x_2\) 在定义域内, 对于任意 \(\lambda \in (0,1)\), 有:
\[f(\lambda x_1 + (1 - \lambda) x_2) \leq \lambda f(x_1) + (1 - \lambda) f(x_2)
\]
- 函数 \(f\) 是严格凸函数 (strictly convex), 如果对于任意 \(x_1 \neq x_2\) 在定义域内, 对于任意 \(\lambda \in (0,1)\), 有:
\[f(\lambda x_1 + (1 - \lambda) x_2) < \lambda f(x_1) + (1 - \lambda) f(x_2)
\]
- 若 \(f\) 是凸函数, 则 \(-f\) 是凹函数 (concave function).

如下是一些常见的凸函数:
- 线性函数: \(f(x) = a^\top x + b\).
- 二次函数: \(f(x) = x^\top Q x + b^\top x + c\), 其中 \(Q \succeq 0\).
- 最小二乘误差: \(f(x) = \|Ax - b\|_2^2\).
- p-范数: \(f(x) = \|x\|_p\), 其中 \(p \geq 1\).

***Properties of Convex Functions**:
1. 凸函数必然是连续的.
2. \(f\) 是凸函数等价于对任意给定的 \(x_1, x_2\) 在定义域内, 有 \(\phi(\alpha) = f(x_1 + \alpha x_2)\) 也是凸函数.
3.  \(f(x)\) 是 \(C\) 上的凸函数的充要条件是 
    \[f(y) \ge f(x) + \nabla f(x)^\top (y - x), \quad \forall x, y \in C
    \]
    - 该不等式称为 *一阶凸性条件* (first-order condition for convexity), 表示函数在任意点的切平面位于函数图像的下方.
4. 设 \(C \subseteq \mathbb{R}^n\) 非空开凸集, 且 \(f: C \to \mathbb{R}\) 二阶可微, 则 \(f\) 是凸函数的充要条件是其 Hessian 矩阵在 \(C\) 上半正定, 即:
    \[\nabla^2 f(x) \succeq 0, \quad \forall x \in C
    \]
    - 该条件称为 *二阶凸性条件* (second-order condition for convexity).

如下是一些保持凸性的操作:
1. Perspective Function: 设 \(f: \mathbb{R}^n \to \mathbb{R}\) 是凸函数, 则其透视函数 (perspective function) 定义为:
   \[g(x, t) = t f\left(\frac{x}{t}\right), \quad t > 0
   \]
   则 \(g: \mathbb{R}^{n+1} \to \mathbb{R}\) 也是凸函数.

2. 非负加权和 (Nonnegative Weighted Sum): 设 \(f_1, f_2, \ldots, f_k\) 是凸函数, 且 \(\alpha_1, \alpha_2, \ldots, \alpha_k \geq 0\), 则函数:
   \[f(x) = \sum_{i=1}^k \alpha_i f_i(x)
   \]
   也是凸函数.


3. 最大值 (Maximum): 设 \(f_1, f_2, \ldots, f_k\) 是凸函数, 则函数:
   \[f(x) = \max_{1 \leq i \leq k} f_i(x)
   \]
   也是凸函数.


***Definition* (Level Set)**: 给定函数 \(f: \mathbb{R}^n \to \mathbb{R}\) 和标量 \(\alpha \in \mathbb{R}\), 定义水平集 (level set) 为:
\[ L_\alpha = \{x \mid f(x) \leq \alpha, x \in C\]
- 若 \(f\) 是凸函数, 则其水平集 \(L_\alpha\) 是凸集.
