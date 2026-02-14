# Convexity (I)

> Ref: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/

## Introduction

**What is a convex optimization problem?**

一般而言, 一个凸优化问题可以表示为如下形式:

$$
\begin{aligned}
\min_{x\in D}\quad & f(x) \\
\text{s.t. }\quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, r
\end{aligned}
$$

**Why Convexity?**
- 凸优化问题是我们能够一般化解决的一类优化问题. 对于凸优化问题, 其局部最优解即为全局最优解.
- 非凸问题往往需要具体情况具体分析, 难以给出一个统一的解决范式.

## Convex Sets

***Definition* (Convex Set)**: 对于集合 $C\subseteq \mathbb{R}^n$, 其是一个凸集, 若满足对任意 $x,y \in C$, 有:

$$
tx + (1-t) y \in C, \quad \forall t \in [0,1]
$$

- 即凸集中的任意两点连线段均在该集合内.

***Definition* (Convex Combination)**: 对于集合 $C\subseteq \mathbb{R}^n$ 中的点 $x_1, x_2, \ldots, x_k$, 若存在非负实数 $\theta_1, \theta_2, \ldots, \theta_k$ 满足 $\sum_{i=1}^k \theta_i = 1$, 则点

$$
y = \sum_{i=1}^k \theta_i x_i
$$

称为 $x_1, x_2, \ldots, x_k$ 的**凸组合**.

***Definition* (Convex Hull)**: 对于集合 $S\subseteq \mathbb{R}^n$, 其**凸包**定义为包含 $S$ 的所有元素的所有凸组合的最小凸集, 记为 $\text{conv}(S)$. 


如下是一些典型的凸集的例子:

- Trivial examples: 空集, 点, 线等必然是凸集.
- **Norm Ball**: $\{x: \|x\| \leq r\}$ (给定$\|\cdot\|$及半径 $r$).
- **Hyperplane**: $\{x: a^\top x = b\}$ (给定 $a,b$)
- **Halfspace**: $\{x: a^\top x\leq b\}$ (给定 $a,b$)   
- **Affine Space**: $\{x: Ax= b\}$ (给定 $A,b$)
- **Polyhedron**: $\{x: Ax \leq b, Cx = d\}$ (给定 $A,b,C,d$)
- **Simplex**: 对于 affinely independent 的 $k+1$ 个点 $x_0, x_1, \ldots, x_k$, 其凸包称为 $k$-simplex, 记为 $\text{conv}\{x_0, x_1, \ldots, x_k\}$.
  - affinely independent: 点集 $\{x_0, x_1, \ldots, x_k\}$ 称为仿射独立的, 若 $\{x_1 - x_0, x_2 - x_0, \ldots, x_k - x_0\}$ 线性独立. 线形独立需要考虑原点, 而仿射独立只考虑点之间的相对位置关系. 在一维空间不重合的两个点, 二维空间中不共线的三个点等等均是仿射独立的例子.
  - 数学上, 若 $x_0, x_1, \ldots, x_k$ 是标准基, 则 $k$-simplex 即为单位 simplex, 或称为概率 simplex:

    $$
\text{conv}\{e_0, e_1, \ldots, e_k\} = \{\boldsymbol{w} \in \mathbb{R}^{k+1}: \boldsymbol{w} \geq 0, \boldsymbol{1}^\top \boldsymbol{w} = 1\}
    $$


## Cones

***Definition* (Cone)**: 对于集合 $C\subseteq \mathbb{R}^n$, 其是一个锥, 若满足对任意 $x \in C$ 及非负实数 $t \geq 0$, 有:

$$
tx \in C
$$

- 即锥中的任意点沿射线方向延伸后仍在该集合内.

***Definition* (Convex Cone)**: 对于集合 $C\subseteq \mathbb{R}^n$, 其是一个凸锥, 若满足对任意 $x,y \in C$ 及非负实数 $t_1, t_2 \geq 0$, 有:

$$
t_1 x + t_2 y \in C
$$

- 并不是所有的锥都是凸锥. 例如在二维空间中的坐标轴集合 $C = \{ (x, y) \in \mathbb{R}^2 \mid x=0 \text{ or } y=0 \}$ 就不是凸锥. 锥只要求该集合关于正数乘法封闭, 并不要求其关于加法封闭.

***Definition* (Conic Combination)**: 对于集合 $C\subseteq \mathbb{R}^n$ 中的点 $x_1, x_2, \ldots, x_k$, 若存在非负实数 $\theta_1, \theta_2, \ldots, \theta_k \geq 0$, 则点 $\sum_{i=1}^k\theta_i x_i$ 称为一个锥组合. 

***Definition* (Conic Hull)**: 对于集合 $S\subseteq \mathbb{R}^n$, 其**锥包**定义为包含 $S$ 的所有元素的所有锥组合的最小凸锥, 记为 $\text{cone}(S)$.

如下是一些典型的凸锥的例子:

- **Norm cone**: $\{(x,t): \|x\| \leq t\}$ (给定$\|\cdot\|$). 特别地在二范数下, 该集合称为**Second-Order Cone (SOC)** 或**Lorentz Cone**.
- **Normal cone**: 给定任何集合 $C$ 以及点 $x\in C$, 其法锥定义为:

    $$
    \mathcal{N}_C(x) = \{g: g^\top (y-x) \leq 0, \forall y \in C\}
    $$

    - Normal cone 是对法线概念的一个推广. 其相当于是指, 在点 $x$ 处, 所有指向集合外部的向量的集合 (相当于夹角至少垂直或为钝角的向量集合).
    - 对于集合的内部点, 其法锥仅包含零向量. 对于光滑的边界点, 其法锥仅包含唯一的法向量的非负倍数. 对于非光滑的边界点, 其法锥包含多个法向量的非负倍数.
    - 图示链接: [Code_Generated_Image.png](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/Code_Generated_Image.png)
- **Positive semidefinite cone**: $\mathbb{S}^n_+ = \{X \in \mathbb{R}^{n\times n}: X = X^\top, X \succeq 0\}$ 即由所有 $n\times n$ 的对称正半定矩阵组成的集合.
  - PSD 作为 cone, 其满足对于 PSD 的两个矩阵 $X,Y\in \mathbb{S}^n_+$, 满足任意非负实数 $t_1, t_2 \geq 0$, 有 $t_1 X + t_2 Y \in \mathbb{S}^n_+$ 即其仍然是 PSD 矩阵. 
  - 根据 PSD 矩阵的定义, $a^\top (t_1 X+t_2 Y) a = t_1 a^\top X a + t_2 a^\top Y a \geq 0, \forall a \in \mathbb{R}^n$ 因此 $t_1 X + t_2 Y$ 仍然是 PSD 矩阵.


## Key properties of Convex Sets


***Theorem*(Separating Hyperplane Theorem)**: 对于两个不相交的凸集 $C,D \subset \mathbb{R}^n$, 存在一个超平面将其分开, 即存在非零向量 $\boldsymbol{a} \in \mathbb{R}^n$ 及标量 $b \in \mathbb{R}$, 使得:

$$
\boldsymbol{a}^\top \boldsymbol{x} \leq b, \quad \forall \boldsymbol{x} \in C, \quad \text{and} \quad \boldsymbol{a}^\top \boldsymbol{y} \geq b, \quad \forall \boldsymbol{y} \in D.
$$

*Proof.* 
- 给定两个不相交的凸集 $C$ 和 $D$, 定义距离函数 $d(C,D) = \inf\{\|\boldsymbol{c} - \boldsymbol{d}\| : \boldsymbol{c} \in C, \boldsymbol{d} \in D\}$, 并记各自最短处的点为 $\boldsymbol{c}_0 \in C$ 和 $\boldsymbol{d}_0 \in D$. 事实上, 这里的 hyperplane 恰好在 $\boldsymbol{c}_0$ 和 $\boldsymbol{d}_0$ 的中垂线上.
    ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202601072148122.png)
- 记 $\boldsymbol{m} = (\boldsymbol{c}_0 + \boldsymbol{d}_0)/2$ 为 $\boldsymbol{c}_0$ 和 $\boldsymbol{d}_0$ 的中点, 则定义仿射函数 $f(\boldsymbol{x}) = (\boldsymbol{d}_0 - \boldsymbol{c}_0)^\top (\boldsymbol{x} - \boldsymbol{m})$. 
  - $f(\boldsymbol{x}) = 0$ 的集合是一个超平面, 这个超平面经过点 $\boldsymbol{m}$, 且法向量为 $\boldsymbol{d}_0 - \boldsymbol{c}_0$. 事实上, $f(\boldsymbol{x})=0$ 等价于 $\|\boldsymbol{x} - \boldsymbol{c}_0\| = \|\boldsymbol{x} - \boldsymbol{d}_0\|$. 故 $f(\boldsymbol{x'})>0$ 表示点 $\boldsymbol{x'}$ 更靠近 $\boldsymbol{d}_0$, 而 $f(\boldsymbol{x''})<0$ 则表示点 $\boldsymbol{x''}$ 更靠近 $\boldsymbol{c}_0$.
- 下证明对于任意 $\boldsymbol{d} \in D$, $f(\boldsymbol{d}) = (\boldsymbol{d}_0 - \boldsymbol{c}_0)^\top (\boldsymbol{d} - \boldsymbol{m}) \geq 0$.
  - 用反证法. 假设存在 $\boldsymbol{d}' \in D$ 使得 $f(\boldsymbol{d}') < 0$. 下试图说明, 若存在这样的距离$\boldsymbol{c}_0$ 近于 $\boldsymbol{d}_0$ 的 $D$ 中的点 $\boldsymbol{d}'$, 则我们可以让 $\boldsymbol{d}_0$ 向 $\boldsymbol{d}'$ 移动, 从而找到一个更近的点对, 而这与 $\boldsymbol{c}_0$ 和 $\boldsymbol{d}_0$ 是最短距离点对矛盾.
    - $f(\boldsymbol{d}') < 0$ 等价于 $(\boldsymbol{d}_0 - \boldsymbol{c}_0)^\top (\boldsymbol{d}' - \boldsymbol{m})  =  (\boldsymbol{d}_0-\boldsymbol{c}_0)^\top (\boldsymbol{d}'-\boldsymbol{d}_0+\frac{\boldsymbol{d}_0-\boldsymbol{c}_0}{2})< 0$, 即 $(\boldsymbol{d}_0 - \boldsymbol{c}_0)^\top (\boldsymbol{d}_0-\boldsymbol{d}' ) > 0$. 这说明从 $\boldsymbol{d}_0$ 指向 $\boldsymbol{d}'$ 的向量与从 $\boldsymbol{d}_0$ 指向 $\boldsymbol{c}_0$ 的向量夹角为锐角, 即 $\boldsymbol{d}'$ 在 $\boldsymbol{d}_0$ 的“前方”.
    - 由于 $D$ 是凸集, 则对于任意 $t \in [0,1]$, 点 $\boldsymbol{d}_t = t \boldsymbol{d}' + (1-t) \boldsymbol{d}_0 \in D$.
    - 此时, 考虑距离函数 $h(t) = \|\boldsymbol{c}_0 - \boldsymbol{d}_t\|^2$. 则有:

      $$
h(t) = \|\boldsymbol{c}_0 - (t \boldsymbol{d}' + (1-t) \boldsymbol{d}_0)\|^2 = \|\boldsymbol{c}_0 - \boldsymbol{d}_0 + t(\boldsymbol{d}_0 - \boldsymbol{d}')\|^2
      $$

      对 $t$ 求导, 有:

      $$
h'(t) = 2(\boldsymbol{c}_0 - \boldsymbol{d}_0 + t(\boldsymbol{d}_0 - \boldsymbol{d}'))^\top (\boldsymbol{d}_0 - \boldsymbol{d}')
      $$

      在 $t=0$ 处, 有:

      $$
h'(0) = 2(\boldsymbol{c}_0 - \boldsymbol{d}_0)^\top (\boldsymbol{d}_0 - \boldsymbol{d}') < 0
      $$

      这说明在 $t=0$ 附近, $h(t)$ 是递减的. 因此, 存在一个足够小的正数 $\Delta t > 0$, 使得 $h(\Delta t) < h(0)$, 即 $\|\boldsymbol{c}_0 - \boldsymbol{d}_{\Delta t}\| < \|\boldsymbol{c}_0 - \boldsymbol{d}_0\|$. 这与 $\boldsymbol{c}_0$ 和 $\boldsymbol{d}_0$ 是最短距离点对矛盾.
- 类似地, 可证明对于任意 $\boldsymbol{c} \in C$, $f(\boldsymbol{c}) = (\boldsymbol{d}_0 - \boldsymbol{c}_0)^\top (\boldsymbol{c} - \boldsymbol{m}) \leq 0$.
- 因此, 取 $\boldsymbol{a} = \boldsymbol{d}_0 - \boldsymbol{c}_0$ 及 $b = \boldsymbol{a}^\top \boldsymbol{m}$, 则有:

$$
\boldsymbol{a}^\top \boldsymbol{x} \leq b, \quad \forall \boldsymbol{x} \in C, \quad \text{and} \quad \boldsymbol{a}^\top \boldsymbol{y} \geq b, \quad \forall \boldsymbol{y} \in D.
$$

$\square$

**Notes**
- 严格的分割 ($\boldsymbol{a}^\top \boldsymbol{x} < b$ 和 $\boldsymbol{a}^\top \boldsymbol{y} > b$) 并不是对所有不相交的凸集都成立的. 
  - 例如, 若 $C=\{(x,y): y \geq x\}, D =\{(x,y): y < x\}$, 则其不相交, 但不存在严格分割它们的超平面. 并且这与集合是否是闭集无关, 例如对于 $C=\{(x,y): y\ge 1/x, x\ge 0\}, D=\{(x,y): y \leq 0\}$, 其也是不相交的凸集, 但不存在严格分割它们的超平面.
  - 总的而言, 我们需要满足两个集合的距离 $d(C,D) > 0$ 时, 才能保证存在严格分割它们的超平面.

***Theorem* (Supporting Hyperplane Theorem)**: 对于非空凸集 $C \subset \mathbb{R}^n$ 及其边界上的任意点 $x_0 \in \text{bd}(C)$, 定存在一个超平面支持该点, 即存在非零向量 $\boldsymbol{a} \in \mathbb{R}^n$ 使得:

$$
C \subseteq \{\boldsymbol{x} : \boldsymbol{a}^\top \boldsymbol{x} \leq \boldsymbol{a}^\top \boldsymbol{x}_0\},
$$

> ***补充* ($\mathbb{R}^n$ 中的简单拓扑)**: 
> - **Open Ball**: 对于点 $x \in \mathbb{R}^n$ 及实数 $r>0$, 定义开球为 $B(x,r) = \{y \in \mathbb{R}^n: \|y-x\| < r\}$.
> - **Interior Point**: 对于集合 $C \subseteq \mathbb{R}^n$, 若存在 $r>0$ 使得开球 $B(x,r) \subset C$, 则称点 $x$ 为 $C$ 的内点. 数学上表示为: $\text{int}(C) = \{x \in C: \exists r>0, B(x,r) \subset C\}$.
> - **Exterior Point**: 对于集合 $C \subseteq \mathbb{R}^n$, 若存在 $r>0$ 使得开球 $B(x,r) \cap C = \emptyset$, 则称点 $x$ 为 $C$ 的外点.
> - **Boundary**: 对于集合 $C \subseteq \mathbb{R}^n$, 其边界定义为: $\text{bd}(C) = \text{cl}(C) \setminus \text{int}(C) = \mathbb{R}^n \setminus (\text{int}(C) \cup \text{ext}(C))$. 即边界上的点既不是内点, 也不是外点.
>    - **Closure**: 对于集合 $C \subseteq \mathbb{R}^n$,  其闭包定义为: $\text{cl}(C) = \{x \in \mathbb{R}^n: \forall r>0, B(x,r) \cap C \neq \emptyset\}$. 即对于 closure 中的任意点, 其任意小的邻域内均包含 $C$ 中的点.故 $\text{cl}(C) = \text{int}(C) \cup \text{bd}(C)$.
> - 给定 $C \subseteq \mathbb{R}^n$, 则空间中的任意点 $x$ 要么是 $C$ 的内点, 要么是外点, 要么是边界点.

*Proof.*

- 首先考虑一个非退化的场景. 设 $x_0 \in \text{bd}(C)$, 考虑 $\text{int}(C) \neq \emptyset$ 的情形.
  - 此时, 令 $A:=\{x_0\}, B:=\text{int}(C)$. 则 $A$ 和 $B$ 是不相交的凸集. 根据**Separating Hyperplane Theorem**, 存在非零向量 $\boldsymbol{a} \in \mathbb{R}^n$ 及标量 $b \in \mathbb{R}$, 使得:

    $$
\boldsymbol{a}^\top \boldsymbol{x} \leq b, \quad \forall \boldsymbol{x} \in A, \quad \text{and} \quad \boldsymbol{a}^\top \boldsymbol{y} \geq b, \quad \forall \boldsymbol{y} \in B.
    $$

    - 由于 $A$ 仅包含点 $x_0$, 则 $\boldsymbol{a}^\top \boldsymbol{x}_0 \leq b$. 故对于任意 $\boldsymbol{y} \in \text{int}(C)$, 有 $\boldsymbol{a}^\top \boldsymbol{y} \geq b \geq \boldsymbol{a}^\top \boldsymbol{x}_0$. 因此, 我们确定了这样一个超平面, 使得 $\text{int}(C)$ 在该超平面的另一侧.
  - 接下来, 我们需要从 $\text{int}(C)$ 推广到整个 $C$. 而推广的核心在于下述引理, 其核心思想为对于 $C$ 中的任意一个点 $\boldsymbol{x}$, 我们总能构造一串内点 $\boldsymbol{y}_{\epsilon_k}, \cdots$ 序列以逼近之. 其正式叙述如下. 
    - ***引理***. 设 $C\subset \mathbb{R}^n$ 且 $\text{int}(C)\neq \emptyset$. 任取 $x^* \in \text{int}(C)$, 定存在 $r>0$ 使得 (1) $B(x^\ast,r)\subseteq C$ ; (2) 对任意 $x \in C, \epsilon \in (0,1)$, 点 $y_\varepsilon:=(1-\varepsilon)x+\varepsilon x^\ast$ 属于 $\text{int}(C)$. 故当 $\varepsilon \to 0^+$ 时, $y_\varepsilon \to x$. (3) 更进一步, 以 $y_\varepsilon$ 为球心, 以 $\varepsilon r$ 为半径的开球 $B(y_\varepsilon, \varepsilon r)$ 亦包含于 $C$ 中. 
      - 对于 (1), 其根据内点的定义自然成立. 此时固定这个半径 $r$. 
      - 下证明 (2). 由于 $C$ 是凸的, 故 $y_\varepsilon \in C$. 为证明 $y_\varepsilon \in \text{int}(C)$, 需证 $\exists \rho>0, B(y_\varepsilon, \rho) \subseteq C$.  这等价于去证明对于 $B(y_\varepsilon, \rho)$ 内的任意一点 $z$, 有 $z\in C$. 而 $z\in B(y_\varepsilon, \rho)$ 又等价于 $\|z - y_\varepsilon\| < \rho$. 下证明, 当取 $\rho = \varepsilon r$ 时, 对任意 $z$ 满足 $\|z - y_\varepsilon\| < \epsilon r$, 有 $z\in C$.
        - 构造性地引入 (1) 中 $B(x^*,r)$ 的一个点 $w:=x^\ast+\frac{1}{\varepsilon}(z-y_\varepsilon)$ (可以证明, $\|w-x^*\|<r$, 即这样的到的 $w$ 定是在 (1) 中开球内的). 又由于 $B(x^*,r)\subset C$, 故 $w\in C$. 
        - 我们有 $x\in C$ (待逼近的点), $w\in C$, 且 $C$ 是凸的. 根据 $w = x^\ast+\frac{1}{\varepsilon}(z-y_\varepsilon)$ 以及 $y_\varepsilon = (1-\varepsilon)x + \varepsilon x^\ast$ 的定义, 可知

            $$
    z = y_\varepsilon + \varepsilon (w - x^\ast) = (1-\varepsilon)x + \varepsilon w
            $$

        - 因此, 根据凸集的定义, 可知 $z \in C$. 这就证明了 (2): $y_\varepsilon \in \text{int}(C)$.
    -  下使用这个引理完成推广的证明. 对于给定的任意 $\boldsymbol{x}\in C$, 我们要证 $\boldsymbol{a}^\top \boldsymbol{x} \leq \boldsymbol{a}^\top \boldsymbol{x}_0$. 根据引理, 取 $\varepsilon_k:=1/k$, 定义 $\boldsymbol{y}_k:=(1-\varepsilon_k)\boldsymbol{x}+\varepsilon_k \boldsymbol{x}^\ast.$ 则 $\boldsymbol{y}_k \in \text{int}(C)$ 且 $\lim_{k\to \infty} \boldsymbol{y}_k = \boldsymbol{x}$. 又由于 $\boldsymbol{x} \mapsto \boldsymbol{a}^\top \boldsymbol{x}$ 是 $\mathbb{R}^n$ 中的连续函数, 故 $\lim_{k\to \infty} \boldsymbol{a}^\top \boldsymbol{y}_k = \boldsymbol{a}^\top x$.
- 另一方面, 考虑 $\text{int}(C) = \emptyset$ 的情形 (例如在二维空间中的一个线段或三维空间中的一个平面多边形). 此时可以认为 $C$ 是“躺”在一个低维的仿射平面中. 此时这个仿射平面本身即为一个超平面, 且显然支持 $C$ 上的任意点. 故其也是平凡成立的. (此处具体数学证明略去, 可参考相关凸分析教材.)
  
$\square$

- **推论**: 给定一个闭集 $C$ 且 $\text{int}(C)\neq \emptyset$, 若对于边界上的任意一点 $c\in \text{bd}(C)$ 都存在对应的 supporting hyperplane, 则 $C$ 是凸集.
  
 
## Operations that Preserve Convexity

如下操作均能保持凸集的凸性:

- **Intersection**: 可列交的凸集的交集仍然是凸集. 即对于任意凸集族 $\{C_i\}_{i\in \mathcal{I}}$, 其交集 $\bigcap_{i\in \mathcal{I}} C_i$ 仍然是凸集.
- **Affine Transformation**: 设 $C \subseteq \mathbb{R}^n$ 是凸集, 且 $A \in \mathbb{R}^{m\times n}, \boldsymbol{b} \in \mathbb{R}^m$. 则仿射变换 $f(\boldsymbol{x}) = A\boldsymbol{x} + \boldsymbol{b}: \mathbb{R}^n \to \mathbb{R}^m$ 下的像 $f(C) = \{A\boldsymbol{x} + \boldsymbol{b}: \boldsymbol{x} \in C\}$ 仍然是凸集. 另外, 凸集 $D \subseteq \mathbb{R}^m$ 的逆像 $f^{-1}(D) = \{\boldsymbol{x} \in \mathbb{R}^n: A\boldsymbol{x} + \boldsymbol{b} \in D\}$ 亦是凸集.

  - **Example** 给定 $A_1, \cdots, A_k, B \in \mathbb{S}^n$, 定义 linear matrix inequality 为 $\boldsymbol{x}_1 A_1 + \boldsymbol{x}_2 A_2 + \cdots + \boldsymbol{x}_k A_k \preceq B$, 其中 $x\in \mathbb{R}^k$. 则该不等式所定义的集合 $\{\boldsymbol{x} \in \mathbb{R}^k: \boldsymbol{x}_1 A_1 + \boldsymbol{x}_2 A_2 + \cdots + \boldsymbol{x}_k A_k \preceq B\}$ 是凸集. 
  - *Proof*
    - Method 1: 可直接由定义, 给定 $x,y\in C$, 证明对任意 $t\in [0,1]$, 有 $tx + (1-t)y \in C$. 
    - Method 2: 该集合可表示为 $\{x\in \mathbb{R}^k: B - \sum_{i=1}^k x_i A_i \succeq 0\} = f^{-1}(\mathbb{S}^n_+)$, 其中 $f(x) = B - \sum_{i=1}^k x_i A_i$ 是一个仿射映射, 而 $\mathbb{S}^n_+$ 是 PSD 锥, 因此根据仿射映射下凸集的逆像仍然是凸集的性质, 可知该集合是凸集.

- **Perspective image and preimage**: 设 $C \subseteq \mathbb{R}^{n+1}$ 是凸集. 定义透视映射 $P: \mathbb{R}^{n} \times \mathbb{R}_{+} \to \mathbb{R}^n$ 为 $P(\boldsymbol{z}, t) = \boldsymbol{z}/t$. 则透视映射下的像 $P(C) = \{\boldsymbol{z}/t: (\boldsymbol{z}, t) \in C, t > 0\}$ 仍然是凸集. 另外, 凸集 $D \subseteq \mathbb{R}^n$ 的逆像 $P^{-1}(D) = \{(\boldsymbol{z}, t) \in \mathbb{R}^n \times \mathbb{R}_+: \boldsymbol{z}/t \in D\}$ 亦是凸集.

- **Linear-fractional image and preimage**: 设 $C \subseteq \mathbb{R}^{m}$ 是凸集, 且 $A \in \mathbb{R}^{m\times n}, \boldsymbol{b} \in \mathbb{R}^m, \boldsymbol{c} \in \mathbb{R}^n, d \in \mathbb{R}$. 定义线性分式映射 $F: \mathbb{R}^n \to \mathbb{R}^m$ 为 $F(\boldsymbol{x}) = (A\boldsymbol{x} + \boldsymbol{b})/(\boldsymbol{c}^\top \boldsymbol{x} + d)$, 其中定义域为 $\{\boldsymbol{x} \in \mathbb{R}^n: \boldsymbol{c}^\top \boldsymbol{x} + d > 0\}$. 则线性分式映射下的像 $F(C) = \{(A\boldsymbol{x} + \boldsymbol{b})/(\boldsymbol{c}^\top \boldsymbol{x} + d): \boldsymbol{x} \in C, \boldsymbol{c}^\top \boldsymbol{x} + d > 0\}$ 仍然是凸集. 另外, 凸集 $D \subseteq \mathbb{R}^m$ 的逆像 $F^{-1}(D) = \{\boldsymbol{x} \in \mathbb{R}^n: (A\boldsymbol{x} + \boldsymbol{b})/(\boldsymbol{c}^\top \boldsymbol{x} + d) \in D, \boldsymbol{c}^\top \boldsymbol{x} + d > 0\}$ 亦是凸集.


## Convex Functions

***Definition* (Convex Function)**: 对于函数 $f: \mathbb{R}^n \to \mathbb{R}$, 其是一个凸函数, 若满足 (1) 其定义域 $\text{dom}(f) \subseteq \mathbb{R}^n$ 是凸集; (2) 对任意 $x,y \in \text{dom}(f)$ 及 $t \in [0,1]$, 有:

$$
f(tx + (1-t)y) \leq t f(x) + (1-t) f(y)
$$

- **Strictly Convex Function**: 若对任意 $x \neq y \in \text{dom}(f)$ 及 $t \in (0,1)$, 有:

$$
f(tx + (1-t)y) < t f(x) + (1-t) f(y)
$$

  - 可以认为, strictly convex function 要 “更凸于“ 线性函数.
- **Strongly Convex Function**: 若存在常数 $m > 0$, 使得对于给定函数 $f$ 有: $f(x)-\frac{m}{2}\|x\|^2$ 是凸函数, 则称 $f$ 是 $m$-强凸函数. 
  - 可以认为, strongly convex function 要 “更凸于“ 二次函数.

显然有: strongly convex function $\implies$ strictly convex function $\implies$ convex function.

如下是一些典型的凸函数的例子:

- **Exponential Function**: $f(x) = e^{ax}$, 其中 $a \in \mathbb{R}$.
- **Power Function**: $f(x) = x^a$, 其中 $a \geq 1$ 或 $a \leq 0$, 定义域为 $\mathbb{R}_{++}$.
- **Affine Function**: $f(x) = a^\top x + b$, 其中 $a \in \mathbb{R}^n, b \in \mathbb{R}$. 其既是凸函数, 也是凹函数.
- **Quadratic Function**: 若给定 $Q\succeq 0$, 则 $f(x) = \frac{1}{2} x^\top Q x + b^\top x + c$ 是凸函数, 其中 $b \in \mathbb{R}^n, c \in \mathbb{R}$.
- **Least Squares Function**: $f(x) = \|Ax - b\|_2^2$, 其中 $A \in \mathbb{R}^{m\times n}, b \in \mathbb{R}^m$.
- **Norms**: 任何范数 $\|\cdot\|$ 都是凸函数.
    - $\ell_p$ norm: 对于 $p \geq 1$, 定义 $f(x) = \|x\|_p = (\sum_{i=1}^n |x_i|^p)^{1/p}$.
    - $\ell_\infty$ norm: 定义 $f(x) = \|x\|_\infty = \max_{1\leq i \leq n} |x_i|$.
    - operator norm: 对于给定矩阵 $X$ 及其对应 singular values $\sigma_1(X) \geq \sigma_2(X) \geq \sigma_r(X) \geq 0$, 则定义 $\|X\|_{\text{op}} = \sigma_1(X)$.
    - nuclear norm / trace norm: 定义 $\|X\|_{\text{nuc}} = \sum_{i=1}^r \sigma_i(X)$.
    - 注意: $\ell_0$ 'norm': 定义 $f(x) = \|x\|_0 = |\{i: x_i \neq 0\}|$. 虽然其被称为“范数”, 但实际上并不满足范数的定义 (不满足正齐次性及三角不等式). 并且其不是凸函数.
- **Indicator Function**: 对凸集 $C$, 定义指标函数为:

    $$
    I_C(x) = \begin{cases}
    0, & x \in C \\
    +\infty, & x \notin C
    \end{cases}
    $$

  - 显然, 指标函数是凸函数. 因为其定义域即为凸集 $C$, 且在该定义域内函数值恒为零 (线性函数), 在定义域外函数值恒为无穷大.
- **Support Function**: 对任意集合 $C$ (不对其凸性作要求), 其 support function 定义为:

    $$
    S_C(x) = \sup_{y \in C} x^\top y
    $$

  - 支持函数是凸函数. 因为对于任意 $x_1, x_2 \in \mathbb{R}^n$ 及 $t \in [0,1]$, 有:

    $$
    \begin{aligned}
    S_C(tx_1 + (1-t)x_2)
    &= \sup_{y \in C} (tx_1 + (1-t)x_2)^\top y \\
    &\le t \sup_{y \in C} x_1^\top y + (1-t) \sup_{y \in C} x_2^\top y \\
    &= t S_C(x_1) + (1-t) S_C(x_2).
    \end{aligned}
    $$

- **Max Function**: $f(x) = \max\{x_1, x_2, \ldots, x_n\}$ 是凸函数. 

## Key Properties of Convex Functions

***Claim* (Epigraph Characterization)**: 函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是凸函数的充分必要条件为其**上图** (epigraph) 是凸集. 其中, 上图定义为:

$$
\text{epi}(f) = \{(x,t) \in \mathbb{R}^n \times \mathbb{R}: t \geq f(x)\}
$$

***Claim* (Sublevel Set Characterization)**: 函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是凸函数, 可以推出其任意**次水平集** (sublevel set) 是凸集. 其中, 次水平集定义为:

$$
\text{sublevel}_\alpha(f) = \{x \in \text{dom}(f): f(x) \leq \alpha\}, \quad \forall \alpha \in \mathbb{R}
$$

- 反之不成立, 即任意次水平集均为凸集并不能推出函数是凸函数, 只能称其为 **Quasi-Convex Function**.
  - 例如如下 $f(x) = \log(|x| + 1)$, 其任意次水平集均为凸集, 但该函数并非凸函数. 
       ![y=log(|x|+1)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260108115514.png)


***Theorem* (First-Order Characterization)**: 函数 $f: \mathbb{R}^n \to \mathbb{R}$ 在凸集 $\text{dom}(f)$ 上可微, 则 $f$ 是凸函数的充分必要条件为:

$$
f(y) \geq f(x) + \nabla f(x)^\top (y - x), \quad \forall x,y \in \text{dom}(f)
$$

- 该不等式表明, 函数在任意点处的切线 (或切平面) 都在函数图像的下方.

***Theorem* (Second-Order Characterization)**: 函数 $f: \mathbb{R}^n \to \mathbb{R}$ 在凸集 $\text{dom}(f)$ 上二阶可导, 则 $f$ 是凸函数的充分必要条件为:

$$
\nabla^2 f(x) \succeq 0, \quad \forall x \in \text{dom}(f)
$$

- 即函数的 Hessian 矩阵在定义域内处处为正半定矩阵.
- 注意, strictly convex function 并不能推出 $\nabla^2 f(x) \succ 0$. 例如 $f(x) = x^4$ 是严格凸函数, 但其 Hessian (即二阶导数) 在 $f''(x) = 12x^2$, 在 $x=0$ 处并不严格正定.

***Theorem* (Jensen's Inequality)**: 设 $f: \mathbb{R}^n \to \mathbb{R}$ 是凸函数, 且 $X$ 是取值于 $\text{dom}(f)$ 的随机变量. 则有:

$$
f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]
$$

***Example* (Convexity of Log-Sum-Exp Function)**: 函数 $g(x) = \log(\sum_{i=1}^n \exp(\boldsymbol{a}_i^\top x + b_i))$, 其中 $\boldsymbol{a}_i \in \mathbb{R}^n, b_i \in \mathbb{R}$, 是凸函数.


## Operations that Preserve Convexity of Functions

如下操作均能保持凸函数的凸性:

- **Nonnegative Linear Combination**: 设 $f_1, f_2, \ldots, f_m: \mathbb{R}^n \to \mathbb{R}$ 是凸函数, 且 $a_1, a_2, \ldots, a_m \geq 0$ 是非负实数. 则函数 $f(x) = \sum_{i=1}^m a_i f_i(x)$ 仍然是凸函数.
- **Pointwise Maximum**: 设 $f_s: \mathbb{R}^n \to \mathbb{R}, s \in S$ 是一族凸函数, 则函数 $f(x) = \sup_{s \in S} f_s(x)$ 仍然是凸函数.
- **Partial Minimization**: 设 $f: \mathbb{R}^{n+m} \to \mathbb{R}$ 是凸函数, 则对于任意固定的 $y \in \mathbb{R}^m$, 函数 $g(x) = \inf_{y \in \mathbb{R}^m} f(x,y)$ 仍然是凸函数.
- **Affine Composition**: 设 $f: \mathbb{R}^m \to \mathbb{R}$ 是凸函数, 且 $A \in \mathbb{R}^{m\times n}, b \in \mathbb{R}^m$. 则函数 $g(x) = f(Ax + b)$ 仍然是凸函数.
- **General Composition**: 给定 $f: \mathbb{R}^n \to \mathbb{R}, g: \mathbb{R}^n \to \mathbb{R}$ 及 $h = \mathbb{R}\to\mathbb{R}$, 且 $f(x) = h(g(x))$, 则有如下组合关系:
    - 若 $h$ 是 convex 非减, 且 $g$ 是 convex, 则 $f$ 是 convex.
    - 若 $h$ 是 convex 非增, 且 $g$ 是 concave, 则 $f$ 是 convex.
