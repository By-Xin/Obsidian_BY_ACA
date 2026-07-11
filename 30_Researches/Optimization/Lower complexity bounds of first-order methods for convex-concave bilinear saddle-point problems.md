# Lower complexity bounds of first-order methods for convex-concave bilinear saddle-point problems

## 1. Introduction

### 1.1 Intuition 

本文从理论上尝试分析对于大规模 convex-concave bilinear saddle-point problems, 任何 deterministic first-order method 的算法收敛下界. 

回顾基本的一阶算法及其收敛性质. 
- 对于 Projected Gradient Method, 其更新为:
    $$
    \mathbf{x}^{t+1} = \Pi_{\mathcal{X}}(\mathbf{x}^t - \alpha \nabla f(\mathbf{x}^t)),
    $$
    其中 $\Pi_{\mathcal{X}}$ 是投影算子, $\alpha$ 是步长.
- 若 $f$ 是 convex, 且 $L_f$-smooth, 则通过合适的步长选择, Projected Gradient Method 可以达到收敛率
    $$
    f(\mathbf{x}^t) - f(\mathbf{x}^*) = \mathcal{O}\left(\frac{1}{t}\right).
    $$
- 进一步, 若使用 Nesterov's accelerated gradient method, 则可以达到更快的收敛率
    $$
    f(\mathbf{x}^t) - f(\mathbf{x}^*) = \mathcal{O}\left(\frac{1}{t^2}\right).
    $$
    并且已有证明说明, 该收敛率就是最优的.

本文则将考虑如下 bilinear saddle-point problem (SPP):
$$
\min_{\mathbf{x} \in \mathcal{X}} \max_{\mathbf{y} \in \mathcal{Y}}
\mathcal{L}(\mathbf{x}, \mathbf{y}) 
= 
f(\mathbf{x}) + \langle \mathbf{A}\mathbf{x} - \mathbf{b}, \mathbf{y} \rangle - g(\mathbf{y}), \tag{SPP}
$$
- 其中 $\mathcal{X} \subseteq \mathbb{R}^{n}$ 和 $\mathcal{Y} \subseteq \mathbb{R}^{m}$ 是 closed and convex, $f: \mathbb{R}^n \to \mathbb{R}$ 和 $g: \mathbb{R}^m \to \mathbb{R}$ 是 closed and convex, $\mathbf{A} \in \mathbb{R}^{m\times n}$, $\mathbf{b} \in \mathbb{R}^{m}$. 
- 假设 $f$ 是 $L_f$-smooth:
    $$
    \|\nabla f(\mathbf{x_1}) - \nabla f(\mathbf{x}_2)\| \leq L_f \|\mathbf{x_1} - \mathbf{x}_2\|, \quad \forall \mathbf{x}_1, \mathbf{x}_2 \in \mathcal{X}.
    $$
- 假设 $g$ 是一个较为简单的函数, 其 proximal operator 可以被高效计算.

对于该问题, 有 primal 和 dual 两重视角:
- Primal: 即先对内层 $\mathbf{y}$ 进行最大化, 再对外层 $\mathbf{x}$ 进行最小化, 得到 primal problem:
    $$
    \phi^* := \min_{\mathbf{x} \in \mathcal{X}} \left\{
        \phi(\mathbf{x}) := f(\mathbf{x}) + \max_{\mathbf{y} \in \mathcal{Y}} [\langle \mathbf{A}\mathbf{x} - \mathbf{b}, \mathbf{y} \rangle - g(\mathbf{y})]
        \right\}
    $$

- Dual: 即先对外层 $\mathbf{x}$ 进行最小化, 再对内层 $\mathbf{y}$ 进行最大化, 得到 dual problem:
    $$
    \psi^* := \max_{\mathbf{y} \in \mathcal{Y}} \left\{
        \psi(\mathbf{y}) := -g(\mathbf{y}) + \min_{\mathbf{x} \in \mathcal{X}} [\langle \mathbf{A}\mathbf{x} - \mathbf{b}, \mathbf{y} \rangle + f(\mathbf{x})]
        \right\}
    $$

- 不难得到, 弱对偶性是一直成立的, 即
    $$
    \psi^* \leq \phi^*.
    $$
    因为对任意固定的 $\mathbf{x} \in \mathcal{X}$ 与 $\mathbf{y} \in \mathcal{Y}$, 有
    $$
    \psi = 
    \min_{\mathbf{x}' \in \mathcal{X}} \mathcal{L}(\mathbf{x}', \mathbf{y}) \leq \mathcal{L}(\mathbf{x}, \mathbf{y}) \leq \max_{\mathbf{y}' \in \mathcal{Y}} \mathcal{L}(\mathbf{x}, \mathbf{y}') = \phi.
    $$
    故各自的最优值满足 $\psi^* \leq \phi^*$.


- 在一些温和条件, 如 $\mathcal{X}$ 和 $\mathcal{Y}$ 是 compact, 强对偶性也可以得到, 即 $\psi^* = \phi^*$. 此时有 saddle point $(\mathbf{x}^*, \mathbf{y}^*)$ 满足
    $$
    \mathcal{L}(\mathbf{x}^*, \mathbf{y}) \leq \mathcal{L}(\mathbf{x}^*, \mathbf{y}^*) \leq \mathcal{L}(\mathbf{x}, \mathbf{y}^*), \quad \forall \mathbf{x} \in \mathcal{X}, \forall \mathbf{y} \in \mathcal{Y}.
    $$

事实上, 许多的优化问题都可以整理为 SPP, 一个很重要的子类是 affinely constrained smooth convex optimization:
$$
f^* := \min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x}) \quad \text{s.t. } \mathbf{A}\mathbf{x} = \mathbf{b}, \tag{ACSCP}
$$
- 下说明, 当前这个例子就等价于将标准形式中 $\mathcal{Y} = \mathbb{R}^m$, $g \equiv 0$.  具体地, 此时标准形式为:
    $$
    \min_{\mathbf{x} \in \mathcal{X}}  f(\mathbf{x}) +  \max_{\mathbf{y} \in \mathbb{R}^m} \langle \mathbf{A}\mathbf{x} - \mathbf{b}, \mathbf{y} \rangle.
    $$
    因此对于后面的最大化问题, 当且仅当 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 时, 其最大值才是有限的且为零, 否则其最大值为 $+\infty$.  因此二者等价. 

### 1.2 Main Goal

#### 1.2.1 Objective

本文的主要结构如下. 文中想要证明的核心命题是, 对于 SPP 问题, 任意的 deterministic first-order method, 其下界为 $\Omega(1/t)$ (凸) 或 $\Omega(1/t^2)$ (强凸). 不过, 由于 SPP 的 generality, 文章会先从 ACSCP 问题出发, 证明出其下界, 然后再将其推广到 SPP 问题.

#### 1.2.2 Analysis Framework

在本文的 deterministic first-order method分析中涉及到如下几个概念和问题. 

- Deterministic 表明算法不涉及随机性, 给定相同的初始点和参数, 算法每次迭代都会产生相同的结果.
- First-order method 可以理解为黑箱 Oracle 能够返回的信息仅限于函数值和梯度信息. 对于本文的 SPP, 其 Oracle 可以返回如下信息:
  $$
  O(\mathbf{x}, \mathbf{y}) := (\nabla f(\mathbf{x}), \mathbf{A}\mathbf{x}, \mathbf{A}^\top \mathbf{y}). 
  $$
  - 对 SPP 目标函数求关于 $\mathbf{x}$ 的梯度, 其结果为 $\nabla f(\mathbf{x}) + \mathbf{A}^\top \mathbf{y}$; 求关于 $\mathbf{y}$ 的梯度, 其结果为 $\mathbf{A}\mathbf{x} - \mathbf{b} - \nabla g(\mathbf{y})$. 当然其他的一些信息如 $\mathbf{b}$ 是已知的; 和 $g$ 有关的信息, 文中认为其是简单的, 因此同样不作为 Oracle 的返回信息.
  - 具体的黑箱计算过程如下:
    - 算法的迭代初始点为 $(\mathbf{x}^{(0)}, \mathbf{y}^{(0)}) \in \mathcal{X} \times \mathcal{Y}$, 迭代次数为 $t = 0, 1, 2, \ldots$.
    - 在第 $t$ 次迭代中, 算法会在当前迭代点的 inquiry point $(\mathbf{x}^{(t)}, \mathbf{y}^{(t)})$ 处调用 Oracle, 得到
      $$
      O(\mathbf{x}^{(t)}, \mathbf{y}^{(t)}) = (\nabla f(\mathbf{x}^{(t)}), \mathbf{A}\mathbf{x}^{(t)}, \mathbf{A}^\top \mathbf{y}^{(t)}).
      $$
    - 然后算法根据当前迭代点和 Oracle 返回的信息, 计算出下一次迭代的 inquiry point $(\mathbf{x}^{(t+1)}, \mathbf{y}^{(t+1)})$ 和最终用来返回作为输出的点 $(\bar{\mathbf{x}}^{(t+1)}, \bar{\mathbf{y}}^{(t+1)})$. 迭代过程可以表示为:
      $$
      (\mathbf{x}^{(t+1)}, \mathbf{y}^{(t+1)}, \bar{\mathbf{x}}^{(t+1)}, \bar{\mathbf{y}}^{(t+1)}) = \mathcal{I}_t\left( \boldsymbol{\vartheta}; O(\mathbf{x}^{(0)}, \mathbf{y}^{(0)}), \ldots, O(\mathbf{x}^{(t)}, \mathbf{y}^{(t)})\right), \quad \text{(1)}
      $$
      - 这里之所以区分 $(\mathbf{x}^{(t+1)}, \mathbf{y}^{(t+1)})$ 和 $(\bar{\mathbf{x}}^{(t+1)}, \bar{\mathbf{y}}^{(t+1)})$, 是因为有些算法 (例如 Nesterov's accelerated gradient method), 用来查询梯度信息和最终返回作为决策变量的点是不同的. 因此这样表达可以更为一般化.
      - $\boldsymbol{\vartheta}$ 是本身问题包含的所有静态信息, 例如 $\mathbf{A}$, $\mathbf{b}$, $L_f$, $\mathcal{X}$, $\mathcal{Y}$ 等等. 这些信息独立于迭代之外, 是随着问题的定义而固定的. 
      - $\mathcal{I}_t$ 是算法在第 $t$ 次迭代中能够利用全部历史信息, 通过任意方式组合的任意规则. 

#### 1.2.3 Example: LALM on Affinely Constrained Problem

下面通过一个具体的优化算法例子辅助理解整个算法框架. 针对仿射约束问题:
$$
\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x}) \quad \text{s.t. } \mathbf{A}\mathbf{x} = \mathbf{b},
$$
考虑如下的 linearized augmented Lagrangian method (LALM) 优化算法:
$$
\begin{aligned}
\mathbf{x}^{(t+1)} &= \text{Proj}_{\mathcal{X}}\left(\mathbf{x}^{(t)} - \frac{1}{\eta} \left(\nabla f(\mathbf{x}^{(t)}) \right) + \mathbf{A}^\top (\boldsymbol{\lambda}^{(t)} + \mathbf{r}^{(t)})\right), \\
\boldsymbol{\lambda}^{(t+1)} &= \boldsymbol{\lambda}^{(t)} + \mathbf{r}^{(t+1)}, \\
\end{aligned}
$$
其中 $\mathbf{r}^{(t)} = \mathbf{A}\mathbf{x}^{(t)} - \mathbf{b}$ 是当前的残差, $\eta>0$ 是步长参数. 

- 回顾, 在正常的 ALM 中, $\mathbf{x}^{(t+1)}$ 需要通过最小化二次的 augmented Lagrangian function $L_\rho(\mathbf{x}, \boldsymbol{\lambda}^{(t)}) := f(\mathbf{x}) + \langle \boldsymbol{\lambda}^{(t)}, \mathbf{A}\mathbf{x} - \mathbf{b} \rangle + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ 来得到, 然而其求解往往是很昂贵或困难的. 
- 故 ALAM 只会对 $L_\rho$ 进行一步梯度下降, 也就是 linearized, 然后再投影回 feasible set $\mathcal{X}$. 此外论文中取 $\rho = 1$ 作为 penalty parameter, 因此在这里就不再显式地写出 $\rho$.  


文中为分析方便, 进一步假设 $\mathcal{X} = \mathbb{R}^n$, 也就是没有约束, 因此 $\text{Proj}_{\mathcal{X}}$ 可以省略. 且令初始化 $\boldsymbol{\lambda}^{(0)} = \mathbf{x}^{(0)} = \mathbf{0}$. 则 LALM 的迭代过程可以展开并整理为:
$$
\begin{aligned}
\mathbf{x}^{(t+1)} &= - \frac{1}{\eta} \left(
    \sum_{j=0}^{t}  \left(\nabla f(\mathbf{x}^{(j)}) + \mathbf{A}^\top \mathbf{r}^{(j)}\right) +
    \mathbf{A}^\top \sum_{j=1}^{t} \mathbf{r}^{(j)}
\right)
\end{aligned}
$$
- 其中 $\boldsymbol{\lambda}^{(t)} = \sum_{j=1}^{t} \mathbf{r}^{(j)}$ 是通过累加残差得到的, 并且规定 $\sum_{b}^{a} (\cdot)= 0$ 若 $b > a$.
- 观察该表达式, 即可发现, 这样的 $\mathbf{x}^{(t+1)}$ 就是通过所有历史梯度 $\nabla f(\mathbf{x}^{(j)})$ 和所有历史 $\mathbf{A}^\top \mathbf{r}^{(j)}$ 的线性组合得到的. 换言之, $\mathbf{x}^{(t+1)}$ 落在其历史梯度和残差的 linear span 中. 
  - **这里同时还指出, 这个 linear span 的形式是由 $\mathcal{X} = \mathbb{R}^n$ 决定的, 并且会给后续的分析提供便利. 若是更一般的有约束情景, 则确实仍需要进行投影操作, 则此时 linear span 不再成立, 需要引入新的表示方法**
- 此外文中具体给出了如何将 LALM 的迭代过程嵌入到 (1) 中, 也就是如何定义 $\mathcal{I}_t$ 来表示 LALM 的迭代过程. 这里具体细节略, 其核心目的主要是为展示该方法的通用性框架.


最后, 注意到 LALM 的误差迭代收敛率 $|f(\mathbf{x}^{(t)}) - f(\mathbf{x}^\star)| = \mathcal{O}(1/t)$, 其可行性误差 $\|\mathbf{A}\mathbf{x}^{(t)} - \mathbf{b}\| = \mathcal{O}(1/t)$. 本文试图说明, 针对这样的约束仿射问题, 任何 deterministic first-order method 都无法突破 $\mathcal{O}(1/t)$ 的收敛率下界. 也就是说, LALM 已经是最优的了.


## 2. Lower Complexity Bounds under Linear Span Assumption for Affinely Constrained Problems

第一个部分首先考虑上述 ACSCP 问题. 要证明对于该类问题, 任意 deterministic first-order method 都无法突破 $\mathcal{O}(1/t)$ 的收敛率下界, 只需构造出一个 hard instance, 使得任意 deterministic first-order method 都无法在该实例上突破 $\mathcal{O}(1/t)$ 的收敛率下界.

这里考虑如下凸二次规划问题作为 hard instance:
$$
\begin{aligned}
f^* := \min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}) := \frac{1}{2}\mathbf{x}^\top \mathbf{H}\mathbf{x} - \mathbf{h}^\top \mathbf{x}
\quad \text{s.t. } \mathbf{A}\mathbf{x} = \mathbf{b}, \tag{Q}
\end{aligned}
$$
- 其中 $\mathbf{H} \in \mathbb{S}_+^n$, $\mathbf{A} \in \mathbb{R}^{m\times n}$ (假设 $m \leq n$). 注意到 $\nabla f$ 是 Lipschitz continuous. 此外给定一个固定的 horizon $k < m/2$, 这在大规模问题上往往是合理的. 
- 之所以选择二次规划问题作为 hard instance, 是因为其梯度 $\nabla f(\mathbf{x}) = \mathbf{H}\mathbf{x} - \mathbf{h}$ 是线性的便于分析, 然而其又可以通过具体的设计来为优化迭代制造出足够的麻烦. 

在当前 section 中, 假设我们只考虑符合 linear span assumption 形式的 deterministic first-order method.

***Assumption* (Linear Span Assumption)**: 对于迭代序列 $\{\mathbf{x}^{(t)}\}_{t=0}^{\infty}$, 其满足 $\mathbf{x}^{(0)} = \mathbf{0}$, 且对于任意 $t \geq 1$, 有
$$
\mathbf{x}^{(t)} \in \text{span}\{\nabla f(\mathbf{x}^{(0)}), \mathbf{A}^\top \mathbf{r}^{(0)}), \ldots, \nabla f(\mathbf{x}^{(t-1)}), \mathbf{A}^\top \mathbf{r}^{(t-1)}\}.
$$
其中 $\mathbf{r}^{(j)} = \mathbf{A}\mathbf{x}^{(j)} - \mathbf{b}$ 是第 $j$ 次迭代的残差.

- 上文的 LALM 算法在无约束 $\mathcal{X} = \mathbb{R}^n$ 下就是一个满足该假设的例子. 
- $\mathbf{x}^{(0)} = \mathbf{0}$ 是不失一般性的, 因为总可以通过平移将任意的初始点 $\mathbf{x}^{(0)}$ 转换为 $\mathbf{0}$, 并且相应地调整 $\mathbf{h}$ 和 $\mathbf{b}$.
- 不过也需要指出, 如果算法中包含 Projection 并且要作用到一个有约束的 $mathcal{X}$ 上, 则由于引入非线性映射, 该假设就不再成立. 这将在下一个 section 通过引入新的技术来解决.

### 2.1 Special Linear Constraints

这个小节将具体给出上述 quadratic problem (Q) 的 hard instance 的约束部分的设计. 

规定 $\mathbf{O}$ 是零矩阵, $\mathbf{1}$ 是全 1 向量, $\mathbf{0}$ 是全 0 向量. 定义分块矩阵和对应向量
$$
\boldsymbol{\Lambda}:=
\begin{bmatrix}
\mathbf{B}_{2k \times 2k} & \mathbf{O} \\
\mathbf{O} & \mathbf{G}
\end{bmatrix} \in \mathbb{R}^{m\times n},\qquad 
\mathbf{c} :=
\begin{bmatrix}\mathbf{1}_{2k} \\
\mathbf{0}
\end{bmatrix} \in \mathbb{R}^{m},
$$ 
- 其中
    $$
    \mathbf{B} :=
    \begin{bmatrix}
    &&&-1&1\\
    &&\vdots&\vdots&\\
    &-1&1&&\\
    -1&1&&&\\
    1&&&&
    \end{bmatrix} \in \mathbb{R}^{2k\times 2k},
    $$
    是一个双副对角的矩阵结构, 是整个构造的 '麻烦来源'. 其作用在任意向量 $\mathbf{u} = (u_1, \ldots, u_{2k})^\top$ 上, 其结果为
    $$
    \mathbf{B}\mathbf{u} = (u_{2k} - u_{2k-1}, \ldots, u_2 - u_1, u_1)^\top.
    $$
    注意到这是一种相邻元素差分的形式, 故 $\mathbf{B}$ 相当于离散的差分算子 (类似求导).
    - 这个顺序至关重要: 验证 $\mathbf{B}\mathbf{1}_{2k} = (0,\ldots,0,1)^\top = \mathbf{e}_{2k}$, 这正是 $\mathcal{K}_0 = \text{span}\{\mathbf{e}_{2k,n}\}$ 的来源, 也解释了为什么后面 $\mathcal{F}_i$ 从左边 ($\mathbf{e}_1,\ldots,\mathbf{e}_i$) 增长, 而 $\mathcal{K}_i$ 从右边 ($\mathbf{e}_{2k-i},\ldots,\mathbf{e}_{2k}$) 增长. 根据 $\mathbf{B}$ 的结构, 可以立刻得到如下性质:
    - $\|\mathbf{B}\| \leq 2$.
      - *Proof*. 这是由于 $\|\mathbf{B} \mathbf{u}\|^2 = u_1^2 + \sum_{i=1}^{2k-1} (u_{i+1} - u_i)^2 \leq 4\sum_{i=1}^{2k} u_i^2 = 4\|\mathbf{u}\|^2$, 因此 $\|\mathbf{B}\| \leq 2$.
    - $\|\boldsymbol{\Lambda}\| = \max\{\|\mathbf{B}\|, \|\mathbf{G}\|\} = 2$.
      - *Proof Sketch*.  这可以根据 $\mathbf{B}$ 和 $\mathbf{G}$ 的分块结构直接得到.
    - $\mathbf{B}$ 是 full row rank, 且 $\mathbf{B}^{-1}$ 的结构为
    $$
    \mathbf{B}^{-1} =
    \begin{bmatrix}
    1 &  && \\
    1 & 1 &  & \\
    \vdots &  & \ddots & \\
    1 & 1 & \cdots & 1 
    \end{bmatrix} \in \mathbb{R}^{2k\times 2k}.
    $$

- 对应的 $\mathbf{G} \in \mathbb{R}^{(m-2k)\times (n-2k)}$ 是任意 full row rank 矩阵, 且 $\|\mathbf{G}\| = 2$.  其用于进行提高问题维度, 不过对应的是 $\mathbf{c}$ 中的零块, 不会影响到后续的分析, 故具体形式并不重要. 

根据定义好的 $\boldsymbol{\Lambda}$ 和 $\mathbf{c}$, 则可以正式定义出 hard instance 的约束部分为
$$
\mathbf{A} = \frac{L_A}{2}\boldsymbol{\Lambda},\quad \mathbf{b} = \frac{L_A}{2}\mathbf{c},
$$
- 其中 $L_A \geq 0$ 是一个可调的参数. 通过缩放, 可以得到变换后的谱范数为
    $$
    \|\mathbf{A}\| = \frac{L_A}{2}\|\boldsymbol{\Lambda}\| = L_A.
    $$


通过这样的设计, 可以得到满足 $\mathbf{A}\mathbf{x}^* = \mathbf{b}$ 的方程的解 $\mathbf{x}^*$ 的结构为
$$
x_i^* = i, \quad i = 1, \ldots, 2k, \quad x_i^* = 0, \quad i = 2k+1, \ldots, n.
$$
即
$\mathbf{x}^* = (1, 2, \ldots, 2k, 0, \ldots, 0)^\top \in \mathbb{R}^n$.
- *Proof Sketch.* 该结构可以通过分块矩阵直接计算得到. 



### 2.2 Krylov Subspace

#### Krylov Subspace Introduction

首先介绍 Krylov subspace 的概念. 

***Definition* (Krylov Subspace)**: 给定一个矩阵 $\mathbf{M} \in \mathbb{R}^{n\times n}$ 和一个向量 $\mathbf{v} \in \mathbb{R}^n$, 则其 Krylov subspace of order $j$ 定义为
$$
\mathcal{K}_j(\mathbf{M}, \mathbf{v}) := \text{span}\{\mathbf{v}, \mathbf{M}\mathbf{v}, \ldots, \mathbf{M}^{j-1}\mathbf{v}\}.
$$
- 直观理解, Krylov subspace 就是反复对 $\mathbf{v}$ 进行矩阵 $\mathbf{M}$ 的线性变换所生成的向量所能够张成的线性空间. 
- 之所以需要引入 Krylov subspace, 是因为某种意义上, 任何的 deterministic first-order method 都相当于是在反复地进行 Matrix-vector multiplication. 因此算法的迭代点 $\mathbf{x}^{(t)}$ 都会落在某个 Krylov subspace 中. 这将为后续的分析提供便利.

#### Krylov Subspace in Current Hard Instance

具体落实在当前的 hard instance 上, 定义如下两个 Krylov subspace:
$$
\mathcal{J}_i := \text{span}\{\mathbf{c}, (\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top)\mathbf{c}, \ldots, (\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top)^i\mathbf{c}\} \subseteq \mathbb{R}^m, \quad i = 0, 1, \ldots
$$
以及
$$
\mathcal{K}_i := \boldsymbol{\Lambda}^\top \mathcal{J}_i \subseteq \mathbb{R}^n, \quad i = 0, 1, \ldots
$$
- 其中 $\mathcal{J}_i \subseteq \mathbb{R}^m$ 是在约束空间上的 Krylov subspace. 而 $\mathcal{K}_i \subseteq \mathbb{R}^n$ 是在决策变量空间 上的 Krylov subspace, 其是通过 $\boldsymbol{\Lambda}^\top$ 将 $\mathcal{J}_i$ 映射到 $\mathbb{R}^n$ 上得到的.
- 从代数的角度看, 整个线性映射的关系如下. 给定一个决策 $\mathbf{x} \in \mathbb{R}^n$, 其通过 $\mathbf{A} = \frac{L_A}{2}\boldsymbol{\Lambda}$ 映射到约束空间 $\mathbb{R}^m$ 上, 得到 $\mathbf{A}\mathbf{x} \in \mathbb{R}^m$ 衡量其在各个约束上的表现. 然后通过 $\mathbf{A}^\top = \frac{L_A}{2}\boldsymbol{\Lambda}^\top$ 将约束空间的向量映射回决策变量空间 $\mathbb{R}^n$ 指导更新下一次的优化迭代. 这里 $\mathbf{A}^\top$ 是 $\mathbf{A}$ 的 adjoint operator, 其满足:
    $$
    \langle \mathbf{A}\mathbf{x}, \mathbf{r} \rangle_{\mathbb{R}^m} = \langle \mathbf{x}, \mathbf{A}^\top \mathbf{r} \rangle_{\mathbb{R}^n}, \quad \forall \mathbf{x} \in \mathbb{R}^n, \forall \mathbf{r} \in \mathbb{R}^m.
    $$

#### Reduced Krylov Subspace

注意到, 对于 $\mathcal{J}_i$ 和 $\mathcal{K}_i$, 或者说其对应的矩阵 $\boldsymbol{\Lambda}$ 和向量 $\mathbf{c}$, 其在结构上只有前 $2k$ 个元素是 active 的.  通过矩阵计算, 可以发现
$$
(\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top)^i \mathbf{c} =
\begin{bmatrix}
    (\mathbf{B}\mathbf{B}^\top)^i \mathbf{1}_{2k} \\
    \mathbf{0}_{m-2k}
\end{bmatrix}, \quad i = 0, 1, \ldots 
$$
以及
$$
\boldsymbol{\Lambda}^\top (\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top)^i \mathbf{c} =
\begin{bmatrix}
    \mathbf{B}^{2i+1} \mathbf{1}_{2k} \\
    \mathbf{0}_{n-2k}
\end{bmatrix}, \quad i = 0, 1, \ldots
$$
因此我们事实上只需要考虑 $\mathcal{J}_i$ 和 $\mathcal{K}_i$ 的前 $2k$ 个元素, 得到约简后的 Krylov subspace:
$$
\mathcal{F}_i := \text{span}\{\mathbf{1}_{2k}, (\mathbf{B}\mathbf{B}^\top)\mathbf{1}_{2k}, \ldots, (\mathbf{B}\mathbf{B}^\top)^i\mathbf{1}_{2k}\} \subseteq \mathbb{R}^{2k}, \quad i = 0, 1, \ldots
$$
以及
$$
\mathcal{R}_i := \text{span}\{\mathbf{B}\mathbf{1}_{2k}, \mathbf{B}(\mathbf{B}\mathbf{B}^\top)\mathbf{1}_{2k}, \ldots, \mathbf{B}(\mathbf{B}\mathbf{B}^\top)^i\mathbf{1}_{2k}\} \subseteq \mathbb{R}^{2k}, \quad i = 0, 1, \ldots
$$
- 这里需要指出, 在文中简记 $\mathbf{B}\mathbf{B}^\top = \mathbf{B}^2$. 事实上 $\mathbf{B}$ 是一个 Hankel 矩阵 ($B_{ij}$ 仅依赖于 $i+j$: 当 $i+j=2k$ 时为 $-1$, 当 $i+j=2k+1$ 时为 $1$), 因此 $\mathbf{B}^\top = \mathbf{B}$, $\mathbf{B}\mathbf{B}^\top = \mathbf{B}^2$ 是严格等式. 因此, 按照文章的写法, 有:
    $$
    \mathcal{F}_i := \text{span}\{\mathbf{1}_{2k}, \mathbf{B}^2\mathbf{1}_{2k}, \ldots, \mathbf{B}^{2i}\mathbf{1}_{2k}\} \subseteq \mathbb{R}^{2k}, \quad i = 0, 1, \ldots
    $$
    以及
    $$
    \mathcal{R}_i := \text{span}\{\mathbf{B}\mathbf{1}_{2k}, \mathbf{B}^3\mathbf{1}_{2k}, \ldots, \mathbf{B}^{2i+1}\mathbf{1}_{2k}\} \subseteq \mathbb{R}^{2k}, \quad i = 0, 1, \ldots
    $$
    - 进一步, 观察到二者有迭代关系:
        $$
        \mathcal{R}_i = \mathbf{B}\mathcal{F}_i, \quad i = 0, 1, \ldots
        $$
        或反过来
        $$
        \mathcal{F}_i = \text{span}\{\mathbf{1}_{2k}\}  + \mathbf{B}\mathcal{R}_{i-1}, \implies \mathbf{B}\mathcal{R}_i \subseteq \mathcal{F}_{i+1}\quad i = 1, 2, \ldots 
        $$
        - 回顾 $\text{span}\{\mathbf{1}_{2k}\}$ 就是一维直线. 两个子空间 $\mathbf{U}$ 和 $\mathbf{V}$ 的和定义为 $\mathbf{U} + \mathbf{V} := \{\mathbf{u} + \mathbf{v}: \mathbf{u} \in \mathbf{U}, \mathbf{v} \in \mathbf{V}\}$.
        

- 若进一步代入 $\mathbf{B}$ 的具体结构, 则可以单纯通过计算得到:
    $$
    \mathcal{F}_i = \text{span}\{\mathbf{1}_{2k}, \mathbf{e}_{2k}^{(1)}, \mathbf{e}_{2k}^{(2)}, \ldots, \mathbf{e}_{2k}^{(i)}\} \subseteq \mathbb{R}^{2k}, \quad i = 0, 1, \ldots
    $$
    以及
    $$
    \mathcal{R}_i = \text{span}\{\mathbf{e}_{2k}^{(2k-i)}, \mathbf{e}_{2k}^{(2k-i+1)}, \ldots, \mathbf{e}_{2k}^{(2k)}\} \subseteq \mathbb{R}^{2k}, \quad i = 0, 1, \ldots
    $$
    - 其中 $\mathbf{e}_{2k}^{(j)}$ 是 $\mathbb{R}^{2k}$ 中的标准基向量, 其第 $j$ 个元素为 1, 其他元素为 0.

由约简关系 $\mathcal{J}_i = \mathcal{F}_i \times \mathbf{0}_{m-2k}$ 和 $\mathcal{K}_i = \mathcal{R}_i \times \mathbf{0}_{n-2k}$, 则可以得到 $\mathcal{J}_i$ 和 $\mathcal{K}_i$ 的具体结构为:
$$
\mathcal{J}_i = \text{span}\{\mathbf{c}, \mathbf{e}^{(1)}_{m}, \mathbf{e}^{(2)}_{m}, \ldots, \mathbf{e}^{(i)}_{m}\} \subseteq \mathbb{R}^{m}, \quad i = 0, 1, \ldots
$$
以及
$$
\mathcal{K}_i = \text{span}\{\mathbf{e}^{(2k-i)}_{n}, \mathbf{e}^{(2k-i+1)}_{n}, \ldots, \mathbf{e}^{(2k)}_{n}\} \subseteq \mathbb{R}^{n}, \quad i = 0, 1, \ldots
$$
- 有转换规则
    $$
    \boldsymbol{\Lambda} \mathcal{K}_i \subseteq \mathcal{J}_{i+1}, \quad i = 0, 1, \ldots
    $$
    以及
    $$
    \boldsymbol{\Lambda}^\top \mathcal{J}_i = \mathcal{K}_i, \quad i = 0, 1, \ldots
    $$

- 以及如下严格递增性质
    $$
    \mathcal{K}_{i-1} \subsetneq \mathcal{K}_i, \quad \mathcal{J}_{i-1} \subsetneq \mathcal{J}_i, \quad i = 1, 2, \ldots
    $$



### 2.3 A lower complexity bound with positive $L_A$

这一小节的目标如下. 若能够说明, 对于任意的 deterministic first-order method, 其迭代点 $\mathbf{x}^{(t)}$ 都落在 $\mathcal{K}_{t-1}$ 中, 则可以通过估计
$$
\min_{\mathbf{x} \in \mathcal{K}_{t-1}} |f(\mathbf{x}) - f^\star| \quad {\small\text{and}} \quad \min_{\mathbf{x} \in \mathcal{K}_{t-1}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|
$$
分别给出收敛率下界. 


#### Iteration Restriction

下面给出迭代点 $\mathbf{x}^{(t)}$ 落在 $\mathcal{K}_{t-1}$ 中的详细说明. 

在上述问题形式约束下, 回顾其梯度为 $\nabla f(\mathbf{x}) = \mathbf{H}\mathbf{x} - \mathbf{h}$. 此时只要同时满足
1. $h \in \mathcal{K}_0 = \text{span}\{\mathbf{e}^{(2k)}_{n}\}$,
   - 这要求 $\mathbf{h} \in \mathbb{R}^n$ 中只有第 $2k$ 个元素 (active 区块中的最后一个元素) 是非零的, 其他元素都是零. 
   - 并且再次强调, 这里的证明只需要找到一个 hard instance, 因此只要存在一个 (尽管特殊) 的 $\mathbf{h}$ 能够造成足够的 '麻烦' 即可. 
2. $\mathbf{H}\mathcal{K}_{t-1} \subseteq \mathcal{K}_t, \quad t = 1, 2, \ldots k$,
   - 由于 $\mathcal{K}_t$ 是决策变量空间 $\mathbb{R}^n$ 上的 Krylov subspace, 并且其在取值上表示由最后 $t$ 个元素位置的基向量构成的空间, 因此这个假设的意思是, 当 $\mathbf{H}$ 作用在 $\mathcal{K}_{t-1}$ 上时, 每次只会递进拓展一个相邻位置元素的取值. 

则可以通过 induction 说明, 对于任意的 deterministic first-order method, 其迭代点 
$$
\mathbf{x}^{(t)} \in \mathcal{K}_{t-1} = \text{span}\{\mathbf{e}^{(2k-t+1)}_{n}, \mathbf{e}^{(2k-t+2)}_{n}, \ldots, \mathbf{e}^{(2k)}_{n}\}
, \quad t = 1, 2, \ldots, k.
$$

- 其可以理解为, 对于任意一阶迭代算法, 在第 $t$ 次迭代时, 只有后 $t$ 个元素是非零的, 前面的元素都为 0. 例如, 当 $t=k$ 时, $\mathbf{x}^{(k)} = \text{span}\{\mathbf{e}^{(k+1)}, \ldots, \mathbf{e}^{(2k)}\}$.

*Proof*. 根据 Linear Span Assumption, 已假设 $\mathbf{x}^{(t)} \in \text{span}\{\nabla f(\mathbf{x}^{(0)}), \mathbf{A}^\top \mathbf{r}^{(0)}, \ldots, \nabla f(\mathbf{x}^{(t-1)}), \mathbf{A}^\top \mathbf{r}^{(t-1)}\}$, 因此原命题只需证明对于任意 $t = 1, 2, \ldots, k$, 
$$
\text{span}\{\nabla f(\mathbf{x}^{(0)}), \mathbf{A}^\top \mathbf{r}^{(0)}, \ldots, \nabla f(\mathbf{x}^{(t-1)}), \mathbf{A}^\top \mathbf{r}^{(t-1)}\} \subseteq \mathcal{K}_{t-1}.
$$

总体思路如下. 
- 根据归纳法, 当 $t=1$ 时, 要证 $\nabla f(\mathbf{x}^{(0)}) \in \mathcal{K}_0$ 且 $\mathbf{A}^\top \mathbf{r}^{(0)} \in \mathcal{K}_0$. 
  - 由于 $\mathbf{x}^{(0)} = \mathbf{0}$, 故 
    $$
    \begin{aligned}
    \mathbf{A}^\top \mathbf{r}^{(0)} &= \mathbf{A}^\top (\mathbf{A}\mathbf{x}^{(0)} - \mathbf{b}) = -\mathbf{A}^\top \mathbf{b} \\
    &= -\frac{L_A^2}{4} \boldsymbol{\Lambda}^\top \mathbf{c} \quad {\small{\text{by definition of } \mathbf{A} \text{ and } \mathbf{b}}} \\
    \\&= -\frac{L_A^2}{4} \mathbf{e}^{(2k)}_{n} \in \mathcal{K}_0 \quad {\small{\text{by calculation of } \boldsymbol{\Lambda}^\top \mathbf{c}}}.
    \end{aligned}
    $$
  - 由于 $\nabla f(\mathbf{x}^{(0)}) = \mathbf{H}\mathbf{x}^{(0)} - \mathbf{h} = -\mathbf{h}$, 且假设 $\mathbf{h} \in \mathcal{K}_0$, 故 $\nabla f(\mathbf{x}^{(0)}) \in \mathcal{K}_0$. 因此当 $t=1$ 时, 命题成立.

- 假设 $t = s$ 时问题仍然成立, 即 $\mathbf{x}^{(s)} \in \mathcal{K}_{s-1}$. 要证 当 $t = s+1$ 时, $\mathbf{x}^{(s+1)} \in \mathcal{K}_{s}$. 
  - 考虑 $\nabla f(\mathbf{x}^{(s)}) = \mathbf{H}\mathbf{x}^{(s)} - \mathbf{h}$. 由于 $\mathbf{x}^{(s)} \in \mathcal{K}_{s-1}$, 且假设 $\mathbf{H}\mathcal{K}_{s-1} \subseteq \mathcal{K}_s$, 故 $\mathbf{H}\mathbf{x}^{(s)} \in \mathcal{K}_s$. 另一方面, $\mathbf{h} \in \mathcal{K}_0 \subseteq \mathcal{K}_s$, 故 $\nabla f(\mathbf{x}^{(s)})= \mathbf{H}\mathbf{x}^{(s)} - \mathbf{h} \in \mathcal{K}_s$.
  - 考虑 $\mathbf{A}^\top \mathbf{r}^{(s)} = \mathbf{A}^\top \mathbf{A} \mathbf{x}^{(s)} - \mathbf{A}^\top \mathbf{b}$. 由于 $\mathbf{x}^{(s)} \in \mathcal{K}_{s-1}$, 且前文已说明有 Krylov 结构关系:
    $$
    \boldsymbol{\Lambda} \mathcal{K}_i \subseteq \mathcal{J}_{i+1}, \quad \boldsymbol{\Lambda}^\top \mathcal{J}_i = \mathcal{K}_i, \quad i = 0, 1, \ldots
    $$
    因此 $\mathbf{A}^\top \mathbf{A} \mathbf{x}^{(s)} \in \mathcal{K}_s$. 另一方面, $\mathbf{A}^\top \mathbf{b}$ 在 $t=1$ 时已证 $\mathbf{A}^\top \mathbf{b} \in \mathcal{K}_0 \subseteq \mathcal{K}_s$. 因此 $\mathbf{A}^\top \mathbf{r}^{(s)} = \mathbf{A}^\top \mathbf{A} \mathbf{x}^{(s)} - \mathbf{A}^\top \mathbf{b} \in \mathcal{K}_s$.

故根据归纳法, 原命题得证. 

$\square$

#### Hard Instance Construction

在确立了所有必要组件后, 给出 hard instance 的具体构造. 考虑如下问题:
$$
\begin{aligned}

\min_{\mathbf{x} \in \mathbb{R}^n} \left\{
    f(\mathbf{x}) := L_f \left(
        \frac{1}{2} x_k^2 + \frac{1}{2} \sum_{i=2k+1}^{n} x_i^2
    \right)
\right\}
\quad \text{s.t.} \quad \mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned}
$$

可以验证, 该 instance 符合全部假设. 故有 
- $\mathbf{x}^{(t)} \in \mathcal{K}_{t-1}$, $t = 1, 2, \ldots, k$. 即, 第 $t$ 步的迭代点只在后 $t$ 个元素上非零. 然而注意这里的目标函数 $f$ 只依赖于 $x_k$ 这一个坐标以及 $x_{2k+1}, \ldots, x_n$, 完全不触及 $x_{k+1}, \ldots, x_{2k}$ 这些坐标. 
- 因此, 在前 $k$ 步迭代中, $\nabla f(\mathbf{x}) = 0$, $f(\mathbf{x}) = 0$, $\mathbf{A}\mathbf{x} - \mathbf{b} \neq 0$. 任何 deterministic first-order method 都无法获得任何关于目标函数的下降信息, 也无法满足约束条件.