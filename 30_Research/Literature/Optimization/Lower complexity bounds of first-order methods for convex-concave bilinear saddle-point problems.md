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

在本文的 deterministic first-order method 分析中涉及到如下几个概念和问题.

- Deterministic 表明算法不涉及随机性, 给定相同的初始点和参数, 算法每次迭代都会产生相同的结果.
- First-order method 可以理解为黑箱 Oracle 能够返回的信息仅限于函数值和梯度信息. 对于本文的 SPP, 其 Oracle 可以返回如下信息:
  $$
  \mathrm{O}(\mathbf{x}, \mathbf{y}) := (\nabla f(\mathbf{x}), \mathbf{A}\mathbf{x}, \mathbf{A}^\top \mathbf{y}).
  $$
  - 对 SPP 目标函数求关于 $\mathbf{x}$ 的梯度, 其结果为 $\nabla f(\mathbf{x}) + \mathbf{A}^\top \mathbf{y}$; 求关于 $\mathbf{y}$ 的梯度, 其结果为 $\mathbf{A}\mathbf{x} - \mathbf{b} - \nabla g(\mathbf{y})$. 当然其他的一些信息如 $\mathbf{b}$ 是已知的; 和 $g$ 有关的信息, 文中认为其是简单的, 因此同样不作为 Oracle 的返回信息.
  - 具体的黑箱计算过程如下:
    - 算法的迭代初始点为 $(\mathbf{x}^{(0)}, \mathbf{y}^{(0)}) \in \mathcal{X} \times \mathcal{Y}$, 迭代次数为 $t = 0, 1, 2, \ldots$.
    - 在第 $t$ 次迭代中, 算法会在当前迭代点的 inquiry point $(\mathbf{x}^{(t)}, \mathbf{y}^{(t)})$ 处调用 Oracle, 得到
      $$
      \mathrm{O}(\mathbf{x}^{(t)}, \mathbf{y}^{(t)}) = (\nabla f(\mathbf{x}^{(t)}), \mathbf{A}\mathbf{x}^{(t)}, \mathbf{A}^\top \mathbf{y}^{(t)}).
      $$
    - 然后算法根据当前迭代点和 Oracle 返回的信息, 计算出下一次迭代的 inquiry point $(\mathbf{x}^{(t+1)}, \mathbf{y}^{(t+1)})$ 和最终用来返回作为输出的点 $(\bar{\mathbf{x}}^{(t+1)}, \bar{\mathbf{y}}^{(t+1)})$. 迭代过程可以表示为:
      $$
      (\mathbf{x}^{(t+1)}, \mathbf{y}^{(t+1)}, \bar{\mathbf{x}}^{(t+1)}, \bar{\mathbf{y}}^{(t+1)}) = \mathcal{I}_t\left( \boldsymbol{\vartheta}; \mathrm{O}(\mathbf{x}^{(0)}, \mathbf{y}^{(0)}), \ldots, \mathrm{O}(\mathbf{x}^{(t)}, \mathbf{y}^{(t)})\right), \quad \text{(1)}
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
\mathbf{x}^{(t+1)} &= \text{Proj}_{\mathcal{X}}\left(\mathbf{x}^{(t)} - \frac{1}{\eta} \left(\nabla f(\mathbf{x}^{(t)}) + \mathbf{A}^\top (\boldsymbol{\lambda}^{(t)} + \mathbf{r}^{(t)})\right)\right), \\
\boldsymbol{\lambda}^{(t+1)} &= \boldsymbol{\lambda}^{(t)} + \mathbf{r}^{(t+1)}, \\
\end{aligned}
$$
其中 $\mathbf{r}^{(t)} = \mathbf{A}\mathbf{x}^{(t)} - \mathbf{b}$ 是当前的残差, $\eta>0$ 是步长参数.

- 回顾, 在正常的 ALM 中, $\mathbf{x}^{(t+1)}$ 需要通过最小化二次的 augmented Lagrangian function $L_\rho(\mathbf{x}, \boldsymbol{\lambda}^{(t)}) := f(\mathbf{x}) + \langle \boldsymbol{\lambda}^{(t)}, \mathbf{A}\mathbf{x} - \mathbf{b} \rangle + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ 来得到, 然而其求解往往是很昂贵或困难的.
- 故 LALM 只会对 $L_\rho$ 进行一步梯度下降, 也就是 linearized, 然后再投影回 feasible set $\mathcal{X}$. 此外论文中取 $\rho = 1$ 作为 penalty parameter, 因此在这里就不再显式地写出 $\rho$.

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
  - **这里同时还指出, 这个 linear span 的形式是由 $\mathcal{X} = \mathbb{R}^n$ 决定的, 并且会给后续的分析提供便利. 若是更一般的有约束情景, 则确实仍需要进行投影操作, 则此时 linear span 不再成立, 需要引入新的表示方法.**
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

> ***Assumption 1* (Linear Span Assumption)** 
>
> 迭代序列 $\{\mathbf{x}^{(t)}\}_{t=0}^{\infty}$ 满足 $\mathbf{x}^{(0)} = \mathbf{0}$, 且对于任意 $t \geq 1$, 有
> $$
> \mathbf{x}^{(t)} \in \text{span}\{\nabla f(\mathbf{x}^{(0)}), \mathbf{A}^\top \mathbf{r}^{(0)}, \ldots, \nabla f(\mathbf{x}^{(t-1)}), \mathbf{A}^\top \mathbf{r}^{(t-1)}\},
> $$
> 其中 $\mathbf{r}^{(j)} = \mathbf{A}\mathbf{x}^{(j)} - \mathbf{b}$ 是第 $j$ 次迭代的残差.

- 上文的 LALM 算法在无约束 $\mathcal{X} = \mathbb{R}^n$ 下就是一个满足该假设的例子.
- $\mathbf{x}^{(0)} = \mathbf{0}$ 是不失一般性的, 因为总可以通过平移将任意的初始点 $\mathbf{x}^{(0)}$ 转换为 $\mathbf{0}$, 并且相应地调整 $\mathbf{h}$ 和 $\mathbf{b}$.
- 不过也需要指出, 如果算法中包含 Projection 并且要作用到一个有约束的 $\mathcal{X}$ 上, 则由于引入非线性映射, 该假设就不再成立. 这将在下一个 section 通过引入新的技术 (旋转不变性) 来解决.

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
    是一个反对角双带状 (Hankel 型) 的矩阵结构, 是整个构造的 '麻烦来源'. 由于 $B_{ij}$ 仅依赖于 $i+j$ (当 $i+j=2k$ 时为 $-1$, 当 $i+j=2k+1$ 时为 $1$), 故 **$\mathbf{B}$ 是对称矩阵**: $\mathbf{B}^\top = \mathbf{B}$. 其作用在任意向量 $\mathbf{u} = (u_1, \ldots, u_{2k})^\top$ 上, 其结果为
    $$
    \mathbf{B}\mathbf{u} = (u_{2k} - u_{2k-1}, \ldots, u_2 - u_1, u_1)^\top.
    $$
    注意到这是一种相邻元素差分的形式 (且顺序被翻转), 故 $\mathbf{B}$ 相当于离散的差分算子 (类似求导).
  - 这个分量顺序至关重要: 验证 $\mathbf{B}\mathbf{1}_{2k} = (0,\ldots,0,1)^\top = \mathbf{e}_{2k}^{(2k)}$, 这正是后面 $\mathcal{K}_0 = \text{span}\{\mathbf{e}^{(2k)}_{n}\}$ 的来源, 也解释了为什么后面 $\mathcal{F}_i$ 从左边 ($\mathbf{e}^{(1)},\ldots,\mathbf{e}^{(i)}$) 增长, 而 $\mathcal{K}_i$ 从右边 ($\mathbf{e}^{(2k-i)},\ldots,\mathbf{e}^{(2k)}$) 增长.
- 根据 $\mathbf{B}$ 的结构, 可以立刻得到如下性质:
  - $\|\mathbf{B}\| \leq 2$.
    - *Proof*. 由 $(a-b)^2 \leq 2a^2 + 2b^2$ 得
      $$
      \|\mathbf{B} \mathbf{u}\|^2 = u_1^2 + \sum_{i=1}^{2k-1} (u_{i+1} - u_i)^2 \leq u_1^2 + \sum_{i=1}^{2k-1} 2(u_{i+1}^2 + u_i^2) \leq 4\sum_{i=1}^{2k} u_i^2 = 4\|\mathbf{u}\|^2,
      $$
      因此 $\|\mathbf{B}\| \leq 2$. $\square$
  - $\|\boldsymbol{\Lambda}\| = \max\{\|\mathbf{B}\|, \|\mathbf{G}\|\} = 2$.
    - *Proof Sketch*. 这可以根据 $\mathbf{B}$ 和 $\mathbf{G}$ 的分块对角结构直接得到.
  - $\mathbf{B}$ 可逆 (非奇异), 且 $\mathbf{B}^{-1}$ 是**反三角**全 1 矩阵:
    $$
    \mathbf{B}^{-1} =
    \begin{bmatrix}
     &  && 1\\
     &  & 1 & 1\\
     &  & \vdots & \vdots \\
    1 & \cdots & 1 & 1
    \end{bmatrix} \in \mathbb{R}^{2k\times 2k},
    $$

- 对应的 $\mathbf{G} \in \mathbb{R}^{(m-2k)\times (n-2k)}$ 是任意 full row rank 矩阵, 且 $\|\mathbf{G}\| = 2$.  其用于进行提高问题维度, 不过对应的是 $\mathbf{c}$ 中的零块, 不会影响到后续的分析, 故具体形式并不重要. (full row rank 这一要求的真正用处在后面对偶解唯一性的推导中体现.)

在此基础上给出 hard instance 的约束部分.

> ***Lemma 1***
>
> 设 $\boldsymbol{\Lambda}, \mathbf{c}$ 如上, 给定 $L_A > 0$, 令
> $$
> \mathbf{A} = \frac{L_A}{2}\boldsymbol{\Lambda},\quad \mathbf{b} = \frac{L_A}{2}\mathbf{c}.
> $$
> 则 $\|\mathbf{A}\| = L_A$, 且任意满足 $\mathbf{A}\mathbf{x}^* = \mathbf{b}$ 的 $\mathbf{x}^*$ 均有
> $$
> x_i^* = i, \quad i = 1, \ldots, 2k.
> $$

- *Proof Sketch.* $\|\mathbf{A}\| = \frac{L_A}{2}\|\boldsymbol{\Lambda}\| = L_A$. 对 $\mathbf{x} = (\mathbf{u}^\top, \mathbf{v}^\top)^\top$ 分块, 约束的 $\mathbf{u}$-块化为 $\mathbf{B}\mathbf{u} = \mathbf{1}_{2k}$; 由 $\mathbf{B}$ 可逆, 唯一解为 $\mathbf{u}^* = \mathbf{B}^{-1}\mathbf{1}_{2k} = (1, 2, \ldots, 2k)^\top$ (反三角结构第 $i$ 行恰有 $i$ 个 1, 行和为 $i$). $\square$
- 注: 本小节要求 $L_A > 0$; 后续将讨论如何放宽到 $L_A \geq 0$, 从而能覆盖 proximal gradient ($L_A = 0$) 的情形.

### 2.2 Krylov Subspace

#### Krylov Subspace Introduction

首先介绍 Krylov subspace 的概念.

> ***Definition* (Krylov Subspace)**
>
> 给定矩阵 $\mathbf{M} \in \mathbb{R}^{n\times n}$ 和向量 $\mathbf{v} \in \mathbb{R}^n$, 其 Krylov subspace of order $j$ 定义为
> $$
> \mathcal{K}_j(\mathbf{M}, \mathbf{v}) := \text{span}\{\mathbf{v}, \mathbf{M}\mathbf{v}, \ldots, \mathbf{M}^{j-1}\mathbf{v}\}.
> $$

- 直观理解, Krylov subspace 就是反复对 $\mathbf{v}$ 进行矩阵 $\mathbf{M}$ 的线性变换所生成的向量所能够张成的线性空间.
- 之所以需要引入 Krylov subspace, 是因为某种意义上, 任何的 deterministic first-order method 都相当于是在反复地进行 Matrix-vector multiplication. 因此算法的迭代点 $\mathbf{x}^{(t)}$ 都会落在某个 Krylov subspace 中. 这将为后续的分析提供便利.

#### Krylov Subspace in Current Hard Instance

具体落实在当前的 hard instance 上, 定义如下两个 Krylov subspace:
$$
\mathcal{J}_i := \text{span}\{\mathbf{c}, (\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top)\mathbf{c}, \ldots, (\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top)^i\mathbf{c}\} \subseteq \mathbb{R}^m, \quad i \geq 0,
$$
以及
$$
\mathcal{K}_i := \boldsymbol{\Lambda}^\top \mathcal{J}_i \subseteq \mathbb{R}^n, \quad i \geq 0.
$$
- 其中 $\mathcal{J}_i \subseteq \mathbb{R}^m$ 是在约束空间上的 Krylov subspace. 而 $\mathcal{K}_i \subseteq \mathbb{R}^n$ 是在决策变量空间上的 Krylov subspace, 其是通过 $\boldsymbol{\Lambda}^\top$ 将 $\mathcal{J}_i$ 映射到 $\mathbb{R}^n$ 上得到的.
- 从代数的角度看, 整个线性映射的关系如下. 给定一个决策 $\mathbf{x} \in \mathbb{R}^n$, 其通过 $\mathbf{A} = \frac{L_A}{2}\boldsymbol{\Lambda}$ 映射到约束空间 $\mathbb{R}^m$ 上, 得到 $\mathbf{A}\mathbf{x} \in \mathbb{R}^m$ 衡量其在各个约束上的表现. 然后通过 $\mathbf{A}^\top = \frac{L_A}{2}\boldsymbol{\Lambda}^\top$ 将约束空间的向量映射回决策变量空间 $\mathbb{R}^n$ 指导更新下一次的优化迭代. 这里 $\mathbf{A}^\top$ 是 $\mathbf{A}$ 的 adjoint operator, 其满足:
    $$
    \langle \mathbf{A}\mathbf{x}, \mathbf{r} \rangle_{\mathbb{R}^m} = \langle \mathbf{x}, \mathbf{A}^\top \mathbf{r} \rangle_{\mathbb{R}^n}, \quad \forall \mathbf{x} \in \mathbb{R}^n, \forall \mathbf{r} \in \mathbb{R}^m.
    $$

#### Reduced Krylov Subspace

注意到, 对于 $\mathcal{J}_i$ 和 $\mathcal{K}_i$, 或者说其对应的矩阵 $\boldsymbol{\Lambda}$ 和向量 $\mathbf{c}$, 其在结构上只有前 $2k$ 个元素是 active 的.  通过矩阵计算, 可以发现
$$
(\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top)^i \mathbf{c} =
\begin{bmatrix}
    \mathbf{B}^{2i} \mathbf{1}_{2k} \\
    \mathbf{0}_{m-2k}
\end{bmatrix},
\qquad
\boldsymbol{\Lambda}^\top (\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top)^i \mathbf{c} =
\begin{bmatrix}
    \mathbf{B}^{2i+1} \mathbf{1}_{2k} \\
    \mathbf{0}_{n-2k}
\end{bmatrix}, \quad i \geq 0.
$$
- 这里 $\mathbf{B}\mathbf{B}^\top = \mathbf{B}^2$ 是严格等式, 因为如 §2.1 所述 $\mathbf{B}$ 是对称的 (Hankel 型).

因此我们事实上只需要考虑 $\mathcal{J}_i$ 和 $\mathcal{K}_i$ 的前 $2k$ 个元素, 得到约简后的 Krylov subspace:
$$
\mathcal{F}_i := \text{span}\{\mathbf{1}_{2k}, \mathbf{B}^2\mathbf{1}_{2k}, \ldots, \mathbf{B}^{2i}\mathbf{1}_{2k}\} \subseteq \mathbb{R}^{2k},
\qquad
\mathcal{R}_i := \text{span}\{\mathbf{B}\mathbf{1}_{2k}, \mathbf{B}^3\mathbf{1}_{2k}, \ldots, \mathbf{B}^{2i+1}\mathbf{1}_{2k}\} \subseteq \mathbb{R}^{2k}.
$$
- 二者有迭代关系:
    $$
    \mathcal{R}_i = \mathbf{B}\mathcal{F}_i, \qquad
    \mathcal{F}_i = \text{span}\{\mathbf{1}_{2k}\} + \mathbf{B}\mathcal{R}_{i-1} \implies \mathbf{B}\mathcal{R}_i \subseteq \mathcal{F}_{i+1}, \quad i \geq 1.
    $$
  - 回顾 $\text{span}\{\mathbf{1}_{2k}\}$ 就是一维直线. 两个子空间 $\mathbf{U}$ 和 $\mathbf{V}$ 的和定义为 $\mathbf{U} + \mathbf{V} := \{\mathbf{u} + \mathbf{v}: \mathbf{u} \in \mathbf{U}, \mathbf{v} \in \mathbf{V}\}$.

> ***Lemma 2* (约简 Krylov 子空间的显式结构)**
>
> 对任意 $0 \leq i \leq 2k-1$, 有
> $$
> \mathcal{F}_i = \text{span}\{\mathbf{1}_{2k}, \mathbf{e}_{2k}^{(1)}, \mathbf{e}_{2k}^{(2)}, \ldots, \mathbf{e}_{2k}^{(i)}\},
> \qquad
> \mathcal{R}_i = \text{span}\{\mathbf{e}_{2k}^{(2k-i)}, \mathbf{e}_{2k}^{(2k-i+1)}, \ldots, \mathbf{e}_{2k}^{(2k)}\},
> $$
> 以及 $\mathbf{B}\mathcal{R}_i = \text{span}\{\mathbf{e}_{2k}^{(1)}, \ldots, \mathbf{e}_{2k}^{(i+1)}\} \subseteq \mathcal{F}_{i+1}$ (约定 $\mathbf{e}_{2k}^{(0)} = \mathbf{0}$).

- 其中 $\mathbf{e}_{2k}^{(j)}$ 是 $\mathbb{R}^{2k}$ 中的标准基向量, 其第 $j$ 个元素为 1, 其他元素为 0.
- *Proof Sketch.* 由 $\mathbf{B}\mathbf{1}_{2k} = \mathbf{e}^{(2k)}_{2k}$, $\mathbf{B}\mathbf{e}^{(2k)}_{2k} = \mathbf{e}^{(1)}_{2k}$, $\mathbf{B}\mathbf{e}^{(i)}_{2k} = \mathbf{e}^{(2k-i+1)}_{2k} - \mathbf{e}^{(2k-i)}_{2k}$ 逐层归纳可得. $\square$
- 注意 $i \leq 2k-1$ 的范围限制: 超过此范围后空间饱和 (占满 $\mathbb{R}^{2k}$), 严格递增性不再成立. 这也是需要维度足够大 ($k < m/2$) 的原因.

> ***Lemma 3* (原始 Krylov 子空间的显式结构)**
>
> 由约简关系 $\mathcal{J}_i = \mathcal{F}_i \times \{\mathbf{0}_{m-2k}\}$ 和 $\mathcal{K}_i = \mathcal{R}_i \times \{\mathbf{0}_{n-2k}\}$, 对任意 $0 \leq i \leq 2k-1$:
> $$
> \mathcal{J}_i = \text{span}\{\mathbf{c}, \mathbf{e}^{(1)}_{m}, \mathbf{e}^{(2)}_{m}, \ldots, \mathbf{e}^{(i)}_{m}\},
> \qquad
> \mathcal{K}_i = \text{span}\{\mathbf{e}^{(2k-i)}_{n}, \mathbf{e}^{(2k-i+1)}_{n}, \ldots, \mathbf{e}^{(2k)}_{n}\}.
> $$
> 且有转换规则与严格递增性质:
> $$
> \boldsymbol{\Lambda} \mathcal{K}_i \subseteq \mathcal{J}_{i+1}, \qquad \boldsymbol{\Lambda}^\top \mathcal{J}_i = \mathcal{K}_i,
> \qquad
> \mathcal{K}_{i-1} \subsetneq \mathcal{K}_i, \quad \mathcal{J}_{i-1} \subsetneq \mathcal{J}_i, \quad 1 \leq i \leq 2k-1.
> $$

### 2.3 A Lower Complexity Bound with Positive $L_A$

这一小节的目标如下. 若能够说明, 对于任意满足 Assumption 1 的 first-order method, 其迭代点 $\mathbf{x}^{(t)}$ 都落在 $\mathcal{K}_{t-1}$ 中, 则可以通过估计
$$
\min_{\mathbf{x} \in \mathcal{K}_{t-1}} |f(\mathbf{x}) - f^\star| \quad {\small\text{and}} \quad \min_{\mathbf{x} \in \mathcal{K}_{t-1}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|
$$
分别给出目标误差和可行性误差的收敛率下界.

#### Iteration Restriction

下面给出迭代点 $\mathbf{x}^{(t)}$ 落在 $\mathcal{K}_{t-1}$ 中的充分条件. 回顾问题 (Q) 的梯度为 $\nabla f(\mathbf{x}) = \mathbf{H}\mathbf{x} - \mathbf{h}$.

> ***Lemma 4* (迭代点封锁的充分条件)**
>
> 设 $\mathbf{A}, \mathbf{b}$ 如 Lemma 1. 若实例 (Q) 满足
> 1. $\mathbf{h} \in \mathcal{K}_0 = \text{span}\{\mathbf{e}^{(2k)}_{n}\}$;
> 2. $\mathbf{H}\mathcal{K}_{t-1} \subseteq \mathcal{K}_t, \quad t = 1, 2, \ldots, k$,
>
> 则在 Assumption 1 下, 迭代点满足
> $$
> \mathbf{x}^{(t)} \in \mathcal{K}_{t-1} = \text{span}\{\mathbf{e}^{(2k-t+1)}_{n}, \mathbf{e}^{(2k-t+2)}_{n}, \ldots, \mathbf{e}^{(2k)}_{n}\}, \quad t = 1, 2, \ldots, k.
> $$

关于两个条件的直观解读:
1. 条件 1 要求 $\mathbf{h} \in \mathbb{R}^n$ 中只有第 $2k$ 个元素 (active 区块中的最后一个元素) 是非零的, 其他元素都是零.
   - 再次强调, 这里的证明只需要找到一个 hard instance, 因此只要存在一个 (尽管特殊) 的 $\mathbf{h}$ 能够造成足够的 '麻烦' 即可.
2. 由于 $\mathcal{K}_t$ 在取值上表示由最后 $t+1$ 个 (active 区块内) 元素位置的基向量构成的空间, 条件 2 的意思是, 当 $\mathbf{H}$ 作用在 $\mathcal{K}_{t-1}$ 上时, 每次至多只会递进拓展一个相邻位置元素的取值.

结论的直观解读: 对于任意满足 Assumption 1 的一阶迭代算法, 在第 $t$ 次迭代时, 迭代点只有 (active 区块内的) 后 $t$ 个元素可以非零, 前面的元素都为 0. 例如, 当 $t=k$ 时, $\mathbf{x}^{(k)} \in \text{span}\{\mathbf{e}^{(k+1)}_n, \ldots, \mathbf{e}^{(2k)}_n\}$ — 前 $k$ 个坐标仍然全为零.

*Proof*. 根据 Assumption 1, 只需证明对于任意 $t = 1, 2, \ldots, k$,
$$
\mathcal{S}_t := \text{span}\{\nabla f(\mathbf{x}^{(0)}), \mathbf{A}^\top \mathbf{r}^{(0)}, \ldots, \nabla f(\mathbf{x}^{(t-1)}), \mathbf{A}^\top \mathbf{r}^{(t-1)}\} \subseteq \mathcal{K}_{t-1}.
$$
对 $t$ 进行归纳.

- **Base case** ($t=1$): 要证 $\nabla f(\mathbf{x}^{(0)}) \in \mathcal{K}_0$ 且 $\mathbf{A}^\top \mathbf{r}^{(0)} \in \mathcal{K}_0$.
  - 由于 $\mathbf{x}^{(0)} = \mathbf{0}$, 故
    $$
    \begin{aligned}
    \mathbf{A}^\top \mathbf{r}^{(0)} &= \mathbf{A}^\top (\mathbf{A}\mathbf{x}^{(0)} - \mathbf{b}) = -\mathbf{A}^\top \mathbf{b} \\
    &= -\frac{L_A^2}{4} \boldsymbol{\Lambda}^\top \mathbf{c} \quad {\small{\text{by definition of } \mathbf{A} \text{ and } \mathbf{b}}} \\
    &= -\frac{L_A^2}{4} \mathbf{e}^{(2k)}_{n} \in \mathcal{K}_0 \quad {\small{\text{by } \boldsymbol{\Lambda}^\top\mathbf{c} = (\mathbf{B}\mathbf{1}_{2k}; \mathbf{0}) = \mathbf{e}^{(2k)}_n}}.
    \end{aligned}
    $$
  - 由于 $\nabla f(\mathbf{x}^{(0)}) = \mathbf{H}\mathbf{x}^{(0)} - \mathbf{h} = -\mathbf{h}$, 且条件 1 给出 $\mathbf{h} \in \mathcal{K}_0$, 故 $\nabla f(\mathbf{x}^{(0)}) \in \mathcal{K}_0$. 因此 $t=1$ 时命题成立.

- **Induction step**: 假设 $\mathcal{S}_s \subseteq \mathcal{K}_{s-1}$ 对某个 $1 \leq s < k$ 成立 (从而由 Assumption 1 得 $\mathbf{x}^{(s)} \in \mathcal{K}_{s-1}$). 要证 $\mathcal{S}_{s+1} \subseteq \mathcal{K}_s$, 由 $\mathcal{S}_{s+1} = \mathcal{S}_s + \text{span}\{\nabla f(\mathbf{x}^{(s)}), \mathbf{A}^\top\mathbf{r}^{(s)}\}$ 且 $\mathcal{S}_s \subseteq \mathcal{K}_{s-1} \subseteq \mathcal{K}_s$, 只需证明新增的两项落在 $\mathcal{K}_s$ 中.
  - 考虑 $\nabla f(\mathbf{x}^{(s)}) = \mathbf{H}\mathbf{x}^{(s)} - \mathbf{h}$. 由于 $\mathbf{x}^{(s)} \in \mathcal{K}_{s-1}$, 且条件 2 给出 $\mathbf{H}\mathcal{K}_{s-1} \subseteq \mathcal{K}_s$, 故 $\mathbf{H}\mathbf{x}^{(s)} \in \mathcal{K}_s$. 另一方面, $\mathbf{h} \in \mathcal{K}_0 \subseteq \mathcal{K}_s$, 故 $\nabla f(\mathbf{x}^{(s)}) \in \mathcal{K}_s$.
  - 考虑 $\mathbf{A}^\top \mathbf{r}^{(s)} = \mathbf{A}^\top \mathbf{A} \mathbf{x}^{(s)} - \mathbf{A}^\top \mathbf{b}$. 由于 $\mathbf{x}^{(s)} \in \mathcal{K}_{s-1}$, 且 Lemma 3 给出转换规则
    $$
    \boldsymbol{\Lambda} \mathcal{K}_{s-1} \subseteq \mathcal{J}_{s}, \quad \boldsymbol{\Lambda}^\top \mathcal{J}_{s} = \mathcal{K}_{s},
    $$
    因此 $\mathbf{A}^\top \mathbf{A} \mathbf{x}^{(s)} \in \mathcal{K}_s$. 另一方面, base case 已证 $\mathbf{A}^\top \mathbf{b} \in \mathcal{K}_0 \subseteq \mathcal{K}_s$. 因此 $\mathbf{A}^\top \mathbf{r}^{(s)} \in \mathcal{K}_s$.

故根据归纳法, 原命题得证. 

$\square$

#### Hard Instance Construction

在确立了所有必要组件后, 给出 hard instance 的具体构造.

给定 $L_f, L_A > 0$, 考虑如下问题:
$$
\min_{\mathbf{x} \in \mathbb{R}^n} \left\{
     f(\mathbf{x}) := L_f \left(
         \frac{1}{2} x_k^2 + \frac{1}{2} \sum_{i=2k+1}^{n} x_i^2
     \right)
 \right\}
 \quad \text{s.t.} \quad \mathbf{A} \mathbf{x} = \mathbf{b},
 $$
其中 $\mathbf{A}, \mathbf{b}$ 如 Lemma 1.

可以验证, 该 instance 是 (Q) 的特例 ($\mathbf{h} = \mathbf{0}$, $\mathbf{H}$ 为对角阵且 $\|\mathbf{H}\| = L_f$, 故 $\nabla f$ 是 $L_f$-Lipschitz 的), 且符合 Lemma 4 的两个条件 ($\mathbf{h} = \mathbf{0} \in \mathcal{K}_0$; $\mathbf{H}$ 对角故 $\mathbf{H}\mathcal{K}_{t-1} \subseteq \mathcal{K}_{t-1} \subseteq \mathcal{K}_t$).

> ***Lemma 5* (信息真空)**
>
> 对上述 instance 应用任意满足 Assumption 1 的一阶方法, 有 $\mathbf{x}^{(t)} \in \mathcal{K}_{t-1}$, $t = 1, \ldots, k$. 此外,
> $$
> f(\mathbf{x}) = 0 \quad \text{且} \quad \nabla f(\mathbf{x}) = \mathbf{0}, \qquad \forall\, \mathbf{x} \in \mathcal{K}_{k-1}.
> $$

- 关键观察: 目标函数 $f$ 只依赖于 $x_k$ 这一个坐标以及 $x_{2k+1}, \ldots, x_n$, 完全不触及 $x_{k+1}, \ldots, x_{2k}$ 这些坐标. 而 $\mathcal{K}_{k-1} = \text{span}\{\mathbf{e}^{(k+1)}_n, \ldots, \mathbf{e}^{(2k)}_n\}$ 中的点恰好只在 $x_{k+1}, \ldots, x_{2k}$ 上非零 — 与 $f$ 的 active 坐标完全错开.
- 因此, 在前 $k$ 步迭代中, oracle 返回的目标函数信息恒为零 ($f(\mathbf{x}^{(t)}) = 0$, $\nabla f(\mathbf{x}^{(t)}) = \mathbf{0}$), 而 $\mathbf{A}\mathbf{x}^{(t)} - \mathbf{b} \neq \mathbf{0}$. 任何满足假设的一阶方法都无法获得任何关于目标函数的下降信息, 也无法满足约束条件.

> ***Lemma 6* (Primal-Dual Solution)**
>
> 设 $L_f, L_A > 0$. 上述 instance 有唯一最优解 $\mathbf{x}^\star$ 与唯一对应的 Lagrange multiplier $\mathbf{y}^\star$:
> $$
> \mathbf{x}^\star = (1, 2, \ldots, 2k, 0, \ldots, 0)^\top, \qquad
> \mathbf{y}^\star = \frac{2 k L_f}{L_A} \begin{bmatrix} \mathbf{0}_{k} \\ \mathbf{1}_{k} \\ \mathbf{0}_{m-2k} \end{bmatrix},
> $$
> 且最优目标值为 $f^\star = \dfrac{L_f}{2} k^2$.

*Proof*. 分三步: 解耦、求原始解、求对偶解.

**(i) 解耦.** 记 $\mathbf{x} := (\mathbf{u}^\top, \mathbf{v}^\top)^\top \in \mathbb{R}^{2k + (n-2k)}$. 由 $\mathbf{A}, \mathbf{b}$ 的分块结构与目标函数的坐标分离性 ($x_k$ 属于 $\mathbf{u}$-块, $\sum_{i>2k}x_i^2 = \|\mathbf{v}\|^2$ 属于 $\mathbf{v}$-块), 原问题解耦为两个独立子问题:
$$
\begin{aligned}
&\min_{\mathbf{u}} \frac{L_f}{2} u_k^2 \quad \text{s.t. } \frac{L_A}{2} \mathbf{B}\mathbf{u} = \frac{L_A}{2}\mathbf{1}_{2k}, \qquad (\text{a}) \\
&\min_{\mathbf{v}} \frac{L_f}{2} \|\mathbf{v}\|^2 \quad \text{s.t. } \frac{L_A}{2} \mathbf{G}\mathbf{v} = \mathbf{0}. \qquad (\text{b})
\end{aligned}
$$

**(ii) 原始解.**
- 对于问题 (a), 由 Lemma 1, 约束的可行集是单点集 $\{\mathbf{u}^\star\}$, $\mathbf{u}^\star = (1, 2, \ldots, 2k)^\top$, 故唯一可行点自动就是最优点 — **目标函数在问题 (a) 中没有起到任何筛选作用**.
  - *这正是 hard instance 的设计哲学: 让可行集是单点集, 使得难度全部来自 '凑齐约束' 而非 '下降'; 同时 (由 Lemma 5) 目标函数的任何信息在前 $k$ 步内均被隐藏.*
  - 代入 $u_k^\star = k$ (第 $k$ 个分量, 不是第 $2k$ 个!), 得子问题最优值 $\frac{L_f}{2} k^2$.
- 对于问题 (b), 显然 $\mathbf{v}^\star = \mathbf{0}$ 是唯一最优解, 最优值为 $0$.
- 综上,
    $$
    \boxed{\mathbf{x}^\star = (1, 2, \ldots, 2k, 0, \ldots, 0)^\top}, \qquad f^\star = \frac{L_f}{2} k^2.
    $$

**(iii) 对偶解.** 引入 Lagrange multiplier $\mathbf{y} \in \mathbb{R}^m$, 取 Lagrangian 为
$$
\mathcal{L}(\mathbf{x}, \mathbf{y}) = f(\mathbf{x}) + \langle \mathbf{y},  \mathbf{b} - \mathbf{A}\mathbf{x} \rangle,
$$
其 KKT 条件为
$$
\nabla f(\mathbf{x}^\star) - \mathbf{A}^\top \mathbf{y}^\star = \mathbf{0}, \qquad \mathbf{A}\mathbf{x}^\star - \mathbf{b} = \mathbf{0}.
$$
- *符号约定注*: 若 Lagrangian 取 $f + \langle \mathbf{y}, \mathbf{A}\mathbf{x} - \mathbf{b} \rangle$, 则 $\mathbf{y}^\star$ 整体反号; 不影响 $\|\mathbf{y}^\star\|$ 及后续所有下界结论.

- **左边**: 根据 $f(\mathbf{x}) = \frac{L_f}{2} (x_k^2 + \sum_{i=2k+1}^{n} x_i^2)$, 逐坐标求偏导:
    $$
    \frac{\partial f}{\partial x_i}= \begin{cases}
        L_f x_k & \text{if } i = k, \\
        L_f x_i & \text{if } i \in \{2k+1, \ldots, n\}, \\
        0 & \text{otherwise}.
    \end{cases}
    $$
    代入 $\mathbf{x}^\star$ ($x_k^\star = k$; $x_i^\star = 0$ 对 $i > 2k$), 得
    $$
    \nabla f(\mathbf{x}^\star) = L_f k\, \mathbf{e}^{(k)}_{n}.
    $$
- **右边**: 对 $\mathbf{y} = (\boldsymbol{\lambda}^\top, \boldsymbol{\pi}^\top)^\top \in \mathbb{R}^{2k + (m-2k)}$ 分块, 则
    $$
    \mathbf{A}^\top \mathbf{y}^\star = \frac{L_A}{2} \boldsymbol{\Lambda}^\top \mathbf{y}^\star = \frac{L_A}{2}
    \begin{bmatrix}
    \mathbf{B}^\top \boldsymbol{\lambda}^\star \\
    \mathbf{G}^\top \boldsymbol{\pi}^\star
    \end{bmatrix}.
    $$
    逐块与左边对应, 得
    $$
    \frac{L_A}{2} \mathbf{B}^\top \boldsymbol{\lambda}^\star = L_f k\, \mathbf{e}^{(k)}_{2k}, \qquad
    \frac{L_A}{2} \mathbf{G}^\top \boldsymbol{\pi}^\star = \mathbf{0}.
    $$
- **解 $\boldsymbol{\pi}^\star$**: $\mathbf{G}$ 是 full row rank $\implies \mathbf{G}^\top$ 列满秩 (单射) $\implies \mathbf{G}^\top\boldsymbol{\pi} = \mathbf{0}$ 只有零解, 故 $\boldsymbol{\pi}^\star = \mathbf{0}$ 且唯一. (这正是当初要求 $\mathbf{G}$ full row rank 的用处 — 保证对偶解唯一.)
- **解 $\boldsymbol{\lambda}^\star$**: 由 $\mathbf{B}$ 对称, $(\mathbf{B}^\top)^{-1} = \mathbf{B}^{-1}$, 于是
    $$
    \boldsymbol{\lambda}^\star = \frac{2}{L_A} \mathbf{B}^{-1} \left(L_f k\, \mathbf{e}^{(k)}_{2k}\right) = \frac{2 L_f k}{L_A}\, \mathbf{B}^{-1}\mathbf{e}^{(k)}_{2k}.
    $$
    而 $\mathbf{B}^{-1}\mathbf{e}^{(k)}_{2k}$ 即取 $\mathbf{B}^{-1}$ 的第 $k$ 列: 由反三角结构 $(\mathbf{B}^{-1})_{ij}=1 \iff i+j\geq 2k+1$, 有 $i + k \geq 2k+1 \iff i \geq k+1$, 故
    $$
    \mathbf{B}^{-1}\mathbf{e}^{(k)}_{2k} = \begin{bmatrix} \mathbf{0}_{k} \\ \mathbf{1}_{k} \end{bmatrix}
    \quad\implies\quad
    \boldsymbol{\lambda}^\star = \frac{2 L_f k}{L_A} \begin{bmatrix} \mathbf{0}_{k} \\ \mathbf{1}_{k} \end{bmatrix}.
    $$
- 综上, 对偶最优解为
    $$
    \boxed{\mathbf{y}^\star = (\boldsymbol{\lambda}^{\star\top}, \boldsymbol{\pi}^{\star\top})^\top = \frac{2 L_f k}{L_A} \begin{bmatrix} \mathbf{0}_{k} \\ \mathbf{1}_{k} \\ \mathbf{0}_{m-2k} \end{bmatrix}}
    $$
$\square$

#### Lower Complexity Bound 

总结前文重要结论:
- 最优解 $\mathbf{x}^\star = (1, 2, \ldots, 2k, 0, \ldots, 0)^\top$, 故 $\|\mathbf{x}^\star\|^2 = \sum_{i=1}^{2k} i^2 = \frac{k(2k+1)(4k+1)}{3}$.
- 对偶最优解 $\mathbf{y}^\star = \frac{2 L_f k}{L_A} \begin{bmatrix} \mathbf{0}_{k} \\ \mathbf{1}_{k} \\ \mathbf{0}_{m-2k} \end{bmatrix}$, 故 $\|\mathbf{y}^\star\|^2= {4 L_f^2 k^3}/{L_A^2}$.
- 对于任意 $\mathbf{x} \in \mathcal{K}_{k-1}$, 有 $f(\mathbf{x}) = 0$ 且 $\nabla f(\mathbf{x}) = \mathbf{0}$. 
- 对应 Optimality gap, 有
    $$
    |f(\mathbf{x}) - f^\star| = |0 - f^\star| = \frac{L_f}{2} k^2.
    $$
- 对应 feasibility gap, 有
    $$
    \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 \geq \frac{L_A^2}{4} k. 
    $$
    - *Proof*. 
        $$
        \begin{aligned}
        \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 &= \left\|\frac{L_A}{2} \boldsymbol{\Lambda}\mathbf{x} - \frac{L_A}{2}\mathbf{c}\right\|^2 \quad {\small (\text{by definition of } \mathbf{A} \text{ and } \mathbf{b})} \\
        &= \frac{L_A^2}{4} \left\|\begin{bmatrix} \mathbf{B}_{2k} & \mathbf{O} \\ \mathbf{O} & \mathbf{G}     \end{bmatrix}\begin{bmatrix} \mathbf{0}_k \\ \mathbf{x}_{k+1:2k} \\ \mathbf{0}_{n-2k}     \end{bmatrix} - \begin{bmatrix} \mathbf{1}_{2k} \\ \mathbf{0}_{m-2k} \end{bmatrix}\right\|^2 \quad {\small (\text{since } \mathbf{x} \in \mathcal{K}_{k-1})} \\
        \\ 
        &=\frac{L_A^2}{4} \left\|\mathbf{B}_{2k}\begin{bmatrix} \mathbf{0}_k \\ \mathbf{x}_{k+1:2k} \end{bmatrix} - \mathbf{1}_{2k}\right\|^2
        \\
        &= \frac{L_A^2}{4} \left\| \underbrace{\sum_{j=1}^{k-1} \left(x_{2k-j+1} - x_{2k-j} - 1\right)^2 + (x_{k+1} - 1)^2}_{{\small\text{first } k \text{ row} \geq 0} } + \sum_{j=k+1}^{2k} (0 - 1)^2\right\|^2 \\
        & \geq \frac{L_A^2}{4} \sum_{j=k+1}^{2k} 1^2 = \frac{L_A^2}{4} k.
        \end{aligned}
        $$
    $\square$

根据上述命题, 可以直接给出如下 Optimality gap 和 Feasibility gap 的下界结论.

> ***Lemma 7* (Optimality gap 下界)**
>
> 对于 $L_A, L_f > 0$, 任意满足 Assumption 1 的一阶方法, 在前 $k$ 步迭代中, 有
> $$
> \min_{\mathbf{x} \in \mathcal{K}_{k-1}} |f(\mathbf{x}) - f^\star| = |f(\mathbf{0}) - f^\star| \geq \frac{3 L_f \|\mathbf{x}^\star\|^2}{32(k+1)} + \frac{\sqrt{6}}{32(k+1)} L_A \|\mathbf{x}^\star\| \cdot \|\mathbf{y}^\star\|,
> $$
> 以及
> $$
> \min_{\mathbf{x} \in \mathcal{K}_{k-1}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\| \geq \frac{\sqrt{3} L_A \|\mathbf{x}^\star\|}{4\sqrt{2}(k+1)}.
> $$

*Proof*. 

考虑 feasibility gap:
$$
\begin{aligned}
\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 &\geq \frac{L_A^2}{4} k 
\\
&= \frac{L_A^2}{4} \cdot \frac{3\|\mathbf{x}^\star\|^2}{(2k+1)(4k+1)} \quad {\small (\text{by } \|\mathbf{x}^\star\|^2 = {k(2k+1)(4k+1)}/{3})} \\
&\geq \frac{3 L_A^2 \|\mathbf{x}^\star\|^2}{8(k+1)^2} \quad {\small (\text{since } (2k+1)(4k+1) \leq 8(k+1)^2)} \\
\end{aligned}
$$

考虑 optimality gap:
- 一方面
    $$
    \begin{aligned}
        |f(\mathbf{x}) - f^\star|&= f^\star =  \frac{L_f}{2} k^2 \\
        &= \frac{3 L_f k \|\mathbf{x}^\star\|^2}{2(2k+1)(4k+1)} \\&
        \geq \frac{3 L_f \|\mathbf{x}^\star\|^2}{16(k+1)} \quad {\small (\text{since } \frac{k}{(2k+1)(4k+1)} \geq \frac{1}{16(k+1)})} \\
    \end{aligned}
    $$

- 另一方面
    $$
    \begin{aligned}
        |f(\mathbf{x}) - f^\star| &= \frac{L_f}{2} k^2 \\
        &= \frac12 \cdot k^{1/2} \cdot L_f k^{3/2} \\
        &= \frac12 \cdot  \left(
            \frac{\sqrt{3} \|\mathbf{x}^\star\|}{\sqrt{(2k+1)(4k+1)}}
        \right)
        \left(\frac{L_A}{2} \cdot \|\mathbf{y}^\star\| \right)  \quad {\small (\text{by } \|\mathbf{x}^\star\|^2 = {\frac{k(2k+1)(4k+1)}{3}}, \|\mathbf{y}^\star\|^2 = {\frac{4 L_f^2 k^3}{L_A^2}})} \\
        &= \frac{\sqrt{3}}{4} \cdot \frac{L_A \|\mathbf{x}^\star\| \cdot \|\mathbf{y}^\star\|}{\sqrt{(2k+1)(4k+1)}} \\
        &\geq \frac{\sqrt{6}}{16(k+1)} L_A \|\mathbf{x}^\star\| \cdot \|\mathbf{y}^\star\| \quad {\small (\text{since } \frac{\sqrt{3}}{4\sqrt{(2k+1)(4k+1)}} \geq \frac{\sqrt{6}}{16(k+1)})}.
    \end{aligned}
    $$


- 综上两项, 根据 $\max\{a,b\} \geq (a+b)/2$, 得
    $$
    f(\mathbf{x}) - f^\star \geq \frac{3 L_f \|\mathbf{x}^\star\|^2}{32(k+1)} + \frac{\sqrt{6}}{32(k+1)} L_A \|\mathbf{x}^\star\| \cdot \|\mathbf{y}^\star\|.
    $$


$\square$

将上述结论整理成算法语言, 即得到如下定理. 

> ***Theorem 3* (Lower Complexity Bound with Positive $L_A$)**
>
> 给定正系数 $L_f, L_A > 0$, 正整数维度 $m\leq n$. 对于任意迭代次数 $t < m/2$, 对于问题 $\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$ s.t. $\mathbf{A}\mathbf{x} = \mathbf{b}$, 其中 $f$ 是 $L_f$-Lipschitz 光滑的, $\|\mathbf{A}\| = L_A$, 都存在一种 instance 的构造, 使得任意满足 Assumption 1 的一阶方法在前 $t$ 步迭代中, 其迭代点 $\mathbf{x}^{(t)}$ 与 primal-dual 最优解 $(\mathbf{x}^\star, \mathbf{y}^\star)$ 满足关系:
>
> $$
> \begin{aligned}
> &|f(\mathbf{x}^{(t)}) - f(\mathbf{x}^\star)| \geq \frac{3 L_f \|\mathbf{x}^\star\|^2}{32(t+1)} + \frac{\sqrt{6}}{32(t+1)} L_A \|\mathbf{x}^\star\| \cdot \|\mathbf{y}^\star\|, \\
> &\|\mathbf{A}\mathbf{x}^{(t)} - \mathbf{b}\| \geq \frac{\sqrt{3} L_A \|\mathbf{x}^\star\|}{4\sqrt{2}(t+1)}.
> \end{aligned}
> $$
>

*Proof*. 由 Lemma 7, 取 $k = t < m/2$, 且由于对于任意迭代 $\mathbf{x}^{(t)} \in \mathcal{K}_{t-1}$, 故任意算法的表现不可能优于该空间内的 lower bound, 故证毕.

$\square$

对于该结论有如下说明:

1. 误差下界的收敛率为 $\Omega (1/t)$, 等价于对应达到规定误差 $\epsilon$ 所需的迭代次数为 $\Omega (1/\epsilon)$.
2. 注意到, 若使用 proximal gradient 算法, 能够达到的收敛率下界为 $\Omega(\sqrt{L_f/\epsilon})$, 快于这里的 $\Omega(L_f /\epsilon)$. 不过这并不矛盾, 因为 proximal gradient method 并不能归纳到当前的 oracle 类算法中. 
   - 根据 proximal gradient 算法, 当前约束问题等价于引入 $0-\infty$ 示性函数 $\delta_{\{\mathbf{A}\mathbf{x} = \mathbf{b}\}}(\mathbf{x})$, 进而将原问题转化为 $\min_{\mathbf{x}} f(\mathbf{x}) + \delta_{\{\mathbf{A}\mathbf{x} = \mathbf{b}\}}(\mathbf{x})$. 从而得到迭代:
        $$
        \mathbf{x}^{(t+1)} = \text{Proj}_{\{\mathbf{A}\mathbf{x} = \mathbf{b}\}}\left(\mathbf{x}^{(t)} - \frac{1}{L_f} \nabla f(\mathbf{x}^{(t)})\right),
        $$
        而向仿射集 $\{\mathbf{A}\mathbf{x} = \mathbf{b}\}$ 的投影在 $\mathbf{A}$ 满秩的前提下有 closed-form solution: $\text{Proj}_{\{\mathbf{A}\mathbf{x} = \mathbf{b}\}}(\mathbf{z}) = \mathbf{z} - \mathbf{A}^\top (\mathbf{A}\mathbf{A}^\top)^{-1} (\mathbf{A}\mathbf{z} - \mathbf{b})$. 该迭代由于需要 $(\mathbf{A}\mathbf{A}^\top)^{-1}$ 的矩阵求逆, 因此相当于 $\mathbf{A}$ 无穷次的级数求和, 故并不属于当前的 oracle 类算法. 并且这样的投影也是十分昂贵的. 

3. 之所以需要引入对偶变量 $\mathbf{y}$, 是因为在约束优化问题中, $f(\mathbf{x}) - f^\star \geq 0$ 只在可行域 $\{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{b}\}$ 内成立, 而在不可行域中, $f(\mathbf{x}) - f^\star$ 可能为负数. 但由于我们希望同时衡量 Optimality gap 和 feasiblity gap, 因此需要引入对偶变量 $\mathbf{y}$ 来将两者联系起来. 具体来说, 对于任意 $\mathbf{x} \in \mathbb{R}^n$, 设 $(\mathbf{x}^\star, \mathbf{y}^\star)$ 为 primal-dual 最优解, 则有
    $$
    f(\mathbf{x}) - f^\star \geq \langle \mathbf{y}^\star, \mathbf{A}\mathbf{x} - \mathbf{b} \rangle \geq -\|\mathbf{y}^\star\| \cdot \|\mathbf{A}\mathbf{x} - \mathbf{b}\|,
    $$


### 2.4 A Lower Complexity Bound with Nonnegative $L_A$

回顾在上一个小节, 优化的目标函数为:
$$
f(\mathbf{x}) = L_f \left(\frac{1}{2} x_k^2 + \frac{1}{2} \sum_{i=2k+1}^{n} x_i^2\right), \quad \text{s.t. } \mathbf{A}\mathbf{x} = \mathbf{b},
$$
其中 $\mathbf{A}$ 是 $m \times n$ 的矩阵, 且 $\|\mathbf{A}\| = L_A > 0$.  然而该问题不能取 $L_A = 0$ 的退化场景. 

在这个小节中, 仍然考虑如下优化问题:
$$
f^\star := \min_{\mathbf{x} \in \mathbb{R}^n} \left\{
     f(\mathbf{x}) := 
     \frac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x} - \mathbf{h}^\top \mathbf{x}
 \right\}
 \quad \text{s.t.} \quad \mathbf{A} \mathbf{x} = \mathbf{b},
$$
不过具体构造如下, 令
$$
\mathbf{H} = \frac{L_f}{4} \begin{bmatrix} 
\mathbf{B}^\top \mathbf{B} & \mathbf{O} \\ 
\mathbf{O} & \mathbf{I}_{n-2k}
\end{bmatrix} \in \mathbb{R}^{n \times n}, \quad
\mathbf{h} = \left(\frac{L_f}{4} + \frac{L_A}{4\sqrt{2}}\right) \mathbf{e}^{(2k)}_{n}, \\
\,\\
\mathbf{A} = \frac{L_A}{2} \boldsymbol{\Lambda} \in \mathbb{R}^{m \times n}, \quad
\mathbf{b} = \frac{L_A}{2} \mathbf{c} \in \mathbb{R}^{m}.
$$

根据构造, 可知 $\mathbf{H} \succeq \mathbf{O}$, 故 $f$ 是 convex quadratic function, 且是 $L_f$-Lipschitz 光滑的. 
- *Proof*. 光滑性: $\nabla f(\mathbf{x}) = \mathbf{H}\mathbf{x} - \mathbf{h}$, 且 $\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| = \|\mathbf{H}(\mathbf{x}-\mathbf{y})\| \leq \|\mathbf{H}\| \cdot \|\mathbf{x}-\mathbf{y}\|$. 由于 $\mathbf{H}$ 是 block diagonal, 故 $\|\mathbf{H}\| = \frac{L_f}{4} \max\{\|\mathbf{B}^\top \mathbf{B}\|, 1\} \leq \frac{L_f}{4} \cdot 4 = L_f$ (由 Lemma 1 可知 $\|\mathbf{B}\| \leq 2$). 因此 $\nabla f$ 是 $L_f$-Lipschitz 的. 

    $\square$


> ***Lemma 8***
>
> 根据上述方法构造的 instance, 对任意满足 Assumption 1 的一阶方法, 有 $\mathbf{x}^{(t)} \in \mathcal{K}_{t-1}= \text{span}\{\mathbf{e}^{(2k-t+1)}_n, \ldots, \mathbf{e}^{(2k)}_n\}$, 即 $\mathbf{x}^{(t)}$ 的前 $2k-t$ 个坐标恒为零. 其中 $t = 1, \ldots, k$.

*Proof*. 由前面的 Lemma 4, 证明该命题只需验证 $\mathbf{h} \in \mathcal{K}_0$ 且 $\mathbf{H}\mathcal{K}_{t-1} \subseteq \mathcal{K}_t$, $t = 1, \ldots, k$.
- 由于 $\mathbf{h} = \left(\frac{L_f}{4} + \frac{L_A}{4\sqrt{2}}\right) \mathbf{e}^{(2k)}_{n}$, 根据定义, $\mathcal{K}_0 = \text{span}\{\mathbf{e}^{(2k)}_n\}$, 故 $\mathbf{h} \in \mathcal{K}_0$.
- 由于 $\mathbf{H}$ 是 block diagonal, 对于 $\mathcal{K}_{t-1} = \text{span}\{\mathbf{e}^{(2k-t+1)}_n, \ldots, \mathbf{e}^{(2k)}_n\}$, 其非零的元素处在第 $2k-t+1, \ldots, 2k$ 个坐标上, 因此 $\mathbf{H}\mathcal{K}_{t-1}$  只会同 $\mathbf{H}$ 中核心的 block $\mathbf{B}^\top \mathbf{B} \in \mathbb{R}^{2 k\times 2k}$ 产生交互, 展开计算有:
    $$
    \mathbf{B}^\top \mathbf{B} = \begin{bmatrix}
        2 & -1 & 0 & \cdots & 0 \\
        -1 & 2 & -1 & \cdots & 0 \\
        0 & -1 & 2 & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & -1 \\
        0 & 0 & 0 & -1 & 1
    \end{bmatrix}.
    $$
    故对于中间区域的基向量 $\mathbf{e}^{(i)}_{2k}$, $1 < i < 2k$, 有 $\mathbf{B}^\top \mathbf{B} \mathbf{e}^{(i)}_{2k} = -\mathbf{e}^{(i-1)}_{2k} + 2\mathbf{e}^{(i)}_{2k} - \mathbf{e}^{(i+1)}_{2k}$, 而端点 $\mathbf{e}^{(2k)}_{2k}$ 有 $\mathbf{B}^\top \mathbf{B} \mathbf{e}^{(2k)}_{2k} = -\mathbf{e}^{(2k-1)}_{2k} + \mathbf{e}^{(2k)}_{2k}$. 因此核心洞察为: $\mathbf{B}^\top \mathbf{B}$ 作用在 $\mathbf{e}^{(i)}_{2k}$ 上, 只会涉及 $\mathbf{e}^{(i-1)}_{2k}, \mathbf{e}^{(i)}_{2k}, \mathbf{e}^{(i+1)}_{2k}$ 三个坐标, 因此 $\mathbf{H}\mathcal{K}_{t-1} \subseteq \mathcal{K}_t$.

$\square$

下求解最优值. 

> ***Lemma 9* (Primal-Dual Solution)**
>
> 给定 $L_f, L_A > 0$, 上述 instance 有唯一最优解 $\mathbf{x}^\star = (1, 2, \ldots, 2k, 0, \ldots, 0)^\top$ 与唯一对应的对偶解 $\mathbf{y}^\star$:
> $$
> \mathbf{y}^\star = - \frac{1}{2\sqrt{2}} \begin{bmatrix} \mathbf{1}_{2k} \\ \mathbf{0}_{m-2k} \end{bmatrix} \implies \|\mathbf{y}^\star\| = \frac{\sqrt{k}}{2},
> $$
> 对应的最优值
> $$
> f^\star = - \left(\frac{L_f}{4} + \frac{L_A}{2\sqrt{2}}\right) k.
> $$

*Proof*. 对于原问题, 同样对 $\mathbf{x}$ 按照前 $2k$ 个坐标与后 $n-2k$ 个坐标分块, 记 $\mathbf{x} = (\mathbf{u}^\top, \mathbf{v}^\top)^\top$, 则原问题解耦为两个独立子问题:
$$
\begin{aligned}
&\min_{\mathbf{u}} \frac{1}{2} \mathbf{u}^\top \mathbf{S} \mathbf{u} - \mathbf{s}^\top \mathbf{u} \quad &&\text{s.t. } \frac{L_A}{2} \mathbf{B}\mathbf{u} = \frac{L_A}{2}\mathbf{1}_{2k}, \qquad &&& (\text{a}) \\
&\min_{\mathbf{v}} \frac{L_f}{8} \|\mathbf{v}\|^2 \quad &&\text{s.t. } \frac{L_A}{2} \mathbf{G}\mathbf{v} = \mathbf{0}. \qquad &&& (\text{b})
\end{aligned}
$$
其中, 记 $\mathbf{S} = \frac{L_f}{4} \mathbf{B}^\top \mathbf{B}$, $\mathbf{s} = \left(\frac{L_f}{4} + \frac{L_A}{4\sqrt{2}}\right) \mathbf{e}^{(2k)}_{2k}$.

- 对于问题 $\text{(b)}$, 显然 $\mathbf{v}^\star = \mathbf{0}$ 是唯一最优解, 最优值为 $0$, 对于任意 $L_A \geq 0$ 都成立. 
- 对于问题 $\text{(a)}$, 
  - 若 $L_A > 0$, 此时约束为 $\mathbf{B}\mathbf{u} = \mathbf{1}_{2k}$, 由于 $\mathbf{B}$ 是 full rank, 故唯一可行点为 $\mathbf{u}^\star = (1, 2, \ldots, 2k)^\top$, 代入目标函数得最优值为
    $$
    f^\star = \frac{1}{2} (\mathbf{u}^\star)^\top \mathbf{S} \mathbf{u}^\star - \mathbf{s}^\top \mathbf{u}^\star = - \left(\frac{L_f}{4} + \frac{L_A}{2\sqrt{2}}\right) k.
    $$
  - 若 $L_A = 0$, 此时约束为 $\mathbf{0} = \mathbf{0}$, 故 $\mathbf{u}$ 无约束. 由 $\mathbf{B}$ non-singular, 可知 $\mathbf{S} = \frac{L_f}{4} \mathbf{B}^\top \mathbf{B}$ 是正定的, 目标函数此时 strictly convex. 且最优点须满足 $\nabla_{\mathbf{u}} f(\mathbf{u}) = \mathbf{S}\mathbf{u} - \mathbf{s} = \mathbf{0}$. 只需检查 $\mathbf{u}^\star = (1, 2, \ldots, 2k)^\top$ 是否满足该条件:
    $$
    \mathbf{S}\mathbf{u}^\star - \mathbf{s} = \frac{L_f}{4} \mathbf{B}^\top \mathbf{B} \mathbf{u}^\star - \left(\frac{L_f}{4} + \frac{L_A}{4\sqrt{2}}\right) \mathbf{e}^{(2k)}_{2k}.
    $$
    由于 $\mathbf{B}\mathbf{u}^\star = \mathbf{1}_{2k}$, 故 $\mathbf{B}^\top \mathbf{B} \mathbf{u}^\star = \mathbf{B}^\top \mathbf{1}_{2k}$, 而 $\mathbf{B}^\top \mathbf{1}_{2k}$ 的最后一个分量为 $1$, 其他分量为 $0$. 因此
    $$
    \mathbf{S}\mathbf{u}^\star - \mathbf{s} = \frac{L_f}{4} \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix} - \left(\frac{L_f}{4}\right) \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix} = \mathbf{0}.
    $$
    因此 $\mathbf{u}^\star$ 确实是最优解, 且最优值为
    $$
    f^\star = -\frac{L_f}{4} k.
    $$

对于对偶问题, 同样分块 $\mathbf{y} = (\boldsymbol{\lambda}_{2k}^\top, \boldsymbol{\pi}_{m-2k}^\top)^\top$, 则对应写出上述 $\text{(a)}, \text{(b)}$ 两个子问题的 KKT stationarity 条件为
$$
\begin{aligned}
&\text{(a)} \quad \mathbf{S}\mathbf{u}^\star - \mathbf{s} = \frac{L_A}{2} \mathbf{B}^\top \boldsymbol{\lambda}^\star, \\
&\text{(b)} \quad \frac{L_f}{4} \mathbf{v}^\star = \frac{L_A}{2} \mathbf{G}^\top \boldsymbol{\pi}^\star \stackrel{\mathbf{v}^\star = \mathbf{0}}{\implies} \mathbf{G}^\top \boldsymbol{\pi}^\star = \mathbf{0}.
\end{aligned}
$$
- 对于 $\text{(b)}$, 当 $L_A > 0$, 由于 $\mathbf{G}$ 是 full row rank, 因此 $\boldsymbol{\pi}^\star = \mathbf{0}$ 是唯一解. 当 $L_A = 0$, 则方程有无数解, 不妨取 $\boldsymbol{\pi}^\star = \mathbf{0}$.
- 对于 $\text{(a)}$, LHS 经展开计算有
    $$
    \mathbf{S}\mathbf{u}^\star - \mathbf{s} = - \frac{L_A}{4\sqrt{2}} \mathbf{e}^{(2k)}_{2k} = \frac{L_A}{2} \mathbf{B}^\top \boldsymbol{\lambda}^\star
    $$
    - 当 $L_A > 0$, 由于 $\mathbf{B}$ 是 non-singular, 故唯一解为
        $$
        \boldsymbol{\lambda}^\star = - \frac{1}{2\sqrt{2}} (\mathbf{B}^\top)^{-1} \mathbf{e}^{(2k)}_{2k} = - \frac{1}{2\sqrt{2}} \mathbf{1}_{2k}.
        $$
    - 当 $L_A = 0$, 则 $\boldsymbol{\lambda}^\star$ 无约束, 不妨取 $\boldsymbol{\lambda}^\star = - \frac{1}{2\sqrt{2}} \mathbf{1}_{2k}$. 此时 $\text{(a)}$ 的 stationarity 条件仍然成立, 因为 $\mathbf{S}\mathbf{u}^\star - \mathbf{s} = \mathbf{0}$.

$\square$

整合上述结果, 得到 objective 和 feasibility gap 的收敛下界如下. 

> ***Lemma 10* (Lower Complexity Bound with Nonnegative $L_A$)**
>
> 给定 $L_f > 0$ 与 $L_A \geq 0$, 且假设 $L_f \geq L_A$.  则对于任意满足 Assumption 1 的一阶方法, 在前 $k$ 步迭代中, 有
> $$
> \min_{\mathbf{x} \in \mathcal{K}_{k-1}} |f(\mathbf{x}) - f^\star| \geq \frac{3 L_f \|\mathbf{x}^\star\|^2}{128 (k+1)^2} + \frac{\sqrt{3} L_A \|\mathbf{x}^\star\| \cdot \|\mathbf{y}^\star\|}{8(k+1)}, 
> $$
> $$
> \min_{\mathbf{x} \in \mathcal{K}_{k-1}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\| \geq \frac{\sqrt{3} L_A \|\mathbf{x}^\star\|}{4\sqrt{2}(k+1)}.
> $$
>

*Proof*. 由于本小节的约束条件 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 与上一小节相同, 因此 feasibility gap 的下界与上一小节相同. 
对于 objective gap,  由于 $\mathbf{x}^{(t)} \in \mathcal{K}_{t-1}$, 故不放记 $\mathbf{x}^{(t)} = (\mathbf{0}_k^\top, \mathbf{z}^\top, \mathbf{0}_{n-2k}^\top)^\top$, 其中非零部分为 $\mathbf{z} \in \mathbb{R}^{k}$, 且有对应关系 $x_{k+i} = z_i$, $i = 1, \ldots, k$. 
代入 $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x} - \mathbf{h}^\top \mathbf{x}$ 以及具体 $\mathbf{H}, \mathbf{h}$ 的构造, 得
$$
\begin{aligned}
f(\mathbf{x})
&= \frac{L_f}{8} \|\mathbf{\bar{B}}\mathbf{z}\|^2 - \left(\frac{L_f}{4} + \frac{L_A}{4\sqrt{2}}\right) z_k 
\end{aligned}
$$
- 其中, $\mathbf{\bar{B}} \in \mathbb{R}^{k \times k}$ 是 $\mathbf{B} \in \mathbb{R}^{2k \times 2k}$ 的右下角 $k \times k$ 子矩阵, 根据
    $$
    \mathbf{x}^\top \mathbf{H} \mathbf{x} = \frac{L_f}{4} \left\| \begin{bmatrix} \mathbf{B}_{2k} & \mathbf{O} \\ \mathbf{O} & \mathbf{I}_{n-2k} \end{bmatrix} \begin{bmatrix} \mathbf{0}_k \\ \mathbf{z}_k \\ \mathbf{0}_{n-2k} \end{bmatrix}\right\|^2 = \frac{L_f}{4} \left\|\mathbf{B} \begin{bmatrix} \mathbf{0}_k \\ \mathbf{z}_k \end{bmatrix}\right\|^2 := \frac{L_f}{4} \|\mathbf{\bar{B}}\mathbf{z}\|^2.
    $$
    可得到其具体形式为:
    $$
    \mathbf{\bar{B}} = \begin{bmatrix}
         &  &   & -1 & 1 \\
         &  &   & \vdots & \vdots \\
         & -1 & 1 &  &  \\
        -1 & 1 &  &  &  \\
        1 &  &  &  & 
    \end{bmatrix} \in \mathbb{R}^{k \times k}.
    $$ 

故上述 $n$ 维空间中关于 $\mathbf{x}$ 的优化问题, 可以等价转化为 $k$ 维空间中关于 $\mathbf{z}$ 的无约束优化问题:
$$
\min_{\mathbf{x} \in \mathcal{K}_{k-1}} f(\mathbf{x}) = \min_{\mathbf{z} \in \mathbb{R}^{k}}  \left\{\frac{L_f}{8} \|\mathbf{\bar{B}}\mathbf{z}\|^2 - \left(\frac{L_f}{4} + \frac{L_A}{4\sqrt{2}}\right) z_k\right\}.
$$
该问题同样是 strictly convex 的, 经过求解, 得到最优解
$$
\mathbf{z}^\star = \frac{4}{L_f} \left(\frac{L_f}{4} + \frac{L_A}{4\sqrt{2}}\right) (1,2, \ldots, k)^\top
$$
代入计算其最小值有
$$
\min_{\mathbf{x} \in \mathcal{K}_{k-1}} f(\mathbf{x}) = - \frac{1}{8} \left(L_f +\sqrt{2} L_A + \frac{L_A^2}{2 L_f}\right) k.
$$
故代入上述求解的最优值 $f^\star = - \left(\frac{L_f}{4} + \frac{L_A}{2\sqrt{2}}\right) k$, 得到
$$
\min_{\mathbf{x} \in \mathcal{K}_{k-1}} f(\mathbf{x}) - f^\star = \frac{1}{8} \left(L_f +\sqrt{2} L_A - \frac{L_A^2}{2 L_f}\right) k  \stackrel{L_f \geq L_A}{\geq}
\frac{1}{8} \left(\frac{L_f}{2} + \sqrt{2} L_A\right) k 
$$

最后, 利用 $\|\mathbf{x}^\star\|^2 = \frac{k(2k+1)(4k+1)}{3}$, $\|\mathbf{y}^\star\|^2 = \frac{k}{8}$, 并进行代数放缩, 得到
$$
\min_{\mathbf{x} \in \mathcal{K}_{k-1}} f(\mathbf{x}) - f^\star \geq \frac{3 L_f \|\mathbf{x}^\star\|^2}{128 (k+1)^2} + \frac{\sqrt{3} L_A \|\mathbf{x}^\star\| \cdot \|\mathbf{y}^\star\|}{8(k+1)}.
$$

$\square$

综合上述各个结论, 得到如下定理.

> ***Theorem 4* (Lower Complexity Bound with Nonnegative $L_A$)**
>
> 给定 $m \leq n, L_f > 0, L_A \geq 0, L_f \geq L_A$, 给定任意正整数 $t < m/2$, 存在上述形式的构造使得 $f$ 是 $L_f$-Lipschitz 光滑的, $\|\mathbf{A}\| = L_A$, 且对于任意满足 Assumption 1 的一阶方法, 在前 $t$ 步迭代中, 其迭代点 $\mathbf{x}^{(t)}$ 与 primal-dual 最优解 $(\mathbf{x}^\star, \mathbf{y}^\star)$ 满足关系:
>
> $$
> \begin{aligned}
> &|f(\mathbf{x}^{(t)}) - f(\mathbf{x}^\star)| \geq \frac{3 L_f \|\mathbf{x}^\star\|^2}{128 (t+1)^2} + \frac{\sqrt{3} L_A \|\mathbf{x}^\star\| \cdot \|\mathbf{y}^\star\|}{8(t+1)}, \\
> &\|\mathbf{A}\mathbf{x}^{(t)} - \mathbf{b}\| \geq \frac{\sqrt{3} L_A \|\mathbf{x}^\star\|}{4\sqrt{2}(t+1)}.
> \end{aligned}
> $$


上述定理基本可由 Lemma 10 直接得到, 只需注意到 $\mathbf{x}^{(t)} \in \mathcal{K}_{t-1}$, 故任意算法的表现不可能优于该空间内的 lower bound.

对于该结论有如下说明:
- 关于 optimality gap, 忽略常数, 其结构为
    $$
    f(\mathbf{x}^{(t)}) - f^\star \gtrsim \frac{L_f \|\mathbf{x}^\star\|^2}{(t+1)^2} + \frac{L_A \|\mathbf{x}^\star\| \cdot \|\mathbf{y}^\star\|}{(t+1)}.
    $$
    - 其中第一项是 smooth convex objective 本身带来的困难, 经过加速, 可以达到 $\Omega(1/t^2)$ 的收敛率. 这也是无约束凸优化的经典收敛率 lower bound.
    - 第二项是约束条件带来的困难, 来自 affine constraint 或 primal-dual coupling, 其中 $L_A = \|\mathbf{A}\|$ 是约束条件的难度系数, $\|\mathbf{x}^\star\|$ 表示 primal solution 的规模, $\|\mathbf{y}^\star\|$ 表示 dual solution 的规模. 回顾在 saddle point 问题中, $\langle \mathbf{A}\mathbf{x}, \mathbf{y} \rangle$ 是 primal-dual coupling 的核心, 而其中的尺度自然就是 $\|\mathbf{A}\| \cdot \|\mathbf{x}^\star\| \cdot \|\mathbf{y}^\star\|$. 且这一项只能以 $1/t$ 的收敛率下降.
    - 此外, 文中提出, 已有的 first-order method 的 upper bound 也具有
        $$
        \mathcal{O}\left(\frac{L_f \|\mathbf{x}^\star\|^2}{t^2} + \frac{L_A \|\mathbf{x}^\star\| \cdot \|\mathbf{y}^\star\|}{t}\right),
        $$
        的形式, 因此这个 rate 是 tight 的. 
- 关于 feasibility gap, 其结构为
    $$
    \|\mathbf{A}\mathbf{x}^{(t)} - \mathbf{b}\| \gtrsim \frac{L_A \|\mathbf{x}^\star\|}{(t+1)}.
    $$
    - 同样也在说明, 由于约束的存在, 即使目标函数很快收敛, 也无法保证可行性, 其收敛率只能是 $\Omega(1/t)$.

### 2.5 A Lower Complexity Bound for Strongly Convex Case

上述的研究是在目标函数 $f$ 是 convex 的前提下给出的. 若进一步假设 $f$ 是 $\mu$-strongly convex:
$$
\langle \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}), \mathbf{x} - \mathbf{y} \rangle \geq \mu \|\mathbf{x} - \mathbf{y}\|^2, \quad \forall \mathbf{x},\mathbf{y} \in \mathbb{R}^n,
$$
理论上对于普通 unconstrained 问题, 可以有 linear convergence, 即误差按照 $q^t, q \in (0,1)$ 的速率收敛. 然而若额外考虑 affine constraint $\mathbf{A}\mathbf{x} = \mathbf{b}$, 则文中指出, 往往 linear convergence 将无法得到保证. 

考虑到强凸性保证原问题必有唯一解, 因此可以直接考虑迭代点 $\mathbf{x}^{(t)}$ 与最优解 $\mathbf{x}^\star$ 的距离 $\|\mathbf{x}^{(t)} - \mathbf{x}^\star\|^2$ 作为衡量指标. 

> ***Theorem 5* (Lower Complexity Bound for Strongly Convex Case)**
>
> 给定 $m \leq n, \mu > 0, L_A \geq 0$, 以及任意 $t < m/2$, 可以构造一个 affinely constrained problem
> $$
> \min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}), \quad \text{s.t. } \mathbf{A}\mathbf{x} = \mathbf{b},
> $$
> 其中 $f$ 是 $\mu$-strongly convex 的, $\|\mathbf{A}\| = L_A$, 且这个问题具有唯一的 primal-dual 最优解 $(\mathbf{x}^\star, \mathbf{y}^\star)$, 使得对于任意满足 Assumption 1 的一阶方法, 在前 $t$ 步迭代中, 其迭代点 $\mathbf{x}^{(t)}$ 与最优解 $\mathbf{x}^\star$ 满足关系:
> $$
> \|\mathbf{x}^{(t)} - \mathbf{x}^\star\|^2 \geq \frac{5 L_A^2 \|\mathbf{y}^\star\|^2}{256 \mu^2 (t+1)^2}.
> $$

*Proof*. 令 $k=t$, 考虑 quadratic problem:
$$
\min_{\mathbf{x} \in \mathbb{R}^n} \left\{ f(\mathbf{x}) :=  \frac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x} - \mathbf{h}^\top \mathbf{x}
\right\}, \quad \text{s.t. } \mathbf{A}\mathbf{x} = \mathbf{b},
$$
并令 $\mathbf{H} = \mu \mathbf{I}_n$, $\mathbf{h} = \mathbf{0}$, $\mathbf{A} = \frac{L_A}{2} \boldsymbol{\Lambda}$, $\mathbf{b} = \frac{L_A}{2} \mathbf{c}$. 如此构造, 可知 $f$ 是 $\mu$-strongly convex 的, 且 $\|\mathbf{A}\| = L_A$. 

显然如此构造的问题, 仍然满足 $\mathbf{h} \in \mathcal{K}_0$ 且 $\mathbf{H}\mathcal{K}_{k-1} = \mu \mathcal{K}_{k-1} \subseteq \mathcal{K}_k$, 因此 Lemma 4 仍然成立, 即 $\mathbf{x}^{(k)} \in \mathcal{K}_{k-1}$. 对 $\mathbf{x}$ 同样进行分块计算, 得到唯一最优解
$$
\mathbf{x}^\star = (1, 2, \ldots, 2k, 0, \ldots, 0)^\top
$$
以及对偶解
$$
y_i^\star = \begin{cases}
\frac{\mu}{L_A} i(4k - i + 1), & i = 1, \ldots, 2k, \\
0, & i = 2k+1, \ldots, m.
\end{cases}
$$
故对于任意 $\mathbf{x} \in \mathcal{K}_{k-1}$, 有
$$
\|\mathbf{x} - \mathbf{x}^\star\|^2 = \sum_{i=1}^{2k} (x_i - x_i^\star)^2 + \sum_{i=2k+1}^{n} (0 - 0)^2 \geq \sum_{i=1}^{k} (0 - i)^2 = \frac{k(k+1)(2k+1)}{6}.
$$

另外, 注意到根据单纯计算整理
$$
\|\mathbf{y}^\star\|^2 = \frac{2k(2k+1)(4k+1)}{15L_A^2}(16k^2+8k+2)
$$
代入上述 $\|\mathbf{x} - \mathbf{x}^\star\|^2$ 的下界, 并进行代数放缩即证. 具体细节略.  

## 3. Lower Complexity Bounds of General First-Order Methods for Affinely Constrained Problems

在上一个 section 中, 无论具体 setting 如何, 其迭代点都依赖于如下 linear span 的一阶算法假设:
$$
\mathbf{x}^{(t)} \in \text{span}\{\nabla f(\mathbf{x}^{(0)}), \mathbf{A}^\top \mathbf{r}^{(0)}, \ldots, \nabla f(\mathbf{x}^{(t-1)}), \mathbf{A}^\top \mathbf{r}^{(t-1)}\}, \quad t = 1, 2, \ldots
$$
并且由此归纳证明了 $\mathbf{x}^{(t)} \in \mathcal{K}_{t-1}$. 
本文则进一步考虑更 general 的 first-order methods, 认为 $\mathcal{I}_t$ 可以是任意 deterministic 的规则:
$$
(\mathbf{x}^{(t+1)}, \mathbf{y}^{(t+1)}, \mathbf{\bar{x}}^{(t+1)}, \mathbf{\bar{y}}^{(t+1)}) = \mathcal{I}_t(\boldsymbol{\theta}; \mathrm{O}(\mathbf{x}^{(0)}, \mathbf{y}^{(0)}), \ldots, \mathrm{O}(\mathbf{x}^{(t)}, \mathbf{y}^{(t)})), \quad t = 0, 1, 2, \ldots
$$
这将允许包括 projection 等在内的许多非线性操作. 文章将使用 Nemirovski 等提出的 **rotation variance** 技巧, 将任意 instance 转化回 Section 2 所定义的示例当中. 

首先明确问题定义. 给定半正定矩阵 $\mathbf{H} \in \mathbb{S}_+^n$, $\mathbf{A} \in \mathbb{R}^{m \times n}$, 以及参数 $\boldsymbol{\theta} = (\mathbf{h}, \mathbf{b}, R_X, R_Y, \lambda)$, 其中
- $\mathbf{h} \in \mathbb{R}^n$ 是目标函数的线性项,
- $\mathbf{b} \in \mathbb{R}^m$ 是约束条件的右端项,
- $R_X, R_Y  \in [0, \infty]$ 是 primal, dual 约束球的半径, 当取 $\infty$ 时表示无约束.
- $\lambda \in [0, \infty)$ 是对偶目标中 $\|\mathbf{y}\|^2$ 的正则化系数.

根据上述符号, 定义 instance $\mathrm{P}(\boldsymbol{\theta}; \mathbf{H}, \mathbf{A})$ 为如下的 affinely constrained problem:
$$
\phi^* = \min_{\|\mathbf{x}\| \leq R_X} \left\{
\frac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x} - \mathbf{h}^\top \mathbf{x} + \max_{\|\mathbf{y}\| \leq R_Y} \left[\langle \mathbf{A}\mathbf{x} - \mathbf{b}, \mathbf{y}\rangle - \frac{\lambda}{2} \|\mathbf{y}\|^2\right]
\right\}
$$
- 说明: 这样的定义是 general 的, 其包含了 Section 2 中的形式, 只需令 $R_X = R_Y = \infty, \lambda = 0$ 即可.

$\diamond$

下正式给出 rotation 的相关性质. 

> ***Proposition 1* (Rotated Instance Properties)**
>
> 给定如下前提条件:
> - 给定问题维度 $m \leq n$, Krylov 子空间维度 $k < m/2$, 迭代步数 $t \leq k/2 -1$.
> - 给定 $f$ 的 Lipschitz 光滑常数 $L_f \geq 0$, 以及 $\mathbf{A}$ 的谱范数 $L_A \geq 0$.
>
> 考虑由上面定义的 original instance $\mathrm{P}(\boldsymbol{\theta}; \mathbf{H}, \mathbf{A})$, 且该 instance 须满足如下条件:
> - $\|\mathbf{H}\| \leq L_f$: 其保证 $f$ 是 $L_f$-Lipschitz 光滑的.
> - $\mathbf{A} = \frac{L_A}{2} \boldsymbol{\Lambda}$, $\mathbf{b} = \frac{L_A}{2} \mathbf{c}$, 其中 $\boldsymbol{\Lambda}, \mathbf{c}$ 的构造如 Section 2.1 所述. 这是要转化的 hard instance 目标. 
> - $\mathbf{H} \in \mathbb{S}_+^n$, 且 $\mathbf{H} \mathcal{K}_{2s-1} \subseteq \mathcal{K}_{2s}$, $\forall s \leq k/2$: 这为了保证 Krylov 子空间的嵌套关系, 每次 $\mathbf{H}$ 的作用只会增加一个有效维度. 
> - $\mathbf{h} \in \mathcal{K}_0$: 回忆 $\mathcal{K}_0 = \text{span}\{\mathbf{e}_{n}^{(2k)}\}$.
>
> 对于满足上述条件的 original instance, 在给定一个一阶方法 $\mathcal{M}$ 的前提下, 都能找到另一组旋转后的 instance $\mathrm{P}(\boldsymbol{\theta}; \mathbf{\tilde{H}}, \mathbf{\tilde{A}})$, 其中 $\mathbf{\tilde{H}} = \mathbf{U}^\top \mathbf{H} \mathbf{U}$, $\mathbf{\tilde{A}} = \mathbf{V}^\top \mathbf{A} \mathbf{V}$, 且 $\mathbf{U}, \mathbf{V}$ 是 orthogonal matrices, 其依赖于具体迭代 $t$, 并满足 $\mathbf{U} \mathbf{h} = \mathbf{h}$, $\mathbf{V}\mathbf{b} = \mathbf{b}$, 则有结论如下: 
>
> 1. 若原问题的 saddle point 为 $(\mathbf{x}^\star, \mathbf{y}^\star)$, 则旋转后的 instance 的 saddle point 为 $(\mathbf{\hat{x}}, \mathbf{\hat{y}}) = (\mathbf{U}^\top \mathbf{x}^\star, \mathbf{V}^\top \mathbf{y}^\star)$, 且 $\|\mathbf{\hat{x}}\| = \|\mathbf{x}^\star\|$, $\|\mathbf{\hat{y}}\| = \|\mathbf{y}^\star\|$.
>
> 2. 当使用 $\mathcal{M}$ 在 rotated instance  进行 $t$ 步迭代后得到的迭代点 $(\mathbf{\bar{x}}^{(t)}, \mathbf{\bar{y}}^{(t)})$, 在经过旋转变换映射回 original space 后, 仍然能落在 section 2 中分析的 $\mathcal{K}_{t-1}$ 中, 故有优化下界:
>   $$
>  \begin{aligned}
> & \tilde{\phi}(\mathbf{\bar{x}}^{(t)}) - \tilde{\phi}^\star \geq \min_{\mathbf{x} \in \mathcal{K}_{t-1}} \phi(\mathbf{x}) - \phi^\star, \\
> & \tilde{f}(\mathbf{\bar{x}}^{(t)}) - \tilde{f}(\mathbf{\hat{x}}) \geq \min_{\mathbf{x} \in \mathcal{K}_{t-1}} f(\mathbf{x}) - f(\mathbf{x}^\star),\\
> &\|\mathbf{\tilde{A}}\mathbf{\bar{x}}^{(t)} - \mathbf{b}\| \geq \min_{\mathbf{x} \in \mathcal{K}_{t-1}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|,\\
> & \|\mathbf{\tilde{x}}^{(t)} - \mathbf{\hat{x}}\|^2 \geq \min_{\mathbf{x} \in \mathcal{K}_{t-1}} \|\mathbf{x} - \mathbf{x}^\star\|^2.
>  \end{aligned}
> $$

- 这个 proposition 将任意算法 $\mathcal{M}$ 的收敛情况都被 Section 2 中的 lower bound 所控制, 因此可以得到 general first-order methods 的 lower complexity bound.

- 该 proposition 的证明较为繁杂, 不妨先承认该定理的合法性, 其具体证明将单独给出在如下链接中. 

### 3.1 Lower Complexity Bounds

这里首先明确一下我们黑盒优化的世界观. 
- 首先会给定一个确定但 arbitrary 的一阶方法 $\mathcal{M}$, 这个方法是固定的, 然而具体利用的信息是任意的(只要是一阶的).  这会导致一个特点: 尽管算法可能很强大, 然而只要在第 $s$ 步查询点 $\mathbf{x}^{(s)}, \mathbf{y}^{(s)}$ 是历史 oracle 回答的一个固定函数, 那么算法便无法区分具体的 instance 是什么. (换言之, 类似插值函数的比喻, 只要每次 instance 给出的查询点所需的信息是相同的, 我们可以任意的调整 instance 的构造, 使得算法无法区分, 从而构造一个困难的优化问题.)

- 因此, 我们可以以博弈的方式针对 $\mathcal{M}$ 每个 iteration 时的查询去调整 instance, 构造出一条 instance 序列 $\mathrm{P}_0, \mathrm{P}_1, \ldots$, 只要保证在在第 $s$ 次迭代时, $\mathrm{P}_s$ 的历史轨迹和 $\mathrm{P}_{<s}$ 的历史轨迹是相同的, 这样的任意构造都是合理的. 


> ***Theorem 6* (Lower Complexity Bound (I) of General First-Order Methods)**
>
> 考虑前提条件:
> - $8 < m \leq n$ (保证 Krylov 子空间的维度 $k$ 足够大)
> - $L_f > 0$, $L_A > 0$. 
> 
> 对于任意 $t < m/4 - 1$, 任意上述一阶方法 $\mathcal{M}$, 存在某种 instance 的构造 
> $$
> \tilde{f}^\star = \min_{\mathbf{x} \in \mathbb{R}^n} \left\{\tilde{f}(\mathbf{x}), ~ \text{s.t. } \mathbf{\tilde{A}}\mathbf{x} = \mathbf{b},\right\}
> $$
> 满足 $\tilde{f}$ 是 $L_f$-Lipschitz 光滑的, $\|\mathbf{\tilde{A}}\| = L_A$, 且该问题具有唯一的 primal-dual 最优解 $(\mathbf{\hat{x}}, \mathbf{\hat{y}})$,
> 并且 $\mathcal{M}$ 在前 $t$ 步迭代中, 其迭代点 $\mathbf{\bar{x}}^{(t)}$ 满足
>  $$
>  \tilde{f}(\mathbf{\bar{x}}^{(t)}) - \tilde{f}^\star \geq \frac{3 L_f \|\mathbf{\hat{x}}\|^2}{64 (2t+5)^2} + \frac{\sqrt{3}}{16(2t+5)} L_A \|\mathbf{\hat{x}}\| \cdot \|\mathbf{\hat{y}}\|,
> $$
> 以及
> $$
> \|\mathbf{\tilde{A}}\mathbf{\bar{x}}^{(t)} - \mathbf{b}\| \geq \frac{\sqrt{3} L_A \|\mathbf{\hat{x}}\|}{4\sqrt{2}(2t+5)}.
> $$
> 其中 $(\mathbf{\hat{x}}, \mathbf{\hat{y}})$ 是 rotated instance 的 primal-dual 最优解.

*Proof Sketch*.  
- 设 $k = 2t + 2$ (以满足定理中的各种维度要求), 并 accordingly 可以确定 $\boldsymbol{\Lambda}, \mathbf{c}$ 的构造. 
- 按照 Section 2 中 convex case 方法的形式构造 original instance:
    $$
    \mathbf{H} = \frac{L_f}{4} \begin{bmatrix} \mathbf{B}^\top \mathbf{B} & \mathbf{0} \\ \mathbf{0} & \mathbf{I}_{n-2k} \end{bmatrix} , \quad \mathbf{h} = \frac{L_f}{2} \mathbf{e}^{(2k)}_{n}, \quad \mathbf{A} = \frac{L_A}{2} \boldsymbol{\Lambda}, \quad \mathbf{b} = \frac{L_A}{2} \mathbf{c}.
    $$
- 可以验证其满足上述 proposition 中的所有条件.
- 根据 Proposition 1, 可以得到 rotated instance $\mathrm{P}(\boldsymbol{\theta}; \mathbf{\tilde{H}}, \mathbf{\tilde{A}})$, 且 $\mathbf{\tilde{H}} = \mathbf{U}^\top \mathbf{H} \mathbf{U}$, $\mathbf{\tilde{A}} = \mathbf{V}^\top \mathbf{A} \mathbf{V}$, 其中 $\mathbf{U}, \mathbf{V}$ 是 orthogonal matrices, 且 $\mathbf{U}\mathbf{h} = \mathbf{h}$, $\mathbf{V}\mathbf{b} = \mathbf{b}$.
- 根据 Proposition 1, 可以得到 rotated instance 的对应优化下界, 并代入 Section 2 的具体数值即证. 

以及当 strongly convex 时, 也可以 accordingly 给出如下定理.

> ***Theorem 7* (Lower Complexity Bound (II) of General First-Order Methods)**
>
> 令 $8 < m \leq n$, $\mu > 0$, $L_A \geq 0$. 对于任意 $t < m/4 - 1$, 任意上述一阶方法 $\mathcal{M}$, 存在相同的构造使得
> $$
> \|\mathbf{\bar{x}}^{(t)} - \mathbf{\hat{x}}\|^2 \geq \frac{5 L_A^2 \|\mathbf{\hat{y}}\|^2}{256 \mu^2 (2t+5)^2}.
> $$

