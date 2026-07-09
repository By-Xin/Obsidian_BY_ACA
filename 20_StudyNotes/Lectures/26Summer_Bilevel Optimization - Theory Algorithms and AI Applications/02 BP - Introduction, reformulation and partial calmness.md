# Bilevel Programming - Introduction, Reformulation and Partial Calmness

> Speaker: Zhang Jin (Southern University of Science and Technology, China)
>
> Date: July 06, 2026

## 1. Introduction to Bilevel Programming

### 1.1 Basic Problem Formulation

考虑一般的 Bilevel Programming 问题, 其形式为:
$$
\begin{aligned}
&\min_{x, y}  \quad && F(x, y) \\
&\text{s.t.} \quad && y \in S(x) := \arg\min_{y \in Y(x)} f(x, y),
\end{aligned}
\tag{BP}
$$
其中 
- $x$ 是上层决策变量, 在博弈论中为 leader; $y$ 为下层决策变量, 在博弈论中为 follower. 
- $\min_{y \in Y(x)} f(x, y)$ 是下层优化问题 $\text{P}_x$, 其可行域为 $Y(x) := \{y: g(x, y) \leq 0\}$, $g(x, y)$ 为某种下层约束. 
- 在上层 $x$ 决策时, 下层的反应是已知的 (数学上即已知下层 $\text{P}_x$ 的最优解集 $S(x)$). 并且最终的目标函数 $F(x, y)$ 也依赖于下层的决策变量 $y$.
- 若不加特别说明, 暂时假设 $F, f, g$ 均为光滑的.

### 1.2 Single-level Reformulation Intuition

若下层问题 $\text{P}_x$ 的解对于每个给定的 $x$ 都是唯一的, 则可以将其记为 $y^\star(x)$, 并且此时双层问题可以被简化为一个单层优化问题:
$$
\min_{x} F(x, y^\star(x)).
$$

- 例如考虑如下的双层问题:
    $$
    \min_{x, y} F(x, y) \quad \text{s.t.} \quad y \in \arg\min_{y'} f(x, y') 
    $$
    这里若假设 $f(x, y)$ 对于 $y$ 是凸且光滑的, 则 $y^\star(x)$ 唯一且满足 $\nabla_y f(x, y^\star(x)) = 0$. 因此, 双层问题可以被简化为单层问题:
    $$
    \min_{x, y} F(x, y) \quad \text{s.t.} \quad \nabla_y f(x, y(x)) = 0.
    $$

### 1.3 Optimistic vs. Pessimistic BP 

然而在非凸甚至更一般的情况下, 解的非唯一性会给问题的求解带来麻烦: 当下层问题有多个最优解时, 其对于 $\text{P}_x$ 的最优性是相等的, 然而对于上层问题来说则可能有很大差别. 因此根据不同的假设, 双层问题可以被分为两类:
- **Optimistic BP**: 在下层问题有多个最优解时, 上层决策者假设下层会选择对上层最有利的解 (该假设相对好处理, 这里也主要讨论这种情况), 即:
    $$
    \min_{x, y} \{ F(x, y) : ~ y \in S(x)\}.
    $$
- **Pessimistic BP**: 相当于 worst-case 的假设, 即下层会选择对上层最不利的解, 然后讨论这个 worst-case 下的最优解:
    $$
    \min_{x, y} \max_{y \in S(x)} F(x, y). 
    $$


## 2. Single-level Reformulation: Lower Level Unconstrained

首先考虑下层问题没有约束的情况:
$$
\begin{aligned}
&\min_{x, y}  \quad && F(x, y) \\
&\text{s.t.} \quad && y \in S(x)
\end{aligned} \tag{BP}
$$
其中 $S(x)$ 是下层问题 $\text{P}_x$ 的最优解集, 即:
$$
\begin{aligned}
&\min_{y}  \quad && f(x, y) \tag{P$_x$}
\end{aligned}
$$


### 2.1 First-order Approach and Implicit Function Reformulation

即使 $S(x) = \{y(x)\}$ 是单点集能够转化为单层问题, 其往往也需要一些进一步的条件才能让问题变得可解. 
- 例如, 通常需要 $y(x)$ 是 Lipschitz 连续的. 则需要满足以下条件:
  - **LICQ**: 下层问题的约束 $g(x, y)$ 是 LICQ 的. 即在可行点 $\bar{y}$ 处, 活跃约束 (即 $g_i(x, \bar{y}) = 0$) 的梯度向量是线性无关的. 这保证了 KKT 的乘子是唯一的. 
  - **Strong SOSC**: 对于下层问题的 Lagrange 函数 $L(y, \lambda; x) := f(x, y) + \lambda^\top g(x, y)$, 若 $(\bar{y}, \bar{\lambda})$ 是给定 $\bar{x}$ 处的一个 KKT 点, 则 SOSC 要求: 对所有 $\lambda_i > 0$ 的活跃约束的梯度向量 $\nabla_y g_i(\bar{x}, \bar{y})$, 其正交的空间上, Hessian $\nabla^2_{yy} L(\bar{y}, \bar{\lambda}; \bar{x})$ 是正定的. 即对任意方向向量 $d$, 若 $\nabla_y g_i(\bar{x}, \bar{y})^\top d = 0$ 且 $d \neq 0$, 则应有 $d^\top \nabla^2_{yy} L(\bar{y}, \bar{\lambda}; \bar{x}) d > 0$. 这保证了 KKT 方程组关于 $(y, \lambda)$ 的 Jacobian 在 $(\bar{y}, \bar{\lambda})$ 处是非奇异的.
- 当上述连续性条件满足时, 对于单层优化问题, 可以通过 Implicit Function Theorem 来求外层目标 $F$ 关于 $x$ 的导数. 首先根据得到的梯度等式关系, 有
    $$
    \frac{\mathrm{d}}{\mathrm{d} x} \left[
        \nabla_y f(x, y(x)) 
    \right] = \frac{\partial}{\partial x} \nabla_y f(x, y) + \frac{\partial}{\partial y} \nabla_y f(x, y) \frac{\mathrm{d} y(x)}{\mathrm{d} x} = 0.
    $$
    因此可以得到:
    $$
    0 = \nabla^2_{xy} f(x, y) + \nabla^2_{yy} f(x, y) \nabla y(x) \implies \nabla y(x) = - \left[ \nabla^2_{yy} f(x, y) \right]^{-1} \nabla^2_{xy} f(x, y).
    $$
    故最后有:
    $$
    \begin{aligned}
    \frac{\mathrm{d}}{\mathrm{d} x} F(x, y(x)) &= \nabla_x F(x, y) + \nabla_y F(x, y) \nabla y(x) \\
    &= \nabla_x F(x, y) - \nabla_y F(x, y) \left[ \nabla^2_{yy} f(x, y) \right]^{-1} \nabla^2_{xy} f(x, y).
    \end{aligned}
    $$

### 2.2 Failure in the Nonconvex Case: Mirrlees' Example

若在非凸问题上, 仍同样使用 $\nabla_y f(x, y) = 0$ 来替代下层问题的最优性条件, 则可能无法得到正确的最优解 (尽管在实践中这一做法往往是十分诱人且常见的).

***Example* (Mirrlees' Problem)**: 考虑如下的具体优化问题
$$
\begin{aligned}
&\min_{x, y}  \quad && F(x, y) = (x-2)^2 + (y-1)^2 \\
&\text{s.t.} \quad && y \in \arg\min_{y} f(x, y) := - x \exp(- (y+1)^2) - \exp(- (y-1)^2).
\end{aligned}
$$

- 求解这个下层目标函数关于 $y$ 的一阶导数并令之为零, 得到如下方程:
    $$
    -2 x (y+1) \exp(- (y+1)^2) + 2 (y-1) \exp(- (y-1)^2) = 0 \iff x = \frac{1-y}{1+y} \exp(4y). 
    $$
    - 换言之, 这条曲线上的点都是下层问题的 stationary points. 

- 下图展示的是下层问题 $f(x, y)$ 的 contour 图, 其中红色曲线表示 $\nabla_y f(x, y) = 0$ 的点. 可以观察到, 在 $y=1$ 和 $y=-1$ 附近有两条 ‘峡谷’, 分别对应着下层问题可能的最优解. 
- 然而, 尽管图中的几个点都是 stationary points, 但 pt1 和 pt2 都不是下层问题的最优解, pt3 和 BP optimal solution 才是下层问题的最优解. 因此, 若直接使用 $\nabla_y f(x, y) = 0$ 来替代下层问题的最优性条件, 则可能会得到错误的最优解.

- 换言之, 如果按照 SP 问题直接用一阶条件来分析, 整条红色曲线都是可行的, 其并不区分局部/全局最小, 甚至局部最大. 然而如果真的考虑原始的 BP 问题, 则可行集只是红色曲线中的子集. 如果真正的答案真的发生在跳变的位置, 则 SP 就会错误的进行光滑化, 甚至可能会得到错误的最优解.

- 并且这里指出, 即使上文的 LICQ + 强 SOSC 性质得到满足, 其衡量的仍然只是局部的微分几何性质, 并不能保证全局的最优性. 因此在该例中, 仍然无法保证 $\nabla_y f(x, y) = 0$ 的点就是下层问题的最优解.

![mirrlees_contour](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/mirrlees_contour.png)

### 2.3 Value Function Reformulation

从前面的例子可以看到, 即使下层问题解唯一, 通过隐函数的方法也并不总能得到正确的最优解. 因为在非凸等几何复杂的情况下, 上述的最优性条件等通常考虑的都是局部的性质, 并不能保证全局的最优性. 因此, 另一种常用的处理方法是通过引入下层问题的 value function 来 reformulate 双层问题. 这是一种全局的操作. 

#### 2.3.1 Definition of Value Function

考虑同样的 BP 问题:
$$
\begin{aligned}
&\min_{x, y}  \quad && F(x, y) \\
&\text{s.t.} \quad && y \in S(x) := \arg\min_{y \in Y(x)} f(x, y),
\end{aligned}
$$
- 其中 $Y(x) := \{y: g(x, y) \leq 0\}$ 是下层问题的可行域. 

定义下层问题的 value function 为:
$$
V(x) := \inf_y \{ f(x, y) \}
$$
- 其中 $y$ 可以在全空间取值. 

则可以根据 value function 定义下面的 BP 问题的等价问题:
$$
\begin{aligned}
&\min_{x, y}  \quad && F(x, y) \\
&\text{s.t.} \quad && f(x, y) - V(x)\leq 0.
\end{aligned}
\tag{VP}
$$
- 其等价性是因为: 根据 $V(x)$ 的定义, $f(x, y) \geq \inf_y f(x, y) = V(x)$ 对于所有 $y$ 都成立, 因此 $f(x, y) - V(x) \leq 0$ 等价于 $f(x, y) = V(x)$, 即 $y$ 是下层问题的最优解.
- VP 方法把全局最小的要求直接通过引入一个新的函数的方式添加到约束中. 区别于前面通过一阶条件来替代下层问题的一阶必要性条件, VP 的刻画是全局的. 

为进一步分析 VP 问题, 需考察约束 $h(x,y) := f(x, y) - V(x)$ 的梯度. 
- 从简单情况开始. 若 $S(x) = \{y(x)\}$ 是单点集, 则 $V(x) = f(x, y(x))$. 因此有:
    $$
    \nabla_x V(x) = \nabla_x f(x, y(x)) + \nabla_y f(x, y(x)) \nabla y(x) = \nabla_x f(x, y(x)),
    $$
    其中第二个等号是因为 $y(x)$ 是下层问题的最优解, 因此 $\nabla_y f(x, y(x)) = 0$.
- 扩展到非单点集, 则 $V(x)$ 的梯度 (这里是广义 Clarke subgradient) 可以通过 Danskin 定理来计算, 结果为:
    $$
    \partial^c V(x)=\mathrm{Cvx}\{\nabla_x f(x,y):y\in S(x)\}
    $$
    - 其中 $\mathrm{Cvx}\{\cdot\}$ 表示凸包. 换言之, $V(x)$ 的梯度是所有下层最优解的梯度的凸包.
- 因此最终的梯度为:
    $$
     \partial h(x,y) = \{ 
        (\nabla_x f(x, y) - \xi, \nabla_y f(x, y)): \xi \in \partial^c V(x)
        \}
    $$
    若写紧凑些, 则有:
    $$
    \partial h(x,y) = \nabla f(x, y) - \partial^c V(x) \times \{0\}.
    $$
    - 这表示, 把整个梯度向量 $(\nabla_x f(x, y), \nabla_y f(x, y))$ 的 $x$ 部分减去 $\partial^c V(x)$ 的所有可能的梯度, 而 $y$ 部分保持不变.

#### 2.3.2 Problems of VP

然而这样的 reformulation 也带来了一些新的问题:
1. 尽管下层目标函数 $f(x, y)$ 本身是光滑的, 然而 $V(x) = \inf_y f(x, y)$ 并不一定是光滑的. 例如下图是 Mirrlees 问题中 $V(x)$ 的图像, 可以看到其存在着 kinks, 即不可微的点. 
    <!-- ![20260708230728](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260708230728.png) -->
2. 即使 $V(x)$ 以及其余所有的函数都是光滑的, VP 问题对应的 MFCQ 要求仍然不被满足. 换言之, 这里的 KKT 乘子可能是退化的. 
   - 回顾 MFCQ 定义. 若其成立, 需要有一个方向 $d$ 使得 $\partial h_i(x, y)^\top d < 0$ 对于所有活跃约束 $i \in I(x, y)$ 都成立. 因此立刻得到 $\text{MFCQ Holdes} \iff (0,0) \notin \partial h(x, y)$.
   - 然而对于 VP 问题的任意可行解 $(\bar{x}, \bar{y})$, 其必满足 $h(\bar{x}, \bar{y}) = 0$ , 即 $(\bar{x}, \bar{y})$ 是下层问题的最优解. 因此 Clarke subgradient 必然有 $(0, 0) \in \partial h(\bar{x}, \bar{y})$, 因此 MFCQ 对于 VP 中任意可行点均不成立. 

如果不考虑 MFCQ, 直接形式上写出 VP 问题的 KKT 条件, 则有:
$$
\begin{aligned}
0 &\in \nabla F(\bar{x}, \bar{y}) + \mu \partial^c h(\bar{x}, \bar{y}) 
\end{aligned}
$$
将其按分量展开:
$$
\begin{aligned}
0 &\in \nabla_x F(\bar{x}, \bar{y}) + \mu (\nabla_x f(\bar{x}, \bar{y}) - \partial^c V(\bar{x})), \\
0 &= \nabla_y F(\bar{x}, \bar{y}) + \mu \nabla_y f(\bar{x}, \bar{y}) = \nabla_y F(\bar{x}, \bar{y}) 
\end{aligned}
$$
- 其中第二个等式是因为 $(\bar{x}, \bar{y})$ 是下层问题的最优解, 因此 $\nabla_y f(\bar{x}, \bar{y}) = 0$. 因此在该方向上, 乘子 $\mu$ 对于 $\nabla_y f(\bar{x}, \bar{y})$ 的影响是消失的. 这也就是 MFCQ 不成立的直观体现. 

### 2.4 Combined Reformulation

综上, Value function 方法和 Implicit function 方法在实践中都会各自遇到问题导致求解困难. 因此考虑将二者结合起来, 得到 combined reformulation. 其形式为:
$$
\begin{aligned}
&\min_{x, y}  \quad && F(x, y) \\
&\text{s.t.} \quad && f(x, y) - V(x) \leq 0, \\
&&& \nabla_y f(x, y) = 0. 
\end{aligned}
$$

尽管在约束层面, 对于任意的可行点 $(\bar{x}, \bar{y})$, $\nabla_y f(\bar{x}, \bar{y}) = 0$ 都是自动成立的, 然而通过额外引入的约束, 会给 KKT 系统带来新的乘子. 
- 对于不等式约束 $h_1(x, y) := f(x, y) - V(x) \leq 0$, 其对应的 KKT 乘子为 $\mu \geq 0$;
- 对于等式约束 $h_2(x, y) := \nabla_y f(x, y) = 0$, 其对应的 KKT 乘子为 $\beta$ 且无符号限制. 

因此标准的 Lagrangian 平稳条件为:
$$
0 \in \nabla F(\bar{x}, \bar{y}) + \mu \partial^c h_1(\bar{x}, \bar{y}) + \beta \nabla h_2(\bar{x}, \bar{y}).   
$$
- 其中 $h_1$ 中由于包含非光滑的 $V(x)$, 因此使用了 Clarke subgradient, 也就是刚才提到的 $\partial^c h_1(x, y) = \nabla f(x, y) - \partial^c V(x) \times \{0\}$.
- $\nabla h_2(x, y)$ 可以进一步展开为: $\nabla h_2(x, y) = \nabla [\nabla_y f(x, y)] =(\nabla^2_{xy} f(x, y), \nabla^2_{yy} f(x, y))$.

若将上述平稳条件分别按分量 $x$ 和 $y$ 展开, 则有:
$$
\begin{aligned}
0 &\in \nabla_x F(\bar{x}, \bar{y}) + \mu (\nabla_x f(\bar{x}, \bar{y}) - \partial^c V(\bar{x})) + \beta \nabla^2_{xy} f(\bar{x}, \bar{y}), \\
0 &= \nabla_y F(\bar{x}, \bar{y}) + \mu \nabla_y f(\bar{x}, \bar{y}) + \beta \nabla^2_{yy} f(\bar{x}, \bar{y}) = \nabla_y F(\bar{x}, \bar{y}) + \beta \nabla^2_{yy} f(\bar{x}, \bar{y}).
\end{aligned}
$$
- 这里发现在 $y$ 分量上, 由于引入了 $\nabla_y f(x, y) = 0$ 的约束, 因此 $\beta \nabla^2_{yy} f(\bar{x}, \bar{y})$ 不再消失, 而是对 KKT 系统产生了影响. 这也就是 combined reformulation 的优势所在. 

#### Combined Reformulation with Second-order Condition (CP-SOC)

特别指出, CP 问题的 KKT 平稳条件在 $y$ 分量上的表达式的成立
$$
0 = \nabla_y F(\bar{x}, \bar{y}) + \beta \nabla^2_{yy} f(\bar{x}, \bar{y}).
$$
事实上暗含假设 $\nabla^2_{yy} f(\bar{x}, \bar{y}) \neq 0$. 因此正如 combined reformulation 时在做的事情, 我们可以通过引入事实上恒成立的约束显式地添加到约束问题中, 在不改变原问题的可行集的情况下, 引入新的乘子, 避免 KKT 系统的退化. 

故定义 CP-SOC 问题为:
$$
\begin{aligned}
&\min_{x, y}  \quad && F(x, y) \\
&\text{s.t.} \quad && f(x, y) - V(x)\leq 0, \\
&&& \nabla_y f(x, y) = 0, \\
&&& \nabla^2_{yy} f(x, y) \in \mathbb{S}^n_{+}.
\end{aligned}
$$

通过引入 $\nabla^2_{yy} f(x, y) \in \mathbb{S}^n_{+}$ 的约束, 可以引入新的乘子, 给 KKT 系统带来新的自由度.  不难看出, 上述问题有如下推导关系:
$$
\text{KKT(VP)} \implies \text{KKT(CP)} \implies \text{KKT(CP-SOC)}.
$$


## 3. Single-level Reformulation: Lower Level Constrained

进一步考虑下层问题有约束的情况, 即下层问题 $\text{P}_x$ 的形式为:
$$
\begin{aligned}
&\min_{y}  \quad && f(x, y) \\
&\text{s.t.} \quad && g(x, y) \leq 0.
\end{aligned}
$$

约束的存在会给问题的求解带来进一步的困难. 

### 3.1 MPEC for BP with Lower Level Constraints


- Mirrlees 例子展示了当下层无约束但非凸, 一阶最优性条件只是必要但不充分时, 若直接使用 $\nabla_y f(x, y) = 0$ 来替代下层问题的最优性条件, 则可能无法得到正确的最优解. 
- 在下层问题有约束的情况下, 一阶最优性条件就变成了 KKT 条件. 因此为方便研究, 先做简化假设: 假设下层问题 $\text{P}_x$ 是凸的, 且满足 Slater 条件. 则 KKT 条件是充分必要的.  这使得我们可以先只处理由于约束带来的困难. 

给定 $x$, 写出下层优化问题的 Lagrangian:
$$
L(y, \lambda; x) := f(x, y) + \lambda^\top g(x, y)
$$
对于 $y \in S(x)$ 在凸且 Slater 满足的条件下, 等价于 KKT 条件成立, 即存在 $\lambda \geq 0$ 使得:
$$
\begin{aligned}
\nabla_y L(y, \lambda; x) &= \nabla_y f(x, y) + \nabla_y g(x, y)^\top \lambda = 0, \\
g(x, y) &\leq 0, \\
\lambda^\top g(x, y) &= 0.
\end{aligned}
$$

上述 KKT 条件可以等价压缩为:
$$
\exists \lambda \geq 0: \quad 
0 = \nabla_y f(x, y) + \nabla_y g(x, y)^\top \lambda, \quad
0 \geq g(x, y) \perp \lambda \geq 0
$$

因此既然 $y \in S(x)$ 等价于 KKT 条件成立, 则 BP 问题可以 reformulate 为如下的 Mathematical Program with Equilibrium Constraints (MPEC):
$$
\begin{aligned}
&\min_{x, y, \lambda}  \quad && F(x, y) \\
&\text{s.t.} \quad && 0 = \nabla_y f(x, y) + \nabla_y g(x, y)^\top \lambda, \\
&&& 0 \geq g(x, y) \perp \lambda \geq 0.
\end{aligned}
$$

不过这样的 reformulation 也存在相应的问题: 目前的讨论都是在下层问题凸的基础上的; 若下层非凸, 则会同样出现类似 Mirrlees 例子中的情况. 并且 MPCC 本身也存在例如, LICQ 不成立, 乘子不唯一等问题. 因此在下层问题非凸的情况下, 需要进一步的处理.

### 3.2 Value Function Reformulation under Constrained Lower Level

再次回到 value function 方法, 考虑在有约束情况下的处理. 

#### Gradient of Value Function under Unique Solution

同样先考虑 $S(x) = \{y(x)\}$ 是单点集的情况, 且假设 $y(x) \in C^1$. 故由链式法则:
$$
\nabla V(x) = \nabla_x f(x, y(x)) + \nabla_y f(x, y(x)) \nabla y(x) \tag{1}
$$
我们需要获取 $V(x)$ 的梯度以进行最优性刻画以及算法设计.

- 由于下层问题约束的存在, 上式中的 $\nabla_y f(x, y(x))$ 不再恒为零, 而是一阶 KKT 条件:
    $$
    \nabla_y f(x, y(x)) + \nabla_y g(x, y(x))^\top \lambda(x) = 0,  \tag{2}
    $$
    以及
    $$
    g(x, y(x))^\top \lambda(x) = 0 . \tag{3}
    $$

  - 对 $\text{(3)}$ 两边对 $x$ 求导, 得到:
      $$
      \begin{aligned}
      \frac{\mathrm{d}}{\mathrm{d} x} g(x, y(x))^\top \lambda(x) &= \frac{\mathrm{d}}{\mathrm{d} x} \left[g(x, y(x))\right]^\top \lambda(x) + g(x, y(x))^\top \frac{\mathrm{d}}{\mathrm{d} x} \lambda(x) \\
      &= \left[\nabla_x g(x, y(x)) + \nabla_y g(x, y(x)) \nabla y(x)\right]^\top \lambda(x) + g(x, y(x))^\top \lambda'(x) \\ &= 0 
      \end{aligned}
      $$
  - 并且注意到, 这里 $g(x, y(x))^\top \lambda'(x) \equiv 0$, 因为根据互补松弛, $g(x, y(x)) = 0$ 对于所有活跃约束 $i \in I(x, y(x))$ 都成立; 对于非活跃约束, $\lambda_i(x) = 0$, 因此 $\lambda_i'(x)$ 对应的项也为零. 故有
      $$
      0 = \nabla_x g(x, y(x))^\top \lambda(x) + \underbrace{\nabla_y g(x, y(x))^\top \lambda(x)}_{\text{By (2):}~ -\nabla_y f(x, y(x))} \nabla y(x) .
      $$
  - 而这里的下划线部分正是由一阶 KKT 条件 (2) 得到的. 因此最终得到:
  $$
  \nabla_y f(x, y(x)) \nabla y(x) = \nabla_x g(x, y(x))^\top \lambda(x).
  $$
- 最终代入 (1) 式, 得到:
    $$
    \boxed{\nabla V(x) = \nabla_x f(x, y(x)) + \nabla_x g(x, y(x))^\top \lambda(x)}
    $$

#### Sensitivity Analysis 

上述 $V(x)$ 的梯度计算依赖于两个假设: 
1. 下层问题的解集 $S(x)$ 中的元素是唯一的; 
2. 在下层问题最优解 $y \in S(x)$ 处的 KKT 乘子集合 $\text{KT}(x, y)$ 也是唯一的. 
   - 回顾: 这个性质往往需要 LICQ 等 CQ 来保证. 如果有一些约束之间是线性相关的, 则可能会出现多个 KKT 乘子对应同一个最优解的情况. 

然而在实际问题中, 这两个假设并不会自动成立. 因此需要进一步的敏感性分析来研究 $V(x)$ 的性质.

根据 Gauvin (1979) 的结果, 若下层问题 MFCQ 成立 (保证 $\text{KT}(x, y)$ 非空且 compact), 且可行域本身一致有界, 则 $V(x)$ 在 $x$ 是 Lipschitz 连续的, 并且其 Clarke subgradient 可以计算为
$$
\partial^c V(x) \subseteq \text{conv}\left\{ \bigcup_{y \in S(x), \lambda \in \text{KT}(x, y)} \left\{\nabla_x f(x, y) + \nabla_x g(x, y)^\top \lambda\right\} \right\}.
$$

然而, 在这样良定义的情况下, 将 $\partial^c V(x)$ 代入到 VP 问题中, 整个大的 VP 问题的 NNAMCQ 仍然不成立, 因此同样会出现 KKT 退化的问题. 故自然想到类似前文, 尝试通过引入新的约束来避免退化:
$$
\begin{aligned}
&\min_{x, y, u}  \quad && F(x, y) \\
&\text{s.t.} \quad && f(x, y) - V(x)\leq 0, \\
&&& \nabla_y f(x, y) + u \nabla_y g(x, y) = 0, \\
&&& g(x, y) \leq 0, \\
&&& u \geq 0, \\
&&& u^\top g(x, y) = 0.
\end{aligned} \tag{CP}
$$

然而有近期研究指出, 即使引入了新的约束, 仍然无法保证 KKT 系统的非退化性. 因此需要进一步的处理, 这就引出了 partial calmness 的概念.


## 4. Partial Calmness

直观上, partial calmness 不再通过引入新的约束来避免 KKT 系统的退化, 而是将约束 $f(x, y) - V(x) \leq 0$ 作为 penalty 放入到目标函数中:
$$
\begin{aligned}
&\min_{x, y, \mu}  \quad && F(x, y) + \mu (f(x, y) - V(x)) \\
&\text{s.t.} \quad && \nabla_y f(x, y) + u \nabla_y g(x, y) = 0, \\
&&& g(x, y) \leq 0, \\
&&& u \geq 0, \\
&&& u^\top g(x, y) = 0.
\end{aligned}
$$

可以证明, 这样的方法几乎处处成立, 即对于大多数的 BP 问题, 其 KKT 系统是非退化的. 