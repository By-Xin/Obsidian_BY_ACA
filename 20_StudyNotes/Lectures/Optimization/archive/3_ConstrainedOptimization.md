# Constrained Optimization

## Overview

考虑如下约束优化问题 $P$:
\[
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m, \\
& h_j(x) = 0, \quad j = 1, \ldots, l,
\end{aligned}
\]

其中 $f: \mathbb{R}^n \to \mathbb{R}$ 是目标函数, $g_i: \mathbb{R}^n \to \mathbb{R}$ 是不等式约束函数, $h_j: \mathbb{R}^n \to \mathbb{R}$ 是等式约束函数, 且**要求都是连续可微的**. 

有一些非光滑函数可以通过引入辅助变量和等式/不等式约束来转化为上述形式, 例如:
- $f(x) = \max\{g_1(x), g_2(x)\}$ 可以转化为引入辅助变量 $t$ 并添加约束 $g_1(x) \leq t$, $g_2(x) \leq t$, 然后最小化 $t$:
\[
\begin{aligned}
\min_{x, t} \quad & t \\
\text{s.t.} \quad & g_1(x) - t \leq 0, \\
& g_2(x) - t \leq 0.
\end{aligned}
\]

## Karush-Kuhn-Tucker (KKT) 条件 (最优解的一阶必要条件)

假设 $x^*$ 是问题 $P$ 的一个局部最优解, 且满足某个**线性独立约束资格条件** (LICQ)成立, 则存在拉格朗日乘子 $\boldsymbol{\lambda}^* \in \mathbb{R}^m$ 和 $\boldsymbol{\mu}^* \in \mathbb{R}^l$, 使得以下 KKT 条件成立:
- **Stationarity (驻点条件)**:
\[\nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(x^*) + \sum_{j=1}^l \mu_j^* \nabla h_j(x^*) = 0.\]
- **Primal feasibility (原始可行性)**:
\[\begin{aligned}
& g_i(x^*) \leq 0, \quad i = 1, \ldots, m, \\
& h_j(x^*) = 0, \quad j = 1, \ldots, l.
\end{aligned}\]
- **Complementary slackness (互补松弛性)**:
\[\lambda_i^* g_i(x^*) = 0, \quad i = 1, \ldots, m.\]
- **Dual feasibility (对偶可行性)**:
\[\lambda_i^* \geq 0, \quad i = 1, \ldots, m.\]

> **优化问题的物理理解**
>
> 想象一个山谷. 目标函数 $f(x)$ 描述了山谷的高度分布. 我们的目标是找到最低点 (最小化 $f(x)$). 
> - 如果没有限制, 我们可以直接沿着重力的分量 (即梯度的反方向, $-\nabla f(x)$) 下山. 
> - 但是, 现在我们有一些障碍物 (约束条件 $g_i(x) \leq 0$ 和 $h_j(x) = 0$), 我们必须绕过这些障碍物才能找到最低点.
>   - 对于等式约束, 其定义了一些轨道或路径, 我们只能沿着这些轨道移动. 
>   - 对于不等式约束, 它们建立了一些围栏, 我们必须在围栏内或沿着围栏边界移动.
> 
> 在最优点处, 我们无法再沿着任何允许的方向下山,因为所有允许的方向都被约束所限制. 这就是 KKT 条件的物理意义: 在最优点处, 目标函数的梯度 (表示下山的方向) 可以表示为约束梯度的线性组合, 这意味着我们已经达到了一个平衡点, 既不能再下山也不能违反约束. 
> - 这可以理解为, 重力的分量被铁轨和围栏的反作用力所平衡. 故当且仅当所有约束(如铁轨等)提供的合力方向与重力方向一致时, 我们才能到达最低点. 这就是 KKT 条件中的驻点条件.
>
> 互补松弛性条件则表示, 对于每个不等式约束, 要么该约束是严格满足的 (即我们在围栏内, inactive,  $g_i(x) < 0$), 要么该约束是紧绑定的 (即我们在围栏上, active, $g_i(x) = 0$), 但不能两者兼得. 这个条件的意义在于, 如果我们在围栏内, 那么该围栏对我们的运动没有影响 (拉格朗日乘子 $\lambda_i^* = 0$); 反之, 如果我们在围栏上, 那么该围栏对我们的运动有影响 (拉格朗日乘子 $\lambda_i^* > 0$). 我们在计算最优点的驻点条件时, 只需要考虑那些紧绑定的约束, 因为只有它们对我们的运动有影响.
>
> dual feasibility 条件确保了拉格朗日乘子 $\lambda_i^*$ 是非负的, 这符合物理直觉, 因为围栏只能提供阻碍运动的力 (即反作用力), 而不能提供推动运动的力.


## Lagrange 对偶性

考虑约束优化问题 $P$ 的拉格朗日函数:
\[
L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\boldsymbol{x}) + \sum_{i=1}^m \lambda_i g_i(\boldsymbol{x}) + \sum_{j=1}^l \mu_j h_j(\boldsymbol{x}).
\]

在此基础上, 定义对偶函数:
\[
    d(\boldsymbol{\lambda}, \boldsymbol{\mu}) = \inf_{\boldsymbol{x}\in\mathcal{X}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}).
\]

---

注意到, 对于原问题而言, 对于所有可行的 $\boldsymbol{x}$, 都有:
\[ g_i(\boldsymbol{x}) \leq 0, \quad h_j(\boldsymbol{x}) = 0. \] 因此, 对于所有 $\boldsymbol{\lambda} \geq 0$, $\boldsymbol{\mu}$, 都有:
\[L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) \leq f(\boldsymbol{x}).\] 这意味着, 对于所有 $\boldsymbol{\lambda} \geq 0$, $\boldsymbol{\mu}$, 都有:
\[d(\lambda,\mu)=\inf_x L(x,\lambda,\mu)\le L(x,\lambda,\mu)\le f(x)\quad (\forall\ x\ \text{可行}).\]

因为我们想找的primal的最优值是 $p^* := \inf_{\text{可行 } x} f(x)$, 所以对于所有 $\boldsymbol{\lambda} \geq 0$, $\boldsymbol{\mu}$, 都有:
\[d(\lambda,\mu)\le \inf_{\text{可行 } x} f(x)= p^*.\]

因此, $d(\lambda,\mu)$ 是原问题的一个下界, 而 $\lambda, \mu$ 类似这个下界的生成参数. 这就是**弱对偶性** (Weak Duality). 我们还可以进一步通过优化 $\lambda, \mu$ 来得到这个下界的最佳值(尽可能贴近 $p^*$):
\[d^*:=\max_{\lambda \geq 0, \mu} d(\lambda,\mu) \le p^*.\]

这个优化问题被称为**对偶问题** (Dual Problem), 其最优值 $d^*$ 被称为**对偶最优值** (Dual Optimal Value). 如果 $d^* = p^*$, 则称为**强对偶性** (Strong Duality) 成立. $p^* - d^*$ 被称为**对偶间隙** (Duality Gap).

如果原问题 $P$ 是一个凸优化问题 (即 $f$ 是凸函数, 所有 $g_i$ 是凸函数, 所有 $h_j$ 是仿射函数), 且满足 **Slater 条件** (存在一个严格可行点, 即存在 $\boldsymbol{x}$ 使得 $g_i(\boldsymbol{x}) < 0$ 对所有 $i$ 成立, 且 $h_j(\boldsymbol{x}) = 0$ 对所有 $j$ 成立), 则强对偶性成立, 即 $d^* = p^*$.

---

接着考虑如下二者之间的关系:
\[\min_x \max_{\lambda\ge 0,\mu} L(x,\lambda,\mu)
\quad \text{vs}\quad
\max_{\lambda\ge 0,\mu}\min_x L(x,\lambda,\mu).\]

回顾 $L(x,\lambda,\mu)$ 的定义:
\[L(x,\lambda,\mu)=f(x)+\sum_{i=1}^m \lambda_i g_i(x)+\sum_{j=1}^l \mu_j h_j(x).\]


对于 primal $\min_x\max_{\lambda,\mu} L$, 首先看内部, 将其看作 $x$ 是给定的, 则 $L(x,\lambda,\mu)$ 是 $\lambda,\mu$ 的线性函数. 
- 如果 $x$ 违反了某个不等式约束 (即存在某个 $i$ 使得 $g_i(x) > 0$), 则可以通过让对应的 $\lambda_i \to +\infty$ 来使得 $L(x,\lambda,\mu) \to +\infty$.
- 如果 $x$ 违背了某个等式约束 (即存在某个 $j$ 使得 $h_j(x) \neq 0$), 则可以通过让对应的 $\mu_j \to +\infty$ 或 $\mu_j \to -\infty$ 来使得 $L(x,\lambda,\mu) \to +\infty$. 因此, 对于任何不可行的 $x$, 都有:
\[\max_{\lambda\ge 0,\mu} L(x,\lambda,\mu)= +\infty.\]

- 如果 $x$ 是可行的, 则对于所有的 $g_i(x) \leq 0$, 我们要确保 $\lambda_i = 0$ 才能确保最大. 同理, 对于所有的 $h_j(x) = 0$, $\mu_j$ 可以取任意值, 但不会影响最大值. 因此, 对于任何可行的 $x$, 都有:
\[\max_{\lambda\ge 0,\mu} L(x,\lambda,\mu)= f(x).\]
- 因此, 再看外层的 $\min_x$, 则有:
\[\min_x \max_{\lambda\ge 0,\mu} L(x,\lambda,\mu)
= \min_{\text{可行 } x} f(x) = p^*.\]
即对于不可行的点, 其对应的值为 $+\infty$, 直接排除. 我们只需在可行域内寻找最小值, 即得到原问题的最优值.

对于 dual $\max_{\lambda\ge 0,\mu}\min_x L(x,\lambda,\mu)$, 其相当于是先给出 Lagrangian 的下界, 然后再在这些下界中寻找最大的一个. 这就是我们前面提到的对偶问题, 其最优值为 $d^*$.