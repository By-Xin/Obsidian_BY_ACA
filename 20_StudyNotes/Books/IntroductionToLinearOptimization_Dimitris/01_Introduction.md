# Section 1 Introduction

## 1.1 Variants of the linear programming problem

### Problem Formulations

考虑问题标准形式:
$$
\begin{aligned}
    \text{minimize} \quad  \mathbf{c}^\top &\mathbf{x} \\
    \text{subject to} \quad  \mathbf{A} &\mathbf{x} = \mathbf{b}, \\
    & \mathbf{x} \geq \mathbf{0}.
\end{aligned}
$$
其中 $\mathbf{A}\in \mathbb{R}^{m\times n}$. 从直观上可以认为, 一共有 $n$ 种不同的资源以及 $m$ 种不同的约束条件, 其中 $\mathbf{b}$ 表示每种资源的可用量, $\mathbf{c}$ 表示每种资源的单位成本, $\mathbf{x}$ 表示每种资源的使用量. 目标是最小化总成本.

***Example* (Diet Problem)**: 假设有 $n$ 种食物, 每种食物的价格为 $c_i$, 每种食物中含有 $m$ 种营养成分, 其中第 $i$ 种食物中第 $j$ 种营养成分的含量为 $a_{ji}$, 每种营养成分的最低需求量为 $b_j$. 目标是选择每种食物的购买量 $x_i$ 以满足所有营养需求并最小化总成本. 该问题可以建模为:
$$
\begin{aligned}
    \text{minimize} \quad  \sum_{i=1}^{n} c_i &x_i \\
    \text{subject to} \quad  \sum_{i=1}^{n} a_{ji} &x_i = b_j, \quad j=1,\ldots,m, \\
    & x_i \geq 0, \quad i=1,\ldots,n.
\end{aligned}
$$

$\diamond$

任意线性规划问题都可以转化为标准形式. 考虑以下 general form:
$$
\begin{aligned}
    \text{minimize} \quad  \mathbf{c}^\top &\mathbf{x} \\
    \text{subject to} \quad  \mathbf{a}_i^\top \mathbf{x} &\geq b_i, \quad i \in M_1, \\
    \mathbf{a}_i^\top \mathbf{x} &\leq b_i, \quad i \in M_2, \\
    \mathbf{a}_i^\top \mathbf{x} &= b_i, \quad i \in M_3, \\
    x_j &\geq 0, \quad j \in N_1, \\
    x_j &\leq 0, \quad j \in N_2,
\end{aligned}
$$

从 general form 到 standard form 的转化方法如下:
- **消除自由变量**: 如果 $x_j$ 是自由变量, 则可以将其表示为两个非负变量的差值, 即 $x_j = x_j^+ - x_j^-$, 其中 $x_j^+, x_j^- \geq 0$.
- **将不等式约束转化为等式约束**: 对于形如
    $$
    \sum_{i=1}^{n} a_{ji} x_i \leq b_j,
    $$
    的不等式约束, 总可以通过引入一个松弛变量 $s_j \geq 0$ 将其转化为等式约束:
    $$
    \sum_{i=1}^{n} a_{ji} x_i + s_j = b_j, \quad s_j \geq 0.
    $$

### Examples

#### Multiperiod planning of electric power generation

某地未来 $T$ 年每年有电力需求 $d_t$. 现有旧电厂每年产能 $e_t$, 其不会增加不会减少. 政府可以在每年年年初新建两类电厂:
- 煤电厂: 第 $t$ 年新建的煤电厂产能为 $x_t$, 单位发电成本为 $c_t$, 使用寿命为 $20$ 年;
- 核电厂: 第 $t$ 年新建的核电厂产能为 $y_t$, 单位发电成本为 $n_t$, 使用寿命为 $15$ 年.

优化目标为用最小的总成本满足未来 $T$ 年的电力需求, 且核电厂的总产能不能超过总产能的 $20\%$. 



该问题可以建模如下. 

对于目标, 第 $T$ 年的总建设成本为:
$$
\sum_{t=1}^{T} c_t x_t + n_t y_t.
$$

对于约束,
- 考虑到煤电厂的使用寿命,在第 $t$ 年, 其能够积累的发电总额为过去 $20$ 年内新建的煤电厂的产能之和, 即:
    $$
    w_t = \sum_{i=\max(1, t-19)}^{t} x_i.
    $$
- 对于核电同理:
    $$
    z_t = \sum_{i=\max(1, t-14)}^{t} y_i.
    $$
- 因此要满足未来每年的电力需求, 需要满足以下约束:
    $$
    w_t + z_t + e_t \geq d_t, \quad t=1,\ldots,T.
    $$
- 为了保证核电厂的总产能不超过总产能的 $20\%$, 需要满足以下约束:
    $$
    z_t \leq 0.2 (w_t + z_t + e_t), \quad t=1,\ldots,T.
    $$
- 以及非负约束:
    $$
    x_t, y_t \geq 0, \quad t=1,\ldots,T.
    $$

综上, 该问题可以建模为如下线性规划问题:
$$
\begin{aligned}
    \text{minimize} \quad  &\sum_{t=1}^{T} c_t x_t + n_t y_t \\
    \text{subject to} \quad  & w_t - \sum_{i=\max(1, t-19)}^{t} x_i = 0, \quad t=1,\ldots,T, \\
    & z_t - \sum_{i=\max(1, t-14)}^{t} y_i = 0, \quad t=1,\ldots,T, \\
    & w_t + z_t + e_t \geq d_t, \quad t=1,\ldots,T, \\
    & 0.8 z_t - 0.2 w_t - 0.2 e_t \leq 0, \quad t=1,\ldots,T, \\
    & x_t, y_t, w_t, z_t \geq 0, \quad t=1,\ldots,T.
\end{aligned}
$$