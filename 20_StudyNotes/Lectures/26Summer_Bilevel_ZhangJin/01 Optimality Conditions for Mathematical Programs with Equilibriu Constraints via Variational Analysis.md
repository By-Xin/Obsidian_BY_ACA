# Optimality Conditions for Mathematical Programs with Equilibrium Constraints via Variational Analysis

> Speaker: Zhang Jin (Southern University of Science and Technology, China)
>
> Date: July 06, 2026

## 1. Introduction

本 talk 将从普通非线性规划问题的 KKT 条件出发, 逐步扩展介绍什么是 MPCC / MPEC 以及这类问题中的最优性条件, 以及如何利用变分分析的工具来刻画这些条件.

### 1.1 Recap: Nonlinear Programming (NLP)

#### Standard NLP

给定标准 NLP:
$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} & \quad f(\mathbf{x}) \\
\text{s.t.} & \quad h_i(\mathbf{x}) = 0, \quad i = 1, \ldots, p \\
& \quad g_j(\mathbf{x}) \leq 0, \quad j = 1, \ldots, q
\end{aligned}
$$
其中, $f, h_i, g_j: \mathbb{R}^n \to \mathbb{R}$ 是连续可微的. 

#### KKT Conditions

对于候选的局部最优解 $\bar{\mathbf{x}}$, 通常考虑其 KKT 条件:
$$
\begin{aligned}
\nabla f(\bar{\mathbf{x}}) + \sum_{i=1}^p \lambda_i^h \nabla h_i(\bar{\mathbf{x}}) + \sum_{j=1}^q \lambda_j^g \nabla g_j(\bar{\mathbf{x}}) = 0, \\ 
\lambda_j^g g_j(\bar{\mathbf{x}}) = 0, \quad j = 1, \ldots, q, \\
\lambda_j^g \geq 0, \quad j = 1, \ldots, q.
\end{aligned}
$$

若进一步记 active constraints 的集合为 $\mathcal{I}_g(\bar{\mathbf{x}}) = \{ j \in \{1, \ldots, q\} : g_j(\bar{\mathbf{x}}) = 0 \}$, 则 KKT 条件可以简化为:
$$
\begin{aligned}
\nabla f(\bar{\mathbf{x}}) + \sum_{i=1}^p \lambda_i^h \nabla h_i(\bar{\mathbf{x}}) + \sum_{j \in \mathcal{I}_g(\bar{\mathbf{x}})} \lambda_j^g \nabla g_j(\bar{\mathbf{x}}) = 0, \\ 
\boldsymbol{\lambda}^g \geq 0.
\end{aligned}
$$

**强调, 对于 NLP, KKT 需要在一些 *constraint qualification (CQ)* 下才是局部最优解的必要条件**. 并且后文将看到, 在 MPCC / MPEC 等问题中, 传统的 CQ 往往不再适用, 需要引入新的概念来刻画最优性条件.

### 1.2 Fritz John Conditions and Several Constraint Qualifications 

#### Fritz John Conditions

除了 KKT 条件之外, 还有一个更为一般的一阶必要条件, 即 **Fritz John 条件**. 其与 KKT 条件的区别在于, Fritz John 条件不需要任何 CQ 的假设, 但其引入了一个额外的非负标量 $\lambda_0$ 来刻画最优性条件.

***Proposition* (Fritz John Conditions [Fritz John, 1948])**: 对于上述光滑有限维 NLP 中, 任意局部最优解 $\bar{\mathbf{x}}$, 均存在一组 **不全为 $0$** 的 $\lambda_0 \geq 0$, $\lambda_i^h \in \mathbb{R}$, $\lambda_j^g \geq 0$, 使得
$$
\lambda_0 \nabla f(\bar{\mathbf{x}}) + \sum_{i=1}^p \lambda_i^h \nabla h_i(\bar{\mathbf{x}}) + \sum_{j \in \mathcal{I}_g(\bar{\mathbf{x}})} \lambda_j^g \nabla g_j(\bar{\mathbf{x}}) = 0.
$$
- 若 $\lambda_0 > 0$, 则可以将 FJ 条件左右两侧同时除以 $\lambda_0$, 从而得到 KKT 条件.
- 若 $\lambda_0 = 0$, 则称这种情况为 **abnormal case**, 对应的非零的 $(\boldsymbol{\lambda}^g, \boldsymbol{\lambda}^h)$ 称为 **abnormal multiplier**.
  - 在这种情况下, 一阶的信息退化掉了, 目标函数本身的梯度信息无法提供任何有用的最优性条件. 
  - 这也事实上就是 KKT 条件中所依赖的 CQ 在本质上想要排除的情况.

#### No Nonzero Abnormal Multiplier Constraint Qualification (NNAMCQ / PLICQ)

因此, 这里总结了如下 constraint qualification, 称为 **NNAMCQ (No Nonzero Abnormal Multiplier Constraint Qualification)** (或等价地, **PLICQ (Positive Linear Independence Constraint Qualification)**). 简单地讲, 该 CQ 的作用就是排除 abnormal case 的发生.

***Definition* (NNAMCQ / PLICQ)**: 对于上述 NLP, 称 $\bar{\mathbf{x}}$ 满足 NNAMCQ / PLICQ, 若不存在 $(\boldsymbol{\lambda}^g, \boldsymbol{\lambda}^h) \neq 0$ 同时满足 $\boldsymbol{\lambda}^g \geq 0$ 且
$$
\sum_{i=1}^p \lambda_i^h \nabla h_i(\bar{\mathbf{x}}) + \sum_{j \in \mathcal{I}_g(\bar{\mathbf{x}})} \lambda_j^g \nabla g_j(\bar{\mathbf{x}}) = 0.
$$
或等价地考虑其逆否命题: 若在 $\bar{\mathbf{x}}$ 处, 对于任意满足如下条件的 $(\boldsymbol{\lambda}^g, \boldsymbol{\lambda}^h)$:
$$
\begin{cases}~
\sum_{i=1}^p \lambda_i^h \nabla h_i(\bar{\mathbf{x}}) + \sum_{j \in \mathcal{I}_g(\bar{\mathbf{x}})} \lambda_j^g \nabla g_j(\bar{\mathbf{x}}) = 0, \\~
\boldsymbol{\lambda}^g \geq 0,
\end{cases}
$$
均能够推出 $(\boldsymbol{\lambda}^g, \boldsymbol{\lambda}^h) = 0$, 则称 $\bar{\mathbf{x}}$ 满足 NNAMCQ / PLICQ.

$\diamond$

- 直观地, NNAMCQ / PLICQ 的几何意义是: 不存在非零的合法 multiplier $(\boldsymbol{\lambda}^g, \boldsymbol{\lambda}^h)$ 是的约束梯度的非负线性组合为零向量. 也就是说, 在 $\bar{\mathbf{x}}$ 处, 等式约束的梯度和 active 不等式约束的梯度之间没有任何非零的线性依赖关系.

#### Mangasarian-Fromovitz Constraint Qualification (MFCQ)

借助 Motzkin's Theorem, 可以证明 NNAMCQ / PLICQ 等价于 MFCQ (Mangasarian-Fromovitz Constraint Qualification). 具体地, MFCQ 的定义如下. 

***Definition* (MFCQ)**: 对于上述 NLP, 称 $\bar{\mathbf{x}}$ 满足 MFCQ, 若满足:
1. 等式约束的梯度向量之间 $\{ \nabla h_i(\bar{\mathbf{x}}) \}_{i=1}^p$ 是线性无关的;
2. 存在一个方向 $\mathbf{d} \in \mathbb{R}^n$ 使得
    $$
    \begin{cases}
    \nabla h_i(\bar{\mathbf{x}})^\top \mathbf{d} = 0, & i = 1, \ldots, p, \\
    \nabla g_j(\bar{\mathbf{x}})^\top \mathbf{d} < 0, & j \in \mathcal{I}_g(\bar{\mathbf{x}}).
    \end{cases}
    $$

$\diamond$

- 直观地, MFCQ 的几何意义是: 存在一个 "好" 方向 $\mathbf{d}$, 使得沿着该方向, 等式约束保持不变, 而所有 active 不等式约束都严格减小. 反过来, 若找不到任何一个这样的方向 $\mathbf{d}$, 则意味着存在一个非零的合法 multiplier $(\boldsymbol{\lambda}^g, \boldsymbol{\lambda}^h)$ 使得约束梯度的非负线性组合为零, 从而违反了 NNAMCQ / PLICQ.

#### Relationship between CQs

总结一下, 对于这些 CQ, 有如下推导关系:
- LICQ (即等式约束和 active 不等式约束的梯度线性无关) 可以推出 MFCQ;
- Slater 条件 (在凸优化中, 对于凸约束, 存在一个严格可行点) 可以推出 MFCQ;
- MFCQ 等价于 NNAMCQ / PLICQ;
- MFCQ / NNAMCQ / PLICQ 可以推出 FJ 条件中的 $\lambda_0 > 0$. 
- FJ 条件中的 $\lambda_0 > 0$ 等价于 KKT 条件成立.

## 2. Mathematical Programs with Complementarity Constraints (MPCC)

### 2.1 Definition of MPCC

首先先给出 MPCC 的标准形式:
$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} & \quad f(\mathbf{x}) \\
\text{s.t.} & \quad \mathbf{G}(\mathbf{x}) \leq \mathbf{0}, \\
& \quad \mathbf{H}(\mathbf{x}) \leq \mathbf{0}, \\
& \quad  \mathbf{G}(\mathbf{x})^\top \mathbf{H}(\mathbf{x}) = 0,
\end{aligned}
$$
其中 $\mathbf{G}, \mathbf{H}: \mathbb{R}^n \to \mathbb{R}^m$, $\mathbf{G}(\mathbf{x})^\top \mathbf{H}(\mathbf{x}) = \sum_{i=1}^m G_i(\mathbf{x}) H_i(\mathbf{x})$.  
- 并且立即可以推得, 由于 $\mathbf{G}_i(\mathbf{x}) \leq 0, \mathbf{H}_i(\mathbf{x}) \leq 0$, 因此 $\mathbf{G}_i(\mathbf{x})^\top \mathbf{H}_i(\mathbf{x}) = 0$ 等价于 $\mathbf{G}_i(\mathbf{x}) = 0$ 或 $\mathbf{H}_i(\mathbf{x}) = 0$. 因此也称这里的约束为 **complementarity constraints**.

### 2.2 Game Theoretic Motivation for MPCC

下面将从博弈论角度解释, 许多均衡博弈等问题都会自然产生 MPCC 形式的互补约束. 并且在其中还能看到暗含的 bilevel 结构. 

#### Nash Game

考虑两个玩家的 Nash equilibrium. 称 $(\bar{\mathbf{x}}, \bar{\mathbf{y}})$ 是 Nash equilibrium, 若同时优化:
$$
\bar{\mathbf{x}} \in \arg\min_{\mathbf{x}} f_1(\mathbf{x}, \bar{\mathbf{y}}), \quad
\bar{\mathbf{y}} \in \arg\min_{\mathbf{y}} f_2(\bar{\mathbf{x}}, \mathbf{y}).   
$$

#### Stackelberg Game

Stackelberg game 是一种 leader-follower 的博弈, 其博弈双方并不是平等对称的, 而具有先后的行动顺序: 
1. Leader 会先决定 $\mathbf{x}$ 并且是公开的;
2. Follower 会在观察到 $\mathbf{x}$ 后, 决定 $\mathbf{y}$ 来使得其自己的目标 $g(\mathbf{x}, \mathbf{y})$ 最小:
    $$
    \mathbf{y} \in \arg\min_{\mathbf{y}'} g(\mathbf{x}, \mathbf{y}').
    $$
3. Leader 在选择 $\mathbf{x}$ 之前就会预判 follower 的反应, 并且假设 follower 会选择最优的 $\mathbf{y}$ 来使得其自己的目标 $g(\mathbf{x}, \mathbf{y})$ 最小. 因此 leader 会在这个前提下优化自己的目标 $f$. 

因此, 整个 Stackelberg 均衡可以写作如下 bilevel 的形式:
$$
\begin{aligned}
\min_{\mathbf{x}, \mathbf{y}} & \quad f(\mathbf{x}, \mathbf{y}) \\
\text{s.t.} & \quad \mathbf{y} \in \arg\min_{\mathbf{y}'} g(\mathbf{x}, \mathbf{y}').
\end{aligned}
$$

该问题也是 MPCC 的根本动机来源. 

#### Cournot-Nash

Cournot-Nash 是一个多人博弈结构. 这里假设 $N$ 个企业同时决定各自产量 $q_i$, 并且市场价格 $p(\cdot)$ 是总产量 $\sum_{i=1}^N q_i$ 的函数, 且假设产量越高, 价格越低. 对应每个企业 $i$ 的目标都是最大化自己的利润 = 收入 - 成本. 因此 Cournot-Nash 的均衡可以写作如下形式:
$$
\begin{aligned}
     q_i^* \in \arg\max & \quad q_i \cdot p\left(\sum_{j\neq i} q_j^* + q_i\right) - c_i(q_i) \\
        \text{s.t.} & \quad q_i \geq 0, \quad i = 1, \ldots, N.
\end{aligned}
$$

并且对于企业 $i$, 其面对的 KKT 条件为:
$$
0 = p\left(\sum_{j\neq i} q_j^* + q_i^*\right) + q_i^* p'\left(\sum_{j\neq i} q_j^* + q_i^*\right) - c_i'(q_i^*) - \eta_i := F_i(q^*) - \eta_i, \\
\eta_i \geq 0, \quad q_i^* \geq 0, \quad \eta_i q_i^* = 0.
$$
若将全部企业的 KKT 整理到一起, 写作向量形式, 则有:
$$
F(\mathbf{q}) \geq 0, \quad \mathbf{q} \geq 0, \quad F(\mathbf{q})^\top \mathbf{q} = 0,
$$
这是事实上就是后面讨论的 MPCC 的互补形式. 当然具体的形式在后面给出. 

#### Stackelberg-Cournot-Nash

若再 Cournot-Nash 的 $N$ 个竞争企业的基础上, 增加一个 leader (原有企业为 follower), 则可以得到一个 Stackelberg-Cournot-Nash 的博弈结构: Leader 先决定自己的产量 $x$; 并且能够预判 $N$ 个 follower 在看到 $x$ 后会如何调整均衡产量 $q$, 并且 accordingly 地调整自己的产量 $x$ 来使得自己的利润最大化. 该问题可以写作如下 bilevel 的形式:
$$
\begin{aligned} 
\max_{x, q} & \quad x \cdot p(x + \sum_{i=1}^N q_i) - c_{N+1}(x) \\
\text{s.t.} & \quad F(q+x) \geq 0, \quad q \geq 0, \quad F(q+x)^\top q = 0, \quad x \geq 0.
\end{aligned}
$$

## 3. Variational Analysis and Optimality Conditions for MPCC

### 3.1 Difficulty of Solving MPCC and Reformulation

#### PLICQ Never Holds for MPCC

对于 MPCC, 一个非常棘手的问题在于: 如果我们将其视作一个普通的 NLP, 则可以证明上文的 MFCQ 等经典 CQ 条件在任意可行点处都不成立. 

*Proof*. 设 $\bar{\mathbf{x}}$ 是任意一个 MPCC 的可行点, 故 $\mathbf{G}(\bar{\mathbf{x}}) \leq 0, \mathbf{H}(\bar{\mathbf{x}}) \leq 0, \mathbf{G}(\bar{\mathbf{x}})^\top \mathbf{H}(\bar{\mathbf{x}}) = 0$. 
- 构造下列辅助问题, 不难看出 $\bar{\mathbf{x}}$ 是该辅助问题的全局最优解:
    $$
    \min_{\mathbf{x} \in \mathbb{R}^n} \quad \phi(\mathbf{x}) := \mathbf{G}(\mathbf{x})^\top \mathbf{H}(\mathbf{x}) \quad \text{s.t.} \quad \mathbf{G}(\mathbf{x}) \leq 0, \mathbf{H}(\mathbf{x}) \leq 0.
    $$
    因为 $\mathbf{G}_i(\bar{\mathbf{x}}) \leq 0, \mathbf{H}_i(\bar{\mathbf{x}}) \leq 0$, 因此最小值为 $\phi(\bar{\mathbf{x}}) = 0$. 并且 $\phi(\mathbf{x}) \geq 0$ 对于所有可行点 $\mathbf{x}$ 都成立. 因此 $\bar{\mathbf{x}}$ 是该辅助问题的最优解.

- 对这个辅助问题用 Fritz John 条件, 则存在不全为 $0$ 的 $\lambda_0, \boldsymbol{\lambda}^G, \boldsymbol{\lambda}^H \geq 0$ 使得
    $$
    \lambda_0 \nabla \phi(\bar{\mathbf{x}}) + \sum_{i \in \mathcal{I}_G(\bar{\mathbf{x}})} \lambda_i^G \nabla G_i(\bar{\mathbf{x}}) + \sum_{i \in \mathcal{I}_H(\bar{\mathbf{x}})} \lambda_i^H \nabla H_i(\bar{\mathbf{x}}) = 0, \quad \dagger
    $$
    其中 $\mathcal{I}_G(\bar{\mathbf{x}}) = \{ i : G_i(\bar{\mathbf{x}}) = 0 \}$, $\mathcal{I}_H(\bar{\mathbf{x}}) = \{ i : H_i(\bar{\mathbf{x}}) = 0 \}$.

- 而 $\dagger$ 就给出了 PLICQ 的反例. 因为, 对于 MPCC-as-NLP, 其约束分别为 $\mathbf{G}(\mathbf{x}) \leq 0, \mathbf{H}(\mathbf{x}) \leq 0, \phi(\mathbf{x}) = 0$. 因此, PLICQ 要求, 若
    $$
    \eta \nabla \phi(\bar{\mathbf{x}}) + \sum_{i \in \mathcal{I}_G(\bar{\mathbf{x}})} \mu_i^G \nabla G_i(\bar{\mathbf{x}}) + \sum_{i \in \mathcal{I}_H(\bar{\mathbf{x}})} \mu_i^H \nabla H_i(\bar{\mathbf{x}}) = 0, \quad \boldsymbol{\mu}^G, \boldsymbol{\mu}^H \geq 0,
    $$
    则必须有 $\eta = 0, \boldsymbol{\mu}^G = 0, \boldsymbol{\mu}^H = 0$. 而 $\dagger$ 给出了一个反例, 因此 PLICQ 不成立. 由于 MFCQ 等价于 PLICQ, 因此 MFCQ 也不成立.

$\square$

#### Reformulation of MPCC

传统的分析方法全部失效, 需要引入新的分析工具, 也就是 **Variational Analysis** 来刻画 MPCC 的最优性条件. 因此这里也不再将其当作普通 NLP 来接, 而是将其 reform. 

对于原先的互补约束:
$$
\mathbf{G}(\mathbf{x}) \leq 0, \quad \mathbf{H}(\mathbf{x}) \leq 0, \quad \mathbf{G}(\mathbf{x})^\top \mathbf{H}(\mathbf{x}) = 0,
$$
事实上就是说, 对于每一个 $i = 1, \ldots, m$, 都有 $G_i(\mathbf{x}) = 0$ 或 $H_i(\mathbf{x}) = 0$. 即, 至少有一个约束必须在边界上. 

因此, 一方面可以将其写成如下 geometric constraint:
$$
\boldsymbol{\Phi} (\mathbf{x}) :=
(\mathbf{G}(\mathbf{x}), \mathbf{H}(\mathbf{x})) \in \Omega_C := \{ (\mathbf{y}, \mathbf{z}) \in \mathbb{R}_{\leq 0}^m \times \mathbb{R}_{\leq 0}^m \mid \mathbf{y} \perp \mathbf{z} \}.
$$
- 其中 $\perp$ 表示互补约束, 即 $\mathbf{y}\perp\mathbf{z}  \iff y_i z_i = 0, ~i=1, \ldots, m$. 
- $\boldsymbol{\Phi} (\mathbf{x})$ 是将 $\mathbf{G}(\mathbf{x})$ 和 $\mathbf{H}(\mathbf{x})$ 拼接成一个 $2m$ 维向量的映射.

这里, $\Omega_C$ 的几何结构比较特殊. 例如在二维空间中, $\Omega_C = (\mathbb{R}_{\leq 0} \times \{0\}) \cup (\{0\} \times \mathbb{R}_{\leq 0})$, 即两个半轴的并集. 下面给出 $\Omega_C$ 的二维几何图示:

![20260707105532](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260707105532.png)

这样的几何既不是凸集, 也不是光滑流形, 因此也是普通 NLP 失效的原因, 也是变分分析工具能够发挥作用的地方. 

另一方面还可以考虑将互补约束 reform 为
$$
\max\{\mathbf{G}(\mathbf{x}), \mathbf{H}(\mathbf{x})\} = 0,
$$
其中 $\max$ 是逐元素的最大值. 然而函数 $\max\{\cdot, \cdot\}$ 是非光滑的, 因此也无法直接使用传统的 NLP 分析方法.

### 3.2 Variational Analysis Tools 1: Normal Cone

Variational Analysis 是一门研究非凸/非光滑分析的领域分支, 其主要工具包括 **tangent cone**, **normal cone**, **subdifferential**, **coderivative** 等. 这些工具可以帮助我们刻画非光滑约束的几何结构, 从而推导出 MPCC 的最优性条件.

前文的分析指出, MPCC 问题常见的 CQ 都不成立, 因此有两种 reformulation 的方法. 其中之一是将互补约束 reform 为 geometric constraint $\boldsymbol{\Phi}(\mathbf{x}) \in \Omega_C$. 然而, $\Omega_C$ 是非凸的, 因此无法直接使用传统的 KKT 条件. 这里将引入 **normal cone** 的概念来刻画 $\Omega_C$ 的几何结构, 从而推导出 MPCC 的最优性条件.

#### Motivation: Normal Cone for Convex Set and Optimality Conditions

先给出一般的在凸集上 normal cone 的定义.

***Definition* (Normal Cone for Convex Set)**: 对于一个 closed and convex 集合 $C$, 在 $\bar{\mathbf{x}} \in C$ 处的 normal cone 定义为:
$$
\mathcal{N}_C(\bar{\mathbf{x}}) := \{ \mathbf{w} : \langle \mathbf{w}, \mathbf{x} - \bar{\mathbf{x}} \rangle \leq 0, ~\forall \mathbf{x} \in C
 \}.
$$
- 即, normal cone 中的向量 $\mathbf{w}$ 是指向集合 $C$ 外部的向量, 并且与集合 $C$ 中任意点 $\mathbf{x}$ 的连线都形成一个钝角.

作为导入, 先考虑一个简单的优化问题: $\min f(x), ~\text{s.t.}~ x \in C$, 其中 $C$ 是 closed and convex 集合. 由于约束的存在, 此时 $\bar{x}$ 往往仍具有下降的梯度方向, 但是不可行的. 故, 最优解的一阶条件应为: $-\nabla f(\bar{x})$ 是 $C$ 在 $\bar{x}$ 处的一个法向量, 即 $-\nabla f(\bar{x}) \in \mathcal{N}_C(\bar{x})$. 或等价地, $0 \in \nabla f(\bar{x}) + \mathcal{N}_C(\bar{x})$. 

再考虑更一般的 geometric constraint 问题:
$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} & \quad \boldsymbol{f}(\mathbf{x}) \\
\text{s.t.} & \quad \boldsymbol{\Phi}(\mathbf{x}) \in C,
\end{aligned}
$$
- 其中 $\boldsymbol{f}, \boldsymbol{\Phi}: \mathbb{R}^n \to \mathbb{R}^m$ 属于 $C^1$, 且集合 $C$ 是 closed and convex. 


Accordingly, 最优性 KKT 条件也可以用 normal cone 来刻画. 对于 $\bar{\mathbf{x}}$ 是该问题的局部最优解, 则存在 $\boldsymbol{\lambda^\Phi} \in \mathcal{N}_C(\boldsymbol{\Phi}(\bar{\mathbf{x}}))$ 使得
$$
\mathbf{0} = \nabla \boldsymbol{f}(\bar{\mathbf{x}}) + \nabla \boldsymbol{\Phi}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda^\Phi}.
$$
- 其中 $\boldsymbol{\lambda^\Phi} \in \mathbb{R}^m$ 是 Lagrange multiplier (这里的上标只是一个标记, 表示它是对应于 $\boldsymbol{\Phi}$ 的 multiplier). $\boldsymbol{\lambda^\Phi} \in \mathcal{N}_C(\boldsymbol{\Phi}(\bar{\mathbf{x}}))$ 表示 $\boldsymbol{\lambda^\Phi}$ 是集合 $C$ 在 $\boldsymbol{\Phi}(\bar{\mathbf{x}})$ 处的一个法向量. 
- 常见的等式约束与不等式约束都可以用 normal cone 来刻画, 例如:
    - 对于等式约束问题 
        $$
        \min_{\mathbf{x}\in\mathbb{R}^n} f(\mathbf{x}), ~\text{s.t.}~ h(\mathbf{x}) = 0
        $$
        则 $\Phi(\mathbf{x}) = h(\mathbf{x})$, $C = \{0\}$, 因此 $\mathcal{N}_C(0) = \mathbb{R}^m$, 从而 KKT 条件为 $\nabla f(\bar{\mathbf{x}}) + \nabla h(\bar{\mathbf{x}})^\top \lambda^h = 0$. 
    - 对于不等式约束问题 
        $$
        \min_{\mathbf{x}\in\mathbb{R}^n} f(\mathbf{x}), ~\text{s.t.}~ g(\mathbf{x}) \leq 0
        $$
        则 $\Phi(\mathbf{x}) = g(\mathbf{x})$, $C = \mathbb{R}_{\leq 0}^m$. 下考虑求解 $\mathcal{N}_{\mathbb{R}_{\leq 0}^m}(\bar{\mathbf{y}})$, 其中 $\bar{\mathbf{y}} = g(\bar{\mathbf{x}}) \leq 0$. 
      - 先考虑一维的简化情况. 若 $\bar{y} < 0$, 则根据定义, $\mathcal{N}_{\mathbb{R}_{\leq 0}}(\bar{y}) = \{w\in \mathbb{R}: w(y - \bar{y}) \leq 0, ~\forall y \leq 0\} = \{0\}$. 若 $\bar{y} = 0$, 则 $\mathcal{N}_{\mathbb{R}_{\leq 0}}(0) = \{w\in \mathbb{R}: w(y - 0) \leq 0, ~\forall y \leq 0\} = \mathbb{R}_{\geq 0}$. 因此, 一维情况下, 
        $$
        \mathcal{N}_{\mathbb{R}_{\leq 0}}(\bar{y}) = \begin{cases} \{0\}, & \bar{y} < 0 \\ \mathbb{R}_{\geq 0}, & \bar{y} = 0 \end{cases}.
        $$
      - 拓展到 $m$ 维, 其中 $C = \mathbb{R}_{\leq 0}^m$ 由于是一个 product set, 因此 $\mathcal{N}_{\mathbb{R}_{\leq 0}^m}(\bar{\mathbf{y}}) = \prod_{i=1}^m \mathcal{N}_{\mathbb{R}_{\leq 0}}(\bar{y}_i) = \prod_{i=1}^m \begin{cases} \{0\}, & \bar{y}_i < 0 \\ \mathbb{R}_{\geq 0}, & \bar{y}_i = 0 \end{cases}$.
      - $\boldsymbol{\lambda^\Phi} \in \mathcal{N}_{\mathbb{R}_{\leq 0}^m}(\bar{\mathbf{y}})$ 就是 complementary slackness condition 的另一种刻画方式, 即 $\lambda_i^\Phi \geq 0, ~\lambda_i^\Phi \bar{y}_i = 0$.

#### Normal Cone for Nonconvex Set

对于凸集的 normal cone 是较为简单的. 但是对于非凸集合, 并不是总能找到一个良定义的法向方向. 这里将给出三种不同的 normal cone 的定义, 分别是 **regular normal cone**, **limiting normal cone**, **Clarke normal cone**. 

***Definition* (Regular Normal Cone)**: 对于一个 closed nonempty 集合 $C$, 给定 $\bar{\mathbf{x}} \in C$, 其 regular normal cone 定义为:
$$
\widehat{\mathcal{N}}_C(\bar{\mathbf{x}}) := \left\{ \mathbf{w} : \langle \mathbf{w}, \mathbf{x} - \bar{\mathbf{x}} \rangle \leq o(\|\mathbf{x} - \bar{\mathbf{x}}\|), ~\forall \mathbf{x} \in C \right\}.
$$
或更严谨地
$$
\widehat{\mathcal{N}}_C(\bar{\mathbf{x}}) := \left\{ \mathbf{w} : \limsup_{\mathbf{x} \to \bar{\mathbf{x}}, \mathbf{x} \in C, \mathbf{x} \neq \bar{\mathbf{x}}} \frac{\langle \mathbf{w}, \mathbf{x} - \bar{\mathbf{x}} \rangle}{\|\mathbf{x} - \bar{\mathbf{x}}\|} \leq 0 \right\}.
$$
- 该定义可以粗略理解为: $\langle \mathbf{w}, \mathbf{x} - \bar{\mathbf{x}} \rangle \lesssim 0$ 在一阶近似意义下成立 (对比凸集中 normal cone 的定义, 其要求 $\langle \mathbf{w}, \mathbf{x} - \bar{\mathbf{x}} \rangle \leq 0$ 对所有 $\mathbf{x} \in C$ 都成立).
- 上述极限过程还可以理解如下. 令 $R(\mathbf{x}) := \langle \mathbf{w}, \mathbf{x} - \bar{\mathbf{x}} \rangle / \|\mathbf{x} - \bar{\mathbf{x}}\| = \|\mathbf{w}\| \cos \theta$, 其中 $\theta$ 是 $\mathbf{w}$ 与 $\mathbf{x} - \bar{\mathbf{x}}$ 的夹角. 故 regular normal cone 的定义等价于要求 $\lim\sup_{\mathbf{x} \to \bar{\mathbf{x}}, \mathbf{x} \in C, \mathbf{x} \neq \bar{\mathbf{x}}} \cos \theta \leq 0$, 即 $\theta$ 在一阶近似意义下大于等于 $\pi/2$ (朝向集合外部). 而这里采用 limsup 而不是 lim, 是需要考虑到 $\mathbf{x}$ 可能从不同的方向趋近 $\bar{\mathbf{x}}$, 因此需要考虑所有可能的方向, 并取最坏情况 (最有可能朝向内部的方向, 即 limsup).

*Example.* 考虑刚才 MPCC 中的二维简化版本中的互补锥集合 $\Omega_C = \{(y, z) \in \mathbb{R}^2: y \leq 0, z\leq 0, yz = 0\}$ (相当于坐标轴中的两条非正半轴之并集). 记 $(u,v) \in \widehat{\mathcal{N}}_{\Omega_C}(\bar{\mathbf{x}})$, 则有如下几种情况:
1. 若 $\bar{y} < 0, \bar{z} = 0$ (对应于 $y$ 轴的负半轴), 则对应的法向向量 $(u,v)$ 必须满足 $u = 0$ 而 $v$ 任意; 反之对于 $\bar{y} = 0, \bar{z} < 0$ (对应于 $z$ 轴的负半轴), 则对应的法向向量 $(u,v)$ 必须满足 $v = 0$ 而 $u$ 任意.
2. 若 $\bar{y} = 0, \bar{z} = 0$ (对应于原点), 则对应的法向向量 $(u,v)$ 必须满足 $u \leq 0, v \leq 0$. 也就是说, regular normal cone 在原点处是一个闭合的第一象限的集合. 

故
$$
\widehat{\mathcal{N}}_{\Omega_C}(\bar{\mathbf{x}}) = \begin{cases}
\{(0, v): v \in \mathbb{R}\}, & \bar{y} < 0, \bar{z} = 0, \\
\{(u, 0): u \in \mathbb{R}\}, & \bar{y} = 0, \bar{z} < 0, \\
\mathbb{R}_{\geq 0}^2, & \bar{y} = 0, \bar{z} = 0.
\end{cases}
$$

***Definition* (Limiting / Mordukhovich Normal Cone)**: 对于一个 closed nonempty 集合 $C$, 给定 $\bar{\mathbf{x}} \in C$, 其 limiting normal cone 定义为:
$$
\mathcal{N}^{\text{M}}_C(\bar{\mathbf{x}}) := \left\{
    \mathbf{w}: \exists \mathbf{x}^k \in C, \mathbf{x}^k \to \bar{\mathbf{x}}, \exists \mathbf{w}^k \to \mathbf{w}, \mathbf{w}^k \in \widehat{\mathcal{N}}_C(\mathbf{x}^k) 
\right\}. 
$$
- 该定义可以理解如下: $C$ 的 limiting normal cone 中的向量 $\mathbf{w}$ 是指, 首先确定集合中国年的一个趋近于 $\bar{\mathbf{x}}$ 的点列 $\{\mathbf{x}^k\}$, 对该点列的每个点 $\mathbf{x}^k$, 都能确定一个 regular normal cone 中的向量 $\mathbf{w}^k \in \widehat{\mathcal{N}}_C(\mathbf{x}^k)$, 并且该向量列 $\{\mathbf{w}^k\}$ 收敛到 $\mathbf{w}$. 也就是说, limiting normal cone 中的向量 $\mathbf{w}$ 是由 regular normal cone 中的向量列极限得到的.

*Example*. 继续考虑刚才 MPCC 中的二维简化版本中的互补锥集合 $\Omega_C = \{(y, z) \in \mathbb{R}^2: y \leq 0, z\leq 0, yz = 0\}$ (相当于坐标轴中的两条非正半轴之并集). 记 $(u,v) \in \mathcal{N}^{\text{M}}_{\Omega_C}(\bar{\mathbf{x}})$. 
- 则除了 regular normal cone 中的情况之外, $\mathcal{N}^{\text{M}}_{\Omega_C}(\bar{\mathbf{x}})$ 还额外包含了横纵坐标轴的负半轴的向量. 
- 这是因为, 考虑点列 $(y^k, z^k) = (-1/k, 0) \to (0, 0)$, 从横轴负半轴接近原点, 对于每一个点其 regular normal 都是整个竖直方向 $\{(0, v): v \in \mathbb{R}\}$. 因此当去其极限时, 则得到 $\{(0, v): v \in \mathbb{R}\}$ 也包含在 limiting normal cone 中. 
- 同理, 纵轴负半轴的 regular normal cone 也会贡献 $\{(u, 0): u \in \mathbb{R}\}$ 到 limiting normal cone 中. 
- 因此, limiting normal cone 包含了整个第一象限, 以及完整的横纵坐标轴. 

***Definition* (Clarke Normal Cone)**: 对于一个 closed nonempty 集合 $C$, 给定 $\bar{\mathbf{x}} \in C$, 其 Clarke normal cone 定义为 Mordukhovich normal cone 的闭凸包:
$$
\mathcal{N}^{\text{C}}_C(\bar{\mathbf{x}}) := \text{cl conv} \mathcal{N}^{\text{M}}_C(\bar{\mathbf{x}}).
$$

*Example*. 不难看出, 若进一步对 limiting normal cone 取闭凸包, 则得到 Clarke normal cone, 而其在当前例子中就是整个 $\mathbb{R}^2$ 空间. 

综上, 三种非凸集合的 normal cone 的关系为:
$$
\text{regular normal cone} \subseteq \text{limiting normal cone} \subseteq \text{Clarke normal cone}.
$$


### 3.3 Variational Analysis Tools 2: Generalized Gradient

另一种 reformulation 的方法是将互补约束 reform 为 $\max\{\mathbf{G}(\mathbf{x}), \mathbf{H}(\mathbf{x})\} = 0$. 然而 $\max$ 函数是非光滑, 且并不总能保证是凸的, 同样无法直接使用传统的梯度或次梯度 (注意 subgradient 也是针对凸函数的概念) 进行分析. 因此, 这里将引入 **generalized gradient** 的概念, 其中最主要的是 **Clarke generalized gradient**. 

***Definition* (Clarke Generalized Gradient)**: 对于一个 locally Lipschitz 函数 $f: \mathbb{R}^n \to \mathbb{R}$, $f$ 在 $\bar{\mathbf{x}}$ 处的 Clarke generalized gradient 定义为:
$$
\partial^{\text{C}} f(\bar{\mathbf{x}}) := \text{conv} \left\{ \lim_{k \to \infty} \nabla f(\mathbf{x}^k) : f~ \text{is differentiable at } \mathbf{x}^k \to \bar{\mathbf{x}}  \right\}.
$$

- 该定义可以理解为: 在 $f$ 的可微点处任取趋近于 $\bar{\mathbf{x}}$ 的点 $\{\mathbf{x}^k\}$, 对每个点 $\mathbf{x}^k$ 计算其梯度 $\nabla f(\mathbf{x}^k)$, 并取该梯度列的极限. 将所有的极限梯度取凸包, 就得到了 Clarke generalized gradient. 简单说, Clarke generalized gradient 是在附近可微点的梯度极限信息的凸包. 
- 若 $f$ 是 locally Lipschitz, 则 Rademacher 定理保证了 $f$ 在几乎所有点都是可微的, 因此 Clarke generalized gradient 是良定义的.

*Example* 考虑 $f(x) = \max\{1-x^2, x-1\}$. 则该函数在非交界点处的梯度就是各自的梯度, 而在交界点处 $x = 1$, $f_1(x) = 1-x^2$ 的梯度为 $f'_1(1) = -2$, $f_2(x) = x-1$ 的梯度为 $f'_2(1) = 1$. 因此, Clarke generalized gradient 在 $x=1$ 处为 $\partial^{\text{C}} f(1) = \text{conv}\{-2, 1\} = [-2, 1]$. 同理, 在 $x = -2$ 处的 Clarke generalized gradient 为 $\partial^{\text{C}} f(-2) = \text{conv}\{4, 1\} = [1, 4]$.

![20260707140905](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260707140905.png)


### 3.4 Variational Analysis: Optimality Conditions for MPCC

该部分的叙事主线仍然要回到 Normal Cone 这一分支上. 在 variational 的开始部分介绍了通过 normal cone 来刻画 KKT 条件的方式. 因此, 在定义了regular normal cone, limiting normal cone, Clarke normal cone 三种非凸集合的 normal cone 之后, 可以类似地给出 MPCC 的最优性条件.  -->

#### (Strong) Optimality Conditions for Regular Normal Cone

考虑一般的一个 geometric constraint 问题:
$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}), ~\text{s.t.}~ \boldsymbol{\Phi}(\mathbf{x}) \in C,
$$
- 其中 $\boldsymbol{\Phi}: \mathbb{R}^n \to \mathbb{R}^m$ 是连续可微的, $f: \mathbb{R}^n \to \mathbb{R}$ 且 $C$ 是 closed nonempty 集合. 记 $\bar{\mathbf{x}}$ 是该问题的局部最优解. 

则通过 regular normal cone 可以得到 $\bar{\mathbf{x}}$ 如下 Strong (S-) Optimality Condition:

***Proposition* (S-Optimality Condition for Regular Normal Cone)**: 若 $\bar{\mathbf{x}}$ 是上述问题的局部最优解, 且满足正则条件 $\nabla \boldsymbol{\Phi}(\bar{\mathbf{x}}) \in \mathbb{R}^{m \times n}$ 的 $m$ 行线性无关 (防止退化), 则存在某个乘子 $\boldsymbol{\lambda}^\Phi \in \widehat{\mathcal{N}}_C(\boldsymbol{\Phi}(\bar{\mathbf{x}}))$ 使得
$$
\mathbf{0} = \nabla f(\bar{\mathbf{x}}) + \nabla \boldsymbol{\Phi}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^\Phi.
$$

*Proof*. 
- 记原问题的可行域为 $\mathcal{F} := \{ \mathbf{x} \in \mathbb{R}^n : \boldsymbol{\Phi}(\mathbf{x}) \in C \}$. 
- 由于 $\bar{\mathbf{x}}$ 是局部最优解, 故
    $$
    \mathbf{0} \in \nabla f(\bar{\mathbf{x}}) + \widehat{\mathcal{N}}_{\mathcal{F}}(\bar{\mathbf{x}}).
    $$
    - 这是因为, 因为 $\bar{\mathbf{x}}$ 是局部最优解, 因此在其可行域 $\mathcal{F}$ 中的一个小邻域内, 所有 $\mathbf{x} \in \mathcal{F}$ 都满足 $f(\mathbf{x}) \geq f(\bar{\mathbf{x}})$. 由于 $f$ 可微, 对其一阶 Taylor 展开, 则有 $f(\mathbf{x}) = f(\bar{\mathbf{x}}) + \langle \nabla f(\bar{\mathbf{x}}), \mathbf{x} - \bar{\mathbf{x}} \rangle + o(\|\mathbf{x} - \bar{\mathbf{x}}\|)$. 因此, 对于所有 $\mathbf{x} \in \mathcal{F}$, 有
        $$
        0 \stackrel{\text{local min}}{\leq} f(\mathbf{x}) - f(\bar{\mathbf{x}}) \stackrel{\text{Taylor}}{=} \langle \nabla f(\bar{\mathbf{x}}), \mathbf{x} - \bar{\mathbf{x}} \rangle + o(\|\mathbf{x} - \bar{\mathbf{x}}\|).
        $$
        这恰满足 regular normal cone 的定义, $\widehat{\mathcal{N}}_{\mathcal{F}}(\bar{\mathbf{x}}) := \{ \mathbf{w} : \langle \mathbf{w}, \mathbf{x} - \bar{\mathbf{x}} \rangle \leq o(\|\mathbf{x} - \bar{\mathbf{x}}\|), ~\forall \mathbf{x} \in \mathcal{F} \}$, 故 $- \nabla f(\bar{\mathbf{x}}) \in \widehat{\mathcal{N}}_{\mathcal{F}}(\bar{\mathbf{x}})$, 即 $\mathbf{0} \in \nabla f(\bar{\mathbf{x}}) + \widehat{\mathcal{N}}_{\mathcal{F}}(\bar{\mathbf{x}})$.

- 然而, 我们无法直接处理 $\widehat{\mathcal{N}}_\mathcal{F}(\bar{\mathbf{x}})$, 因为这个集合是通过 $\boldsymbol{\Phi}(\mathbf{x}) \in C$ 反解出来的 $\mathbf{x}$ 构成的, 我们并不显示地知道其构成的元素. 因此承认并利用 *Change of Coordinates*  公式, 有如下关系成立
    $$
    \widehat{\mathcal{N}}_{\mathcal{F}}(\bar{\mathbf{x}}) = \nabla \boldsymbol{\Phi}(\bar{\mathbf{x}})^\top
    \widehat{\mathcal{N}}_C(\boldsymbol{\Phi}(\bar{\mathbf{x}}))
    $$
    其中满秩的条件也是应用在了这一步上. 

- 最终整理故有, 
    $$
    \mathbf{0} \in \nabla f(\bar{\mathbf{x}}) + \nabla \boldsymbol{\Phi}(\bar{\mathbf{x}})^\top
    \widehat{\mathcal{N}}_C(\boldsymbol{\Phi}(\bar{\mathbf{x}})).
    $$

$\square$

具体应用到 MPCC 问题中, 考虑问题:
$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}), \quad\text{s.t.}~ \boldsymbol{\Phi}(\mathbf{x}) \in \Omega_C,
$$
则可以通过 regular normal cone 得到 S-Optimality Condition. 

***Proposition* (S-Optimality Necessary Condition for MPCC)**: 若 $\bar{\mathbf{x}}$ 是上述 MPCC 问题的局部最优解, 且满足正则条件 MPCC-LICQ, 即:
$$
\nabla \mathbf{G}_i(\bar{\mathbf{x}}), \nabla \mathbf{H}_j (\bar{\mathbf{x}}), ~i \in \mathcal{I}_G(\bar{\mathbf{x}}), j \in \mathcal{I}_H(\bar{\mathbf{x}})
$$
是线性无关的. 则存在某个乘子 $(\boldsymbol{\lambda}^G, \boldsymbol{\lambda}^H) \in \widehat{\mathcal{N}}_{\Omega_C}(\mathbf{G}(\bar{\mathbf{x}}), \mathbf{H}(\bar{\mathbf{x}}))$ 使得
$$
\mathbf{0} = \nabla f(\bar{\mathbf{x}}) + \nabla \mathbf{G}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^G + \nabla \mathbf{H}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^H.
$$

- 注意到, 这里的正则条件不需要全部的约束行满秩, 而是所有 active set 的梯度向量线性无关即可. 并且对于 biactive 的退化情况, 若 $\mathbf{G}_i(\bar{\mathbf{x}}) = \mathbf{H}_i(\bar{\mathbf{x}}) = 0$, 则 $\nabla \mathbf{G}_i(\bar{\mathbf{x}})$ 和 $\nabla \mathbf{H}_i(\bar{\mathbf{x}})$ 都需要纳入正则条件的线性无关性要求中. 因此也规避了传统 NLP 中的 MFCQ 等 CQ 条件不成立的问题.

若再带入具体的 $\widehat{\mathcal{N}}_{\Omega_C}(\mathbf{G}(\bar{\mathbf{x}}), \mathbf{H}(\bar{\mathbf{x}}))$ 的表达式:
$$
\widehat{\mathcal{N}}_{\Omega_C}(\mathbf{G}(\bar{\mathbf{x}}), \mathbf{H}(\bar{\mathbf{x}})) = \left\{
    (\mathbf{u}, \mathbf{v}) : 
    \begin{aligned}
        & u_i = 0, v_i \in \mathbb{R}, & \quad \text{if } ~\mathbf{G}_i(\bar{\mathbf{x}}) < 0, \mathbf{H}_i(\bar{\mathbf{x}}) = 0 \\
        & u_i \in \mathbb{R}, v_i = 0, & \quad \text{if } ~\mathbf{G}_i(\bar{\mathbf{x}}) = 0, \mathbf{H}_i(\bar{\mathbf{x}}) > 0\\
        & u_i \geq 0, v_i \geq 0, & \quad \text{if } ~ \mathbf{G}_i(\bar{\mathbf{x}}) = \mathbf{H}_i(\bar{\mathbf{x}}) = 0
    \end{aligned}
\right\},
$$
则可以得到 MPCC 的 S-Optimality Condition 的具体形式: 若 $\bar{\mathbf{x}}$ 是 MPCC 的可行点, 称 $\bar{\mathbf{x}}$ 是 S-Stationary Point, 若存在 multiplier $\boldsymbol{\lambda}^G, \boldsymbol{\lambda}^H \in \mathbb{R}^{m}$, 使得如下条件同时成立:
1. Stationarity:
    $$
    \mathbf{0} = \nabla f(\bar{\mathbf{x}}) + \nabla \mathbf{G}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^G + \nabla \mathbf{H}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^H.
    $$
2. 对每个 index $i$, 根据 $\mathbf{G}_i(\bar{\mathbf{x}})$ 和 $\mathbf{H}_i(\bar{\mathbf{x}})$ 的取值, 有如下约束:
    - 若 $\mathbf{G}_i(\bar{\mathbf{x}}) < 0, \mathbf{H}_i(\bar{\mathbf{x}}) = 0$, 则 $\lambda_i^G = 0, \lambda_i^H$ 任意.
    - 若 $\mathbf{G}_i(\bar{\mathbf{x}}) = 0, \mathbf{H}_i(\bar{\mathbf{x}}) < 0$, 则 $\lambda_i^G$ 任意, $\lambda_i^H = 0$.
    - 若 $\mathbf{G}_i(\bar{\mathbf{x}}) = 0, \mathbf{H}_i(\bar{\mathbf{x}}) = 0$, 则 $\lambda_i^G \geq 0, \lambda_i^H \geq 0$.

#### Optimality Conditions for Limiting Normal Cone

对于 strong optimality condition, 其正则条件要求 $\nabla \boldsymbol{\Phi}(\bar{\mathbf{x}})$ 行满秩, 对应 MPCC 中要求 MPCC-LICQ. 然而该条件较为强, 因此在实际问题中可能不容易满足. 因此, 这里尝试将放宽这个条件, 并配合 limiting normal cone 来得到稍弱但更一般的 optimality condition.

暂时仍然考虑 general 的 geometric constraint 问题:
$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}), ~\text{s.t.}~ \boldsymbol{\Phi}(\mathbf{x}) \in C,
$$
- 其中 $\boldsymbol{\Phi}: \mathbb{R}^n \to \mathbb{R}^m$ 是连续可微的, $f: \mathbb{R}^n \to \mathbb{R}$ 是连续可微的, $C$ 是 closed nonempty 集合. 记 $\bar{\mathbf{x}}$ 是该问题的局部最优解.

首先给出局部误差有界 *local error bound* 的定义, 该定义是后续 optimality condition 的一个重要假设.

***Definition* (Local Error Bound)**: 对于上述问题, 若存在常数 $\mu >0$, 存在关于 $\bar{\mathbf{x}}$ 的邻域 $\mathcal{U}(\bar{\mathbf{x}})$, 使得对于所有 $\mathbf{x} \in \mathcal{U}(\bar{\mathbf{x}})$, 有
$$
\text{dist}(\mathbf{x}, \mathcal{F}) \leq \mu \cdot \text{dist}(\boldsymbol{\Phi}(\mathbf{x}), C),
$$
则称该问题在 $\bar{\mathbf{x}}$ 处满足局部误差有界 (local error bound) 条件.
- 其中 $\mathcal{F} := \{\mathbf{x} \in \mathbb{R}^n: \boldsymbol{\Phi}(\mathbf{x}) \in C\}$ 是问题的可行域, $\text{dist}(\mathbf{x}, \mathcal{S}) := \inf_{\mathbf{s} \in \mathcal{S}} \|\mathbf{x} - \mathbf{s}\|$ 是点 $\mathbf{x}$ 到集合 $\mathcal{S}$ 的距离.
- 在 variational analysis 中, 其有时也称映射 $M(\mathbf{x}) := \boldsymbol{\Phi}(\mathbf{x}) - C$ 是 metrically regular 的, 或 $M^{-1}$ 是 calm 的. 不过, 这里不展开讨论这些概念, 在看到这些术语时, 就理解成是 local error bound 即可.

相比于 regular normal cone, limiting normal cone 的 optimality condition 不需要 $\nabla \boldsymbol{\Phi}(\bar{\mathbf{x}})$ 行满秩, 只需要满足 local error bound 条件即可. 这是一个更为宽泛的条件. 

***Proposition* (Necessary Condition for Limiting Normal Cone / Mordukhovich Optimality Condition)**: 若 $\bar{\mathbf{x}}$ 是上述问题的局部最优解, 且满足 local error bound 条件, 则存在某个 limiting normal cone 中的乘子 $\boldsymbol{\lambda}^\Phi \in \mathcal{N}^{\text{M}}_C(\boldsymbol{\Phi}(\bar{\mathbf{x}}))$ 使得如下最优性条件成立:
$$
\mathbf{0} = \nabla f(\bar{\mathbf{x}}) + \nabla \boldsymbol{\Phi}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^\Phi.
$$


*Proof*. 

- 首先, 直接给出如下引理, 称为 *Clarke's Exact Penalty Principle (精确罚函数)*. 给定在集合 $S$ 上 $L_f$-Lipschitz 连续的函数 $f$. 若 $\bar{\mathbf{x}} \in C \subset S$ 是问题 $\min_{\mathbf{x} \in C} f(\mathbf{x})$ 的最优解, 则 $\bar{\mathbf{x}}$ 也是问题 $\min_{\mathbf{x} \in S} f(\mathbf{x}) + L_f \cdot \text{dist}(\mathbf{x}, C)$ 的最优解.   
  - *Proof of Lemma*. 任取 $\mathbf{x} \in S$, 且记 $\tilde{\mathbf{x}} := \text{Proj}_C(\mathbf{x})$ 为 $\mathbf{x}$ 在 $C$ 上的投影点, 则有
    $$
    \begin{aligned}
    f(\bar{\mathbf{x}}) & \leq f(\tilde{\mathbf{x}}) \quad \text{\small (By optimality)} \\
    & = f(\mathbf{x}) + f(\tilde{\mathbf{x}}) - f(\mathbf{x}) \\
    & \leq f(\mathbf{x}) + L_f \|\tilde{\mathbf{x}} - \mathbf{x}\| \quad \text{\small (By Lipschitz continuity)} \\
    & = f(\mathbf{x}) + L_f \cdot \text{dist}(\mathbf{x}, C) \quad {\small (\| \mathbf{z} - \text{Proj}_C(\mathbf{z}) \| = \min_{\mathbf{c} \in C} \|\mathbf{z} - \mathbf{c}\| = \text{dist}(\mathbf{z}, C))}.
    \end{aligned}
    $$
  - 应用该引理在当前问题中. 因为 $\bar{\mathbf{x}}$ 是 $\min_{\mathbf{x} \in \mathcal{F}} f(\mathbf{x})$ 的局部最优解,  其中 $\mathcal{F} := \{\mathbf{x} \in \mathbb{R}^n: \boldsymbol{\Phi}(\mathbf{x}) \in C\}$, 因此其也是问题 $\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}) + L_f \cdot \text{dist}(\mathcal{F}, \mathbf{x})$ 的局部最优解, 其中 $L_f$ 是 $f$ 在 $\bar{\mathbf{x}}$ 附近的 Lipschitz 常数. 故有不等式:
    $$
    f(\bar{\mathbf{x}}) \leq f(\mathbf{x}) + L_f \cdot \text{dist}(\mathcal{F}, \mathbf{x}), \quad \forall \mathbf{x} \in \mathcal{U}(\bar{\mathbf{x}}).
    $$

- 因此由 local error bound 条件, 存在 $\mu > 0$ 使得在 $\bar{\mathbf{x}}$ 的邻域内, 对于所有 $\mathbf{x}$ 有
    $$
    \text{dist}(\mathbf{x}, \mathcal{F}) \leq \mu \cdot \text{dist}(\boldsymbol{\Phi}(\mathbf{x}), C).
    $$
    因此上述不等式可以进一步写为
    $$
    f(\bar{\mathbf{x}}) \leq f(\mathbf{x}) + L_f \mu \cdot \text{dist}(\boldsymbol{\Phi}(\mathbf{x}), C), \quad \forall \mathbf{x} \in \mathcal{U}(\bar{\mathbf{x}}).
    $$
    故这意味着, $\bar{\mathbf{x}}$ 也是如下问题的局部最优
    $$
    \min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}) + \gamma \cdot \text{dist}(\boldsymbol{\Phi}(\mathbf{x}), C), \quad \text{where } \gamma \geq L_f \mu.
    $$
    - 至此我们将原问题转化为一个无约束优化问题. 对于该问题, 其在最优点 $\bar{\mathbf{x}}$ 处有 Clarke generalized gradient 的最优性条件:
        $$
        \begin{aligned}
        \mathbf{0} &\in \partial^{\text{C}} \left(f(\bar{\mathbf{x}}) + \gamma \cdot \text{dist}(\boldsymbol{\Phi}(\bar{\mathbf{x}}), C) \right) \\
        &\quad = \nabla f(\bar{\mathbf{x}}) + \gamma \cdot \partial^{\text{C}} \text{dist}(\boldsymbol{\Phi}(\bar{\mathbf{x}}), C) \\
        &\quad = \nabla f(\bar{\mathbf{x}}) + \gamma \nabla \boldsymbol{\Phi}(\bar{\mathbf{x}})^\top \mathcal{N}^{\text{M}}_C(\boldsymbol{\Phi}(\bar{\mathbf{x}})).
        \end{aligned}
        $$
        - 其中最后一个不等式是因为 (1) 链式法则; (2) 变分分析的一个结论: 距离函数的 Clarke generalized gradient 就是 limiting normal cone, 即 $\partial^{\text{C}} \text{dist}(\mathbf{y}, C) = \mathcal{N}^{\text{M}}_C(\mathbf{y})$.

$\square$


M-stationary 只需要 local error bound 条件. 该条件尽管宽松, 但想要验证本身也并不总是容易. 因此在 MPCC 中, 也有一些更为具体的 verifiable 的 CQ 条件等. 其有两个经典的推导关系:
- $\nabla \boldsymbol{\Phi}(\bar{\mathbf{x}})$ 行满秩 $\implies$ NNAMCQ $\implies$ local error bound. 
- $\boldsymbol{\Phi}(\bar{\mathbf{x}})$ 是 affine 且 $C$ 是有限个 convex polyhedral set 的并集 $\implies$ local error bound.

若利用这里的推导关系, 可有如下命题. 

***Proposition* (M-Stationary Condition for MPCC)**: 若 $\bar{\mathbf{x}}$ 是 MPCC 的局部最优解, 且或者 (1) $\mathbf{G}, \mathbf{H}$ 是 affine, 或者 (2)  NNAMCQ, 即不存在全为零的乘子 $\boldsymbol{\lambda}^G, \boldsymbol{\lambda}^H$ 使得
$$
\mathbf{0} = \nabla \mathbf{G}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^G + \nabla \mathbf{H}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^H, \quad (\boldsymbol{\lambda}^G, \boldsymbol{\lambda}^H) \in \mathcal{N}^{\text{M}}_{\Omega_C}(\mathbf{G}(\bar{\mathbf{x}}), \mathbf{H}(\bar{\mathbf{x}})),
$$
则存在某个乘子 $(\boldsymbol{\lambda}^G, \boldsymbol{\lambda}^H) \in \mathcal{N}^{\text{M}}_{\Omega_C}(\mathbf{G}(\bar{\mathbf{x}}), \mathbf{H}(\bar{\mathbf{x}}))$ 使得如下最优性条件成立:
$$
\mathbf{0} = \nabla f(\bar{\mathbf{x}}) + \nabla \mathbf{G}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^G + \nabla \mathbf{H}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^H.
$$

若同理代入 MPCC 的 limiting normal cone 的具体表达式, 有最终的 M-Stationary Condition 的具体形式: 对于每个 index $i$, 若满足约束:
- $\mathbf{0} = \nabla f(\bar{\mathbf{x}}) + \nabla \mathbf{G}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^G + \nabla \mathbf{H}(\bar{\mathbf{x}})^\top \boldsymbol{\lambda}^H$.
- 根据 $\mathbf{G}_i(\bar{\mathbf{x}})$ 和 $\mathbf{H}_i(\bar{\mathbf{x}})$ 的取值, 有如下约束:
    - 若 $\mathbf{G}_i(\bar{\mathbf{x}}) < 0, \mathbf{H}_i(\bar{\mathbf{x}}) = 0$, 则 $\lambda_i^G = 0, \lambda_i^H$ 任意.
    - 若 $\mathbf{G}_i(\bar{\mathbf{x}}) = 0, \mathbf{H}_i(\bar{\mathbf{x}}) < 0$, 则 $\lambda_i^G$ 任意, $\lambda_i^H = 0$.
    - 若 $\mathbf{G}_i(\bar{\mathbf{x}}) = 0, \mathbf{H}_i(\bar{\mathbf{x}}) = 0$, 则 $\lambda_i^G > 0, \lambda_i^H > 0$ 或 $\lambda_i^G = \lambda_i^H = 0$.

则称 $\bar{\mathbf{x}}$ 是 M-Stationary Point.


#### Clarke Stationary Condition

Clarke stationary condition 本身的得到不再沿着 $\Omega_C$ 的 geometric constraint 的路径, 而是沿着 $\max\{\mathbf{G}(\mathbf{x}), \mathbf{H}(\mathbf{x})\} = 0$ 的 reformulation 的路径. 由于 $\max$ 函数是非光滑的, 因此需要利用 Clarke generalized gradient 来刻画其最优性条件.

这里考虑如下的优化问题:
$$
\min_{\mathbf{x} \in \mathbb{R}^n} \mathbf{f}(\mathbf{x}), \quad \text{s.t. }~\boldsymbol{\Phi}(\mathbf{x}) = \mathbf{0},
$$
- 其中, $\mathbf{f}, \boldsymbol{\Phi} : \mathbb{R}^n \to \mathbb{R}^m$ 是 $\bar{\mathbf{x}}$ 附近的 locally Lipschitz 连续的, 但并不假设其可微性. 

若假设 $\bar{\mathbf{x}}$ 同样也是罚问题:
$$
\min_{\mathbf{x} \in \mathbb{R}^n} \mathbf{f}(\mathbf{x}) + \mu \|\boldsymbol{\Phi}(\mathbf{x})\|
$$
的局部最优解 (对于某个 $\mu > 0$), 则存在乘子 $\boldsymbol{\lambda}^\Phi$ 使得 Clarke stationary condition 成立:
$$
\mathbf{0} \in \partial^{\text{C}} \mathbf{f}(\bar{\mathbf{x}}) + \sum_{i=1}^m \lambda_i^\Phi \partial^{\text{C}} \boldsymbol{\Phi}_i(\bar{\mathbf{x}}).
$$

## 4. Mathematical Program with Equilibrium Constraints (MPEC)

### 4.1 MPEC and MPCC

在前面的所有讨论中, 都是在说从一些均衡问题出发, 能够自然的发现其互补的约束, 并通过引入乘子将其表示为 MPCC 的结构. 然而这样的引入处理并不一定是最方便的, 有时也可能会使的问题的求解更为复杂. 

考虑如下 MPEC 问题:
$$
\begin{aligned}
\min_{x, y} & \quad F(x, y) \\
\text{s.t.} \quad & 0 \in \phi(x, y) + \mathcal{N}_\Gamma (y),
\end{aligned}
$$
其中 $\Gamma := \{ y: g(y) \leq 0 \}$ 是一个凸集, 表示问题的可行域约束. 

MPEC 和 MPCC 的区别在于, 其约束条件不再是一个固定不变的集合 $C$, 而变成了一个随 $y$ 变化的 normal cone. 因此, MPEC 的一个研究核心就是当约束条件本身是变化的, 其最优性条件该如何刻画. 



