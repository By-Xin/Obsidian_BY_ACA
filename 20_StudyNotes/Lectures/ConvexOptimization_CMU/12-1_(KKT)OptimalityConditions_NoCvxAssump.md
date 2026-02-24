# Optimality Conditions for Constrained Problems (General Case without Convexity Assumption)

> [!cite] References
> - Lecture: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/
> - Reading: 最优化: 建模、算法与理论, 刘浩洋等, 5.5 小节.

## 0. TL;DR

对于不要求凸性的约束优化问题, 有如下关键结论:

- **切锥**: 从可行域内某点 $\mathbf{x}^*$ 出发, 所有能够满足约束条件的行动方向之集合. 定义为 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) = \left\{ \mathbf{d} \in \mathbb{R}^n \;\middle|\; \exists t_k \downarrow 0, \exists \mathbf{x}_k \in \mathcal{X}, \frac{\mathbf{x}_k - \mathbf{x}^*}{t_k} \to \mathbf{d} \right\}$.
  - 由切锥定义的最优条件: 若 $\mathbf{x}^*$ 是局部极小点, 则有 $\mathbf{d}^\top \nabla f(\mathbf{x}^*) \geq 0, \quad \forall \mathbf{d} \in \mathcal{T}_{\mathcal{X}}(\mathbf{x}^*)$.
- **活跃集**: 对于可行域 $\mathcal{X}$ 内的点 $\mathbf{x}$, 其活跃集定义为 $\mathcal{A}(\mathbf{x}) = \mathcal{E} \cup \{i \in \mathcal{I} \mid c_i(\mathbf{x}) = 0\}$. 即所有等式约束和所有不等式约束中, 在 $\mathbf{x}$ 处取等号的约束的集合.
- **线性化可行方向锥:** 对于可行域 $\mathcal{X}$ 内的点 $\mathbf{x}$, 其线性化可行方向锥定义为 $\mathcal{F}(\mathbf{x}) = \left\{ \mathbf{d} \in \mathbb{R}^n \;\middle|\; \begin{aligned} \mathbf{d}^\top \nabla c_j(\mathbf{x}) = 0, & \quad \forall j \in \mathcal{E} \\ \mathbf{d}^\top \nabla c_i(\mathbf{x}) \leq 0, & \quad \forall i \in \mathcal{A}(\mathbf{x}) \cap \mathcal{I} \end{aligned} \right\}$. 即从当前点 $\mathbf{x}$ 出发, 所有能满足约束的一阶移动方向的集合, 其保证在等式约束上沿着等式约束的切向移动, 在不等式约束上沿着使约束函数不增加(朝可行域内部)的方向移动. 
  - 在约束连续可微的情况下, 切锥一般包含于线性化可行方向锥, 即 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}) \subseteq \mathcal{F}(\mathbf{x})$.
- **约束的品性**: 约束的品性是指约束条件符合某些特定条件的性质. 这些性质往往可以保证在局部最优点 $\mathbf{x}^*$ 处, 满足 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) = \mathcal{F}(\mathbf{x}^*)$. 常见的约束品性有 LICQ, MFCQ, LCQ 等.
  - LICQ: 活跃集中的约束之梯度线性无关. 即对于任意 $i\in \mathcal{A}(\mathbf{x}^*)$ 有 $\nabla c_i(\mathbf{x}^*)$ 线性无关.
  - MFCQ: 活跃集中, 若存在向量 $\mathbf{w}$ 使得等式约束 $\nabla c_j(\mathbf{x}^*)^\top \mathbf{w} = 0$ 成立, 不等式约束 $\nabla c_i(\mathbf{x}^*)^\top \mathbf{w} \lt 0$ 成立, 则称该点满足 MFCQ.
  - LCQ: 若所有的约束函数均是线性函数, 则称该点满足 LCQ.
- **Karush-Kuhn-Tucker (KKT) 条件**: 若 $\mathbf{x}^*$ 是局部极小点, 则有:
  - Stationarity: $\nabla f(\mathbf{x}^*) + \sum_{j\in\mathcal{E}} \lambda_j^* \nabla c_j(\mathbf{x}^*) + \sum_{i\in\mathcal{I}} \mu_i^* \nabla c_i(\mathbf{x}^*) = 0$
  - Primal Feasibility 1: $c_i(\mathbf{x}^*) \leq 0, i\in \mathcal{I}$
  - Primal Feasibility 2: $c_j(\mathbf{x}^*) = 0, j\in \mathcal{E}$
  - Dual Feasibility: $\mu_i^* \geq 0, i\in \mathcal{I}$
  - Complementary Slackness: $\mu_i^* c_i(\mathbf{x}^*) = 0, i\in \mathcal{I}$

- **临界锥**: 对于可行域满足 KKT 的点 $\mathbf{x}^*$, 其 Critical Cone 定义为 $\mathcal{C}(\mathbf{x}^*) = \left\{ \mathbf{d} \in \mathcal{F}(\mathbf{x}^*) \;\middle|\; \mathbf{d}^\top \nabla f(\mathbf{x}^*) = 0 \right\}$. 即在满足 KKT 条件的基础上, 所有一阶线性可行方向中, 那些根据一阶梯度信息无法判断是否为上升下降方向的线性化可行方向. 

- **二阶必要条件**: 若 $\mathbf{x}^*$ 是局部极小点, 且 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) = \mathcal{F}(\mathbf{x}^*)$ 成立, 则对于 KKT 点 $(\mathbf{x}^*, \lambda_j^*, \mu_i^*)$ 有: $\mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^*, \lambda_j^*, \mu_i^*) \mathbf{d} \geq 0, \quad \forall \mathbf{d} \in \mathcal{C}(\mathbf{x}^*; \lambda_j^*, \mu_i^*)$.
  
- **二阶充分条件**: 若在可行点 $\mathbf{x}^*$ 处, 存在 Lagrange Multiplier $(\lambda_j^*, \mu_i^*)$ 使得 KKT 条件成立, 如果: $\mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^*, \lambda_j^*, \mu_i^*) \mathbf{d} \gt 0, \quad \forall \mathbf{d} \in \mathcal{C}(\mathbf{x}^*; \lambda_j^*, \mu_i^*), ~\mathbf{d} \neq \mathbf{0}$. 则 $\mathbf{x}^*$ 是严格局部极小点.


## 1. First-Order Optimality Conditions

回顾, 考虑如下一般的含约束的优化问题 (不要求是凸的):
$$\begin{aligned}
& \min_{\mathbf{x}\in \mathbb{R}^n} && f(\mathbf{x}) \\
& \text{subject to} && c_i(\mathbf{x}) \leq 0, i\in \mathcal{I} \\
& && c_j(\mathbf{x}) = 0, j\in \mathcal{E}
\end{aligned}$$

其 Lagrangian 函数为 (统一记号: 等式约束乘子为 $\lambda_j$, 不等式约束乘子为 $\mu_i \ge 0$):
$$
L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\mathbf{x}) + \sum_{j\in \mathcal{E}} \lambda_j c_j(\mathbf{x}) + \sum_{i\in \mathcal{I}} \mu_i c_i(\mathbf{x})
$$

其 Lagrange Dual Function 为:
$$
g(\boldsymbol{\lambda}, \boldsymbol{\mu}) = \inf_{\mathbf{x}\in \mathbb{R}^n} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})
$$

### 1.1. Optimality Conditions by Tangent Cone

为定义可行域内的一系列点列的极限状态, 引入切向量和切锥的概念. 

***Definition* (Tangent Vector)**: 对于可行域 $\mathcal{X}$ 内的点列 $\{\mathbf{x}_k\}_{k=1}^\infty \subseteq \mathcal{X}\subset \mathbb{R}^n$, 其极限状态为 $\lim_{k\to\infty} \mathbf{x}_k = \mathbf{x}^*\in \mathcal{X}$ (即该点列逼近 $\mathbf{x}^*$). 若存在向量 $\mathbf{d}\in \mathbb{R}^n$, 以及一个正数标量序列 $\{t_k\}_{k=1}^\infty$ 且 $t_k \to 0$ 使得:
$$
\lim_{k\to\infty} \frac{\mathbf{x}_k - \mathbf{x}^*}{t_k} = \mathbf{d}
$$
则称 $\mathbf{d}$ 为 $\mathbf{x}^*$ 处的切向量.

***Definition* (Bouligand (Contingent) Tangent Cone)**: 对于上述点 $\mathbf{x}^*$ 处的全部切向量之集合, 称为该点处的切锥, 记作 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*)$, 其数学表达为:
$$
\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) = \left\{
\mathbf{d} \in \mathbb{R}^n \;\middle|\; 
\exists t_k \downarrow 0, \exists \mathbf{x}_k \in \mathcal{X}, \frac{\mathbf{x}_k - \mathbf{x}^*}{t_k} \to \mathbf{d}
\right\}
$$

- **切锥表示从可行域内某点 $\mathbf{x}^*$ 出发, 所有能够满足约束条件的行动方向之集合.**

> [!example] **切锥的例子**
>
> ![Ref: 最优化: 建模、算法与理论, 刘浩洋等, 5.5 小节](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202602202338194.png)
>
> - 如上图所示. 图中两条曲线分别代表两条约束方程 $c_1(\mathbf{x})$ 和 $c_2(\mathbf{x})$ 的图像. 左侧为不等式约束, 图中深色阴影部分表示两约束方程构成的可行域 $\mathcal{X}$. 右侧为等式约束, 故可行域只有轮廓本身. 
>   - 对于不等式约束, 其切锥为整个深浅阴影区域, 为一个凸锥. 
>   - 对于等式约束, 其切锥只能取在左图轮廓线上, 即图中两条射线. 
>   - 该例中切锥是凸锥; 但一般非凸可行域下的 Bouligand 切锥未必是凸集.

***Theorem* (Optimality Conditions by Tangent Cone)**: 设 $\mathbf{x}^*$ 是可行域 $\mathcal{X}$ 内的一个局部极小点. 若 $f$ 和 $c_i, c_j (\forall i,j)$ 在 $\mathbf{x}^*$ 处可微, 则有:
$$
\mathbf{d}^\top \nabla f(\mathbf{x}^*) \geq 0, \quad \forall \mathbf{d} \in \mathcal{T}_{\mathcal{X}}(\mathbf{x}^*)
$$

或等价地:
$$
\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) \cap \{\mathbf{d}\in \mathbb{R}^n \mid \mathbf{d}^\top \nabla f(\mathbf{x}^*) \lt 0\} = \emptyset
$$

- 其直观的理解为, 从最优点 $\mathbf{x}^*$ 出发, 所有能够满足约束条件的行动方向, 其与梯度方向的夹角都不应为钝角 (允许直角); 即任何从最优点出发的可行方向都不应是一阶下降方向. 
- *Proof.*
  - 用反证法, 假设在 $\mathbf{x}^*$ 处有 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) \cap \{\mathbf{d}\in \mathbb{R}^n \mid \mathbf{d}^\top \nabla f(\mathbf{x}^*) \lt 0\} \neq \emptyset$, 则记该集合中的某个向量为 $\mathbf{d}^*$.
  - 根据切向量的定义, 存在 $\{t_k\}_{k=1}^\infty$ 且 $t_k \to 0$ 以及对应的切向量 $\{\mathbf{d}_k\}_{k=1}^\infty$ 使得 $\mathbf{x}^*+ t_k \mathbf{d}_k \in \mathcal{X}$. 
  - 对 $f$ 在 $\mathbf{x}^*$ 进行 Taylor 展开, 有:
    $$
    \begin{aligned}
    f(\mathbf{x}^*+ t_k \mathbf{d}_k) &= f(\mathbf{x}^*) + t_k \mathbf{d}_k^\top \nabla f(\mathbf{x}^*) + o(t_k) \\
    &= f(\mathbf{x}^*) + \underbrace{t_k (\mathbf{d}^*)^\top \nabla f(\mathbf{x}^*)}_{<0} + \underbrace{t_k (\mathbf{d}_k - \mathbf{d}^*)^\top \nabla f(\mathbf{x}^*)}_{\to 0} + o(t_k)
    \end{aligned}
    $$
  - 由于 $(\mathbf{d}^*)^\top \nabla f(\mathbf{x}^*) < 0$ 且 $\mathbf{d}_k \to \mathbf{d}^*$, 对足够大的 $k$ 有:
    $$
    f(\mathbf{x}^*+ t_k \mathbf{d}_k) < f(\mathbf{x}^*)
    $$
    与 $\mathbf{x}^*$ 为局部极小点矛盾.

  $\square$


### 1.2. Optimality Conditions by Linearized Feasible Direction Cone

上述在几何上给出了可行域的判定定理, 然而其计算往往是不容易的. 如下我们需要给出更容易计算的可行方向集合之定义. 

***Definition* (Active Set)**: 对于可行域 $\mathcal{X}$ 内的点 $\mathbf{x}$, 其 active set 定义为:
$$
\mathcal{A}(\mathbf{x}) = \mathcal{E} \cup \{i \in \mathcal{I} \mid c_i(\mathbf{x}) = 0\}
$$

即所有等式约束和所有不等式约束中, 在 $\mathbf{x}$ 处取等号的约束的集合. 

- Active set 是对于当前点 $\mathbf{x}$ 处, 所有真正起到约束作用的约束的集合. 对于所有 $c_i(\mathbf{x}) < 0$ 的约束, 其并没有起到约束作用, 在这些该点的微小领域内, 这些约束仍然可以被满足. 

***Definition* (Linearized Feasible Direction Cone)**: 对于可行域 $\mathcal{X}$ 内的点 $\mathbf{x}$, 其 linearized feasible direction cone 定义为:
$$
\mathcal{F}(\mathbf{x}) = \left\{
\mathbf{d} \in \mathbb{R}^n \;\middle|\; \begin{aligned}
\mathbf{d}^\top \nabla c_j(\mathbf{x}) = 0, & \quad \forall j \in \mathcal{E} \\
\mathbf{d}^\top \nabla c_i(\mathbf{x}) \leq 0, & \quad \forall i \in \mathcal{A}(\mathbf{x}) \cap \mathcal{I}
\end{aligned}
\right\}
$$

- 该定义的 intuition 如下: 
  - 我们尝试寻找从当前点 $\mathbf{x}$ 出发, 所有能够满足约束条件的行动方向之集合. 
  - 希望存在微小量 $t > 0$ 使得 $\mathbf{x} + t \mathbf{d} \in \mathcal{X}$. 故需要对每个约束 $c_j, c_i$ 求解一阶 Taylor 近似 (即线性化):
    $$
    c_i(\mathbf{x} + t \mathbf{d}) \approx c_i(\mathbf{x}) + t \nabla c_i(\mathbf{x})^\top \mathbf{d} \quad \forall i
    $$

  - 对于等式约束 $j\in \mathcal{E}$, 要求有 $c_j(\mathbf{x} + t \mathbf{d}) = 0$, 又由于 $c_j(\mathbf{x}) = 0$, 代入上述展开故有:
    $$
    \nabla c_j(\mathbf{x})^\top \mathbf{d} = 0,\quad \forall j \in \mathcal{E}
    $$
    - 站在当前点 $\mathbf{x}$ 处, 只能沿着等式约束的切向量方向移动, 即"沿着等式约束的轮廓线"移动.
  - 对于不等式约束 $i\in \mathcal{A}(\mathbf{x}) \cap \mathcal{I}$, 要求有 $c_i(\mathbf{x} + t \mathbf{d}) \leq 0$, 又由于 $c_i(\mathbf{x}) = 0$, 故有:
    $$
    \nabla c_i(\mathbf{x})^\top \mathbf{d} \leq 0,\quad \forall i \in \mathcal{A}(\mathbf{x}) \cap \mathcal{I}
    $$
    - 站在当前点 $\mathbf{x}$ 处, 可行的移动方向必与梯度方向夹角为钝角或直角, 即必往约束的内部(或切向)移动.
      - *为什么梯度方向的钝角或直角方向对应约束内部(或边界切向)?* 事实上, 这是因为该约束为 $\leq$ 的不等式约束, 而梯度方向本质上为最陡上升方向, 因此沿梯度方向的正分量为上升方向; 对于 active set 中的点而言, 任何严格上升分量都将导致该不等式约束不成立. 
    

***Corollary* (Contingent Tangent Cone and Linearized Feasible Direction Cone)**: 若存在 $\mathbf{x}^*$ 的邻域 $U$, 使得全部约束函数 $\{c_i\}_{i\in \mathcal{I}}$ 与 $\{c_j\}_{j\in \mathcal{E}}$ 在 $U$ 上一阶连续可微, 则对于任意可行点 $\mathbf{x}\in U\cap \mathcal{X}$, 满足
$$
\mathcal{T}_{\mathcal{X}}(\mathbf{x}) \subseteq \mathcal{F}(\mathbf{x})
$$
- 观察下述例子:
  - 考虑问题
    $$
    \begin{aligned}
    \min_{x\in \mathbb{R}} & f(x) = x \\
    \text{s.t.} & (-x+3)^3 \leq 0
    \end{aligned}
    $$
    其可行域为 $\mathcal{X}=[3,\infty)$. 
    - 根据切锥的定义, 可知在 $x = 3$ 处, 其切锥为 $\mathcal{T}_{\mathcal{X}}(3) = \{d \mid d \geq 0\}$. 
    - 又根据线性化可行方向锥的定义, 其梯度(导数)方向为 $c'(x) = -3(-x+3)^2\mid_{x=3} = 0$, 且该点 $x=3$ 处该不等式约束是 active 的, 故其线性化可行方向需满足 $0\cdot d \leq 0$, 即 $d \in \mathbb{R}$. 
    - 故有 $\mathcal{T}_{\mathcal{X}}(3) \subset \mathcal{F}(3)$.
  - 另一方面, 若将约束条件改为
    $$
    \begin{aligned}
    \min_{x\in \mathbb{R}} & f(x) = x \\
    \text{s.t.} & -x+3 \leq 0
    \end{aligned}
    $$
    其可行域仍为 $\mathcal{X}=[3,\infty)$. 
    - 根据定义, 由于可行域没有改变, 故在 $x = 3$ 处, 其切锥仍为 $\mathcal{T}_{\mathcal{X}}(3) = \{d \mid d \geq 0\}$. 
    - 然而, 其导数发生了变化, 此时 $c'(x) = -1\mid_{x=3} = -1$, 故其线性化可行方向需满足 $-1\cdot d \leq 0$, 即 $d \geq 0$. 
    - 此时有 $\mathcal{T}_{\mathcal{X}}(3) = \mathcal{F}(3) = \{d \mid d \geq 0\}$. 
  - 该例子说明, 即使对于同一个可行域, 只是其约束条件的代数表示发生了变化, 其线性化可行方向锥可能发生变化. 本质上, 这是因为线性化可行方向锥的定义是基于一阶 Taylor 近似, 而在高维表述中该一阶信息可能丢失, 从而影响对于可行方向的判定. 


综上, 我们有观察: 
- 线性化可行方向锥易于计算和使用, 但其本身会受到问题的代数表示的影响
- 切锥相对稳健, 然而其计算往往需要计算极限等复杂操作. 

### 1.3. Constraint Qualification

根据上述观察, 引入约束的品性 (Constraint Qualification) 这一概念, 满足该品性的约束往往保证了在最优点 $\mathbf{x}^*$ 处可以有诸如 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) = \mathcal{F}(\mathbf{x}^*)$ 的优秀性质. 

***Definition* (Linear Independence Constraint Qualification)**: 对于可行域 $\mathcal{X}$ 内的点 $\mathbf{x}$, 任意 Active set $\mathcal{A}(\mathbf{x})$ 中的约束 $\nabla c_i(\mathbf{x}), \quad i\in \mathcal{A}(\mathbf{x})$ 线性无关, 则称该约束在 $\mathbf{x}$ 处满足 LICQ.

>[!warning] 注意
> 线性无关的是约束的梯度, 而不是约束本身. 

***Lemma* (LICQ Property)**: 若任意可行点 $\mathbf{x}\in \mathcal{X}$ 满足 LICQ, 则有:
$$
\mathcal{T}_{\mathcal{X}}(\mathbf{x}) = \mathcal{F}(\mathbf{x})
$$

- *Proof.*
  - 不失一般性, 假设 active set $\mathcal{A}(\mathbf{x}) = \mathcal{E} \cup \mathcal{I}$ 且 $|\mathcal{A}(\mathbf{x})| = m$. 记矩阵:
    $$
    \mathbf{A}(\mathbf{x}) = \begin{bmatrix}
    \nabla c_1(\mathbf{x}) & \nabla c_2(\mathbf{x}) & \dots & \nabla c_m(\mathbf{x})
    \end{bmatrix}^\top \in \mathbb{R}^{m\times n}
    $$
  - 由 LICQ 之假设, 各约束之间是线性独立的, 故有 $\text{rank}(\mathbf{A}(\mathbf{x})) = m$. 取 $\mathbf{Z}\in \mathbb{R}^{n\times (n-m)}$ 的列向量张成 $\text{Null}(\mathbf{A}(\mathbf{x}))$, 则有 $\mathbf{A}(\mathbf{x})\mathbf{Z} = \mathbf{0}_{m\times (n-m)}$ (即为等式约束的一阶可行方向对应的空间). 此外, 根据 rank-nullity 定理, 有 $\text{rank}(\mathbf{A}(\mathbf{x})) + \text{nullity}(\mathbf{A}(\mathbf{x})) = n$, 故 $\text{rank}(\mathbf{Z}) = n-m$. 
    - $\mathbf{Z}$ 张成的空间即为贴着活跃约束的边界, 沿着该方向移动不会违反任何约束. 
  - 给定任意可行点 $\mathbf{x}_0$ 与 $\mathbf{d}\in \mathcal{F}(\mathbf{x}_0)$, 欲证 $\mathbf{d}\in \mathcal{T}_{\mathcal{X}}(\mathbf{x}_0)$, 即 $\mathcal{F}(\mathbf{x}_0)\subseteq\mathcal{T}_{\mathcal{X}}(\mathbf{x}_0)$. 若命题得证, 再加之 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}_0) \subseteq \mathcal{F}(\mathbf{x}_0)$, 则将有 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}_0) = \mathcal{F}(\mathbf{x}_0)$. 
    <!-- - 根据切锥定义, 我们需要构造一串可行点列 $\{\mathbf{z}_k\}_{k=1}^\infty \subseteq \mathcal{X}\subset \mathbb{R}^n$ 和 $t_k \downarrow 0$ 使得 $\dfrac{\mathbf{z}_k - \mathbf{x}_0}{t_k} \to \mathbf{d}$, 或者说 $\mathbf{z}_k = \mathbf{x}_0 + t_k \mathbf{d} + \delta_{t_k}$, 且 $\delta_{t_k} = o(t_k)$. 本质在于说明, 每给定一个步长 $t_k$ 以及想要移动的方向 $\mathbf{d}$, 我们总能找到一个可行点 $\mathbf{z}_k$ (事实上依赖于步长故 $\mathbf{z}(t_k)$) 保证 $\mathbf{z}(t_k) \to \mathbf{x}_0$. 并且这种趋近关系事实上应当是对于所有 $t_k \downarrow 0$ 都成立的. 下面的证明将说明, 我们可以构造出一个线性系统 $R(\mathbf{z},t_k; \mathbf{d}) = \mathbf{0}$ 使得可以从中解出 $\mathbf{z}(t_k)$.
    - 记 $\mathbf{c}(\mathbf{z}) = \begin{bmatrix} c_1(\mathbf{z}) & c_2(\mathbf{z}) & \dots & c_m(\mathbf{z}) \end{bmatrix}^\top \in \mathbb{R}^m$, 是一个所有活跃函数组成的向量函数. 则上述映射 $R: \mathbb{R}^n \times \mathbb{R} \to \mathbb{R}^{m+(n-m)}$ 可以表示为:
      $$
      \begin{aligned}
      R(\mathbf{z},t_k; \mathbf{d}) &=
        \begin{bmatrix}
        \mathbf{c}(\mathbf{z}) - t_k A(\mathbf{x}_0) \mathbf{d} \\
        \mathbf{Z}^\top (\mathbf{z} - \mathbf{x}_0 - t_k \mathbf{d})
        \end{bmatrix} = \mathbf{0}
      \end{aligned}
      $$
      - 第一部分 $\mathbf{c}(\mathbf{z}) - t_k A(\mathbf{x}_0) \mathbf{d} = \mathbf{0}$ 等价于对于任意 $i\in \mathcal{A}(\mathbf{x}_0)$ 有 $c_i(\mathbf{z}) = t_k \mathbf{d}^\top \nabla c_i(\mathbf{x}_0)$. 其 LHS 代表我们想要找出的点的约束取值; 而 RHS 代表在 $\mathcal{F}(\mathbf{x}_0)$ 中的约束状态. 若 RHS $i\in\mathcal{E}$, 则 $\mathbf{d}^\top \nabla c_i(\mathbf{x}_0) = 0$, 对应 LHS $c_i(\mathbf{z}) = 0$; 若 $i\in\mathcal{I}$, 则 $\mathbf{d}^\top \nabla c_i(\mathbf{x}_0) \leq 0$, 对应 LHS $c_i(\mathbf{z}) \leq 0$. 因此通过第一部分, 我们可以很好地控制 $\mathbf{z}$ 仍为可行点.
      - 第二部分 $\mathbf{Z}^\top (\mathbf{z} - \mathbf{x}_0 - t_k \mathbf{d}) = \mathbf{0}$ 相当于说明 $\mathbf{z} - (\mathbf{x}_0 + t_k \mathbf{d}) = \mathbf{z} - \delta_{t_k}$ 与 $\mathbf{Z}$ 的列向量正交, 即只能沿着约束梯度方向进行移动修正. 
    - 在定义 $R(\mathbf{z},t_k; \mathbf{d}) = \mathbf{0}$ 后, 根据隐函数定理, 能够保证对于任意给定充分小的 $t_k > 0$, 存在唯一的 $\mathbf{z}(t_k)$, 使得 $R(\mathbf{z}(t_k),t_k) = \mathbf{0}$.
      - 具体而言, 对于 $R: \mathbb{R}^n \times \mathbb{R} \to \mathbb{R}^n$ 连续可微. 若 (1) $R(\mathbf{x}_0,0) = \mathbf{0}$; (2) $\dfrac{\partial R}{\partial \mathbf{z}}(\mathbf{x}_0,0)$ 可逆, 则定理可以保证: 存在 $\epsilon>0$, 以及唯一的函数 $\mathbf{z}(t)$, 使得 $R(\mathbf{z}(t),t)=\mathbf{0},~ \mathbf{z}(0)=\mathbf{z}$.
    - 具体求解 $\mathbf{z}(t_k)$ 时, 对 $\mathbf{c}(\mathbf{z})$ 在 $\mathbf{x}_0$ 处进行一阶 Taylor 展开, 有:
      $$
      \mathbf{c}(\mathbf{z}_k) = \mathbf{c}(\mathbf{x}_0) + \nabla \mathbf{c}(\mathbf{x}_0)(\mathbf{z}_k - \mathbf{x}_0) + \mathbf{e}_k
      $$
    - 将上表达式进行代入整理, 可得 (具体过程略)
      $$
      \begin{aligned}
      \frac{\mathbf{z}_k - \mathbf{x}_0}{t_k} &= \mathbf{d} + \frac{1}{t_k} \left(\begin{bmatrix} A(\mathbf{x}_0) \\ \mathbf{Z}^\top \end{bmatrix}^{-1}\right) \begin{bmatrix} \mathbf{e}_k \\\mathbf{0}\end{bmatrix}
      \end{aligned}
      $$
    - 根据 $\|\mathbf{e}_k\|=o(t_k)$, 命题得证. -->


***Definition* (Mangasarian-Fromovitz Constraint Qualification, MFCQ)** 给定可行点 $\mathbf{x}_0$ 及其 active set $\mathcal{A}(\mathbf{x}_0)$, 若存在一个向量 $\mathbf{w}\in \mathbb{R}^n$ 满足:
$$
\begin{aligned}
\nabla c_i(\mathbf{x}_0)^\top \mathbf{w} &\lt 0, \quad &&\forall i \in \mathcal{A}(\mathbf{x}_0) \cap \mathcal{I} \\
\nabla c_j(\mathbf{x}_0)^\top \mathbf{w} &= 0, \quad &&\forall j \in  \mathcal{E}
\end{aligned}
$$
且等式约束的梯度集合 $\{\nabla c_j(\mathbf{x}_0)\}_{j\in \mathcal{E}}$ 线性无关, 则称该约束在 $\mathbf{x}_0$ 处满足 MFCQ.
- 可以证明，由 LICQ 可以推出 MFCQ, 但反之不然. 
- 若 MFCQ 成立, 同样可知 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}_0) = \mathcal{F}(\mathbf{x}_0)$. 


***Definition* (Linear Constraint Qualification, LCQ)** 若优化问题中的全部约束函数 $c_k(\mathbf{x}), k\in \mathcal{E} \cup \mathcal{I}$ 都是线性的, 则称线性约束品性满足. 
- 对于线性约束品性, 有 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}) = \mathcal{F}(\mathbf{x})$. 
- LP, QP 等优化问题自然满足线性约束品性. 
- LCQ 和 LICQ 直接一般没有必然关联.


### 1.4. Karush-Kuhn-Tucker (KKT) Conditions

回顾含约束问题(不要求凸)的几何最优性条件: 对于局部最优解 $\mathbf{x}^*$ 和可行域 $\mathcal{X}$, 则任意可行方向 $\mathbf{d}$ 都满足:
$$
\mathbf{d}^\top \nabla f(\mathbf{x}^*) \geq 0, \quad \forall \mathbf{d} \in \mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) \quad (1)
$$

而我们也同样讨论, 这个最优性条件的求解是困难的. 转而我们考虑 Linear Feasible Direction Cone 的定义:
$$
\mathcal{F}(\mathbf{x}^*) = \left\{ \mathbf{d} \in \mathbb{R}^n \mid \mathbf{d}^\top \nabla c_j(\mathbf{x}^*) = 0, \forall j \in \mathcal{E} ~;~ \mathbf{d}^\top \nabla c_i(\mathbf{x}^*) \leq 0, \forall i \in \mathcal{A}(\mathbf{x}^*) \cap \mathcal{I} \right\} \quad (2)
$$
但也同时指出 $\mathcal{F}(\mathbf{x}^*)$ 并不能直接指定 $\mathbf{x}^*$ 处的最优性条件. 因此我们将验证一些约束品性(Constraint Qualifications), 当 CQ 满足时往往将有 $\mathcal{F}(\mathbf{x}^*) = \mathcal{T}_{\mathcal{X}}(\mathbf{x}^*)$ 作为桥梁. 这里使用的 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) = \mathcal{F}(\mathbf{x}^*)$ 是一个较强但常用的充分条件.

因此, 对于既是最优点, 又满足例如 LICQ 的约束品性时, 则 $(1), (2)$ 将同时成立, 故换言之, 下述集合为空集:
$$
\left\{ \mathbf{d} \ \middle| \
\begin{aligned}
\mathbf{d}^\top \nabla f(\mathbf{x}^*) &\lt 0, \\
\mathbf{d}^\top \nabla c_i(\mathbf{x}^*) &\leq 0, \forall i \in \mathcal{A}(\mathbf{x}^*) \cap \mathcal{I} \\
\mathbf{d}^\top \nabla c_j(\mathbf{x}^*) &= 0, \forall j \in \mathcal{E}
\end{aligned}
\right\} = \emptyset
$$

- 这意味着, 在局部最小点 $\mathbf{x}^*$ 处, 不存在一个可行方向 $\mathbf{d}$ 同时满足:
    - 一阶可行 (即后两个条件) , 使得 active set 的约束仍然不违反;
    - $\mathbf{d}^\top \nabla f(\mathbf{x}^*) < 0$ (第一个条件), 即该方向是下降方向.
- 然而这一条件的判断仍然不够直接, 下述引理会进一步改进. 

***Lemma* (Farkas' Lemma)**: 设 $p,q$ 是两个非负整数, 给定向量组 $\left \{ \mathbf{a}_i\in \mathbb{R}^n \right \}_{i=1}^p$ 和 $\left \{ \mathbf{b}_j\in \mathbb{R}^n \right \}_{j=1}^q$, 以及 $\mathbf{c}\in \mathbb{R}^n$, 如下两组命题恰有其一成立:

(1) 存在 $\mathbf{d}\in \mathbb{R}^n$ 使得如下条件同时成立:
- $\mathbf{d}^\top \mathbf{a}_i = 0, \forall i = 1, \ldots, p \quad \text{\small (F.L.1)}$
- $\mathbf{d}^\top \mathbf{b}_j \ge 0, \forall j = 1, \ldots, q \quad \text{\small (F.L.2)}$
- $\mathbf{d}^\top \mathbf{c} < 0 \quad \text{\small (F.L.3)}$

(2) 存在 $\left \{ \lambda_i \right \}_{i=1}^p \in \mathbb{R}$ 和 $\left \{ \mu_j \right \}_{j=1}^q \in \mathbb{R}_{\geq 0}$ 使得:
$$
\mathbf{c} = \sum_{i=1}^p \lambda_i \mathbf{a}_i + \sum_{j=1}^q \mu_j \mathbf{b}_j \quad \text{\small (F.L.4)}
$$

- *Proof*
    - 若 $\text{\small (F.L.4)}$ 成立, 对其左右两侧同时乘以 $\mathbf{d}^\top$, 则有:
        $$
        \mathbf{d}^\top \mathbf{c} = \sum_{i=1}^p \lambda_i \mathbf{d}^\top \mathbf{a}_i + \sum_{j=1}^q \mu_j \mathbf{d}^\top \mathbf{b}_j
        $$
      -  此时对于满足 $\text{\small (F.L.1)}$ 和 $\text{\small (F.L.2)}$ 的 $\mathbf{d}$, 其能推出 $\mathbf{d}^\top \mathbf{c} \geq 0$, 此时证明 $\text{\small (F.L.3)}$ 不成立. 
    - 若 $\text{\small (F.L.1)}\sim \text{\small (F.L.3)}$ 解不存在, 则用反证法结合分离超平面定理, 可以推出 $\text{\small (F.L.4)}$ 成立. 

---

对照前述的空集条件, 可与 Farkas 引理中的 (1) 对齐: 取 $\mathbf{c}=\nabla f(\mathbf{x}^*)$, $\mathbf{a}_j=\nabla c_j(\mathbf{x}^*)$ ($j\in\mathcal{E}$), $\mathbf{b}_i=-\nabla c_i(\mathbf{x}^*)$ ($i\in \mathcal{A}(\mathbf{x}^*)\cap\mathcal{I}$). 由 $\text{\small (F.L.4)}$ 得
$$
\nabla f(\mathbf{x}^*) = \sum_{j\in\mathcal{E}} \lambda_j^* \nabla c_j(\mathbf{x}^*) - \sum_{i\in\mathcal{I}\cap \mathcal{A}(\mathbf{x}^*)} \mu_i^* \nabla c_i(\mathbf{x}^*)
$$
- 其中 $\lambda_j^* \in \mathbb{R}, j\in \mathcal{E}$ 和 $\mu_i^* \ge 0, i\in \mathcal{I}\cap \mathcal{A}(\mathbf{x}^*)$.
- 由于等式乘子符号自由, 重新命名后可写成常见 stationarity 形式:
$$
\nabla f(\mathbf{x}^*) + \sum_{j\in\mathcal{E}} \lambda_j^* \nabla c_j(\mathbf{x}^*) + \sum_{i\in\mathcal{I}\cap \mathcal{A}(\mathbf{x}^*)} \mu_i^* \nabla c_i(\mathbf{x}^*) = 0
$$

若进一步补充, 对于 $i\in \mathcal{I}\setminus \mathcal{A}(\mathbf{x}^*)$ 的部分 (即 inactive 的不等式约束), 令 $\mu_i^* = 0$, 则有:
$$
\begin{aligned}
\nabla f(\mathbf{x}^*) + \sum_{j\in\mathcal{E}} \lambda_j^* \nabla c_j(\mathbf{x}^*) + \sum_{i\in\mathcal{I}} \mu_i^* \nabla c_i(\mathbf{x}^*) &= 0\\
\mu_i^* c_i(\mathbf{x}^*) &= 0,\quad \forall i\in \mathcal{I}
\end{aligned}
$$
  
  - 其中 $\lambda_j^* \in \mathbb{R}, j\in \mathcal{E}$ 和 $\mu_i^* \ge 0, i\in \mathcal{I}$.

第二个条件 $\mu_i^* c_i(\mathbf{x}^*) = 0, \forall i\in \mathcal{I}$ 也称为 *Complementary Slackness Condition (CSC)*. 

- 其表示对于 inactive 的不等式约束, 其对应的 multiplier 为 $0$; 对于不为 $0$ 的 multiplier, 其约束一定是 active 的 (即 $c_i(\mathbf{x}^*) = 0$).
- 对于 CSC, 若能够保证 $\mu_i^* = 0$ 和 $c_i(\mathbf{x}^*) = 0$ 有且仅有其一成立, 则说明当前的约束是严格互补松弛的 (Strict Complementary Slackness Condition, SCSC). 一般满足 SCSC 的约束的最优点具有良好性质, 算法收敛速度较快.

> [!example] **不满足 SCSC 的例子**
>
> 考虑如下问题:
> $$
> \min_{x\in \mathbb{R}} \quad x^2 ,\quad \text{s.t.} \quad x \leq 0.
> $$
> 最优解是 $x^* = 0$, 并且该位置对于约束来说也是 active 的. 
>
> 另一方面, 考虑 KKT stationarity 条件, 其要求:
> $$
> \nabla f(x^*) + \mu^* \nabla c(x^*) = 0 \implies (2x^* + \mu^*) \mid_{x^*=0} = 0 \implies \mu^* = 0.
> $$
> 综上, 该问题是 active 的, 但同时 $\mu^* = 0$, 不满足 SCSC. 其直观是, 这个约束虽然卡在边界上, 但并没有阻止目标函数减小. 因为在该点的梯度本来就是 $0$, 本身便不需要额外的约束来限制梯度. 

---

综上, 总结出如下一阶必要条件, 即 KKT 条件, 并称满足 $(\mathbf{x}^*, \lambda_j^*, \mu_i^*)$ 为 KKT 点.

***Theorem* (KKT Conditions)** 考虑如下约束优化问题 (不要求是凸的):
$$
\begin{aligned}
\min_{\mathbf{x}\in \mathbb{R}^n} & \quad f(\mathbf{x}) \\
\text{subject to} & \quad c_i(\mathbf{x}) \leq 0, i\in \mathcal{I} \\
& \quad c_j(\mathbf{x}) = 0, j\in \mathcal{E}
\end{aligned}
$$
对于局部最优解 $\mathbf{x}^* \in \mathcal{X}$, 若 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) = \mathcal{F}(\mathbf{x}^*)$ 成立, 则存在 Lagrange Multiplier $\lambda_j^*, \mu_i^*$ 使得如下条件成立:
$$
\begin{aligned}
&\text{\small{Stationarity:}}  &&  \nabla f(\mathbf{x}^*) + \sum_{j\in\mathcal{E}} \lambda_j^* \nabla c_j(\mathbf{x}^*) + \sum_{i\in\mathcal{I}} \mu_i^* \nabla c_i(\mathbf{x}^*) = 0\\
&\text{\small{Primal Feasibility 1:}}  && c_i(\mathbf{x}^*) \leq 0, \quad \forall i\in \mathcal{I} \\
&\text{\small{Primal Feasibility 2:}}  && c_j(\mathbf{x}^*) = 0, \quad \forall j\in \mathcal{E} \\
&\text{\small{Dual Feasibility:}} &&  \mu_i^* \geq 0, \quad \forall i\in \mathcal{I} \\
&\text{\small{Complementary Slackness:}} &&  \mu_i^* c_i(\mathbf{x}^*) = 0, \quad \forall i\in \mathcal{I}
\end{aligned}
$$
- 这里的 Stationarity 条件是前述 Farkas' Lemma 的直接推论, 其代表最优点处不存在一阶下降的可行方向. 一般也将其记为 $\nabla_{\mathbf{x}} L(\mathbf{x}^*, \lambda_j^*, \mu_i^*) = 0$.

- 需要指出, 该条件成立是建立在 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) = \mathcal{F}(\mathbf{x}^*)$ 成立的前提下的. 这是一个较强假设, 通常由 LICQ、MFCQ、LCQ 等 CQ 保证. 因此 KKT 是一个必要条件, 满足 KKT 条件并不一定是最优点. 


## 2. Second-Order Optimality Conditions

对 KKT 点而言, 结合 Stationarity、Dual Feasibility 与 Complementary Slackness 可推出 $\mathbf{d}^\top \nabla f(\mathbf{x}^*) \geq 0, \forall \mathbf{d} \in \mathcal{F}(\mathbf{x}^*)$ (类比 $y=x^3$ 的这种驻点但非局部极值点, 其排除所有能够在一阶情况下让目标函数减小的可行方向). 下面需要通过二阶最优性条件来进一步判断最优点.

***Definition* (Critical Cone)**: 设 $(\mathbf{x}^*, \lambda_j^*, \mu_i^*)$ 是 KKT 点, 其 Critical Cone 定义为:
$$
\mathcal{C}(\mathbf{x}^*; \lambda_j^*, \mu_i^*) = \left\{ \mathbf{d} \in \mathcal{F}(\mathbf{x}^*) \mid \nabla c_i(\mathbf{x}^*)^\top \mathbf{d} = 0, ~ \forall i \in \mathcal{A}(\mathbf{x}^*) \cap \mathcal{I}\ \text{with}\ \mu_i^* \gt 0\right\}
$$
在 KKT 条件下也等价于:
$$
\mathcal{C}(\mathbf{x}^*; \lambda_j^*, \mu_i^*) = \left\{ \mathbf{d} \in \mathcal{F}(\mathbf{x}^*) \mid \mathbf{d}^\top \nabla f(\mathbf{x}^*) = 0\right\}
$$


- Critical Cone 的 intuition 如下:
    - Critical cone 作为 $\mathcal{F}(\mathbf{x}^*)$ 的子集, 其继承 $\mathcal{F}(\mathbf{x}^*)$ 的性质, 在等式约束下 $\nabla c_j(\mathbf{x}^*)^\top \mathbf{d} = 0,~ j \in \mathcal{E}$; 在活跃不等式约束下 $\nabla c_i(\mathbf{x}^*)^\top \mathbf{d} \leq 0,~ i \in \mathcal{A}(\mathbf{x}^*) \cap \mathcal{I}$.
    - 同时, 由于满足 KKT 的 Stationarity 条件, 等式左右同乘 $\mathbf{d}^\top$, 有:
        $$
        \mathbf{d}^\top \nabla f(\mathbf{x}^*) + \sum_{j\in\mathcal{E}} \lambda_j^* \mathbf{d}^\top \nabla c_j(\mathbf{x}^*) + \sum_{i\in\mathcal{I}} \mu_i^* \mathbf{d}^\top \nabla c_i(\mathbf{x}^*) = 0
        $$
        - 由 $\mathcal{F}(\mathbf{x}^*)$ 的定义, 全部等式约束均有 $\mathbf{d}^\top \nabla c_j(\mathbf{x}^*) = 0$; 由互补松弛性全部非活跃不等式约束均有 $\mu_i^* = 0$. 综合后, 得到
            $$
            \mathbf{d}^\top \nabla f(\mathbf{x}^*) + \sum_{i\in\mathcal{I}\cap \mathcal{A}(\mathbf{x}^*)} \mu_i^* \mathbf{d}^\top \nabla c_i(\mathbf{x}^*) = 0 \quad (\dagger)
            $$
        - 分析该条件, 已知 $\mu_i^* \ge 0$, $\nabla c_i(\mathbf{x}^*)^\top \mathbf{d} \leq 0$, 因此 $\mathbf{d}^\top \nabla f(\mathbf{x}^*) \geq 0$.

    - 综上, 由 KKT + $\mathcal{F}(\mathbf{x}^*)$ 的定义, 目前得到的集合为 $\{ \mathbf{d} \in \mathcal{F}(\mathbf{x}^*) \mid \mathbf{d}^\top \nabla f(\mathbf{x}^*) \geq 0 \}$.  即所有一阶线性可行方向上, 目标的一阶变化都不可能是负的. 既然如此, 进一步讨论两种情况:
        - 若 $\nabla f(\mathbf{x}^*)^\top \mathbf{d} \gt 0$, 说明一阶情况下该方向立即会导致目标函数增加, 故可以直接忽略;
        - 若 $\nabla f(\mathbf{x}^*)^\top \mathbf{d} = 0$, 说明该方向是一阶线性可行方向, 但是否能够在完整情况下同样确保最小, 这是在之前的一阶条件下没办法判断, 而需要进一步研究的. 
    - 因此在概念上, Critical Cone 即在上述基础上, 提取 $\nabla f(\mathbf{x}^*)^\top \mathbf{d} = 0$ 的那些方向作为二阶情况研究的基本对象. 
    - 其也等价于如下命题: $\nabla c_i(\mathbf{x}^*)^\top \mathbf{d} = 0,~ \mu_i^* \gt 0,~  \forall i \in \mathcal{A}(\mathbf{x}^*) \cap \mathcal{I}$ , 下面将说明这两个命题是等价的.
        - 根据 $(\dagger)$ 式, 可以得到 $\nabla f(\mathbf{x}^*)^\top \mathbf{d} = - \sum_{i\in\mathcal{I}\cap \mathcal{A}(\mathbf{x}^*)} \mu_i^* \nabla c_i(\mathbf{x}^*)^\top \mathbf{d}$. 故欲让 $\nabla f(\mathbf{x}^*)^\top \mathbf{d} = 0$, 则需要 RHS 的$- \sum_{i\in\mathcal{I}\cap \mathcal{A}(\mathbf{x}^*)} \mu_i^* \nabla c_i(\mathbf{x}^*)^\top \mathbf{d} = 0$, 即要求全部 $\mu_i^* \gt 0$ 的不等式约束都满足 $\nabla c_i(\mathbf{x}^*)^\top \mathbf{d} = 0$.
    - 综上, critical cone 就是在讨论线性化可行方向中, 那些根据一阶梯度信息无法判断是否为上升下降方向的线性化可行方向. 

***Theorem* (Second-Order Optimality Necessary Condition)**: 假设 $f$ 与全部活跃约束在 $\mathbf{x}^*$ 邻域二阶连续可微, 且 $\mathbf{x}^*$ 是局部最小值, $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) = \mathcal{F}(\mathbf{x}^*)$ 成立, $(\mathbf{x}^*, \lambda_j^*, \mu_i^*)$ 是 KKT 点, 则有:
$$
\mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^*, \lambda_j^*, \mu_i^*) \mathbf{d} \geq 0, \quad \forall \mathbf{d} \in \mathcal{C}(\mathbf{x}^*; \lambda_j^*, \mu_i^*)
$$

***Theorem* (Second-Order Optimality Sufficient Condition)**: 假设对于可行点 $\mathbf{x}^*$, 存在一个 Lagrange Multiplier $(\lambda_j^*, \mu_i^*)$ 使得 KKT 条件成立 (通常配合 LICQ 等 CQ). 如果
$$
\mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^*, \lambda_j^*, \mu_i^*) \mathbf{d} \gt 0, \quad \forall \mathbf{d} \in \mathcal{C}(\mathbf{x}^*; \lambda_j^*, \mu_i^*), ~\mathbf{d} \neq \mathbf{0}
$$
则 $\mathbf{x}^*$ 是严格局部最小值. 

> [!note] 
> 
> 1. 上述充分条件和必要条件并不互为充要. 必要条件允许半正定的退化情景, 然而充分条件是严格正定的.
> 2. 二阶最优性条件也同样需要某种正定性的保证, 但其只需要在 critical cone 中成立, 而无需考虑在整个可行域内.

> [!example] 考虑如下一个具体的二元约束优化问题:
> $$
> \begin{aligned}
> &\min_{x_1, x_2} \quad x_1^2 + x_2^2 \\
> &\text{subject to} \quad x_1^2/4 + x_2^2 - 1 = 0
> \end{aligned}
> $$
> 
> - 其 Lagrangian 为:
>   $$
>   L(x_1, x_2, \lambda) = x_1^2 + x_2^2 + \lambda (x_1^2/4 + x_2^2 - 1)
>   $$
>
> - 对于可行域内任意一点 $(x_1, x_2)$, 其线性方向可行锥可求解如下:
>   - 首先求解等式约束的梯度:
>     $$
>     \nabla c(x_1, x_2) = \begin{pmatrix}
>     {\partial c}/{\partial x_1} \\
>     {\partial c}/{\partial x_2}
>     \end{pmatrix} = \begin{pmatrix}
>     {x_1}/{2} \\
>     2x_2
>     \end{pmatrix}
>     $$
> 
>   - 根据定义, 线性方向可行锥为: $\mathcal{F}(\mathbf{x}^*) = \left\{ \mathbf{d} \in \mathbb{R}^2 \mid \nabla c(\mathbf{x}^*)^\top \mathbf{d} = 0 \right\}$. 故:
>     $$
>     \mathcal{F}(x_1, x_2) = \left\{ (d_1, d_2) \mid \frac{x_1}{2} d_1 + 2x_2 d_2 = 0 \right\}
>     $$
> 
> - 由于只有一个等式约束, 且该约束梯度非零, 故 LICQ 成立. 
>
> - 求解 Critical Cone $\mathcal{C}(\mathbf{x}^*; \lambda_j^*, \mu_i^*) = \left\{ \mathbf{d} \in \mathcal{F}(\mathbf{x}^*) \mid \mathbf{d}^\top \nabla f(\mathbf{x}^*) = 0\right\}$:
>   - 首先求原函数之梯度: 
>     $$
>     \nabla f(x_1, x_2) = \begin{pmatrix}
>     {\partial f}/{\partial x_1} \\
>     {\partial f}/{\partial x_2}
>     \end{pmatrix} = \begin{pmatrix}
>     2x_1 \\
>     2x_2
>     \end{pmatrix}
>     $$
> 
>   - 根据定义, Critical Cone 为: 
>     $$
>     \mathcal{C}(x_1, x_2; \lambda_j^*, \mu_i^*) = \left\{ \frac{x_1}{2}d_1 + 2x_2 d_2 = 0, x_1 d_1 + x_2 d_2 = 0 \right\}
>     $$
>
> - 求解 KKT 点, 由 Stationarity 条件: 
>   $$
>   \begin{cases}
>   \frac{\partial L}{\partial x_1} = 2x_1 + \frac{\lambda_1}{2} x_1 = 0 \\
>   \frac{\partial L}{\partial x_2} = 2x_2 + 2\lambda_1 x_2 = 0 \\
>   \frac{\partial L}{\partial \lambda} = \frac{x_1^2}{4} + x_2^2 - 1 = 0
>   \end{cases} \implies (x_1^*, x_2^*, \lambda_1^*) = (\pm2,0,-4){\text{\small{ or }}} (0,\pm1,-1)
>   $$
> - 讨论此处的 KKT 点是否满足二阶最优性条件. 
>   - 首先求解二阶导数:
>     $$
>     \nabla^2_{\mathbf{x}\mathbf{x}} L(x_1, x_2, \lambda_1) = \begin{pmatrix}
>     \frac{\partial^2 L}{\partial x_1^2} & \frac{\partial^2 L}{\partial x_1 \partial x_2}  \\
>     \frac{\partial^2 L}{\partial x_2 \partial x_1} & \frac{\partial^2 L}{\partial x_2^2} \\
>     \end{pmatrix} = \begin{pmatrix}
>     2 + \frac{1}{2} \lambda_1 & 0 \\
>     0 & 2 + 2\lambda_1 \\
>     \end{pmatrix}
>     $$
>   - 对于 $\mathbf{y}_1 = (2,0,-4)^\top$:
>     - 其二阶梯度为:
>       $$
>       \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{y}_1) = \begin{pmatrix}
>       0 & 0 \\
>       0 & -6 \\
>       \end{pmatrix}
>       $$
>     - 代入 Critical Cone 条件 $\mathcal{C}(\mathbf{y}_1)$ 有:
>       $$
>       \mathcal{C}(\mathbf{y}_1) = \left\{ \frac{2}{2}d_1 + 2\cdot 0 \cdot d_2 = 0, d_1 \cdot 2 + d_2 \cdot 0 = 0 \right\} = \left\{ d_1 = 0 \right\}
>       $$
>     - 显然 $\mathbf{y}_1$ 不满足局部最优的二阶必要条件.
>       - 例如, 取 $\mathbf{d} = (0,1)^\top \in \mathcal{C}(\mathbf{y}_1)$, 则 $\mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{y}_1) \mathbf{d} = -6 \lt 0$ 不满足二阶必要条件.
>
>   - 对于 $\mathbf{y}_2 = (0,1,-1)^\top$:
>     - 类似地, 其二阶梯度为:
>       $$
>       \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{y}_2) = \begin{pmatrix}
>       3/2 & 0 \\
>       0 & 0 \end{pmatrix}
>       $$
>     - 代入 Critical Cone 条件 $\mathcal{C}(\mathbf{y}_2)$ 有:
>       $$
>       \mathcal{C}(\mathbf{y}_2) = \left\{ 0d_1 + 2\cdot 1 \cdot d_2 = 0, d_1 \cdot 0 + d_2 \cdot 1 = 0 \right\} = \left\{ d_2 = 0 \right\}
>       $$
>     - $\mathbf{y}_2$ 满足局部最优的二阶必要条件.
>       - 对于任意的 $\mathbf{d}\neq \mathbf{0} \in \mathcal{C}(\mathbf{y}_2) \Leftrightarrow (d_1, 0)^\top$ , 其中 $d_1 \neq 0$, 有 $\mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{y}_2) \mathbf{d} = 3d_1^2/2 \gt 0$ 满足二阶必要条件.
>
> - 综上, $\mathbf{y}_1$ 不满足局部最优的二阶必要条件, $\mathbf{y}_2$ 满足局部最优的二阶必要条件. 
