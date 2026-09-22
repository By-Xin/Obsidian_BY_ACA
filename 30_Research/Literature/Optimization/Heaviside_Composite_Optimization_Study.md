# Heaviside Composite Optimization Problems (HSCOPs)
<!-- 
> - Jong-Shi Pang, Jul 2026

## 1. Introduction


Heaviside function, 即阶跃函数或阈值函数, 最开始在信号处理中提出, 并近来在优化问题中得到了广泛的应用. 一般地, 考虑如下两种 Heaviside function:

- *Closed Heaviside function*: 
    $$
    H_{\text{cl}}(s) := \mathbb{1}_{[0, \infty)}(s) = \begin{cases}
    1, & s \geq 0 \\
    0, & s < 0
    \end{cases}
    $$
    - **upper semi-continuous**: $\lim_{s \to 0^-} H_{\text{cl}}(s) = 0$, $\lim_{s \to 0^+} H_{\text{cl}}(s) = 1$. 
    - 最大化问题中常希望目标函数是 closed Heaviside (因为此时 $H_{\text{cl}}(s) = 1$ 的部分是闭集, 便于求解).
- *Open Heaviside function*:
    $$
    H_{\text{op}}(s) := \mathbb{1}_{(0, \infty)}(s) = \begin{cases}
    1, & s > 0 \\
    0, & s \leq 0
    \end{cases}
    $$
    - **lower semi-continuous**: $\lim_{s \to 0^-} H_{\text{op}}(s) = 0$, $\lim_{s \to 0^+} H_{\text{op}}(s) = 1$.
    - 最小化问题中常希望目标函数是 open Heaviside.

![H_thresholds_panel](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/H_thresholds_panel.png)

可进一步定义 composite Heaviside function (CHF). 给定 $\mathbf{x} \in \mathbb{R}^n$, 以及 $K$ 个函数 $\phi_k: \mathbb{R}^n \to \mathbb{R}$, 与 $\psi_k: \mathbb{R}^n \to \mathbb{R}$, 则 composite Heaviside function 定义为:
$$
H_{\text{comp}}(\mathbf{x}) := \sum_{k=1}^{K} \psi_k(\mathbf{x}) H(\phi_k(\mathbf{x})),
$$

说明:
- CHF 表示了一种 *conditional function*. 对于其中的每一项, 其表示:
    $$
    \psi_k(\mathbf{x}) H(\phi_k(\mathbf{x})) =
    \begin{cases}
    \psi_k(\mathbf{x}), & \text{if } \phi_k(\mathbf{x}) \geq 0 \\
    0, & \text{if } \phi_k(\mathbf{x}) < 0
    \end{cases}
    $$
    - 这种形式在神经网络中类似于激活函数的作用.
- $\psi_k(\mathbf{x})$ 通常是 friendly 的, 但 $\phi_k(\mathbf{x})$ 可以是任意的, 甚至是不可微的. 
- 经过 composite Heaviside function 的处理, 其可以表示任意有限跳跃的 step functions. 

故给出 general 的 Heaviside Composite Optimization Problem (HSCOP) 的定义如下:
$$
\begin{aligned}
\max_{\mathbf{x}\in P} \quad & \theta_{\text{GHS}}(\mathbf{x})
\triangleq c(\mathbf{x}) + \sum_{k=1}^{K_0} \psi_{0k}(\mathbf{x}) \mathbf{1}_{[0,\infty)}\bigl(\phi_{0k}(\mathbf{x})\bigr) \\[1mm]
\text{s.t.} \quad & \mathbf{A}_{i,} \mathbf{x} + \sum_{k=1}^{K_i} \psi_{ik}(\mathbf{x}) \mathbf{1}_{[0,\infty)}\bigl(\phi_{ik}(\mathbf{x})\bigr) \geq \eta_i, \quad i=1,\ldots,I.
\end{aligned}
$$

其中 $\mathbf{x} \in \mathbb{R}^n$ 是决策变量, $P \subseteq \mathbb{R}^n$ 是 *friendly set*, $c(\cdot)$ 是 *friendly function* 如 affine 或 quadratic, $\mathbf{A} \in \mathbb{R}^{I \times n}$, $\mathbf{A}_{i,}$ 是 $\mathbf{A}$ 的第 $i$ 行. 并且注意这里的 indicator function 是 closed Heaviside function, 因为这是一个 maximization problem. 

***Example* ($\ell_0$-norm minimization)**: 对于 $\ell_0$ function, 其可以表达为 $|\mathbf{x}|_0 := \# \{i: x_i \neq 0\} = \mathbb{1}_{(0, \infty)}(|\mathbf{x}|) = 1 - \mathbb{1}_{[0, \infty)}(|\mathbf{x}|)$, 这便是一个内层不可微的 composite Heaviside function. 更进一步, 若考虑 $n$ 个变量的 mixed-sign combination:
$$
\sum_{j = 1}^{n} a_j |x_j|_0 \leq b
$$
其中 $a_j \in \mathbb{R}$, 则由于 $a_j$ 的符号不定, 导致其函数可能既不是 lower semi-continuous 也不是 upper semi-continuous, 这边是后续分析的一个重点和难点来源. 

总结一下, HSCOPs 在处理中将会遇到一下几个难点:
- Discontinuity and non-differentiability: Heaviside function 本身是 discontinuous 的. 而与之复合的其他函数同时也有可能是不光滑的. 
- Lack of semi-continuity in objective: 由于 $\psi_k(\mathbf{x})$ 的符号不定, 导致 objective function 可能既不是 lower semi-continuous 也不是 upper semi-continuous, 此时 feasible set 可能不是 closed 的, 则最优值未必 attainable. 
- 凸分析的传统工具在当前非连续等情况下未必完全适用, 需要重新界定. 
- 全局最优解此时可能不存在, 需要考虑局部最优解的定义和性质.
- 宏观上, 其既有连续的部分, 也有离散的部分, 类似于二者的一个 fusion 地带, 但又有别于传统的 MIP. 这类问题的求解方法和理论分析都需要新的思路.



## 2. Preliminaries

下给出一些基本的定义和符号, 以便后续分析.

***Definition* (Bouligand differentialbility)**: 给定 $f: \mathcal{O} \subseteq \mathbb{R}^n \to \mathbb{R}^m$ 其中 $\mathcal{O}$ 是一个 open set, 称 $f$ 在点 $\mathbf{\bar{x}} \in \mathcal{O}$ 上是 Bouligand differentiable (简称 B-differentiable), 若同时满足:
1. $f$ 在 $\mathbf{\bar{x}}$ 附近 locally Lipschitz continuous:
    $$
    \exists L > 0, \exists \delta > 0, \forall \mathbf{x}, \mathbf{y} \in B(\mathbf{\bar{x}}, \delta), \quad
     |f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.
    $$
2. 对任一方向 $\mathbf{v} \in \mathbb{R}^n$, 方向导数
    $$
    f'(\mathbf{\bar{x}}; \mathbf{v}) := \lim_{\tau \downarrow 0} \frac{f(\mathbf{\bar{x}} + \tau \mathbf{v}) - f(\mathbf{\bar{x}})}{\tau}
    $$
    存在且有穷. 

说明: 
- 开集是避免了边界点的影响, 以便在任意方向进行方向导数的计算.
- Local Lipschitz continuity 排除了函数在该点附近的剧烈变化 (如剧烈震荡, 跳跃等)
- Bouligand differentiability 只要求任意方向的导数均在, 但不要求存在一个统一的 gradient 对应于所有方向, 故有别于传统的 Fréchet differentiability. 因而 partial differentiability 一般也不成立:
    $$
    f'(\mathbf{\bar{x}}; \mathbf{v}) \neq \sum_{i=1}^{n} f'(\mathbf{\bar{x}}; v_i \mathbf{e}_i),  
    $$
    其中 $\mathbf{e}_i$ 是标准基向量, $v_i$ 是 $\mathbf{v}$ 的第 $i$ 个分量.
- 一般的 calculus rules, 包括 加减乘除, chain rule 等, 在 B-differentiable 的情况下仍然成立. 


***Definition* (Piecewise Affine (PA) function)**: 给定连续函数 $f: \mathbb{R}^n \to \mathbb{R}^m$, 若 $f$ 能将 $\mathbb{R}^n$ 分割为有限个 polyhedral sets, 且在每个 polyhedral set 上 $f$ 是 affine, 则称 $f$ 是 piecewise affine (PA) function. 

- 特别地, 对于 $m=1$, $f$ 可以表示为:
    $$
    f(\mathbf{x}) = \max_{1 \leq k \leq K} [{\mathbf{a}_k}^\top \mathbf{x} + \alpha_k]  - \max_{1 \leq l \leq L} [{\mathbf{b}_l}^\top \mathbf{x} + \beta_l],
    $$
    其中 $\mathbf{a}_k, \mathbf{b}_l \in \mathbb{R}^n$, $\alpha_k, \beta_l \in \mathbb{R}$. 

- 任何 PA function 都是处处 B-differentiable 的. 

- 对于 PA function $f$, 对于任意 $\mathbf{\bar{x}} \in \mathbb{R}^n$, 其可以被任意足够近的点 $\mathbf{x}$ 精确表示:
    $$
    f(\mathbf{x}) = f(\mathbf{\bar{x}}) + f'(\mathbf{\bar{x}}; \mathbf{x} - \mathbf{\bar{x}}), \quad \forall \mathbf{x} \text{\small{ sufficiently close to }} \mathbf{\bar{x}}.
    $$

下考虑优化问题的稳定性条件. 一般而言, 若 $f$ 是 B-differentiable 的, 则可以定义 Bouligand stationary (B-stationary) point, 其是局部最优解的一个必要条件.

***Definition* (Bouligand stationary)**: 考虑优化问题 $\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$, 其中 $f$ 是 B-differentiable 的. 称 $\mathbf{\bar{x}} \in \mathcal{X}$ 是 Bouligand stationary (B-stationary) point, 若满足:
$$
f'(\mathbf{\bar{x}}; \mathbf{v}) \geq 0, \quad \forall \mathbf{v} \in \mathcal{T}_{\mathcal{X}}(\mathbf{\bar{x}}),
$$
其中 $\mathcal{T}_{\mathcal{X}}(\mathbf{\bar{x}})$ 是 $\mathcal{X}$ 在 $\mathbf{\bar{x}}$ 处的 tangent cone, 定义为:
$$
\mathcal{T}_{\mathcal{X}}(\mathbf{\bar{x}}) := \left\{ \mathbf{v} \in \mathbb{R}^n : \mathbf{v} = \lim_{k \to \infty} \frac{\mathbf{x}^k - \mathbf{\bar{x}}}{\tau_k}, {\small\text{ for some sequence }} \{\mathbf{x}^k\} \subseteq \mathcal{X} {\small\text{converging to }} \mathbf{\bar{x}}
\land ~ \tau_k \downarrow 0 \right\}.
$$

- 本质上, Tangent cone 表示从 $\mathbf{\bar{x}}$ 出发, 沿着可行集可以产生的瞬时移动方向. 

若 $f$ 连方向导数都不存在, 则可以考虑如下 lifted problem:
$$
\begin{aligned}
\min_{(t, \mathbf{x}) \in \mathbb{R} \times \mathcal{X}} \quad & t \\
\text{s.t.} \quad & f(\mathbf{x}) - t \leq 0.
\end{aligned}
$$ 
即考虑 $f$ 的 epigraph. Lifted problem 与 $\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$ 是等价的, 但其将目标函数的复杂性转移到了约束几何上. 

***Definition* (Epigraphical stationary)**: 考虑优化问题 $\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$. 称 $\mathbf{\bar{x}} \in \mathcal{X}$ 是 epigraphical stationary point, 若 $(f(\mathbf{\bar{x}}), \mathbf{\bar{x}})$ 是 lifted problem 的 B-stationary point.

- 若 $f$ 是 B-differentiable 的, 则 $\mathbf{\bar{x}}$ 是 epigraphical stationary point $\iff$ $\mathbf{\bar{x}}$ 是 B-stationary point. 
- 若 $f$ 不是 B-differentiable 的, 则 epigraphical stationary point 的定义仍然成立, 而 B-stationary point 的定义则不再适用.

进一步, 如下两个局部几何性质将保证 B-stationary 与 local minimizer 的等价性. 

***Definition* (Locally star-shaped)**: 集合 $\mathcal{X} \subseteq \mathbb{R}^n$ 称为 locally star-shaped, 若对于任意足够靠近 $\mathbf{\bar{x}}$ 的点 $\mathbf{x} \in \mathcal{X}$, 存在与 $\mathbf{x}$ 相关的标量 $\bar{\tau}$, 使得对于任意 $\tau \in [0, \bar{\tau}]$, 有 $\mathbf{\bar{x}} + \tau (\mathbf{x} - \mathbf{\bar{x}}) \in \mathcal{X}$.

- Star-shaped property 是一个弱化的 convexity property:
  - 只要求从 $\mathbf{\bar{x}}$ 出发到周围的任何 *sufficiently close* 的点 $\mathbf{x}$, 沿着 $\mathbf{\bar{x}}$ 向 $\mathbf{x}$ 走出一小段距离 (而不需要需要全程) 在 $\mathcal{X}$ 内即可. 
    
  - 只需要考虑每个 $\mathbf{x}$ 各自和 $\mathbf{\bar{x}}$ 之间的关系, 而不需要考虑 $\mathbf{x}$ 之间的关系.
     - 注意, 这里的 $\mathbf{x}$ 不是在任意邻域内取到的, 而是在 $\mathcal{X}$ 内的, 且足够靠近 $\mathbf{\bar{x}}$ 的点. 例如考虑 $\mathcal{X}=\left\{(s,0):s\geq0\right\}\cup\left\{(0,s): s\geq0\right\}$ 即坐标轴的两条半轴, 则 $\mathbf{\bar{x}}=(0,0)$ 是 locally star-shaped 的, 但不是 convex 的. 其只考虑两条射线即可, 并不是稠密的. 
     ![star_shaped_not_convex](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/star_shaped_not_convex.png) 

>  这个 star 其实主要想说的是这种放射形的感觉, 不是那种五角星六芒星 (笑) ![starburst](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/starburst.png)


***Definition* (Locally convex-like)**: 集合 $\mathcal{X} \subseteq \mathbb{R}^n$ 称为  在 $\mathbf{\bar{x}}$ 处 locally convex-like, 若存在 $\mathbf{\bar{x}} \in \mathcal{X}$ 的邻域 $\mathcal{N}(\mathbf{\bar{x}})$, 使得 $\mathcal{X} \cap \mathcal{N}(\mathbf{\bar{x}}) \subseteq \mathbf{\bar{x}} + \mathcal{T}_{\mathcal{X}}(\mathbf{\bar{x}})$.
- locally convex-like property 是一个弱化的 star-shaped property, 其甚至不要求 $\mathbf{\bar{x}}$ 到 $\mathbf{x}$ 能够连续的走出一小段距离, 而只要求 $\mathbf{x}$ 在 $\mathbf{\bar{x}}$ 的 tangent cone 内即可.

    ![convex_like_not_star_shaped](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/convex_like_not_star_shaped.png)


对于 reference point $\mathbf{\bar{x}} \in \mathcal{X}$, 若 $\mathcal{X}$ 是 convex-like 的, 且 $f$ 是 B-differentiable 的且 
$$
f(\mathbf{x}) \geq f(\mathbf{\bar{x}}) + f'(\mathbf{\bar{x}}; \mathbf{x} - \mathbf{\bar{x}}) \qquad  (\star)
$$ 
对任意 sufficiently close 的 $\mathbf{x} \in \mathcal{X}$ 成立, 则 $\mathbf{\bar{x}}$ 是 local minimizer $\iff$ $\mathbf{\bar{x}}$ 是 B-stationary point.

*Proof Sketch*: 
- 由 locally convex-like: $\mathbf{x} - \mathbf{\bar{x}} \in \mathcal{T}_{\mathcal{X}}(\mathbf{\bar{x}})$
- 由 Bouligand stationary: $f'(\mathbf{\bar{x}}; \mathbf{x} - \mathbf{\bar{x}}) \geq 0$
- 由 $(\star)$: $f(\mathbf{x}) \geq f(\mathbf{\bar{x}}) + f'(\mathbf{\bar{x}}; \mathbf{x} - \mathbf{\bar{x}}) \geq f(\mathbf{\bar{x}})$

## 3. Paradigm changes: Sources of discontinuity

## 4. Solution Analysis: NLP Approach

### Existence of optimal solution

首先讨论解的存在性问题. 如下例子说明, 若不加任何限制, HSCOP 的最优解可能是不可达的.

***Ex le* (Non-attainable optimal value)**:

首先考虑如下 HSCOP, 其最优解是不可达的.
$$
    \begin{aligned}
      \min_{x_1,x_2} \quad & x_1^2+2(1-x_2)^2+|x_1|_0+\tfrac{1}{2}|x_2|_0 \\[2pt]
      \text{s.t.} \quad & |x_1|_0\geq|x_2|_0,\quad -1\leq x_1,x_2\leq 1.
    \end{aligned}
$$

分析可得, 该约束问题的可行域为 $\mathcal{X}_{\text{GHS}} = [-1,1]^2 \setminus \{(0, x_2): x_2 \neq 0\}$. 由于 slit $\{(0, x_2): x_2 \neq 0\}$ 的存在, 导致 $\mathcal{X}_{\text{GHS}}$ 不是 closed 的, 因而最优值不可达.

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/l0_nonattainable_3d.png)

为保证最优解的存在性, 需要额外条件限制. 一个重要的条件是 **upper semi-continuity**, 其保证了约束的 superlevel sets 是 closed 的.

***Proposition* (Upper semi-continuity)**: 给定连续函数 $\phi_{ij} : \mathbb{R}^n \to \mathbb{R}$, $\psi_{ij}: \mathbb{R}^n \to \mathbb{R}$, 其中 $i \in [I], j\in [J_i]$. 给定点 $\mathbf{\bar{x}} \in \mathbb{R}^n$, 若存在一个 neighborhood $\mathcal{N}(\mathbf{\bar{x}})$, 使得对于任意
$$
(i,j) \in  \mathcal{J}_0 := \{(i,j): \phi_{ij}(\mathbf{\bar{x}}) = 0 \geq \psi_{ij}(\mathbf{x}), \forall i,j\},
$$
下列条件之一成立:
- 对于所有 $\mathbf{x} \in \mathcal{N}(\mathbf{\bar{x}})$, 若 $\psi_{ij}(\mathbf{x}) < 0$ , $\phi_{ij}(\mathbf{x}) \geq 0$;
- 对于所有 $\mathbf{x} \in \mathcal{N}(\mathbf{\bar{x}})$, $\psi_{ij}(\mathbf{x}) \geq 0$.

则函数
$$
\mathbf{x} \mapsto  \sum_{j=1}^{J_i} \psi_{ij}(\mathbf{x}) \mathbb{1}_{[0,\infty)}(\phi_{ij}(\mathbf{x}))
$$
对于任意固定 $i \in [I]$ 都在 $\mathbf{\bar{x}}$ 处是 upper semi-continuous 的.

$\diamond$

说明: 

- 观察这里的每一个单项 $\psi_{ij}(\mathbf{x}) \mathbb{1}_{[0,\infty)}(\phi_{ij}(\mathbf{x}))$, 回顾其可以表示为:
    $$
    \psi_{ij}(\mathbf{x}) \mathbb{1}_{[0,\infty)}(\phi_{ij}(\mathbf{x})) =
    \begin{cases}
    \psi_{ij}(\mathbf{x}), & \text{if } \phi_{ij}(\mathbf{x}) \geq 0 \\
    0, & \text{if } \phi_{ij}(\mathbf{x}) < 0
    \end{cases}
    $$
    而又知, upper semi-continuous 是 $\lim\sup_{\mathbf{x} \to \mathbf{\bar{x}}} f(\mathbf{x}) \leq f(\mathbf{\bar{x}})$, 即允许函数值不低于附近的极限值.  


- 因此, 对于 closed Heaviside function, 若 $\psi_{ij}(\mathbf{\bar{x}}) < 0$, 则此时在 $\phi_{ij}(\mathbf{\bar{x}}) = 0$ 左右会存在一个非法的跳跃, 即 $\lim\sup_{\mathbf{x} \to \mathbf{\bar{x}}} \psi_{ij}(\mathbf{x}) \mathbb{1}_{[0,\infty)}(\phi_{ij}(\mathbf{x})) = 0 > \psi_{ij}(\mathbf{\bar{x}})$, 其破坏了 upper semi-continuity. 故要对这类 index $\mathcal{J}_0$ 进行限制排除.

- 集合 $\mathcal{J}_0$ 收集了所有的临界情况. 第一个 bullet 相当于直接去掉了 当 $\psi < 0$ 时 $\phi$ 负半轴的情况; 第二个 bullet 则直接要求 $\psi$ 非负. 故二者分别对应, 保证了 upper semi-continuity.

    ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/aa.png)


故在此基础上, 讨论 global maximizer 的存在性. 

***Proposition* (Existence of global maximizer)**:
对于最大化问题:
$$
\sup_{\mathbf{x}\in \mathcal{X}_{\text{GHS}}} \theta_{\text{GHS}} (\mathbf{x}) = c(\mathbf{x}) + \sum_{k=1}^{K_0} \psi_{0k}(\mathbf{x}) \mathbf{1}_{[0,\infty)}(\phi_{0k}(\mathbf{x}))
$$
假设:
- $\mathcal{X}_{\text{GHS}} \neq \varnothing$ (not necessarily closed)
- $c$ 是连续的, 且 $-c$ 在 $\mathcal{X}_{\text{GHS}}$ 上是 coercive 的, 即 $\lim_{\|\mathbf{x}\| \to \infty, \mathbf{x} \in \mathcal{X}_{\text{GHS}}} c(\mathbf{x}) = -\infty$.
- 最优值有限: $\sup_{\mathbf{x}\in \mathcal{X}_{\text{GHS}}} \theta_{\text{GHS}} (\mathbf{x}) \in (-\infty, \infty)$

若存在可行点列 $\{\mathbf{x}^\nu\} \subseteq \mathcal{X}_{\text{GHS}}$, 其函数值趋近于最优值 $\lim_{\nu \to \infty} \theta_{\text{GHS}} (\mathbf{x}^\nu) = \sup_{\mathbf{x}\in \mathcal{X}_{\text{GHS}}} \theta_{\text{GHS}} (\mathbf{x})$. 则对于任意满足上述 Upper Semi-continuity Proposition 的条件的 accumulation point $\mathbf{x}^\infty$ (即子列收敛极限点, 其存在性定被保证), 都是 $\theta_{\text{GHS}}$ 在 $\mathcal{X}_{\text{GHS}}$ 上的 global maximizer.


*Proof Sketch*:

- 由 coercive property, 可行点列 $\{\mathbf{x}^\nu\}$ 是 bounded 的, 而有限维空间中的有界序列定存在收敛子列, 故 $\mathbf{x}^\infty$ 存在性得证. 
- 由 Upper Semi-continuity Proposition, 保证即使 $\mathcal{X}_{\text{GHS}}$ 不是 closed 的, 也有 $\mathbf{x}^\infty \in \mathcal{X}_{\text{GHS}}$ 可行. 
- 且目标函数 $\theta_{\text{GHS}}$ 在 $\mathbf{x}^\infty$ 处是 upper semi-continuous 的, 故 supremum 可达, 即
    $$
    \theta_{\text{GHS}} (\mathbf{x}^\infty) \geq \lim_{\nu \to \infty} \theta_{\text{GHS}} (\mathbf{x}^\nu) = \sup_{\mathbf{x}\in \mathcal{X}_{\text{GHS}}} \theta_{\text{GHS}} (\mathbf{x}).
    $$

### Stationarity and local maximizer


在讨论完 global maximizer 的存在性后, 进一步讨论何时一个 stationary  point 能够成为 local maximizer. 并且需要如下的权衡: 一方面希望其能够尽量简单地求解, 另一方面又希望其能尽量排除掉一些非 local maximizer 的 stationary point. 这里主要考虑如下两种:
- *Pseudo B-stationarity* 
- *Epi-stationarity*

#### Pseudo-Bouligand stationary

回到 GHSOP 问题, 给定一个 feasible point $\mathbf{\bar{x}} \in \mathcal{X}_{\text{GHS}}$, 根据其 Heaviside 对应的激活情况, 可以将下标划分如下:
$$
\mathcal{J}_{i>} (\mathbf{\bar{x}}) := \{j: \phi_{ij}(\mathbf{\bar{x}}) > 0\}, \quad 
\mathcal{J}_{i=} (\mathbf{\bar{x}}) := \{j: \phi_{ij}(\mathbf{\bar{x}}) = 0\}, \\
\mathcal{J}_{i<} (\mathbf{\bar{x}}) := \{j: \phi_{ij}(\mathbf{\bar{x}}) < 0\}, \quad
\mathcal{J}_{i\geq} (\mathbf{\bar{x}}) := \{j: \phi_{ij}(\mathbf{\bar{x}}) \geq 0\}.
$$
由于是 closed Heaviside function, 故事实上只有 $\mathcal{J}_{i\geq} (\mathbf{\bar{x}})$ 的部分是 active 的, 而 $\mathcal{J}_{i<} (\mathbf{\bar{x}})$ 的部分的 indicator 取 $0$. 

若只保留 active 的部分, 则可以将 GHSOP 问题转化为标准的 NLP 问题, 该过程称为 pulled-out, 其形式为:
$$
    \begin{aligned}
      \max_{\mathbf{x}\in P} \quad & \theta(\mathbf{x};\bar{\mathbf{x}})\triangleq c(\mathbf{x})
        +\sum_{j\in\mathcal{J}_{0\geq}(\bar{\mathbf{x}})}\psi_{0j}(\mathbf{x})\\
      \text{s.t.} \quad & \left\{
        \begin{array}{l}
          \text{for all }i=1,\ldots,I\\[4pt]
          \mathbf{A}_{i\cdot}\mathbf{x}
            +\displaystyle\sum_{j\in\mathcal{J}_{i\geq}(\bar{\mathbf{x}})}
            \psi_{ij}(\mathbf{x})\geq\eta_i
        \end{array}\right.\\[2pt]
      \text{and} \quad & \left\{
        \begin{array}{l}
          \text{for all }i=0,1,\ldots,I:\\[4pt]
          \phi_{ij}(\mathbf{x})\geq 0\quad
            \forall\,j\in\mathcal{J}_{i\geq}(\bar{\mathbf{x}})\\[3pt]
          \phi_{ij}(\mathbf{x})\leq 0\quad
            \forall\,j\in\mathcal{J}_{i<}(\bar{\mathbf{x}})
        \end{array}\right.
    \end{aligned}
$$

- 注意, 这里的第二组约束要求, 在 pulled-out 的过程中, 决策变量 $\mathbf{x}$ 不能离开当前 $\bar{\mathbf{x}}$ 的 Heaviside 激活状态, 即 $\mathcal{J}_{i\geq}(\bar{\mathbf{x}})$ 的部分必须保持 $\phi_{ij}(\mathbf{x}) \geq 0$, 而 $\mathcal{J}_{i<}(\bar{\mathbf{x}})$ 的部分必须保持 $\phi_{ij}(\mathbf{x}) \leq 0$.
- 而且事实上这种自指关系在本质上揭示 pull-out 过程是一个 fixed-point 迭代.


***Definition* (Pseudo-Bouligand stationary)**: 称 $\mathbf{\bar{x}} \in \mathcal{X}_{\text{GHS}}$ 是 G-HSCOP 的 pseudo-Bouligand stationary point $\iff$ $\mathbf{\bar{x}}$ 是其 pulled-out NLP 问题的 Bouligand stationary point.

可以证明, 一般而言有如下推导关系:
$$
\text{local min} \stackrel{\text{(1)}}{\implies} \text{epi-stationary} \stackrel{\text{(2)}}{\implies} \text{pseudo-B-stationary}
$$

*Proof Sketch*:
- (1) : 若 $\mathbf{\bar{x}}$ 是 local min, 则 $(f(\mathbf{\bar{x}}), \mathbf{\bar{x}})$ 是 lifted problem 的 local min, 而 B-stationary 是 local min 的必要条件 (因为 B-Stationary 要求不存在一个可行的切向量 $\mathbf{v}$ 使得函数一阶下降, 而 local min 必然满足这个条件), 故 $\mathbf{\bar{x}}$ 是 epi-stationary.
- (2) : 首先有观察, pulled-out 问题的 feasible point 必然是原问题的 feasible point. 此时用反证法, 假设 $\mathbf{\bar{x}}$ 是 epi-stationary, 但不是 pseudo-B-stationary, 则存在一个 pulled-out 问题的可行切向量使得 pull-out 函数值下降. 然而根据构造关系, 这个下降方向必然也是原问题的真实下降方向, 这会破坏 epi-stationary 的定义, 故矛盾.


若额外满足 $\mathcal{Z} := \text{epi}(f) \cap (\mathcal{X} \times \mathbb{R})$ 在 $(f(\mathbf{\bar{x}}), \mathbf{\bar{x}})$ 处是 locally convex-like 的, 则 (1)  反向成立, 即 local min $\iff$ epi-stationary $\implies$ pseudo-B-stationary.

*Proof Sketch*:
由 epi-stationary 表示 tangent cone 没有下降方向, 而 locally convex-like 表示每个附近真实下降点都会产生一个向下的 tangent direction, 故 epi-stationary 没有下降放下 $\iff$ 附近没有真实下降点 $\iff$ local min.




### Simplified HSCOP and Two Techniques of Solving Stationary Point

考虑如下简化问题:
$$
    \begin{aligned}
      \min_{\mathbf{x}\in P} \quad & \Phi(\mathbf{x})\triangleq c(\mathbf{x})
        +\sum_{k=1}^{K}\varphi_k(\mathbf{x})
        \mathbb{1}_{(0,\infty)}\bigl(g_k(\mathbf{x})\bigr)\\[4pt]
      \text{s.t.} \quad & \sum_{\ell=1}^{L}\phi_\ell(\mathbf{x})
        \mathbb{1}_{[0,\infty)}\bigl(h_\ell(\mathbf{x})\bigr)\leq b
    \end{aligned} \qquad  {\small{\text{(S-HSCOP)}}}
$$

- 相比于 G-HSCOP, 其只有一个标量约束, 去掉了 affine 约束. 并且注意这里改为了最小化问题, 因而 indicator function 改为了 open Heaviside function.

同样由于 Heaviside 的存在, 我们不能直接照搬普通的平稳条件, 而是转为研究 pulled-out B-stationary conditions, 作为原问题的 pseudo-B-stationary conditions. 故同样, 假设我们手里已经有一个候选点 $\mathbf{\bar{x}} \in P$, 并且根据其 Heaviside 激活情况, 可以将下标划分, 得到 pulled-out 的问题:
$$
    \begin{aligned}
      \min_{\mathbf{x}} \quad & \Phi(\mathbf{x};\bar{\mathbf{x}})\triangleq c(\mathbf{x})
        +\sum_{k\in\mathcal{K}_{>}(\bar{\mathbf{x}})}\varphi_k(\mathbf{x})\\[2pt]
      \text{s.t.} \quad & \mathbf{x} \in \widehat{\mathcal{S}}_{\mathrm{ps}}(\bar{\mathbf{x}}), \quad
        \sum_{\ell\in\mathcal{L}_{>}(\bar{\mathbf{x}})}\phi_\ell(\mathbf{x})
        \leq b
    \end{aligned} \qquad {\small{\text{(PO} (\mathbf{\bar{x}})\text{)}}}
$$
其中 $\mathcal{K}$ 是与 $\varphi_k$ 和 $g_k$ 相关的 index 集合, $\mathcal{L}$ 是与 $\phi_\ell$ 和 $h_\ell$ 相关的 index 集合. $\widehat{\mathcal{S}}_{\mathrm{ps}}(\bar{\mathbf{x}})$ 是 $\mathbf{\bar{x}}$ 的 Heaviside 激活状态下的 feasible set, 其定义为:
$$
    \widehat{\mathcal{S}}_{\mathrm{ps}}(\bar{\mathbf{x}})
    =\left\{\mathbf{x}\in P:
    \begin{array}{ll}
        g_k(\mathbf{x})\leq 0, & \forall k\in\mathcal{K}_{\leq}(\bar{\mathbf{x}})\\[3pt]
        g_k(\mathbf{x})\geq 0, & \forall k\in\mathcal{K}_{>}(\bar{\mathbf{x}})\\[3pt]
        h_\ell(\mathbf{x})\leq 0, & \forall\ell\in\mathcal{L}_{\leq}(\bar{\mathbf{x}})\\[3pt]
        h_\ell(\mathbf{x})\geq 0, & \forall\ell\in\mathcal{L}_{>}(\bar{\mathbf{x}})
    \end{array}\right\}.
$$

- 回顾, $\mathbf{\bar{x}}$ 是原问题的 pseudo-B-stationary point $\iff$ $\mathbf{\bar{x}}$ 是生成的 pulled-out problem $\text{PO} (\mathbf{\bar{x}})$ 的 B-stationary point. 而这本身是一个自举 /  fixed-point 的定义, 因为我们需要给出一个候选点 $\mathbf{\bar{x}}$ 来生成 pulled-out problem, 而 pulled-out problem 的 B-stationary point 又需要回到 $\mathbf{\bar{x}}$ 来验证. 
- 因此, pull-out problem 虽然定义了 pseudo-B-stationary point, 但无法直接求解. 因此下面给出了具体的求解方法. 

#### Penalized Epigraphical Formulation

上面的 pulled-out problem 需要给定 $\mathbf{\bar{x}}$ 来确定 Heaviside 的激活状态. 而下面的 penalized epigraphical formulation 则不需要预先知道 $\mathbf{\bar{x}}$, 而是把所有可能的分支都考虑进来. 

对于原问题 S-HSCOP, 其含有两类 Heaviside function: $\varphi_k(\mathbf{x}) \mathbb{1}_{(0,\infty)}(g_k(\mathbf{x}))$ 和 $\phi_\ell(\mathbf{x}) \mathbb{1}_{[0,\infty)}(h_\ell(\mathbf{x}))$. 对于两者, 分别对应引入 $t_k$ 和 $s_\ell$ 两组 slackness variable 进行 epi-form lifting, 并且对所有的 functional constraint (即 $\sum_{\ell=1}^{L}\phi_\ell(\mathbf{x}) \mathbb{1}_{[0,\infty)}(h_\ell(\mathbf{x}))\leq b$) 进行罚函数处理, 得到如下的 penalized epigraphical formulation:

假设 $\varphi_k$ 在 $P \cap g_k^{-1}(0)$ 以及 $\phi_\ell$ 在 $P \cap h_\ell^{-1}(0)$ 上非负, 此时有:
$$
\begin{aligned}
\min_{\mathbf{x} \in P,\mathbf{t},\mathbf{s}} \quad & \Phi_\lambda(\mathbf{x},\mathbf{t},\mathbf{s}) := \underbrace{c(\mathbf{x}) + \sum_{k=1}^{K} t_k}_{\small \Phi(\mathbf{x}) \text{ in epi-form}}  + \lambda \max \left( 
    \sum_{\ell=1}^{L} s_\ell - b, 0
\right) \\
\text{s.t.} \quad & \forall k: \min\bigl(\max(\varphi_k(\mathbf{x})-t_k,\,-g_k(\mathbf{x})),\,
          \max(g_k(\mathbf{x}),\,-t_k)\bigr)\leq 0 \\
& \forall \ell: \min\bigl(\max(\phi_\ell(\mathbf{x})-s_\ell,\,-h_\ell(\mathbf{x})),\,
          \max(h_\ell(\mathbf{x}),\,-s_\ell)\bigr)\leq 0
\end{aligned} 
\qquad  {\small{\text{(P-HSCOP)}}}
$$

- 该形式总的而言相当于进行了如下三步处理: (1) 目标中的 Heaviside 通过 $t_k$ lifiting; (2) 约束中的 Heaviside 通过 $s_\ell$ lifting; (3) $\sum_\ell s_\ell \leq b$ 被罚入目标. 

  - 目标中的 Heaviside 提升为 epi-form. 对于其中第 $k$ 项 $\varphi_k(\mathbf{x}) \mathbb{1}_{(0,\infty)}(g_k(\mathbf{x}))$, 其 epi-form 为 $t_k \geq \varphi_k(\mathbf{x}) \mathbb{1}_{(0,\infty)}(g_k(\mathbf{x}))$, 根据 Heaviside 的定义, 其等价于
    $$
    \begin{cases}
    t_k \geq \varphi_k(\mathbf{x}) & \text{if } g_k(\mathbf{x}) > 0 \\
    t_k \geq 0 & \text{if } g_k(\mathbf{x}) \leq 0
    \end{cases}
    $$
    在逻辑上, 每一条 if-then 相当于一个且关系, 而两条 if-else 之间是或关系, 故可以将其转化为如下的逻辑表达式:
    $$
    \bigl(g_k(\mathbf{x}) > 0 \land t_k \geq \varphi_k(\mathbf{x})\bigr) \lor \bigl(g_k(\mathbf{x}) \leq 0 \land t_k \geq 0\bigr)
    $$
    而根据逻辑编码, $\max(a,b) \leq 0 \iff a \leq 0 \land b \leq 0$, $\min(a,b) \leq 0 \iff a \leq 0 \lor b \leq 0$, 故可以将其转化为如下的 min-max 表达式:
    $$
    \min\bigl(\max(\varphi_k(\mathbf{x})-t_k,\,-g_k(\mathbf{x})),\,
          \max(g_k(\mathbf{x}),\,-t_k)\bigr)\leq 0 
    $$
     - 不过严谨地说, 当 $g_k(\mathbf{x}) = 0$ 时, 该表达式可约简为 $t_k \geq \min\{\varphi_k(\mathbf{x}), 0\}$, 故为了保证等价性, 需要假设 $\varphi_k(\mathbf{x}) \geq 0$ 当 $g_k(\mathbf{x}) = 0$ 时成立. 此时才有 $\min\{\varphi_k(\mathbf{x}), 0\} = 0$, 与 Heaviside 的定义一致.

  - 另一方面对于 function constraint $\sum_{\ell=1}^{L} \phi_\ell(\mathbf{x}) \mathbb{1}_{[0,\infty)}(h_\ell(\mathbf{x})) \leq b$, 其每一项等价于
    $$
    \begin{cases}
    \phi_\ell(\mathbf{x}) & \text{if } h_\ell(\mathbf{x}) \geq 0 \\
    0 & \text{if } h_\ell(\mathbf{x}) < 0   
    \end{cases}
    $$
    故用 lifting 的方式, 每项对应一个 slack variable $s_\ell$, 则
    $$
    \begin{cases}
    s_\ell \geq \phi_\ell(\mathbf{x}) & \text{if } h_\ell(\mathbf{x}) \geq 0 \\
    s_\ell \geq 0 & \text{if } h_\ell(\mathbf{x}) < 0
    \end{cases}
    $$
    故 epi-form 为 $s_\ell \geq \phi_\ell(\mathbf{x}) \mathbb{1}_{[0,\infty)}(h_\ell(\mathbf{x}))$, 同理可转化为如下的 min-max 表达式:
    $$
    \min\bigl(\max(\phi_\ell(\mathbf{x})-s_\ell,\,-h_\ell(\mathbf{x})),\,
          \max(h_\ell(\mathbf{x}),\,-s_\ell)\bigr)\leq 0
    $$
    - 类似地, 此时当 $h_\ell(\mathbf{x}) = 0$ 时, 上述表达式为 $s_\ell \geq \min\{\phi_\ell(\mathbf{x}), 0\}$, 此时虽然假设了 $\phi_\ell(\mathbf{x}) \geq 0$, 但只能保证当 $h_\ell(\mathbf{x}) = 0$ 时, $s_\ell \geq 0$, 而不能保证 $s_\ell \geq \phi_\ell(\mathbf{x})$. 其确实没有保证逐点相等. 

  - 最终当用 $s_\ell$ 代替 $\phi_\ell(\mathbf{x}) \mathbb{1}_{[0,\infty)}(h_\ell(\mathbf{x}))$ 后, 故可以用 $\max\{\sum_{\ell=1}^{L} s_\ell - b, 0\}$ 来衡量 S-HSCOP 中约束 $\sum_{\ell=1}^{L} \phi_\ell(\mathbf{x}) \mathbb{1}_{[0,\infty)}(h_\ell(\mathbf{x})) \leq b$ 的违反程度, 并且用 $\lambda \max(\sum_{\ell=1}^{L} s_\ell - b, 0)$ 来进行罚函数处理. 当 $\lambda$ 足够大时, 该罚函数将会强制 $\sum_{\ell=1}^{L} s_\ell \leq b$ 成立, 从而保证原问题的约束被满足.


- 如此构造出的 penalized epigraphical formulation 有如下几个问题需要被回答:
  - 罚问题的 Stationary point 是否能满足 $\sum_{\ell=1}^{L} s_\ell \leq b$? $\lambda$ 需要多大才能保证, 是否是一个可计算的常数? (因为 $\lambda$ 太大可能会导致数值问题, 过小则无法保证约束满足)
  - 罚问题的 B-stationary point 是否能保证原问题的 pseudo-B-stationary point? 


***Theorem* (Exact penalization for finite $\lambda$)**: 假设 $c$ 和 $\varphi_k$ 是 Lipschitz continuous 的 (对应 $\operatorname*{Lip}(c)$ 和 $\operatorname*{Lip}(\varphi_k)$), 假设 $(\mathbf{\bar{x}}, \mathbf{\bar{t}}, \mathbf{\bar{s}})$ 是 P-HSCOP 的 B-stationary point. 若
- 罚参数 $\lambda > \operatorname*{Lip}(\varphi) + \operatorname*{Lip}(c)$, 
- 存在单位向量 $\mathbf{v}$, 使得 $\mathbf{v} \in \mathcal{T} (\mathcal{\hat{S}}_{\mathrm{ps}}(\bar{\mathbf{x}}) ; \mathbf{\bar{x}})$ 且 $\sum_{\ell \in \mathcal{L}_{>} (\bar{\mathbf{x}})} \phi_\ell'(\mathbf{\bar{x}}; \mathbf{v}) \leq -1$, 

则 $\mathbf{\bar{x}}$ 是原 S-HSCOP 问题的 pseudo-B-stationary point. 

说明:
- 其中第二个条件的说明如下
  - 回顾对于 S-HSCOP, 其约束为 $\sum_{\ell=1}^{L} \phi_\ell(\mathbf{x}) \mathbb{1}_{[0,\infty)}(h_\ell(\mathbf{x})) \leq b$. 其中, active 的约束是 $\mathcal{L}_{>} (\bar{\mathbf{x}}) = \{\ell: h_\ell(\mathbf{\bar{x}}) > 0\}$, 因此 pulled-out 的约束为 $\sum_{\ell \in \mathcal{L}_{>} (\bar{\mathbf{x}})} \phi_\ell(\mathbf{x}) \leq b$. 因此其整体含义是, 对于这些 active 的约束, 当约束被违反时, 存在下降方向. 
  - 对于具体下降方向 $\mathbf{v}$, 其要求 $\mathbf{v} \in \mathcal{T} (\mathcal{\hat{S}}_{\mathrm{ps}}(\bar{\mathbf{x}}) ; \mathbf{\bar{x}})$, 其中 $\mathcal{\hat{S}}_{\mathrm{ps}}(\bar{\mathbf{x}})$ 是去掉 functional constraint $\sum_{\ell=1}^{L} \phi_\ell(\mathbf{x}) \mathbb{1}_{[0,\infty)}(h_\ell(\mathbf{x})) \leq b$ 后的其余约束的集合, 要求 $\mathbf{v} \in \mathcal{T} (\mathcal{\hat{S}}_{\mathrm{ps}}(\bar{\mathbf{x}}) ; \mathbf{\bar{x}})$ 意味着从 $\mathbf{v}$ 出发不会破坏其余约束的可行性. 


#### Smoothing

观察上面的 penalized epigraphical formulation, 其目标函数和约束都是 min-max 的形式, 其实某种意义上仍然并没有解决 Heaviside 的不连续性问题 (因为 min-max 事实上仍然可以写成 Heaviside 的形式, 其本质上区别于原问题的 Heaviside 的是 lifted, 但仍然是非连续的). 因此, 还需要进一步的 smoothing 技术, 将 min-max 转化为 smooth 的形式, 以便于使用光滑优化方法求解.

仍然考虑罚函数, 但将 Heaviside 的形式显示写出, 得到 Heavisde 的 penalized epigraphical formulation:
$$
\min_{\mathbf{x} \in P} \left\{
\Psi_{\lambda}(\mathbf{x}) := c(\mathbf{x}) + \sum_{k=1}^{K} \varphi_k(\mathbf{x}) \mathbb{1}_{(0,\infty)}(g_k(\mathbf{x})) + \lambda \max\left( \sum_{\ell=1}^{L} \phi_\ell(\mathbf{x}) \mathbb{1}_{(0,\infty)}(h_\ell(\mathbf{x})) - b, 0 \right)
\right\}
$$
故仍需要对 Heaviside 进行 smoothing. 不过这里的 smoothing 区别于数学的 $C^1$ 或 $C^\infty$ 的光滑. 总体而言有两种 smoothing 技术. (其中 $\delta$ 是 smoothing 的参数, 越小越接近原问题的 Heaviside, 但也越不光滑)
- Truncation based: 构造一个上升函数 $\hat{\theta}(\cdot)$, 然后将其截断到 $[0,1]$ 区间. 对于截断可能产生非光滑拐角. 而又仔细分为 symmetric / asymmetric 两种截断方式等. 而截断仍然是一种 difference of convex (DC) 结构, 仍然可以使用 DC programming 的方法求解. 
    $$
    \theta_{\text{trunc}}(t, \delta) = T_{[0,1]}(\hat{\theta}(t, \delta)) = \min\{\max\{\hat{\theta}(t, \delta), 0\}, 1\}
    $$
- Average based: 通过类似卷积等方式, 通过积分进行平均. 



![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/234_horizontal_concat.png)


综上, 通过上述两种技巧, 可以将原简化问题 S-HSCOP:
$$
\begin{aligned}
\min_{\mathbf{x}\in P} \quad & c(\mathbf{x})
+\sum_{k=1}^{K}
\varphi_k(\mathbf{x})
\mathbf{1}_{(0,\infty)}
\bigl(g_k(\mathbf{x})\bigr)\\
\text{s.t.} \quad & \sum_{\ell=1}^{L}
\phi_\ell(\mathbf{x})
\mathbf{1}_{(0,\infty)}
\bigl(h_\ell(\mathbf{x})\bigr)
\le b.
\end{aligned}
$$

通过精确罚+光滑化, 转化为如下的光滑优化问题:

$$
\begin{aligned}
    \min_{\mathbf{x}\in P} \quad & \widehat{\Phi}_\lambda(\mathbf{x},\delta)\triangleq c(\mathbf{x})
    +\underbrace{\sum_{k=1}^{K}\varphi_k(\mathbf{x})
        \theta_k^\varphi(g_k(\mathbf{x}),\delta)}_{
        \text{approx.\ of Heaviside in objective}}\\
    & +\underset{\text{\scriptsize fixed}}{\lambda}\,
    \overbrace{\max\Bigl(\underbrace{\sum_{\ell=1}^{L}
        \phi_\ell(\mathbf{x})\theta_\ell^\phi(h_\ell(\mathbf{x}),\delta)}_{
        \text{approx.\ of Heaviside in constraint}}-b,\ 0\Bigr)}^{
        \text{constraint penalization}}
\end{aligned}
$$

此时只保留了相对便于求解的简单约束 $\mathbf{x} \in P$. 

说明: 
- 对于每一个给定的具体问题, 其都是一个 DC programming 问题. 对于 DC 问题我们有更为全面的分析工具可以使用, 因此 Heaviside 问题在当前的重点只要关注到转化为 DC 即可. 
- 这里的 $\lambda$ 是固定的, 不会在渐进分析中让 $\lambda \to \infty$, 而是固定为一个足够大的常数, 以保证约束的满足. 
- $\delta$ 是 smoothing 的参数 (当 $\delta \to 0$ 时为原问题的 Heaviside). 因此, 在实践中, 将同样进行 variable smoothing, 已通过递减的 $\delta_1>\delta_2 > \cdots \delta_\nu\downarrow 0$ 来得到一连串的光滑优化问题 $\mathbf{x}^{(\nu)} \in \arg\min_{\mathbf{x}\in P} \widehat{\Phi}_\lambda(\mathbf{x},\delta_\nu)$. 我们关注, 当 $\mathbf{x}^{(\nu)} \to \mathbf{x}^\star$ 时, $\mathbf{x}^\star$ 对于原问题 S-HSCOP 的平稳性等性质. 可以证明, 当如下问题被满足, $\mathbf{x}^\star$ 是原问题的 pseudo-B-stationary point.

  - *(C1) Pointwise sign condition:* 存在邻域 $\mathcal{N}_*$，使得在 $P\cap\mathcal{N}_*$ 上，
    $\{\varphi_k\}_{k\in\mathcal{K}_=(\mathbf{x}^*)}$ 与
    $\{\phi_\ell\}_{\ell\in\mathcal{L}_=(\mathbf{x}^*)}$ 均为非负函数。

  - *(C2) Satisfied by a convex PA function:*
    - 若 $\mathbf{v}\in\mathcal{T}(P;\mathbf{x}^*)$ 且
      $g_k'(\mathbf{x}^*;\mathbf{v})\leq 0$ 对所有
      $k\in\mathcal{K}_=(\mathbf{x}^*)$ 成立，则
      $g_k'(\mathbf{x};\mathbf{v})\leq 0$ 对所有
      $k\in\mathcal{K}_=(\mathbf{x}^*)$ 及
      $\mathbf{x}\in\mathcal{N}_*$ 成立。
    - 若 $\mathbf{v}\in\mathcal{T}(P;\mathbf{x}^*)$ 且
      $h_\ell'(\mathbf{x}^*;\mathbf{v})\leq 0$ 对所有
      $\ell\in\mathcal{L}_=(\mathbf{x}^*)$ 成立，则
      $h_\ell'(\mathbf{x};\mathbf{v})\leq 0$ 对所有
      $\ell\in\mathcal{L}_=(\mathbf{x}^*)$ 及
      $\mathbf{x}\in\mathcal{N}_*$ 成立。

  - *(C3) Functional consistency:* 对所有
    $k\in\mathcal{K}_=(\mathbf{x}^*)$ 和
    $\ell\in\mathcal{L}_=(\mathbf{x}^*)$，有
    $$
    \lim_{\nu\to\infty}
    \theta_k^\varphi\bigl(g_k(\mathbf{x}^\nu),\delta_\nu\bigr)=0,
    \qquad
    \lim_{\nu\to\infty}
    \theta_\ell^\phi\bigl(h_\ell(\mathbf{x}^\nu),\delta_\nu\bigr)=0.
    $$

  - *(C4) Descent for exact penalization:* 对任意满足
    $$
    \mathbf{x}\in P\cap\mathcal{N},
    \qquad
    \sum_{\ell\in\mathcal{L}_>(\mathbf{x})}\phi_\ell(\mathbf{x})>b
    $$
    的 $\mathbf{x}$，存在单位向量
    $\bar{\mathbf{v}}\in\mathcal{T}(\widehat{\mathcal{S}}_{\mathrm{ps}}(\mathbf{x});\mathbf{x})$，使得
    $$
    \sum_{\ell\in\mathcal{L}_>(\mathbf{x})}
    \phi_\ell'(\mathbf{x};\bar{\mathbf{v}})\leq -1.
    $$

  - *(C5) Clarke regularity:* 函数 $c$、
    $\{\varphi_k\}_{k\in\mathcal{K}_>(\mathbf{x}^*)}$ 与
    $\{\phi_\ell\}_{\ell\in\mathcal{L}_>(\mathbf{x}^*)}$ 在
    $\mathbf{x}^*$ 处均为 Clarke regular。对于 B-differentiable 函数 $f$，这表示对任意收敛至
    $\bar{\mathbf{x}}$ 的序列 $\{\mathbf{z}^\nu\}$ 及任意
    $\mathbf{v}\in\mathbb{R}^n$，
    $$
    \limsup_{\nu\to\infty}f'(\mathbf{z}^\nu;\mathbf{v})
    \leq f'(\bar{\mathbf{x}};\mathbf{v}).
    $$ -->


## 5. Affine Heaviside Composite Problems

在当前 section, 将进一步研究一个更具体的问题, 即 affine Heaviside composite problem (A-HSCOP) with mixed signed constant coefficient $\{\psi_{ik}\}$.

$$
\begin{aligned}
\max_{\mathbf{x}\in P} \quad & c(\mathbf{x})
+\sum_{k=1}^{K_0}
\psi_{0k}\,
\mathbf{1}_{[0,\infty)}
\bigl(\phi_{0k}(\mathbf{x})\bigr)\\
\text{s.t.} \quad & \mathbf{A}_{i\cdot}\mathbf{x}
+\sum_{k=1}^{K_i}
\psi_{ik}\,
\mathbf{1}_{[0,\infty)}
\bigl(\phi_{ik}(\mathbf{x})\bigr)
\geq\eta_i,
\qquad i=1,\ldots,I.
\end{aligned}
$$

注意, 这里的 $\psi_{ik}$ 从前面依赖于 $\mathbf{x}$ 的函数, 变为常数, 但其符号不定. 
并且, 强调内层的 $\phi_{ik}(\mathbf{x})$ 仍然可以是非凸, 不可微的. 后面主要考虑 concave / piecewise affine 两种情况. 


本 section  的研究重点将在于研究由于 $\psi_{ik}$ 的符号不定 (负系数) 导致的 semi-continuity 和 closedness 的问题. 

### Local geometric properties of A-HSCOP

首先研究可行集附近的局部几何特征. 

- 若给定 $P$ 是凸集, 内层函数 $\phi_{ik}$ 是 piecewise affine 的, surprisingly, 可行集是 locally star-shaped 的. 
  - Recall: 对任意 $\mathbf{\bar{x}} \in \mathcal{X}_{\text{AHS}}$, 都存在一个邻域 $\mathcal{N}(\mathbf{\bar{x}})$, 使得领域内的任何点 $\mathbf{x} \in \mathcal{N}(\mathbf{\bar{x}}) \cap \mathcal{X}_{\text{AHS}}$, 都存在 scalar $\bar{\tau} > 0$, 使得 $\mathbf{\bar{x}} + \tau (\mathbf{x} - \mathbf{\bar{x}}) \in \mathcal{X}_{\text{AHS}}$ 对所有 $\tau \in [0, \bar{\tau}]$ 成立.
  - 其作用在于, 若附近存在一个更好的可行点 $\mathbf{x}$, 则从 $\mathbf{\bar{x}}$ 出发, 沿着 $\mathbf{x}$ 的方向, 存在着至少一小段可行的下降方向. 

- 若进一步假设 $c$ 是 $B$-differentiable 的, 并且在 $\mathbf{\bar{x}}$ 附近是 locally concave-like: 对所有足够接近 $\mathbf{\bar{x}}$ 的 $\mathbf{x}$, 都有 $c(\mathbf{x}) \leq c(\mathbf{\bar{x}}) + c'(\mathbf{\bar{x}}; \mathbf{x} - \mathbf{\bar{x}})$, 则此时有如下三个命题等价:
    1. $\mathbf{\bar{x}}$ 是 A-HSCOP 的 local max.
    2. $\mathbf{\bar{x}}$ 是 A-HSCOP 的 epi-stationary point.
    3. 对于任意足够接近 $\mathbf{\bar{x}}$ 的 $\mathbf{x} \in \mathcal{X}_{\text{AHS}}$, $\mathbf{\bar{x}}$ 会首先局部最大化 Heaviside 和的部分, 只有在 Heaviside 和的部分相等时, 才会考虑 $c$ 的下降方向. 也即, 对于任意 $\mathbf{x} \in \mathcal{X}_{\text{AHS}}$ 足够接近 $\mathbf{\bar{x}}$, 都有
        $$
        \sum_{j=1}^{J_0}
        \psi_{0j}
        \mathbf 1_{[0,\infty)}
        \bigl(\phi_{0j}(\mathbf{x})\bigr)
        \leq
        \sum_{j=1}^{J_0}
        \psi_{0j}
        \mathbf 1_{[0,\infty)}
        \bigl(\phi_{0j}(\bar{\mathbf{x}})\bigr).
        $$
        并且若 $\sum_{j=1}^{J_0} \psi_{0j} \mathbf 1_{[0,\infty)} (\phi_{0j}(\mathbf{x})) = \sum_{j=1}^{J_0} \psi_{0j} \mathbf 1_{[0,\infty)} (\phi_{0j}(\bar{\mathbf{x}}))$, 则有
        $$
        c(\mathbf{x}) \leq c(\mathbf{\bar{x}}), \qquad c'(\mathbf{\bar{x}}; \mathbf{x} - \mathbf{\bar{x}}) \leq 0.
        $$
       - 直观上, 当 $\mathbf{x}$ 足够接近 $\mathbf{\bar{x}}$ 时, 连续的 $c$ 会变得任意小, 而 Heaviside 的部分则是离散的, 因此在局部, Heaviside 的部分会主导 $c$ 的变化. 

### $\epsilon$-adjusted Heaviside function

对上述 A-HSCOP, 根据 $\psi_{ij}$ 的符号进行修改如下: 
- 若 $\psi_{ij} > 0$, 则 $\psi_{ij} \mathbf{1}_{[0,\infty)}(\phi_{ij}(\mathbf{x}))$ 保留不变.
- 若 $\psi_{ij} < 0$, 则 $\psi_{ij} \mathbf{1}_{[0,\infty)}(\phi_{ij}(\mathbf{x}))$ 改为 $\psi_{ij} \mathbf{1}_{(-\epsilon,\infty)}(\phi_{ij}(\mathbf{x}))$, 其中 $\epsilon > 0$ 是一个小的正数.
  - 从直观上, 由于负系数会破坏 closed Heaviside 的 upper semi-continuity, 理论上将其改为 open Heaviside 即可. 但又为了保证函数在 $0$ 点的取值不变, 故将临界点左移一个微小单位, 并使用开区间, 以同时保证 upper-semi-continuity 和原先函数值的不变. 


因此, 根据恒等式 $\psi_{ij} \equiv \psi_{ij}^+ - \psi_{ij}^-$, 其中 $\psi_{ij}^+ = \max\{\psi_{ij}, 0\}$, $\psi_{ij}^- = \max\{0, -\psi_{ij}\}$, 因此可以通过对其直接进行 $\epsilon$-adjustment, 得到:
$$
\begin{aligned}
\psi_{ij}\equiv\psi_{ij}^+ - \psi_{ij}^- \approx \psi_{ij}^+ \mathbf{1}_{[0,\infty)}(\phi_{ij}(\mathbf{x})) - \psi_{ij}^- \mathbf{1}_{(-\epsilon,\infty)}(\phi_{ij}(\mathbf{x})).
\end{aligned}
$$

故得到如下的 $\epsilon$-adjusted A-HSCOP:
$$
\begin{aligned}
    \max_{\mathbf{x}\in P} \quad & \theta^\varepsilon_{\mathrm{AHS}}\triangleq c(\mathbf{x})
    +\underbrace{\sum_{j=1}^{J_0}\psi_{0j}^+\mathbb{1}_{[0,\infty)}
        \bigl(\phi_{0j}(\mathbf{x})\bigr)
        -\sum_{j=1}^{J_0}\psi_{0j}^-\mathbb{1}_{(-\varepsilon,\infty)}
        \bigl(\phi_{0j}(\mathbf{x})\bigr)}_{\text{upper semicontinuous}}\\[2pt]
    \text{s.t.} \quad & \text{for all }i=1,\ldots,I:\\[2pt]
    & \mathbf{A}_{i\cdot}\mathbf{x}
    +\underbrace{\sum_{j=1}^{J_i}\psi_{ij}^+\mathbb{1}_{[0,\infty)}
        \bigl(\phi_{ij}(\mathbf{x})\bigr)
        -\sum_{j=1}^{J_i}\psi_{ij}^-\mathbb{1}_{(-\varepsilon,\infty)}
        \bigl(\phi_{ij}(\mathbf{x})\bigr)}_{\text{upper semicontinuous}}
    \geq\eta_i,
\end{aligned}
$$

显然, 对于 $\epsilon' > \epsilon > 0$, 有 $\mathcal{X}_{\text{AHS}}^{\epsilon'} \subseteq \mathcal{X}_{\text{AHS}}^{\epsilon} \subseteq \mathcal{X}_{\text{AHS}}$. 而关于最优值和最优值点, 有如下结论.

***Theorem* (Optimality of $\epsilon$-adjusted A-HSCOP)**: 给定 $\mathbf{\bar{x}} \in \mathcal{X}_{\text{AHS}}$, 假设 $\phi_{ij}$ 是连续的. 
- 若 $\mathbf{\bar{x}}$ 是原 A-HSCOP 的 local max, 则存在 $\bar{\epsilon} > 0$, 使得任意 $\epsilon \in (0, \bar{\epsilon})$, $\mathbf{\bar{x}}$ 也是 $\epsilon$-adjusted A-HSCOP $\text{P}_{\text{AHS}}^{\epsilon}$ 的 local max.
- 若 $\mathbf{\bar{x}}$ 是 $\epsilon$-adjusted A-HSCOP $\text{P}_{\text{AHS}}^{\epsilon}$ 的 local max, 且满足 local sign invariance 条件, 则 $\mathbf{\bar{x}}$ 也是原 A-HSCOP 的 local max.
  - local sign invariance: 对于所有 $i,j$ 满足 $\psi_{ij} < 0$, 若 $\phi_{ij}(\mathbf{\bar{x}}) = 0$, 则存在邻域 $\mathcal{N}(\mathbf{\bar{x}})$, 使得 $\phi_{ij}(\mathbf{x}) \geq 0$ 对所有 $\mathbf{x} \in \mathcal{N}(\mathbf{\bar{x}}) \cap P$ 成立 (其确保了在 $\mathbf{1}_{(-\epsilon,\infty)}(\phi_{ij}(\mathbf{x}))$ 和 $\mathbf{1}_{[0,\infty)}(\phi_{ij}(\mathbf{x}))$ 之间的局部一致性).


## 6. IP-Based Solution Algorithms for Scalar Combinations

本 section 将讨论当我们成功地通过 $\epsilon$-opening 的方法对原问题进行转化后, 如何使用 IP 等方法进行具体求解. 这里讨论的问题仍然控制在 constant coefficient Heaviside 问题上, 即 $\psi_{ij}$ 是常数, 但符号不定.

$$
\begin{aligned}
\max_{\mathbf{x}\in P} \quad & c(\mathbf{x})
+\sum_{j=1}^{J_0}
\psi_{0j}
\mathbf{1}_{[0,\infty)}
\bigl(\phi_{0j}(\mathbf{x})\bigr)\\
\text{s.t.} \quad & \mathbf{A}_{i\cdot}\mathbf{x}
+\sum_{j=1}^{J_i}
\psi_{ij}
\mathbf{1}_{[0,\infty)}
\bigl(\phi_{ij}(\mathbf{x})\bigr)
\geq\eta_i,
\quad i=1,\ldots,I.
\end{aligned}
$$

按照问题难度的不同, 可以进一步划分为如下几类具体问题:
- $\psi_{ij} \geq 0$ 且内层函数 $\phi_{ij}$ 是 concave: 最简单, 最终可以整理成 mixed-integer convex programming
- $\psi_{ij} \geq 0$ 且内层函数 $\phi_{ij}$ 是 piecewise affine: 由于 piecewise affine 未必是 concave, 因此需要先用 decomposition / piece selection 等方法进行转化
- $\psi_{ij}$ 符号不定, 且内层函数 $\phi_{ij}$ 是 piecewise affine: 需要 $\epsilon$-opening 以及 piece selection 技术结合
- $\psi_{ij}$ 符号不定且包含 multiplicative HSCOP: 包含不同 Heaviside 的乘积等更复杂情况.

首先在本 section, 若不加说明, 默认 $\phi_{ij}$ 是 bounded 的, 即存在 $M > 0$, 使得 $|\phi_{ij}(\mathbf{x})| \leq M$ 对所有 $(i,j)$ 和 $\mathbf{x} \in P$ 成立. 

### Base case and full IP

首先考虑最简单的 base case, 即 $\psi_{ij} \geq 0$ 且 $\phi_{ij}$ 是 concave 的情况. 对于每个 Heaviside 项, 引入一个 binary variable $z_{ij} \in \{0,1\}$, 使得 $z_{ij} = \mathbf{1}_{[0,\infty)}(\phi_{ij}(\mathbf{x}))$. 则此时通过 big-M 的方式, 可以将 Heaviside 的约束转化为如下的 mixed-integer linear constraints:
$$
\begin{aligned}
\max_{\mathbf{x}\in P, \mathbf{z}} \quad & \theta_{\text{AHS}}^\oplus (\mathbf{x}, \mathbf{z}) := c(\mathbf{x}) + \sum_{j=1}^{J_0} \psi_{0j} z_{0j} \\
\text{s.t.} \quad & \mathbf{A}_{i\cdot} \mathbf{x} + \sum_{j=1}^{J_i} \psi_{ij} z_{ij} \geq \eta_i, \quad i=1,\ldots,I, \\
& z_{ij} \in \{0,1\}, \quad  \phi_{ij}(\mathbf{x}) \geq -M (1 - z_{ij}), \quad \forall i,j.
\end{aligned}
$$

- 该 big-M 表示法要求只 $z_{ij} = 1 \implies \phi_{ij}(\mathbf{x}) \geq 0$. 不过对于当前最大化问题且 $\psi_{ij} \geq 0$ 的情况下是合理的.

- 可以证明, $\mathbf{\bar{x}}$ 是原问题的 local max, 当且仅当 $(\mathbf{\bar{x}}, \mathbf{\bar{z}})$ 是 IP 的 local max, 其中 $\bar{z}_{ij} = \mathbf{1}_{[0,\infty)}(\phi_{ij}(\mathbf{\bar{x}}))$. 

- 事实上, 只要有有效的 big-M, 且 $\psi_{ij} \geq 0$, 原问题与 IP 的等价性就是成立的, 其不依赖于 concavity. 而对于 concave 的要求是处于当前 IP 本身的求解技术希望连续约束是凸的以便于求解. 

### PIP subproblems

上述方法在理论上可行, 但一个实际问题是, 由于需要给每个 Heaviside 项引入一个 binary variable, 这对于 IP solver 而言是计算不可行的. 因此在实践中, 希望通过渐近披露的方式, 先求解只含有少量自由变量的子问题 

给定一个已知可行点 $\mathbf{\bar{x}} \in \mathcal{X}_{\text{AHS}}$ (未必是最优点), 计算每个内层函数 $\phi_{ij}(\mathbf{\bar{x}})$ 的值. 另外引入一个阈值 $\delta > 0$, 将 $\phi_{ij}(\mathbf{\bar{x}})$ 的值分为三类:
- $\phi_{ij}(\mathbf{\bar{x}}) \geq \delta$: 认为明显为 active, 则将其对应的 $z_{ij} = 1$ 固定, 不再作为自由变量. 记为 $\mathcal{J}^\delta_{i, >}(\mathbf{\bar{x}}) \subseteq \{j: \phi_{ij}(\mathbf{\bar{x}}) \geq \delta\}$.
- $\phi_{ij}(\mathbf{\bar{x}}) \leq -\delta$: 认为明显为 inactive, 则将其对应的 $z_{ij} = 0$ 固定, 不再作为自由变量. 记为 $\mathcal{J}^\delta_{i, <}(\mathbf{\bar{x}}) \subseteq \{j: \phi_{ij}(\mathbf{\bar{x}}) \leq -\delta\}$.
- $-\delta < \phi_{ij}(\mathbf{\bar{x}}) < \delta$: 认为不确定, 是 PIP 唯一需要求解的变量集合. 记为 $\mathcal{J}^\delta_{i, 0}(\mathbf{\bar{x}})$ 为前述两项的补集. 

通过这样的划分, 我们可以 focus 到真正不确定的变量范围上, 以大幅缩减计算. 因此最终整理后的子问题如下. 其中 $\rho > 0$ 是一个 proximal 参数, 用于保证 $\mathbf{x}$ 不会偏离 $\mathbf{\bar{x}}$ 太远, 以保证 $\mathcal{J}^\delta_{i, 0}(\mathbf{\bar{x}})$ 的划分仍然是合理的. 

$$
\begin{aligned}
    \max_{\mathbf{x}\in P,\,\mathbf{z}} \quad & \theta^{\oplus;\delta}_{\mathrm{AHS};\rho}
    (\mathbf{x},\mathbf{z};\mathbf{\bar{x}})\triangleq c(\mathbf{x})
    +\sum_{j\in\mathcal{J}^\delta_{0;0}(\mathbf{\bar{x}})}\psi_{0j}z_{0j}
    -\frac{\rho}{2}\|\mathbf{x}-\mathbf{\bar{x}}\|_2^2 \\
    \text{s.t.} \quad & \mathbf{A}_{i\cdot}\mathbf{x}
    +\sum_{j\in\mathcal{J}^\delta_{i;0}(\mathbf{\bar{x}})}\psi_{ij}z_{ij}
    +\sum_{j\in\mathcal{J}^\delta_{i;>}(\mathbf{\bar{x}})}\psi_{ij}
    \geq\eta_i,\quad i=1,\ldots,I, \\
    & \phi_{ij}(\mathbf{x})\geq -M(1-z_{ij}),\quad
    z_{ij}\in\{0,1\},\quad
    j\in\mathcal{J}^\delta_{i;0}(\mathbf{\bar{x}}),\; i=0,\ldots,I, \\
    & \phi_{ij}(\mathbf{x})\geq 0,\quad
    j\in\mathcal{J}^\delta_{i;>}(\mathbf{\bar{x}}),\; i=0,\ldots,I, \\
    & \phi_{ij}(\mathbf{x})\ \text{free},\quad
    j\in\mathcal{J}^\delta_{i;<}(\mathbf{\bar{x}}),\; i=0,\ldots,I.
\end{aligned} \qquad \qquad \bigl(\mathrm{IP}^{\oplus;\delta}_{\mathrm{AHS};\rho}(\mathbf{\bar{x}})\bigr)
$$

说明:
- 在目标函数中, 对于 $\mathcal{J}^\delta_{0;>}(\mathbf{\bar{x}})$ 的部分, 由于其对应的 $z_{0j} = 1$ 已经固定为常数, 故不再出现在目标函数中. 
- 在约束中, 对于不确定项 $\mathcal{J}^\delta_{i;0}(\mathbf{\bar{x}})$, 需要用 big-M 进行约束; 对于 $\mathcal{J}^\delta_{i;>}(\mathbf{\bar{x}})$, 希望通过约束要求新求出的 $\mathbf{x}$ 仍然保持 $\phi_{ij}(\mathbf{x}) \geq 0$, 避免矛盾; 对于 $\mathcal{J}^\delta_{i;<}(\mathbf{\bar{x}})$, 由于本身就是 inactive, 故不再对其进行约束.
- 这是一个 convex-constrained mixed-binary program, 可由 GUROBI 等完整求解.