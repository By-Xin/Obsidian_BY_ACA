# Heaviside Composite Optimization Problems (HSCOPs)

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
\underset{\mathbf{x}\in P}{\operatorname{maximize}}
\quad
\theta_{\text{GHS}} (\mathbf{x})
\triangleq
c(\mathbf{x})
+
\sum_{k=1}^{K_0}
\psi_{0k}(\mathbf{x})
\mathbf{1}_{[0,\infty)}
\bigl(\phi_{0k}(\mathbf{x})\bigr)
\\[1mm]
\text{subject to}
\quad
\mathbf {A}_{i,} \mathbf{x}
+
\sum_{k=1}^{K_i}
\psi_{ik}(\mathbf{x})
\mathbf{1}_{[0,\infty)}
\bigl(\phi_{ik}(\mathbf{x})\bigr)
\geq \eta_i,
\qquad
i=1,\ldots,I.
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
\min_{(t, \mathbf{x}) \in \mathbb{R} \times \mathcal{X}} t, \quad \text{subject to } f(\mathbf{x}) - t \leq 0.
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

## 4. Solution Analysis

### Existence of optimal solution

首先讨论解的存在性问题. 如下例子说明, 若不加任何限制, HSCOP 的最优解可能是不可达的.

***Example* (Non-attainable optimal value)**:

首先考虑如下 HSCOP, 其最优解是不可达的.
$$
    \begin{aligned}
      \min_{x_1,x_2}\quad
      &x_1^2+2(1-x_2)^2+|x_1|_0+\tfrac{1}{2}|x_2|_0\\[2pt]
      \text{subject to}\quad
      &|x_1|_0\geq|x_2|_0\ \text{ and }\ -1\leq x_1,x_2\leq 1.
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
      \operatorname*{\text{maximize}}_{\mathbf{x}\in P}\quad
      &\theta(\mathbf{x};\bar{\mathbf{x}})\triangleq c(\mathbf{x})
        +\sum_{j\in\mathcal{J}_{0\geq}(\bar{\mathbf{x}})}\psi_{0j}(\mathbf{x})\\
      \operatorname*{\text{subject to}}\quad
      &\left\{
        \begin{array}{l}
          \text{for all }i=1,\ldots,I\\[4pt]
          \mathbf{A}_{i\cdot}\mathbf{x}
            +\displaystyle\sum_{j\in\mathcal{J}_{i\geq}(\bar{\mathbf{x}})}
            \psi_{ij}(\mathbf{x})\geq\eta_i
        \end{array}\right.\\[2pt]
      \text{and}\quad
      &\left\{
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


#### Simplified Bouligand 

考虑如下简化问题:
$$
\begin{aligned}
\min_{\mathbf{x}\in P}\quad \Phi(\mathbf{x}) := c(\mathbf{x}) + \sum_{k=1}^K \varphi_k(\mathbf{x}) \mathbb{1}_{[0,\infty)}(g_k(\mathbf{x})), \\
\text{subject to}\quad \sum_{\ell=1}^{L} \varphi_\ell(\mathbf{x}) \mathbb{1}_{[0,\infty)}(h_\ell(\mathbf{x})) \leq b, 
\end{aligned}
$$

- 相比于 G-HSCOP, 其只有一个标量约束, 去掉了 affine 约束. 并且注意这里改为了最小化问题. 