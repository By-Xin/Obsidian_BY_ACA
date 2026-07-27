# 6. The VC-Dimension

>- Book Reference: Understanding Machine Learning: From Theory to Algorithms, Shai Shalev-Shwartz and Shai Ben-David.

回顾上一个 section 的 decomposition, 对于 ERM 产生的 $h_S \in \arg\min_{h \in \mathcal{H}} L_S(h)$, 其 true risk 可以被分解为 approximation error 和 estimation error:
$$
L_{\mathcal{D}}(h_S) = \underbrace{L_{\mathcal{D}}(h^*)}_{\text{approximation error}} + \underbrace{(L_{\mathcal{D}}(h_S) - L_{\mathcal{D}}(h^*))}_{\text{estimation error}}
$$
其中
- Approximation error: 反映了 domain knowledge, 是在应用层面上根据不同的专家知识来选择的, 其在理论层面无法进行任何提升.
- Estimation error: 反映的是泛化能力, 是理论层面关注的核心. 在 agnostic PAC learning 中, 回顾其定义:
    $$
    \exists m_{\mathcal{H}}(\epsilon, \delta) \text{ s.t. } \forall \mathcal{D}, \forall m \geq m_{\mathcal{H}}(\epsilon, \delta): \mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(h_S) - L_{\mathcal{D}}(h^*) \leq \epsilon] \geq 1 - \delta
    $$
    其本身也是在考虑 estimation error 的上界.

之前的推导确定了在有限 hypothesis space $\mathcal{H}$ 下是 agnostic PAC learnable 的, 其 sample complexity 为
$$
\mathcal{O}\left(\frac{\log(|\mathcal{H}|/\delta)}{\epsilon^2}\right)
$$
而反过来若 $\mathcal{H}$ 是全体的 hypothesis space, 则由  No-Free-Lunch theorem 可知其是不可学习的. 而 VC-dimension 的提出就是为了在 $\mathcal{H}$ 是无限 hypothesis space 的情况下对其进行学习能力的刻画.

## 6.1 Infinite-Size Classes Can Be Learnable

首先给出一个无限可学习的 hypothesis space 的例子, 其为 threshold functions:
$$
\mathcal{H} = \{h_a: a \in \mathbb{R}\}, \quad h_a(x) = \mathbf{1}[x < a]
$$
其中 $a \in \mathbb{R}$ 是待学习的参数, 其 hypothesis space 是无限的. 

下说明其是 realizable PAC learnable 的.

***Lemma 6.1***. 上述的 threshold functions 的 hypothesis space $\mathcal{H}$ 是 PAC learnable 的, 其 ERM 能够在 $m_{\mathcal{H}}(\epsilon, \delta) = \lceil \frac{\log(2/\delta)}{\epsilon} \rceil$ 个样本下有:
$$
\mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(h_S) \leq \epsilon] \geq 1 - \delta
$$

*Proof*. 

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/lemma_6_1_threshold_proof_layout.png)

- 考虑 realizable 的情况. 假设存在 $a^\star$ 使得 $h^\star = h_{a^\star}$ 是最优的 hypothesis, 即 $L_{\mathcal{D}}(h^\star) = 0$.
- 记 $\mathcal{X}$ 的边缘分布为 $\mathcal{D}_\mathcal{X}$, 故可以确定两个区间端点 $-\infty \leq a_0 < a^\star < a_1 \leq \infty$ 使得 $\mathcal{X}$ 在 $(a_0, a^\star)$ 和 $(a^\star, a_1)$ 上的概率质量均不超过 $\epsilon$:
    $$
    \mathcal{D}_\mathcal{X}((a_0, a^\star)) \leq \epsilon, \quad \mathcal{D}_\mathcal{X}((a^\star, a_1)) \leq \epsilon
    $$
- 给定样本集 $S = \{(x_1, y_1), \ldots, (x_m, y_m)\}$. 取所有 $y_i = 1$ 的样本中最大的 $x_i$ 记为 $b_0$, 取所有 $y_i = 0$ 的样本中最小的 $x_i$ 记为 $b_1$: 
    $$
    b_0 = \max\{x_i: (x_i, y_i) \in S, y_i = 1\}, \quad b_1 = \min\{x_i: (x_i, y_i) \in S, y_i = 0\}
    $$
     则 ERM algorithm 选择的 $h_S$ (由于当前 case 是 realizable 的, 故必须是正确分类的) 对应的 threshold $b_S$ 定介于二者之间: $b_0 \leq b_S \leq b_1$.

- 而可以看到, 只要 $b_0 \geq a_0$ 且 $b_1 \leq a_1$, 则一定能够保证 $L_{\mathcal{D}}(h_S) \leq \epsilon$.  故根据集合的关系, 有:
    $$
    \begin{aligned}
        \mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(h_S) > \epsilon] &\leq \mathbb{P}_{S \sim \mathcal{D}^m}[b_0 < a_0 \lor b_1 > a_1] \\
        &\leq \mathbb{P}_{S \sim \mathcal{D}^m}[b_0 < a_0] + \mathbb{P}_{S \sim \mathcal{D}^m}[b_1 > a_1]
    \end{aligned}
    $$
  - 上述充分条件关系是因为: 只要 $[b_0, b_1] \subseteq [a_0, a_1]$, 则 $b_S$ 就一定落在 $[a_0, a_1]$ 中, 即使落在 $a^\star$ 的一侧, 其错误率也不会超过 $[a_0, a^\star]$ 或 $[a^\star, a_1]$ 的概率质量, 即 $\epsilon$.

- 最终, 注意到事件 $\{b_0 < a_0\}$ 等价于没有任何样本落入 $(a_0, a^\star)$, $\{b_1 > a_1\}$ 等价于没有任何样本落入 $(a^\star, a_1)$ (由 realizability: 若有正例落入 $(a_0, a^\star)$, 则 $b_0$ 至少会推至该点, 故 $b_0 \geq a_0$; 负例方向对称). 因此:
    $$
    \begin{aligned}
        \mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(h_S) > \epsilon]
        &\leq \mathbb{P}[b_0 < a_0] + \mathbb{P}[b_1 > a_1] \\
        &= \mathbb{P}[\forall i,\ x_i \notin (a_0,a^\star)] + \mathbb{P}[\forall i,\ x_i \notin (a^\star,a_1)] \\
        &\leq (1-\epsilon)^m + (1-\epsilon)^m \\
        &\leq 2e^{-\epsilon m}\\
        &\leq \delta \qquad ( \text{when } m \geq \log(2/\delta)/\epsilon)
    \end{aligned}
    $$

$\square$

## 6.2 The VC-Dimension

本节正式定义 VC-dimension. 

首先回顾在有限维情况下 NFL 的证明过程, 其可以看成是如下的博弈步骤:
1. Learner: 选择并公开 hypothesis space $\mathcal{H}$, 算法 $\mathcal{A}$, 以及 sample size $m$.
2. Adversary: 根据 Learner 的算法, 选择一个 distribution $\mathcal{D}$, 并从中采样 $S \sim \mathcal{D}^m$.
3. Learner: 根据采样的样本集 $S$, 选择一个 hypothesis $h_S = \mathcal{A}(S) \in \mathcal{H}$, 并计算其 true risk $L_{\mathcal{D}}(h_S)$.

而在证明时, 参照的逻辑为:
- 首先挑选有限 feature space $\mathcal{C} \subseteq \mathcal{X}$, 其大小为 $2m$, 其中训练集 $|S| = m$.
- 考虑 $\mathcal{C}$ 上的均匀分布, 且给每个分布分配一个真实标注函数 $(x, f(x))$.
- Adversary 可以从 $\mathcal{C} \mapsto \{0,1\}$ 的全部 $2^\|\mathcal{C}|$ 个标注函数中选择一个