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

***Definition* (Restriction of a Hypothesis Class)**. 考虑从 $\mathcal{X}$ 到 $\{0,1\}$ 的 hypothesis space $\mathcal{H}$. 从 feature space $\mathcal{X}$ 中取出一个 finite set $C = \{c_1, \ldots, c_m\} \subseteq \mathcal{X}$. 对于每一个 $h \in \mathcal{H}$, 其在 $C$ 上的 restriction 可以被表示为一个 binary vector:
$$
h(C) = (h(c_1), \ldots, h(c_m)) \in \{0,1\}^m
$$
遍历 $\mathcal{H}$ 中所有的 hypothesis, 可以得到一个 binary vector 的集合, 称之为 restriction of $\mathcal{H}$ to $C$:
$$
\mathcal{H}_C = \{h(C): h \in \mathcal{H}\} \subseteq \{0,1\}^m
$$
表示将 $\mathcal{H}$ 中所有的 hypothesis 限制在一个有限的集合 $C$ 上的所有可能的 labelings. 

$\diamond$

一个重要的结论是: 即使 $\mathcal{H}$ 是无限的, 其 restriction $\mathcal{H}_C$ 仍然是有限的. 这是因为:

- $\mathcal{H}_C \subseteq \{0,1\}^{|C|}$. 后者是所有从 $C$ 到 $\{0,1\}$ 的映射的全集, 而 $\mathcal{H}_C$ 是真正能够通过 $\mathcal{H}$ 中的 hypothesis 实现的 labelings 的子集. 因此, 有 $|\mathcal{H}_C| \leq 2^{|C|}$.
- 此外, $\mathcal{H}$ 中可能存在两个或多个不同的 hypothesis 在 $C$ 上的 restrictions 是相同的 (其不同的 labelings 可能在 $C$ 之外的其他点上才体现出来). 故由于重复点的存在, 也可能进一步减少 $|\mathcal{H}_C|$ 的大小. 

***Definition* (Shattering)**. 给定 $C = \{c_1, \ldots, c_m\} \subseteq \mathcal{X}$, 以及 $\mathcal{H}_C = \{h(C): h \in \mathcal{H}\}$. 如果 $\mathcal{H}_C$ 能够实现 $C$ 上的所有可能的 labelings, 即 $\mathcal{H}_C = \{0,1\}^{|C|}$, 则称 $\mathcal{H}$ shatters $C$:
$$
\forall (y_1, \ldots, y_m) \in \{0,1\}^{|C|}, \exists h \in \mathcal{H} \text{ s.t. } h(c_i) = y_i, \forall i = 1, \ldots, m
$$

对 shattering 的理解如下:
- 一个 $\mathcal{H}$ 能够 shatter 一个 finite set $C$, 意味着 $\mathcal{H}$ 在 $C$ 上并不会排除 label 的组合可能; 反过来, 若不能够 shatter, 则说明至少有一种 labelling 的组合的结构被排除了.  
  - 例如, 若 $\mathcal{H} = \{0, 1\}^|C|$, 则 $\mathcal{H}$ 可以实现 $C$ 上的所有 labelings, 此时的 assignment 是没有限制的, 每个 input 的 label 可以被任意, 独立地指定.
  - 而若比如 threshold functions $h_a(x) = \mathbf{1}\{x \leq a\}$, 则其在 $(c_1, c_2)$ 其中 $c_1 < c_2$ 上, 只能实现三种 restrictions: $\mathcal{H}_C = \{(0,0), (1,0), (1,1)\}$, 而不能实现 $(0,1)$ 的 restriction. 

- 若 $\mathcal{H}$ 能够 shatter $C$, 说明 $\mathcal{H}$ 在 $C$ 上是如此复杂, 以至于其仅凭部分的数据的 label, 无法推断剩余的数据的 label. 对应到具体学习问题中, 由于训练集的有限性, shattering 的存在意味着 $\mathcal{H}$ 在 $C$ 上的泛化能力是无法保证的, $\mathcal{H}$ 中存在着两个 hypothesis, 在已有的训练集上的预测完全相同, 而在训练集之外的预测却完全不同. 


***Corollary* (Shattering and NFL)**. 若 $\mathcal{H}$ 能够 shatter $C$, 且 $|C| = 2 |S| = 2m$, 则无论如何选择什么 learning algorithm $A$, 都存在一个完全无噪声, 且 realizable 的分布 $\mathcal{D}$, 使得 $A$ 在 $S \sim \mathcal{D}^m$ 上训练得到的 hypothesis $h_S$ 的泛化误差至少为 $1/8$, w.p. $\geq 1/7$.

> *If someone can explain every phenomenon, his explanations are worthless*

***Definition* (VC-Dimension)**. 对于 hypothesis space $\mathcal{H}$, 其 VC-dimension 定义为 $\mathcal{H}$ 能够 shatter 的最大 finite set $C$ 的大小:
$$
\operatorname{VCdim}(\mathcal{H}) = \max\{|C|: C \subseteq \mathcal{X}, \mathcal{H} \text{ shatters }C\}
$$

- VC-dim 刻画了 $\mathcal{H}$ 的复杂度, 最多在多少个点上能够实现全部的 labelings 组合. 
- $\operatorname{VCdim}(\mathcal{H}) = d$ 意味着: (1) 至少存在一个 finite set $C$ 的大小为 $d$, 使得 $\mathcal{H}$ 能够 shatter $C$; (2) 不存在任何大小为 $d+1$ 的 finite set $C'$ 能够被 $\mathcal{H}$ shatter.

- 若对于任意大的 finite set $C$, $\mathcal{H}$ 都能够 shatter, 则称 $\operatorname{VCdim}(\mathcal{H}) = \infty$. 显然, 无穷 VC-dim 的 hypothesis space 是 PAC 不可学习的. 