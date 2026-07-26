# 5. The Bias-Complexity Tradeoff

>- Book Reference: Understanding Machine Learning: From Theory to Algorithms, Shai Shalev-Shwartz and Shai Ben-David.

在之前的章节中表明, 在机器学习任务中, 为了避免过拟合, 我们需要预先指定一个 hypothesis class $\mathcal{H}$, 而后在其中进行学习. 而 predefine 的 $\mathcal{H}$ 就表达了 learner 的先验知识. 因此本章的核心就是关于 $\mathcal{H}$ 的选择. 总体思路如下:
- **No-Free-Lunch Theorem**: 不存在一个 universal 的 learning algorithm 在所有任务上表现好. 因此我们必须要有偏好, 必须要对 $\mathcal{H}$ 进行选择.
- **Error Decomposition**: 既然选择了 $\mathcal{H}$, 就一定会存在 error, 而该 error 可以被分解为 approximation 和 estimation error.
- **Bias-Complexity Tradeoff**: 既然误差可以被分解为 approximation 和 estimation error, 那么我们就需要在两者之间进行权衡. 也就是说, 我们需要选择一个合适的 $\mathcal{H}$, 使得 approximation error 和 estimation error 的和最小.

## 5.1 No-Free-Lunch Theorem

***Theorem 5.1 (No-Free-Lunch Theorem)***: 给定定义域 $\mathcal{X}$ 和标签集 $\mathcal{Y}$ (这里暂时 specifiy 为二分类问题 $\mathcal{Y} = \{0, 1\}$), 对于其上的任意学习算法 $\mathcal{A}$, 以及任意训练集大小 $m < |\mathcal{X}|/2$, 存在一个定义在 $\mathcal{X} \times \mathcal{Y}$ 上的分布 $\mathcal{D}$, 使得:
- *(该学习任务是 realizable 的, 即存在完美的 predictor)* 存在函数 $f: \mathcal{X} \to \mathcal{Y}$, 满足 $L_{\mathcal{D}}(f) = 0$.
- *(即使存在完美的 predictor, 任意预先固定的算法 $\mathcal{A}$ 也有概率在某个任务上失败)* 当训练集 $S \sim \mathcal{D}^m$ 时, 对于 $\mathcal{A}$ 输出的 predictor $h = \mathcal{A}(S)$ 满足 
    $$
    \mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(h) \geq 1/8] \geq 1/7
    $$

> [!Note]
>
> - 在当前语境中, 机器学习 '任务' 指的是一个在给定输入输出空间 $\mathcal{X} \times \mathcal{Y}$ 上的分布 $\mathcal{D}$. 不同的分布便代表了不同的任务.
>
> - 注意这里的量词顺序: 对于每一个算法 $\mathcal{A}$, 都能构造一个反例分布 $\mathcal{D}$, 使得该算法在该分布下可能失败. 因此 NFL 的结论是: *不存在一个算法能在任意任务上成功*, 而不是 *存在无法解决的任务*.
>
> - 该定理的总体直觉为: 对于监督学习, $S$ 是唯一信息的来源. 若不对 $\mathcal{H}$ 进行限制, 那么在未见过的样本点上, 算法只能随机猜测 (猜中概率 50-50). 因此若在一个 $2m$ 大小的总体内抽 $m$ 个样本, 则至少有一半的信息没有见过, 在没见过的信息上有一半的几率猜错, 因此总的误差即为 $1/4$.

*Proof*. 

首先构造一个反例任务分布 $\mathcal{D}$, 并说明其满足 NFL 定理的 realizable 条件:

- 取输入空间的一个有限子集 $\mathcal{C} \subseteq \mathcal{X}$, 使得 $|\mathcal{C}| = 2 |S| := 2m$. 
  - 由于样本集 $S$ 的大小为 $m$, 因此 $\mathcal{C}$ 中至少有一半的样本点算法 $\mathcal{A}$ 没有见过.
  - 由于 $\mathcal{X}$ 可能是无限的, 故引入一个有限子集 $\mathcal{C}$ 来首先简化问题, 然后再将结论推广到 $\mathcal{X}$ 上.

- 考虑所有从 $\mathcal{C}$ 到 $\{0, 1\}$ 的映射之可能 (共有 $T = 2^{2m}$ 种), 记为 $f_1, f_2, \ldots, f_T$, 其中每个 $f_i: \mathcal{C} \to \{0, 1\}$ 给出了 $\mathcal{C}$ 中每个点的标签. 故对于其中第 $i$ 个映射 (任务), 考虑构造如下对应的分布 $\mathcal{D}_i$ (作为证明的反例): 从 $\mathcal{C}$ 中均匀采样一个输入点 $X\sim \text{Unif}(\mathcal{C})$, 然后将其标签 $Y$ 确定性地定为 $f_i(X)$. 故其概率分布为:
    $$
    \mathcal{D}_i\{(x, y)\} = \begin{cases}
        1/|\mathcal{C}| & \text{if } y = f_i(x) \\
        0 & \text{otherwise}
    \end{cases}
    $$
     显然, $f_i$ 是 $\mathcal{D}_i$ 下的完美 predictor, 即 $L_{\mathcal{D}_i}(f_i) = 0$. 这对应了 NFL 定理中 '该学习任务是 realizable 的' 的要求.
  - 其穷尽了当前输入空间内的全部输入到全部 0-1 标签的映射可能. 每个 $f_i$ 就可以理解为一个具体的任务, 任务之间彼此是平行的.
  - 另外, 由于 $\mathcal{D}_i$ 是 $C$ 上的均匀分布且无噪声, 因此对于任意 hypothesis $h$, 有:
    $$
    L_{\mathcal{D}_i}(h) = \frac{1}{|\mathcal{C}|} \sum_{x \in \mathcal{C}} \mathbb{1}[h(x) \neq f_i(x)]
    $$

接着, 证明 NFL 定理中 '任意预先固定的算法 $\mathcal{A}$ 都存在一个任务分布 $\mathcal{D}$ 使得其在该分布下可能失败' 的要求:

- 在每个任务分布 $\mathcal{D}_i$ 下, 独立同分布地抽取 $m$ 个样本点, 得到训练集 $S = \{(X_l, Y_l)\}_{l=1}^m \sim \mathcal{D}_i^m$. 考虑定义在 $\mathcal{C}$ 上的算法: $\mathcal{A}_{\mathcal{C}}(S): \mathcal{C} \to \{0, 1\}$, 计算其 true risk 关于样本的期望: $\mathbb{E}_{S \sim \mathcal{D}_i^m}[L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S))]$. 取上述期望关于全部任务的最大值, 可以证明, 有:
    $$
    \max_{i \in [T]} \mathbb{E}_{S \sim \mathcal{D}_i^m}[L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S))] \geq \frac{1}{4} \qquad \text{(1)}
    $$
    即在 $T$ 个可能的任务中, 至少有一个任务的期望真实风险不小于 $1/4$, 不妨记之为任务 $i^*$, 对应分布 $\mathcal{D}_{i^*}$.

- 又由于 $\mathcal{C}$ 是 $\mathcal{X}$ 的子集, 因此可以推广到全输入空间中, 考虑定义在 $\mathcal{X}$ 上的算法 $\mathcal{A}_{\mathcal{X}}(S): \mathcal{X} \to \{0, 1\}$, 其在 $\mathcal{C}$ 上的行为与 $\mathcal{A}_{\mathcal{C}}(S)$ 逐点相同, 并且分布 $\mathcal{D}_{i^*}$ 同样可以进行扩展, 其在 $\mathcal{X} \setminus \mathcal{C}$ 上的概率为 0. 总而言之, 由 (1) 可得, 在全空间 $\mathcal{X}$ 上, 仍然有:
    $$
    \mathbb{E}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(\mathcal{A}_{\mathcal{X}}(S))] = \mathbb{E}_{S \sim \mathcal{D}_{i^*}^m}[L_{\mathcal{D}_{i^*}}(\mathcal{A}_{\mathcal{C}}(S))] \geq 1/4 \qquad \text{(2)}
    $$
    结合 (2) 与 $L_{\mathcal{D}}(\mathcal{A}_{\mathcal{X}}(S)) \in [0, 1]$ 的有界性, 可以通过 Markov 不等式立即得到, 对于任意尾部阈值 $a \in [0, 1]$, 有:
    $$
    \mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(\mathcal{A}_{\mathcal{X}}(S)) \geq a] \geq  \frac{1/4 - a}{1 - a}
    $$
    不妨定性地取 $a = 1/8$, 则有:
    $$
    \mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(\mathcal{A}_{\mathcal{X}}(S)) \geq 1/8] \geq 1/7
    $$
    这就完成了 NFL 定理的证明. 不过我们前面直接承认了 (1) 的结论, 下面给出其证明.

下重新完整 claim 一下 (1) 的命题并给出证明. 已知 $\mathcal{C} \subseteq \mathcal{X}$, $|\mathcal{C}| = 2m$, $f_1, \ldots, f_T$ 枚举 $\mathcal{C} \to \{0, 1\}$ 所有的映射, 其中 $T = 2^{2m}$. 对于每个 $f_i$, 定义对应的分布 $\mathcal{D}_i$ 为: $X \sim \text{Unif}(\mathcal{C})$, $Y = f_i(X)$. 

欲证: 对于任意学习算法 $\mathcal{A}_{\mathcal{C}}$, 有:
$$
\max_{i \in [T]} \mathbb{E}_{S \sim \mathcal{D}_i^m}[L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S))] \geq 1/4
$$
- 首先理解一下证明对象. 
  - 首先固定一个 arbitrarily 定义在 $\mathcal{C}$ 上的学习算法 $\mathcal{A}_{\mathcal{C}}$, 其输入是训练集 $S = \{(X_l, Y_l)\}_{l=1}^m$, 输出是一个 hypothesis $h: \mathcal{C} \to \{0, 1\}$. 
  - 在给定一个具体的任务分布 $\mathcal{D}_i$ 下, 抽取 $m$ 个样本点, 得到训练集 $S \sim \mathcal{D}_i^m$, 并计算其输出 hypothesis 的 true risk: $L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S))$. 由于训练集是随机的, 因此我们考虑其期望 $\mathbb{E}_{S \sim \mathcal{D}_i^m}[L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S))]$. 
  - 接着在我们构造出的 $T$ 个任务分布上, 取上述期望的最大值 (即 worst-case 的任务), 得到 $\max_{i \in [T]} \mathbb{E}_{S \sim \mathcal{D}_i^m}[L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S))]$. 该量即为我们欲证的对象.
- 考虑样本集 $S = \{(X_l, Y_l)\}_{l=1}^m \sim \mathcal{D}_i^m$, 由于其相当于从 $\mathcal{C}$ 中有放回等权重地抽取 $m$ 个样本点, 因此一共有 $|\mathcal{C}|^m = (2m)^m$ 种可能的样本集. 记这些样本集为 $S_1, S_2, \ldots, S_{k}$, 其中 $k = (2m)^m$. 若强调对应任务分布 $\mathcal{D}_i$, 则记为 $S_j^i$. 因此期望 $\mathbb{E}_{S \sim \mathcal{D}_i^m}[L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S))]$ 可以写为:
    $$
    \mathbb{E}_{S \sim \mathcal{D}_i^m}[L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S))] = \frac{1}{k} \sum_{j=1}^{k} L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S_j^i))
    $$

- 对上式取关于 $i \in [T]$ 的最大值, 定大于其关于 $i$ 的平均值, 即:
    $$
    \begin{aligned}
    \max_{i \in [T]} \mathbb{E}_{S \sim \mathcal{D}_i^m}[L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S))] &\geq \frac{1}{T} \sum_{i=1}^{T} \mathbb{E}_{S \sim \mathcal{D}_i^m}[L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S))] \\
    & = \frac{1}{T} \sum_{i=1}^{T} \frac{1}{k} \sum_{j=1}^{k} L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S_j^i)) \\
    & = \frac{1}{k} \sum_{j=1}^{k} \frac{1}{T} \sum_{i=1}^{T} L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S_j^i)) \\
    & \geq \min_{j \in [k]} \frac{1}{T} \sum_{i=1}^{T} L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S_j^i)) .
    \end{aligned}
    $$
    故命题等价于证明, 对于任意样本集 $S_j$, 有:
    $$
    \frac{1}{T} \sum_{i=1}^{T} L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S_j^i)) \geq 1/4
    $$

- 固定 $j$, 由于 $S_j = (x_1, \ldots, x_m)$ 只有 $m$ 笔信息, 故在 $2m$ 大小的输入空间 $\mathcal{C}$ 中, 至少还有 $m$ 笔信息没有被学习, 记为 $\mathcal{V} := \mathcal{C} \setminus \{x_1, \ldots, x_m\} := \{v_1, \ldots, v_p\}$, 其中 $p \geq m$.  故对于任意 $h = \mathcal{A}_{\mathcal{C}}(S_j^i)$, 其 true risk 为:
    $$
    \begin{aligned}
    L_{\mathcal{D}_i}(h) &= \frac{1}{|\mathcal{C}|} \sum_{x \in \mathcal{C}} \mathbb{1}[h(x) \neq f_i(x)] \\
    &\geq \frac{1}{|\mathcal{C}|} \sum_{x \in \mathcal{V}} \mathbb{1}[h(x) \neq f_i(x)] \\
    & \geq \frac{1}{2p} \sum_{r=1}^{p} \mathbb{1}[h(v_r) \neq f_i(v_r)] .
    \end{aligned}
    $$
    故代入上式, 有:
    $$
    \begin{aligned}
    \frac{1}{T} \sum_{i=1}^{T} L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S_j^i)) &\geq \frac{1}{T} \sum_{i=1}^{T} \frac{1}{2p} \sum_{r=1}^{p} \mathbb{1}[\mathcal{A}_{\mathcal{C}}(S_j^i)(v_r) \neq f_i(v_r)] \\
    &= \frac{1}{2p} \sum_{r=1}^{p} \frac{1}{T} \sum_{i=1}^{T} \mathbb{1}[\mathcal{A}_{\mathcal{C}}(S_j^i)(v_r) \neq f_i(v_r)] \\
    &\geq \frac{1}{2} \min_{r \in [p]} \frac{1}{T} \sum_{i=1}^{T} \mathbb{1}[\mathcal{A}_{\mathcal{C}}(S_j^i)(v_r) \neq f_i(v_r)] .
    \end{aligned}
    $$
    下证明最后面的 $\frac{1}{T} \sum_{i=1}^{T} \mathbb{1}[\mathcal{A}_{\mathcal{C}}(S_j^i)(v_r) \neq f_i(v_r)] = 1/2$, 即对于任意一个未见过的样本点 $v_r$, 其在所有可能的任务分布下, 有一半的概率被算法 $\mathcal{A}_{\mathcal{C}}$ 错误预测. 
    - 首先将 $f_1, \ldots, f_T$ 按照 $v_r$ 的标签两两配对: 如果两个映射 $f_i, f_i'$ 在且仅在 $v_r$ 上的标签不同, 则将其配对 (对于任意 $c \in \mathcal{C}, f_i(c) \neq f_i'(c)$ iif $c = v_r$). 
    - 对于这样的两组任务, 其训练集 $S_j^i$ 和 $S_j^{i'}$ 是完全相同的, 因为训练集不包含 $v_r$. 因此算法 $\mathcal{A}_{\mathcal{C}}$ 对于这两个任务的输出 hypothesis 是相同的, 即 $h(v_r) = \mathcal{A}_{\mathcal{C}}(S_j^i) = \mathcal{A}_{\mathcal{C}}(S_j^{i'})$. 然而, 对于预测 $h(v_r)$, 其只能与 $f_i(v_r)$ 或 $f_i'(v_r)$ 中的一个相同, 另一个不同, 即:
        $$
        \mathbb{1}[\mathcal{A}_{\mathcal{C}}(S_j^i)(v_r) \neq f_i(v_r)] + \mathbb{1}[\mathcal{A}_{\mathcal{C}}(S_j^{i'})(v_r) \neq f_i'(v_r)] = 1
        $$
    - 由于所有的 $T$ 个任务分布都可以两两配对, 因此对于任意一个未见过的样本点 $v_r$, 有
        $$
        \frac{1}{T} \sum_{i=1}^{T} \mathbb{1}[\mathcal{A}_{\mathcal{C}}(S_j^i)(v_r) \neq f_i(v_r)] = 1/2
        $$
    综上, 将 $frac{1}{T} \sum_{i=1}^{T} \mathbb{1}[\mathcal{A}_{\mathcal{C}}(S_j^i)(v_r) \neq f_i(v_r)] = 1/2$ 代入上式, 得到:
    $$
    \frac{1}{T} \sum_{i=1}^{T} L_{\mathcal{D}_i}(\mathcal{A}_{\mathcal{C}}(S_j^i)) \geq \frac{1}{2} \min_{r \in [p]} \frac{1}{T} \sum_{i=1}^{T} \mathbb{1}[\mathcal{A}_{\mathcal{C}}(S_j^i)(v_r) \neq f_i(v_r)] = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}
    $$

$\square$


### 5.1.1 No-Free-Lunch and Prior Knowledge

$\mathcal{H}$ 的选择体现了我们对任务的先验知识. PAC Learning 的理论框架中需要预先指定 $\mathcal{H}$. 反过来, 若不加任何先验, 则该问题是不可学习的. 

***Corollary* (No-Free-Lunch Theorem for Realizable PAC Learning)**: 对于无限的输入空间 $\mathcal{X}$, 若不对 $\mathcal{H}$ 进行限制 (允许其为 $\mathcal{H} = \{h: \mathcal{X} \to \mathcal{Y} = \{0, 1\}\}$ 的全体), 则 $\mathcal{H}$ 不是 realizable PAC learnable 的.

*Proof*. 
- 用反证法. 假设 $\mathcal{H}$ 是 PAC 可学习的, 则存在一个学习算法 $\mathsf{A}$, 使得对于任意 $\epsilon, \delta \in (0, 1)$ (不妨 specify 为 $\epsilon < 1/8, \delta < 1/7$), 存在一个多项式函数 $m_{\mathcal{H}}(\epsilon, \delta)$, 使得对于任意定义在 $\mathcal{X} \times \mathcal{Y}$ 上的分布 $\mathcal{D}$, 当该分布本身为 realizable 的 (即存在 $h^* \in \mathcal{H}$, 使得 $L_{\mathcal{D}}(h^*) = 0$), 及当训练集大小 $m \geq m_{\mathcal{H}}(\epsilon, \delta)$ 时, 有
    $$
    \mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(\mathcal{A}(S)) \leq \epsilon] \geq 1 - \delta \implies \mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(\mathcal{A}(S)) > 1/8] \geq 6/7
    $$

- 然而根据 PAC 的定义, 由于 $\mathcal{X}$ 是无限的, 则对于任意训练集大小 $m$, 均存在一个 '坏' 分布 $\mathcal{D}$, 使得上述算法 $\mathsf{A}$ 在该分布下失败, 即
    $$
    \mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(\mathsf{A}(S)) > 1/8] > 1/7
    $$

- 二者相加将使得概率之和大于 1, 故矛盾, 因此 $\mathcal{H}$ 不是 realizable PAC learnable 的.

$\square$

## 5.2 Error Decomposition

对于 ERM 的输出 $h_S \in \argmin_{h \in \mathcal{H}} L_S(h)$, 其 true risk 可以被分解为 approximation error 和 estimation error 之和:
$$
L_{\mathcal{D}}(h_S) = \underbrace{L_{\mathcal{D}}(h^*)}_{\epsilon_\text{app}} + \underbrace{(L_{\mathcal{D}}(h_S) - L_{\mathcal{D}}(h^*))}_{\epsilon_\text{est}}
$$

该式子的成立性本身是 trivial 的, 而其背后的结构是更为重要的. 
- **Approximation Error** ($\epsilon_\text{app} = L_{\mathcal{D}}(h^*) = \min_{h \in \mathcal{H}} L_{\mathcal{D}}(h)$): 
  - 含义: $\mathcal{H}$ 的 expressiveness 的理论上限, 是 $\mathcal{H}$ 中最优的 hypothesis 的 true risk.
  - 其刻画了 **inductive bias** 的大小 (bias-complexity tradeoff 中的 bias).
  - 具有单调性: 若 $\mathcal{H}_1 \subseteq \mathcal{H}_2$, 则 $\epsilon_\text{app}(\mathcal{H}_1) \geq \epsilon_\text{app}(\mathcal{H}_2)$, 即更大的 hypothesis class 可以更好地拟合数据, 减少 approximation error.
  - 与 $m$ 无关, 仅与 $\mathcal{H}$ 的选择有关.
  - 在 realizable 的情况下, $\epsilon_\text{app} = 0$.

    
- **Estimation Error** ($\epsilon_\text{est} = L_{\mathcal{D}}(h_S) - L_{\mathcal{D}}(h^*)$):
  - 含义: 由于训练集的有限性, 学习算法只能在有限的训练集上进行学习, 因此其输出的 hypothesis $h_S$ 可能无法达到 $\mathcal{H}$ 中最优的 true risk $h^*$. 
  - 该误差刻画了学习算法的泛化能力 (bias-complexity tradeoff 中的 complexity).

因此上述的 decomposition 就引出了最重要的 tradeoff: **bias-complexity tradeoff**. $\mathcal{H}$ 的选择越大, approximation error 越小, 但 estimation error 越大; 反之亦然.


> [!note]
>
> - 理论上还应考虑 Bayes error. 即由于数据条件分布本身的噪声 ($\mathcal{D}(Y|X)$, 即同一个输入 $X$ 可能对应多个标签 $Y$), 即使 $\mathcal{H}$ 是全体函数, 也无法拟合数据. 此时记 $\epsilon_\text{Bayes} = L_{\mathcal{D}}(h^*_\text{Bayes})$, 其中 $h^*_\text{Bayes} = \argmin_{h: \mathcal{X} \to \mathcal{Y}} L_{\mathcal{D}}(h)$, 则此时的 Decomposition 为:
>   $$
>   L_{\mathcal{D}}(h_S) = \underbrace{L_{\mathcal{D}}(h^*_\text{Bayes})}_{\epsilon_\text{Bayes}} + \underbrace{L_{\mathcal{D}}(h^*) - L_{\mathcal{D}}(h^*_\text{Bayes})}_{\epsilon_\text{app}} + \underbrace{L_{\mathcal{D}}(h_S) - L_{\mathcal{D}}(h^*)}_{\epsilon_\text{est}}
>   $$
>   其数值上, 这里定义的 approximation error 只与前文定义的 approximation error 相差一个常数 $\epsilon_\text{Bayes}$, 而 estimation error 则不变. 因此在进行 tradeoff 比较时, 该常数项可以忽略不计. 然而若讨论其绝对数值时, 则需要考虑 Bayes error 的存在.
>
> - 若考虑实际, 可能还要额外考虑 optimization error, 即由于算法本身的优化能力有限, 可能无法找到 $\mathcal{H}$ 中最优的 hypothesis $h^*$. 这时的 decomposition 为:
>   $$
>   L_{\mathcal{D}}(h_S) = \underbrace{L_{\mathcal{D}}(h^*_\text{Bayes})}_{\epsilon_\text{Bayes}} + \underbrace{L_{\mathcal{D}}(h^*) - L_{\mathcal{D}}(h^*_\text{Bayes})}_{\epsilon_\text{app}} + \underbrace{L_{\mathcal{D}}(h_S) - L_{\mathcal{D}}(h^*)}_{\epsilon_\text{est}} + \underbrace{L_{\mathcal{D}}(\tilde h_S) - L_{\mathcal{D}}(h_S)}_{\epsilon_\text{opt}}
>   $$