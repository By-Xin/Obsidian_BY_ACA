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

- 首先取输入空间的一个子集 $\mathcal{C} \subseteq \mathcal{X}$, 使得 $|\mathcal{C}| = 2 |S| := 2m$. 
  - 由于样本集 $S$ 的大小为 $m$, 因此 $\mathcal{C}$ 中至少有一半的样本点算法 $\mathcal{A}$ 没有见过
- 考虑所有从 $\mathcal{C}$ 到 $\{0, 1\}$ 的映射之可能 (共有 $T = 2^{2m}$ 种), 记为 $f_1, f_2, \ldots, f_T$, 其中每个 $f_i: \mathcal{C} \to \{0, 1\}$ 给出了 $\mathcal{C}$ 中每个点的标签.
  - 其穷尽了当前输入空间内的全部输入到全部 0-1 标签的映射可能. 每个 $f_i$ 就可以理解为一个具体的任务, 任务之间彼此是平行的.
  - 故对于其中第 $i$ 个映射 (任务), 考虑其对应的分布 $\mathcal{D}_i$: 从 $\mathcal{C}$ 中均匀采样一个输入点 $X\sim \text{Unif}(\mathcal{C})$, 然后将其标签 $Y$ 确定性地定为 $f_i(X)$. 故其概率分布为:
    $$
    \mathcal{D}_i\{(x, y)\} = \begin{cases}
        1/|\mathcal{C}| & \text{if } y = f_i(x) \\
        0 & \text{otherwise}
    \end{cases}
    $$
     显然, $f_i$ 是 $\mathcal{D}_i$ 下的完美 predictor, 即 $L_{\mathcal{D}_i}(f_i) = 0$.
  - 另外, 由于 $\mathcal{D}_i$ 是 $C$ 上的均匀分布且无噪声, 因此对于任意 hypothesis $h$, 有:
    $$
    L_{\mathcal{D}_i}(h) = \frac{1}{|\mathcal{C}|} \sum_{x \in \mathcal{C}} \mathbb{1}[h(x) \neq f_i(x)]
    $$

- 接着在每个任务分布 $\mathcal{D}_i$ 下, 独立同分布地抽取 $m$ 个样本点, 得到训练集 $S = \{(X_l, Y_l)\}_{l=1}^m \sim \mathcal{D}_i^m$. 考虑定义在 $\mathcal{C}$ 上的算法: $\mathcal{A}(S): \mathcal{C} \to \{0, 1\}$, 计算其 true risk 关于样本的期望: $\mathbb{E}_{S \sim \mathcal{D}_i^m}[L_{\mathcal{D}_i}(\mathcal{A}(S))]$. 取上述期望关于全部任务的最大值, 可以证明, 有:
    $$
    \max_{i \in [T]} \mathbb{E}_{S \sim \mathcal{D}_i^m}[L_{\mathcal{D}_i}(\mathcal{A}(S))] \geq \frac{1}{4} \qquad \text{(1)}
    $$
    即在 $T$ 个可能的任务中, 至少有一个任务 $A$ 的期望真实风险不小于 $1/4$:

- 