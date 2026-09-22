# Online Learning and Online Convex Optimization: Introduction

> [!quote]
> Ref: Online Learning and Online Convex Optimization, Shai Shalev-Shwartz. 

## Brief Introduction

首先给出总览.  稍后会在后续章节中逐步展开具体的数学定义. 

### Intuition

Online Learning 的信息是序列式的, 逐渐到达的. 在第 $t$ 轮, learner 从 Instance Domain $\mathcal{X}$ 中观察到一个实例 (问题) $\mathrm{x}_t$, 并被要求进行预测 $p_t$ (例如, 分类或回归). 在做出预测之后, learner 收到一个标签 (答案) $y_t \in \mathcal{Y}$, 并得到一个对应损失 $l(p_t, y_t)$. 有时为方便起见, 也往往会让输出 $p_t$ 来自一个更大的预测空间 $\mathcal{D}$ (例如对于分类问题, $\mathcal{Y} = \{0, 1\}$, 但往往 $\mathcal{D} = [0, 1]$). 

总的而言, 我们希望从历史序列 $\{(\mathrm{x}_1, y_1), \ldots, (\mathrm{x}_{t-1}, y_{t-1})\}$ 中学习数据模式, 以便在第 $t$ 轮做出更好的预测:
- 在传统的统计学习 sequential prediction 中, 我们往往假设数据是独立同分布 (i.i.d.) 的, 以捕捉数据模式.
- 在 Online Learning  中, 往往会放宽这个假设, 允许数据是任意的, 其可以是 deterministic, stochastic (但未必是固定的分布), 甚至是 adversarial (可以根据 learner 本身的行为来进行适应性的对抗调整).

不过为了确保能学到非平凡的 learner, 一般考虑如下限制:
- **Realizability**: 尽管数据的生成是任意的, 但所有答案都来自某个目标映射:
    $$
    h^\star : \mathcal{X} \to \mathcal{Y}, \quad h^\star \in \mathcal{H}.
    $$
    其中 $\mathcal{H}$ 是一个 fix 的 hypothesis class, 并且是对 learner 已知的. 称符合这种设置的情况为 realizable case, 因为存在一个假设 $h^\star$ 能够完美地拟合数据:
    $$
    \exists h^\star \in \mathcal{H}, \quad \text{s.t.} \quad h^\star(\mathrm{x}_t) = y_t, \forall t.
    $$
    - 在 realizable 的情况下, 可能存在多个 realizable 的 $h^\star$. 给定一个 online learning algorithm $A$ (目前可以粗略的理解成是一个根据历史决策数据和当期输入来估计当前标签的 $(\mathcal{X} \times \mathcal{Y})^{t-1} \times \mathcal{X} \to \mathcal{D}$ 的决策映射), 则将 $A$ 在这些 $h^\star$ 上的 worst-case performance (最大犯错数), 记为 $M_A(\mathcal{H})$, 作为 $A$ 的性能指标. 注意, 这里只要求 $h^\star$ 是给定的, 具体的序列 $\{(\mathrm{x}_t, y_t)\}$ 仍然可以是任意甚至 adversarial 的.
    - 一个关于 $M_A(\mathcal{H})$ 的上界就是 $A$ 的 mistake bound. Online Learning 的一个重要目标就是设计 mistake bound 尽可能小的算法 $A$.


- **Regret**:  Realizability 的假设过于严格, 往往现实世界中由于数据的噪声等, 并不能找到一个 $h^\star$ 能够完美拟合所有的数据. 这时, 可以放宽让 learner 的表现不必完美, 而是和 $\mathcal{H}$ 中表现最好的假设进行比较. 
  - 假设算法 $A$ 在 $T$ 轮中给出了序列 $\{p_1, \ldots, p_T\}$ 的预测, 以及对应的数据序列 $\{(\mathrm{x}_1, y_1), \ldots, (\mathrm{x}_T, y_T)\}$, 则对于 $\mathcal{H}$ 中的任意一个假设 $h^\star$, 可以定义该 $h^\star$ 的 regret 为:
    $$
    \text{Regret}_T(h^\star) = \sum_{t=1}^T l(p_t, y_t) - \sum_{t=1}^T l(h^\star(\mathrm{x}_t), y_t).
    $$
    - 其含义为: 累计的 $T$ 轮中, learner predict 出的损失与 $h^\star$ 的理想损失之间的差距. 
  - 将 $\mathcal{H}$ 中的所有 $h^\star$ 的 regret 的 worst-case (最大) 作为 $A$ 的性能指标, 记为
    $$
    \text{Regret}_T(\mathcal{H}) = \max_{h^\star \in \mathcal{H}} \text{Regret}_T(h^\star).
    $$
    或等价地
    $$
    \text{Regret}_T(\mathcal{H}) = \sum_{t=1}^T l(p_t, y_t) - \min_{h^\star \in \mathcal{H}} \sum_{t=1}^T l(h^\star(\mathrm{x}_t), y_t).
    $$
  - 在当前的假设下, 最终的学习目标为希望最终的 regret 是 sublinear 的, 即
    $$
    \text{Regret}_T(\mathcal{H}) = o(T).
    $$
    这时, learner 的平均 regret 就会趋近于 0, 即
    $$
    \lim_{T \to \infty} \frac{\text{Regret}_T(\mathcal{H})}{T} = 0.
    $$
    - 这在说明: 随着轮次的累计, 长期的平均表现会趋近于 $\mathcal{H}$ 中的最优表现.

### Examples

下面给出几个 online learning 在不同场景下的具体的例子, 以说明其在不同场景下的一般性和适用性.

-  **Online Linear Regression**:
   - 考虑 $\mathcal{X} = \mathbb{R}^d$, $\mathcal{Y} = \mathcal{D} = \mathbb{R}$, 以及 MSE (mean squared error) $l(p, y) = (p - y)^2$. Hypothesis class 考虑最简单的线性模型:
   $$
   \mathcal{H} = \{h_w : h_w(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}, \mathbf{w} \in \mathbb{R}^d\}.
   $$
   其整体的设定和一般的 OLS 无异.


 - **Prediction with Expert Advice**:
   - 这是一个非常经典的 online learning 的例子. 设 $\mathbf{x}_t \in \mathcal{X} \subset \mathbb{R}^d$ 是每一轮的输入, 其第 $i$ 个分量 $\mathbf{x}_t[i]$ 可以被看成是第 $i$ 个专家 (expert) 在第 $t$ 轮的建议. 在每一轮 learner 需要从中选择一个专家的建议来进行预测. 在进行选择后, learner 会收到 label $\mathbf{y}_t \in \mathcal{Y} = [0,1]^d$, 一个 $d$ 维的 $[0,1]$ 区间内的向量, 其第 $i$ 个分量 $\mathbf{y}_t[i]$ 可以被看成是如果选择了第 $i$ 个专家的建议, 则后验的真实 cost. 因此, 在每轮决策中, 我们的实际决策损失也可以直接通过 $\mathbf{y}_t$ 来计算:
    $$
    l(p_t, \mathbf{y}_t) = \mathbf{y}_t[p_t].
    $$
    - 一个常见的 $\mathcal{H}$ 的设定是 $\mathcal{H} = \{h_1, \ldots, h_d\}$, 其中 $h_i(\mathbf{x}) = i, \forall \mathbf{x}$. 也就是说, $\mathcal{H}$ 中的每个假设 $h_i$ 都表示, 在每轮的建议就是永远选择第 $i$ 个专家的建议, 即一个 constant predictor. 则此时的总的 $\mathcal{H}$ 的 regret 记为 learner 的损失, 与 $\mathcal{H}$ 一直 follow 着一个固定专家的最优损失之间的差距. 
        $$
        \text{Regret}_T(\mathcal{H}) = \sum_{t=1}^T l(p_t, \mathbf{y}_t) - \min_{i \in [d]} \sum_{t=1}^T l(h_i(\mathbf{x}_t), \mathbf{y}_t) = \sum_{t=1}^T \mathbf{y}_t[p_t] - \min_{i \in [d]} \sum_{t=1}^T \mathbf{y}_t[i].   
        $$

      - 这是一个相对简化的例子, 我们只需要比较单一的固定专家, 而不是每一轮都切换当前轮的最优专家. 后者则为更困难的 dynamic 问题. 


- **Online Ranking**
  - 在第 $t$ 轮, learner 收到一个 query $\mathbf{x}_t$ (例如, 一个搜索引擎的查询), 以及一个 document set $\mathcal{D}_t$ (例如, 搜索引擎返回的一系列文档). Learner 需要根据 $\mathbf{x}_t$ 和 $\mathcal{D}_t$ 来给出一个 ranking $\pi_t$ (例如, 对 $\mathcal{D}_t$ 中的文档进行排序). 在做出 ranking 之后, learner 会收到真实答案 $\mathbf{y}_t \in \mathcal{Y} = \{1, \ldots, |\mathcal{D}_t|\}$, 其表示用户最终选择的文档在 $\mathcal{D}_t$ 中的排名位置. 因此, learner 的损失可以通过 $\mathbf{y}_t$ 来计算:
    $$
    l(\pi_t, \mathbf{y}_t) = \pi_t^{-1}(\mathbf{y}_t).
    $$
    - 这里 $\pi_t^{-1}(\mathbf{y}_t)$ 表示 $\mathbf{y}_t$ 在 ranking $\pi_t$ 中的位置. 例如, 如果 $\pi_t$ 将 $\mathbf{y}_t$ 排在第一位, 则损失为 1; 如果排在第二位, 则损失为 2, 以此类推. 因此, learner 的目标是尽可能地将用户选择的文档排在前面, 从而最小化损失


### A Gentle Start 

#### Cover's Impossibility

考虑二分类问题. 在每一轮, 输入 $\mathbf{x}_t \in \mathcal{X}$, 真实标签 $y_t \in \mathcal{Y} = \mathcal{D} = \{0, 1\}$, learner 的预测 $p_t \in \mathcal{D} \in \{0, 1\}$. 损失函数为 $l(p_t, y_t) = |p_t - y_t|$, 即预测错误则损失为 1, 预测正确则损失为 0. 此外, 假设总的的 hypothesis class $\mathcal{H} < \infty$ 有限. 定义总的 regret 为:
$$
\text{Regret}_T(\mathcal{H}) = \max_{h^\star \in \mathcal{H}} \Bigl (\sum_{t=1}^T |p_t - y_t| - \sum_{t=1}^T |h^\star(\mathbf{x}_t) - y_t| \Bigr).
$$
并且我们的目标是找到一个 sublinear 的 regret 的算法 $A$. 

然而我们这里说明: 如果没有上文提到的例如 realizable 的假设, 这样的 sublinear regret 的算法 $A$ 是不存在的, 即使 $\mathcal{H} = \{h_0, h_1\}$ 只有两个假设, 其中 $h_0(\mathbf{x}) \equiv 0$ 和 $h_1(\mathbf{x}) \equiv 1$ 是两个 constant predictor.
- 回顾, online learning 的交互顺序为: learner 看到 $\mathbf{x}_t \rightarrow$ learner 预测 $p_t \rightarrow$ 环境 (adversary) 看到 $p_t$ 后揭示 $y_t \rightarrow$ learner 计算损失 $l(p_t, y_t)$.  因此, adversary 可以每次都选择与预测完全相反的标签来进行揭示, 以确保 learner 每轮都犯错, 故总的犯错次数为
    $$
    \sum_{t=1}^T |p_t - y_t| = T.
    $$

- 然而, 在总的 $T$ 轮的过程中, 总有一个答案 $\{0, 1\}$ 是出现次数较多的, 例如, 如果 $0$ 出现的次数较多, 则 $h_0$ 的损失就会较小. 因此, learner 的 regret 就会是
    $$
    \text{Regret}_T(\mathcal{H}) = T - \min_{h^\star \in \mathcal{H}} \sum_{t=1}^T |h^\star(\mathbf{x}_t) - y_t| = T - \min\{\text{count of } 0, \text{count of } 1\} \geq T/2.
    $$
    这显然是一个 linear 的 regret, 而非 sublinear 的.


#### Realizability 假设, Consistent Algorithm 与 Halving Algorithm


在 realizable 的假设下 (即存在一个 $h^\star \in \mathcal{H}$ 对于任意 $t$, 都有 $y_t = h^\star(\mathbf{x}_t)$).

***Consistent Algorithm***

一个自然的算法设计思路是: 既然 $\mathcal{H}$ 是一个 finite 的 hypothesis class 且对 learner 已知, 并且 realizability 的假设保证了 $\mathcal{H}$ 中至少存在一个 $h^\star$ 能够完美拟合数据, 则 learner 可以在每轮中维护一个 version space $\mathcal{V}_t \subseteq \mathcal{H}$, 其包含了所有在前 $t-1$ 轮中都没有犯错的假设. 在第 $t$ 轮, learner 可以从 $\mathcal{V}_t$ 中选择一个假设 $h_t$ 来进行预测. 其算法思路如下:

```
Algorithm: Consistent Algorithm for Realizable Case
INPUT: Finite hypothesis class H, version space V_1 = H
FOR t = 1, 2, ...
    Receive instance x_t
    Choose any h_t in V_t
    Predict p_t = h_t(x_t)
    Receive true answer y_t = h^*(x_t)
    Update version space: V_{t+1} <- {h in V_t : h(x_t) = y_t}
```

- 注意, 每次更新的时候, 并不是只关注当前 predict 用的那个 $h_t$, 而是将 version space 中所有在当前轮犯错的假设都剔除掉. 这样, 随着轮次的累计, version space 会逐渐缩小, 每当有一次犯错, 就会剔除掉至少一个假设. 因此, 若总共有 $M$ 次犯错, 则 version space 中的元素数量至多为:
    $$
    |\mathcal{V}_{t}| \leq |\mathcal{H}| - M.
    $$

- 由于 realizability 的假设保证了 $\mathcal{H}$ 中至少存在一个 $h^\star$ 能够完美拟合数据, 因此 version space 中至少还会剩下一个 $h^\star$, 即
    $$
    |\mathcal{V}_{t}| \geq 1.
    $$

- 综上, 可以得到
    $$
    M_C \leq |\mathcal{H}| - 1.
    $$
    也就是说, 在 realizable 的假设下, 这个 consistent algorithm 最多犯错的次数是 $|\mathcal{H}| - 1$.


上述的算法是符合直觉的, 但并不有效. 因为我们总可以构造一个序列使得算法恰需要 $|\mathcal{H}| - 1$ 次犯错, 以达到上界. 例如, 让每个错误的假设有且仅有一处错误, 并且每次犯错都恰好剔除掉一个假设. 这时, learner 的犯错次数就会达到 $|\mathcal{H}| - 1$ 的上界.

***Halving Algorithm***

Halving 是一个更聪明的算法设计, 其核心思路在于: 在每轮的预测时, 不再任选一个假设 $h$ 来进行预测, 而是选择 version space 中的 majority vote 来进行预测 (即, 对于当前轮的输入 $\mathbf{x}_t$, 统计 version space 中所有假设 $h$ 的预测 $h(\mathbf{x}_t)$ 的结果, 选择出现次数较多的那个结果作为预测 $p_t$), 这样能够保证, 如果当前步犯错了, 则至少能够剔除掉 version space 中一多半的假设, 从而更快地缩小 version space. 
- 注意, halving 和 consistent algorithm 的区别在并不在于其更新 version space 的方式, 而是在于其预测的方式. Halving 的核心目的是让每次犯错都能够剔除掉更多的假设, 从而更快地缩小 version space. 

- 另一方面, Halving 算法的意义也不在于加速收敛次数找到最优的 $h^\star$, 而是在于减少犯错的次数 (让每次犯错更有价值), 这样能够让下一个 turn 的预测更有可能是正确的.

```
Algorithm: Halving Algorithm for Realizable Case
INPUT: Finite hypothesis class H, version space V_1 = H
FOR t = 1, 2, ...
    Receive instance x_t
    Predict p_t = majority vote of {h(x_t) : h in V_t}
    Receive true answer y_t 
    Update version space: V_{t+1} <- {h in V_t : h(x_t) = y_t}
```

不难看出, 由于是众数投票, 故如果当前轮犯错了, 则至少能够剔除掉 version space 中的一半的假设, 故总的犯错次数 $M$ 满足
$$
M_H \leq \log_2(|\mathcal{H}|).
$$

在后续的章节中还将给出更快的算法收敛设计. 


#### Randomization

另一种试图突破 Cover's impossibility 的思路是引入随机化, 不过这事实上相当于对于整体的 online learning 的机制设定上做了一个放松. 具体而言, 在第 $t$ 轮, 其信息流为:

1. Learner 看到 $\mathbf{x}_t$. Learner 根据 $\mathbf{x}_t$ 和历史数据 $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_{t-1}, y_{t-1})\}$ 来计算一个概率参数 $\mathrm{p}_t \in [0, 1]$, 其表示在当前轮选择 $1$ 的概率.
2. Adversary 知道这一轮的真实标签 $y_t \in \{0,1\}$, 历史的所有数据, 整体的随机化策略, 以及 learner 的随机参数 $\mathrm{p}_t$, 但不知道 learner 在当前轮的具体随机采样结果 (即, learner 最终选择了 $0$ 还是 $1$). Adversary 根据这些信息来揭示 $y_t$.

此时, 由于随机性的引入, 评价指标应当转而变成期望意义上的. 在每一轮 $t$ 上, learner 的输出 $\hat{y}_t$ 是一个 Bernoulli 随机变量, 其 $\mathbb{P}(\hat{y}_t = 1) = \mathrm{p}_t$, 因此期望意义上的错误率为
$$
\mathbb{E}[ \boldsymbol{1}\{\hat{y}_t \neq y_t\} ] = \mathbb{P}(\hat{y}_t \neq y_t) = |\mathrm{p}_t - y_t|.
$$
- 最后一个等式可以简单分类讨论得到. 若 $y_t = 0$, 则 $\mathbb{P}(\hat{y}_t \neq y_t) = \mathbb{P}(\hat{y}_t = 1) = \mathrm{p}_t$. 若 $y_t = 1$, 则 $\mathbb{P}(\hat{y}_t \neq y_t) = \mathbb{P}(\hat{y}_t = 0) = 1 - \mathrm{p}_t$. 因此, 无论 $y_t$ 是 $0$ 还是 $1$, 都有 $\mathbb{P}(\hat{y}_t \neq y_t) = |\mathrm{p}_t - y_t|$.

可以证明, 在当前的随机化设定下, 对于有限的 $\mathcal{H}$, 确实存在一个 sublinear 的 low regret:
$$
\sum_{t=1}^T |\mathrm{p}_t - y_t| - \min_{h^\star \in \mathcal{H}} \sum_{t=1}^T |h^\star(\mathbf{x}_t) - y_t| \leq \sqrt{0.5 T \log(|\mathcal{H}|)}.
$$

事实上, 对于上述的两种假设 (realizable 和 randomization), 都起到相当于是一种 convexification 的技巧, 而这种凸化的思想也将是后续章节中一个重要的设计思路. 

### Structure of the Notes

事实上, Online Learning 作为一个非常 general 的学习框架, 其可以从不同的领域进行解释, 如 game theory, optimization, machine learning, information theory 等等. 而不同领域的结果往往可以通过 prediction with expert advice 来进行统一的解释. 而这个 note 主要将从 Online Convex Optimization 的角度来进行展开.

在后续的章节中, 主体的安排如下:
- [[OL02_Online_Convex_Optimization]]: 介绍 Online Convex Optimization 的基本设定
- [[OL03_Online_Classification]]: 在熟悉 Online Convex Optimization 的基础算法后, 说明其是如何解决具体的学习问题的.
- [[OL04_Limited_Feedback_Bandits]]: 介绍 bandit setting 的 online learning, 以及其与 full information setting 的区别. 该小节在讨论一些信息更为受限的 online learning 的设定, 以及其对应的算法设计和分析.
- [[OL05_Online_to_Batch_Conversion]]: 通过统计学习理论的角度来说明, 如何将 online learning 的算法和分析结果转化为 batch learning 的算法和分析结果. 