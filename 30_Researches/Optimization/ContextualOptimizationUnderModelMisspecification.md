# Contextual Optimization Under Model Misspecification: A Tractable and Generalizable Approach

## Introduction

- **Contextual Optimization**: 在现实决策中, 很多优化目标参数是不确定的. 但在决策之前能观测到一些有关的 contextual information 来帮助了解决策参数. 
  - 例如: 一个交通路网有 $d$ 条连边, 定义 $\mathbf{c} \in \mathbb{R}^d$ 为每条边的通行成本, 其在决策前是未知的. 但在决策前可以观测到一些 contextual information $\mathbf{x} \in \mathbb{R}^k$ (例如天气, 时间, 交通流量等), 算法会根据这些 contextual information, 根据某种 policy $\pi(\mathbf{x})$ 来进行具体决策 $\mathbf{w} = \pi(\mathbf{x})$ (例如选择一条路径, 或流量分配等). 
  - 文章定义, 标准的 contextual optimization 问题为:
    $$
    \min_{\pi \in \Pi} \quad \mathbb{E}_{ (\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \mathbf{c}^\top \pi(\mathbf{x}) \right]
    $$
    - $(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}$ 表示二者的联合分布, 强调二者之间的相关性, contextual information $\mathbf{x}$ 可以帮助了解 $\mathbf{c}$ 的分布.
    - $\Pi$ 表示 policy 的函数空间, 其中 $\pi: \mathbb{R}^k \to \mathcal{W} \subseteq \mathbb{R}^d$.  $\mathcal{W}$ 表示决策空间 (例如路径选择集合, 资产权重, 订货量等), 且假设其是 convex & bounded. 
  - 然而由于真实的联合分布 $\mathcal{P}$ 是未知的, 因此上述优化问题无法直接求解. 我们有的是一些历史数据 $\{(\mathbf{x}_i, \mathbf{c}_i)\}_{i=1}^n$ , 因此需要从某个 hypothesis class $\mathcal{H}$ 中学习一个参数化 predictor $\mathbf{\hat{c}}_\theta(\mathbf{x}) \in \mathbb{R}^d$ 来预测 $\mathbf{c}$, 进而求解.

- 对于 contextual optimization, 最经典的方法是 *predict-then-optimize (PTO)*: 先学习一个 predictor $\mathbf{\hat{c}}_\theta(\mathbf{x})$, 而后求解
    $$
    \mathbf{\widehat{w}} \in \arg\min_{\mathbf{w} \in \mathcal{W}} \quad \mathbf{\hat{c}}_\theta(\mathbf{x})^\top \mathbf{w}
    $$
    注意, 这里的实际支付成本应当是 $\mathbf{c}^\top \mathbf{\widehat{w}}$, 而不是 $\mathbf{\hat{c}}_\theta(\mathbf{x})^\top \mathbf{\widehat{w}}$. 落实到具体实现, 还有如下两个路径:
    - Sequential Learning and Optimization (SLO): 即传统的先预测后决策, 两阶段方法. 
    - Integrated Learning and Optimization (ILO): 即端到端方法, 直接在预测时就考虑到后续优化效果. 

- 然而上述两种方法都假设 hypothesis class $\mathcal{H}$ 是 well-specified 的, 即存在一个 $\mathbf{\widehat{c}}_\theta(\mathbf{x}) \in \mathcal{H}$ 使得
    $$
    \mathbf{\widehat{c}}_\theta(\mathbf{x}) = \mathbf{c}^\star(\mathbf{x}) \quad \text{\small{almost everywhere}}
    $$
    但这并非现实中常见的情况. 更为现实的场景往往是 misspecified 的, 即使样本量趋于无穷大, 算法收敛到全局最优解, 但可能由于模型本身表现能力不足, 仍然无法得到最优的决策. 例如有如下几个重要的 misspecification 来源:
    - **Incomplete features**: 存在重要的遗漏特征
    - **Unmodeled dependencies**: 存在未建模的依赖关系, 例如 $\mathbf{c}$ 的分布可能依赖于 $\mathbf{x}$ 的某些非线性组合, 但模型假设是线性的.
    - **Distribution shift**: 训练数据和测试数据的分布不一致, 例如训练数据来自于过去的交通流量, 而测试数据来自于未来的交通流量, 其分布可能发生了变化.

- 本文核心重点区分了如下两种 misspecification:
  - **Prediction misspecification**: 即 $\mathcal{H}$ 中不存在一个 predictor $\mathbf{\hat{c}}_\theta(\mathbf{x})$ 能够准确预测 $\mathbf{c}$. 典型的度量是:
    $$
    \inf_{\mathbf{\hat{c}}_\theta \in \mathcal{H}} \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \| \mathbf{\hat{c}}_\theta(\mathbf{x}) - \mathbf{c} \|^2_2 \right] > 0
    $$

  - **Decision misspecification**: 即 $\mathcal{H}$ 不存在一个 predictor $\mathbf{\hat{c}}_\theta(\mathbf{x})$, 使其能够诱导的决策 $\mathbf{\widehat{w}}$ 与真实 cost 所诱导的最优决策 $\mathbf{w}^\star$ 几乎处处相同. 
  
  
- 类似 SPO 当时的论述, 预测的好坏并不能决定性的影响最终决策的好坏. 
  - 最重要的不是 predicion , 而是 decision misspecification, 即当前的 $\mathbf{\widehat{c}}$ 能否有导出 $\mathcal{H}$ 内最好的 decision.
  - 然而文中指出, 即使是 SPO 尽管已经是 decision awareness 的算法, 但仍没有给出在 misspecification 下的 consistency guarantee. 
    <!-- - *Counter-example*: 
    - 考虑一个一维情景. $x$ 是输入 context 特征, $c \in \{-1,1\}$ 是真实标签. 预测模型给出分数 $\hat{c}(x)$, 并可以通过 $\text{sign}(\hat{c}(x))$ 来做出标签预测. 此外, 决策空间 $\mathcal{W} = [-1/2, 1/2]$. 下游决策问题希望在给定预测值 $\hat{c}(x)$ 的情况下, 最大化决策收益 $\hat{c}(x) \cdot w$:
        $$
        w(\hat{c}(x)) \in \arg\max_{w \in [-1/2, 1/2]} \hat{c}(x) \cdot w 
        $$
        下根据预测出的 $\hat{c}(x)$ 的符号分情况讨论:
        - 若 $\hat{c}(x) > 0$, 若使得 $\hat{c}(x) \cdot w$ 最大化, 则 $w(\hat{c}(x)) = 1/2$.
        - 若 $\hat{c}(x) < 0$, 若使得 $\hat{c}(x) \cdot w$ 最大化, 则 $w(\hat{c}(x)) = -1/2$.
        
        综上, 
         $$
         w(\hat{c}(x)) = \frac{1}{2} \cdot \text{sign}(\hat{c}(x))
         $$
    - 假设这里的 context $x$ 只有两个等可能的取值 $\mathbb{P}(x=1) = \mathbb{P}(x=2) = 1/2$, 且真实标签 $c \equiv 1$. 因此不论输入 $x$ 是什么, 对应的正确分类都是 $c = 1$, 对应的正确决策都是 $w^\star = 1/2$.  -->

### Problem Setup

下正式给出问题 formulation. 在观察到某个具体的 context $\mathbf{x}$ 后, 我们希望求解如下的决策问题:
$$
\min_{\mathbf{w} \in \mathcal{W}} \quad \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \mathbf{c}^\top \mathbf{w} \mid \mathbf{x} \right] = \min_{\mathbf{w} \in \mathcal{W}} \quad \mathbb{E} \left[ \mathbf{c} \mid \mathbf{x} \right]^\top \mathbf{w}
$$
由于分布 $\mathcal{P}$ 未知, 因此先根据历史数据 $\{(\mathbf{x}_i, \mathbf{c}_i)\}_{i=1}^n$ 在参数化 hypothesis class $\mathcal{H}$ 中学习一个 predictor $\mathbf{\hat{c}}_\theta(\mathbf{x})$ (可能是 misspecified 的), 而后求解 decision
$$
\mathbf{\widehat{w}}(\mathbf{\hat{c}}_\theta(\mathbf{x})) \in \arg\min_{\mathbf{w} \in \mathcal{W}} \quad \mathbf{\hat{c}}_\theta(\mathbf{x})^\top \mathbf{w} =: \mathcal{W}^\star(\mathbf{\hat{c}}_\theta(\mathbf{x}))
$$
由于最优决策 $\mathbf{w}^\star(\mathbf{x})$ 可能不唯一, 因此定义 $\mathcal{W}^\star(\mathbf{c})$ 为所有最优决策的集合. 而最终的优化目标损失函数则定义为, 在最优决策集合中的, 以真实成本 $\mathbf{c}$ (而不是预测成本 $\mathbf{\hat{c}}_\theta(\mathbf{x})$) 计算的 worst-case 成本之期望:
$$
\ell_\mathcal{P}(\theta) := \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \max_{\mathbf{w} \in \mathcal{W}^\star(\mathbf{\hat{c}}_\theta(\mathbf{x}))} \mathbf{c}^\top \mathbf{w} \right]
$$


## Main Approach

### Rewriting target loss under uniqueness assumption

文章指出, $\mathcal{W}^\star(\mathbf{\hat{c}}_\theta(\mathbf{x}))$ 尽管理论上可能多解, 但实践中, 若 $\mathbf{\hat{c}}_\theta(\mathbf{x})$ 是连续的, 且 $\mathcal{W}$ 是 polyhedron, 则往往具有唯一解. 更严谨地说, 若对于任意模型参数 $\theta \in \mathbb{R}^m$, almost surely 有 $\mathbf{\hat{c}}_\theta(\mathbf{x}) \neq 0$, 则 $\mathcal{W}^\star(\mathbf{\hat{c}}_\theta(\mathbf{x}))$ almost surely 是唯一的. 

在唯一性假设成立下, 最小化上述 target loss 就可以改写为:
$$
\begin{aligned}
\min_{\theta \in \mathbb{R}^m} \left\{\ell_\mathcal{P}(\theta)  =  \min_{w_{\mathcal{P}} \in \mathcal{W}_{\mathcal{P}}} \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \mathbf{c}^\top {w}_\mathcal{P}(\mathbf{x}) \right]\right \}\\
\text{s.t.} \quad w_\mathcal{P}(\mathbf{x}) \in \arg\min_{\mathbf{w} \in \mathcal{W}} \quad \mathbf{\hat{c}}_\theta(\mathbf{x})^\top \mathbf{w}, \quad \text{for all  } \mathbf{x} \in \mathbb{R}^k
\end{aligned}
$$
其中强调 $\mathcal{W}_{\mathcal{P}}$ 是决策规则的集合, 其元素 $w_\mathcal{P}(\mathbf{x})$ 不是具体的决策, 而是一个函数, 其输入是 context $\mathbf{x}$, 输出是 $\mathcal{W}$ 中的一个决策.

### Surrogate loss: Consistent Integrated Learning and Optimization (CILO)

给定所有 measurable decision mappings 集合 $\mathcal{W}_{\mathcal{P}}$, 提取所有在真实成本下诱导出的 cost 不超过 $\beta \in \mathbb{R}$ 的 policy subset 为
$$
\mathcal{\bar{W}}_{\mathcal{P}}^\beta := \left\{ \bar{w}_\mathcal{P} \in \mathcal{W}_{\mathcal{P}}: \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \mathbf{c}^\top \bar{w}_\mathcal{P}(\mathbf{x}) \right] \leq \beta \right\}
$$
由此给出 CILO surrogate loss: 对于任意 $\beta \in \mathbb{R}$, 任意 $\theta \in \mathbb{R}^m$, 定义
$$
\begin{aligned}
\ell_\mathcal{P}^\beta(\theta) := \min_{\bar{w}_\mathcal{P} \in \mathcal{\bar{W}}_{\mathcal{P}}^\beta} \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \mathbf{\hat{c}}_\theta(\mathbf{x})^\top \bar{w}_\mathcal{P}^\beta(\mathbf{x}) \right]
-
\min_{w_\mathcal{P} \in \mathcal{W}_{\mathcal{P}}} \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \mathbf{\hat{c}}_\theta(\mathbf{x})^\top {w}_\mathcal{P}(\mathbf{x}) \right]
\end{aligned}
$$

该 loss 可以理解为如下两个部分的差值:
- 第一个 loss 表示: 在真实 cost 确定的 performance 达到 $\beta$ 的 policy subset 中, 根据预测 cost $\mathbf{\hat{c}}_\theta$ 的评价下, 最好的 objective value.
- 第二个 loss 表示: 在所有 policy 中, 根据预测 cost $\mathbf{\hat{c}}_\theta$ 的评价下, 最好的 objective value.

因此这个 loss 的意义在于, 在限制 policy 为真实 cost 不超过 $\beta$ 的 subset 中, 其在预测 cost 下的表现与在所有 policy 中的表现的 gap.

不难看出, $\ell_\mathcal{P}^\beta(\theta) \geq 0$ 对于任意 $\theta, \beta$ 都成立. 而当且仅当存在某个 policy $w_\mathcal{P} \in \mathcal{W}_{\mathcal{P}}$ 使得其在真实 cost 下的 performance 不超过 $\beta$, 且在预测 cost 下的 performance 与所有 policy 中的 best performance 一致时, $\ell_\mathcal{P}^\beta(\theta) = 0$. 

接着定义 
$$
\beta^\star_{\mathcal{H}, \mathcal{P}} := \min_{\hat{c}_\theta \in \mathcal{H}} \ell_\mathcal{P}(\mathbf{\hat{c}}_\theta) = \min_{\hat{c}_\theta \in \mathcal{H}} \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \max_{\mathbf{w} \in \mathcal{W}^\star(\mathbf{\hat{c}}_\theta(\mathbf{x}))} \mathbf{c}^\top \mathbf{w} \right]
$$
即在 hypothesis class $\mathcal{H}$ 中最佳 predictor $\mathbf{\hat{c}}_\theta$ 所诱导的 policy 的真实 loss. 代表当前 hypothesis class $\mathcal{H}$ 的 best-in-class performance. 以及
$$
\beta_{\max, \mathcal{P}} := \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \max_{\mathbf{w} \in \mathcal{W}} \mathbf{c}^\top \mathbf{w} \right]
$$
即在所有 policy 中, 真实 cost 的 worst-case performance. 代表了当前问题的 inherent difficulty.

文章的第一个重要结论是, 对于任意 $\beta_{\mathcal{H}, \mathcal{P}}^\star \leq \beta < \beta_{\max, \mathcal{P}}$, 且对每个满足 $\mathbf{\hat{c}}_\theta \neq 0$ 的 $\theta \in \mathbb{R}^m$, 有
$\ell_\mathcal{P}(\theta) \leq \beta$当且仅当 $\theta$ 是 $\ell_\mathcal{P}^\beta$ 的 minimizer. 特别地, 当 $\beta = \beta_{\mathcal{H}, \mathcal{P}}^\star$ 时, $\theta$ 是 $\ell_\mathcal{P}$ 的 minimizer 当且仅当 $\theta$ 是 $\ell_\mathcal{P}^\beta$ 的 minimizer. 