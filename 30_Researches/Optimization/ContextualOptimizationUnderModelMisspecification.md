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

因此这个 loss 的意义在于, 在限制 policy 为真实 cost 不超过 $\beta$ 的 subset 中, 其在预测 cost 下的表现与在所有 policy 中的表现的 gap. 故有关系:
$$
\ell_\mathcal{P}^\beta(\theta) = 0 \iff \ell_\mathcal{P}(\theta) \leq \beta
$$
不难看出, $\ell_\mathcal{P}^\beta(\theta) \geq 0$ 对于任意 $\theta, \beta$ 都成立.  因此, $\ell_\mathcal{P}^\beta(\theta) = 0$ 又等价于 $\theta$ 是 $\ell_\mathcal{P}^\beta$ 的 minimizer. 因此有如下定理: 

***Theorem 1*** 对于任意 $\beta \in \mathbb{R}$, 任意 $\theta \in \mathbb{R}^m$, 当 $\mathbf{\hat{c}}_\theta(\mathbf{x}) \neq 0$ 几乎处处成立时, 有 $\ell_\mathcal{P}(\theta) \leq \beta$ 当且仅当 $\theta$ 是 $\ell_\mathcal{P}^\beta$ 的 minimizer.
- 最小化 CILO loss $\ell_\mathcal{P}^\beta$ 等价于找出所有在真实 cost 下 performance 不超过 $\beta$ 的 predictor $\mathbf{\hat{c}}_\theta$.

$\diamond$

接着定义 
$$
\beta^\star_{\mathcal{H}, \mathcal{P}} := \min_{\hat{c}_\theta \in \mathcal{H}} \ell_\mathcal{P}(\mathbf{\hat{c}}_\theta) = \min_{\hat{c}_\theta \in \mathcal{H}} \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \max_{\mathbf{w} \in \mathcal{W}^\star(\mathbf{\hat{c}}_\theta(\mathbf{x}))} \mathbf{c}^\top \mathbf{w} \right]
$$
即整个 hypothesis class $\mathcal{H}$ 能达到的最优真实 decision loss. 定义
$$
\beta_{\max, \mathcal{P}} = \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left[ \max_{\mathbf{w} \in \mathcal{W}} \mathbf{c}^\top \mathbf{w} \right]
$$
即在 $\mathcal{W}$ 中效果最差的决策在真实 cost 下的 performance. 故应取 $\beta \in [\beta^\star_{\mathcal{H}, \mathcal{P}}, \beta_{\max, \mathcal{P}}]$ 来保证 surrogate loss 的 non-triviality.

此外, 若令 $\beta = \beta^\star_{\mathcal{H}, \mathcal{P}}$, 则由定理:
$$
\theta \in \arg\min_{\theta \in \mathbb{R}^m} \ell_\mathcal{P}^{\beta^\star_{\mathcal{H}, \mathcal{P}}}(\theta) \iff \ell_\mathcal{P}(\theta) \leq \beta^\star_{\mathcal{H}, \mathcal{P}} = \min_{\hat{c}_\theta \in \mathcal{H}} \ell_\mathcal{P}(\mathbf{\hat{c}}_\theta) \implies \ell_\mathcal{P}(\theta) = \beta^\star_{\mathcal{H}, \mathcal{P}}
$$
即
$$
\theta \in \arg\min_{\theta \in \mathbb{R}^m} \ell_\mathcal{P}^{\beta^\star_{\mathcal{H}, \mathcal{P}}}(\theta) \iff \theta \in \arg\min_{\theta \in \mathbb{R}^m} \ell_\mathcal{P}(\mathbf{\hat{c}}_\theta)
$$

这个定理给出了最小化真正 decision loss $\ell_\mathcal{P}$ 的 surrogate loss $\ell_\mathcal{P}^{\beta^\star_{\mathcal{H}, \mathcal{P}}}$ 的构造方法. 并且当 $\beta = \beta^\star_{\mathcal{H}, \mathcal{P}}$ 时, $\ell_\mathcal{P}^{\beta^\star_{\mathcal{H}, \mathcal{P}}}$ 的 minimizer 与 $\ell_\mathcal{P}$ 的 minimizer 是完全一致的. 然而在实践中, $\beta^\star_{\mathcal{H}, \mathcal{P}}$ 是未知的, 因此需要通过 line search 的方法, 确定若干 $\beta$ 值, 并在每个 $\beta$ 下最小化 $\ell_\mathcal{P}^\beta$, 进而找到最优的 $\beta^\star_{\mathcal{H}, \mathcal{P}}$.


## Technical Approach

这里展示了具体优化过程中的技术问题和处理方法. 主要问题包括
1. $\mathcal{P}$ 是未知的, 因此期望需要用样本均值 empirically $\mathcal{P}_n$  近似 $\mathcal{P}$. 故存在泛化性之问题. 
2. $\ell_\mathcal{P}^\beta$ 本身也是 non-convex, non-smooth  的, 故还需要额外的光滑手段. 
3.  上述定理还需要 $\mathbf{\hat{c}}_\theta(\mathbf{x}) \neq 0$ 几乎处处成立以保证下游决策的唯一性, 因此还需要具体的手段确保其成立. 

约定这里的 norm 取 $\ell_2$ norm.

### Generalization Guarantee


为确保泛化性, 做如下假设:

- Hypothesis set 是 linear 的形式, 即存在一个 linear map $\Phi : \mathbb{R}^k \to \mathbb{R}^{m \times d}$ 使得对于任意的 $\mathbf{\hat{c}}(\mathbf{x})$, 都有 $\mathbf{\hat{c}}_\theta(\mathbf{x}) = \Phi(\mathbf{x})^\top \theta$. 

- 决策的可行域 $\mathcal{W}$ 是 closed and bounded, 故存在一个常数 $B_\mathcal{W} > 0$, 使得 $\|\mathbf{w}\|_2 \leq B_\mathcal{W}$ 对于任意 $\mathbf{w} \in \mathcal{W}$ 都成立. 给定 context $\mathbf{x}$, 真实 cost $\mathbf{c}$ bounded: 存在 $K$, 使得 $\|\mathbf{c}\|_2 \leq K$ 对于任意 $(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}$ 都成立. 从而有:
    $$
    \langle \mathbf{c}, \mathbf{w} \rangle \leq K B_\mathcal{W}, \quad \forall (\mathbf{x}, \mathbf{c}) \sim \mathcal{P}, \forall \mathbf{w} \in \mathcal{W}
    $$

- 当 $\theta = 0$ 时, $\mathbf{\hat{c}}_\theta(\mathbf{x}) = 0$ 对于任意 $\mathbf{x}$ (这个很基本的假设, 只是为了推导方便). 此外, 要求 $\mathbf{\hat{c}}_\theta(\mathbf{x})$ 的梯度  $\nabla_\theta \mathbf{\hat{c}}_\theta(\mathbf{x}) \in \mathbb{R}^{d \times m}$ 有上界 $B_\Phi$, 即
    $$
    \|\nabla_\theta \mathbf{\hat{c}}_\theta(\mathbf{x})\|_2 \leq B_\Phi, \quad \forall \mathbf{x} \in \mathbb{R}^k, \forall \theta \in \mathbb{R}^m
    $$
    此外要求梯度 $\nabla_\theta \mathbf{\hat{c}}_\theta(\mathbf{x})$ 是 piecewise continuous 的 (只在有限个点不连续) 以便后续的积分操作. 

    $\diamond$

    根据上述假设, 由积分放缩, 可以得到如下不等式:
    $$
    \|\mathbf{\hat{c}}_\theta(\mathbf{x}) \| \leq B_\Phi \|\theta\|_2, \quad \forall \mathbf{x} \in \mathbb{R}^k, \forall \theta \in \mathbb{R}^m
    $$
    这说明 predictive cost $\mathbf{\hat{c}}_\theta(\mathbf{x})$ 的大小可以被 $\|\theta\|_2$ 控制.


由此可以得到如下的泛化性定理. 

***Theorem 2*** 令 $\beta \geq \beta^\star_{\mathcal{H}, \mathcal{P}}$. 在根据历史数据 $S = \{(\mathbf{x}_i, \mathbf{c}_i)\}_{i=1}^n$ 形成的经验分布 $\mathcal{P}_n$ 下, 取某个 $\theta^\star \in \mathbb{R}^m$ 为 empirical CILO loss 足够小的 minimizer, 即 $\ell_{\mathcal{P}_n}^\beta(\theta^\star) \leq \varepsilon$. 接着假设 $\theta^\star$ 同样可以被一个正常数 $D$ 控制, 即  $\|\theta^\star\|_2 \leq D$. 则在上述假设基础上, 以及 $\beta_{\mathcal{H}, \mathcal{P}}^\star > \beta_{\min, \mathcal{P}}$ 的前提下 (其中 RHS 是一个给定的 oracle 下界), 以概率至少 $1 - \delta$ , 有:
$$
\ell_\mathcal{P}^\beta(\theta^\star) \leq \underbrace{\varepsilon}_{\small\text{{empirical CILO}}} + \underbrace{\mathcal{O}\left( 
    \frac{1}{\beta-\beta_{\min, \mathcal{P}}} \cdot \sqrt{\frac{\log(1/\delta)}{n}}
\right)}_{\small\text{{generalization gap}}}
$$
其中 $\beta_{\min, \mathcal{P}} := \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}} \left( \min_{\mathbf{w} \in \mathcal{W}} \mathbf{c}^\top \mathbf{w} \right)$ 为理想的 oracle 水平, 即能够看到真实 cost $\mathbf{c}$ 后, 在 $\mathcal{W}$ 中做出的最优决策的 performance. 
- 注意区分 $\beta_{\min, \mathcal{P}}$ 与 $\beta^\star_{\mathcal{H}, \mathcal{P}}$, 前者是 oracle 下界, 后者是当前 hypothesis class $\mathcal{H}$ 能够达到的最优真实 decision loss. 故恒有 $\beta_{\min, \mathcal{P}} \leq \beta^\star_{\mathcal{H}, \mathcal{P}}$.
  - 若 $\mathcal{H}$ 是 well-specified 的, 则确实能够产生 oraclly optimal 的决策, 故 $\beta_{\min, \mathcal{P}} = \beta^\star_{\mathcal{H}, \mathcal{P}}$. 
  - 若 $\mathcal{H}$ 是 misspecified 的, 则 $\beta_{\min, \mathcal{P}} < \beta^\star_{\mathcal{H}, \mathcal{P}}$.
- 在表达式中, $1/(\beta - \beta_{\min, \mathcal{P}})$ 表明, 当 CILO 中规定的允许真实成本门槛 $\beta$ 越接近 oracle 下界 $\beta_{\min, \mathcal{P}}$, 则要求越严格, 表明当前只允许十分接近 oracle 的 policy subset, 这会导致泛化性很差, 反之亦然. 
- 不过反而天然的, 当模型是 misspecified 的时, $\beta^\star_{\mathcal{H}, \mathcal{P}} > \beta_{\min, \mathcal{P}}$, 因此 $\beta - \beta_{\min, \mathcal{P}} \geq \beta^\star_{\mathcal{H}, \mathcal{P}} - \beta_{\min, \mathcal{P}} > 0$, 因此泛化性是有保证的. 对应的, 当模型是 well-specified 的, 则需要适当调大 $\beta$ 来保证泛化性.


### Stronger Consistency Guarantee

*Theorem 1* 给出了 $\ell_\mathcal{P}^\beta = 0 \implies \ell_\mathcal{P} \leq \beta$ 的 guarantee, 然而在实践中往往无法严格保证 $\ell_\mathcal{P}^\beta = 0$, 因此需要更强的 guarantee. 

假设: 
1. 只要 $\theta \neq 0$ a.s., 则 $\mathbf{\hat{c}}_\theta(\mathbf{x}) \neq 0$ a.s. 成立. 这结合之前的 assumption 保证 $\min_{\mathbf{w} \in \mathcal{W}} \mathbf{\hat{c}}_\theta(\mathbf{x})^\top \mathbf{w}$ 的唯一性成立. 
2. $\mathcal{W}$ 是一个 polyhedron, 记其所有 extreme points 为 $\mathcal{W}_\text{ext}$. 由于目标 $\mathbf{\hat{c}}_\theta(\mathbf{x})^\top \mathbf{w}$ 是线性的, 因此最优点 $\mathbf{w}^\star$ 必然是 $\mathcal{W}_\text{ext}$ 中取到. 因此考虑如下差值以衡量解的稳定性: 当 $\mathcal{W}^\star(\mathbf{\hat{c}}_\theta(\mathbf{x})) \neq \mathcal{W}_\text{ext}$ 时,
    $$
    \Delta_\theta(\mathbf{x}) := \min_{\mathbf{w} \in \mathcal{W}_\text{ext} \setminus \mathcal{W}^\star(\mathbf{\hat{c}}_\theta(\mathbf{x}))} \mathbf{\hat{c}}_\theta(\mathbf{x})^\top \mathbf{w} 
    - 
    \min_{\mathbf{w} \in \mathcal{W}_\text{ext}} \mathbf{\hat{c}}_\theta(\mathbf{x})^\top \mathbf{w}
    $$
    - 第一项表示在所有非最优 extreme points 中, 其在 $\mathbf{\hat{c}}_\theta(\mathbf{x})$ 下的最小值. 第二项表示在所有 extreme points 中, 其在 $\mathbf{\hat{c}}_\theta(\mathbf{x})$ 下的最小值. 因此 $\Delta_\theta(\mathbf{x})$ 表示最优点与次优点之间的 gap. 这个 gap 越大, 则 $\mathbf{\hat{c}}_\theta(\mathbf{x})$ 的唯一性越强, 也就越不容易被噪声干扰. 
    - 当 $\mathcal{W}^\star(\mathbf{\hat{c}}_\theta(\mathbf{x})) = \mathcal{W}_\text{ext}$ 时, 定义 $\Delta_\theta(\mathbf{x}) = 0$. 

    且假设, 存在常数 $\alpha > 0$, $\gamma \geq 0$, 使得对于任意 $\theta \in \mathbb{R}^m$, 有
    $$
    \mathbb{P} \left( 0 < \Delta_\theta(\mathbf{x}) \leq \|\theta\| t \right) \leq \left(\frac{\gamma t}{B_{\mathcal{W}}}\right)^\alpha, \quad \forall t > 0
    $$
    即, 解不稳定的 gap 只以 $\mathcal{O}(t^\alpha)$ 的小概率出现, 通常的解的 gap 都是稳定的, 这保证了 $\mathbf{\hat{c}}_\theta(\mathbf{x})$ 的唯一性.


在上述假设下, 有如下定理:
***Theorem 3*** 存在 $\alpha >0$, 使得对于任意 $\mathbf{0} \in \mathbb{R}^m$, 任意 $\theta \in \mathbb{R}^m$ 且不为零, 以及 $\beta \geq \beta^\star_{\mathcal{H}, \mathcal{P}}$, 有
$$
\ell_\mathcal{P}(\theta) \leq \beta + \mathcal{O}\left( \ell_\mathcal{P}^\beta(\theta)^{\frac{\alpha}{1+\alpha}} \right)
$$

- 即使 $\ell_\mathcal{P}^\beta(\theta)$ 不是严格为零, 也可以保证 $\ell_\mathcal{P}(\theta)$ 不会偏离 $\beta$ 太远, 其以 $\mathcal{O}\left( \ell_\mathcal{P}^\beta(\theta)^{\frac{\alpha}{1+\alpha}} \right)$ 的速度收敛到 $\beta$. 

### Optimizing the Surrogate

根据上面的推导, 只需要找到一个 $\theta \neq 0$ 且有界, 使得 empirical surrogate $\ell_{\mathcal{P}_n}^\beta(\theta)$ 足够小, 即可保证 target loss $\ell_\mathcal{P}(\theta)$ 足够小.  故这一小节的目标即为优化该经验 CILO.

对于任意 $\theta \in \mathbb{R}^m$, 其 empirical CILO loss 可以拆分如下 (纯粹代数整理):
$$
\ell_{\mathcal{P}_n}^\beta(\theta) = g_{\mathcal{P}_n} (\theta) - \bar{g}^\beta_{\mathcal{P}_n} (\theta)
$$
其中
$$
g_{\mathcal{P}_n} (\theta) :=  - \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}_n} \left[ \min_{\mathbf{w} \in \mathcal{W}} \mathbf{\hat{c}}_\theta(\mathbf{x})^\top \mathbf{w} \right] = -\min_{\mathbf{w}_{\mathcal{P}_n} \in \mathcal{W}_{\mathcal{P}_n}} \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}_n} \left[ \mathbf{\hat{c}}_\theta(\mathbf{x})^\top {w}_{\mathcal{P}_n}(\mathbf{x}) \right]
$$
$$
\bar{g}^\beta_{\mathcal{P}_n} (\theta) := - \min_{\bar{w}_{\mathcal{P}_n} \in \mathcal{\bar{W}}_{\mathcal{P}_n}^\beta} \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}_n} \left[ \mathbf{\hat{c}}_\theta(\mathbf{x})^\top \bar{w}_{\mathcal{P}_n}^\beta(\mathbf{x}) \right]
$$

假设 $\mathbf{\hat{c}}_\theta(\mathbf{x})$ 是一个连续函数, 且 $\nabla_\theta \mathbf{\hat{c}}_\theta(\mathbf{x})$ 是 $B_L > 0$ Lipschitz continuous 的, 则可以证明这两个函数都是 weakly convex 的, 因此 $\ell_{\mathcal{P}_n}^\beta(\theta)$ 是两个 weakly convex 函数的差, 即 DC function. 

这里又重新进行了一次 rename: $g^1_{\mathcal{P}_n}(\theta) := g_{\mathcal{P}_n}(\theta)$, $g^2_{\mathcal{P}_n}(\theta) := \bar{g}^\beta_{\mathcal{P}_n}(\theta)$, 则 $\ell_{\mathcal{P}_n}^\beta(\theta) = g^1_{\mathcal{P}_n}(\theta) - g^2_{\mathcal{P}_n}(\theta)$. 此时对每一个股份进行 Moreau Smoothing:
$$
M_{\mathcal{P}_n}^i(\theta) := \min_{\theta \in \mathbb{R}^m} \left\{ g^i_{\mathcal{P}_n}(\theta) + \frac{1}{2} \|\lambda - \theta\|_2^2 \right\}, \quad i = 1, 2
$$
以及
$$
\theta_{\mathcal{P}_n}^i (\lambda) := \arg\min_{\theta \in \mathbb{R}^m} \left\{ g^i_{\mathcal{P}_n}(\theta) + \frac{1}{2} \|\lambda - \theta\|_2^2 \right\}, \quad i = 1, 2
$$

然后就可以定义 s-CILO (smoothed CILO) loss:
$$
r_{\mathcal{P}_n}^\beta(\lambda) := M_{\mathcal{P}_n}^1(\lambda) - M_{\mathcal{P}_n}^2(\lambda)
$$
根据 gradient: 
$$
\nabla r_{\mathcal{P}_n}^\beta(\lambda) =  \theta_{\mathcal{P}_n}^2 (\lambda) - \theta_{\mathcal{P}_n}^1 (\lambda)
$$
因此若 $\lambda$ 是 s-CILO 的 stationary point, 则  $\theta_{\mathcal{P}_n}^1 (\lambda) = \theta_{\mathcal{P}_n}^2 (\lambda)$. 且这个共同的 $\theta$ 也是原始 empirical CILO $\ell_{\mathcal{P}_n}^\beta(\theta)$ 的 stationary point. 因此可以通过优化 s-CILO 来间接优化原始的 empirical CILO.

下讨论如何避免出现 $\theta = 0$ 的情况. 定义如下的 log-CILO loss:
$$
f_{\mathcal{P}_n}^\beta(\lambda) := \log d(\lambda, - \bar{V}_{\mathcal{P}_n}^\beta) - \log d(\lambda, V_{\mathcal{P}_n})
$$
其中
$$
V_{\mathcal{P}_n} = \{ \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}_n} \left[ \Phi(\mathbf{x})^\top {w}_{\mathcal{P}_n}(\mathbf{x}) \right] : \mathbf{w}_{\mathcal{P}_n} \in \mathcal{W}_{\mathcal{P}_n} \}
$$
$$
\bar{V}_{\mathcal{P}_n}^\beta = \{ \mathbb{E}_{(\mathbf{x}, \mathbf{c}) \sim \mathcal{P}_n} \left[ \Phi(\mathbf{x})^\top \bar{w}_{\mathcal{P}_n}^\beta(\mathbf{x}) \right] : \bar{w}_{\mathcal{P}_n}^\beta \in \mathcal{\bar{W}}_{\mathcal{P}_n}^\beta \}
$$
前者对应所有 policies, 后者对应所有在真实 cost 下 performance 不超过 $\beta$ 的 policies. $d$ 是欧氏距离. 

恒有 $f_{\mathcal{P}_n}^\beta(\lambda) \geq 0$, 当且仅当 $d(\lambda, - \bar{V}_{\mathcal{P}_n}^\beta) = d(\lambda, V_{\mathcal{P}_n})$, 即 $\lambda$ 到两个集合的距离相等时, $f_{\mathcal{P}_n}^\beta(\lambda) = 0$. 这与 s-CILO 的 stationary point 是等价的. 
总结下来, log-CILO 继承了 s-CILO 的所有性质, 且避免了 $\theta = 0$ 的情况. 因此可以通过优化 log-CILO 来间接优化原始的 empirical CILO, 并且保证 $\theta \neq 0$.

