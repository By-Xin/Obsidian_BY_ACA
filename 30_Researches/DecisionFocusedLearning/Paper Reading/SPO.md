# Smart "Predict, then Optimize"

## Introduction 

核心关切在于: 先用 ML 根据特征 $\mathbf{x} \in \mathcal{X} \subseteq \mathbb{R}^p$ 预测一个优化问题的参数 $\mathbf{c} \in \mathcal{C} \subseteq \mathbb{R}^d$, 然后基于这个预测 $\hat{\mathbf{c}} = f_\theta(\mathbf{x})$ 来求解一个优化问题, 以此来做出决策 $\mathbf{w} \in \mathcal{S} \subseteq \mathbb{R}^d$.  

传统的预测-决策的分离观点下, 模型预测的核心是最终的预测准确性. 然而这假设了所有参数都是同质性的. 然而事实往往并不是如此, 有一些分量即使有很大的改变, 也不会改变最终的决策, 反之亦然. 

##  统计学习理论框架

本文考虑一个 contextual stochastic optimization 的问题. 并试图通过一个统计学习理论的框架来分析这个问题. 
- 考虑 $\mathbf{x}, \mathbf{c}$ 是一对随机变量. 当一个 instance $(\mathbf{x}, \mathbf{c})$ 发生后, 决策者在决策步骤能够观察到的特征是 $\mathbf{x} = x$, 但是无法直接观测到成本参数 $\mathbf{c}$. 此时会记这个给定 $\mathbf{x}$ 下的 $\mathbf{c}$ 的条件分布为 $\mathcal{D}_\mathbf{x}$. 
- 这就是 contextual stochastic optimization 的含义, 决策不是针对 $\mathbf{c}$ 的无条件分布, 而是针对 $\mathbf{x}$ 条件下的 $\mathcal{D}_\mathbf{x}$ 的分布.

因此对应的目标问题, 即 Contextual Stochastic Optimization 问题:
$$
\min_{\mathbf{w} \in \mathcal{S}} \mathbb{E}_{\mathbf{c} \sim \mathcal{D}_\mathbf{x}} [\mathbf{c}^\top \mathbf{w} \mid \mathbf{x}] = \min_{\mathbf{w} \in \mathcal{S}} \mathbb{E}_{\mathbf{c} \sim \mathcal{D}_\mathbf{x}} [\mathbf{c} \mid \mathbf{x}]^\top \mathbf{w}
$$
- 此处的等式成立只要求 $\mathbf{c}$ 的条件期望存在和目标函数的线性关系给出. 此时, 若记 $\mu(\mathbf{x}) = \mathbb{E}_{\mathbf{c} \sim \mathcal{D}_\mathbf{x}} [\mathbf{c} \mid \mathbf{x}]$, 则问题等价于 $\min_{\mathbf{w} \in \mathcal{S}} \mu(\mathbf{x})^\top \mathbf{w}$. 因此, 在当前模型设定 (线性) 下, 条件分布 $\mathcal{D}_\mathbf{x}$ 只通过条件均值 $\mu(\mathbf{x})$ 来影响决策. 这也意味着, $\mu(\mathbf{x})$ 是关于最优决策的一个充分统计量.

下面提出 Nominal (Downstream) Optimization 问题, 即在 $\mathbf{c}$ 已经被观测到的情况下的确定性优化问题:
$$
P(\mathbf{c}): \quad z^\star(\mathbf{c}) := \min_{\mathbf{w}} \mathbf{c}^\top \mathbf{w}, \quad \text{s.t.}~ \mathbf{w} \in \mathcal{S}
$$

- Nominal 是指 $\mathbf{c}$ 已经被观测到的情况下的理想情况; Downstream 是指 $\mathbf{c}$ 是下游优化问题的输入.
- 这里 $z^\star(\mathbf{c})$ 是一个 oracle 决策, 即在 $\mathbf{c}$ 已经被观测到的情况下的最优决策. 
- 这里要求 $\mathcal{S}$ 是 non-empty, convex, compact (closed and bounded). 此外, 还假设 $\mathcal{S}$ 是给定且已知的.
  - Compact 是为了满足目标值有限, 并且最优值是可达的. 
  - Convex 的假设实际上并不失一般性. 对于任意 non-convex 或 non-closed 的 $\mathcal{\tilde{S}}$, 我们总可以用其 closed convex hull $\mathcal{S} := \overline{\text{conv}}(\mathcal{\tilde{S}})$ 来替代. 在 LP 问题上, 可以证明二者的最优值是一样的, 即:
    $$
    z^\star(\mathbf{c}) = \min_{\mathbf{w} \in \mathcal{\tilde{S}}} \mathbf{c}^\top \mathbf{w} = \min_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w}
    $$
  - 额外的关于 $\mathcal{S}$ 的给定且已知的假设, 是为了将随机性剔除在决策之外, 也就是每个 instance 之间的差异只存在于 $\mathbf{c}$ 的差异, 而不是 $\mathcal{S}$ 的差异. 因此每个 instance 本身也就可以 1-1 地对应到一个 $\mathbf{c}$ 上. 
- 注意, 这里虽然优化的变量是 $\mathbf{w}$, 但对于这个优化问题而言, 其是一个关于 $\mathbf{c}$ 的函数. 我们认为, 每给定一个 $\mathbf{c}$, 我们总会给出这个 specific $\mathbf{c}$ 下的最优决策. 因此我们并不关注具体的每个 LP 的求解细节, 而是关注 $\mathbf{c}$ 的变化如何影响 $z^\star(\mathbf{c})$ 的变化. 这也是为什么我们后续的训练目标为 $\ell(\hat{\mathbf{c}}, \mathbf{c})$. 
- 对应地, 我们将这个 Oracle 的 Nominal Optimization 问题的最优解, 称为 Oracle 决策, 记为:
    $$
    \mathbf{w}^\star(\mathbf{c}) \in \arg\min_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w} := W^\star(\mathbf{c})
    $$

-  在实做中, 若 $\mathbf{c}$ 无法直接观测到, 那么就是用 **plug-in** 的方式, 实际的求解
  $$
  P(\hat{\mathbf{c}}): \quad z^\star(\hat{\mathbf{c}}) := \min_{\mathbf{w}} \hat{\mathbf{c}}^\top \mathbf{w}, \quad \text{s.t.}~ \mathbf{w} \in \mathcal{S}
  $$


下面考虑具体的训练过程. 
- 在训练中, 考虑如下训练数据 $\{(\mathbf{x}_i, \mathbf{c}_i)\}_{i=1}^n$, 其中 $\mathbf{x}_i$ 是特征 / context, $\mathbf{c}_i$ 是对应的成本参数 (并且暗含假设其是独立同分布的).
- 按照 Learning theory 的做法, 这里考虑的 hypothesis class 是 
  $$
  \mathcal{H} = \{f_\theta: \mathcal{X} \to \mathcal{C} \subseteq \mathbb{R}^d \mid \theta \in \Theta\},
  $$
   其中 $\Theta$ 是参数空间. 
  - 特别地, 在下文中, 一个常见的假设是 $\mathcal{H}$ 是一个线性模型类, 即 $f_\theta(\mathbf{x}) = \mathbf{B} \mathbf{x}$, 其中 $\theta = \mathbf{B} \in \mathbb{R}^{d \times p}$ 是一个矩阵.
- 定义损失函数 $\ell: \mathcal{C} \times \mathcal{C} \to \mathbb{R}_+$, 其衡量了预测 $\hat{\mathbf{c}}$ 和真实 $\mathbf{c}$ 之间的差异. 其实本质上, 本文的核心创新点就是对这个损失函数的定义. 

明确了训练的数据和模型假设之后, 下面就可以定义训练的目标了. 其 Empirical Risk Minimization 的目标为:
$$
\mathcal{\widehat{R}}_n (f)=\min_{f \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n \ell(f(\mathbf{x}_i), \mathbf{c}_i)
$$
得到 prediction model $f^\star(\mathbf{x})$ 之后, 就可以用 plug-in 的方式来求解对应的 optimization problem, 从而得到决策规则:
$$
\delta_{f^\star}(\mathbf{x}) := \mathbf{w}^\star(f^\star(\mathbf{x})) 
$$
- 这表示, 给定一个特征 $\mathbf{x}$, 先用 $f^\star$ 来预测 $\hat{\mathbf{c}} = f^\star(\mathbf{x})$, 然后再用 plug-in 的方式来求解对应的 optimization problem, 从而得到决策 $\delta_{f^\star}(\mathbf{x})$.

此外, 为了后续分析方便, 这里引入 $\mathcal{S}$ 的 support function $\xi_\mathcal{S}: \mathbb{R}^d \to \mathbb{R}$, 定义如下:
$$
\xi_\mathcal{S}(\mathbf{c}) := \max_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w}, \quad \forall \mathbf{c} \in \mathbb{R}^d.
$$
- 这是一个凸分析中比较常用的工具. 表示沿着方向 $\mathbf{c}$ 看, 集合 $\mathcal{S}$ 上的所有向量 $\mathbf{w}$ 在 $\mathbf{c}$ 方向上的投影的最大值. (这个支撑函数就是 supporting hyperplane 的那个函数)
- 根据凸分析的性质, 由于 $\mathcal{S}$ 是compact 的, 因此 $\xi_\mathcal{S}(\cdot)$ 点点有限, 并且这个定义中的最大值对于任意 $\mathbf{c}$ 都是可取到的.
- 下面 claim: $\xi_\mathcal{S}(\mathbf{c}) = - z^\star(-\mathbf{c})$. 这是因为:
  $$
  \begin{aligned}
  z^\star(\mathbf{c}) & = \min_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w} \\
  \implies z^\star(-\mathbf{c}) & = \min_{\mathbf{w} \in \mathcal{S}} (-\mathbf{c})^\top \mathbf{w} = - \max_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w} = - \xi_\mathcal{S}(\mathbf{c})
  \end{aligned}
  $$
- 接着有 $\xi_\mathcal{S}(\mathbf{c}) = \mathbf{c}^\top \mathbf{w}^\star(-\mathbf{c})$.
  - 这是因为, $\mathbf{w}^\star(-\mathbf{c})$ 是 $\min_{\mathbf{w} \in \mathcal{S}} (-\mathbf{c})^\top \mathbf{w}$ 的一个最优解, 因此也是 $\max_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w}$ 的一个最优解.
  - 因此带入到 $-\xi_\mathcal{S}(\mathbf{c}) =  - \max_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w}$ 中, 就可以得到 $\xi_\mathcal{S}(\mathbf{c}) = \mathbf{c}^\top \mathbf{w}^\star(-\mathbf{c})$.
- 接着可以证明: $\xi_\mathcal{S}(\mathbf{c})$ 是一个 convex function. 
  - 因为这是很多 affine function 的 pointwise maximum, 因此是 convex 的.
  - 由于在后面的分析中, 有分解如下, 因此我们关注 $\xi_\mathcal{S}$ 的 convexity.
    $$
    \ell_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c}) = \xi_\mathcal{S}(\hat{\mathbf{c}} - \mathbf{c}) + 2 \hat{\mathbf{c}}^\top \mathbf{w}^\star(\mathbf{c}) - z^\star(\mathbf{c})
    $$
- 对于凸函数 $\xi_\mathcal{S}(\cdot)$, 其 subgradient 的定义为:
  $$
  \partial \xi_\mathcal{S}(\mathbf{c}) := \{\mathbf{g} \in \mathbb{R}^d \mid \xi_\mathcal{S}(\mathbf{c}') \geq \xi_\mathcal{S}(\mathbf{c}) + \mathbf{g}^\top (\mathbf{c}' - \mathbf{c}), \forall \mathbf{c}' \in \mathbb{R}^d\}
  $$
  并且自然有性质, 如果 $\bar{\mathbf{w}} \in \arg\max_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w}$, 则 $\bar{\mathbf{w}} \in \partial \xi_\mathcal{S}(\mathbf{c})$. 

## SPO Loss Function

### True SPO Loss

考虑最本质的 SPO loss 的定义. 
- 给定 feature 与 cost 的训练数据 $\{(\mathbf{x}_i, \mathbf{c}_i)\}_{i=1}^n$, 以及一个 prediction model $f_\theta: \mathcal{X} \to \mathcal{C}$, 首先可以训练出一个预测 $\hat{\mathbf{c}}_i = f_\theta(\mathbf{x}_i)$.
- 然后根据预测出的 cost vector $\hat{\mathbf{c}}_i$, plug in 到下游优化问题中:
  $$
   z^\star({\mathbf{c}}) := \min_{\mathbf{w} \in \mathcal{S}} {\mathbf{c}}^\top \mathbf{w} \tag{P(c)}
  $$
  得到
  $$
  \mathbf{w}^\star(\hat{\mathbf{c}}) = \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}
  $$

- 因此 $\mathbf{w}^\star(\hat{\mathbf{c}})$ 是在预测的 cost vector $\hat{\mathbf{c}}$ 下的最优决策, 而 $\mathbf{c}$ 是要支付的真实 cost. 因此可以定义 SPO loss 为在真实成本下, 用户能够做出的最优决策与实际全局最优决策之间的差异, namely regret:
  $$
  \ell^{w^*}_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c}) := \mathbf{c}^\top \mathbf{w}^\star(\hat{\mathbf{c}}) - z^\star(\mathbf{c}) = \mathbf{c}^\top \mathbf{w}^\star(\hat{\mathbf{c}}) - \min_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w} \geq 0 .
  $$
  - 然而这里的一个问题是, $\mathbf{w}^*(\hat{\mathbf{c}})$ 是 $\arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}$ 的一个解, 可能不是唯一的, 记这个最优解集为 $W^\star(\hat{\mathbf{c}})$. 这些解在 $\hat{\mathbf{c}}$ 下都是最优的, 但是在真实的 $\mathbf{c}$ 下却有可能是各不相同的. 

    -  为避免歧义. 往往会考虑 worst-case 的情况, 此时定义 SPO loss 为:
      $$
      \ell_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c}) := \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w} - z^\star(\mathbf{c}) .
      $$
     - 文中也指出, 这个多解问题 practically 通常只会在 degenerate 的情况下才会发生, 因此在实际中通常不会有太大影响.


> **Figure 1**: 下图是 SPO 的一个示例. 
> ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260409143223.png)
> - 左图中, $\mathcal{S}$ 是一个多边形区域. 上方的 $\mathbf{c}$ 是真实的最优 cost 向量. 上方顶点即为真实 cost $\mathbf{c}$ 下的 Oracle 决策 $\mathbf{w}^\star(\mathbf{c})$. 阴影区域表示这个顶点的 norm cone, 即如果优化得到的 $\hat{\mathbf{c}}$ (的反向) 落在这个阴影区域内, 那么就可以保证 $\mathbf{w}^\star(\hat{\mathbf{c}}) = \mathbf{w}^\star(\mathbf{c})$, 从而保证 $\ell_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c}) = 0$. 在这里, $\hat{\mathbf{c}}_A$ 即为一个符合要求的预测, 而 $\hat{\mathbf{c}}_B$ 则不符合要求. 此外, 观察到 $\hat{\mathbf{c}}_A, \hat{\mathbf{c}}_B$ 共圆, 且圆心为 $\mathbf{c}$, 这就说明在传统的 MSE loss 下, 二者距离 $\mathbf{c}$ 是一样的, 但是在 SPO loss 下, 二者的损失却是完全不同的.
> - 右图中, $\mathcal{S}$ 是一个椭圆. 这也说明, SPO 的适用场景除了 LP, 也可以是在一些二次规划等场景下, 例如 Markowitz portfolio optimization 问题. 在椭圆可行域下, 发现之前的 norm cone 退化成了一条直线, 因此只有当 $\hat{\mathbf{c}}$ 落在这条直线上 (与 $\mathbf{c}$ 共线) 时, 才能保证 $\ell_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c}) = 0$. 因此在这个场景下, 预测 $\hat{\mathbf{c}}$ 的方向比距离更重要. 并且变得更为敏感. 换言之, 问题可行域的几何结构, 会对 SPO loss 的敏感性产生重要影响.

文中进一步说明, 事实上 $0-1$ 的 binary classification loss 可以看作是一个特殊的 SPO loss, 其对应下游优化问题的 feasible set $\mathcal{S} = [-1/2, 1/2]$, 且 $\mathbf{c} \in \{-1, 1\}$.


按照 ERM 的做法, 我们要寻找一个最优的预测模型 $f_\theta$ 来最小化 empirical risk:
$$
\min_{f_\theta \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n \ell_\text{SPO}(f_\theta(\mathbf{x}_i), \mathbf{c}_i) = \min_{f_\theta \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n [\mathbf{c}_i^\top \mathbf{w}^\star(f_\theta(\mathbf{x}_i)) - z^\star(\mathbf{c}_i)].
$$
然而, 优化这样的 ERM 往往是非常困难的. 一个直观观察若给定成本 $\mathbf{c}$, 求解 $\mathbf{w}^*(\mathbf{c}) \in \arg\min_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w}$ 可能是不连续的. 这随着预测 $\hat{\mathbf{c}}$ 的变化可能会发生跳变 (类比分类问题). 事实上, 由于分类问题的 NP-hardness, 一般求解 SPO 也会是十分困难的.


### SPO+ Surrogate

由于前面 SPO loss
$$
\ell_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c}) := \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w} - z^\star(\mathbf{c}), \quad W^\star(\hat{\mathbf{c}}) = \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}
$$
的求解困难, 因此这里的目标是构建一个 SPO 的 surrogate. 


- 首先进行一个等价改写. 对于任意 $\alpha \in \mathbb{R}$, 有
  $$
  \ell_{\text{SPO}}(\hat{\mathbf{c}}, \mathbf{c}) = \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \{\mathbf{c}^\top \mathbf{w} ~\underline{-~ \alpha \hat{\mathbf{c}}^\top \mathbf{w}}~\} \underline{+ \alpha z^\star(\hat{\mathbf{c}})} - z^\star(\mathbf{c}) 
  $$
  - 其中, 根据定义, $z^\star(\hat{\mathbf{c}}) = \min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}$, 而 $\mathbf{w} \in W^\star(\hat{\mathbf{c}})$, 因此 $\hat{\mathbf{c}}^\top \mathbf{w} = z^\star(\hat{\mathbf{c}})$. 因此上式中的下划线部分为 0.

- 接着, 注意到
  $$
  W^\star(\hat{\mathbf{c}}) = \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w} \subseteq \mathcal{S}
  $$
  因此
  $$
  \ell_{\text{SPO}}(\hat{\mathbf{c}}, \mathbf{c}) =
  \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) - z^\star(\mathbf{c}) \leq \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) - z^\star(\mathbf{c})
  $$
  再对上式 RHS 取关于 $\alpha$ 的 infimum, 则有
  $$
  \ell_{\text{SPO}}(\hat{\mathbf{c}}, \mathbf{c}) \leq \inf_{\alpha \in \mathbb{R}} \left\{ \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) \right\} - z^\star(\mathbf{c}) 
  $$
  
下面我们就要对
$$
\inf_{\alpha \in \mathbb{R}} \left\{ \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) \right\} - z^\star(\mathbf{c}) 
$$
这个 upper bound 进行细致讨论. 

***Proposition* 2** 当 $\alpha \to \infty$ 时, 该上界趋于 $\ell_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c})$:
$$
\lim_{\alpha \to \infty} \left\{ \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}})\right\} - z^\star(\mathbf{c})  = \ell_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c})
$$


在该 Proposition 2 的基础上, 用上述上界的极限形式来替代 $\ell_\text{SPO}$, 则有
$$
\begin{aligned}
\min_{ f \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n \ell_\text{SPO}(f(\mathbf{x}_i), \mathbf{c}_i) &\stackrel{(1)}{=} \min_{f \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n \lim_{\alpha_i \to \infty} \left\{ \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}_i^\top \mathbf{w} - \alpha_i f(\mathbf{x}_i)^\top \mathbf{w}\} + \alpha_i z^\star(f(\mathbf{x}_i))\right\} - z^\star(\mathbf{c}_i) \\
&\stackrel{\text{(2)}}{=} \min_{f \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n \lim_{\alpha_i \to \infty} \left\{ \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}_i^\top \mathbf{w} - \alpha_i f(\mathbf{x}_i)^\top \mathbf{w}\} + \alpha_i f(\mathbf{x}_i)^\top \mathbf{w}^*(\alpha_i f(\mathbf{x}_i))\right\} - z^\star(\mathbf{c}_i)  \\
&\stackrel{(3)}{=} \min_{f \in \mathcal{H}} \frac{1}{n} \lim_{\alpha \to \infty} \left\{\sum_{i=1}^n \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}_i^\top \mathbf{w} - \alpha f(\mathbf{x}_i)^\top \mathbf{w}\} + \alpha f(\mathbf{x}_i)^\top \mathbf{w}^*(\alpha f(\mathbf{x}_i))\right\} - z^\star(\mathbf{c}_i) \\
&\stackrel{\text{(4)}}{\leq} \min_{f \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}_i^\top \mathbf{w} - 2 f(\mathbf{x}_i)^\top \mathbf{w}\} + 2 f(\mathbf{x}_i)^\top \mathbf{w}^*(2 f(\mathbf{x}_i)) - z^\star(\mathbf{c}_i) \\
&\stackrel{\text{(5)}}{\leq} \min_{f \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}_i^\top \mathbf{w} - 2 f(\mathbf{x}_i)^\top \mathbf{w}\} + 2 f(\mathbf{x}_i)^\top \mathbf{w}^*(\mathbf{c}_i) - z^\star(\mathbf{c}_i) + 0.
\end{aligned}
$$

其中:
- (1) 直接代入可得. 注意到 $f(\mathbf{x}_i) = \hat{\mathbf{c}}_i$.
- (2) 的成立依赖于等式关系 $\alpha_i z^\star(f(\mathbf{x}_i)) = \alpha_i f(\mathbf{x}_i)^\top \mathbf{w}^*(\alpha_i f(\mathbf{x}_i))$ 或等价地 $\alpha_i z^\star(\hat{\mathbf{c}}_i) = \alpha_i \hat{\mathbf{c}}_i^\top \mathbf{w}^*(\alpha_i \hat{\mathbf{c}}_i)$. 分别回顾定义. $z^\star(\hat{\mathbf{c}}_i) = \min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}_i^\top \mathbf{w}$, 而 $\mathbf{w}^*(\alpha_i \hat{\mathbf{c}}_i) \in \arg\min_{\mathbf{w} \in \mathcal{S}} (\alpha_i \hat{\mathbf{c}}_i)^\top \mathbf{w}$. 因此显然左右二者相等. 
- (3) 由于 $\alpha_i$ 是对每个 instance 都不同的但都趋于 $\infty$, 因此统一为一个整体的 $\alpha \to \infty$.
- (4) 是由于上式求极限部分本身是关于 $\alpha$ 的单调递减函数, 故数学上成立. 之所以要进行这个放缩, 是因为 $\alpha \to \infty$ 的极限在实际中是无法实现的, 因此只能用一个有限的 $\alpha$ 来近似. 这里选择 $\alpha = 2$ 是为了方便后续 Bayes risk minimizer 分析上的简洁. 
- (5) 本质上在说 $2f(\mathbf{x}_i)^\top \mathbf{w}^*(2f(\mathbf{x}_i)) = 2\hat{\mathbf{c}}_i^\top \mathbf{w}^*(2\hat{\mathbf{c}}_i) \leq 2 \hat{\mathbf{c}}_i^\top \mathbf{w}^*(\mathbf{c}_i)$. 这是由于, 根据定义, LHS 中 $\mathbf{w}^*(2\hat{\mathbf{c}}_i) \in \arg\min_{\mathbf{w} \in \mathcal{S}} (2\hat{\mathbf{c}}_i)^\top \mathbf{w}$, 故 $2\hat{\mathbf{c}}_i^\top \mathbf{w}^*(2\hat{\mathbf{c}}_i) \leq 2\hat{\mathbf{c}}_i^\top \mathbf{w}$ 对任意 $\mathbf{w} \in \mathcal{S}$ 都成立, 因此也对 $\mathbf{w}^*(\mathbf{c}_i) \in \mathcal{S}$ 成立. 这就得到了 (5) 的不等式.

综上, 经过一系列的放缩, 我们得到了最终的 SPO+ surrogate loss:
$$
\begin{aligned}
\ell_\text{SPO+}(\hat{\mathbf{c}}, \mathbf{c}) &:= \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - 2 \hat{\mathbf{c}}^\top \mathbf{w}\} + 2 \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}) - z^\star(\mathbf{c}) 
\\ &= \max_{\mathbf{w} \in \mathcal{S}} \langle \mathbf{c} - 2 \hat{\mathbf{c}}, \mathbf{w} \rangle + 2 \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}) - z^\star(\mathbf{c})
\end{aligned}
$$
- $\max_{\mathbf{w} \in \mathcal{S}} \langle \mathbf{c} - 2 \hat{\mathbf{c}}, \mathbf{w} \rangle$ 表示 optimization oracle, 其仍然是在考虑下游 feasible set $\mathcal{S}$ 的情况下, 考虑实际成本与预测成本的差异.
- $2 \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c})$ 表示真实成本下的最优决策, 但是在预测成本下的权重. 
- $- z^\star(\mathbf{c})$ 表示真实成本下的最优值, 这是一个常数项, 因此在训练中可以忽略掉.

回顾, 若引入 support function $\xi_\mathcal{S}(\cdot) = \max_{\mathbf{w} \in \mathcal{S}} \langle \cdot, \mathbf{w} \rangle$, 则可以将 SPO+ surrogate loss 改写为:
$$
\ell_\text{SPO+}(\hat{\mathbf{c}}, \mathbf{c}) = \xi_\mathcal{S}(\mathbf{c} - 2 \hat{\mathbf{c}}) + 2 \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}) - z^\star(\mathbf{c})
$$
这在后文的一些凸性的分析中会更方便. 

