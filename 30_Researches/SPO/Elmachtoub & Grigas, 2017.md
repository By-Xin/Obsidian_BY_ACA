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
P(\mathbf{c}): \quad \mathbf{z}^\star(\mathbf{c}) := \min_{\mathbf{w}} \mathbf{c}^\top \mathbf{w}, \quad \text{s.t.}~ \mathbf{w} \in \mathcal{S}
$$

- Nominal 是指 $\mathbf{c}$ 已经被观测到的情况下的理想情况; Downstream 是指 $\mathbf{c}$ 是下游优化问题的输入.
- 这里 $\mathbf{z}^\star(\mathbf{c})$ 是一个 oracle 决策, 即在 $\mathbf{c}$ 已经被观测到的情况下的最优决策. 
- 这里要求 $\mathcal{S}$ 是 non-empty, convex, compact (closed and bounded). 此外, 还假设 $\mathcal{S}$ 是给定且已知的.
  - Compact 是为了满足目标值有限, 并且最优值是可达的. 
  - Convex 的假设实际上并不失一般性. 对于任意 non-convex 或 non-closed 的 $\mathcal{\tilde{S}}$, 我们总可以用其 closed convex hull $\mathcal{S} := \overline{\text{conv}}(\mathcal{\tilde{S}})$ 来替代. 在 LP 问题上, 可以证明二者的最优值是一样的, 即:
    $$
    \mathbf{z}^\star(\mathbf{c}) = \min_{\mathbf{w} \in \mathcal{\tilde{S}}} \mathbf{c}^\top \mathbf{w} = \min_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w}
    $$
  - 额外的关于 $\mathcal{S}$ 的给定且已知的假设, 是为了将随机性剔除在决策之外, 也就是每个 instance 之间的差异只存在于 $\mathbf{c}$ 的差异, 而不是 $\mathcal{S}$ 的差异. 因此每个 instance 本身也就可以 1-1 地对应到一个 $\mathbf{c}$ 上. 
- 注意, 这里虽然优化的变量是 $\mathbf{w}$, 但对于这个优化问题而言, 其是一个关于 $\mathbf{c}$ 的函数. 我们认为, 每给定一个 $\mathbf{c}$, 我们总会给出这个 specific $\mathbf{c}$ 下的最优决策. 因此我们并不关注具体的每个 LP 的求解细节, 而是关注 $\mathbf{c}$ 的变化如何影响 $\mathbf{z}^\star(\mathbf{c})$ 的变化. 这也是为什么我们后续的训练目标为 $\ell(\hat{\mathbf{c}}, \mathbf{c})$. 
- 对应地, 我们将这个 Oracle 的 Nominal Optimization 问题的最优解, 称为 Oracle 决策, 记为:
    $$
    \mathbf{w}^\star(\mathbf{c}) \in \arg\min_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w} := W^\star(\mathbf{c})
    $$

-  在实做中, 若 $\mathbf{c}$ 无法直接观测到, 那么就是用 **plug-in** 的方式, 实际的求解
  $$
  P(\hat{\mathbf{c}}): \quad \mathbf{z}^\star(\hat{\mathbf{c}}) := \min_{\mathbf{w}} \hat{\mathbf{c}}^\top \mathbf{w}, \quad \text{s.t.}~ \mathbf{w} \in \mathcal{S}
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
- 下面 claim: $\xi_\mathcal{S}(\mathbf{c}) = - \mathbf{z}^\star(-\mathbf{c})$. 这是因为:
  $$
  \begin{aligned}
  \mathbf{z}^\star(\mathbf{c}) & = \min_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w} \\
  \implies \mathbf{z}^\star(-\mathbf{c}) & = \min_{\mathbf{w} \in \mathcal{S}} (-\mathbf{c})^\top \mathbf{w} = - \max_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w} = - \xi_\mathcal{S}(\mathbf{c})
  \end{aligned}
  $$
- 接着有 $\xi_\mathcal{S}(\mathbf{c}) = \mathbf{c}^\top \mathbf{w}^\star(-\mathbf{c})$.
  - 这是因为, $\mathbf{w}^\star(-\mathbf{c})$ 是 $\min_{\mathbf{w} \in \mathcal{S}} (-\mathbf{c})^\top \mathbf{w}$ 的一个最优解, 因此也是 $\max_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w}$ 的一个最优解.
  - 因此带入到 $-\xi_\mathcal{S}(\mathbf{c}) =  - \max_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w}$ 中, 就可以得到 $\xi_\mathcal{S}(\mathbf{c}) = \mathbf{c}^\top \mathbf{w}^\star(-\mathbf{c})$.
- 接着可以证明: $\xi_\mathcal{S}(\mathbf{c})$ 是一个 convex function. 
  - 因为这是很多 affine function 的 pointwise maximum, 因此是 convex 的.
  - 由于在后面的分析中, 有分解如下, 因此我们关注 $\xi_\mathcal{S}$ 的 convexity.
    $$
    \ell_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c}) = \xi_\mathcal{S}(\hat{\mathbf{c}} - \mathbf{c}) + 2 \hat{\mathbf{c}}^\top \mathbf{w}^\star(\mathbf{c}) - \mathbf{z}^\star(\mathbf{c})
    $$


## SPO Loss Function

