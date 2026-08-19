# Smart "Predict, then Optimize"

## Introduction 

考虑如下场景. 
- 在实际决策问题中, 定义决策变量 $\mathbf{w} \in \mathcal{S} \subseteq \mathbb{R}^d$, 其中假定 $\mathcal{S}$ 是一个 well-defined 且 known 的可行集合, 并往往假定具有 convex, compact 等良好性质 (当然对于更一般的问题也可以考虑其 convex hull). 
- 每个决策会对应一定成本 $\mathbf{c} \in \mathcal{C} \subseteq \mathbb{R}^d$, 其在做出决策时是无法直接观测到的, 故视作某种随机变量. 
- 对于这样的决策系统, 能够直接观察到的是一些特征 (或称上下文 context) $\mathbf{x} \in \mathcal{X} \subseteq \mathbb{R}^p$, 根据特征可以定义 $\mathbf{c}$ 服从的条件分布 $\mathcal{D}_\mathbf{x}$. 

因此, 总的而言, 我们考虑如下的线性 contextual stochastic optimization 问题:
$$
\min_{\mathbf{w} \in \mathcal{S}} \mathbb{E}_{\mathbf{c} \sim \mathcal{D}_\mathbf{x}} [\mathbf{c}^\top \mathbf{w} \mid \mathbf{x}] = \min_{\mathbf{w} \in \mathcal{S}} \mathbb{E}_{\mathbf{c} \sim \mathcal{D}_\mathbf{x}} [\mathbf{c} \mid \mathbf{x}]^\top \mathbf{w}
$$
即在给定特征 $\mathbf{x}$ 的条件下, 使得决策 $\mathbf{w}$ 的期望成本最小化. 并且在线性的假设下, 类比 OLS, $\mathbb{E}_{\mathbf{c} \sim \mathcal{D}_\mathbf{x}} [\mathbf{c} \mid \mathbf{x}]$ 事实上为一个关于最优决策的充分统计量. 并记之为 $\hat{\mathbf{c}}$. 

故传统的 Predict-then-Optimize (PtO) 方法如下. 
- 首先, 通过经典的统计学习方法, 根据特征 $\mathbf{x}$ 来预测 $\hat{\mathbf{c}}$.
- 在已知 $\hat{\mathbf{c}}$ 的情况下, 该问题就退化为一个确定性优化问题, 即 Nominal Optimization 问题:
    $$
    P(\hat{\mathbf{c}}): \quad z^\star(\hat{\mathbf{c}}) := \min_{\mathbf{w}} \hat{\mathbf{c}}^\top \mathbf{w}, \quad \text{s.t.}~ \mathbf{w} \in \mathcal{S}
    $$
    该优化问题的最优解 $\mathbf{w}^\star(\hat{\mathbf{c}})$ 即为最终的决策.
- 最终决策的结果又将在真实的成本 $\mathbf{c}$ 下产生实际的损失, 即 $\mathbf{c}^\top \mathbf{w}^\star(\hat{\mathbf{c}})$ 进行评估. 

然而, 传统的 PtO 方法存在一个问题, 即在训练预测模型时, 其损失函数往往是 MSE 或 MAE 等传统的回归损失函数, 其关注的是预测的 $\hat{\mathbf{c}}$ 与真实的 $\mathbf{c}$ 之间的差异, 而不是最终决策 $\mathbf{w}^\star(\hat{\mathbf{c}})$ 在真实成本下的损失 $\mathbf{c}^\top \mathbf{w}^\star(\hat{\mathbf{c}})$. 故由 SPO 开始的一系列工作 (往往成为 Decision-Focused Learning / End-to-End Learning) 就是试图将最终决策的损失直接作为训练预测模型的目标, 通过预测-决策端到端融合的方式, 来提升最终决策的性能.

> [!example]
>
> 下图说明了传统的 PtO 方法可能的问题. 
>
> ![20260817195937](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260817195937.png)
> - $S$ 是定义在决策空间 $(w_1, w_2) \in \mathbb{R}^2$ 的 feasible set, 假设为一个多边形. 由于当前的问题相当于一个 LP, 故最优值应当在 feasible set 的顶点上. 
> - 另外在成本/预测空间 $(c_1, c_2) \in \mathbb{R}^2$ 中, 真实的成本向量为 $\mathbf{c}$, 而预测的成本向量为 $\hat{\mathbf{c}}$. 圆圈表示 $\ell_2 = \frac{1}{2} \|\hat{\mathbf{c}} - \mathbf{c}\|_2^2$ 的等高线. 故若以 MSE 为估计标准, 则圆上的任意一点都是等 loss 的. 
> - 注意, 决策空间和成本空间本身是两个不同的空间. 然而, 由于最终的目标函数为 $\min_{\mathbf{w}} \mathbf{c}^\top \mathbf{w}$, 其关于 $\mathbf{w}$ 的梯度为 $\mathbf{c}$, 因此 $-\mathbf{c}$ 即为决策空间中目标函数的下降方向. 因此, 不妨通过观察 $\mathbf{c}$ 的方向来判断最终决策的结果.
> - 对于决策空间中的粉色区域, 其为上方顶点对应的 normal cone, 即, 如果梯度方向 $-\hat{\mathbf{c}}$ 落在这个 normal cone 中, 则最终的决策 $\mathbf{w}^\star(\hat{\mathbf{c}})$ 将会落在顶点上. 反之, 没有指向该顶点的梯度方向, 则最终的决策 $\mathbf{w}^\star(\hat{\mathbf{c}})$ 将会落在其他顶点上.
>
> 综上, 该图展示了预测和决策的不一致性问题. 并且注意到, 往往在线性约束中, 这样的变化还是不连续的. 在一段预测范围内, 最终的决策可能都是一样的, 但是一旦预测超出了这个范围, 最终的决策就会发生跳变. 

关于本文的余下部分, 其分为两大主线进行阐释: 一方面是提出了 SPO loss, 其直接将最终决策的损失作为训练预测模型的目标, 并且给出了一个 convex surrogate, 使得训练预测模型的目标函数是可优化的; 另一方面是从统计学习理论的角度, 对该问题进行了泛化分析. 这里将首先专注于其优化方面的内容, 在完整理解了 SPO loss 的定义和优化之后, 再从统计学习理论的角度来分析其泛化性能.


## SPO Loss Function

### True SPO Loss

首先, 对于问题
$$
P(\mathbf{c}): \quad z^\star(\mathbf{c}) := \min_{\mathbf{w}} \mathbf{c}^\top \mathbf{w}, \quad \text{s.t.}~ \mathbf{w} \in \mathcal{S}
$$
记在估计的成本 $\hat{\mathbf{c}}$ 下的最优解为 $\mathbf{w}^\star(\hat{\mathbf{c}})$, 对应的在真实成本 $\mathbf{c}$ 下的真实损失为 $\mathbf{c}^\top \mathbf{w}^\star(\hat{\mathbf{c}})$. 另一方面, 若真的知道真实成本 $\mathbf{c}$, 则能够做出的最优决策为 $\mathbf{w}^\star(\mathbf{c})$, 对应的理论最优损失为 $z^\star(\mathbf{c}) = \mathbf{c}^\top \mathbf{w}^\star(\mathbf{c})$.

因此, 可以定义 regret 为在真实成本下, 用户(通过估计)能够做出的最优决策与实际全局最优决策之间的差异:
$$
\ell^{w^*}_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c}) := \mathbf{c}^\top \mathbf{w}^\star(\hat{\mathbf{c}}) - z^\star(\mathbf{c}) = \mathbf{c}^\top \mathbf{w}^\star(\hat{\mathbf{c}}) - \mathbf{c}^\top \mathbf{w}^\star(\mathbf{c}) \geq 0 .
$$

不过上述定义有一点不严谨. 这是由于对于同样一个预测 $\hat{\mathbf{c}}$, 其对应的最优解 $\mathbf{w}^\star(\hat{\mathbf{c}}) \in \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w} := W^\star(\hat{\mathbf{c}})$ 可能不是唯一的, 而是一个解集. 这些解在 $P(\hat{\mathbf{c}})$ 问题里都是最优的, 但是在真实的 $P(\mathbf{c})$ 问题里却有可能是各不相同的. 因此, 为避免歧义, 往往会考虑 worst-case 的情况, 即取 $\max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w}$, 于是最终的 SPO loss 定义为:
$$
\ell_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c}) := \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w} - z^\star(\mathbf{c}) = \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w} - \mathbf{c}^\top \mathbf{w}^\star(\mathbf{c}) \geq 0 .
$$

> [!example]
>
> 文中举了一个重要的例子, 这也是后文 spo+ 的 intuition 所在. 其简单描述为: 对于 $c \in \{-1, 1\}$ 的而二分类问题, 其 classification error $\ell_\text{CE}(\hat{c}, c) = \mathbf{1}_{\operatorname{sign}(\hat{c}) \neq c}$. 作者生成, $\ell_\text{CE}$ 等价于一个 feasible set $S= [-1/2, 1/2]$ 的 $P(c)$ 问题对应的 SPO loss. 
>
> *Proof*.
> 首先对于二分类问题, $\ell_\text{CE}(\hat{c}, c) = \mathbf{1}_{\operatorname{sign}(\hat{c}) \neq c}$ 的设定是自然的 (注意这里原文中似乎有一个笔误, 其写成了 $\mathbf{1}_{\operatorname{sign}(\hat{c}) = c}$, 但显然不符合直觉). 
>
> 另一方面考虑优化问题 $\min_{w \in [-1/2, 1/2]} c w$. 由于 $c \in \{-1, 1\}$, 因此分类讨论:
>   - 当 $c = 1$ 时, $\min_{w \in [-1/2, 1/2]} c w = \min_{w \in [-1/2, 1/2]} w = -1/2$, 对应的最优解为 $w^\star(c) = -1/2$.
>  - 当 $c = -1$ 时, $\min_{w \in [-1/2, 1/2]} c w = \min_{w \in [-1/2, 1/2]} -w = -(-1/2) = 1/2$, 对应的最优解为 $w^\star(c) = 1/2$.
>
> 因此, $w^\star(c) = -c/2$. 另一方面, 对于预测 $\hat{c}$, 其对应预测问题的最优解为 $w^\star(\hat{c}) \in \arg\min_{w \in [-1/2, 1/2]} \hat{c} w$. 同样分类讨论:
>  - 当 $\hat{c} > 0$ 时, $w^\star(\hat{c}) = -1/2$.
> - 当 $\hat{c} < 0$ 时, $w^\star(\hat{c}) = 1/2$.
>
> 因此, 对于 $cw$ 与 $\hat{c}w$ 的符号关系, 有 $\operatorname{sign}(\hat{c}) = c$ 当且仅当 $w^\star(\hat{c}) = w^\star(c)$. 而这恰恰就是 classification error 的定义. 
>
> $\square$
>
> 综上, 这个例子想要试图说明, $0-1$ loss 的分类问题, 其实就是一个简单的 SPO loss 的特殊情况. 


### SPO+ Surrogate

由于前面 SPO loss
$$
\ell_\text{SPO}(\hat{\mathbf{c}}, \mathbf{c}) := \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w} - z^\star(\mathbf{c}), \quad W^\star(\hat{\mathbf{c}}) = \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}
$$
的求解困难, 因此这里的目标是构建一个 SPO 的 surrogate. 首先给出其定义, 其形式为:
$$
\ell_{\text{SPO+}} (\hat{\mathbf{c}}, \mathbf{c}) := \max_{\mathbf{w} \in \mathcal{S}} \left\{\mathbf{c}^\top \mathbf{w} - 2 \hat{\mathbf{c}}^\top \mathbf{w}\right\} + 2 \hat{\mathbf{c}}^\top \mathbf{w}^\star({\mathbf{c}}) - z^\star(\mathbf{c})
$$

其具有如下分析性质:
- $\ell_{\text{SPO+}} (\hat{\mathbf{c}}, \mathbf{c})$ 是一个 convex function
- 对于任意 $\hat{\mathbf{c}} \in \mathbb{R}^d$, $\ell_{\text{SPO}} (\hat{\mathbf{c}}, \mathbf{c}) \leq \ell_{\text{SPO+}} (\hat{\mathbf{c}}, \mathbf{c})$, 即 SPO+ 是 SPO 的一个上界.
- 对于任意给定 $\hat{\mathbf{c}} \in \mathbb{R}^d$, $2(\mathbf{w}^\star(\mathbf{c}) - \mathbf{w}^\star(2\hat{\mathbf{c}} - \mathbf{c}))$ 是 $\ell_{\text{SPO+}} (\hat{\mathbf{c}}, \mathbf{c})$ 的 subgradient.

---

下给出具体推导过程. 


- 首先进行一个等价改写. 对于任意 $\alpha \in \mathbb{R}$, 有
  $$
  \ell_{\text{SPO}}(\hat{\mathbf{c}}, \mathbf{c}) = \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \{\mathbf{c}^\top \mathbf{w} ~\underline{-~ \alpha \hat{\mathbf{c}}^\top \mathbf{w}}~\} \underline{+ \alpha z^\star(\hat{\mathbf{c}})} - z^\star(\mathbf{c}) 
  $$
  - 因为根据定义, $z^\star(\hat{\mathbf{c}}) = \min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}$, 而 $\mathbf{w} \in W^\star(\hat{\mathbf{c}})$, 因此 $\hat{\mathbf{c}}^\top \mathbf{w} = z^\star(\hat{\mathbf{c}})$. 因此上式中的下划线部分之和为 0.

- 接着, 将 $\mathbf{w} \in W^\star(\hat{\mathbf{c}})$ 放宽为 $\mathbf{w} \in \mathcal{S}$, 则因为 $W^\star(\hat{\mathbf{c}}) = \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w} \subseteq \mathcal{S}$, 有:
  $$
  \ell_{\text{SPO}}(\hat{\mathbf{c}}, \mathbf{c}) =
  \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) - z^\star(\mathbf{c}) \leq \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) - z^\star(\mathbf{c}) \qquad (\dagger)
  $$

- 由于上式 RHS 对于任意 $\alpha \in \mathbb{R}$ 都成立, 因此可以对 $\alpha$ 取 infimum, 则有
  $$
  \ell_{\text{SPO}}(\hat{\mathbf{c}}, \mathbf{c}) \leq \inf_{\alpha \in \mathbb{R}} \left\{ \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) \right\} - z^\star(\mathbf{c}) 
  $$
  通过严格数学推导 (具体详细过程将在后文单独给出), 可以证明上面的不等式其实为等号, 其在 $\alpha \to \infty$ 时取到, 即
  $$
  \ell_{\text{SPO}}(\hat{\mathbf{c}}, \mathbf{c}) = \lim_{\alpha \to \infty} \left\{ \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) \right\} - z^\star(\mathbf{c})
  $$
  - 下提供一个直观的数学理解. 对于 $\inf$ 里面的部分重新整理, 合并进 $\max$ 里面,
    $$
    \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) = \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha (\hat{\mathbf{c}}^\top \mathbf{w} - z^\star(\hat{\mathbf{c}}))\}
    $$
    当 $\alpha$ 逐渐增大, $\alpha(\hat{\mathbf{c}}^\top \mathbf{w} - z^\star(\hat{\mathbf{c}}))$ 这一项将主导取值, 故 $\max$ 会逐渐趋向于
    $$
    \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha (\hat{\mathbf{c}}^\top \mathbf{w} - z^\star(\hat{\mathbf{c}}))\} \to \max_{\mathbf{w} \in \mathcal{S}} \{- \alpha (\hat{\mathbf{c}}^\top \mathbf{w} - z^\star(\hat{\mathbf{c}}))\} = \min_{\mathbf{w} \in \mathcal{S}} \{\alpha (\hat{\mathbf{c}}^\top \mathbf{w} - z^\star(\hat{\mathbf{c}}))\}
    $$
    注意到, 这里 $\hat{\mathbf{c}}^\top \mathbf{w} - z^\star(\hat{\mathbf{c}})$ 就表示在预测 $\hat{\mathbf{c}}$ 下, 当前决策 $\mathbf{w}$ 与预测情况中最优决策 $\mathbf{w}^\star(\hat{\mathbf{c}})$ 的差异. 因此, 当 $\alpha \to \infty$ 时, 任意非零差异都会被无限放大, 因此若要最小化 $\alpha (\hat{\mathbf{c}}^\top \mathbf{w} - z^\star(\hat{\mathbf{c}}))$, 则必然要求 $\mathbf{w} \in W^\star(\hat{\mathbf{c}})$.而当 $\mathbf{w} \in W^\star(\hat{\mathbf{c}})$ 时, 则自动又回到了放缩前的 SPO loss. 
  
  - 简单讲这个部分的推导逻辑即为:
    $$
    W^*(\hat{\mathbf{c}}) \xrightarrow{\text{relax}} \mathcal{S} \xrightarrow{\alpha \to \infty} W^*(\hat{\mathbf{c}}).
    $$

- 对上述 $\lim$ 形式的 SPO loss 进行放缩, 即可得到最终的 SPO+ surrogate loss, 过程如下:
    $$
    \begin{aligned}
    \ell_{\text{SPO}} (\hat{\mathbf{c}}, \mathbf{c}) 
    &\stackrel{(1)}{=} \lim_{\alpha \to \infty} \left\{ \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha \hat{\mathbf{c}}^\top \mathbf{w}^\star(\hat{\mathbf{c}}) - z^\star(\mathbf{c})\right\} \\
    &\stackrel{(2)}{\leq} \max_{\mathbf{w} \in \mathcal{S}} \left\{\mathbf{c}^\top \mathbf{w} - 2 \hat{\mathbf{c}}^\top \mathbf{w}\right\} + 2 \hat{\mathbf{c}}^\top \mathbf{w}^\star(2\hat{\mathbf{c}}) - z^\star(\mathbf{c}) \\
    &\stackrel{(3)}{\leq} \max_{\mathbf{w} \in \mathcal{S}} \left\{\mathbf{c}^\top \mathbf{w} - 2 \hat{\mathbf{c}}^\top \mathbf{w}\right\} + 2 \hat{\mathbf{c}}^\top \mathbf{w}^\star({\mathbf{c}}) - z^\star(\mathbf{c}) =: \ell_{\text{SPO+}} (\hat{\mathbf{c}}, \mathbf{c})
    \end{aligned}
    $$
    - 其中
      - (1) 是前面 SPO loss 的 $\lim$ 形式, 且注意到 $z^\star(\hat{\mathbf{c}}) = \hat{\mathbf{c}}^\top \mathbf{w}^\star(\hat{\mathbf{c}})$.这里 $\alpha$ 本质上相当于一个 Lagrangian Multiplier, 其通过对偶问题, 将原先难以处理的 $W^\star(\hat{\mathbf{c}})$ 约束放宽为 $\mathcal{S}$, 并通过 $\alpha$ 的放缩来逼近原先的约束.
      - (2) 是对 $\lim_{\alpha \to \infty}$ 进行放缩, 取 $\alpha = 2$.注意到该不等式的成立依赖于 $\max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\}$ 事实上是一个单调递增的函数. 其单调性将在后文进行证明. 至于特值 $\alpha = 2$ 的选择, 其是为了后续的统计理论性质的方便. 与当前 convex surrogate 的定义无关, 理论上也可以选择其他的 $\alpha > 1$.
      - (3) 是将 $2\hat{\mathbf{c}}^\top \mathbf{w}^\star(2\hat{\mathbf{c}})$ 放宽为 $2\hat{\mathbf{c}}^\top \mathbf{w}^\star({\mathbf{c}})$. 这是因为 $\mathbf{w}^\star(\hat{\mathbf{c}}) \in \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}$, 而 $\mathbf{w}^\star({\mathbf{c}}) \in \mathcal{S}$, 因此 $\hat{\mathbf{c}}^\top \mathbf{w}^\star(\hat{\mathbf{c}}) \leq \hat{\mathbf{c}}^\top \mathbf{w}^\star({\mathbf{c}})$. 将 $\hat{\mathbf{c}}$ 替换为 $2\hat{\mathbf{c}}$ 后, 该不等式仍然成立.

*Proof*. 已知 $\ell_\text{SPO} = \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w} - z^\star(\mathbf{c})$, 其中 $W^\star(\hat{\mathbf{c}}) = \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}$. 且已说明, 对于任意 $\alpha \in \mathbb{R}$, SPO loss 可以改写为
$$
\begin{aligned}
\ell_{\text{SPO}}(\hat{\mathbf{c}}, \mathbf{c}) & = \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) - z^\star(\mathbf{c}) \\ 
& \leq \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) - z^\star(\mathbf{c}) \\
&\leq \inf_{\alpha \in \mathbb{R}} \left\{ \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha \hat{\mathbf{c}}^\top \mathbf{w}\} + \alpha z^\star(\hat{\mathbf{c}}) - z^\star(\mathbf{c})\right\} .
\end{aligned}
$$
下证明其实际上是等号, 且 infimum 在 $\alpha \to \infty$ 时取到.

- 对于最初的 SPO 定义, 其中一个比较麻烦的地方在于 $\max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w}$ 的求解困难, 因为本身还内含了一个 $\arg\min$ 的约束. 
  - 注意到, 对于 $\mathbf{w} \in W^\star(\hat{\mathbf{c}})\subseteq \mathcal{S}$, 有 $\hat{\mathbf{c}}^\top \mathbf{w} = z^\star(\hat{\mathbf{c}})$. 且对任意 $\mathbf{w} \in \mathcal{S}$, 有 $\hat{\mathbf{c}}^\top \mathbf{w} \geq z^\star(\hat{\mathbf{c}})$. 因此, $W^\star(\hat{\mathbf{c}})$ 等价于 $\mathcal{S} \cap \{\mathbf{w} \mid \hat{\mathbf{c}}^\top \mathbf{w} \leq z^\star(\hat{\mathbf{c}})\}$. 故 $\max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w}$ 可以改写为:
        $$
        \begin{aligned}
        & \max_{\mathbf{w}} \mathbf{c}^\top \mathbf{w} \quad \text{s.t.}~ \mathbf{w} \in W^\star(\hat{\mathbf{c}}) \\
        \iff &\max_{\mathbf{w}} \mathbf{c}^\top \mathbf{w} \quad \text{s.t.}~ \mathbf{w} \in \mathcal{S},~ \hat{\mathbf{c}}^\top \mathbf{w} \leq z^\star(\hat{\mathbf{c}}) \qquad (1)
        \end{aligned}
        $$
- 对上述等价变形 $(1)$, 其可以通过 Lagrangian duality 来进行求解. 其 Lagrangian 为
  $$
  \mathcal{L}(\mathbf{w}, \alpha) = \mathbf{c}^\top \mathbf{w} - \alpha (\hat{\mathbf{c}}^\top \mathbf{w} - z^\star(\hat{\mathbf{c}})), \quad \alpha \geq 0
  $$
  因此, 根据 weak duality, 对于任意 feasible $\mathbf{w} \in \mathcal{S}$, $\alpha \geq 0$, 有
  $$
  p^\star :=
  \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w} \leq
  \mathcal{L}(\mathbf{w}, \alpha) \leq \max_{\mathbf{w} \in \mathcal{S}} \mathcal{L}(\mathbf{w}, \alpha) =: q(\alpha) \\ \implies \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w} \leq \inf_{\alpha \geq 0} q(\alpha) =: q^\star
  $$

- 进一步, 根据 [Borwein-Lewis 2010, Theorem 4.3.8](https://roke.eecs.ucf.edu/Reading/Papers/ConvAnalysis.pdf), 可以证明其是 strong duality, 即
  $$
  \max_{\mathbf{w} \in W^\star(\hat{\mathbf{c}})} \mathbf{c}^\top \mathbf{w} = \inf_{\alpha \geq 0} q(\alpha).
  $$
  - B-L 定理本身考虑的是: 关于约束优化问题 $p = \inf_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.}~ \mathbf{g}(\mathbf{x}) = [g_1(\mathbf{x}), \ldots, g_m(\mathbf{x})]^\top \leq 0, $
    考虑其 value function: 对于给定 $\mathbf{b} \in \mathbb{R}^m$, 定义 $v(\mathbf{b}) := \inf_{\mathbf{g}(\mathbf{x}) \leq \mathbf{b}} f(\mathbf{x})$. 若假设 $f,g_i$ 是 closed function (即其 epigraph 是 closed set), 且存在 $\hat{\lambda}_0 \geq 0$, $\hat{\boldsymbol{\lambda}} \in \mathbb{R}^m_+$, 使得 $\hat{\lambda}_0 f(\mathbf{x}) + \hat{\boldsymbol{\lambda}}^\top \mathbf{g}(\mathbf{x})$ 具有 compact level sets (对于任意实数 $r$, $\{\mathbf{x} \mid \hat{\lambda}_0 f(\mathbf{x}) + \hat{\boldsymbol{\lambda}}^\top \mathbf{g}(\mathbf{x}) \leq r\}$ 是 compact), 则只要 $v(\mathbf{b})$ 是 finite 的, 其 infimum 就是 attainable 的. 此外, 若进一步 $f, g_i$ 是 convex, 且 dual 是 finite, 则 strong duality 成立.
  - 可以逐一验证, 在当前的 SPO loss 的定义中确实服从上述条件. 具体细节暂时省略. 

- 此外, 为从 $\inf_{\alpha \geq 0}$ 推出 $\lim_{\alpha \to \infty}$, 需要证明 $q(\alpha) = \max_{\mathbf{w} \in \mathcal{S}} \{\mathbf{c}^\top \mathbf{w} - \alpha (\hat{\mathbf{c}}^\top \mathbf{w} - z^\star(\hat{\mathbf{c}}))\}$ 是单调递减的. 这可以根据 $\hat{\mathbf{c}}^\top \mathbf{w} - z^\star(\hat{\mathbf{c}}) \geq 0$ 对 $\alpha$ 的单调性直接得到. 