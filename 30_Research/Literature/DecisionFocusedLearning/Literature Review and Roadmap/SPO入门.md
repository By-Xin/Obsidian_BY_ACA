# SPO 入门指南

## 什么是 SPO

Smart Predict then Optimize (SPO, 先预测再决策) 是在机器学习领域中的既有训练范式的一个补充. 
- 在许多决策问题中, 我们往往是使用机器学习模型对一些未知的变量进行预测, 然后再使用优化求解器利用这些预测值来求解一个优化问题进行决策. 而这样的策略往往是各自独立的. 传统的机器学习模型的训练目标是最小化预测的准确性, 但是真正的 目标其实是做出好的决策. 
- SPO, 或者更广义的 Decision-Focused Learning (DFL, 决策导向学习), 的核心思想即为能否提供一个端到端的训练框架, 是的模型可以直接围绕下游的决策质量来训练, 而不是单纯地追求预测的准确性 ([[Elmachtoub & Grigas, 2017]]; [[Mandi et al., 2023]]; [[Wilder et al., 2018]]).

其数学建模如下. 
- 给定可观测特征 $\mathbf{x}$ (在当前语境下往往也叫 context), 如天气, 交通状况, 历史价格, 宏观因子等. 这相当于我们进行预测的输入. 
- 给定一个优化的问题参数 $\mathbf{c}$, 这是下游优化问题的输入, 但是我们无法直接观测到的, 例如成本, 权重等. 由于 $\mathbf{c}$ 无法直接观测到, 我们需要通过一个预测模型 $f_\theta(\mathbf{x})$ 来预测 $\hat{\mathbf{c}}= f_\theta(\mathbf{x})$.
- 给定一个优化问题的可行集合 $\mathcal{S}$ 表示所有允许的行动集合. 则优化的目标是求解以下问题:
    $$
    \mathbf{w}^\star(\hat{\mathbf{c}}) \in \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}
    $$
    即为基于预测 $\hat{\mathbf{c}}$ 的最优决策

传统的机器学习最小化的是类似类似 $\|\hat{\mathbf{c}} - \mathbf{c}\|^2$ 的损失函数. 然而在 SPO 的观点下, 这样的输出误差并不一定能反映出决策的质量. 因此, 其对应的任务损失, 称为 **Decision Regret**, 定义为:
$$
\ell_{\text{SPO}}(\hat{\mathbf{c}}, \mathbf{c}) = \mathbf{c}^\top \mathbf{w}^\star(\hat{\mathbf{c}}) -  \mathbf{z}^\star(\mathbf{c})
$$
- 其中 $\mathbf{z}^\star(\mathbf{c}) = \arg\min_{\mathbf{w} \in \mathcal{S}} \mathbf{c}^\top \mathbf{w}$ 是基于真实参数 $\mathbf{c}$ 的一个理论最优 oracle 决策.
- 二者的差表示: 实际做出的决策 $\mathbf{w}^\star(\hat{\mathbf{c}})$ 在真实的成本 $\mathbf{c}$ 下的损失,  与在理想状态下, 如果我们一开始就知道 $\mathbf{c}$ 的话, 可以做出的最优决策 $\mathbf{z}^\star(\mathbf{c})$ 之间的差距. 即为 **由于缺少信息使得我们按照经验参数 $\hat{\mathbf{c}}$ 相比于理想参数 $\mathbf{c}$ 做出的决策质量的损失**.

这一思想的本质在于:
- 如果以预测为导向, 那么所有的误差都是同质的
- 如果以决策为导向, 那么只有真正改变了行动活价值的预测误差才真正重要. 从实践中看, 一个模型即使 MSE 再差, 也有可能做出更好决策 ([[Lee et al., 2024]]; [[Mandi et al., 2023]]; [[Wilder et al., 2018]]).

## 重要术语与发展历史

- **SPO**: Smart Predict then Optimize, 先预测再决策. 由 [[Elmachtoub & Grigas, 2017]] 首次提出. 相关文章还有 [[H. Liu & Grigas, 2021]]; [[Balghiti et al., 2019]].
- **SPO+**: 针对困难 SPO 的一个凸 surrogate. 相关文章有 [[Elmachtoub & Grigas, 2017]]; [[H. Liu & Grigas, 2021]].
- **DFL**: Decision-Focused Learning, 相当于这个话题的一个更广义的机器学习表述. 主要指通过 optimizer 进行 end-to-end 训练的学习范式. 相关文章有 [[Wilder et al., 2018]]; [[Mandi et al., 2023]].
- **Differentiable Optimization Layer**: 将优化过程视为神经网络中可微分的一层, 是相关工作的一个核心技术. 相关文章有 [[Amos & Kolter, 2017]]; [[Agrawal et al., 2019]].
- **Contextual Optimization**: 一个在运筹学情境中更广阔的研究领域. Generally 指在不确定下基于 context 进行优化的研究. 相关文章有 [[Donti et al., 2017]]; [[Sadana et al., 2023]].


其重要发展线索如下:
- [[Donti et al., 2017]], [[Amos & Kolter, 2017]], 提出预测与决策之间的 end-to-end 思想. 展现了端到端的任务训练与可微优化. 
- [[Elmachtoub & Grigas, 2017]] 提出 SPO 框架, 定义了 LP 中的 SPO 损失, 以及一个凸 surrogate SPO+.
- [[Wilder et al., 2018]], [[Ferber et al., 2019]] 将组合优化与可微训练联系起来, 是离散问题中早期的关于 DFL 的工作.
- [[Mandi et al., 2019]] 讨论了更困难组合优化中的 SPO, 扩展到了更复杂的离散问题中.
- [[Balghiti et al., 2019]], [[H. Liu & Grigas, 2021]] 进行理论深化, 给出了关于泛化, 校准和优先样本的一些保证等. 
- [[Kotary et al., 2021]], [[Mandi et al., 2023]], [[Tang & Khalil, 2022]] 对领域进行了规范整合和一些baseline的构建.
- 近期扩展: 主要包括鲁棒性, 剃度病态, 约束中的不确定性等问题的讨论. 相关工作有 [[Schutte et al., 2023]]; [[Huang & Gupta, 2024]]; [[Hu et al., 2022; Hu et al., 2023; Hu et al., 2024]].

## 核心奠基论文

-  Smart "Predict, then Optimize" [[Elmachtoub & Grigas, 2017]]

-  Decision focused learning for combinatorial optimization [[Wilder et al., 2018]]


## 理论分支的研究关注

理论方面的核心关注为: 如果模型训练的时候用的是一些 surrogate loss (而不是真实的 SPO regret), 那么在什么条件下, 这样的做法依然是可靠的. 



