# Learning to Warm-Start Fixed-Point Optimization Algorithms

> https://arxiv.org/abs/2309.07835

## TL;DR

- 该论文提出了一种学习方法来为不动点优化算法提供热启动以加速其收敛速度.

## 章节梳理

### Introduction & Background

**Fixed-point optimization**

- 不动点优化本身是指: 寻找 $z$ 使得 $z = T(z)$, 其中 $T$ 是一个映射. 事实上许多优化问题都可以整理为这样的形式. 在迭代中, 即为:
    $$
    z^{(i+1)} = T_\theta(z^{(i)})
    $$
    - 其中 $\theta$ 是算法的参数, 例如学习率, 数据本身等.  $T_\theta$ 表示由 $\theta$ 定义的具体的更新规则.
    - 若算法收敛了, 则 $z^\star = T_\theta(z^\star)$, 即 $z^\star$ 是 $T_\theta$ 的一个不动点.


- 论文中指出, 许多优化算法本身都是不动点优化算法, 例如: Proximal Gradient Descent, ADMM, 等等. 这些算法的迭代过程都可以看作是一个不动点迭代.
  - 例如, 对于 SGD, 其迭代过程可以写成:
    $$
    z^{(i+1)} = z^{(i)} - \eta \nabla f(z^{(i)}) := T(z^{(i)})
    $$
    在收敛点 $\nabla f(z^\star) = 0$, 时, 则有
    $$
    T(z^\star) = z^\star - \eta \nabla f(z^\star) = z^\star
    $$
    因此是一个不动点.
    - 在这个例子中, $T$ 的参数 $\theta$ 相当于一个由数据等抽象出的一个环境参数, 其影响了 $T$ 的具体形式.



**Warm-starting**

- 对于上述的不动点迭代问题, 一个减少迭代次数实现加速的方法是热启动 (warm-starting), 即选择一个好的初始点 $z^{(0)}$ 来加速收敛. warm-start 不会改变原有算法的迭代过程, 只是通过更聪明的初始点来减少迭代次数.
- 当前已有 warm-start 的算法的主要问题:
  - 缺少 generalization guarantee
  - 本身的学习过程和后续的算法是 decoupled 的. 也就是一个 end-to-end 的问题. 容易导致在 unseen problem 上出现 sub-optimal.



### Warm-start Framework

回忆, 给定数据等具体任务环境 $\theta$, 以及一个不动点算法 $T_\theta$, 其迭代过程为:
$$
z^{(i+1)} = T_\theta(z^{(i)})
$$

- 传统的 warm-start 的思路是: 设计一个神经网络 $h_w$, 输入任务 $\theta$, 输出一个初始点 $z^{(0)} = h_w(\theta)$. 后续的思路是, 直接让 $h_w(\theta)$ 接近最优解 $z^\star$.
- 本文提出的 warm-start 框架则是: 设计一个神经网络 $h_w$, 输入任务 $\theta$, 输出一个初始点 $z^{(0)} = h_w(\theta)$, 但考虑的是该迭代点 $z^{(0)}$ 经过 $K$ 步迭代后的结果 $z^{(K)}$, 即:
    $$
    z^{(K)} = T_\theta^K(h_w(\theta))
    $$
    并以此为目标进行训练. 也就是说, 训练的目标不是让初始值 $h_w(\theta)$ 本身接近最优解, 而是让经过 $K$ 步迭代后的结果 $z^{(K)}$ 接近最优解. 这样就将 warm-start 的学习过程和后续的算法紧密结合在一起, 从而提高了 generalization 的能力.


通过一些例子, 文章试图说明: 对于 warm start 的点的选取本身必须考虑后续的迭代过程. 虽然可能一些 warm start 到 $z^\star$ 的距离是相同的, 但是各自可能分别跑出完全不同的迭代轨迹. (也就是说只看静态的初始点是没有意义的).

#### Training

注意, 我们在当前阶段的目标是学习这样一个神经网络 $h_w$, 使得其能够适配不同的任务 $\theta$. 因此, 这个时候, 我们会根据具体的下游任务 $\theta$ 的选择, 来确定一个需要的损失函数 $\ell_\theta$, 来衡量 $z^{(K)}$ 的好坏, 例如:
- Fixed-point residual
  - $\ell_\theta(z) = \|z - T_\theta(z)\|^2$ (因为如果 $z$ 是一个不动点, 则 $z = T_\theta(z)$, 因此残差越小越好).
  - 相当于一个局部梯度信息, solver friendly
  
- Regression loss
  - $\ell_\theta(z) = \|z - z^\star\|^2$ (直接衡量 $z$ 和最优解 $z^\star$ 的距离). 也就是如果真实的 groudn truth 是已知的, 则直接让 $z^{(K)}$ 接近 $z^\star$.
  - 相当于一个全局信息.


在确定了 loss 的具体形式后, 我们总的的训练目标就是:
$$
\min_w \mathbb{E}_{\theta\sim Q}[\ell_\theta(T_\theta^K(h_w(\theta)))]
$$
- 其中 $Q$ 是一个任务分布, 代表了我们希望模型能够适配的任务的分布. 



#### Testing

假设我们已经通过前面的训练阶段, 学习到了一个神经网络 $h_w$, 那么在测试阶段, 给定一个新的任务 $\theta'$, 我们就可以通过 $h_w(\theta')$ 来得到一个初始点, 然后通过 $T_{\theta'}$ 的迭代来得到最终的结果. 

与训练不同, 
- 测试阶段, 我们可以迭代次数 $t\neq K$, 也就是尽管在训练阶段我们是以 $K$ 步迭代后的结果为目标来训练的, 但是在测试阶段我们可以选择任意的迭代次数 $t$ 来得到最终的结果, 这一方面允许我们在测试阶段进行更多的迭代来进一步优化结果, 另一方面也在观测其迭代泛化能力. 
- 在测试时, 我们将统一考虑 fixed-point residual 作为评估指标, 最终考察的是如下的 risk:
    $$
    R^{(t)}(h_w) =    \mathbb{E}_{\theta\sim Q}[\ell_\theta(T_\theta^t(h_w(\theta)))]
    $$


此外，按照机器学习的思路， 由于我们最终只能观测到 有限 $N$ 个来自 $Q$ 的任务样本 $\{\theta_i\}_{i=1}^N$, 因此最终得到的是如下经验风险
$$
\hat{R}^{(t)}(h_w) = \frac{1}{N}\sum_{i=1}^N \ell_{\theta_i}(T_{\theta_i}^t(h_w(\theta_i)))
$$


### PAC Theory Analysis

学到的 warm start 不只在训练问题上有效; 在新问题上,只要原 fixed-point operator 具有合适的收敛性质,那么经过 $t$ 步迭代后的 fixed-point residual 可以由训练中的经验量和一个复杂度项控制.


### Choosing the right computational architecture

- 文中证明, 当训练出 warm-start 之后, 测试时继续迭代原先的优化算法, 总是会带来改进的. 

- 在 loss 选择上，fixed-point residual loss 更便宜且与测试指标一致，regression loss 则利用 ground-truth solution 的全局信息，并且通常能给 future iterations 提供更强的理论保证



### Experiments

- Learned warm-starting 几乎在所有任务上都显著优于 cold start
- $k=0$ 是危险的.
  - 相当于只学习一个看起来不错的 initialization, 但是完全不引入后续的迭代过程. 
  - 这时实验展示, 其可能看起来在 $t=0$ 的时候表现不错, 但是在后续的迭代过程中会迅速退化, 甚至比 cold start 更差. 
- 训练时的 $k$ 的选择也有一定影响. 是一个和实验相关的超参数. 
- regression loss 的表现通常优于 fixed-point residual loss.


### Future Work

- 目前主要还是使用的 MLP 来实现 $h_w$, 未来可以考虑一些更适合结构化数据的架构
- 扩展到 non-convex optimization
- 扩展到更大规模的优化问题


