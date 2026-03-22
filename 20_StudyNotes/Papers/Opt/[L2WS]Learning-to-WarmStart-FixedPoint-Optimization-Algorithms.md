# Learning to Warm-Start Fixed-Point Optimization Algorithms

> https://arxiv.org/abs/2309.07835

## TL;DR

- 该论文提出了一种学习方法来为不动点优化算法提供热启动以加速其收敛速度.


## Introduction & Background

### Parametric Fixed-point Problem

- 首先, 文中考虑的 Parametric Fixed-point Problem 的形式如下:
  $$
  \text{Find } \mathbf{z} \in \mathbb{R}^p \text{ such that } \mathbf{z} = T_\theta(\mathbf{z}) \qquad \text{(1)}
  $$

  - $\mathbf{z} \in \mathbb{R}^p$ 是决策变量; 
  - $\theta \in \Theta \subseteq \mathbb{R}^d$ 是一个环境/问题参数,  不同的 $\theta$ 代表了不同的具体优化问题. 例如 $\theta$ 可以包含了数据集, 任务的具体要求等.  
  - $T_\theta: \mathbb{R}^p \to \mathbb{R}^p$ 是一个由 $\theta$ 定义的映射, 代表了一个优化的 operator.

- 若对应于迭代的过程, 则可以写成如下的迭代形式:
  $$
  \mathbf{z}^{(i+1)} = T_\theta(\mathbf{z}^{(i)}) \quad \text{for } i=0,1,2,\ldots
  $$

  - 若假设算法最终会收敛到一个不动点 $\mathbf{z}^\star(\theta)$, 即:
    $$
    \lim_{i\to\infty} \|\mathbf{z}^{(i)} - \mathbf{z}^\star(\theta)\| = 0
    $$
    则 $\mathbf{z}^\star(\theta)$ 就是满足 $(1)$ 的解.  


  - 对于初值 $\mathbf{z}^{(0)}$, 其选择会影响迭代的收敛速度, 因此一个好的初值可以加速算法的收敛, 这也是 warm-start 的核心思想.


  - 在实际的优化算法中, 往往会考虑 $\epsilon$-approximate 的收敛结构, 即当 **fixed-point residual** 有
    $$
    \|\mathbf{z}^{(i)} - T_\theta(\mathbf{z}^{(i)})\| \leq \epsilon,
    $$
    就认为 $\mathbf{z}^{(i)}$ 已经足够接近一个不动点了.




- 不动点问题本质上相当于参数化凸优化问题的最优性条件, 因此求解 $(1)$ 的过程就等价于求解这样的凸优化问题.  文中表示, 几乎所有的凸优化问题都可以转化为寻找一个不动点的形式, 许多优化算法表面上看是在更新变量, 但实际上都可以表示为寻找某个算子的一个不动点.  下面是文中举的几个具体的例子. 
  - Gradient Descent: 
    - 优化目标: $\min_z f_\theta(z)$
    - 标准迭代过程: $z^{(i+1)} = z^{(i)} - \eta \nabla f_\theta(z^{(i)})$ 
    - 对应的 fixed-point operator: $T_\theta(z) = z - \eta \nabla f_\theta(z)$. 
    - 当达到 optimal point 时, $z^\star = T_\theta(z^\star) = z^\star - \eta \nabla f_\theta(z^\star)$, 也就是 $\nabla f_\theta(z^\star) = 0$, 满足 optimality condition (在凸优化的情况下).
  - Proximal Gradient Descent:
    - 优化目标: $\min_z f_\theta(z) + g_\theta(z)$
    - 标准迭代过程: $z^{(i+1)} = \text{prox}_{\eta g_\theta}(z^{(i)} - \eta \nabla f_\theta(z^{(i)}))$
    - 对应的 fixed-point operator: $T_\theta(z) = \text{prox}_{\eta g_\theta}(z - \eta \nabla f_\theta(z))$.
    - 当达到 optimal point 时, $z^\star = T_\theta(z^\star) = \text{prox}_{\eta g_\theta}(z^\star - \eta \nabla f_\theta(z^\star))$, 也就是 $0 \in \nabla f_\theta(z^\star) + \partial g_\theta(z^\star)$, 满足 optimality condition (在凸优化的情况下).

  - ADMM (Douglas-Rachford Splitting):
    - 优化目标: $\min_{u} f_\theta(u) + g_\theta(u)$ (注意这里的 $u$ 是最终的决策变量, 而下文的 $z$ 是 ADMM 内部的一个迭代变量)
    - 标准迭代过程:
      - $\tilde{u}^{(i+1)} = \text{prox}_{g_\theta}(z^{(i)})$
      - $u^{(i+1)} = \text{prox}_{f_\theta}(2\tilde{u}^{(i+1)} - z^{(i)})$
      - $z^{(i+1)} = z^{(i)} + u^{(i+1)} - \tilde{u}^{(i+1)}$
    - 对应的 fixed-point operator: $T_\theta(z) = z + \text{prox}_{f_\theta}(2\text{prox}_{g_\theta}(z) - z) - \text{prox}_{g_\theta}(z)$.
    - 当达到 optimal point 时, $z^\star = z^\star +  u^\star - \tilde{u}^\star$, 也就是 $u^\star = \tilde{u}^\star$, 这意味着两个子 proximal 的步骤趋于一致, 这个共同点就是最终的 optimal point.




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


