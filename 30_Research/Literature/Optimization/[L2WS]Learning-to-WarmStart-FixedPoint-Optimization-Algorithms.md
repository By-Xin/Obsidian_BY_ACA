# Learning to Warm-Start Fixed-Point Optimization Algorithms

> https://arxiv.org/abs/2309.07835


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


### Learning to Warm-Start

Learning to Warm-Start 的核心思想是, 通过一个神经网络 $h_w: \Theta \to \mathbb{R}^p$ 来学习一个从问题参数 $\theta$ 到一个好的初始点 $z^{(0)}$ 的映射. 也就是说, 给定一个新的问题实例 $\theta$, 我们可以通过 $h_w(\theta)$ 来得到一个初始点, 然后通过迭代 $T_\theta$ 来求解最终的结果.

本文的一个重点不是直接学习一个好的解, 而是两步的过程: 
- 先通过神经网络 $h_w$ 将 $\theta$ 映射到一个初始点 $z^{(0)} = h_w(\theta)$, 这个初始点本身的质量可能并不高, 但是它是对于后续迭代更 promising 的一个起点.
- 然后通过迭代 $T_\theta$ 来逐步优化这个初始点, 得到这个系统最终的输出 $z^{(K)} = T_\theta^K(h_w(\theta))$, 作为整个优化的监督信号或 inference 阶段的最终输出结果.


#### Training

注意, 我们在当前阶段的目标是学习这样一个神经网络 $h_w$, 使得其能够适配不同的任务 $\theta$. 因此, 这个时候, 我们会根据具体的下游任务 $\theta$ 的选择, 来确定一个需要的损失函数 $\ell_\theta$, 来衡量 $z^{(K)}$ 的好坏, 例如:
- Fixed-point residual
  - $\ell^{\text{FP}}_\theta(z) = \|z - T_\theta(z)\|^2$ (因为如果 $z$ 是一个不动点, 则 $z = T_\theta(z)$, 因此残差越小越好).
  - 相当于一个局部梯度信息, solver friendly
  
- Regression loss
  - $\ell^{\text{REG}}_\theta(z) = \|z - z^\star\|^2$ (直接衡量 $z$ 和最优解 $z^\star$ 的距离). 也就是如果真实的 groudn truth 是已知的, 则直接让 $z^{(K)}$ 接近 $z^\star$.
  - 相当于一个全局信息.


在确定了 loss 的具体形式后, 我们总的的训练目标就是:
$$
\min_w \mathbb{E}_{\theta\sim Q}[\ell_\theta(T_\theta^K(h_w(\theta)))]
$$
- 其中 $\mathcal{Q}$ 代表了我们关注的任务的总体分布, 但是未知的. $\theta \sim \mathcal{Q}$ 代表了我们从这个分布中采样得到的具体任务实例.
- 在实际中, 我们只有有限的样本 $\{\theta_i\}_{i=1}^N \sim \mathcal{Q}$, 因此我们只能通过 empirical risk 来近似这个期望.



#### Testing

假设我们已经通过前面的训练阶段, 学习到了一个神经网络 $h_w$, 那么在测试阶段, 给定一个新的任务 $\theta'$, 我们就可以通过 $h_w(\theta')$ 来得到一个初始点, 然后通过 $T_{\theta'}$ 的迭代来得到最终的结果. 

与训练不同, 
- 测试阶段, 我们可以迭代次数 $t\neq K$, 也就是尽管在训练阶段我们是以 $K$ 步迭代后的结果为目标来训练的, 但是在测试阶段我们可以选择任意的迭代次数 $t$ 来得到最终的结果, 这一方面允许我们在测试阶段进行更多的迭代来进一步优化结果, 另一方面也在观测其迭代泛化能力. 
- 在测试时, 我们将统一考虑 fixed-point residual 作为评估指标, 最终感兴趣的的是如下的 risk:
    $$
    R^{(t)}(w) =    \mathbb{E}_{\theta\sim Q}[\ell_\theta(T_\theta^t(h_w(\theta)))]
    $$
    但由于实际只有有限的样本 $\{\theta_i\}_{i=1}^N \sim \mathcal{Q}$, 因此我们只能计算 empirical risk 作为估计:
    $$
    \hat{R}^{(t)}(w) = \frac{1}{N}\sum_{i=1}^N \ell_{\theta_i}(T_{\theta_i}^t(h_w(\theta_i)))
    $$


## Related Work

### Learning warm starts

- 核心: 不改变迭代算法本身, 只学一个从 problem instance 到一个好的 initialization 的映射. 
- 现有算法问题:
  - 没有考虑后续 downstream 的solver的性能表现
  - 缺少 generalization guarantee
  - 本文针对的是 general 的 fixed-point problem, 而不是特定的 optimization problem

### Learning Algorithm Steps for  convex optimization

- 核心: 直接学习 solver 的迭代步骤, 更新公式或超参数
- 如: 用 RL 学习 QP 方法超参数, 直接学固定点问题的加速更新;
- 关键风险
  - 收敛保证困难
  - 缺少 generalization guarantee
  - 工程上缺少和主流优化库的兼容性

### Meta-learning

需要承认在面对许多 ML 任务时, 二者有所重合. 但是, 其核心区别在于:

- MAML: constant  initialization shared over tasks
- 本文: $h_w$ 是一个 task-conditioned 的 predictor, 可以根据不同的 $\theta$ 来输出不同的 initialization

## PAC Theory Analysis

核心思想: 其学习到的 warm-starting 网络, **在没见过的新问题的实例上**, 经过 $t$ 步迭代之后, fixed-point residual 的表现不会比训练集上差太多, with high probability.


### Preliminaries

***Marginal Fixed-point Residual***: 其关注的是对于一个 warm-start 初始化 $z = h_w(\theta)$, 若其收到了一个微小扰动 $\Delta$ (where $\|\Delta\|_2 \leq \gamma$), 其在经过 $t$ 步 Fixed-point 迭代之后的结果 $T_\theta^t(z+\Delta)$ 的 fixed-point residual 的表现的 worst case 会有多差:
$$
g^t_{\gamma,\theta} = \max_{\|\Delta\|_2 \leq \gamma} \ell_\theta^{\text{FP}}(T_\theta^t(z+\Delta))
$$
- 显然, 当 $\gamma=0$ 的时候, 就退化成了我们之前定义的 $R^{(t)}(h_w)$, 也就是没有扰动的情况.
- 这个指标相当于一个关于我们关注的 fixed-point loss 的局部 robustness 的指标. 后续可能如下几个称呼会偶有混用: marginal risk, marginal fixed-point residual, robust risk, worst-case risk 等.

还可以同理定义这个 $g^t_{\gamma,\theta}$ 的 总体风险和经验风险:
$$
R^t_\gamma(w) = \mathbb{E}_{\theta\sim Q}[g^t_{\gamma,\theta}(z)], \quad \hat{R}^t_\gamma(w) = \frac{1}{N}\sum_{i=1}^N g^t_{\gamma,\theta_i} (z)
$$
- 分别表示这个 warm-start 网络, 在总体分布或经验样本上, 在 $\gamma$-perturbation 下的 worst-case fixed-point residual 的表现.

***PAC-Bayes: McAllester's Bound***: PAC-Bayes 的分析对象是对一个随机化的 predictor 的 generalization bound. 因此需要构造一个随机的 predictor:
- 已知一个 deterministic predictor $h_w$, 此时可以给这个 predictor (的模型参数权重) 添加一个随机扰动 $u$, 来构造一个随机 predictor $h_{w+u}$, 其中 $u$ 是一个随机变量, 例如可以是一个 isotropic Gaussian noise.

此时, 对于 $w+u$ 这个随机 predictor, 其总体风险可以被如下的 bound 来界定:
$$
\underbrace{\mathbb{E}_u[R_\gamma^t(w+u)]}_{\text{总体随机扰动marginal风险}}
 \leq 
 \underbrace{\mathbb{E}_u[\hat{R}_\gamma^t(w+u)]}_{\text{经验随机扰动marginal风险}} + 
 \underbrace{2C_\gamma(t)\sqrt{2\frac{\text{KL}(w+u\|\pi) + \log 2N/\delta}{N-1}}}_{\text{复杂度惩罚}}
$$
其以高概率 $1-\delta$ 成立. 其中:
- $R_\gamma^t(w+u)$:  warm-start 网络 $h_w$ 在添加了随机扰动 $u$ 之后, 在 $\gamma$-perturbation 下的 worst-case fixed-point residual 的总体风险. 即随机化后的邻域 robust 风险.
- $\hat{R}_\gamma^t(w+u)$: 随机化 predictor 在训练集上的邻域 robust 风险.
- $C_\gamma(t) = \max_{\theta \in \Theta, w \in W} g^t_{\gamma,\theta}(h_w(\theta))$: 这个指标相当于对于所有的 $\theta$ 和 $w$ (即所有可能的任务实例和 warm-start 网络), 在 $\gamma$-perturbation 下的 worst-case fixed-point residual 的上界. 该量作为一个常数用来平衡总体的损失量级.
  - 同时, 为了使得这个量级是一个常数而不发散, 要约束 predictor 的的输出 $z = h_w(\theta)$ 的范围. 具体地, 文中限制 predictor 要求其输出距离 fix-point 的集合 $\text{Fix}(T_\theta) = \{z: z = T_\theta(z)\}$ 的距离不超过一个常数 $D$, 即:
    $$
    \operatorname{dist}(h_w(\theta), \text{Fix}(T_\theta)) \leq D, \quad \forall \theta \in \Theta, w \in W
    $$
- $\text{KL}(w+u\|\pi)$: Kullback-Leibler 散度. 其中 $\pi$ 是一个独立于训练数据的先验分布. KL 散度用来衡量随机 predictor 和先验分布之间的差距, 其越小, 代表随机 predictor 的复杂度越低, 惩罚项越小. 类似于保证, 没有为了了拟合训练数据而引入过于复杂的模型, 将分布扭曲的过于离谱.
  - 具体地, 文中假设训练后得到的权重为 $w \in \mathbb{R}^d$, 选择 $\pi \sim \mathcal{N}(0, \sigma^2 I_d)$, $u \sim \mathcal{N}(0, \sigma^2 I_d)$, 则 $w+u \sim \mathcal{N}(w, \sigma^2 I_d)$, 此时 KL 散度可以计算为 $\text{KL}(w+u\|\pi) = \frac{\|w\|^2}{2\sigma^2}$.
  - 这里, $\pi$ 的选取并不意味着真实网络权重需要服从这个先验分布, 只是为了分析的方便, 其需要满足下面几个条件. 而高斯分布恰好是一个非常标准的选择, 起满足下面的要求, 并且是一个各项同性, 中心为0的分布, 使得分析更为简洁.
    - 必须与数据无关
    - KL 散度 $\text{KL}(w+u\|\pi)$ 需要是可计算的, 以便于分析.
    - 要控制 predictor 的复杂度, 以便于得到一个有意义的 bound.
- $\delta$: 置信水平

### Generalization Bound

***Lemma 1***: 其核心思想是, 对于 predictor  $h_w$ 的随机化 $h_{w+u}$, 只要扰动项 $u$ 满足一些条件 (以保证扰动不是过于剧烈), 则原 predictor $h_w$ 的总体风险 $R_\gamma^t(w)$ 可以被随机 predictor 的总体风险 $\hat{R}_\gamma^t(w+u)$ 来 bound 住:
$$
R^t(w)
\le
\hat R_\gamma^t(w)
+
4C_{\gamma/2}(t)
\sqrt{
\frac{\mathrm{KL}(w+u\|\pi)+\log(6N/\delta)}{N-1}
}.
$$

具体地, 给定
- Warm-start predictor $h_w: \Theta \to \mathbb{R}^p$, 并且对所有的 $\theta \in \Theta$, 都满足
  $$
  g^t_{\gamma/2,\theta}(h_w(\theta)) \leq C_{\gamma/2}(t) 
  $$
  - 即我们讨论的是一个正常的, 能够保证在 $\gamma/2$-perturbation 下的 worst-case fixed-point residual 不超过一个常数 $C_{\gamma/2}(t)$ 的 predictor. 这是一个基本假定. 

- 一个与数据无关的先验分布 $\pi$ 

若对于随机扰动 $u$, 其满足如下的条件:
- 对任意 $\delta, \gamma > 0$, 在训练集 $\{\theta_i\}_{i=1}^N \sim \mathcal{Q}$ 上, 只要
  $$
  \mathbb{P}(\max_{\theta \in \Theta} \|h_{w+u}(\theta) - h_w(\theta)\|_2 \leq \gamma/2) \geq 1/2
  $$
  - 即, 即使我们施加 perturbation $u$ 之后, 这个随机 predictor $h_{w+u}$ 和原 predictor $h_w$ 在所有的 $\theta$ 上的输出都不会相差太大 (不超过 $\gamma/2$), 并且至少有一半的概率满足这个条件. 这里暗含着如下三个层次的含义:
    - 该控制在整个输入空间 $\Theta$ 上都成立, 这保证了整体的泛化能力.
    - 这个控制是以概率的形式来描述的, 我们至少要求有一半的扰动是"mild"的好的扰动即可在期望意义上得到一个有意义的 bound.
    - 这个控制的程度是 $\gamma/2$, 这是说如果扰动的 predictor $h_{w+u}$ 在原 predictor $h_w$ 的一个 $\gamma/2$ 的球内, 则这个扰动的 predictor 的风险就可以被原 predictor 的 marginal 风险控制住.

则将有高概率 $1-\delta$ 成立如下的 bound:
$$
R^t(w) \leq \hat{R}_\gamma^t(w) + 4C_{\gamma/2}(t)\sqrt{\frac{\text{KL}(w+u\|\pi) + \log 6N/\delta}{N-1}}
$$
- LHS 表示我们真正感兴趣的, 给定一个 deterministic predictor $h_w$, 其在总体分布上的 fixed-point residual 的风险.
- RHS 的第一项 $\hat{R}_\gamma^t(w)$ 是一个经验的 marginal 风险, 表示的是在训练集上, 给定一个 $\gamma$-邻域的最坏风险(之平均).
- RHS 的第二项仍然是一个复杂度惩罚项, 并且其系数的不同刻画了由于从 random predictor 转化为 deterministic predictor 所带来的额外常数代价.

***Theorem 2:*** 其形式上与 Lemma 1 非常类似. 其相当于是具体化了 Lemma 1 中的一些假设, 得到了一个更为具体的 bound (体现在复杂度惩罚项的具体形式上). 
$$
R^{(t)}(w) \leq \hat{R}_\gamma^{(t)}(w) + \text{complexity penalty}
$$
具体地, 施加的假设如下:
- predictor 是一个 $L$ 层的 ReLU MLP
- 给定了一个具体的 Gaussian 的 prior 和 perturbation, 并证明其确实满足 Lemma 1 中的条件.
- 给出了具体的 KL 的形式.

略去细节, 整体而言我们能得到如下性质:
- 当总体迭代次数 $t \to \infty$ 时, generalization gap (即这里的 complexity penalty) 会趋于0, 这表明当我们在测试阶段进行足够多的迭代时, 其 generalization 性能会得到保证.
  - 即, 因为这里是一个 warm start, 因此在后面如果进行迭代, 则会有改进的 test gap.
- 当样本量 $N \to \infty$ 时, generalization gap 也会趋于0
  - 其符合统计学习直觉, 因为当我们有足够多的样本时, 我们就能够更好地估计总体分布, 从而使得训练得到的 predictor 能够更好地泛化到新的任务实例上. 简单讲, 当样本量足够多, 抽样误差消失, 唯一的 gap 就是 marginal 经验误差, 和population risk 之间的 gap.
- 若令 $h_w(\theta) \equiv 0$, 则退化为一个普通的 cold start, 此时的分析同样仍然成立.

### Bounding the empirical marginal risk

前面的定理已经给出了一个 generalization bound, 但是其中的 RHS 仍然是一个 $\gamma$-邻域的 worst-case fixed-point residual 的经验风险. 这里的任务是将这个 marginal 的成分进一步控制住. 具体而言, 还有待分析的内容有两项:
- $\hat{R}_\gamma^{(t)}(w)$ 
- $C_{\gamma/2}(t)$

这里将使用 fixed-point operator 的一些性质来对其进行分析. 

- 假设 fixed-point operator $T_\theta$ 是 non-expansive 的, 即
  $$
  \|T_\theta(x) - T_\theta(y)\| \leq \|x-y\|, \quad \forall x,y \in \mathbb{R}^p
  $$
- 记迭代点到 fixed-point 的距离为 $r_\theta(z) = \operatorname{dist}(z, \text{Fix}(T_\theta))$, 记在 $\gamma$-邻域的 worst-case 距离为
  $$
  f^t_{\gamma,\theta}(z) = \max_{\|\Delta\|_2 \leq \gamma} r_\theta(T_\theta^t(z+\Delta))
  $$

对于 non-expansive 的 fixed-point operator, 可以证明, 若迭代点 $z$ 的距离 fixed-point 集很近, 那么其 fixed-point residual 也不会太大:
$$
\ell_\theta^{\text{FP}}(z) \leq 2r_\theta(z).
$$
- 这使得我们对 fixed-point residual 的分析可以转化为对迭代点到 fixed-point 集的距离的分析.

再进一步, 根据 $T_\theta$ 的具体收缩性质, 其还可以细分为如下三层:
- **Contractive**: 存在一个 $\beta < 1$, 使得 $\|T_\theta(x) - T_\theta(y)\| \leq \beta \|x-y\|$. 即任意两点经过 $T_\theta$ 的映射之后的距离都会缩小一个常数 $\beta$. 
- **Linearly convergent**: 不要求任意两点距离都缩小, 只要求到 fixed-point set 的距离线性收缩, 即 $\operatorname{dist}(T_\theta(z), \text{Fix}(T_\theta)) \leq \beta \operatorname{dist}(z, \text{Fix}(T_\theta))$, 对任意 $z$ 和 $\theta$ 都成立.
- **Averaged**: 存在一个 $\alpha \in (0,1)$, 以及一个恒等映射 $I$, 使得 $T_\theta = (1-\alpha)I + \alpha R_\theta$, 其中 $R_\theta$ 是一个 non-expansive 的映射. 这是一类很 weak  的假设, 其得到的结论是 sublinear 的. 其在文中的作用是扩大理论的覆盖范围.

其中, $\text{Contractive} \implies \text{Linearly convergent}$, 而 average 隶属于这两者之外, 无直接从属关系.

最终, 三种情况分别得到的结论如下:
- $\beta$-contractive: 
  $$
  R^t(w)
  \le
  \hat R^t(w) + 2\beta^t\gamma + \text{generalization penalty},
  $$
  - 其中, penalty 也包含 $\beta^t(D+\gamma/2)$ 的缩减因子
  - 这说明, 只要算子 $T_\theta$ 是 contractive 的, 则其 generalization gap 会随着迭代次数的增加而指数级地缩小, 这表明在测试阶段进行更多的迭代会带来更好的泛化性能.

- Linearly convergent:
    扰动后的最坏误差, 随着迭代次数的增加, 以一个线性的速率缩小:
    $$
    f_{\gamma,\theta}^{t+1}(z)\le \beta f_{\gamma,\theta}^{t}(z),\quad g_{\gamma,\theta}^{t}(z)\le 2f_{\gamma,\theta}^{t}(z),
    $$
    $$
    f_{\gamma,\theta}^{t}(z)\le r_\theta(T_\theta^t(z))+2\gamma.
    $$
    尺度项也可以以 $\beta^t$ 的速率缩小:
    $$
    C_{\gamma/2}(t)\le 2\beta^t(D+\gamma/2).
    $$

- Averaged: 扰动后的最坏误差会以约 $\mathcal{O}(\frac{1}{\sqrt{t}})$ 的速率缩小.

## Choosing the right computational architecture

这里具体回答了如下问题:
- 训练时网络优化了 $k$ 步, 而测试的时候用的是 $t$ 步 ($t>k$), 则测试时的 fixed-point residual loss 是否可以被训练的 loss 来控制住? 
- 训练时使用 fixed-point residual loss / regression loss, 其是否会对测试时的 fixed-point residual loss 的表现有影响?

其结果如下:
- Contractive: 
  $$
  \frac{\ell_{\theta}^{\mathrm{fp}}(T_\theta^t(z))}
  {\ell_{\theta}^{\mathrm{fp}}(T_\theta^k(z))}
  \le \beta^{\,t-k},
  \qquad
  \frac{\ell_{\theta}^{\mathrm{fp}}(T_\theta^t(z))}
  {\ell_{\theta}^{\mathrm{reg}}(T_\theta^k(z))}
  \le 2\beta^{\,t-k}.
  $$

- Linearly convergent:
  $$
  \frac{\ell_{\theta}^{\mathrm{fp}}(T_\theta^t(z))}
  {\ell_{\theta}^{\mathrm{fp}}(T_\theta^k(z))}
  \le 1,
  \qquad
  \frac{\ell_{\theta}^{\mathrm{fp}}(T_\theta^t(z))}
  {\ell_{\theta}^{\mathrm{reg}}(T_\theta^k(z))}
  \le 2\beta^{\,t-k}.
  $$


- Averaged:
  $$
  \frac{\ell_{\theta}^{\mathrm{fp}}(T_\theta^t(z))}
  {\ell_{\theta}^{\mathrm{fp}}(T_\theta^k(z))}
  \le 1,
  \qquad
  \frac{\ell_{\theta}^{\mathrm{fp}}(T_\theta^t(z))}
  {\ell_{\theta}^{\mathrm{reg}}(T_\theta^k(z))}
  \le
  \sqrt{\frac{\alpha}{(1-\alpha)(t-k+1)}}.
  $$

以及, 到底该选择 residual loss 还是 regression loss 来训练?
- 在非 contractive 的情况下, 两种 loss 都不错;
- 在 contractive 的情况下, 选择 regression loss 更好 (从未来更多迭代本身的角度看)

二者的优劣比较如下:
- Fixed-point residual loss 优点: 不要求解 ground truth; 测试和训练的目标一致
- Regression loss 优点: 利用了 ground truth 的全局信息, 信号更强; 在非 contractive 的情况下, regression loss 能够给出显式的改进因子, 而 residual loss 只能给出不会更差的保证.


## Experiments

- Learned warm-starting 几乎在所有任务上都显著优于 cold start
- $k=0$ 是危险的.
  - 相当于只学习一个看起来不错的 initialization, 但是完全不引入后续的迭代过程. 
  - 这时实验展示, 其可能看起来在 $t=0$ 的时候表现不错, 但是在后续的迭代过程中会迅速退化, 甚至比 cold start 更差. 
- 训练时的 $k$ 的选择也有一定影响. 是一个和实验相关的超参数. 
- regression loss 的表现通常优于 fixed-point residual loss.


## Future Work

- 目前主要还是使用的 MLP 来实现 $h_w$, 未来可以考虑一些更适合结构化数据的架构
- 扩展到 non-convex optimization
- 扩展到更大规模的优化问题


