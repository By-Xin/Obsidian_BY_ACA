# Stochastic Gradient Descent

> [!info] References
> - Lecture: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/
> - Reading: 最优化: 建模、算法与理论, 刘浩洋等, 2.7 小节.

## Stochastic Algorithm

为方便讨论, 这里给出一个随机优化在有监督学习中的典型应用. 

- 有输入特征 $X \in \mathbb{R}^d$ 和输出标签 $Y \in \mathbb{R}$, 且 $(X, Y) \sim \mathcal{P}$. 目标是学习一个函数 $\hat{\phi}: \mathbb{R}^d \to \mathbb{R}$ 使得 $\hat{\phi}(X)$ 能够很好地预测 $Y$. 此外, 往往对 $\phi$ 的假设空间 $\mathcal{H}$ 进行限制以缩小搜索范围, 参数化为 $\phi(\cdot; \theta)$, 其中 $\theta \in \mathbb{R}^p$ 是模型参数. 通过引入一个损失函数 $L: \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ 来衡量预测误差, 以及正则项 $h: \mathbb{R}^d \to \mathbb{R}$ 来保证解的某些性质, 可以将学习问题表述为如下优化问题:
  
    $$
    \min_{\theta \in \mathbb{R}^p} \mathbb{E}_{(X, Y) \sim \mathcal{P}}[L(\phi(X; \theta), Y)]+ h(\theta).
    $$

- 在实践中, 我们通常只能获得一个有限的训练数据集 $\{(x_i, y_i)\}_{i=1}^N$ 来近似 $\mathcal{P}$.  因此, 优化问题可以表述为:
  
$$
\min_{\theta \in \mathbb{R}^p} \frac{1}{N} \sum_{i=1}^N L(\phi(x_i; \theta), y_i)+ h(\theta):= f(\theta).
$$

在下面的讨论中, 为表述习惯, 我们将优化问题重新表述为如下形式, 并且暂时假设 $f_i$ 为可微且凸的函数.
$$
\min_{x\in \mathbb{R}^n} f(x):= \frac{1}{N} \sum_{i=1}^N f_i(x),
$$

## Stochastic Gradient Descent

### SGD and Mini-batch SGD

考虑原始的梯度下降算法:
$$
x_{k+1} = x_k - \alpha_k \nabla f(x_k) = x_k - \alpha_k \frac{1}{N} \sum_{i=1}^N \nabla f_i(x_k).
$$
- 由于经验风险为 $f(x) = \frac{1}{N} \sum_{i=1}^N f_i(x)$, 则全梯度就是 $\nabla f(x) = \nabla \frac{1}{N} \sum_{i=1}^N f_i(x) = \frac{1}{N} \sum_{i=1}^N \nabla f_i(x)$. 计算全梯度需要遍历整个数据集, 当 $N$ 很大时, 计算成本非常高.

SGD 随机梯度法则通过在每次迭代中随机选择一个样本 $i_k$ 来近似梯度:
$$x_{k+1} = x_k - \alpha_k \nabla f_{i_k}(x_k).$$
- 这里, $i_k$ 是从 $\{1, 2, \ldots, N\}$ 中随机等可能抽样得到的索引. 用单个样本的梯度近似整个数据集的梯度, 大大降低了每次迭代的计算成本.
- 该操作的合理性在于, 当给定 $x_k$ 时, $\nabla f_{i_k}(x_k)$ 是 $\nabla f(x_k)$ 的无偏估计, 即 $\mathbb{E}_{i_k}[\nabla f_{i_k}(x_k) \mid x_k] = \nabla f(x_k)$.
    - *Proof*: 由于 $i_k$ 是从 $\{1, 2, \ldots, N\}$ 中随机等可能抽样得到的索引, 则有: 
        $$
        \begin{aligned}
        \mathbb{E}_{i_k}[\nabla f_{i_k}(x_k) \mid x_k] &= \sum_{i=1}^N P(i_k = i) \nabla f_i(x_k) \\
        &= \frac{1}{N} \sum_{i=1}^N \nabla f_i(x_k) \\
        &= \nabla f(x_k).
        \end{aligned}
        $$

不过只选取一个样本的梯度会引入较大的方差, 导致 SGD 的收敛速度较慢. 为了平衡计算效率和收敛速度, 一般使用 mini-batch SGD, 即在每次迭代中随机选择一个小批量的样本 $\mathcal{I}_k \subset \{1, 2, \ldots, N\}$ 来近似梯度:
$$x_{k+1} = x_k - \alpha_k \frac{1}{|\mathcal{I}_k|} \sum_{i \in \mathcal{I}_k} \nabla f_i(x_k).$$

此外, 对于不可微的优化问题, 也可以考虑使用随机次梯度法则:
$$x_{k+1} = x_k - \alpha_k g_{i_k}(x_k),$$
其中 $g_{i_k}(x_k)$ 是 $f_{i_k}$ 在 $x_k$ 处的一个次梯度.

### Variants of SGD

SGD 方法有一系列的变体, 旨在提高收敛速度和稳定性. 

#### Momentum SGD

传统的 SGD 和 GD 类似, 在较为病态的优化问题中可能会出现收敛缓慢的情况. Momentum 方法通过引入一个动量项来加速收敛, 并在高曲率或噪声场景中提供更为有效的更新. 

Momentum SGD 的更新规则如下:

- 给定动量参数 $\mu_k \in [0, 1)$ (通常取 $\mu_k \geq 0.5$), 初始化动量向量 $v_0 = 0$.
- 在迭代 $k = 0, 1, 2, \ldots$ 中, 更新动量和参数:
  - 抽取随机样本索引 $i_k$.
  - $v_{k+1} = \mu_k v_k - \alpha_k \nabla f_{i_k}(x_k)$
  - $x_{k+1} = x_k + v_{k+1}$

此处, $v_k$ 为动量向量, 用于累积历史梯度信息, 从而在更新参数时考虑历史梯度的方向和大小. $\mu_k$ 控制了动量的衰减程度, 较大的 $\mu_k$ 带来较大的惯性, 保留了更多的历史梯度信息, 但可能导致震荡; 较小的 $\mu_k$ 则更快地响应当前梯度, 但可能失去加速效果.


#### Nesterov Accelerated Gradient (NAG)

Nesterov Accelerated Gradient (NAG) 是 Momentum 方法的一种改进, 通过在计算梯度时提前考虑动量的影响来进一步加速收敛. NAG 的更新规则如下:
- 给定动量参数 $\mu_k = \frac{k-1}{k+2}$, 步长 $\alpha_k$ 固定或由线搜索确定.
- 在迭代 $k = 0, 1, 2, \ldots$ 中:
  - $y_{k+1} = x_k + \mu_k (x_k - x_{k-1})$  
  - $x_{k+1} = y_{k+1} - \alpha_k \nabla f_{i_k}(y_{k+1})$

NAG 方法等价于如下的 Momentum 更新:
- $v_{k+1} = \mu_k v_k - \alpha_k \nabla f_{i_k}(x_k + \mu_k v_k)$
- $x_{k+1} = x_k + v_{k+1}$

相比于之前的 Momentum 方法, NAG 方法先计算一个“预更新”位置 $x_k + \mu_k v_k$, 然后在该位置计算梯度, 从而更准确地调整更新方向. 


#### AdaGrad

AdaGrad 是一种自适应学习率方法, 通过根据历史梯度的累积调整每个参数的学习率来提高收敛速度. 在普通的 SGD 中, 每个参数的更新步长是相同的; 而 AdaGrad 将考虑梯度的每个分量的历史累计情况, 来调整每个参数的学习率.

具体而言, 对于梯度 $g_k := \nabla f_{i_k}(x_k) \in \mathbb{R}^n$, 定义从迭代 $t=0$ 到 $k$ 的梯度平方和为:

$$
G_k = \sum_{t=0}^k (g_t \odot g_t),
$$

其中 $\odot$ 表示元素级乘积.  $G_k$ 是一个向量, 其第 $j$ 个分量 $G_{k,j}$ 表示从迭代 $0$ 到 $k$ 的第 $j$ 个参数的梯度平方和. 
- 若某分量的数值较大, 则说明该参数在之前的迭代中经历了较大的梯度累积, 历史变化较为剧烈, 因此需要较小的学习率来稳定更新
- 反之, 若某分量的数值较小, 则说明该参数在之前的迭代中经历了较小的梯度累积, 历史变化较为平稳, 可以使用较大的学习率来加速更新.

AdaGrad 的更新规则为:
- 给定初始学习率 $\alpha > 0$ 和一个小的常数 $\epsilon > 0$ (用于数值稳定), 初始化 $G_0 = \boldsymbol{0}$, 以及起点 $x_0$.
- 在迭代 $k = 0, 1, 2, \ldots$ 中:
  - 抽取随机样本索引 $i_k$ 并计算随机梯度 $g_k = \nabla f_{i_k}(x_k)$.
  - $x_{k+1} = x_k -  \dfrac{\alpha}{\sqrt{G_k + \epsilon \boldsymbol{1}_n}} \odot g_k$.
  - $G_{k+1} = G_k + g_{k+1} \odot g_{k+1}$.

AdaGrad 也可以当作是一种介于一阶方法和二阶方法之间的优化算法. 考虑 $f(x)$ 在 $x_k$ 处的二阶 Taylor 展开:
$$
f(x) \approx f(x_k) + \nabla f(x_k)^\top (x - x_k) + \frac{1}{2} (x - x_k)^\top B_k (x - x_k).
$$
根据 $B_k$ 的不同选择, 可以得到不同的优化算法. 而 AdaGrad 即为令 $B_k = {\alpha}^{-1} \text{diag}(\sqrt{G_k + \epsilon \boldsymbol{1}_n})$ 的特殊情况. 

#### RMSProp

RMSProp (Root Mean Square Propagation) 是 AdaGrad 的一种改进, 该问题在非凸问题上的表现可能更好. 

- 注意到 AdaGrad 的更新步长 $\alpha / \sqrt{G_k + \epsilon \boldsymbol{1}_n}$ 中的 $G_k$ 是从迭代 $0$ 到 $k$ 的梯度平方和, 随着迭代次数的增加, $G_k$ 会不断增大, 导致学习率逐渐单调下降, 最终趋近于零, 从而使得算法在后期的迭代中几乎没有更新. 
- RMSProp 通过引入一个衰减因子 $\rho \in (0, 1)$ 来计算梯度平方的指数加权移动平均, 从而避免了学习率过快下降的问题.

具体地, 将 AdaGrad 中的 $G_k$ 累积项改进为:
$$
M_k = \rho M_{k-1} + (1 - \rho) (g_{k+1} \odot g_{k+1}),
$$

从而得到 RMSProp 的更新规则:
- 给定初始学习率 $\alpha > 0$, 衰减因子 $\rho \in (0, 1)$ (一般取 $\rho=0.9$) 和一个小的常数 $\epsilon > 0$, 初始化 $M_0 = \boldsymbol{0}$, 以及迭代起点 $x_0$.
- 在迭代 $k = 0, 1, 2, \ldots$ 中:
  - 抽取并计算随机梯度 $g_k = \nabla f_{i_k}(x_k)$.
  - 更新 $x_{k+1} = x_k -  \dfrac{\alpha}{\sqrt{M_k + \epsilon \boldsymbol{1}_n}} \odot g_k$.
  - 计算 $M_{k+1} = \rho M_k + (1 - \rho) (g_{k+1} \odot g_{k+1})$.

其中 $\sqrt{M_k + \epsilon \boldsymbol{1}_n}$ 即为所谓的 RMS (Root Mean Square).


#### AdaDelta

AdaDelta 是 RMSProp 的一种改进, 旨在进一步提高优化算法的适应性和鲁棒性. 其和 RMSProp 一样需要维护 $M_k$ 以指数加权移动平均的方式来计算梯度平方的平均值. AdaDelta 在此基础上, 引入了一个新的累积项 $D_k$ 来跟踪参数更新的平方和, 从而将 $\alpha$ 替换为一个自适应的学习率:

- 累积梯度为: $M_k = \rho M_{k-1} + (1 - \rho) (g_{k+1} \odot g_{k+1})$.
- 累积更新为: $D_k = \rho D_{k-1} + (1 - \rho) (\Delta x_k \odot \Delta x_k)$,其中 $\Delta x_k = x_{k+1} - x_k$ 是第 $k$ 次迭代时即将作用在 $x_k$ 上的更新幅度. 

其具体更新规则为:
- 给定衰减因子 $\rho \in (0, 1)$ 和一个小的常数 $\epsilon > 0$, 初始化 $M_0 = \boldsymbol{0}$ 和 $D_0 = \boldsymbol{0}$.
- 在迭代 $k = 0, 1, 2, \ldots$ 中:  
  - 抽取并计算随机梯度 $g_k = \nabla f_{i_k}(x_k)$.
  - 计算累积梯度: $M_{k+1} = \rho M_k + (1 - \rho) (g_{k}\odot g_{k})$.
  - 更新参数: $x_{k+1} = x_k - \dfrac{\sqrt{D_k + \epsilon \boldsymbol{1}_n}}{\sqrt{M_{k+1} + \epsilon \boldsymbol{1}_n}} \odot g_k := x_k + \Delta x_k$.
  - 计算累积更新: $D_{k+1} = \rho D_k + (1 - \rho) ((x_{k+1} - x_k) \odot (x_{k+1} - x_k)):= \rho D_k + (1 - \rho) (\Delta x_k \odot \Delta x_k)$.


#### Adam

Adam (Adaptive Moment Estimation) 本质上是包含了 Momentum 和 RMSProp 的优化算法, 通过同时考虑梯度的一阶矩和二阶矩来调整每个参数的学习率. Adam 的优势在于经过偏差校正后, 每一次迭代步长有确定的范围, 从而使得参数更新更加平稳. 

具体而言, Adam 进行了如下的调整:
- 从之前的梯度作为更新方向改为对梯度的历史指数加权累计: $S_k = \rho_1 S_{k-1} + (1 - \rho_1) g_k$, 其中 $\rho_1$ 是一阶矩的衰减率, 通常取 $\rho_1 = 0.9$.
- 同时其也会记录梯度的二阶矩: $M_k = \rho_2 M_{k-1} + (1 - \rho_2) (g_k \odot g_k)$, 其中 $\rho_2$ 是二阶矩的衰减率, 通常取 $\rho_2 = 0.999$.
- 在正式更新参数之前, 还要额外进行了偏差校正:
  - $\hat{S}_k = \frac{S_k}{1 - \rho_1^k}$, $\hat{M}_k = \frac{M_k}{1 - \rho_2^k}$. 其中 $\rho_1^k$ 和 $\rho_2^k$ 分别是 $\rho_1$ 和 $\rho_2$ 的 $k$ 次幂, 用于校正初始时刻的偏差.

Adam 的更新规则为:
- 给定初始学习率 $\alpha > 0$, 衰减率 $\rho_1, \rho_2 \in (0, 1)$ 和一个小的常数 $\epsilon > 0$, 初始化 $S_0 = \boldsymbol{0}$ 和 $M_0 = \boldsymbol{0}$, 及迭代起点 $x_0$.
- 在迭代 $k = 0, 1, 2, \ldots$ 中:
  - 抽取并计算随机梯度 $g_k = \nabla f_{i_k}(x_k)$.
  - 更新一阶矩: $S_{k+1} = \rho_1 S_k + (1 - \rho_1) g_k$.
  - 更新二阶矩: $M_{k+1} = \rho_2 M_k + (1 - \rho_2) (g_k \odot g_k)$.
  - 进行偏差校正: $\hat{S}_{k+1} = \frac{S_{k+1}}{1 - \rho_1^{k+1}}$, $\hat{M}_{k+1} = \frac{M_{k+1}}{1 - \rho_2^{k+1}}$.
  - 更新参数: $x_{k+1} = x_k - \dfrac{\alpha}{\sqrt{\hat{M}_{k+1} + \epsilon \boldsymbol{1}_n}} \odot \hat{S}_{k+1}$.

## Convergence Analysis of SGD

### Convergence under General Convexity

首先讨论在一般凸函数上的收敛性. 有如下假设:
- 每个 $f_i$ 都是闭凸函数, 存在 subgradient. 
- 随机次梯度的二阶矩有界, 即存在常数 $G > 0$ 使得 $\mathbb{E}[\|g_{i_k}(x_k)\|^2] \leq G^2<\infty$ 对于所有 $k$ 都成立, 其中 $g_{i_k}(x_k)\in\partial f_{i_k}(x_k)$ 是随机样本 $i_k$ 处的一个次梯度.

### Convergence under Strong Convexity