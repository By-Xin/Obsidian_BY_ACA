# Stochastic Gradient Descent

> [!info] References
> - Lecture: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/
> - Reading: 最优化: 建模、算法与理论, 刘浩洋等, 2.7 小节.

## Stochastic Algorithm

为方便讨论, 这里给出一个随机优化在有监督学习中的典型应用. 

- 有输入特征 $X \in \mathbb{R}^d$ 和输出标签 $Y \in \mathbb{R}$, 且 $(X, Y) \sim \mathcal{P}$. 目标是学习一个函数 $\hat{\phi}: \mathbb{R}^d \to \mathbb{R}$ 使得 $\hat{\phi}(X)$ 能够很好地预测 $Y$. 此外, 往往对 $\phi$ 的假设空间 $\mathcal{H}$ 进行限制以缩小搜索范围, 参数化为 $\phi(\cdot; \theta)$, 其中 $\theta \in \mathbb{R}^p$ 是模型参数. 通过引入一个损失函数 $L: \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ 来衡量预测误差, 以及正则项 $h: \mathbb{R}^p \to \mathbb{R}$ 来保证解的某些性质, 可以将学习问题表述为如下优化问题:
  
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
- 此处可将正则项并入每个分量函数(例如 $f_i(x)=\ell_i(x)+h(x)$), 或暂时令 $h\equiv 0$ 以专注讨论随机梯度估计本身; 后续结论不依赖具体拆分方式.

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
- 上述 $\mu_k$ 序列是确定性加速理论中的一种典型选择; 在深度学习随机训练中更常见的是固定动量系数 (如 $\mu\approx 0.9$) 的 Nesterov momentum.
- 在迭代 $k = 0, 1, 2, \ldots$ 中:
  - 初始化时可令 $x_{-1}=x_0$ (或等价地令初始动量为 $0$), 以保证首步定义良好.
  - $y_{k+1} = x_k + \mu_k (x_k - x_{k-1})$  
  - $x_{k+1} = y_{k+1} - \alpha_k \nabla f_{i_k}(y_{k+1})$

NAG 方法等价于如下的 Momentum 更新:
- $v_{k+1} = \mu_k v_k - \alpha_k \nabla f_{i_k}(x_k + \mu_k v_k)$
- $x_{k+1} = x_k + v_{k+1}$

相比于之前的 Momentum 方法, NAG 方法先计算一个“预更新”位置 $x_k + \mu_k v_k$, 然后在该位置计算梯度, 从而更准确地调整更新方向. 


#### AdaGrad

AdaGrad 是一种自适应学习率方法, 通过根据历史梯度的累积调整每个参数的学习率来提高收敛速度. 在普通的 SGD 中, 每个参数的更新步长是相同的; 而 AdaGrad 将考虑梯度的每个分量的历史累计情况, 来调整每个参数的学习率.

具体而言, 对于梯度 $g_k := \nabla f_{i_k}(x_k) \in \mathbb{R}^n$, 定义累计量:
$$
G_0=\boldsymbol{0},\qquad G_{k+1}=G_k+(g_k\odot g_k).
$$

其中 $\odot$ 表示元素级乘积.  $G_k$ 是一个向量, 其第 $j$ 个分量 $G_{k,j}$ 表示到第 $k-1$ 次迭代为止第 $j$ 个参数的梯度平方累计. 
- 若某分量的数值较大, 则说明该参数在之前的迭代中经历了较大的梯度累积, 历史变化较为剧烈, 因此需要较小的学习率来稳定更新
- 反之, 若某分量的数值较小, 则说明该参数在之前的迭代中经历了较小的梯度累积, 历史变化较为平稳, 可以使用较大的学习率来加速更新.

AdaGrad 的更新规则为:
- 给定初始学习率 $\alpha > 0$ 和一个小的常数 $\epsilon > 0$ (用于数值稳定), 初始化 $G_0 = \boldsymbol{0}$, 以及起点 $x_0$.
- 在迭代 $k = 0, 1, 2, \ldots$ 中:
  - 抽取随机样本索引 $i_k$ 并计算随机梯度 $g_k = \nabla f_{i_k}(x_k)$.
  - $G_{k+1} = G_k + g_k \odot g_k$.
  - $x_{k+1} = x_k -  \dfrac{\alpha}{\sqrt{G_{k+1} + \epsilon \boldsymbol{1}_n}} \odot g_k$.

AdaGrad 也可以当作是一种介于一阶方法和二阶方法之间的优化算法. 考虑 $f(x)$ 在 $x_k$ 处的二阶 Taylor 展开:
$$
f(x) \approx f(x_k) + \nabla f(x_k)^\top (x - x_k) + \frac{1}{2} (x - x_k)^\top B_k (x - x_k).
$$
根据 $B_k$ 的不同选择, 可以得到不同的优化算法. 而在上面的索引约定(先更新 $G_{k+1}$, 再更新 $x_{k+1}$)下, AdaGrad 对应
$$
B_k = {\alpha}^{-1} \text{diag}\!\left(\sqrt{G_{k+1} + \epsilon \boldsymbol{1}_n}\right).
$$

#### RMSProp

RMSProp (Root Mean Square Propagation) 是 AdaGrad 的一种改进, 该方法在非凸问题上的表现可能更好. 

- 注意到 AdaGrad 的更新步长 $\alpha / \sqrt{G_{k+1} + \epsilon \boldsymbol{1}_n}$ 中的累计量会随迭代不断增大, 导致学习率逐渐单调下降, 最终趋近于零, 从而使得算法在后期的迭代中几乎没有更新. 
- RMSProp 通过引入一个衰减因子 $\rho \in (0, 1)$ 来计算梯度平方的指数加权移动平均, 从而避免了学习率过快下降的问题.

具体地, 将 AdaGrad 中的 $G_k$ 累积项改进为:
$$
M_{k+1} = \rho M_k + (1 - \rho) (g_k \odot g_k),
$$

从而得到 RMSProp 的更新规则:
- 给定初始学习率 $\alpha > 0$, 衰减因子 $\rho \in (0, 1)$ (一般取 $\rho=0.9$) 和一个小的常数 $\epsilon > 0$, 初始化 $M_0 = \boldsymbol{0}$, 以及迭代起点 $x_0$.
- 在迭代 $k = 0, 1, 2, \ldots$ 中:
  - 抽取并计算随机梯度 $g_k = \nabla f_{i_k}(x_k)$.
  - 计算 $M_{k+1} = \rho M_k + (1 - \rho) (g_k \odot g_k)$.
  - 更新 $x_{k+1} = x_k -  \dfrac{\alpha}{\sqrt{M_{k+1} + \epsilon \boldsymbol{1}_n}} \odot g_k$.

其中 $\sqrt{M_k + \epsilon \boldsymbol{1}_n}$ 即为所谓的 RMS (Root Mean Square).


#### AdaDelta

AdaDelta 是 RMSProp 的一种改进, 旨在进一步提高优化算法的适应性和鲁棒性. 其和 RMSProp 一样需要维护 $M_k$ 以指数加权移动平均的方式来计算梯度平方的平均值. AdaDelta 在此基础上, 引入了一个新的累积项 $D_k$ 来跟踪参数更新的平方和, 从而将 $\alpha$ 替换为一个自适应的学习率:

- 累积梯度为: $M_{k+1} = \rho M_k + (1 - \rho) (g_k \odot g_k)$.
- 累积更新为: $D_{k+1} = \rho D_k + (1 - \rho) (\Delta x_k \odot \Delta x_k)$, 其中 $\Delta x_k = x_{k+1} - x_k$ 是第 $k$ 次迭代的实际更新.

其具体更新规则为:
- 给定衰减因子 $\rho \in (0, 1)$ 和一个小的常数 $\epsilon > 0$, 初始化 $M_0 = \boldsymbol{0}$ 和 $D_0 = \boldsymbol{0}$.
- 在迭代 $k = 0, 1, 2, \ldots$ 中:  
  - 抽取并计算随机梯度 $g_k = \nabla f_{i_k}(x_k)$.
  - 计算累积梯度: $M_{k+1} = \rho M_k + (1 - \rho) (g_k\odot g_k)$.
  - 计算更新方向: $\Delta x_k = - \dfrac{\sqrt{D_k + \epsilon \boldsymbol{1}_n}}{\sqrt{M_{k+1} + \epsilon \boldsymbol{1}_n}} \odot g_k$.
  - 更新参数: $x_{k+1} = x_k + \Delta x_k$.
  - 计算累积更新: $D_{k+1} = \rho D_k + (1 - \rho) (\Delta x_k \odot \Delta x_k)$.


#### Adam

Adam (Adaptive Moment Estimation) 本质上是包含了 Momentum 和 RMSProp 的优化算法, 通过同时考虑梯度的一阶矩和二阶矩来调整每个参数的学习率. Adam 的优势在于偏差校正可缓解零初始化造成的一阶/二阶矩低估, 同时二阶矩分母可抑制坐标方向上的过大更新, 从而使参数更新更平稳. 

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
- 随机次梯度的二阶矩有界, 即存在常数 $M > 0$ 使得 $\mathbb{E}[\|g_{i_k}(x_k)\|^2] \leq M^2<\infty$ 对于所有 $k$ 都成立, 其中 $g_{i_k}(x_k)\in\partial f_{i_k}(x_k)$ 是随机样本 $i_k$ 处的一个次梯度.
  - 这是随机次梯度范数二阶矩有界假设(并非直接等同于方差定义), 它保证了 SGD 更新噪声可控.
- 迭代点到最优点的距离有界. 即存在常数 $R > 0$ 使得 $\|x_k-x^*\| \leq R$ 对于所有 $k$ 都成立. 

注: 在进行 SGD 的收敛分析时, 由于每次迭代的更新方向是随机的, 因此我们通常关注的是算法的**期望行为**或者**高概率行为**. 此外, 对于某一个具体的迭代点 $x_k$, 由于 $i_k$ 的随机性, 其更新方向 $g_{i_k}(x_k)$ 也是随机的, 因此我们通常也会考虑这些迭代点的平均重心 $\bar{x}_k$ 来分析算法的收敛性.

***Lemma* (SGD 的累计误差)**: 在上述假设下,令 $\{\alpha_k\}$ 是任意正步长序列, $\{x_k\}$ 是 SGD 迭代生成的点列, 则对于任意 $K \geq 1$, 都有如下不等式成立:
$$
\sum_{k=1}^K \alpha_k \mathbb{E}[f(x_k) - f(x^*)] \leq \frac{1}{2} \mathbb{E}[\|x_1 - x^*\|^2] + \frac{1}{2} \sum_{k=1}^K \alpha_k^2 M^2,
$$

- *Proof*.
  - 记 $g_k := g_{i_k}(x_k) \in \partial f_{i_k}(x_k)$ 是指在第 $k$ 次迭代中, 随机选择的样本 $i_k$ 在 $x_k$ 处的一个次梯度. 记 $\bar{g}_k := \mathbb{E}[g_{i_k}(x_k) \mid x_k]$ 是 $x_k$ 处的随机次梯度的条件期望, 则由 SGD 估计的无偏性可知 $\bar{g}_k \in \partial f(x_k)$. 记 $\xi_k = g_k - \bar{g}_k$ 为随机次梯度的噪声, 则 $\mathbb{E}[\xi_k \mid x_k] = 0$.
  - 由次梯度的性质 $\langle \bar{g}_k, x^* - x_k \rangle \leq f(x^*) - f(x_k)$, 可以推得:
    $$
    \begin{aligned}
    \frac12\|x_{k+1} - x^*\|^2 &= \frac12\|x_k - \alpha_k g_k - x^*\|^2 \\
    &= \frac12\|x_k - x^*\|^2 - \alpha_k \langle g_k, x_k - x^* \rangle +\frac12 \alpha_k^2 \|g_k\|^2 \\
    &= \frac12\|x_k - x^*\|^2 - \alpha_k \langle \bar{g}_k, x_k - x^* \rangle - \alpha_k \langle \xi_k, x_k - x^* \rangle +\frac12 \alpha_k^2 \|g_k\|^2 \\
    &\leq \frac12\|x_k - x^*\|^2 - \alpha_k (f(x_k) - f(x^*)) - \alpha_k \langle \xi_k, x_k - x^* \rangle +\frac12 \alpha_k^2 \|g_k\|^2.
    \end{aligned}
    $$
  
  - 又根据条件期望 $\mathbb{E}[\langle \xi_k, x_k - x^* \rangle \mid x_k] = 0$, 利用重期望可以得到:$\mathbb{E}[\langle \xi_k, x_k - x^* \rangle] = \mathbb{E}[\mathbb{E}[\langle \xi_k, x_k - x^* \rangle \mid x_k]] = 0$. 因此, 对上述不等式两边取期望, 可以得到:
    $$
    \alpha_k \mathbb{E}[f(x_k) - f(x^*)] \leq \frac12\mathbb{E}[\|x_k - x^*\|^2] - \frac12\mathbb{E}[\|x_{k+1} - x^*\|^2] + \frac12 \alpha_k^2 M^2.
    $$

  - 将上述不等式对 $k=1, 2, \ldots, K$ 进行求和, 可以得到:
    $$
    \sum_{k=1}^K \alpha_k \mathbb{E}[f(x_k) - f(x^*)] \leq \frac{1}{2} \mathbb{E}[\|x_1 - x^*\|^2] + \frac{1}{2} \sum_{k=1}^K \alpha_k^2 M^2.
    $$

  $\square$

- 上述引理在说明: 
  - SGD 的累计误差 (即 $\sum_{k=1}^K \alpha_k \mathbb{E}[f(x_k) - f(x^*)]$) 可以被初始点与最优点之间的距离 $\|x_1 - x^*\|^2$ 和噪声项 $\sum_{k=1}^K \alpha_k^2 M^2$ 控制. 
  - 这为我们分析 SGD 的收敛性提供了一个重要的工具, 因为它将算法的性能与初始条件和随机梯度的方差联系起来.

***Theorem* (SGD 的收敛性 1: 在步长加权平均意义下的收敛)**: 在上述假设下, 定义步长加权平均点 $\bar{x}_K := \dfrac{\sum_{k=1}^K \alpha_k x_k}{\sum_{k=1}^K \alpha_k}$, 则对于任意 $K \geq 1$, 都有如下期望意义下的收敛性保证:
$$
\mathbb{E}[f(\bar{x}_K) - f(x^*)] \leq \frac{R^2 + \sum_{k=1}^K \alpha_k^2 M^2}{2\sum_{k=1}^K \alpha_k}.
$$

- *Proof*.
  - 记 $A_K := \sum_{k=1}^K \alpha_k$, 则 $\bar{x}_K = \frac{1}{A_K} \sum_{k=1}^K \alpha_k x_k$. 由于 $f$ 是凸函数, 由 Jensen Inequality 可以得到:
    $$
    f(\bar{x}_K) = f\left(\frac{1}{A_K} \sum_{k=1}^K \alpha_k x_k\right) \leq \frac{1}{A_K} \sum_{k=1}^K \alpha_k f(x_k).
    $$
  
  - 两侧同时减去 $f(x^*)$ 并取期望, 可以得到:
    $$
    \mathbb{E}[f(\bar{x}_K) - f(x^*)] \leq \frac{1}{A_K} \sum_{k=1}^K \alpha_k \mathbb{E}[f(x_k) - f(x^*)].
    $$

  - 结合之前的引理, 可以得到:
    $$
    \begin{aligned}
    \mathbb{E}[f(\bar{x}_K) - f(x^*)] &\leq \frac{1}{A_K} \left( \frac{1}{2} \mathbb{E}[\|x_1 - x^*\|^2] + \frac{1}{2} \sum_{k=1}^K \alpha_k^2 M^2 \right) \\
    &\leq \frac{R^2 + \sum_{k=1}^K \alpha_k^2 M^2}{2 A_K}.
    \end{aligned} 
    $$

  $\square$

- 由上述定理可以看出, SGD 的收敛速度取决于步长序列 $\{\alpha_k\}$ 的选择. 
  - 例如, 当 $\sum_{k=1}^\infty \alpha_k = \infty$ 且 $\sum_{k=1}^\infty \alpha_k^2 < \infty$ 时, 随机梯度下降算法在期望意义下收敛到最优值, 即 $\lim_{K \to \infty} \mathbb{E}[f(\bar{x}_K) - f(x^*)] = 0$. 
  - 若选择 $\alpha_k$ 为一个固定步长 $\alpha > 0$, 则其在期望意义下是不收敛的, 即 $\mathbb{E}[f(\bar{x}_K) - f(x^*)] \leq \frac{R^2 + K \alpha^2 M^2}{2 K \alpha} \stackrel{K \to \infty}{\longrightarrow} \frac{\alpha M^2}{2} > 0$. 此时只能确定一个次优解的误差上界, 但无法保证其收敛到最优值.


***Theorem* (SGD 的收敛性 2: 不增步长序列下的等权平均收敛)**: 在上述假设下, 定义等权平均点 $\hat{x}_K := \frac{1}{K} \sum_{k=1}^K x_k$, 且要求步长序列 $\{\alpha_k\}$ 是一个不增的正数列, 则对于任意 $K \geq 1$, 都有如下期望意义下的收敛性保证:
$$
\mathbb{E}[f(\hat{x}_K) - f(x^*)] \leq \frac{R^2}{2K \alpha_K} + \frac{1}{2K} \sum_{k=1}^K \alpha_k M^2.
$$

- 该定理与前一定理的主要区别在于, 前者是针对步长加权平均点 $\bar{x}_K$ 的收敛性保证, 而后者则是针对等权平均点 $\hat{x}_K$ 的收敛性保证, 其额外只需要要求步长序列 $\{\alpha_k\}$ 是一个不增的正数列即可. 
- 通过选择合适的步长序列, 例如 $\alpha_k = \mathcal{O}(1/\sqrt{k})$, 可以得到 $\mathbb{E}[f(\hat{x}_K) - f(x^*)] = \mathcal{O}(1/\sqrt{K})$ 的收敛速度, 这也是 SGD 在一般凸函数上的最优收敛速度.
  - 特别地, 取 $\alpha_k = \dfrac{R}{M \sqrt{k}}$, 则可以得到 $\mathbb{E}[f(\hat{x}_K) - f(x^*)] \leq \dfrac{3R M}{2\sqrt{K}}$ 的收敛速度.

> [!info] 讨论:
> 在一般凸且可能非光滑的设定下, 确定性次梯度法与随机次梯度法在最优量级上都可达到 $\mathcal{O}(1/\sqrt{K})$. 但若进一步假设目标函数光滑, 则确定性梯度下降可达到 $\mathcal{O}(1/K)$ (加速法可达 $\mathcal{O}(1/K^2)$), 而朴素 SGD 在不做方差缩减时通常仍是 $\mathcal{O}(1/\sqrt{K})$ 量级.

***Theorem* (SGD 的收敛性 3: 衰减步长下的依概率收敛)**: 在上述假设下, 定义等权平均点 $\hat{x}_K := \frac{1}{K} \sum_{k=1}^K x_k$, 且选择步长 $\alpha_k = \mathcal{O}(1/\sqrt{k})$ (如 $\alpha_k = \dfrac{R}{M \sqrt{k}}$), 则有如下依概率收敛性保证:
$$
f(\hat{x}_K) - f(x^*) \stackrel{P}{\longrightarrow} 0 \quad \text{as } K \to \infty.
$$
或等价地
$$
\lim_{K \to \infty} P(f(\hat{x}_K) - f(x^*) \leq \epsilon) = 1 \quad \text{for any } \epsilon > 0.
$$
- *Proof*.
  - 由于 $\alpha_k = \mathcal{O}(1/\sqrt{k})$, 则由 Theorem 2 可以得到 $\mathbb{E}[f(\hat{x}_K)- f(x^*)] \to 0$ 当 $K \to \infty$.
  - 根据 Markov 不等式, 对任意 $\epsilon > 0$, 可以得到:
    $$
    \mathbb{P}(f(\hat{x}_K) - f(x^*) > \epsilon) \leq \frac{\mathbb{E}[f(\hat{x}_K) - f(x^*)]}{\epsilon} \to 0 \quad \text{as } K \to \infty.
    $$

***Theorem* (SGD 的收敛性 3': 衰减步长下的依概率收敛速度)**: 在上述假设下, 进一步假设对于所有次梯度 $g_{i_k}(x_k)$ 都满足 $\|g_{i_k}(x_k)\| \leq M$ 几乎处处成立, 则对于任意 $\epsilon > 0$, 可保证如下收敛以至少 $1 - \exp(-\epsilon^2/2)$ 的概率成立:
$$
f(\hat{x}_K) - f(x^*) \leq \underbrace{\frac{R^2}{2K \alpha_K} + \frac{1}{2K} \sum_{k=1}^K \alpha_k M^2}_{\scriptsize{\text{Expectation Bound in Thm. 2}}} + \underbrace{\frac{RM}{\sqrt{K}}\epsilon}_{\scriptsize{\text{Prob. Bound}}}.
$$

- 特别地, 若取 $\alpha_k = \dfrac{R}{M \sqrt{k}}$, $\delta = \exp(-\epsilon^2/2)$, 则可以得到如下概率收敛速度:
  $$
  \mathbb{P}\left\{f(\hat{x}_K) - f(x^*) \leq \frac{3R M}{2\sqrt{K}} + \frac{RM}{\sqrt{K}}\sqrt{2\log(1/\delta)}\right\} \geq 1 - \delta.
  $$

### Convergence under Strong Convexity

下面给出一个更常见且自洽的复杂度对比(按**分量梯度计算次数**计; 忽略条件数常数细节). 其中 $N$ 是样本数, $\kappa=L/\mu$ 是条件数:

| 方法 \ 目标类 | 凸, 可能非光滑 | 凸且 $L$-smooth | $\mu$-强凸且 $L$-smooth |
| --- | --- | --- | --- |
| SGD (无方差缩减) | $\mathcal{O}(1/\epsilon^2)$ | $\mathcal{O}(1/\epsilon^2)$ | $\mathcal{O}(1/\epsilon)$ |
| Full GD (每步全梯度) | $\mathcal{O}(N/\epsilon^2)$ | $\mathcal{O}(N/\epsilon)$ | $\mathcal{O}(N\log(1/\epsilon))$ |
| 方差缩减 (SVRG/SAGA/SAG 等) | 通常不主打该设定 | 通常不主打该设定 | $\mathcal{O}((N+\kappa)\log(1/\epsilon))$ |

- 在强凸光滑且追求高精度时, 方差缩减方法通常优于朴素 SGD, 这也是其主要动机.
- 在中低精度或 $N$ 极大时, SGD 由于单步成本低, 仍可能更具实践优势.

## Variance Reduction Techniques for SGD

### Consequences of Variance in SGD

假设目标函数 $f$ 是 $\mu$-强凸的，并且 $L$-光滑的，那么对于任意的 $x,y$，都有:
- 强凸: $\langle \nabla f(x) - \nabla f(y), x-y \rangle \geq \mu \|x-y\|^2$
- 光滑: $\|\nabla f(x) - \nabla f(y)\| \leq L \|x-y\|$

记更新为 $x_{k+1}=x_k-\alpha \nabla f_{i_k}(x_k)$, 并定义噪声
$$
\zeta_k := \nabla f_{i_k}(x_k)-\nabla f(x_k),\qquad \mathbb{E}[\zeta_k\mid x_k]=0.
$$
再记 $\Delta_k = \|x_k - x^*\|^2$. 则可得到如下典型递推:
$$
\begin{aligned}
\mathbb{E}[\Delta_{k+1}]
&=\mathbb{E}[\|x_{k+1} - x^*\|^2 ]\\
&\leq \underbrace{(1-2\alpha\mu+\alpha^2L^2)\mathbb{E}[\Delta_k]}_{\text{A: Deterministic contraction}}
+\underbrace{\alpha^2\mathbb{E}[\|\zeta_k\|^2]}_{\text{B: stochastic-noise term}}
\end{aligned}
$$

- 通过上面的分解可以看到, 总的误差由两部分组成: 
  - A: 确定性收缩项, 由步长 $\alpha$, 强凸参数 $\mu$ 和光滑参数 $L$ 决定; 当 $\alpha$ 合适时该项带来线性收缩趋势.
  - B: 噪声驱动项, 反映随机梯度的条件方差效应.
    - 若使用常数步长, B 项不会自动消失, 常导致收敛到一个与步长相关的误差地板.
    - 若使用递减步长, B 项可随迭代减弱, 在强凸设定下可获得 $\mathcal{O}(1/k)$ 量级结果.
    - 方差缩减方法 (SVRG/SAGA/SAG 等) 的核心就是削弱 B 项对后期收敛的限制.

### Variance Reduction Techniques

#### SAG and SAGA

SAG (Stochastic Average Gradient) 和 SAGA 通过维护历史梯度来减少随机梯度的方差. 

SAG 方法会维护一个梯度存储表 $\{g_i^k\}_{i=1}^N$, 其中 $g_i^k$ 表示“第 $k$ 次迭代开始时”第 $i$ 个样本的存储梯度. 在每次迭代中, 随机选择一个样本索引 $i_k$, 用新梯度覆盖该位置, 并使用全表平均梯度更新参数. 其具体更新规则为:
- 初始化: $g_i^0 = [0, \ldots, 0]$ 对于所有 $i=1, 2, \ldots, N$, 以及起点 $x_0$.
- 在迭代 $k = 0, 1, 2, \ldots$ 中:
  - 随机选择一个样本索引 $i_k \in \{1, 2, \ldots, N\}$.
  - 计算新梯度: $g^{\text{new}} = \nabla f_{i_k}(x_k)$.
  - 更新梯度表: $g_{i_k}^{k+1}=g^{\text{new}}$, 且 $g_i^{k+1}=g_i^k\ (i\neq i_k)$.
  - 更新参数: $x_{k+1} = x_k - \alpha \frac{1}{N} \sum_{i=1}^N g_i^{k+1}$.

在强凸光滑假设下, 对于固定步长 $\alpha = 1/(16L)$, 及零梯度初始化, SAG 的收敛速度为:
$$
\mathbb{E}[f(x_k) - f(x^*)] \leq \left(1 - \min\left\{\frac{\mu}{16L}, \frac{1}{8N}\right\}\right)^k \cdot C_0.
$$

SAG 的缺点是在于其需要维护一个 $N$ 维的梯度列表, 当数据集较大时, 其内存开销较大. 另一方面, SAG 的随机梯度估计是有偏的, 因此 SAGA 则使用一个无偏的随机梯度估计来改进 SAG. 若第 $k$ 步开始时存储表为 $\{g_i^k\}_{i=1}^N$, 则:
$$
g^{\text{new}}=\nabla f_{i_k}(x_k),\qquad
v_k=g^{\text{new}}-g_{i_k}^k+\frac{1}{N}\sum_{i=1}^N g_i^k,
$$
$$
x_{k+1}=x_k-\alpha_k v_k,
$$
并在步末更新 $g_{i_k}^{k+1}=g^{\text{new}}$, 其余分量保持 $g_i^{k+1}=g_i^k\ (i\neq i_k)$.

#### SVRG

SVRG 通过周期性记录全梯度 checkpoint, 并在每次迭代时通过与 checkpoint 的全梯度进行差分来减少随机梯度的方差. 记 $\tilde{x}^{(j)}$ 是第 $j$ 个 checkpoint, 对应的全梯度为 $\nabla f(\tilde{x}^{(j)}) = \frac{1}{N} \sum_{i=1}^N \nabla f_i(\tilde{x}^{(j)})$. 故在每次迭代中, 更新方向为:   

$$
v_k := \nabla f_{i_k}(x_k) - [\nabla f_{i_k}(\tilde{x}^{(j)}) - \nabla f(\tilde{x}^{(j)})].
$$
