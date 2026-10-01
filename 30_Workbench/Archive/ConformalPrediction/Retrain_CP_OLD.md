# Retrain CP

## Introduction

在高风险决策场景下, 我们往往需要给估计的可靠性给出度量. Conformal Prediction (CP) 提供了一种在有限样本下, 对预测区间进行置信度估计的有效方法. 

- 其优势在于:
  - 无需对数据分布做出假设
  - 有限样本下也能提供可靠的置信区间

- CP 的工作处理流程在于:
  - 将数据重新划分为 Train Set 和 Calibration Set
  - 在 Train Set 上训练模型
  - 在 Calibration Set 上计算非符合度分数 (Non-conformity Score)
  - 根据 Non-conformity Score 计算置信区间

Vanilla CP 的一个问题在于, 更多只关注其覆盖率的概率保证, 而忽视了其预测区间的长度信息. 这会导致在某些情况下, 预测区间过宽, 从而降低其实用性.  

例如在回归任务重, 预测区间之长度相当于绝对残差分布的某个分位数. 若残差分布是 ill-conditioned (例如, 非零均值, 高方差等) 时, 预测区间的长度就有很大的改进空间. 这里主要关注两种病态分布的来源: 
-  covariate shift: 训练集和测试集的输入分布不一致
-  model misspecification: 模型假设与真实数据生成过程不一致

因此针对这些问题, 文章提出解决方案: 在 calibration 之前重新训练模型, 来重塑病态分布.

其关键特点在于:
- 简单且可扩展的 retrain 策略
- 不损害 CP 的后处理性质 (post-hoc nature): 不需要修改原始模型的训练过程. 在模型训练后再进行校准. 

不过在实践中我们还需要关注如下指标: 
- 数据分割比例: 如何 trade-off retrain 和 calibration 的数据量
- 有效性条件: 在什么条件下 retrain CP 能够提升性能, 如何能够保证 retrain CP 的有效性

## Preliminaries

### Definitions

定义顺序统计量 (Order Statistics): 对于一组样本 $Z_1, Z_2, \ldots, Z_n$, 其顺序统计量表示为 $Z_{(1)} \geq Z_{(2)} \geq \ldots \geq Z_{(n)}$, 其中 $Z_{(1)}$ 是样本中的最大值, $Z_{(n)}$ 是最小值.

对于随机变量 $Z$ 和分位数水平 $\tau \in (0, 1)$, 总体分位数 $Q_\tau(Z)$ 定义为:
$$Q_\tau(Z) = \inf \{ z \in \mathbb{R} : \mathbb{P}(Z \leq z) \geq \tau \}.$$
对应经验分位数为:
$$\hat{Q}_\tau(Z_1, \ldots, Z_n) = Z_{(\lceil (n+1) (1-\tau) \rceil)},$$

定义绝对 Gaussian 分布 (Absolute Gaussian Distribution): 设 $X \sim \mathcal{N}(\mu, \sigma^2)$, 则 $|X|$ 的分布称为绝对 Gaussian 分布, 记为 $|X| \sim \text{abs}\mathcal{N}(\mu, \sigma^2)$. 

对于矩阵 $A \in \mathbb{R}^{n \times n}$ 和向量 $\boldsymbol{a} \in \mathbb{R}^n$, 定义矩阵的加权范数为:
$$\|\boldsymbol{a}\|^2_A := \boldsymbol{a}^\top A \boldsymbol{a}.$$

### Notations

在我们的 CP 任务中, 假设训练数据 $\{(\boldsymbol{x}_i, y_i)\}_{i=1}^n$ 独立同分布 (i.i.d.) 来自某个联合分布 $\mathcal{P}_{XY}$. 其中 $\boldsymbol{x}_i \in \mathbb{R}^d$ 是特征向量, $y_i \in \mathbb{R}$ 是对应的标量标签. 

我们的任务在于构造预测区间 $C(\boldsymbol{X}_{n+1})\subset \mathbb{R}$, 使得覆盖保证对于任意联合分布 $\mathcal{P}_{XY}$ 和任意样本量 $n$ 均有:
$$
\mathbb{P}\{Y_{n+1} \in C(\boldsymbol{X}_{n+1})\} \geq 1-\alpha
$$
其中 $\alpha$ 是某给定的 miscoverage  rate. 

### CP Procedure

**Step 1: Data Splitting and Training**

首先将数据分割成不相交的两个子集:
- Proper training set: $\mathcal{S}_{\text{train}} := \{(x_i,y_i), i\in \mathcal{I}_{\text{train}}\}$
- Calibration set: $\mathcal{S}_{\text{calib}}:=\{(x_i,y_i), i\in \mathcal{I}_{\text{calib}}\}$


给定某算法 $\mathrm{A}$, 在 proper training set 上进行训练, 得到算法 $\hat{f}: \mathbb{R}^d \to \mathbb{R}$:
$$
\hat{f}(\cdot) \leftarrow \mathrm{A}(\mathcal{S}_{\text{train}})
$$

**Step 2: Non-conformity Scoring**

利用校准集的数据以及在训练集上得到的模型计算绝对残差:
$$
\mathcal{R}_{\text{calib}} = \{|r_i|: r_i = y_i - \hat{f}(\boldsymbol{x}_i), i\in\mathcal{I}_{\text{calib}}\}
$$

然后根据 CP 的理论, 给定原始覆盖水平 $1-\alpha$, 计算校正后的覆盖水平:
$$
1-\alpha' = (1-\alpha)(1+\frac{1}{\mathcal{R}_{\text{calib}} })
$$

根据该覆盖水平计算残差的分位数:
$$
\hat{Q}_{1-\alpha' }(\mathcal{R}_{\text{calib}} )
$$

**Step 3: 根据残差分位数构造预测区间**

对于新样本 $\boldsymbol{X}_{n+1}$, 构造预测区间为:
$$
C(\boldsymbol{X}_{n+1}) = [\hat{f}(\boldsymbol{X}_{n+1}) - \hat{Q}_{1-\alpha' }(\mathcal{R}_{\text{calib}} ), \hat{f}(\boldsymbol{X}_{n+1}) + \hat{Q}_{1-\alpha' }(\mathcal{R}_{\text{calib}} )]
$$

## A Scalable Retraining Framework

### Optimal Prediction Interval

假设回归模型为 
$$Y = f^*(\boldsymbol{X}) +\epsilon, \quad \boldsymbol{X} \sim \mathcal{P}_{\boldsymbol{X}}$$
- $f^*$ 为 oracle 的回归函数 (未知)
- $\epsilon$ 为误差项, 满足
  - $0$ symmetric
  - 与 $\boldsymbol{X}$ 对称

Vanilla CP 中
- 区间为 $[\hat f(\boldsymbol{X}) \pm \hat{Q}_{1-\alpha}(\mathcal{R}_{\text{calib}})]$
- 区间长度为 $2\hat{Q}_{1-\alpha}(\mathcal{R}_{\text{calib}})$
  - 这近似于总体的分位数 $2\hat{Q}_{1-\alpha}(\mathcal{|\hat R|})$, 其中 $\mathcal{\hat R}(\boldsymbol{X},Y) = Y- \hat{f}(\boldsymbol{X}) =\underbrace{f^*(\boldsymbol{X}) - \hat f(\boldsymbol{X})}_{\text{Model Error}} + \epsilon$.

一个理想的情况, 若真实的回归函数 $f^*$ 是已知的, 则此时的残差变为
$$\mathcal{R}_{f^*} = Y - f^*(\boldsymbol{X}) = \epsilon$$

因此我们对于区间长度的优化可以固定为一个优化问题: 
$$
\min_{L\ge0}L, \text{  s.t. } \mathbb{P}(Y \in [f^*(\boldsymbol{X}) - L, f^*(\boldsymbol{X}) + L]) \ge 1- \alpha
$$

当误差分布良好时, 上述优化问题有显式解: 
- **Theorem 4.1**: 若误差 $\epsilon$ 满足, 0 对称, 单峰, 则 $Q_{1-\alpha}(|\mathcal{R}_{f^*}|)$ 即为原问题的解. 

Vanilla CP 的次优性: 在实际问题中, 我们需要用 $\hat{f}$ 替代 $f^*$, 估计的区间为 $[\hat{f}(\boldsymbol{X}) \pm Q_{1-\alpha}(|\mathcal{\hat R}|)]$. 此时的长度可能并非最优, 这是因为残差从 $\mathcal{R}_{f^*}  = \epsilon$ 变为了 $\hat{\mathcal{R}} = f^* - \hat f + \epsilon$, 即多出了模型误差. 此时尤其在两种病态场景下, 我们需要更大的半径才能保证原始的概率覆盖:
- 均值非零 $\mathbb{E}[f^*(\boldsymbol{X}) - \hat{f}(\boldsymbol{X})] \neq 0$: 这使得 $\mathcal{\hat{R}}$ 的分布不再关于 0 对称, 故需要更大半径. 
- 方差 $\mathbb{V}[f^*(\boldsymbol{X}) - \hat{f}(\boldsymbol{X})]$ 过大: 这使得残差方差增大, 区间变长. 

***Algorithm 1: Retrained Conformal Prediction***

**INPUT**: 训练数据 $\mathcal{S}_1$, 校准数据 $\mathcal{S}_2$, 回归算法 $\mathrm{A}$, retrain 比例 $\xi\in(0,1)$, retrain 模型 $g_\theta$, 覆盖率 $1-\alpha$, 新输入 $\boldsymbol{X}$. 

**Step 1**: 进一步按照比例 $xi$ 划分 Calibration 为 retraining 和新的 calibration 两部分, 记为 $\mathcal{S}_2^{\text{ret}}$ 和 $\mathcal{S}_2^{\text{cal}}$, 其中 $|\mathcal{S}_2^{\text{ret}}| = \xi\cdot |\mathcal{S}_2|$.

**Step 2**: 在训练集 $\mathcal{S}_1$ 上训练模型, 得到算法 $\hat f$:
$$\hat{f}(\cdot) \leftarrow \mathrm{A}(\mathcal{S}_1)$$

**Step 3**: 在 retrain $\mathcal{S}_2^{\text{ret}}$ 上对残差 $\hat{R}(\boldsymbol{x}_i,  y_i) = y_i - \hat{f}(\boldsymbol{x}_i)$ 进行拟合, 得到一个 fit 的函数 $g_{\hat{\theta}}(\boldsymbol{x})$:
$$\min_\theta \frac{1}{\mathcal{I}_2^{\text{ret}}} \sum_{\mathcal{I}_2^{\text{ret}}} [\hat{R}(\boldsymbol{x}_i, y_i) - g_\theta(\boldsymbol{x}_i)]^2$$
- 对于该项 $g_\theta$ 的理解见下文的 Retraining Framework 部分.


**Step 4**: 计算调整后的 Non- conformity Score:

在新的 calibration $\mathcal{S}_2^\text{cal}$ 上, 计算:
$$\mathcal{R}_{\text{cal}} = \{|y_i - \hat f(\boldsymbol{x}_i - g_{\hat{\theta}}(\boldsymbol{x}_i): i\in\mathcal{I}_2^{\text{cal}}\}$$

对比 vanilla CP:
- Vanilla: $|y_i - \hat f(\boldsymbol{x}_i)|$
- Retrained: $|y_i - \hat f(\boldsymbol{x}_i) - g_{\hat{\theta}}(\boldsymbol{x}_i)|$

计算新的调整后的覆盖水平: 
$$1-\alpha' = (1-\alpha)(1+\frac{1}{|\mathcal{I}_2^{\text{cal}}|})$$

计算经验分位数:
$$\hat{Q}_{1-\alpha'}(\mathcal{R}_{\text{cal}} )$$

**OUTPUT**: 构造新的预测区间
$$[\underbrace{\hat{f}(\boldsymbol{X})+g_{\hat{\theta}}(\boldsymbol{X})}_{\approx \hat f-\hat f+f^* = f^*} -\hat{Q}_{1-\alpha'}(\mathcal{R}_{\text{cal}} ), \hat{f}(\boldsymbol{X})+g_{\hat{\theta}}(\boldsymbol{X})+\hat{Q}_{1-\alpha'}(\mathcal{R}_{\text{cal}} )]$$

### Retraining Framework

将上面的步骤总结为一个 retraining 的框架.  我们定义基于在 training set $\mathcal{S}_1$ 上训练的模型 $\hat f$ (并且后续亦不再更新) 得到的残差为:
$$\hat{\mathcal{R}}(\boldsymbol{X},Y) = Y - \hat f(\boldsymbol{X})$$

同时引入一个用于拟合残差的 retrain 模型 $g_\theta(\boldsymbol{X})$, 其可以是现行的, 也可以是小 MLP 等. $g_\theta$ 的目标在于希望将拟合降噪后的残差变成纯粹的噪声, 即, 假设 $p_\epsilon \sim \mathcal{WN}(0,\sigma^2)$, 则希望极大如下似然函数:
$$\max_\theta \frac{1}{|\mathcal{I}_2^{\text{ret}}|} \sum_{i\in \mathcal{I}_2^{\text{ret}}} \log p_\epsilon\left(\hat{\mathcal{R}}(\boldsymbol{x}_i,y_i) - g_\theta(\boldsymbol{x}_i)\right)$$
特别地, 在以高斯噪声为例时, 上述目标等价于最小化 MSE:
$$\min_\theta \frac{1}{|\mathcal{I}_2^{\text{ret}}|} \sum_{i\in \mathcal{I}_2^{\text{ret}}} [\hat{\mathcal{R}}(\boldsymbol{x}_i,y_i) - g_\theta(\boldsymbol{x}_i)]^2 \quad (4)$$

## A Case Study on Linear Residual Model

### Problem Setup

上述的算法给出了一个完整的 retrain CP 框架. 这里我们以线性回归为例, 并理论的回答: 1. $\xi = \frac{|\mathcal{S}_2^{\text{ret}}|}{|\mathcal{S}_2|}$ 的选择; 2. 在什么条件下 retrain CP 能够提升性能.

这里引入一个模型的简化假设: 线性残差模型, 即假设模型误差关于输入 $\boldsymbol{X}$ 是线性的:
$$\mathcal{R} = Y - \hat f(\boldsymbol{X}) = f^*(\boldsymbol{X}) + \epsilon - \hat f(\boldsymbol{X}) := \boldsymbol{X}^\top \beta^\circ + \epsilon$$
- 其中 $\boldsymbol{X} \sim \mathcal{N}(0,\Sigma_X)$
- 其中 $\epsilon \sim \mathcal{N}(0,\sigma^2), \epsilon \perp \boldsymbol{X}$

进一步记 总的 Cal Set $\mathcal{S}_2 = \{(\boldsymbol{x}_i,y_i)\}_{i=1}^m$, 分配后的 retrain set $|\mathcal{S}_2^{\text{ret}}| := m_1 = \xi m$, calibration set $|\mathcal{S}_2^{\text{cal}}| := m_2 = (1-\xi)m$.

在 retrain 阶段, 我们通过损失函数 $\min_\theta \frac{1}{|\mathcal{I}_2^{\text{ret}}|} \sum_{i\in \mathcal{I}_2^{\text{ret}}} [\hat{\mathcal{R}}(\boldsymbol{x}_i,y_i) - g_\theta(\boldsymbol{x}_i)]^2$ 来拟合残差 (同样地, 这里的拟合残差 $g_\theta$ 我们也假设其具有线性结构 $g_\theta(\boldsymbol{X}) = \boldsymbol{X}^\top \beta$). 我们可以根据 MSE 很方便的得到其解: $\hat \beta = \arg\min_\beta \frac{1}{m_1} \sum_{i=1}^{m_1} [\hat{\mathcal{R}}(\boldsymbol{x}_i,y_i) - \boldsymbol{x}_i^\top \beta]^2$.

这里整理一下 retrain CP 和 vanilla CP 的核心过程:
- Vanilla CP
  - 回顾, 其残差为 $\hat{\mathcal{R}}^{\text{vanilla}} = |Y - \hat f(\boldsymbol{X})| = |(f^*(\boldsymbol{X}) - \hat f(\boldsymbol{X})) + \epsilon|$
  - 对应分位数来源为整个 calibration set $\mathcal{S}_2$, 对应经验分布为
     $$ \hat{Q}^{\text{vanilla}} = \mathcal{R}^\text{vanilla}_{(\lceil(m+1)(1-\alpha)\rceil)}$$
  - 对应预测区间为: $C^{\text{vanilla}}(\boldsymbol{X}) = [\hat f(\boldsymbol{X}) \pm \hat{Q}^{\text{vanilla}}]$
- Retrain CP
  - 在 $\mathcal{S}_2^{\text{ret}}$ 上拟合一个残差头 $g_\theta(\boldsymbol{X}) = \boldsymbol{X}^\top \hat \beta$, 近似捕捉模型误差 $f^*(\boldsymbol{X}) - \hat f(\boldsymbol{X}) \approx \boldsymbol{X}^\top \beta^\circ$
  - 在剩余的 $\mathcal{S}_2^{\text{cal}}$ 上计算残差, 其残差为 $\hat{\mathcal{R}}^{\text{retrain}} = |Y - \hat f(\boldsymbol{X}) - g_{\hat \theta}(\boldsymbol{X})| = |(f^*(\boldsymbol{X}) - \hat f(\boldsymbol{X}) - g_{\hat \theta}(\boldsymbol{X})) + \epsilon|$
  - 对应分位数来源为 calibration set $\mathcal{S}_2^{\text{cal}}$ (只有较少的 $m_2 = \xi m$ 个样本), 对应经验分位数为
     $$ \hat{Q}^{\text{retrain}} = \mathcal{R}^\text{retrain}_{(\lceil(m_2+1)(1-\alpha)\rceil)}$$
  - 对应预测区间为: $C^{\text{retrain}}(\boldsymbol{X}) = [\hat f(\boldsymbol{X}) + g_{\hat \theta}(\boldsymbol{X}) \pm \hat{Q}^{\text{retrain}}]$

因此这里对比二者的区间长度:
- Vanilla CP: $L^{\text{vanilla}} /2= \hat{Q}^{\text{vanilla}} = \mathcal{R}^\text{vanilla}_{(\lceil(m+1)(1-\alpha^{\text{vanilla}})\rceil)} = |\boldsymbol{X}^\top \beta^\circ + \epsilon| \sim \text{abs}\mathcal{N}(0, s_1^2)$, 其中 $s_1^2 = \|\beta^\circ\|^2_{\Sigma_X} + \sigma^2$
- Retrain CP: $L^{\text{retrain}} /2= \hat{Q}^{\text{retrain}} = \mathcal{R}^\text{retrain}_{(\lceil(m_2+1)(1-\alpha^{\text{retrain}})\rceil)} =  |\boldsymbol{X}^\top (\beta^\circ - \hat \beta) + \epsilon| \sim \text{abs}\mathcal{N}(0, s_2^2)$, 其中 $s_2^2 = \|\beta^\circ - \hat \beta\|^2_{\Sigma_X} + \sigma^2$

直觉上, 若 $\hat\beta$ 拟合的好 ($\hat \beta \approx \beta^\circ$) 则 $s_2^2$ 越小, retrain CP 会结构性的消除模型误差, 从而提升性能.

下依据如下思路给出理论分析: 对于这两个区间长度, 其作为随机变量, 我们给出 $\mathcal{R}^\text{vanilla}$ 一个 lower bound with high probability (w.h.p.), 然后给出 $\mathcal{R}^\text{retrain}$ 的 upper bound w.h.p.. 若在该保守条件下, 仍然能说明 $\text{LB}(\mathcal{R}^\text{vanilla}) > \text{UB}(\mathcal{R}^\text{retrain})$, 则说明 retrain CP 在该条件下优于 vanilla CP, w.h.p..

### High Probability Bounds of Interval Lengths

**Theorem 5.1 (High Probability Bounds of Interval Lengths)**: 给定任意置信水平 $\gamma \in (0,1)$ 及 miscoverage rate $\alpha \in (0,1)$, 可分别给出二者的 bounds 如下:
- $(1-\gamma)$ upper bound of $\mathcal{R}^\text{retrain}$:
  $$
  \mathcal{R}^\text{retrain}\leq \mathbb{E}[\mathcal{R}^\text{retrain}_{(\lceil(m_2+1)\alpha^{\text{ret}}\rceil)}] + \sqrt{2\pi} s_2 \left(\sqrt{\frac{\log(1/\gamma)}{m_2\alpha}} +\frac{\log(1/\gamma)}{m_2\alpha}  \right) \quad (8)
  $$
- $(1-\gamma)$ lower bound of $\mathcal{R}^\text{vanilla}$:
  $$
   \mathcal{R}^\text{vanilla}\geq \mathbb{E}[\mathcal{R}^\text{vanilla}_{(\lceil(m+1)\alpha^{\text{van}}\rceil)}] - \sqrt{2\pi} s_1 \left(\sqrt{\frac{(1-\alpha)\log(1/\gamma)}{m\alpha^2}} \right) \quad (9)
   $$

其中, $s_1^2 = \| \beta^\circ \|_{\Sigma_X}^2 + \sigma_\epsilon^2$, $s_2^2 = \| \beta^\circ - \hat \beta \|_{\Sigma_X}^2+\sigma_\epsilon^2$ 分别为各自 residual 的方差. 

这里注意到, 在 $(8)$ 中, $m_2$ 作为分母, 其越大, $\sqrt{\frac{\log(1/\gamma)}{m_2\alpha}}$ 越小, 即 UB 越紧. 但与此同时, 这也意味着 $\hat \beta$ 的数据量变少, 使得 $s_2^2$ 变大. 因此这里存在一个 trade-off. 故下文将给出一个 $\xi$ 的选择.

### Valid Split Ratio

本小节的核心目的是在于给出一个最优的 retrain 比例 $\xi = \frac{m_1}{m}$ 的选择. 在操作中, 相当于给出 $\mathbb{E}[\mathcal{R}^\text{retrain}_{(\lceil(m_2+1)\alpha^{\text{ret}}\rceil)}]$ 与 $s_2$ 的具体表达, 这样就可以将 UB 表示成一个关于 $\xi$ 的函数, 进而求解最优的 $\xi$.

首先给出如下 lemma, 其给出了 absolute Gaussian 分布的分位数之期望的显式近似. 

**Lemma 5.2 (Approximation for Expectation of Absolute Gaussian Order Statistics)**: 设 $X_1, X_2, \ldots, X_m \sim \text{abs}\mathcal{N}(0,\sigma^2)$. 分别记 $\phi(x) = \sqrt{2/\pi} \exp(-x^2/2)$ 和 $\Phi(x) = \int_{-\infty}^x \phi(t) dt$ 为标准 absolute Gaussian 分布的 pdf 和 cdf. 则对于任意 $\alpha \in (0,1)$, $k = \lceil (m+1)\alpha\rceil$, 有:
$$\begin{aligned}
\mathbb{E}[X_{(k)}] &= Q_{1-\alpha}(X) - \sigma\frac{\Phi^{-1}(1-\alpha)\alpha(1-\alpha)}{2\phi^2(\Phi^{-1}(1-\alpha))}\frac{1}{m} + \mathcal{o}\left(\frac{1}{m}\right) \\&:= Q_{1-\alpha}(X) - \sigma c(\alpha)\frac{1}{m} + \mathcal{o}\left(\frac{1}{m}\right)
\end{aligned}$$

> 这里简要对上述 Lemma 进行说明. 
> - **首先, 不妨只考虑归一化之情形**. $Z_i \sim \text{abs}\mathcal{N}(0,1)$. 否则只需令 $X_i = \sigma Z_i$, 即 $\mathbb{E}[X_{(k)}] = \sigma \mathbb{E}[Z_{(k)}]$ 即可. 
> - **将顺序统计量 $Z_{(k)}$ 映射回 Uniform 分布**. 设 $U_i = \Phi(Z_i)$, 则 $U_i \sim \text{Uniform}(0,1)$. 由统计学之基础知识, 有 $U_{(k)} \sim \text{Beta}(k, m+1-k)$. 且由 Beta 分布的性质, 有 $\mathbb{E}[U_{(k)}] = \frac{k}{m+1}$, $\mathbb{V}[U_{(k)}] = \frac{k(m+1-k)}{(m+1)^2(m+2)}$.  
>   - 注意到, 在 Conformal Prediction 中, 通常取 $k = \lceil (m+1)(1-\alpha)\rceil$, 故 $\mathbb{E}[U_{(k)}]= \frac{k}{m+1} \approx 1-\alpha$, $\mathbb{V}[U_{(k)}] = \frac{\alpha(1-\alpha)}{m} + \mathcal{O}(\frac{1}{m^2}) \asymp \frac{(1-\alpha)\alpha}{m}$.
> - **用 Delta Method 将 $U_{(k)}$ 映射回 $Z_{(k)}$**. 由 Delta Method, 由于 $U_{(k)} = \Phi(Z_{(k)})$, 故 $Z_{(k)} = \Phi^{-1}(U_{(k)})$. 由 Delta Method, 有
> $$Z_{(k)}=\Phi^{-1}(U_{(k)}) \approx \Phi^{-1'}(1-\alpha) +  \frac{1}{2} \Phi^{-1''}(1-\alpha) \mathbb{V}[U_{(k)}]$$

在承认上述 lemma 之后, 我们尝试套用到 retrain 上. 
- 具体地, 令 $X_i \leftarrow \mathcal{R}^\text{retrain}_i \sim \text{abs}\mathcal{N}(0,s_2^2)$, $\sigma^2 \leftarrow s_2^2$, $m \leftarrow m_2$, 则有:
  $$\mathbb{E}[\mathcal{R}^\text{retrain}_{(\lceil(m_2+1)\alpha^{\text{ret}}\rceil)}] = Q_{1-\alpha^{\text{ret}}}(\mathcal{R}^\text{retrain}) - s_2 c(\alpha^{\text{ret}})\frac{1}{m_2} + \mathcal{o}\left(\frac{1}{m_2}\right) \quad (10)$$

- 而根据如下两个事实: 1. 绝对高斯的分位数 $Q_{1-\alpha}(Z) = \Phi^{-1}(1-\alpha)$; 2. 分位数的线性缩放性质 $Q_{1-\alpha}(aZ) = a Q_{1-\alpha}(Z)$, 则有:
  $$Q_{1-\alpha}(\mathcal{R}^\text{retrain}) = s_2 \Phi^{-1}(1-\alpha)$$

- 将其代入 $(10)$ 中, 则有:
  $$\mathbb{E}[\mathcal{R}^\text{retrain}_{(\lceil(m_2+1)\alpha\rceil)}] \approx s_2 \left(\Phi^{-1}(1-\alpha) - c(\alpha)\frac{1}{m_2}\right) \quad (11)$$

- 再将之整理至 Retrain 的 UB $(8)$ 中, 则有:
  $$\begin{aligned}
\text{UB}^{\text{retrain}} &\approx s_2 \left[ \Phi^{-1}(1-\alpha) - c(\alpha)\frac{1}{m_2} + \sqrt{2\pi} \left(\sqrt{\frac{\log(1/\gamma)}{m_2\alpha}} +\frac{\log(1/\gamma)}{m_2\alpha} \right) \right] 
\end{aligned}$$

目前只剩下 $s_2 = \sqrt {\|\beta^\circ - \hat \beta\|^2_{\Sigma_X} + \sigma_\epsilon^2}:=\sqrt{\Delta + \sigma_\epsilon^2}$ 需要处理. 
- 对形如 $f(x) = \sqrt{x^2 + a^2}$ 的函数, 关于 $x=\sigma_\epsilon$ 处进行一阶泰勒展开: $f(x) = f(\sigma_\epsilon) + f'(\sigma_\epsilon)(x-\sigma_\epsilon) + \mathcal{O}((x-\sigma_\epsilon)^2) = \sigma_\epsilon + \frac{\Delta}{2\sigma_\epsilon} + \mathcal{O}(\Delta^2)$. 对应这里,
  $$s_2 = \sqrt{\Delta + \sigma_\epsilon^2} \leq \sigma_\epsilon + \frac{\Delta}{2\sigma_\epsilon}$$
- 另一方面, 关于 $\Delta = \|\beta^\circ - \hat \beta\|^2_{\Sigma_X}$, 其有结论 $\mathbb{E}[\Delta] \lesssim \frac{d}{m_1} = \frac{d}{\xi m}$, 这里 $d$ 为特征维度. 
- 因此, 综上, 有:
  $$s_2 \leq \sigma_\epsilon + \frac{d}{2\sigma_\epsilon \xi m}$$


再对整体的 UB 进行整理, 得到
$$\text{UB}^{\mathrm{ret}}
\ \lesssim\
\Big\{\sigma_\epsilon+\frac{d}{2\sigma_\epsilon m}\cdot\frac{1}{\xi}\Big\}
\cdot
\Bigg[
\underbrace{\Phi^{-1}(1-\alpha)}_{\text{总体分位数}}
-\underbrace{\frac{c(\alpha)}{m(1-\xi)}}_{\text{经验分位数的 }O(1/m)\text{ 偏差}}
+\underbrace{\sqrt{2\pi}\left(
\sqrt{\tfrac{\log(1/\gamma)}{\alpha\,m(1-\xi)}}+\tfrac{\log(1/\gamma)}{\alpha\,m(1-\xi)}
\right)}_{\text{集中误差，样本越多越小}}
\Bigg].$$

再最终进行代数整理, 得到:
$$l(\xi)=\Big\{\sigma+\frac{d}{2\sigma m}\cdot\frac{1}{\xi}\Big\}
\cdot\Bigg\{
\Phi^{-1}(1-\alpha)
+\sqrt{\frac{2\pi\log(1/\gamma)}{\alpha\,m(1-\xi)}}
+\frac{\log(1/\gamma)}{\alpha\,m(1-\xi)}
-\frac{\Phi^{-1}(1-\alpha)\,\alpha(1-\alpha)}{2\,m(1-\xi)\,\phi(\Phi^{-1}(1-\alpha))^2}
\Bigg\}.\quad{(11)}$$

经过一系列的推导, 我们有下述结论:
1. 只要 $\xi$ 不在极端边界, $l(\xi)$ 往往较平, 即 retrain CP 对 $\xi$ 不敏感.
2. 对于最优的 $\xi^*$ 的确定需要求解二次方程, 其计算过程相对繁琐. 
3. 在实践中, 往往对 $\xi$ 进行简单 grid search 即可.

### When Retrained CP Brings Improvement?

在这里通过定义误差的信号强度 SNR 来给出改进条件. 

回顾在 linear residual model 下, vanilla 残差为:
$$\mathcal{R}^\text{vanilla} = \boldsymbol{X}^\top \beta^\circ + \epsilon, \quad \boldsymbol{X} \sim \mathcal{N}(0,\Sigma_X), \epsilon \sim \mathcal{N}(0,\sigma_\epsilon^2)$$
因此定义 SNR 为:
$$\text{SNR} := \frac{\mathbb{V}[\boldsymbol{X}^\top \beta^\circ]}{\mathbb{V}[\epsilon]} = \frac{\|\beta^\circ\|^2_{\Sigma_X}}{\sigma_\epsilon^2}$$
- SNR 越大, 说明残差中有被 $g_\theta$ 捕捉的信号成分越大, retrain CP 的提升空间越大.


***Theorem 5.4***

给定任意置信水平 $\gamma \in (0,1)$ 及 miscoverage rate $\alpha \in (0,1)$, 只要
$$\frac{\|\boldsymbol\beta^\circ\|_{\Sigma_x}^2}{\sigma_\epsilon^2} \ge
\frac{2\sqrt{2\pi}}{\alpha\,\Phi^{-1}(1-\alpha)\,\sqrt m\ -\ \sqrt{2\pi(1-\alpha)\log(1/\gamma)}}\Big[\sqrt{2\alpha\log(1/\gamma)}+\sqrt{(1-\alpha)\log(1/\gamma)}\Big]$$
则 vanilla CP UB w.h.p. 大于 retrain CP LB, 从而以至少 $1-\gamma$ 的概率有:
$$\text{Length}(C^{\text{vanilla}}(\boldsymbol{X})) > \text{Length}(C^{\text{retrain}}(\boldsymbol{X}))$$


在实践中, 我们如果能从数据中估计出 $\widehat{\text{SNR}}$, 则可以根据上式判断 retrain CP 是否有提升的可能性:
1. 在 $\mathcal{S}_2^{\text{ret}}$ 上拟合 $\hat f$ 得到 vanilla 的残差 $\hat{\mathcal{R}}^{\text{vanilla}}_i = y_i - \hat f(\boldsymbol{x}_i)$.
2. 拟合一个带有截距的线性模型 $r\approx \boldsymbol{X}^\top \beta + b$.
3. 计算:
   - $\widehat{\text{sig}} = \mathbb{V}[\boldsymbol{X}^\top \hat \beta]$
   - $\widehat{\sigma_\epsilon^2} = \mathbb{V}[\hat{\mathcal{R}}^{\text{vanilla}} - r]$
4. 计算 $\widehat{\text{SNR}} = \frac{\widehat{\text{sig}}}{\widehat{\sigma_\epsilon^2}}$.