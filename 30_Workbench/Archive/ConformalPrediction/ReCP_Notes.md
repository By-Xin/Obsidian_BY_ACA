# Retrain CP (NEW)

## 1. Background

对于任意的黑箱模型 $\hat{f}: \mathcal{X} \to \mathcal{Y}$ (训练 $\hat{f}$ 的数据未知), 和 (理想情况下) 与原始数据独立同分布的 calibration dataset $\{(\boldsymbol{x}_i, y_i)\}_{i=1}^n$, CP 的宗旨是构造一个 set predictor: $\mathcal{C}(\cdot): \mathcal{X} \to 2^{\mathcal{Y}}$，使得 marginal coverage 得到保证：
$$
\mathbb{P}\{Y \in \mathcal{C}(\mathbf{X})\} \geq 1 - \alpha.
$$
- 这里 $2^{\mathcal{Y}}$ 表示 $\mathcal{Y}$ 的幂集, 即所有 $\mathcal{Y}$ 的可行子集的集合.
  - 若 $\mathcal{Y} = \{1, 2, \ldots, K\}$ 是分类问题的标签空间, 则 $2^{\mathcal{Y}}$ 包含 $2^K$ 个元素 (集合的构成由每个标签是否包含在内($0/1$)决定).
  - 若 $\mathcal{Y} = \mathbb{R}$ 是回归问题的标签空间, 则严格说应当是 $2^{\mathbb{R}}$, 但实际中通常只考虑 Borel 集合族 $\mathcal{B}(\mathbb{R})$ (区间, 可数并等构成的集合) . 
- Marginal coverage 的概率是对 $(\mathbf{X}, Y) \sim P$ 这一联合随机性求得的, 没有对具体的 $\mathbf{X} = \boldsymbol{x}$ 条件化. 这意味着 coverage 保证是对所有可能的输入 $\mathbf{X}$ 的平均意义上的保证, 这一要求相对宽松.

***Vanilla CP***

重新考虑回 Vanilla CP 的构造. 考虑 homogeneous linear regression (可以简单理解为标准的服从 Gaussian-Markov 假设的线性回归模型):
$$
Y = f(\boldsymbol{X}) \boldsymbol{\beta} + \epsilon, \quad \epsilon \stackrel{i.i.d.}{\sim} \mathcal{N}(0, \sigma^2).
$$

Vanilla CP 的核心思想是利用 calibration dataset 来构造一个 nonconformity score function $s(\boldsymbol{x}, y; \hat{f}): \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$, 该函数衡量标签 $y$ 对于输入 $\boldsymbol{x}$ 来说与模型预测 $\hat{f}(\boldsymbol{x})$ 的不符合程度. 对于回归问题, 一个常用的 nonconformity score 是绝对误差: $s(\boldsymbol{x}, y; \hat{f}) = |y - \hat{f}(\boldsymbol{x})|$. 总的而言, CP 的构造过程如下:
1. 在 calibration dataset 上计算 nonconformity scores:
   $$\{s_i = s(\boldsymbol{x}_i, y_i; \hat{f})\}_{i=1}^n.$$
2. 计算 nonconformity scores 的经验分布的 $(1 - \alpha)$-分位数 $Q_{1-\alpha}(\{s_i\}_{i=1}^n)$ (排序后选择第 $\lceil (1 - \alpha)(n + 1) \rceil$ 小的 $s_i$).
3. 构造的 set predictor 为:
   $$
   \hat{\mathcal{C}}(\boldsymbol{x}) = [\hat{f}(\boldsymbol{x}) - Q_{1-\alpha}(\{s_i\}_{i=1}^n), \hat{f}(\boldsymbol{x}) + Q_{1-\alpha}(\{s_i\}_{i=1}^n)].
   $$

***Oracle Conformal Predictor***

这个构造的 $\hat{\mathcal{C}}(\cdot)$ 相当于是对 oracle predictor $\hat{f}(\cdot)$ 的预测区间的一个逼近. 理论上, 如果我们知道 oracle predictor $f(\cdot)$ 以及其误差分布 $\epsilon$ (且在 homogeneous linear regression 的假设下), 则可以直接构造出满足 coverage 要求的预测区间:
$$
\mathcal{C}^*(\boldsymbol{x}) = [f(\boldsymbol{x}) - Q_{1-\alpha}(|\epsilon|), f(\boldsymbol{x}) + Q_{1-\alpha}(|\epsilon|)], 
$$
- 其中 $Q_{1-\alpha}(|\epsilon|)$ 是 $|\epsilon|$ 的 $(1 - \alpha)$-分位数. 这相当于是我们的模型拟合是完美的, 只需要考虑噪声的影响 (而不需要考虑拟合 $\hat{f}$ 与 $f$ 之间的差异). 
- 这个 oracle conformal predictor 是满足覆盖率要求下最 efficient 和 adaptive 的 predictor (因为它只考虑了噪声的影响, 而没有额外的冗余).


***Problem with Vanilla CP***

现有的 Vanilla CP 方法多是在 $\hat{f}$ 是一个 given 的黑箱模型的前提下进行的, 且通常会假设其是 consistent 的, 证明当 calibration dataset 样本量 $n \to \infty$ 时, Vanilla CP 的预测区间会收敛到 oracle 的长度. 

然而, 在实际应用中, 我们并不知道 $\hat{f}$ 的实际性质, 因此就会导致一些问题. 在具体展开分析前, 我们通过以下分解来进一步说明问题:
$$
Y - \hat{f}(\boldsymbol{X}) = \underbrace{f(\boldsymbol{X}) - \hat{f}(\boldsymbol{X})}_{\text{epistemic uncertainty}} + \underbrace{\epsilon}_{\text{aleatoric uncertainty}}.
$$
- Aleatoric uncertainty (数据内在噪声) 是无法通过增加数据量来消除的.
- Epistemic uncertainty (模型不确定性) 是由于模型 $\hat{f}$ 与真实函数 $f$ 之间的差异引起的, 是模型没有学习到的系统性误差. 这种不确定性可以通过更好的拟合, 更多的数据, 或者更好的模型来减少.

在 Vanilla CP 中, 我们使用的分位数来自 $|Y - \hat{f}(\boldsymbol{X})|$ 的经验分布. 因此根据上面的分解, 当 epistemic uncertainty $f(\boldsymbol{X}) - \hat{f}(\boldsymbol{X})$ 较大时, 这个分位数会被拉大, 导致预测区间过宽, 偏离了 $Q_{1-\alpha}(|\epsilon|)$ (oracle 的分位数). 这就意味着, 当模型 $\hat{f}$ 的拟合效果不好时, Vanilla CP 会在估计的准确性和预测区间的宽度之间产生权衡, 可能会导致预测区间过宽, 从而降低了预测的实用性. 这也可以通过残差的两阶矩来理解:
$$\begin{aligned}
\mathbb{E}[(Y - \hat{f}(\boldsymbol{X}))]  &= \mathbb{E}[f(\boldsymbol{X}) - \hat{f}(\boldsymbol{X})] + \mathbb{E}[\epsilon] ,\\
\mathbb{V}[Y - \hat{f}(\boldsymbol{X})] &= \mathbb{V}(f(\boldsymbol{X}) - \hat{f}(\boldsymbol{X})) + \sigma^2.
\end{aligned}$$

而上述问题经常会在如下两种情况下出现:
1. **Model misspecification**: 当 $\hat{f}$ 的假设空间与真实数据生成过程不匹配时, 例如使用线性模型去拟合高度非线性的数据, 会导致较大的 epistemic uncertainty.
2. **Distribution shift**: 当 calibration dataset 与训练 $\hat{f}$ 所用的数据分布不一致时, 例如在不同的时间段或环境下收集的数据, 会导致模型在 calibration dataset 上的表现较差, 从而增加 epistemic uncertainty.

***Retrain CP***

为了解决上述问题, 我们提出了 Retrain CP 的方法. 该方法的核心思想是利用一部分 calibration dataset 来重新训练模型 $\hat{f}$, 以减少 epistemic uncertainty, 然后再使用剩余的 calibration dataset 来构造 nonconformity scores 和预测区间. 

幸运的是, 构建 nonconformity scores 的过程是一个一维的分布估计, 分位数收敛的速度相对较快, 在有限的数据下也能得到较好的估计. 因此, 我们可以牺牲一部分 calibration dataset 来重新训练模型, 以获得更好的拟合效果, 从而减少 epistemic uncertainty, 并最终得到更窄的预测区间.

整体而言, 我们的 Retrain CP 方法包括以下步骤:
1. 将 calibration dataset $\mathcal{S}$ 随机划分为 $\mathcal{S}_1$ 和 $\mathcal{S}_2$.
2. 使用 $\mathcal{S}_1$ 来重新训练模型, 对模型进行 debias, 得到新的模型 $\hat{f}_{\text{retrain}}$.
3. 在 $\mathcal{S}_2$ 上计算 nonconformity scores, 进行分位数估计, 构造预测区间. 

---

## Ideas
- 根据上面的讨论, 我们问题的核心其实就在于这个经验分位数 $Q_{1-\alpha}(\{s_i\}_{i=1}^n)$  (事实上区间的长度也就是 $2Q$), 而再进一步其本质就在于我们定义的 score $s_i$  的分析. 
- 理论上, 我们预期会有如下表现:
	- 对于 $s$ 的分布, retrain 相比 vanilla cp 会更集中在0附近, 尾部更薄, 故相对应的 quantile 更小;
	  ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202511131850372.png)
	- 对于 $Q$ 的分布, retrain 相比 vanilla cp 会更厚尾分布 (因为我们牺牲了一部分的对于 $Q$ 的估计的数据), 但是其分布的均值会相比之下更小. 
		- 我们的核心就相当于, 对这两个 $Q$ 的分布, 各自取一个 CI. 研究这两个置信区间何种条件下能够显著的产生差异. 