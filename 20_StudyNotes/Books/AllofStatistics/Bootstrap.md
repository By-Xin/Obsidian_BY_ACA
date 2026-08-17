---
aliases: [Bootstrap, 自助法, Bootstrap方差估计, Bootstrap置信区间]
tags:
  - concept
  - ml/stats
  - math/probability
source: "All of Statistics, Larry Wasserman"
related_concepts:
  - "[[Central Limit Theorem]]"
  - "[[Confidence Interval]]"
---

# Bootstrap

Bootstrap 的本质是一种通过计算换取推断的策略. 其尝试解决当我们不知道数据背后的真实分布 $\mathcal{F}$ 时, 如何获取统计量的分布以估计其方差或置信区间的问题.

## Intuition

首先给出记号和问题设定. 设我们有数据 $X_1, X_2, \ldots, X_n$ 来自某个未知分布 $\mathcal{F}$, 我们想要估计某个统计量 $T_n = g(X_1, X_2, \ldots, X_n)$ 的方差 $\mathbb{V}_{\mathcal{F}}[T_n]$. 这里 $\mathbb{V}_{\mathcal{F}}$ 强调数据来自分布 $\mathcal{F}$ 时的方差. 由于我们不知道 $\mathcal{F}$, 无法直接计算 $\mathbb{V}_{\mathcal{F}}[T_n]$:
- 传统的做法常假设 $\mathcal{F}$ 属于正态分布等特殊参数分布或 $n$ 足够大以服从中心极限定理. 
- Bootstrap 是一个非参数的估计方法, 在不作出分布假设或样本量不足, 或者统计量过于复杂时, 仍然可以估计 $\mathbb{V}_{\mathcal{F}}[T_n]$. 

一个具体的例子是 Bernoulli 样本下 log-odds 估计量的方差. 
- 设 $X_1, X_2, \ldots, X_n$ 独立同分布于 $\text{Bernoulli}(p)$, $p \in (0,1)$, 同时记样本比例 $\hat{p}_n = \frac{1}{n} \sum_{i=1}^n X_i$. 定义 log-odds 函数 $h(p) = \log\frac{p}{1-p}$ 的估计量为 $T_n = h(\hat{p}_n) = \log \frac{\hat{p}_n}{1 - \hat{p}_n}$. 这里的目标是估计其方差 $\mathbb{V}_{\mathcal{F}}[T_n]$. 
- 对于传统做法, 由于 log-odds 估计量比较复杂, 很难直接计算其方差. 因此需要用 CLT + Delta 方法近似其分布:
  - 由于 $X_i\sim \text{Bernoulli}(p)$, 且 $\mathbb{E}[X_i] = p$, $\mathbb{V}[X_i] = p(1-p)$, 根据 CLT 可知 $\sqrt{n}(\hat{p}_n - p) \xrightarrow{d} \mathcal{N}(0, p(1-p))$.
  - 由 Delta 方法保障, 只要 $h(\cdot)$ 在点 $p$ 处可微, 则有 $\sqrt{n}(h(\hat{p}_n) - h(p)) \xrightarrow{d} \mathcal{N}\left(0, [h'(p)]^2 p(1-p)\right)$. 计算导数 $h'(p) = \frac{1}{p(1-p)}$, 可得
    $$\sqrt{n}(T_n - h(p))\xrightarrow{d}\mathcal{N}\left(0,\ p(1-p)\left[\frac{1}{p(1-p)}\right]^2\right)=\mathcal{N}\left(0,\ \frac{1}{p(1-p)}\right).$$
  - 因此, 当 $n$ 足够大时, 可近似地认为 $T_n$ 服从正态分布, 故有渐进方差
    $$\mathbb{V}_{\mathcal{F}}[T_n] \approx \frac{1}{n p(1-p)}.$$
  - 再进一步, 由于 $p$ 未知, 可用样本比例 $\hat{p}_n$ 替代 $p$, 得到方差估计
    $$\widehat{\mathbb{V}}_{\mathcal{F}}[T_n] = \frac{1}{n \hat{p}_n (1 - \hat{p}_n)}.$$
    
- 而对于 Bootstrap 方法, 其则通过如下核心两个步骤来估计 $\mathbb{V}_{\mathcal{F}}[T_n]$:
  1. **Estimation**: 用经验分布 $\hat{\mathcal{F}}_n= \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{X_i \leq x\}$ (即在样本点处取值为 $\frac{1}{n}$, 其他位置取值为 $0$ 的分布) 近似真实分布 $\text{Bernoulli}(p)$. 即假设:
        $$\mathbb{P}_{\hat{\mathcal{F}}_n}(X = 1) = \hat{p}_n,\quad \mathbb{P}_{\hat{\mathcal{F}}_n}(X = 0) = 1 - \hat{p}_n.$$
  2.  **Resample Approximation**: 重复 $B$ 次 Bootstrap 采样, 每次都计算一次统计量 $T_n^{* (b)}= \log\frac{\hat{p}_n^{* (b)}}{1 - \hat{p}_n^{* (b)}}$, 其中 $\hat{p}_n^{* (b)}$ 是从经验分布 $\hat{\mathcal{F}}_n$ 中采样得到的样本比例, $b = 1, 2, \cdots, B$. 最后用这些 Bootstrap 统计量的样本方差来近似 $\mathbb{V}_{\mathcal{F}}[T_n]$:
        $$\widehat{\mathbb{V}}_{\mathcal{F}}[T_n] = \frac{1}{B-1} \sum_{b=1}^B \left(T_n^{* (b)} - \bar{T}_n^*\right)^2,$$
    其中 $\bar{T}_n^* = \frac{1}{B} \sum_{b=1}^B T_n^{* (b)}$.

> [Visualization](bs.html)


## Simulation

Bootstrap 本质上是 Monte Carlo Simulation 的一种特殊形式. 因此其有关性质可以通过 Simulation 的角度来理解和分析. 具体来说:
- 若我们能够从一个分布 $\mathcal{F}$ 中采样, 则可以通过大量采样来近似计算这个部分下的均值/方差等统计量.
- Bootstrap 的关键在于, 当我们不知道 $\mathcal{F}$ 时, 我们用经验分布 $\hat{\mathcal{F}}_n$ 来近似 $\mathcal{F}$, 然后从 $\hat{\mathcal{F}}_n$ 中采样来进行 Simulation. 

假设我们能够从 $\mathcal{F}$ 中采样 $B$ 次, 得到样本 $Y_1, Y_2, \ldots, Y_B\stackrel{\text{i.i.d.}}{\sim} \mathcal{F}$. 定义样本均值 $\bar{Y}_B = \frac{1}{B} \sum_{b=1}^B Y_b$, 则根据大数定律, 若 $\mathbb{E}[Y_b] < \infty$, 则有
$$\bar{Y}_B \xrightarrow{a.s.} \mathbb{E}_{\mathcal{F}}[Y],\quad \text{if } B \to \infty;$$
更一般地, 若任意可积函数 $h$ 满足 $\mathbb{E}_{\mathcal{F}}[|h(Y)|] < \infty$, 则有
$$\frac{1}{B} \sum_{b=1}^B h(Y_b) \xrightarrow{a.s.} \mathbb{E}_{\mathcal{F}}[h(Y)],\quad \text{if } B \to \infty.$$
- 因此可以立即推出, 对于方差的估计, 若 $\mathbb{V}_{\mathcal{F}}[Y] < \infty$, 则有
  $$\frac{1}{B-1} \sum_{b=1}^B (Y_b - \bar{Y}_B)^2 \xrightarrow{a.s.} \mathbb{V}_{\mathcal{F}}[Y],\quad \text{if } B \to \infty.$$


## Bootstrap Variance Estimation

正如前文所说, 我们现在有两层世界观:
- **真实世界**: 数据 $X_1, X_2, \ldots, X_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{F}$ 来自某个未知的真实分布 $\mathcal{F}$. 我们关心统计量 $T_n = g(X_1, X_2, \ldots, X_n)$ 在真实分布 $\mathcal{F}$ 下的方差 $\mathbb{V}_{\mathcal{F}}[T_n]$.
  \[\boxed{\mathcal{F}
\;\Rightarrow\;
X_1,\ldots,X_n
\;\Rightarrow\;
T_n=g(X_1,\ldots,X_n)}\]
  - 我们无法重复采样 $X_1, X_2, \ldots, X_n$ 来估计 $\mathbb{V}_{\mathcal{F}}[T_n]$, 因为我们只有一组数据.
- **Bootstrap 世界**: 我们用经验分布 $\hat{\mathcal{F}}_n$ 来近似真实分布 $\mathcal{F}$. 然后从经验分布 $\hat{\mathcal{F}}_n$ 中采样 $B$ 次, 得到 Bootstrap 样本 $X_1^{* (b)}, X_2^{* (b)}, \ldots, X_n^{* (b)} \stackrel{\text{i.i.d.}}{\sim} \hat{\mathcal{F}}_n$, $b = 1, 2, \ldots, B$. 定义 Bootstrap 统计量 $T_n^{* (b)} = g(X_1^{* (b)}, X_2^{* (b)}, \ldots, X_n^{* (b)})$. 我们用这些 Bootstrap 统计量的样本方差来近似 $\mathbb{V}_{\mathcal{F}}[T_n]$:
    $$\widehat{\mathbb{V}}_{\mathcal{F}}[T_n] = \frac{1}{B-1} \sum_{b=1}^B \left(T_n^{* (b)} - \bar{T}_n^*\right)^2,$$
其中 $\bar{T}_n^* = \frac{1}{B} \sum_{b=1}^B T_n^{* (b)}$.
  \[\boxed{\hat{\mathcal{F}}_n
\;\Rightarrow\;
X_1^{* (b)},\ldots,X_n^{* (b)}
\;\Rightarrow\;
T_n^{* (b)}=g(X_1^{* (b)},\ldots,X_n^{* (b)})}\]
  - 这个过程可以重复进行 $B$ 次, 因为给定原始数据之后 $\hat{\mathcal{F}}_n$ 是已知的, 可以任意多次采样以得到更好的估计.

而上述 $X_1^{* (b)}, X_2^{* (b)}, \ldots, X_n^{* (b)} \stackrel{\text{i.i.d.}}{\sim} \hat{\mathcal{F}}_n$ 的采样过程, 恰恰等价于从原始样本 $X_1, X_2, \ldots, X_n$ 中有放回地采样 $n$ 次. 这是因为经验分布 $\hat{\mathcal{F}}_n$ 在每个样本点处的概率均为 $\frac{1}{n}$. 故最终可以总结 Bootstrap 方差估计的步骤为:

1. 从原始样本 $X_1, X_2, \ldots, X_n$ 中有放回地采样 $n$ 次 (**注意这里的采出样本数量与原始样本数量相同**), 得到 Bootstrap 样本 $X_1^{* (b)}, X_2^{* (b)}, \ldots, X_n^{* (b)}$.
2. 根据 Bootstrap 样本计算统计量 $T_n^{* (b)} = g(X_1^{* (b)}, X_2^{* (b)}, \ldots, X_n^{* (b)})$.
3. 重复步骤 1 和 2 共 $B$ 次, 得到 $T_n^{* (1)}, T_n^{* (2)}, \ldots, T_n^{* (B)}$.
4. 用这些 Bootstrap 统计量的样本方差来近似 $\mathbb{V}_{\mathcal{F}}[T_n]$:
   $$\widehat{\mathbb{V}}_{\mathcal{F}}[T_n] = \frac{1}{B-1} \sum_{b=1}^B \left(T_n^{* (b)} - \bar{T}_n^*\right)^2,$$
其中 $\bar{T}_n^* = \frac{1}{B} \sum_{b=1}^B T_n^{* (b)}$.

这里要注意, Bootstrap 方差估计的准确性依赖于两个方面:
- 通过样本的经验分布 $\hat{\mathcal{F}}_n$ 来近似真实分布 $\mathcal{F}$ 的准确性. 这个误差和样本量 $n$ 有关, 然而往往由于我们只能在给定的样本下工作, 这个误差是无法消除的.
- 通过有限次采样 $B$ 进行 Monte Carlo 估计的准确性. 这个误差可以通过增加采样次数 $B$ 来减小.
  
即有如下近似链:
\[\mathbb{V}_{\mathcal{F}}[T_n]
\stackrel{\text{B.S. (not so small)}}{\approx} \mathbb{V}_{\hat{\mathcal{F}}_n}[T_n]
\stackrel{\text{M.C. (very small)}}{\approx} \widehat{\mathbb{V}}_{\hat{\mathcal{F}}_n}[T_n].\]

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20251212183047.png)

## Bootstrap Confidence Intervals

Bootstrap 也可以用来构造统计量的置信区间. 假设数据 $X_1, X_2, \ldots, X_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{F}$ 来自某个未知分布 $\mathcal{F}$, 我们关心统计量 $\theta = T(\mathcal{F})=g(X_1, X_2, \ldots, X_n)$ 的置信区间. 其中估计量是 $\hat\theta=T(\hat{\mathcal{F}}_n)$. 我们希望构造一个区间 $[L, U]$ 使得
\[
\mathbb{P}_{\mathcal{F}}(L \leq \hat\theta \leq U) \approx 1 - \alpha.
\]
其中 $1 - \alpha$ 是置信水平.


常见的方法有以下几种:
1. Normal Approximation Method: 基于 Bootstrap 方差估计, 通过正态近似来构造置信区间. 
2. Percentile Method: 直接利用 Bootstrap 统计量的分位数来构造置信区间.
3. Pivotal Method: 构造一个枢轴量 (pivotal quantity), 其分布不依赖于未知参数, 然后利用 Bootstrap 来估计该枢轴量的分布以构造置信区间.

不过不论何种方法, 都依赖于 Bootstrap 采样过程来近似统计量的分布:
1. 从原始样本 $X_1, X_2, \ldots, X_n$ 中有放回地采样 $n$ 次, 得到 Bootstrap 样本 $X_1^{* (b)}, X_2^{* (b)}, \ldots, X_n^{* (b)}$.
2. 根据 Bootstrap 样本计算统计量 $\hat\theta^{* (b)} = g(X_1^{* (b)}, X_2^{* (b)}, \ldots, X_n^{* (b)})$.
3. 重复步骤 1 和 2 共 $B$ 次, 得到 $\hat\theta^{* (1)}, \hat\theta^{* (2)}, \ldots, \hat\theta^{* (B)}$.

### Normal Approximation Method

最简单的方法是基于 Bootstrap 方差估计, 通过正态近似来构造置信区间. 具体来说, 假设我们已经通过 Bootstrap 得到统计量 $ \hat\theta $ 的方差估计 $\widehat{\mathbb{V}}_{\mathcal{F}}[\hat\theta]$, 则可以构造 $100(1-\alpha)\%$ 的置信区间为:
$$\left[\hat\theta - z_{\alpha/2} \sqrt{\widehat{\mathbb{V}}_{\mathcal{F}}[\hat\theta]},\quad \hat\theta + z_{\alpha/2} \sqrt{\widehat{\mathbb{V}}_{\mathcal{F}}[\hat\theta]}\right],$$
其中 $z_{\alpha/2}$ 是标准正态分布的 $1 - \alpha/2$ 分位数.

然而, 这种方法依赖于统计量 $\hat\theta$ 服从近似正态分布的假设, 对于某些复杂或偏态的统计量, 该假设可能不成立, 从而导致置信区间的覆盖率不准确.
### Percentile Method

另一类思路是直接利用 Bootstrap 的结果来构造置信区间, 而不再依赖于正态近似. 回顾置信区的定义, 我们希望找到区间 $[L, U]$ 使得
$$\mathbb{P}_{\mathcal{F}}(L \leq \hat\theta \leq U) \approx 1 - \alpha.$$
粗略地讲, 就是若重复采样多次, 则有约 $100(1-\alpha)\%$ 的区间会包含真实的统计量值 $\theta$. 那么很自然地, 我们可以利用 Bootstrap 采样得到的统计量分布来近似这个区间. 具体来说, 我们可以取 Bootstrap 统计量的 $\alpha/2$ 分位数和 $1 - \alpha/2$ 分位数作为置信区间的端点:
$$ \left[\text{Quantile}_{\alpha/2}(\{\hat\theta^{* (b)}\}_{b=1}^B),\quad \text{Quantile}_{1 - \alpha/2}(\{\hat\theta^{* (b)}\}_{b=1}^B)\right].$$


### Pivotal Method

Pivotal Method 则是构造一个枢轴量 (pivotal quantity), 其分布不依赖于未知参数, 然后利用 Bootstrap 来估计该枢轴量的分布以构造置信区间. 

具体来说, CI 的定义可以重写为
$$\begin{aligned}
&\mathbb{P}_{\mathcal{F}}(\hat\theta - q_L \leq \theta \leq \hat\theta + q_R)  = 1 - \alpha, \\
\Leftrightarrow ~&\mathbb{P}_{\mathcal{F}}(-q_L \leq \theta - \hat\theta \leq q_R) = 1 - \alpha, \\
\Leftrightarrow ~&\mathbb{P}_{\mathcal{F}}(q_R \leq \hat\theta - \theta \leq -q_L) = 1 - \alpha.
\end{aligned}$$

因此, 若我们能够估计出枢轴量 $R:=\hat\theta - \theta$ 的分布, 则可以通过其分位数 $q_L, q_R$ 来构造置信区间:
$$\left[\hat\theta - q_R,\quad \hat\theta - q_L\right].$$

记 $R$ 的 CDF 为 $H$, 即 $H(r) = \mathbb{P}(R \leq r)$, 则根据分位数的定义, 有
$$q_R = -H^{-1}(\alpha/2),\quad q_L = -H^{-1}(1 - \alpha/2).$$

然而由于 $\mathcal{F}$ 未知, 我们无法直接计算 $H$ 和 $H^{-1}$. 这时可以利用 Bootstrap 来近似 $H$:
1. 通过 Bootstrap 重采样得到 $X_1^{* (b)}, X_2^{* (b)}, \ldots, X_n^{* (b)} \stackrel{\text{i.i.d.}}{\sim} \hat{\mathcal{F}}_n$, $b = 1, 2, \ldots, B$.
2. 计算 Bootstrap 统计量 $\hat\theta^{* (b)} = g(X_1^{* (b)}, X_2^{* (b)}, \ldots, X_n^{* (b)})$.
3. 计算 Bootstrap 枢轴量 $R^{* (b)} = \hat\theta^{* (b)} - \hat\theta$ (其中 $\hat\theta = g(X_1, X_2, \ldots, X_n)$ 是原始样本下的统计量).
4. 利用 Bootstrap 枢轴量的经验分布 $\hat{H}(r) = \frac{1}{B} \sum_{b=1}^B \mathbf{1}\{R^{* (b)} \leq r\}$ 来近似 $H$.
5. 计算 $\hat{H}$ 的分位数 $\hat q_R = -\hat{H}^{-1}(\alpha/2)$ 和 $\hat q_L = -\hat{H}^{-1}(1 - \alpha/2)$.
6. 构造置信区间:
   $$\left[\hat\theta - \hat q_R,\quad \hat\theta - \hat q_L\right].$$
   或者等价地,
   $$\left[
2\hat{\theta}-\theta_{1-\alpha/2}^*,\
2\hat{\theta}-\theta_{\alpha/2}^*
\right].$$

## Jackknife vs. Bootstrap

Bootstrap 是进行有放回的重采样, 而 Jackknife 则更省力: 它不进行重采样, 而是通过系统地留出一个样本点来构造子样本. 具体来说, 设统计量为 $T_n = g(X_1, X_2, \ldots, X_n)$, 则对于每个 $i = 1, 2, \ldots, n$, 定义留一子样本统计量
$$T_{n}^{(-i)} = g(X_1, \ldots, X_{i-1}, X_{i+1}, \ldots, X_n).$$

然后将这些留一子样本统计量的均值作为 Jackknife 均值:
$$\bar{T}_n^{(-)} = \frac{1}{n} \sum_{i=1}^n T_n^{(-i)}.$$

- $\{T_{(-i)}\}$  描述了“每删掉一个点, 统计量会变化多少”. 
- 如果统计量对单个观测很敏感, 这些值会波动大; 不敏感则波动小. 所以它可以用来估计 $T_n$ 的方差/标准误.

最终得到的 Jackknife 方差估计为:
$$\widehat{\mathbb{V}}_{\text{Jack}}[T_n] = \frac{n-1}{n} \sum_{i=1}^n \left(T_n^{(-i)} - \bar{T}_n^{(-)}\right)^2.$$