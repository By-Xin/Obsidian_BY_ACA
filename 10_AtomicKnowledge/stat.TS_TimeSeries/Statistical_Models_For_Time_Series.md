# Statistical Models for Time Series

## 1. Introduction

我们希望用统计学的视角来分析时间序列数据. 假定有一组时序数据 $x_1, x_2, \ldots, x_n$, 我们认为这些数据是一系列随机变量 $X_i$ 的观测值, 这些数据背后服从一个联合概率分布 $\mathbb{P}\left(X_1\leq x_1, X_2\leq x_2, \ldots, X_n\leq x_n\right)$, 或称这一系列数据是由一个 stochastic process (随机过程) 生成的. 

随机过程可以粗略地理解为一组随机变量的集合或一个随机变量的序列 (如 $X_t, t=1,2,\ldots$). 对于每一个 $t$ 对应的 $X_t$ 都可以认为是一个随机变量, 而这一系列的随机变量本身也具有一定的相关性. 因此有别于单个随机变量, 随机过程更关注这一系列随机变量的相关关系, 以更整体的视角来分析这一系列随机变量. 

对于一个随机过程 $X_t$, 我们引入 **mean function** 和 **covariance function** 来描述其性质. 之所以称之为函数, 是因不同于随机变量, 一个随机过程对于每个时间点 $t$ 都有一个对应的均值或方差, 故以此强调其与$t$的关系.
- **Mean function**: $\mu(t) \triangleq \mathbb{E}\left[X_t\right]$
- **Auto-covariance function (ACVF)**: $\gamma(s, t) \triangleq \text{Cov}\left(X_s, X_t\right)$

## 2. Stationarity of Stochastic Processes

平稳性 (Stationarity) 是时间序列分析中一个重要的概念. 一个平稳的随机过程具有许多优良的性质, 便于我们进行分析. 对于非平稳的时间序列, 我们往往也是希望将其转化为平稳的时间序列, 再利用这些平稳性质进行分析.

具体而言, stationarity 可以分为两种, 即 **strict stationarity** (严平稳) 和 **weak stationarity** (宽平稳). 

**Strict stationarity**
- 定义: 给定的任意多个有限维时间点 $t_1, t_2, \ldots, t_n$, 以及一个任意大小的 shift (时间偏移) $h$, 要求一个严平稳的随机过程 $X_t$ 满足: $\left(X_{t_1}, X_{t_2}, \ldots, X_{t_n}\right)$ 和 $\left(X_{t_1+h}, X_{t_2+h}, \ldots, X_{t_n+h}\right)$ 的联合分布相同, 即:
$$\begin{aligned}
\mathbb{P}&\left(X_{t_1}\leq x_1, X_{t_2}\leq x_2, \ldots, X_{t_n}\leq x_n\right) \\ = \mathbb{P}&\left(X_{t_1+h}\leq x_1, X_{t_2+h}\leq x_2, \ldots, X_{t_n+h}\leq x_n\right)
\end{aligned}$$
- 性质:
  - $X_t$ 具有相同的边际分布
  - 均值函数和方差函数恒为常数, 不随时间变化
  - 自协方差函数 (ACVF) 有: $\gamma(s, t) = \gamma(s+h, t+h)$, 即只与时间间隔 $|s-t|$ 有关, 而与具体的时间点无关 (自相关函数同理)

**Weak stationarity**
- 定义: 一个宽平稳的随机过程 $X_t$ 满足:
  1. $X_t$ 的均值函数 $\mu(t) = \mathbb{E}\left[X_t\right]$ 为恒为常数与时间 $t$ 无关
  2. $X_t$ 的自协方差函数 $\gamma(t,t-k) = \gamma(0,k)$ 对于任意时间点 $t$ 和 shift $k$ 都成立, 即 ACVF 只与时间间隔 $k$ 有关, 而与具体的时间点无关. 
- 性质:
  - 若一个序列是严平稳的且存在有限的二阶矩, 则一定能推出其是宽平稳的; 但反之不必然. 
  - 如果一个时间序列的联合分布是多元正态的, 则宽平稳和严平稳是等价的, 相互可推出. 
- 严平稳的要求往往过于严格, 因此如无特殊说明, 我们往往默认所讨论的平稳性是指宽平稳性 (weak stationarity), 即前两阶矩的平稳性.

对于一个平稳的时间序列, 由于期望函数是常数, 而 ACVF 只与时间间隔有关而与具体时间点无关, 因此我们可以对平稳时间序列简写其均值函数和自协方差函数为:
- $\mu = \mathbb{E}\left[X_t\right]$
- ACVF: $\gamma_k = \text{Cov}\left(X_t, X_{t-k}\right)$
- ACF: $\rho_k = \text{Corr}\left(X_t, X_{t-k}\right) = \gamma_k / \sqrt{\gamma_0 \gamma_k} = \gamma_k / \gamma_0$

## 3. Examples of Stationary Time Series

下面列举一些常见的平稳时间序列模型. 

### White Noise 白噪声

白噪声是一种最简单的时间序列模型, 类比宽/严平稳的定义, 白噪声也可以分为 **strict white noise** 和 **weak white noise**. 对于一个白噪声序列 $X_t \sim \text{WN}(0, \sigma^2)$, 定义:
- **Weak white noise**: 只要求 $X_t$ 是弱平稳的, 即均值为0, 方差为常数$\sigma^2$, 两两之间的协方差为0.
- **Strict white noise**: 在 weak white noise 的基础上要求这个序列是完全独立同分布的. 

白噪声不对其具体分布作要求, 只要求其均值为0, 方差为常数, 且两两之间的协方差为0. 不过在实践中有时也会进一步要求其服从正态分布, 以便于进行更多的统计推断. 这时的白噪声又称为 **Gaussian white noise**. 

若不加特殊说明, 我们在时间序列分析中所指的白噪声往往是指强平稳的白噪声, 即独立同分布但不加正态分布的要求.

白噪声是时间序列的一个重要基石, 许多时间序列模型都是基于白噪声的变换或组合而来.

### Random Walk 随机游走

#### Simple Random Walk

首先定义最简单的均值为零(无 drift 等)的随机游走模型 (或称simple random walk):

令 $W_t \sim \text{WN}(0, \sigma^2)$ 为独立同分布的白噪声序列, 定义随机游走序列 $X_t$ 为:
$$X_t = \sum_{j=1}^t W_j, ~ t\in\mathbb{N} \quad (1)$$
或等价地写为:
$$X_t = X_{t-1} + W_t, ~ t\in\mathbb{N} \quad (2)$$
其中初始值 $X_0 = 0$.

各阶矩性质如下:
  - 均值函数 $\mu(t) = 0$:
    $$\mu_t = \mathbb{E}\left[X_t\right] = \mathbb{E}\left[\sum_{j=1}^t W_j\right] = \sum_{j=1}^t \mathbb{E}\left[W_j\right] = 0$$
  - 方差函数 $\gamma(t,t) = t \sigma^2$:
    $$\gamma_t = \text{Var}\left[X_t\right] = \text{Var}\left[\sum_{j=1}^t W_j\right] = \sum_{j=1}^t \text{Var}\left[W_j\right] = t \sigma^2$$
    注意到, 随机游走的方差随着时间的推进而线性增长.  
  - 自协方差 ACVF $\gamma(t,s) = \min(t,s) \sigma^2$:  
    $$\begin{aligned}
    \small{\text{不妨令 }} t &\leq s: \\
    \gamma(t,s) &= \text{Cov}\left[X_t, X_s\right] \\
    &= \text{Cov}\left[\sum_{j=1}^t W_j, \sum_{j=1}^s W_j\right] \\
    &= \text{Cov}\left[\sum_{j=1}^t W_j, \sum_{j=1}^t W_j + \sum_{j=t+1}^s W_j\right] \\
    &= \text{Cov}\left[\sum_{j=1}^t W_j, \sum_{j=1}^t W_j\right] + \text{Cov}\left[\sum_{j=1}^t W_j, \sum_{j=t+1}^s W_j\right] \\
    &= \text{Var}\left[\sum_{j=1}^t W_j\right] + 0 \\
    &= t \sigma^2
    \end{aligned}$$
  - 自相关系数 ACF $\rho(t,s) = \sqrt{\min(t,s)/\max(t,s)}$:  
    $$\begin{aligned}
    \small{\text{不妨令 }} t &\leq s: \\
    \rho(t,s) &= \text{Corr}\left[X_t, X_s\right] \\
    &= \frac{\gamma(t,s)}{\sqrt{\gamma(t,t) \gamma(s,s)}} \\
    &= \frac{t \sigma^2}{\sqrt{t \sigma^2 \cdot s \sigma^2}} \\
    &= \sqrt{t/s}
    \end{aligned}$$
    - 如果 $t,s$ 只差一期 (即 $s=t+1$), 且当 $t$ 较大时, 则 $\rho(t,t+1) = \sqrt{t/(t+1)} \approx 1$. 即对于随机游走序列, 当时间较久远时, 两个相邻时间点的相关性会非常高.

注意到, 即使对于 simple random walk, 由于其方差 $t\sigma^2$ 随时间线性增长, 且ACVF $\gamma(t,s) = \min(t,s) \sigma^2$ 也表明了序列见的相关性, 因此**random walk 并不是一个平稳时间序列**.

#### Random Walk with Drift

在随机游走模型的基础上, 我们可以引入一个 drift 参数 $\theta$ 来描述随机游走序列的均值漂移. 这时上面的随机游走模型 (2) 变为:
$$X_t = \theta + X_{t-1} + W_t, ~ t\in\mathbb{N}$$
也就是在每个时间点上, 随机游走序列的均值都会增加一个固定的值 $\theta$.

对应的 (1) 式也可以写为:
$$X_t = \theta t + \sum_{j=1}^t W_j, ~ t\in\mathbb{N}$$

下图是一个简单随机游走和带漂移的随机游走的轨迹图:

![Trajectories for a simple random walk (black) and a random walk with drift = 0.2. The dotted lines show have slope equal to 0 and 0.2 respectively and reflect the mean function for both processes.](https://yanshuo.quarto.pub/nus-ts-book/10-fundamentals_files/figure-html/fig-stationary-rw-1.png)

### Signal + Noise Model

信号加噪声模型是时间序列分析中常见的模型之一. 该模型会将一个时间序列分解为两部分: 一个是具有一定规律性的信号部分, 另一个是随机的噪声部分:
$$
X_t = f_t + W_t
$$
其中 $W_t \sim \text{WN}(0, \sigma^2)$ 为白噪声, $f_t$ 为信号部分为一个确定的函数.

该模型往往用于描述某个时间序列中的趋势或周期性变化, 以及其中的随机波动部分. 如果 $f_t$ 的具体形式是已知的, 则还可以通过回归分析等方法来估计其参数. 例如若 $f_t = \beta_0 + \beta_1 t$ 则该模型就等价于之前讨论的 linear trend method.

### General Linear Process

同样暂时先不考虑 drift 项. 

给定 $t\in \mathbb{Z}$, 考虑一组白噪声 $W_t \sim \text{WN}(0, \sigma^2)$, 以及对应一组参数 $\psi_j$:
| $j$ | $\cdots$ | $-2$ | $-1$ | $0$ | $1$ | $2$ | $\cdots$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\psi_j$ | $\cdots$ | $\psi_{-2}$ | $\psi_{-1}$ | $\psi_0$ | $\psi_1$ | $\psi_2$ | $\cdots$ |
| $W_{t-j}$ | $\cdots$ | $W_{t+2}$ | $W_{t+1}$ | $W_t$ | $W_{t-1}$ | $W_{t-2}$ | $\cdots$ |

对应相乘并相加组成的线性组合 (其实进行的是一个卷积 convolution 操作):
$$
X_t = \sum_{j=-\infty}^{\infty} \psi_j W_{t-j}
$$

对于这样定义的一组线性组合 $X_t$, 进一步假设其系数具有性质 $\sum_{j=-\infty}^{\infty} |\psi_j| < \infty$, 则该线性组合 $X_t$ 的均值函数和自协方差函数分别为:
- 均值函数: $\mu_t = \mathbb{E}\left[X_t\right] = \sum_{j=-\infty}^{\infty} \psi_j \mathbb{E}\left[W_{t-j}\right] = 0$
- 自协方差函数: 
    $$\begin{aligned}
    \gamma(h) &= \text{Cov}\left[X_{t+h}, X_t\right] \\
    &= \text{Cov}\left[\sum_{i=-\infty}^{\infty} \psi_i W_{t+h-i}, \sum_{j=-\infty}^{\infty} \psi_j W_{t-j}\right] \\
    &= \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} \psi_i \psi_j \text{Cov}\left[W_{t+h-i}, W_{t-j}\right] \\
    &= \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} \psi_i \psi_j \gamma(j-i+h) \\
    &= \sum_{i=-\infty}^{\infty} \left( \psi_i \psi_{i-h} \gamma(0)+  \sum_{j\neq i-h} \psi_i \psi_j \gamma(j-i+h)\right)\\
    &= \sum_{i=-\infty}^\infty \psi_i\psi_{i-h}  \sigma^2~ \small{(\text{as iif } j = i-h, \gamma(j-i+h) = \sigma^2 \neq 0)} 
    \end{aligned}$$

注意这里的这个 $\gamma(h) = \sigma^2 \sum_{i=-\infty}^\infty \psi_i\psi_{i-h}$ 与时间间隔 $h$ 有关, 但与具体的时间点无关. 其中的下标 $i$ 是线性组合的, 不是时间点. 因此我们定义的这个线性组合 $X_t$ 是一个平稳时间序列.

若进一步, 我们要求这个序列对于所有 $j<0$, 均有 $\psi_j = 0$, 即
$$
X_t = \sum_{j=0}^{\infty} \psi_j W_{t-j} = \psi_0 W_t + \psi_1 W_{t-1} + \psi_2 W_{t-2} + \cdots
$$
则称这个序列是 **causal** (因果的). 这意味着该序列仅仅依赖于当前时刻及之前的信息 ($W_t, W_{t-1}, \cdots$), 而不依赖于未来的信息 ($W_{t+1}, W_{t+2}, \cdots$).

总的而言, 我们定义这样一个**linear (线性的), stationary (平稳的), causal (因果的) process** 为 **general linear process (GLP)**. 其一般形式为:
$$
X_t = \sum_{j=0}^{\infty} \psi_j W_{t-j}=\psi_0 W_t + \psi_1 W_{t-1} + \psi_2 W_{t-2} + \cdots
$$
其中 $\{W_t\}\sim \text{WN}(0, \sigma^2)$ 为白噪声序列, $\{\psi_j\}$ 为一组参数, 满足 $\sum_{j=0}^{\infty} |\psi_j| < \infty$. 另外常假设 $\psi_0 = 1$. 

整理一下, GLP 的性质如下:
- 均值函数: $\mu_t = \mathbb{E}\left[X_t\right] = 0$
- 方差函数: $\gamma_0 = \text{Var}\left[X_t\right] = \sigma^2 \sum_{j=0}^{\infty} \psi_j^2 < \infty$
- 自协方差函数 ACVF: $\gamma(h) = \sigma^2 \sum_{j=0}^{\infty} \psi_j \psi_{j+h}, ~h \ge 0$
- 自相关系数 ACF: $\rho(h) = \gamma(h) / \gamma(0) = \sum_{j=0}^{\infty} \psi_j \psi_{j+h} / \sum_{j=0}^{\infty} \psi_j^2$

进一步放宽, 当 GLP 的均值非零时, 我们可以引入一个 drift 参数 $\mu$, 使得 $X_t = \mu + \sum_{j=0}^{\infty} \psi_j W_{t-j}$. 其余的讨论是类似的.

## 4. Forecasting with Simple Models

我们在之前已经给出过时间序列预测的一些方法, 如 Mean Method, Naive Method, Seasonal Naive Method 等. 这里将针对这些预测方法给出统计学的解释. 

回顾, 统计模型会针对时序数据 $X_1, X_2, \ldots, X_n$ 建立一个概率模型 $p:=\mathbb{P}\left(X_1\leq x_1, X_2\leq x_2, \ldots, X_n\leq x_n\right)$, 以此来描述这一系列数据的生成过程. 

 $h$ 步预测就是指在已知 $X_{1:n} \triangleq \{X_1, X_2, \ldots, X_n\}$ 的情况下, 预测 $X_{n+h} | X_{1:n}$ 的分布 $p(X_{n+h} | X_{1:n})$. 一个常见的点估计是通过条件期望来进行预测, 即:
$$
\hat{X}_{n+h | n} = \mathbb{E}\left[X_{n+h} | X_n = x_n, \cdots, X_1 = x_1\right]
$$
而条件期望 $\mathbb{E}\left[X_{n+h} | X_{1:n}\right]$ 又可以认为是 $X_{1:n}$ 的一个函数, 即 $\hat{X}_{n+h | n} = \mathbb{E}\left[X_{n+h} | X_{1:n}\right] \triangleq \varphi(X_{1:n})$. 可以证明, 用条件期望进行预测是最小化均方误差MSE的估计, 同时也是在平方损失下的 Bayes optimal estimator (Bayes 最优估计). 也就是说, 在所有 $f: \mathbb{R}^n \to \mathbb{R} ( X_{1:n} \overset{f}{\mapsto} X_{n+h})$ 中, $\varphi$ 是使得 $\mathbb{E}\left[(X_{n+h} - \varphi(X_{1:n}))^2\right]$ 最小的函数, 即:
$$
\varphi = \arg\min_{f : \mathbb{R}^n \to \mathbb{R}} \mathbb{E}\left[(X_{n+h} - f(X_{1:n}))^2\right]
$$

### Mean Method

Mean Method 是一种最简单的预测方法, 即预测未来的值为历史数据的均值:
$$
\hat{X}_{n+h | n} = \bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i
$$

该方法是假设序列服从下面的模型假设:
$$
X_t = \theta + W_t
$$
其中 $\theta$ 是一个未知的常数, $W_t$ 是一个 Gaussian white noise, 即 $W_t \overset{iid}{\sim} \text{W}\mathcal{N}(0, \sigma^2)$.

我们的预测目标是 $p(X_{n+h} | X_{1:n}) = p(\theta + W_{n+h}~ |~ \theta + W_1, \theta + W_2, \ldots, \theta + W_n)$. 由于 $W_t$ 是独立同分布的, 因此 $W_{n+h}$ 与 $W_1, W_2, \ldots, W_n$ 是独立的, 因此 $X_{n+h}$ 与 $X_1, X_2, \ldots, X_n$ 也是独立的. 故
$$
p(X_{n+h} | X_{1:n}) = p(X_{n+h}) = p(\theta + W_{n+h}) 
$$
即 $X_{n+h} \sim \mathcal{N}(\theta, \sigma^2)$.  因此, 该模型的预测是 $\hat{X}_{n+h | n} = \theta$. 而对于正态分布, 很自然有其均值为极大似然估计:
$$
\hat X_{n+h | n} = \hat\theta_{ML} = \frac{1}{n} \sum_{i=1}^n X_i = \bar{X}_n
$$

### Naive Method

Naive Method 是一种更简单的预测方法, 即预测未来的值为历史数据的最后一个值:
$$
\hat{X}_{n+h | n} = X_n
$$

该方法是假设序列服从无漂移的随机游走模型:
$$
X_t = X_{t-1} + W_t = X_1 + \sum_{i=1}^t W_i
$$
同样假设 $W_t$ 是独立同分布的 Gaussian white noise. 

因此可认为:
$$
X_{n+h} = X_n + \sum_{j=n+1}^{n+h} W_j \sim \mathcal{N}(x_n, h\sigma^2)
$$

### Seasonal Naive Method

假设一个季节周期内有 $p$ 个时间点, 则 Seasonal Naive Method 是一种预测方法, 即预测未来的值为历史数据的最后一个季节内对应时点的值:
$$
\hat{X}_{n+h | n} = X_{n- ( (p-h) \mod p)}
$$
例如对于月度数据 ($p=12$), 若当前时点为 2025 年 3 月, 则预测 2025 年 4 月的值为 2024 年 4 月的值.

Naive Method 背后的模型假设都是随机游走模型, 即使考虑 seasonality 也是如此. 只不过这里认为有 $p$ 个独立的子随机游走序列 $Y_t^{(1)}, Y_t^{(2)}, \ldots, Y_t^{(p)}$. 例如同样对于月度数据, 时间跨度为 2011年1月至2024年12月 ($X_{2011/01},\cdots,X_{2024/12}$), 则可以认为有 12 个独立的子序列 ($\{X_{2011/01}, X_{2012/01}, \ldots, X_{2024/01}\}$, $\{X_{2011/02}, X_{2012/02}, \ldots, X_{2024/02}\},$ $\ldots,$ $\{X_{2011/12}, X_{2012/12}, \ldots, X_{2024/12}\}$), 每个子序列都是一个随机游走序列, 序列和序列之间是独立的. 余下的分析与 Naive Method 类似, 每个子序列的预测都是其最后一个值.

### Linear Trend Method

Linear Trend Method 背后的模型假设是 Signal + Noise 模型, 即:
$$
X_t = \beta_0 + \beta_1 t + W_t
$$
而其余的推导与回归分析中的线性回归模型完全一致. 

### Drift Method

Drift Method 是只考虑第一个和最后一个时间点的预测方法, 即:
$$
\hat{X}_{n+h | n} = X_n + h \frac{X_n - X_1}{n-1}
$$

其背后的模型是含有 drift 的随机游走模型:
$$
X_{n+h} = X_n + h\theta + \sum_{j=n+1}^{n+h} W_j
$$
因此其条件期望为:
$$
\hat{X}_{n+h | n} = \mathbb{E}\left[X_{n+h} | X_n, X_1\right] = X_n + h\theta 
$$

因此想要预测 $\hat{X}_{n+h | n}$, 就需要估计 $\theta$. 我们同样尝试通过极大似然估计来估计 $\theta$. 由于 $X_t = X_{t-1} + \theta + W_t$, 且 $W_t \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)$, 因此 $X_t | X_{t-1} \sim \mathcal{N}(X_{t-1} + \theta, \sigma^2)$. 故
$$
p_{\theta}(X_t | X_{t-1}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(X_t - X_{t-1} - \theta)^2}{2\sigma^2}\right)
$$
考虑所有时间步的数据, 得到联合概率(似然函数):
$$\begin{aligned}
\mathcal{L}(\theta) &= p(X_2, X_3, \ldots, X_n | X_1) = \prod_{t=2}^n p_{\theta}(X_t | X_{t-1}) \\
&\propto \exp\left(-\frac{1}{2\sigma^2} \sum_{t=2}^n (X_t - X_{t-1} - \theta)^2\right)
\end{aligned}$$
最终可以求得这个最大似然函数的估计值
$$
\hat\theta_{ML} = \frac{1}{n-1} \sum_{t=2}^n (X_t - X_{t-1}) = \frac{X_n - X_1}{n-1}
$$
因此 Drift Method 的预测值为:
$$
\hat{X}_{n+h | n} = X_n + h \frac{X_n - X_1}{n-1}
$$

## 5. Gaussian Processes

### Definition

Gaussian Process 是一种随机过程, 其中任意有限个时间点的联合分布都是多元正态分布. 也就是说, 对于任意多个时间点 $t_1, t_2, \ldots, t_k (k\in\mathbb{N})$, 其联合分布 $p(X_{t_1}, X_{t_2}, \ldots, X_{t_k})$ 都是多元正态分布. 若记 $\mathbf{X} = (X_{t_1}, X_{t_2}, \ldots, X_{t_k})$, 则有:
$$
p(\mathbf{X}) = (2\pi)^{-k/2} \det(\Sigma)^{-1/2} \exp\left(-\frac{1}{2} (\mathbf{X} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{X} - \boldsymbol{\mu})\right)
$$
其中 $\boldsymbol{\mu} = (\mu_{t_1}, \mu_{t_2}, \ldots, \mu_{t_k})^\top \in \mathbb{R}^k$ 为均值向量, $\Sigma = (\gamma(t_i, t_j))_{i,j} \in \mathbb{R}^{k\times k}$ 为ACVF矩阵.

顺承上面平稳的定义, 若一个 Gaussian Process $(X_t)$ 是一个 weak stationary 的, 那么 $\mathbf{\mu}$ 是一个常数向量, $\Sigma$ 的第 $i,j$ 个元素的取值只与时间间隔 $|t_i - t_j|$ 有关, 而与具体的时间点无关. 特别的, 在 Gaussian 假设下, weak stationarity 和 strict stationarity 是等价的.

### Gaussian Conditional Distribution

***Proposition (Conditional Distribution)***: 若 $(X_1, \cdots, X_{n +1} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma))$ 为一个多元正态分布, 则其条件分布 $X_{n+1} | X_1, \cdots, X_n$ 也是一个正态分布 $\mathcal{N}(\mu^*, \sigma^{*2})$, 其中:
$$
\begin{aligned}
\mu^* &= \mu_{n+1} + \Sigma_{1:n, n+1}^\top \Sigma_{1:n, 1:n}^{-1} (\mathbf{X_{1:n}} - \boldsymbol{\mu_{1:n}}) \\
\sigma^{*2} &= \Sigma_{n+1, n+1} - \Sigma_{1:n, n+1}^\top \Sigma_{1:n, 1:n}^{-1} \Sigma_{1:n, n+1}
\end{aligned}
$$
其中协方差矩阵的分块表示如下图所示:

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250304200917.png)

***Corollary (Conditional Distribution for Stationary Gaussian Process)***: 若 $(X_t)$ 是一个 weak stationary 的 Gaussian Process, 均值为0. 给定前 $n$ 个时间点的值 $X_1, X_2, \ldots, X_n$, 则 $X_{n+h} | X_1, X_2, \ldots, X_n$ 也是一个正态分布 $\mathcal{N}(\mu^*, \sigma^{*2})$, 其中:
$$
\begin{aligned}
\mu^* &=\boldsymbol \gamma_{h~:~n+h-1}^\top \Gamma_{1:n}^{-1} \mathbf{X_{1:n}} \\
\sigma^{*2} &= \gamma(0) - \boldsymbol\gamma_{h~:~n+h-1}^\top \Gamma_{1:n}^{-1} \boldsymbol\gamma_{h~:~n+h-1}
\end{aligned}
$$
其中 
- $\boldsymbol\gamma = \begin{pmatrix} \gamma(0) \\ \gamma(1) \\ \vdots \\ \gamma(n-1) \end{pmatrix} \in \mathbb{R}^n$ 为 ACVF 向量.
- $\boldsymbol\gamma_{h~:~n+h-1} = \begin{pmatrix} \gamma(h) \\ \gamma(h+1) \\ \vdots \\ \gamma(n+h-1) \end{pmatrix} \in \mathbb{R}^{n-h+1}$ 为 $\boldsymbol\gamma$ 的子序列. 
- $\Gamma \in \mathbb{R}^{n\times n}$ 为 ACVF 矩阵, $\Gamma = \begin{pmatrix} \gamma(0) & \gamma(1) & \cdots & \gamma(n-1) \\ \gamma(1) & \gamma(0) & \cdots & \gamma(n-2) \\ \vdots & \vdots & \ddots & \vdots \\ \gamma(n-1) & \gamma(n-2) & \cdots & \gamma(0) \end{pmatrix}$.
- $\Gamma_{1:n} = \begin{pmatrix} \Gamma_{1,1} & \Gamma_{1,2} & \cdots & \Gamma_{1,n} \\ \Gamma_{2,1} & \Gamma_{2,2} & \cdots & \Gamma_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ \Gamma_{n,1} & \Gamma_{n,2} & \cdots & \Gamma_{n,n} \end{pmatrix} \in \mathbb{R}^{n\times n}$ 为 $\Gamma$ 的子矩阵.

可以直接推导置信区间的计算公式:
$$
\begin{aligned}
\text{CI}_{1-\alpha} &= \left(\hat{X}_{n+h | n} - z_{\alpha/2} \sqrt{\sigma^{*2}}, \hat{X}_{n+h | n} + z_{\alpha/2} \sqrt{\sigma^{*2}}\right) \\
\end{aligned}
$$

## 6. Box-Jenkins Methodology 

Box-Jenkins Methodology 是一种时间序列分析的方法, 由 George Box 和 Gwilym Jenkins 于 1970 年提出. 该方法主要包括三个步骤: 模型识别 (Model Identification), 参数估计 (Parameter Estimation), 模型检验 (Model Diagnostic Checking). 如下图所示. 

![图片源自 Box & Jenkins (1976)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250306140845.png)


对于这个建模步骤, 有时在最开始会再加上数据预处理 (Data Preprocessing). 通过 differencing, Box-Cox 变换 或时间序列分解等方法, 将原始数据 $\{X_t\}$ 转换为平稳的时间序列 $\{Y_t\}$, 以便于后续的建模. 在得到可用的模型后, 往往也会先对 $\{Y_t\}$ 进行预测, 然后再通过逆变换得到 $\{X_t\}$ 的预测.

