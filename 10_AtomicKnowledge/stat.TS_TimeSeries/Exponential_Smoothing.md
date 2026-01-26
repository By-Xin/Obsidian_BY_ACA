---
aliases: ['指数平滑', 'Exponential Smoothing']
tags:
  - concept
  - method
  - math/statistics
  - time-series
related_concepts:
  - [[Autoregressive_Model]]
  - [[Forecasting]]
  - [[Smoothing]]
---

# Exponential Smoothing

> Refs: [1] https://yanshuo.quarto.pub/nus-ts-book/07-exponential_smoothing.html; [2] https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/HoltWinters; [3]  https://otexts.com/fpp3/holt-winters.html; [4] https://www.math.pku.edu.cn/teachers/lidf/course/fts/ftsnotes/html/_ftsnotes/index.html 

## Simple Exponential Smoothing (SES)

### Model Introduction

Exponential smoothing (指数平滑) 是一种时间序列预测方法, 其本质是对 ***全部*** 历史数据进行加权平均, 而加权的权重是随时间指数递减的, 即越近期的数据权重越大, 越久远的数据权重越小.

指数平滑预测介于 mean method 和 naive method 之间, 尽量平衡了方差和偏差:
- Mean method: 预测值等于全部历史数据的平均值. 该方法方差 (variance) 较小而偏差 (bias) 较大.
  - 方差小: 对全部历史数据取平均, 等价于把很多数据“平滑”到一个点上. 综合了大量观测值，单个新数据的随机波动对最终预测的影响相对较小，因而在不同样本下的变动幅度也更小，表现为方差较低
  - 偏差大: 如果真实的时间序列具有趋势, 季节性或近期发生了结构性变化, 单纯使用全部数据的平均值可能会“拖后腿”, 因为早期数据会稀释近期的信息, 使得预测无法及时跟上当前水平, 形成系统性的误差（偏差）
- Naive method: 预测值等于最近一个观测值. 该方法方差较大而偏差较小.
  - 偏差小: 由于只使用最后一次观测值作为下一时刻的预测, 如果真实过程当前水平在短期内占主导地位, 那么用最新值作为预测就能够很好地捕捉到当下的状态, 从而偏差较小.
  - 方差大: 完全依赖最后一个观测值也使预测非常“敏感”, 任何一次性的噪声, 极端值或偶然波动都会直接反映到预测上. 由于其不做平滑, 缺少对历史信息的“缓冲”, 导致在不同样本或不同时间点上的预测会有较大的波动幅度, 即方差较高.

### Model Derivation

下推导其数学表达式. 假设当前时刻的时间序列为 $x_n$, 且我们先假定拥有直到无穷远的过去的全部历史数据, 即 $x_n, x_{n-1}, x_{n-2}, \cdots$. 我们希望进行一步向前预测下一个时刻 $\hat{x}_{n+1}$ 的值. 引入一个平滑系数 $\alpha \in (0, 1)$, 用于控制历史数据的权重. 指数平滑就是一种加权平均的预测:
$$
\hat{x}_{n+1} := \alpha x_n + \alpha^2 x_{n-1} + \alpha^3 x_{n-2} + \cdots = \sum_{j=1}^{\infty} \alpha^j x_{n+1-j}
$$
不过目前这个形式还有一点问题. 因为我们是希望进行一个**加权求和**, 因此希望这些权重的和为1 (尽管是一个无穷多项的级数求和). 而我们当前的系数和可以算得:
$$
\sum_{j=1}^{\infty} \alpha^j = \frac{\alpha}{1-\alpha}
$$
因此我们需要对上式进行归一化 (即每个权重都除以 $\frac{\alpha}{1-\alpha}$), 使得权重和为1, 即 $\sum_{j=1}^{\infty} \frac{\alpha^j}{\alpha / (1-\alpha)} = 1$. 于是我们可以得到最终的指数平滑预测公式:
$$\begin{aligned}
\hat{x}_{n+1} &:= \frac{\alpha x_n }{\alpha/(1-\alpha)}  + \frac{\alpha^2 x_{n-1}}{\alpha/(1-\alpha)} + \frac{\alpha^3 x_{n-2}}{\alpha/(1-\alpha)} + \cdots \\
&= (1-\alpha) x_n + \alpha (1-\alpha) x_{n-1} + \alpha^2 (1-\alpha) x_{n-2} + \cdots \\
&= (1-\alpha) (x_n + \alpha x_{n-1} + \alpha^2 x_{n-2} + \cdots) \\
&= (1-\alpha) \sum_{j=0}^{\infty} \alpha^j x_{n-j}
\end{aligned}$$

---

另一个较为常见的版本的(教材中给出的)指数平滑公式是:
$$\begin{aligned}
\hat{x}_{n+1} &:= \alpha \sum_{j=0}^{n-1} (1-\alpha)^j x_{n-j} + (1-\alpha)^n l_0\\
&= \alpha\left(x_n + (1-\alpha)x_{n-1} + (1-\alpha)^2 x_{n-2} + \cdots+ (1-\alpha)^{n-1} x_1\right) + (1-\alpha)^n l_0
\end{aligned}$$

这是考虑到在实际应用中, 我们并不可能拥有无穷远的历史数据, 现实中我们通常只能拥有$x_n, x_{n-1}, \cdots, x_1$这些有限的历史数据. 因此对于 $x_{\leq 0}$ 的数据, 我们统一用一个初始值 $l_0$ 来代替: $x_m \equiv l_0, \forall m \leq 0$. 这样我们就可以从无限求和的形式转化为有限求和的形式:
$$\begin{aligned}
\hat{x}_{n+1} &= \alpha \sum_{j=0}^{\infty} (1-\alpha)^j x_{n-j}\\
&= \alpha \underbrace{\sum_{j=0}^{n-1} (1-\alpha)^j x_{n-j}}_{\small\text{已观测历史数据}} + \alpha \underbrace{\sum_{j=n}^{\infty} (1-\alpha)^j x_{n-j}}_{\small\text{未观测“史前”数据}}\\
&:= \alpha \sum_{j=0}^{n-1} (1-\alpha)^j x_{n-j} + \alpha \sum_{j=n}^{\infty} (1-\alpha)^j l_0\\
&= \alpha \sum_{j=0}^{n-1} (1-\alpha)^j x_{n-j} + (1-\alpha)^n l_0 ~~~\small\text{(由等比数列级数求和公式)}
\end{aligned}$$

---

由此, 在一个指数平滑中:
$$\boxed{\hat{x}_{n+1} = \alpha\sum_{j=0}^{n-1} (1-\alpha)^j x_{n-j} + (1-\alpha)^n l_0}$$
我们主要需要确定的是平滑系数 $\alpha\in(0, 1)$ 和初始值 $l_0\in\mathbb R$ (常为历史均值). 两个极端的情况:
- $\alpha=1$时, $\hat{x}_{n+1} = x_n$, 即 naive method.
- $\alpha=0$时, $\hat{x}_{n+1} = l_0$. 若 $l_0$ 为历史数据的平均值, 则 $\hat{x}_{n+1}$ 为 mean method; 若 $l_0$ 为其他值, 则 $\hat{x}_{n+1}$ 也同样为一个永不更新的常数.
总而言之, $\alpha$ 越大, 越重视最近的数据, 越小, 越重视历史数据 (历史数据的 ‘decay’ 越慢).

除了通过经验 empirically 确定 $\alpha, l_0$ 外, 还可以通过最小化预测误差的方法来确定 $\alpha, l_0$. 例如我们可以定义一个平方和误差函数:
$$\begin{aligned}
\text{SSE}(\alpha, l_0) &:= \sum_{n=1}^N (x_n - \hat{x}_n)^2\\
&= \sum_{n=1}^N \left[x_n - \left(\alpha\sum_{j=0}^{n-1} (1-\alpha)^j x_{n-j} + (1-\alpha)^n l_0)\right)\right]^2
\end{aligned}$$
即对于每一期数据, 我们都可以用指数平滑公式进行预测, 然后计算预测值与真实值的平方误差, 最后对所有误差求和. 我们可以通过最小化这个误差函数来确定 $\alpha, l_0$ 的值.

### Recursive Form of SES

上面的指数平滑公式还可以完全等价的写成递推形式, 而这种形式或许会更符合直觉. 这里不加证明地给出递推公式 (证明细节可以通过归纳法完成):
$$
\boxed{\hat{x}_{t+1} := l_t = \alpha x_t + (1-\alpha) l_{t-1}}
$$
即我们总可以通过上一期的预测值 $l_{t-1}$ 和当前时刻的观测值 $x_t$ 的加权和来预测下一期$x_{t+1}$的值. 这个递推公式的初始条件是 $l_0$, 也就是我们前面提到的初始值 (initial level), 对于其他的 $l_t$ 也类似称为 $t$ 时刻的平滑水平 (level / smoothed value). 其直观理解是:  $\small\text{新水平} = \alpha{\small\text{新信息}} + (1-\alpha){\small\text{旧水平}}$. 并且 SES 可以被看作“持续更新当前水平以预测下一期的方法”.

上面的递推公式把 $(1-\alpha) l_{t-1}$ 项展开重新整理的话, 还可以写作另一种递推形式:
$$l_t = l_{t-1} + \alpha(x_t - l_{t-1})$$
这更体现了“更新”这一概念: 新水平等于旧水平加上一个修正量, 这个修正量正是 $\alpha$ 倍的旧水平的预测误差. 因此当我们获得了一期的新信息时, 我们就可以用这个修正量来更新我们的预测值.

事实上, 这个递推的形式是更 general 的一种形式. 从广义的“状态空间 (state space)” 的角度来看, 我们定义了一个系统的当前状态 $l_t$ (即序列在 $t$ 时刻的平滑水平), 每当我们获得了一个新信息 $x_t$ 时, 我们就可以用这个新信息来更新我们的状态. 这使得我们在面对即使包含了趋势, 季节性等复杂结构的时间序列时, 也可以通过同样类似的思路来进行更新预测 —— 只不过我们需要再多定义几个状态变量 (如趋势项 $\beta_t$, 季节性项 $s_t$ 等), 以及对应的更新规则.

![Comparing the original time series of daily temperature measurements of a cow (black) with a smoothed version of it (blue)](https://yanshuo.quarto.pub/nus-ts-book/07-exponential_smoothing_files/figure-html/fig-exponential-cow-smoothed-1.png)

## Holt's Linear Trend Method

呈上所述, SES 只包含了一个平滑水平 $l_t$, 这使得其往往只能在大体平稳, 没有明显趋势或季节性的时间序列上表现较好. 而对于包含了趋势的时间序列, SES 往往只能给出一个水平的预测, 无法很好地外推一个趋势. 

Holt's linear trend method 是对 SES 的一个扩展, 通过引入一个”斜率“趋势项 $b_t$ 来更好地捕捉时间序列的趋势. 具体而言, Holt‘s Method 定义了两个状态变量: 平滑水平 $l_t$ 和趋势项 $b_t$, 并且对应的更新规则也有所改变:
$$\begin{aligned}
l_t &:= \alpha x_t + (1-\alpha)(l_{t-1} + b_{t-1})\\
b_t &:= \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}
\end{aligned}$$
其中 $0 < \alpha, \beta \leq 1$ 是两个平滑系数. 沿用递推形式的理解思路, Holt's Method 可以被理解为: 
- **$l_t$ 项**: 我们用“前一期水平+前一期趋势” $l_{t-1} + b_{t-1}$ 与新信息 $x_t$ 的一个加权平均来预测当前水平 $l_t$ (某种意义上也表示了我们用“历史数据预测值”与“真实值”进行加权平均来进行预测)
  - 而 $l_{t-1} + b_{t-1}$ 可以被理解为只掌握 $t-1$ 时刻的信息时, 对 $t$ 时刻的水平的预测 (一个均值水平加上一个趋势修正).
  - $x_t$ 则是新的信息, 我们用真正准确的观测值来修正我们的预测.
- **$b_t$ 项**: 我们用“新旧水平的差值 (相当于对当前增长趋势的一个直接估计)” $l_t - l_{t-1}$ 与前一期的趋势 $b_{t-1}$ 的加权平均来预测当前的趋势 $b_t$.
  - 这里的 $\beta$ 可以被理解为对于趋势的一个平滑系数, 即我们对于趋势的更新是基于上一期的趋势的一个加权修正. $\beta$ 越大, 越依赖 $l_t - l_{t-1}$, 更新越激进; 越小, 越依赖 $b_{t-1}$, 更新越保守.
  

当我们有了 $l_t, b_t$ 之后, 我们就可以用这两个状态变量来预测下 $h$ 期的值:
$$\hat{x}_{t+h} = l_t + h b_t$$
- 如果我们认为当前 $t$ 时刻的水平是 $l_t$, 且当下的趋势是 $b_t$, 那么我们可以认为在 $h$ 期后, 时间序列的水平会增加 $h b_t$.

这里的 $\alpha, \beta$ 等参数可以通过经验法则或者最小化预测误差的方法来确定. 例如$l_0$ 常常会取历史数据的平均值, 而 $b_0$ 则可以取前几期的差分值, 如 $x_2 - x_1$, $x_3 - x_2$ 等来估计.

另外, 当 $\beta=0$ 时, 不再更新趋势项 $b_t$, 这就退化为 SES. 


## Damped Trend Method

Holt's Method 通过引入趋势项 $b_t$ 来更好地捕捉时间序列的趋势, 但是在预测时, 我们就是简单地假设时间序列会一直保持当前的趋势, 无脑的用 $l_t + h b_t$ 来预测 $h$ 期后的值. 这对于短期的预测可能是有效的; 但是对于长期的预测, 这样的假设可能会过于乐观, 在很多情况下这种线性的趋势并不能无限的延续下去 (并且通常来讲, 时间序列的趋势是会逐渐减弱的, 如人口增长饱和, 经济增速减缓等).

Damped trend method 是对 Holt's Method 的一个改进, 通过引入一个“衰减”系数 $\phi \in (0, 1)$ 来逐渐减弱趋势的影响, 使得越远期的预测, 趋势的影响越小, 以避免过度外推趋势项带来的大误差. 具体而言, Damped trend method 的更新规则为:
$$\begin{aligned}
l_t &:= \alpha x_t + (1-\alpha)(l_{t-1} + \phi b_{t-1})\\
b_t &:= \beta(l_t - l_{t-1}) + (1-\beta)\phi b_{t-1}
\end{aligned}$$
其中 $\phi$ 是一个衰减系数, 用于控制趋势项的衰减速度. 当 $\phi=1$ 时, 该方法退化为 Holt's Method; 当 $\phi=0$ 时, 该方法退化为 SES. 其实很直观, Damped trend method 就是把所有的 Holt's Method 中的 $b_t$ 都替换为了 $\phi b_t$, 使得趋势项的影响逐渐减弱.

而在估计出 $l_t, b_t$ 之后, 我们的预测公式也会有所改变. 对于 $h$ 期后的预测:
$$
\hat{x}_{t+h} = l_t + \phi b_t + \phi^2  b_t + \cdots + \phi^{h} b_t = l_t + \sum_{j=1}^{h} \phi^{j} b_t
$$
这里从原先简单的 $h\times b_t$ 变成了一个衰减的级数求和 $\sum_{j=1}^{h} \phi^{j} b_t$. 这就使得我们的预测在越远期的预测中, 越来越受到趋势项的衰减影响, 以避免过度外推趋势项带来的大误差.

对于趋势衰减, 补充一个数学上的理解. 对于长期预测 ($h \to \infty$), 我们可以计算出:
$$
\lim_{h\to\infty} \sum_{j=1}^{h} \phi^{j} = \frac{\phi}{1-\phi}
$$
对应的预测值为:
$$
\lim_{h\to\infty} \hat{x}_{t+h} = \lim_{h\to\infty} l_t + \sum_{j=1}^{h} \phi^{j} b_t = l_t + \frac{\phi}{1-\phi} b_t
$$
这意味着, 即使推广到无穷远 ($h\to\infty$ 期之后) 的未来, 我们的预测值仍然是一个有穷的值, 趋势的贡献最多只能是 $\frac{\phi}{1-\phi} b_t$. 因此越小的 $\phi$ 就代表着越强的趋势衰减, 使得远期的预测更快的趋于水平值 $l_t$. 而 $\phi=1$ 则反过来回到了没有衰减的 Holt's Method.

## Holt-Winters Seasonal Method

Holt-Winters Seasonal Method 是对 Holt's Method 的一个扩展, 用于处理**既有趋势又有季节性**的时间序列. 通过同时对水平 $l_t$, 趋势 $b_t$ 和季节性 $s_t$ 进行平滑建模, Holt-Winters Method 可以更好地捕捉时间序列的周期性变化. 

而根据其季节性因素的调整方法不同, 又可以分为 Additive Seasonal Method 和 Multiplicative Seasonal Method 两种.

### Additive Seasonal Method

事实上, Holt-Winters Method 的推广模式和 Holt's Method 类似, 只是在更新规则中又加入了一个季节性项 $s_t$ (和由于考虑了周期性而带来的一些额外调整):
$$\begin{aligned}
l_t &:= \alpha (x_t - s_{t-p}) + (1-\alpha)(l_{t-1} + b_{t-1})\\
b_t &:= \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}\\
s_t &:= \gamma(x_t - l_{t} - b_{t-1}) + (1-\gamma)s_{t-p}
\end{aligned}$$
其中 $p$ 是季节性的周期长度, $\gamma$ 是季节性的平滑系数. 具体而言:
- **$l_t$ 项**: 和 Holt's Method 类似, 我们用“前一期水平+前一期趋势” 来预测当前水平 $l_t$, 只不过当我们拿到了新一期的观测值 $x_t$ 时, 我们需要先减去 $p$ 期前的季节性项 $s_{t-p}$ (即上一个周期同期的季节性强度), 以去除季节性的影响. 相当于我们用 $x_t - s_{t-p}$ 来替代 Holt's Method 中的 $x_t$.
- **$b_t$ 项**: 该项的更新规则和 Holt's Method 完全一致, 用于更新趋势.
- **$s_t$ 项**: 该项略有争议. 在讲义及 *Forecasting: Principles and Practice (2nd ed)* 中采取的是上述的更新公式, `fpp3` 的 R 包中应该也是如此. 但是在其他的一些资料以及 R 的 `stats::HoltWinters` 函数中, 季节性项的更新规则是 $s_t = \gamma(x_t - l_{t}) + (1-\gamma)s_{t-p}$, 即不考虑趋势项的影响. 这里我们依然与讲义保持一致, 采用前者的更新规则. 不过其直观理解都是相似的: 我们用当前的新信息的观测值和当前水平及趋势的总的预测误差 $x_t - (l_{t} + b_{t-1})$ 来更新季节性项 $s_t$.

同样的, 当我们有了 $l_t, b_t, s_t$ 之后, 我们可以用这三个状态变量来预测下 $h$ 期的值:
$$\hat{x}_{t+h} = l_t + h b_t + s_{n - p + (h \mod p)}$$
- $l_t + h b_t$ 和 Holt's Method 类似, 用于预测未来的水平和趋势.
- $s_{n - p + (h \mod p)}$ 则是用于预测未来的季节性项. 这里 $n - p + (h \mod p)$ 是用于确定未来季节性项, 其结果相当于选择与 $h$ 期同期的季节性项. 例如对于 $p=12$ 的月度数据, 当 $h=1$ 时, 我们选择的是 $s_{n-11}$, 即选择与当前月份同期的季节性项, 以此类推.

在实际软件包中, $\alpha, \beta, \gamma$ 以及 $l_0, b_0, s_0$ 等参数往往都是通过最小化预测误差的方法来确定. 而初始的季节性 $s_1, s_2, \cdots, s_p$ 通常会通过第一个周期的数据来估计.

### Multiplicative Seasonal Method

Multiplicative 相对于 Additive 的区别在于, Multiplicative Seasonal Method 是对时间序列的季节性进行乘法调整. 具体而言, 我们同样需要维护三个状态变量 $l_t, b_t, s_t$, 但是在更新规则中, 我们需要对季节性进行乘法调整:
$$\begin{aligned}
l_t &:= \alpha \frac{x_t}{s_{t-p}} + (1-\alpha)(l_{t-1} + b_{t-1})\\
b_t &:= \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}\\
s_t &:= \gamma \frac{x_t}{l_{t} + b_{t-1}} + (1-\gamma)s_{t-p}
\end{aligned}$$
其实相当于前面加性中所有的 $x_t - s_{t-p}$ 都替换为了 $\frac{x_t}{s_{t-p}}$. 对应的 $s_t$ 的更新规则中也从 $x_t - l_{t} - b_{t-1}$ 变为了 $\frac{x_t}{l_{t} + b_{t-1}}$. 其余的更新规则和 Additive Seasonal Method 完全一致.

同样的, 对于 $h$ 期后的预测:
$$\hat{x}_{t+h} = (l_t + h b_t) s_{n - p + (h \mod p)}$$
- 这里不再是对季节性$s_{n - p + (h \mod p)}$ 进行加法调整, 而是对预测的水平和趋势进行乘法调整, $s_{n - p + (h \mod p)}$ 变成了一个成倍的季节因子. 


到底是选择 Additive 还是 Multiplicative Seasonal Method, 最核心的判断依据是季节性的波动幅度与时间序列整体水平之间的关系.  有几个可供参考的判断角度:
- 季节性幅度是否随整体水平而变化
  - 加性: 如果季节性波动幅度与整体水平无关, 且波动幅度在整体水平上是固定的. 例如春夏秋冬气温不会因为平均气温的升高而波动幅度大幅增加.
  - 乘性: 如果季节性波动幅度与整体水平成比例变化. 例如人口基数越大, 季节性的增减波动也会越大.
- 差值 or 比值
  - 加性: 如果季节性的波动是固定的差值, 例如每个月的销售额都会增加 1000 元.
  - 乘性: 如果季节性的波动是固定的比值, 例如每个月的销售额都会增加 10%.
- 经验与试验
  - 在实际应用中, 往往也都会两种模型都试一下, 通过交叉验证等方法借助预测误差, 或者一些 AIC, BIC 等准则来选择合适的模型.

另外需要额外指出的是, 在这个指数平滑的框架下, **对于$\log x_t$ 进行加性的 Holt-Winters 模型是不等价于对于原始数据 $x_t$ 进行乘性的 Holt-Winters 模型的**. 
- 对于对数变换的加性模型, 所有的状态更新都是在对数空间下进行的. 而在转换回原尺度时, 就会引入额外的非线性因素
- 对于原始数据的乘性模型, 其状态更新是在原始尺度下进行平滑的, 只不过在季节性部分才会引入乘法调整. 二者的预测结果是不同的.

### Damping in Holt-Winters Method

Holt-Winters Method 也可以引入**趋势**的衰减项, 使得**趋势在远期的预测中逐渐减弱** (注意 damping 仅仅是对趋势的衰减, 季节性不会受到影响). 具体而言, 我们可以在 Holt-Winters Method 的趋势更新规则中引入一个衰减系数 $\phi$ (这里以乘性季节性为例, 但在加性上也是完全通用的):
$$\begin{aligned}
l_t &:= \alpha \frac{x_t}{s_{t-p}} + (1-\alpha)(l_{t-1} + \phi b_{t-1})\\
b_t &:= \beta(l_t - l_{t-1}) + (1-\beta)\phi b_{t-1}\\
s_t &:= \gamma \frac{x_t}{l_{t} + \phi b_{t-1}} + (1-\gamma)s_{t-p}
\end{aligned}$$
其实还是同样的用 $\phi b_{t-1}$ 来替代原先的 $b_{t-1}$, 使得趋势在远期的预测中逐渐减弱.
未来$h$ 期的预测为:
$$\hat{x}_{t+h} = \left[
    l_t + (\phi + \phi^2 + \cdots + \phi^h) b_t
\right]s_{n - p + (h \mod p)}$$

引入 damping 部分的思路都与前面的 Damped trend method 类似. 不过回忆在不考虑季节性的情况下, 极限状态中会使得对于久远的未来的预测趋于一个有限的常数: $\lim_{h\to\infty} \hat{x}_{t+h} = l_t + \frac{\phi}{1-\phi} b_t$. 这一点在季节性情况下略有不同, 因为虽然趋势项还是会类似的衰减到一个有限的常数, 但是季节项并不会受到衰减的影响, 因此在极限状态下, 预测仍然会呈现持续的季节性波动.

## Exponential Smoothing in R `fable::ETS()`

在 fable 框架中，如果要对某个时间序列变量 X（列名）拟合一个指数平滑模型，可以用下面的通用写法：
```{r}
ETS(X ~ error(O1) + trend(O2) + season(O3))
```
- `error(O1)`: O1 可替换为 `"A"` (additive error) 或 `"M"` (multiplicative error)
  - Additive error: 认为观测值符合: $x_t = \hat x_t + e_t$, 也就是观测值是预测值加上一个误差项的形式.
  - Multiplicative error: 认为观测值符合: $x_t = \hat x_t \cdot (1 + e_t)$, 也就是观测值是预测值乘以一个误差项的形式.
  - **Add 还是 Mult Error 对于点估计的预测差别不大; 其主要影响的是区间估计和估计的分布状况, 因此根据预测的关注不同, 可以选择不同的 error 模型.**
- `trend(O2)`: O2 可替换为 `"N"` (no trend), `"A"` (additive trend) 或 `"Ad"` (damped additive trend)
- `season(O3)`: O3 可替换为 `"N"` (no seasonality), `"A"` (additive seasonality) 或 `"M"` (multiplicative seasonality)

这样就能组合出不同的 ETS 模型，如
- `ETS(X ~ error("A") + trend("N") + season("N"))` 对应 Simple Exponential Smoothing
- `ETS(X ~ error("A") + trend("A") + season("N"))` 对应 Holt-Winters Additive Method
- `ETS(X ~ error("M") + trend("Ad") + season("M"))` 对应 Damped Holt-Winters Multiplicative Method

至于到底该选择哪种模型?
- 我们应当观察时序数据的序列特征: 是否具有趋势? 趋势是否会衰减?是否具有季节性? 季节性是加法还是乘法?
- 如果在代码中不指定具体的 error, trend, season, 则 fable 会在可能的组合中自动选择最优的模型
- 我们也可以手动指定几种可行的备选组合, 然后用 AIC 等准则来选择最优的模型

经验上, `ETS(A,N,M)`,`ETS(A,A,M)`,`ETS(A,Ad,M)` 的数值稳定性较差, 容易使得模型难以收敛或者出现意料之外的结果. 在实践中应当避免使用这些模型.

当序列里存在零值或负值时, 乘法模型会出现问题, 因为乘法模型要求所有的值都是正数. 这时候可以考虑使用加法模型.

> 为什么叫 ETS ? ETS 是 Error, Trend, Seasonality 的缩写. 其核心思想就是把这这三个因素通过状态空间的形式进行建模, 它们都是随时间演变的潜变量 (latent variables). 