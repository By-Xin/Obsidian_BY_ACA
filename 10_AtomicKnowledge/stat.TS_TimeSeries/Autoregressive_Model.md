---
aliases: ['自回归模型', 'AR Model', 'ARIMA']
tags:
  - concept
  - model
  - math/statistics
  - time-series
related_concepts:
  - [[Stationary_Stochastic_Process]]
  - [[Forecasting]]
---

#TimeSeries 
# Autoregressive Model (I)

## The Idea of Autoregressive Model

Autoregressive (AR) 模型是用统计学的方法来描述时间序列的一种典型模型. 假设我们有一个时间序列 $X_t, t=1,2,...,n$, AR 模型关注的是当前时刻的值 $X_t$ 和之前时刻的值 $X_{t-1}, X_{t-2}, ...$ 之间的关系. 其具体形式最开始是由 G.U. Yule 在 1926 年首次提出, 并在 1927 年用于分析解释太阳黑子的数据. 因此我们先暂不直接讨论其具体数学形式, 而是从 Sunspot data 出发, 并且通过额外引入一个 Sinusoidal data 来讨论 AR 模型的基本思想.

### Sunspot Data

Sunspot 是太阳表面的黑子由于太阳周期性活动而产生的一种现象. 从 1700 年开始, 太阳黑子的数量就被记录下来, 从而形成了一个时间序列. 我们可以用这个时间序列来研究太阳活动的周期性变化. 下面是太阳黑子数量的时间序列图:

![Yearly mean total sunspot number from 1700 to 2023.](https://yanshuo.quarto.pub/nus-ts-book/11-ar_files/figure-html/fig-ar-sunspots-1.png)

### Sinusoidal Modeling

对于这种周期性的行为的刻画很自然的联想到通过三角函数来描述. 对于时间序列 $X_t$, 我们希望定义一个 signal + noise 的模型:
$$
X_t = \mu + \alpha_1 \cos(\omega t) + \alpha_2 \sin(\omega t) + W_t
$$
其中 $\mu$ 是整体的均值, $\alpha_1, \alpha_2$ 是振幅, $\omega$ 是频率这里我们根据太阳活动的观测取 $\omega = 2\pi/11$ (太阳黑子的活动周期一般为11年), $W_t\sim \text{WN}(0, \sigma^2)$ 是白噪声且$\sigma^2$ 未知.

我们将 Sunspot 的具体数据带入到这个模型中, 就可以通过诸如最小二乘法等方法来估计参数 $\mu, \alpha_1, \alpha_2, \sigma^2$. 具体而言, 在 R 中我们可以通过以下代码来实现:

```R
model(sin_model = TSLM(Sunspots ~ 
        sin(2 * pi * Year / 11) + cos(2 * pi * Year / 11)))
```

我们将真实的观测 (Observed, 黑色), 模型拟合的结果即 $\hat X_t = \hat \mu + \hat \alpha_1 \cos(\omega t) + \hat \alpha_2 \sin(\omega t)$ (Fitted, 蓝色) 以及模拟的信号即拟合数据加上随机生成的 white noise $\hat X_t + W_t$ (Simulated, 黄色) 画在一起, 如下图所示:
![Sunspot data and fitted sinusoidal model](https://yanshuo.quarto.pub/nus-ts-book/11-ar_files/figure-html/fig-ar-sunspots-sin-model-fitted-1.png)

如果更进一步我们用真实的观测值减去拟合的结果, 就可以得到残差序列, 如下图所示:
![Residuals of the fitted sinusoidal model](https://yanshuo.quarto.pub/nus-ts-book/11-ar_files/figure-html/fig-ar-sunspots-sin-model-residuals-1.png)
然而这个残差序列的模式并不是很符合一个 white noise 的特性, 并且依然具有较强的周期性. 这说明我们的模型还不够完善, 遗留了部分数据信息没有提取出来. 具体而言, 这个模型在如下几个方面存在明显问题:
1. 观测数据的振幅比模型拟合的单一正弦波更大
   - 例如在 1960 年左右，真实数据的波动幅度远超模型预测
     - 太阳黑子数据的变化不只是一个简单的正弦波, 可能还有其他因素影响
     - 振幅可能随时间变化, 而模型没有考虑这一点
2. 观测数据的周期略有变化
    - 真实数据的周期并不是固定的 11 年，而是略微变化的
    - 模型的周期是固定的，所以无法完全匹配数据
3. 生成的 `Sample` 轨迹比真实数据更噪声化
   - `Sample` (即基于拟合值 + 随机噪声的模拟数据) 比 `Observed` (真实数据) 更加波动
     - 噪声项的处理方式可能不合理, 导致模型在模拟数据时引入了过多的随机性
     - 现实中的噪声可能是有结构的(非纯白噪声)，而不仅仅是正态随机变量

出于以上种种原因, Yule 虽然依然基于正弦波模型, 但是提出了更加复杂的 AR 模型来描述时间序列. 

### Understanding Periodicity from ODEs

依然从 Sunspot 的 sinusoidal 模型出发, 观察方程:
$$
X_t = \mu + \alpha_1 \cos(\omega t) + \alpha_2 \sin(\omega t) + W_t
$$


事实上这是如下一个简单的二阶常微分方程 (ODE) 的解:
$$
x''(t) = - \omega^2 \left( x(t) - \mu \right) 
$$

而这个 ODE 经常在物理中用来刻画例如弹簧振子, 单摆等系统的运动行为. 在理想状态下 (对于一个纯粹的确定性系统), 该 ODE 就很好的描述了例如单摆等如何在一个“平衡位置”$\mu$附近的振动行为. 

但是在现实中, 由于各种因素的影响, 我们观测到的数据往往会包含噪声和随机扰动. 因此我们需要将这个 ODE 引入到一个随机微分方程 (SDE) 的框架下, 从而可以更好的描述真实数据的行为. 例如, 我们可以将上述 $x'(t) = - \omega^2 \left( x(t) - \mu \right)$ 改写为:
$$
\mathrm{d}^2X_t = - \omega^2 \left( X_t - \mu \right)+  \mathrm{d}W_t
$$
的形式, 其中 $\mathrm{d}W_t$ 表示随机扰动, 通常为 Brownian Motion. 


### From SDEs to AR Models

为了讨论方便, 首先引入如下几个记号 (算子):
- Backward Shift Operator (滞后算子 $\mathrm B$): 对于任意的时间序列 $X_t$, 定义 
    $$\mathrm B X_t = X_{t-1},\quad  \mathrm B^k X_t = \mathrm B^{k-1} (\mathrm B X_t) = X_{t-k} ~~(\forall k \in \mathbb N).$$
- Identity Operator (恒等算子 $\mathrm I$): 对于任意的时间序列 $X_t$, 定义 
    $$\mathrm I X_t = X_t.$$
- Difference Operator (差分算子 $\nabla$): 对于任意的时间序列 $X_t$, 定义 
    $$\nabla X_t = X_t - X_{t-1} = (\mathrm I - \mathrm B) X_t.$$

在这个记号下, 我们可以将 SDE 离散化处理进行化简, 即用差分算子 $\nabla$ 来代替微分算子 $\mathrm d$. 例如对于二阶导数:
$$
X''(t) \approx X_{t+1} - 2X_t + X_{t-1} = \nabla^2 X_{t+1}.
$$
因此对于我们刚刚提到的 ODE:
$$
X''(t) = - \omega^2 \left( X_t - \mu \right)
$$
我们可以将其离散化为:
$$
\nabla^2 X_{t+1} = (\mathrm I -\mathrm B)^2 X_{t+1} = - \omega^2 \left( X_t - \mu \right).
$$

更进一步, 根据 Taylor 展开, 当 $\omega$ 足够小的时候 $\cos \omega \approx 1 - \frac{\omega^2}{2}$ 即 $- \omega^2 \approx 2(1 - \cos \omega)$. 因此我们可以将上述方程进一步简化为:
$$
(\mathrm I -\mathrm B)^2 X_{t+1} = 2(\cos \omega - 1) \left( X_t - \mu \right).
$$

### Autoregressive Model

因此根据上述的推导, Yule 在 1927 年提出了如下的 AR 模型:
$$
(\mathrm I -\mathrm B)^2 X_{t} = 2( \cos \omega - 1) \left( X_{t-1} - \mu \right) + W_{t}
$$
其中 $W_t \sim \text{WN}(0, \sigma^2)$ 是白噪声. 其中 $\mu, \omega, \sigma^2$ 是未知参数, 通过最小二乘法等方法可以估计出来.

如果将恒等算子 $\mathrm I$ 和差分算子 $\nabla$ 展开, 上述 AR(2) 模型可以进一步写为:
$$
X_t  = \alpha + 2\cos \omega X_{t-1} - X_{t-2} + W_t
$$
其中 $\alpha = 2(1 - \cos \omega) \mu$. 因此, 该模型的本质就是用两阶历史的数据的某个线性组合来预测当前时刻的数据, 并且引入了一个随机噪声项. 由于我们最高只引入了两阶历史数据, 因此这个模型被称为 AR(2) 模型.

若对该模型进行估计并一步向前预测, 最终得到的结果如下图 (fitted) 所示:
![Sunspot data and fitted AR(2) model](https://yanshuo.quarto.pub/nus-ts-book/11-ar_files/figure-html/fig-ar-sunspots-ar-model-fitted-1.png)
可见这个拟合效果要好于之前的正弦模型. 如果我们进一步观察残差序列, 如下图所示:
![Residuals of the fitted AR(2) model](https://yanshuo.quarto.pub/nus-ts-book/11-ar_files/figure-html/fig-ar-sunspots-ar-model-residuals-1.png)
可以看到残差序列的波动性已经大大减小更接近了一个 white noise 的特性. 


## AR Model in General

### Definition

下正式地给出 $\text{AR}(p)$ 模型的定义. 对于一个时间序列 $X_t$, 若其满足如下形式:
$$
X_t = \alpha + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + W_t
$$
或等价地
$$
X_t - \phi_1 X_{t-1} - \phi_2 X_{t-2} - \cdots - \phi_p X_{t-p} = \alpha + W_t
$$
其中 $W_t \sim \text{WN}(0, \sigma^2)$ 是白噪声且独立于历史数据$X_{t-1}, X_{t-2}, ...$, 而 $\alpha, \phi_1, \phi_2, \cdots, \phi_p, \sigma^2$ 是未知参数, 则称该时间序列服从 $\text{AR}(p)$ 模型. 其中 $p$ 被称为 AR 模型的阶数. 

如果考虑使用滞后算子 $\mathrm B$ 来简化上述表达, 则上述 $\text{AR}(p)$ 模型可以写为:
$$

\left(1 - \phi_1 \mathrm B - \phi_2 \mathrm B^2 - \cdots - \phi_p \mathrm B^p \right) X_t \triangleq \boxed{\boldsymbol{\phi}(\mathrm B) X_t = \alpha + W_t 
}
$$
其中 $\phi(\cdot)$ 是一个多项式函数, 表示:
$$
\boldsymbol{\phi}(x) = 1 - \phi_1 x - \phi_2 x^2 - \cdots - \phi_p x^p.
$$
在时序分析中也称为该 $\text{AR}(p)$ 模型的 ***AR 特征多项式 (AR characteristic polynomial)***, 称 $\boldsymbol{\phi}(\mathrm B)$ 为 ***自回归算子 (AutoRegressive Operator)***.

### Stationarity of AR Model

并不是所有具有形如
$$
X_t = \alpha + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + W_t
$$
形式的时间序列都是一个平稳的 AR 模型. 只有当模型的系数 $\phi_1, \phi_2, \cdots, \phi_p$ 满足一定的平稳条件时, 该时间序列才是平稳的. 

> **[回忆]** 我们称一个时间序列 $X_t$ 是(弱)平稳的, 如果: (1)其均值 $\mu$ 恒为常数; (2) 且对于任意的时间$t$和滞后步长$h$, 其自协方差 $\gamma_X(t,t+h) = \text{Cov}(X_t, X_{t+h})$ $= \text{Cov}(X_0, X_h)$ $= \gamma_X(0,h) := \gamma_X(h)$ 仅仅依赖于滞后步长$h$而与时间$t$无关.

首先不加证明地给出下面的判断结论:
对于一个 $\text{AR}(p)$ 模型, 其对应的 AR 特征多项式为:
$$
\boldsymbol{\phi}(x) = 1 - \phi_1 x - \phi_2 x^2 - \cdots - \phi_p x^p.
$$
***当且仅当特征方程 (characteristic equation)*** 
$$
\boldsymbol{\phi}(x) = 1 - \phi_1 x - \phi_2 x^2 - \cdots - \phi_p x^p = 0
$$
***的所有根 $x_1, x_2, \cdots, x_p$ 的绝对值(若有复数根则取模)都大于1***, 即 $\boxed{|x_i| > 1, i=1,2,...,p}$, 时, 该 $\text{AR}(p)$ 模型具有***唯一的平稳解***. 我们又称上述的条件为 ***平稳条件 (stationarity condition)***, 称 AR 特征方程的根的绝对值 (模) 大于1 为 ***在单位圆外 (outside the unit circle)***.

---

比较容易验证, 对于 $\text{AR}(1), \text{AR}(2)$, 上述平稳条件可以进一步简化为:

***Claim 1:*** 对于一个 $\text{AR}(1)$ 模型:
$$
X_t = \alpha + \phi X_{t-1} + W_t
$$
其 AR 特征多项式为 $\boldsymbol{\phi}(x) = 1 - \phi x$. 当且仅当 $\boxed{|\phi| < 1}$ 时, 该 $\text{AR}(1)$ 模型具有唯一的平稳解.

***Claim 2:*** 对于一个 $\text{AR}(2)$ 模型:
$$
X_t = \alpha + \phi_1 X_{t-1} + \phi_2 X_{t-2} + W_t
$$
其 AR 特征多项式为 $\boldsymbol{\phi}(x) = 1 - \phi_1 x - \phi_2 x^2$. 当且仅当 $\boxed{\phi_2 \pm \phi_1 < 1~ ~\text{and}~~  |\phi_2| < 1}$ 时, 该 $\text{AR}(2)$ 模型具有唯一的平稳解.

***Claim 3:*** 对于一个 $\text{AR}(p)$ 模型:
$$
X_t = \alpha + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + W_t
$$
其 AR 特征多项式为 $\boldsymbol{\phi}(x) = 1 - \phi_1 x - \phi_2 x^2 - \cdots - \phi_p x^p$. 
- 按照定理, 我们需要验证所有的根 $|x_i| > 1, i=1,2,...,p$. 但是这个验证过程涉及到高次方程的求解, 通常需要借助数值计算的方法来求解.
- 另外给出一个该模型的平稳的必要条件(即若该条件不满足, 则该模型一定不是平稳的): $\boxed{\phi_1 + \phi_2 + \cdots + \phi_p < 1 \small \text{ and } |\phi_p| < 1}$. 可以用该条件初步筛选出一些不平稳的 AR 模型.

---

#### AR 模型的平稳解与因果解

-  AR 模型的解是什么?
   - 一个可能的角度是: 我们说一个 AR 模型的解, 是因为我们总希望能够将这种自回归的关系拆解开, 从而能够更好地理解时间序列的行为. 
   - 例如对于一个 $\text{AR}(1)$ 模型 $X_t = \alpha + \phi X_{t-1} + W_t$, 我们可以递归写出 $X_{t-1} = \alpha + \phi X_{t-2} + W_{t-1}$, 进而得到 $X_t = \alpha + \phi (\alpha + \phi X_{t-2} + W_{t-1}) + W_t$ $= \alpha(1 + \phi) + \phi^2 X_{t-2} + \phi W_{t-1} + W_t$. 
   - 以此类推, 如果一直递归下去 (到无穷步), 我们就完全解开了这个 AR(1) 模型的关系形式上形如:
    $$\begin{aligned}
        X_t &= \mu + \psi_0 W_t + \psi_1 W_{t-1} + \psi_2 W_{t-2} + \cdots \\
        &= \mu + \sum_{i=0}^\infty \psi_i W_{t-i}
     \end{aligned}$$
    - 某种意义上, 我们说找到了一个 AR 模型的解, 就是确定了一个上述的 $\{\psi_i\}_{i=0}^\infty$ 的序列使得我们的 AR 关系成立. 

   - 例如: 考虑一个 $\text{AR}(1)$ 模型 $X_t =0.5 X_{t-1} + W_t$. 
      -  $X_t = W_t + 0.5 W_{t-1} + 0.5^2 W_{t-2} + \cdots = \sum_{j=0}^\infty 0.5^j W_{t-j}$. 是上述模型的一个解.
          - 这可以通过将 $X_t=\sum_{j=0}^\infty 0.5^j W_{t-j}$ 代入 $X_t =0.5 X_{t-1} + W_t$ 来验证.
            -  $X_{t-1}= 0.5 W_{t-1} + 0.5^2 W_{t-2} + \cdots$, 代入 $X_t =0.5 X_{t-1} + W_t$ 可得 $X_t =0.5 \sum_{j=0}^\infty 0.5^j W_{t-j} + W_t = \sum_{j=0}^\infty 0.5^{j+1} W_{t-j} + W_t = \sum_{j=0}^\infty 0.5^j W_{t-j}$.
      - $X_t = 0.5^t + W_t + 0.5 W_{t-1} + 0.5^2 W_{t-2} + \cdots = 0.5^t + \sum_{j=0}^\infty 0.5^j W_{t-j}$. 是上述模型的另一个解. (验证同上)

    - 综上所述, 对于一个给定的 AR 模型, 我们可以找到无穷多个解满足 AR 关系.
 
- AR 模型的因果解(Causal Solution)
   - 特别指出, 观察上述解的形式 $X_t = \mu + \psi_0 W_t + \psi_1 W_{t-1} + \psi_2 W_{t-2} + \cdots$, 可以发现我们都是在用$t$时刻及以前的数据(白噪声) $W_t, W_{t-1}, W_{t-2}, ...$ 来预测当前时刻的数据 $X_t$. 
   - 这种只利用当前及历史数据来预测未来数据的概念也称为符合***因果的(causal)***. 这样的一个解也称为AR模型的一个***因果解(causal solution of AR model)***. 
   - 其他的可能的关系式, 例如 $X_t = \mu + \psi_0 W_t + \psi_1 W_{t+1} + \psi_2 W_{t+2} + \cdots$ 或许可以满足 AR 模型的关系, 但是并不在我们的考虑范围内. 在这里, 我们只关注因果解.

- AR 模型的平稳解的存在性与唯一性
   - 一个 AR 模型的平稳解是否存在? 若存在, 是否唯一?
   - 针对平稳解的存在性给出如下断言: **对于一个$\text{AR}(p)$ 模型, 若其满足上述平稳条件 (即所有的特征方程的根的模大于1), 则该模型一定存在一个平稳解.**
   - 并且进一步给出唯一性的结论: **若一个 AR 模型的平稳解存在, 则该平稳解是唯一的.**
   - 且该唯一的平稳解一定具有如下形式:
        $$
        X_t = \mu + \psi_0 W_t + \psi_1 W_{t-1} + \psi_2 W_{t-2} + \cdots = \mu + \sum_{j=0}^\infty \psi_j W_{t-j}
        $$
        - 其中 $\mu$ 是常数 (而且不是别的, 恰恰是该 AR 模型的均值), $\{\psi_j\}_{j=0}^\infty$ 是一个序列. 通过后面的讨论我们可以知道, 能够满足条件的$\{\psi_j\}$ 是唯一的, 这也反映了平稳解的唯一性.

  - 换言之, 寻找一个 AR 模型的平稳解, 就等价于寻找符合条件的 $\{\psi_j\}$ 的取值. 

- AR 的平稳解的求解 (待定系数法). (以$\text{AR}(1)$ 模型 $X_t = \alpha + \phi X_{t-1} + W_t$ 的平稳解, 其中 $W_t \sim \text{WN}(0, \sigma^2), |\phi| < 1$ 为例)
  - 由题干给出 $|\phi| < 1$, 故该模型符合平稳条件, 因此存在唯一的平稳解.
  - 根据定理保证, 如果存在这样的平稳解, 则该解的形式一定形如 $X_t = \mu + \psi_0 W_t + \psi_1 W_{t-1} + \psi_2 W_{t-2} + \cdots = \mu + \sum_{j=0}^\infty \psi_j W_{t-j} \quad [\dagger]$. 因此求解的本质就是求解 $\{\psi_j\}$ 的取值.
  - 另外, 根据平稳解的形式, 我们回退一个时间步, 还可以同理给出 $X_{t-1} = \mu + \psi_0 W_{t-1} + \psi_1 W_{t-2} + \psi_2 W_{t-3} + \cdots = \mu + \sum_{j=0}^\infty \psi_j W_{t-1-j} \quad [\circ]$.
  - 把 $\dagger, \circ$ 分别对应代入 $X_t = \alpha + \phi X_{t-1} + W_t$ 的 $X_t$ 和 $X_{t-1}$ 中, 可以得到:
    $$
    \begin{aligned}
      &\mu + \sum_{j=0}^\infty \psi_j W_{t-j} = \alpha + \phi \left( \mu + \sum_{j=0}^\infty \psi_j W_{t-1-j} \right) + W_t \\
      &\Rightarrow \mu + \sum_{j=0}^\infty \psi_j W_{t-j} = \alpha + \phi \mu + \phi \sum_{j=0}^\infty \psi_j W_{t-1-j} + W_t \\
      &\Rightarrow \mu + \psi_0 W_t +  \psi_1 W_{t-1} + \psi_2 W_{t-2} + \cdots = (\alpha + \phi \mu)  + W_t + \phi \psi_0 W_{t-1} + \phi \psi_1 W_{t-2} + \cdots 
        \end{aligned}
    $$
    其中 $\mu,\psi_0, \psi_1, \psi_2, ...$ 是平稳解含有的待定系数. 由于这里的每一个白噪声都是彼此独立的, 因此只有让等式左边的 $W_k$ 项和等式右边的 $W_k$ 项的系数相等, 才能保证等式成立. 因此我们可以得到如下的等式:
      $$
      \begin{cases}
        &\mu = \alpha + \phi \mu \\
          &\psi_0 = 1 \\
          &\psi_1 = \phi \psi_0 = \phi \\
          &\psi_2 = \phi \psi_1 = \phi^2 \\
          &\cdots \\
          &\psi_k = \phi \psi_{k-1} = \phi^k
      \end{cases}
  $$
  - 综上, 我们可以得到 $\text{AR}(1)$ 模型的平稳解为
      $$X_t = \mu + \sum_{j=0}^\infty \psi_j W_{t-j} = \frac{\alpha}{1 - \phi} + W_t + \phi W_{t-1} + \phi^2 W_{t-2} + \cdots = \frac{\alpha}{1 - \phi} + \sum_{j=0}^\infty \phi^j W_{t-j}$$


### Moment of AR Model

这里给出求解一个**平稳的 AR 模型**
$$
X_t = \alpha + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + W_t
$$
的期望和方差的方法. 

从特殊到一般, 我们先考虑一个 $\text{AR}(1)$ 模型. 

#### Moments of AR(1) Model

考虑一个 $\text{AR}(1)$ 模型:
$$
X_t = \alpha + \phi X_{t-1} + W_t \quad (1)
$$
其中 $W_t \sim \text{WN}(0, \sigma^2)$ 是白噪声.

***均值函数***

由于该模型是平稳的, 因此对于任意时刻 $t$, 其均值 $\mathbb E[X_t] = \mu$ 是一个常数. 因此对 $(1)$ 式两边取期望, 可得:
$$
\mu = \alpha + \phi \mu ~~\Rightarrow~~ \mu = \frac{\alpha}{1 - \phi}.
$$


***自相关函数 ACF***

首先为了简化计算, 我们可以对 $X_t$ 进行中心化处理, 即令 $Y_t = X_t - \mu$. 则有:
$$
Y_t = \phi Y_{t-1} + W_t \quad (2)
$$
这个变换不会影响 $X_t$ 的平稳性, 并且可以证明 $\text{Cov}(X_t, X_{t+h}) = \text{Cov}(Y_t, Y_{t+h})$. 因此我们可以考虑 $Y_t$ 的 ACF 和 ACVF $\gamma_Y(h) = \text{Cov}(Y_t, Y_{t+h})$ 来求解 $X_t$ 的 ACF 和 ACVF. 

具体而言, 对于 $k$ 阶滞后的 ACF, 对于 $(2)$ 式, 我们左右两侧同时乘以 $Y_{t-k}$, 并取期望, 可得:
$$
\begin{aligned}
\mathbb E[Y_t Y_{t-k}] &= \phi \mathbb E[Y_{t-1} Y_{t-k}] + \mathbb E[W_t Y_{t-k}] = \phi \mathbb E[Y_{t-1} Y_{t-k}] 
\end{aligned}
$$
第二个等号是因为 $t$ 时刻的白噪声 $W_t$ 与历史($t-k$ 时刻)的数据 $Y_{t-k}$ 独立故该期望为0.

该式可以进一步写为:
$$
\gamma_k = \phi \gamma_{k-1}, ~~ k = 1, 2, ...
$$
若左右两侧同除以 $\gamma_0$, 则有:
$$
\rho_k = \phi \rho_{k-1}, ~~ k = 1, 2, ...
$$
而我们又知道, $\rho_0 = \text{Corr}(Y_t, Y_t) = 1$, 因此可以递推得到:
$$
\rho_k = \phi^k, ~~ k = 0, 1, 2, ...
$$

***方差函数***

同样地, 对 $(2)$ 式两边同时乘以 $Y_t$, 并取期望, 可得:
$$
\begin{aligned}
\mathbb E[Y_t^2] &= \phi \mathbb E[Y_{t-1} Y_t] + \mathbb E[W_t Y_t] \\
\Leftrightarrow \gamma_0 &= \phi \gamma_1 + \mathbb E[W_t Y_t] 
\end{aligned}
$$

对于最后一个期望项, 计算如下:
$$
\begin{aligned}
\mathbb E[W_t Y_t] &= \mathbb E[W_t (\phi Y_{t-1} + W_t)] = \phi \mathbb E[W_t Y_{t-1}] + \mathbb E[W_t^2] = \sigma^2
\end{aligned}
$$
其中第一个等号是将 AR(1) 代入, 最后一个等号是因为白噪声的方差为 $\sigma^2$ 且与历史数据独立. 因此我们可以得到:
$$
\gamma_0 = \phi \gamma_1 + \sigma^2 = \phi (\rho_1 \gamma_0) + \sigma^2 
$$
其中最后一个等号是因为 $\rho_k = \gamma_k / \gamma_0$. 因此我们可以解得:
$$
\gamma_0 = \frac{\sigma^2}{1 - \phi^2}.
$$

***ACVF***

当我们按顺序分别求得
$$
\begin{aligned}
\rho_k &= \phi^k, ~~ k = 0, 1, 2, ... \\
\gamma_0 &= \frac{\sigma^2}{1 - \phi^2}
\end{aligned}
$$
之后, 我们可以通过 $\gamma_k = \rho_k \gamma_0$ 得到所有的 $\gamma_k$.

> **[Note]**: 注意到, 我们对于 AR 模型的矩的求解是要严格按照**期望 -> ACF -> 方差 -> ACVF** 的顺序来进行的. 因为求解方差的过程需要 ACF 的信息, 对于 ACVF 也是如此.

***总结 AR(1) 的性质为***
- $\mu = \frac{\alpha}{1 - \phi}$
- $\rho_k = \phi^k, ~~ k = 0, 1, 2, ...$
- $\gamma_0 = \frac{\sigma^2}{1 - \phi^2}$
- $\gamma_k = \rho_k \gamma_0 =  \phi^k \frac{\sigma^2}{1 - \phi^2}, ~~ k = 1, 2, ...$

#### Moments of AR(2) Model

考虑一个 $\text{AR}(2)$ 模型:
$$
X_t = \alpha + \phi_1 X_{t-1} + \phi_2 X_{t-2} + W_t \quad (3)
$$
其中 $W_t \sim \text{WN}(0, \sigma^2)$ 是白噪声.

为了行文的连续性, 这里将列出 AR(2) 模型的均值, ACF 和 ACVF 的计算过程, 但不再详细展开. 我们将马上讨论如何求解一个一般的 $\text{AR}(p)$ 模型的矩. 而这里的 AR(2) 模型的结果可以直接得到. 

- ***均值函数***: $\mu = \frac{\alpha}{1 - \phi_1 - \phi_2}$
- ***自相关函数 ACF***: $\rho_k = \phi_1 \rho_{k-1} + \phi_2 \rho_{k-2}, ~~ k = 2, 3, ...$
- ***方差函数***: $\gamma_0 = \frac{ (1-\phi_2) \sigma^2}{(1 - \phi_2)(1 - \phi_1^2 - \phi_2^2) -2\phi_2\phi_1^2}$
- ***自协方差函数 ACVF***: $\gamma_k = \rho_k \gamma_0 = \phi_1 \gamma_{k-1} + \phi_2 \gamma_{k-2}, ~~ k = 1, 2, ...$


#### Moments of AR(p) Model

考虑如下平稳的 $\text{AR}(p)$ 模型:
$$
X_t = \alpha + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + W_t \quad (4)
$$
其中 $W_t \sim \text{WN}(0, \sigma^2)$ 是白噪声.

***均值函数***

由于该模型是平稳的, 因此对于任意时刻 $t$, 其均值 $\mathbb E[X_t] = \mu$ 是一个常数, 记之为 $\mu$. 因此对 $(4)$ 式两边取期望, 可得:
$$
\mu = \alpha + \phi_1 \mu + \phi_2 \mu + \cdots + \phi_p \mu ~~\Rightarrow~~ \mu = \frac{\alpha}{1 - \phi_1 - \phi_2 - \cdots - \phi_p}.
$$

***自相关函数 ACF***

在后面的求解中, 我们不妨先将 $\text{AR}(p)$ 模型转化为中心化的形式, 即令 $Y_t = X_t - \mu$. 则上面的等式等价于:
$$
Y_t = \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + W_t \quad (5)
$$
这个变换不会影响 $X_t$ 的平稳性, 并且可以证明 $Y_t$ 和 $Z_t$ 的 ACF 和 ACVF 是相同的. 

按照同样的顺序, 我们先考虑 $k$ 阶滞后的 ACF, 对于 $(5)$ 式, 我们左右两侧同时乘以 $Y_{t-k}$, 并取期望, 可得:
$$\begin{aligned}
\mathbb E[Y_t Y_{t-k}] &= \phi_1 \mathbb E[Y_{t-1} Y_{t-k}] + \phi_2 \mathbb E[Y_{t-2} Y_{t-k}] + \cdots + \phi_p \mathbb E[Y_{t-p} Y_{t-k}] \\
\Leftrightarrow \gamma_k &= \phi_1 \gamma_{k-1} + \phi_2 \gamma_{k-2} + \cdots + \phi_p \gamma_{k-p}, ~~ k = 1, 2, ...
\end{aligned}$$
若左右两侧同除以 $\gamma_0$, 则得到相关系数 $\rho_k = \gamma_k / \gamma_0$:
$$
\rho_k = \phi_1 \rho_{k-1} + \phi_2 \rho_{k-2} + \cdots + \phi_p \rho_{k-p}, ~~ k = 1, 2, ...
$$
而我们又知道, $\rho_0 = \text{Corr}(Y_t, Y_t) = 1$, 且 $\rho_k = \rho_{-k}$, 因此可以从 $k=1$ 开始依次递推遍历所有的 $\rho_k$:
$$\begin{aligned}
(k = 1) \quad&\rho_1 = \phi_1\rho_0  + \phi_2 \rho_{-1} + \cdots + \phi_p \rho_{-p+1} = \phi_1 +\phi_2\rho_1 + \cdots + \phi_p \rho_{p-1} \\
(k = 2) \quad&\rho_2 = \phi_1\rho_1  + \phi_2 + \cdots + \phi_p \rho_{-p+2} = \phi_1^2 + \phi_2 \\
\cdots\\
(k = p) \quad&\rho_p = \phi_1\rho_{p-1}  + \phi_2 \rho_{p-2} + \cdots + \phi_p \rho_0 = \phi_1 \rho_{p-1} + \phi_2 \rho_{p-2} + \cdots + \phi_p
\end{aligned}$$
而这个$p$元线性方程组可以通过矩阵形式表达为:
$$
\begin{bmatrix}
1 & \rho_1 & \rho_2 & \cdots & \rho_{p-1} \\
\rho_1 & 1 & \rho_1 & \cdots & \rho_{p-2} \\
\rho_2 & \rho_1 & 1 & \cdots & \rho_{p-3} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\rho_{p-1} & \rho_{p-2} & \rho_{p-3} & \cdots & 1
\end{bmatrix}
\begin{bmatrix}
\phi_1 \\
\phi_2 \\
\phi_3 \\
\vdots \\
\phi_p
\end{bmatrix}
=
\begin{bmatrix}
\rho_1 \\
\rho_2 \\
\rho_3 \\
\vdots \\
\rho_p
\end{bmatrix}
$$
若进一步记上述左侧矩阵为 $\mathbf R$, $\mathbf \rho = [\rho_1, \rho_2, ..., \rho_p]^\top$, $\boldsymbol \phi = [\phi_1, \phi_2, ..., \phi_p]^\top$, 则上述方程可以写为:
$$
\mathbf R \boldsymbol \phi = \mathbf \rho
$$
其中 $\boldsymbol \phi$ 是 $\text{AR}(p)$ 模型的系数向量, 对于一个具体的模型而言是给定的. 而余下的 $\mathbf R$ 和 $\mathbf \rho$ 是我们需要求解的, 且都是由 $\rho_1, \rho_2, ..., \rho_p$ 组成的, 因此是完全可以求解的. 

这个方程组也称为 ***Yule-Walker 方程组***, 可以通过求解该方程组来得到 $\rho_1, \rho_2, ..., \rho_p$ 的值.

***方差函数***

同样地, 对 $(5)$ 式两边同时乘以 $Y_t$, 并取期望, 可得:
$$
\begin{aligned}
\mathbb E[Y_t^2] &= \phi_1 \mathbb E[Y_{t-1} Y_t] + \phi_2 \mathbb E[Y_{t-2} Y_t] + \cdots + \phi_p \mathbb E[Y_{t-p} Y_t] + \mathbb E[W_t Y_t] \\
\Leftrightarrow \gamma_0 &= \phi_1 \gamma_1 + \phi_2 \gamma_2 + \cdots + \phi_p \gamma_p + \sigma^2 \\
\Leftrightarrow \gamma_0 &= \phi_1 \rho_1 \gamma_0 + \phi_2 \rho_2 \gamma_0 + \cdots + \phi_p \rho_p \gamma_0 + \sigma^2 \quad [\lozenge]
\end{aligned}
$$
其中 $\sigma^2$ 是白噪声的方差. 
- 这里 $\mathbb E[W_t Y_t] = \sigma^2$ 是因为
    $$
    \begin{aligned}
    \mathbb E[W_t Y_t] &= \mathbb E[W_t (\phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + W_t)] \\
    &= \phi_1 \mathbb E[W_t Y_{t-1}] + \phi_2 \mathbb E[W_t Y_{t-2}] + \cdots + \phi_p \mathbb E[W_t Y_{t-p}] + \mathbb E[W_t^2] \\
    &= \sigma^2
    \end{aligned}
    $$ 
    由因果性, $W_t$ 与历史数据 $Y_{t-j}$ 独立, 因此 $\mathbb E[W_t Y_{t-j}] = 0, j = 1, 2, ..., p$.

最终由 $(\lozenge)$ 式可以解得 $\gamma_0$:
$$
\gamma_0 = \frac{\sigma^2}{1 - \phi_1 \rho_1 - \phi_2 \rho_2 - \cdots - \phi_p \rho_p}
$$

***自协方差函数 ACVF***

对于 $k\geq 1$ 阶滞后的自协方差函数, 可以通过 $\gamma_k = \rho_k \gamma_0$ 得到.

### Qualitative Behavior of ACF/ACVF

由于 ACF 和 ACVF 是描述时间序列的重要工具,  这里再着重讨论一下 AR 模型中二者的性质. 考虑一个平稳的 $\text{AR}(p)$ 模型. 我们前面有推导出如下递推关系:
$$
\rho_k = \phi_1 \rho_{k-1} + \phi_2 \rho_{k-2} + \cdots + \phi_p \rho_{k-p}, ~~ k = 1, 2, ...
$$
或者根据 $\rho_k = \gamma_k / \gamma_0$ 可以得到:
$$
\gamma_k - \phi_1 \gamma_{k-1} - \phi_2 \gamma_{k-2} - \cdots - \phi_p \gamma_{k-p} = 0 , ~~ k = 1, 2, ...
$$
因此可以看出, ACF 和 ACVF 二者本身也呈现出了一种自回归的递归关系. 

以 ACVF 为例, 沿用滞后算子 $\mathrm B$ 的记号, 则上述关系可以写为:
$$
\gamma_k - \phi_1 (\mathrm B \gamma_k) - \phi_2 (\mathrm B^2 \gamma_k) - \cdots - \phi_p (\mathrm B^p \gamma_k) = 0
$$
若再记 $\boldsymbol \phi(\mathrm B) = 1 - \phi_1 \mathrm B - \phi_2 \mathrm B^2 - \cdots - \phi_p \mathrm B^p$, 则上述关系可以写为:
$$
\boldsymbol \phi(\mathrm B) \gamma_k = 0
$$
而这个 $\boldsymbol \phi(x) = 1 - \phi_1 x - \phi_2 x^2 - \cdots - \phi_p x^p$ 就是 AR 模型的特征多项式. 因此我们可以得到一个结论: **AR 模型的 ACF 和 ACVF 的取值某种程度上反映了 AR 模型的特征多项式的根的性质.**

因此这里详细讨论一下 AR 模型的 ACF/ACVF 与特征多项式的根之间的关系. 

- 考虑上述特征多项式 $\boldsymbol \phi(x) = 1 - \phi_1 x - \phi_2 x^2 - \cdots - \phi_p x^p$, 则 $p$ 次特征方程 $\boldsymbol \phi(x) = 0$ 会一共有 $p$ 个根, 不妨记为 $\mathbf x_1, \mathbf x_2, ..., \mathbf x_p$ (为方便起见, 这里不考虑重根的情况).

- 则根据一些数学推论我们可以得到 ACVF 的又一表达式:
   $$
   \gamma_k = C_1 \mathbf x_1^{-k} + C_2 \mathbf x_2^{-k} + \cdots + C_p \mathbf x_p^{-k} = \sum_{j=1}^p C_j \mathbf x_j^{-k}
    $$
    因此形式上, ACVF/ACF 的取值是由特征多项式的根的性质决定的 (而且是负幂次的形式)!
- 若更严谨一些考虑特征方程有重根的情况, 则上述表达式中的 $C_j$ 会有一定的变化, 但是对应的 $\mathbf x_j^{-k}$ 的形式不会变化.

我们已经推出了 ACF 与特征多项式的根之间的关系, 当$k$ 趋近于无穷大时 (即考虑时间序列的长程性质), 考虑如下两种情况:
- **若 AR 模型的特征多项式的根都是实数**: 则 $\gamma_k$ 的取值会随着 $k$ 的增大而 **指数衰减 (exponential decay)**.
  ![Exponential Decay of ACF of AR(1)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250321145715.png)
- **若 AR 模型的特征多项式的根有复数**: 则 $\gamma_k$ 的取值会随着 $k$ 的增大而呈现**阻尼振荡 (damped oscillation)** 的形式: $\gamma_k = A r^{-k} \cos(\omega k + \theta)$, 其中 $A,\omega, \theta$ 是常数, $r$ 是特征多项式的根的模. 主要观察 $\gamma_k$ 与 $k$ 的关系. 
  ![Damped Oscillation of ACF of AR(2)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250321145957.png)
- 区分*正负交替的指数衰减*和*阻尼振荡*的关键是观察 ACF/ACVF 的符号. 对于指数衰减, 其是严格正负交替的, 没有固定的周期; 而对于阻尼振荡, 其是有固定的周期的, 且振幅逐渐减小.