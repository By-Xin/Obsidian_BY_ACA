# A Very Simple Note on Spectrual Analysis

> Refs: https://yanshuo.quarto.pub/nus-ts-book/09-spectral_analysis.html; Time Series Analysis with Applications in R (2nd Edition) by Jonathan D. Cryer and Kung-Sik Chan

## Preliminaries

> 首先强烈安利3B1B的该视频 (阅读全文可跳转): 【官方双语】形象展示傅里叶变换 (3B1B):  https://www.bilibili.com/video/BV1pW411J7s8/?spm_id_from=333.337.search-card.all.click&vd_source=8a00dab0be94d29388f2286892ba8d50

## Introduction & General Review

- Spectral analysis, 特别是 Fourier analysis, 是时间序列分析中的一个重要工具. 与之前的直接以时间的流逝角度 (称为时域, time domain) 分析整体数据的变化趋势等不同, spectral analysis 从另一个角度出发 (称为频域, frequency domain), 更关注于提取并分析时序数据中的周期性变化.

- 回忆, 频率 $f$ 是指单位之间内信号震荡的次数, 通常以赫兹 (Hz) 为单位, 表示每秒的周期数. 例如, 2 Hz 的信号表示每秒震荡两个周期. 对应的周期 $T$ 是指信号完成一个周期所需的时间, 通常以秒为单位. 例如, 2 Hz 的信号的周期是 0.5 秒. 因此, 频率和周期是互为倒数的: $f = 1/T$. 因此用频率的角度来研究周期也是非常自然的. 

- 另一方面, 正余弦函数是最具有代表性的周期函数. 以一个一般的余弦函数为例:
    $$
    R~\cos(2\pi f t + \phi)
    $$
    - $R>0$ 是振幅 (amplitude), 表示波的高度;
    - $f$ 是频率, 表示波的周期;
    - $t$ 是时间;
    - $\phi$ 是相位 (phase), 表示波的起始位置.



- 下图展示了一个 $Y_1 = \cos(2\pi \frac{4}{96} t)$ 的余弦波, 其中 $f = \frac{4}{96} = \frac{1}{24}$ Hz, 即周期为 24 个单位. 从图中可以看出, 该余弦波每 24 个单位重复一次. 且该余弦波的振幅为 1 (在 1 和 -1 之间震荡), 相位为 0 (从 $\cos(0) = 1$ 开始).
    ![y=cos(2*pi*4/96*t)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250211145916.png)

- 下图的虚线是一个 $Y_2 = \cos(2\pi (\frac{14}{96} t + 0.3))$ 的余弦波, 其中 $f = \frac{14}{96} = \frac{7}{48}$ Hz, 即周期 $T = \frac{1}{f} = \frac{48}{7} \approx 6.857$ 个单位, 振幅为 1, 相位为 0.3. 
    ![y=cos(2*pi*(t*14/96+0.3))](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250211150336.png)

- 当然, 我们还可以对这两个余弦波进行线性组合(叠加), 例如 $Y_t = 2Y_1 + 3Y_2$, 从而得到一个新的余弦波如下图所示:
    ![Y_t = 2Y_1+3Y_2](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250211150800.png)

- 这时的周期性就开始变得复杂了. 而频谱分析就是要解析这种复杂的周期性. 事实上, 其具体的工作就类似于刚才构造的一个逆过程, 即如果给定这样一个例如 $Y_t = 2Y_1+3Y_2$ 的复杂信号, 我们能否将其分解为一系列简单的具有不同周期的正余弦波的叠加呢? 这就是频谱分析的基本思想.

## Fourier Series & Fourier Transform

### Intuition

- 我们手里现在有这样一组数据 $Y_t$, 其中 $t = 1, 2, \cdots, n$, 我们希望了解其中的周期性变化. 因此延续上面的思路, 我们试图将 $Y_t$ 分解为一系列 (具有不同频率, 即周期) 的正余弦波的叠加. 

- 假设我们一共有 $m$ 组具有不同频率的正余弦波, 分别为: $\cos(2\pi f_1 t), \sin(2\pi f_1 t), \cos(2\pi f_2 t), \sin(2\pi f_2 t), \cdots, \cos(2\pi f_m t), \sin(2\pi f_m t)$, 其中 $f_1, f_2, \cdots, f_m$ 是频率. 
- 那么我们可以将 $Y_t$ 近似表示为:
    $$\begin{aligned}
    Y_t &\approx A_0 + A_1 \cos(2\pi f_1 t) + B_1 \sin(2\pi f_1 t) + \cdots + A_m \cos(2\pi f_m t) + B_m \sin(2\pi f_m t)\\
    &= A_0 + \sum_{j=1}^m A_j \cos(2\pi f_j t) + B_j \sin(2\pi f_j t)
    \end{aligned}$$
    其中 $A_0, A_1, B_1, \cdots, A_m, B_m$ 就是一些用来调整的待定系数.

- 换言之, 如果我们能够有效的确定这些系数 $A_0, A_1, B_1, \cdots, A_m, B_m$ 使得 $Y_t$ 能够被上式很好的近似, 那么我们就真的实现了将 $Y_t$ 分解为一系列不同频率的正余弦波的叠加. 这就是 Fourier series 的基本思想.

### Fourier Coefficients & OLS

- 上面的 
  $$Y_t \approx A_0 + \sum_{j=1}^m A_j \cos(2\pi f_j t) + B_j \sin(2\pi f_j t)$$ 
  中其实还存在一点纰漏没有交代, 也就是这里还没有具体的确定我们希望分解的各个正余弦函数的频率 $f_1, f_2, \cdots, f_m$.  

- 不加证明地指出, 一个重要的观察是 (为后续方便暂时假设 $n$ 是偶数), 当我们取 $f_j = \frac{j}{n}, ~ (j = 1, 2, \cdots, n/2)$ 即 $f_1 = \frac{1}{n}, f_2 = \frac{2}{n}, \cdots, f_{n/2} = \frac{n/2}{n}$ 时, 我们的分解会有很好的性质 (后面马上会介绍). 当取这些频率时 (事实上, 这些频率也被称为 Fourier frequencies)
, 我们对应的正余弦函数变为:$\cos(2\pi \frac{j}{n} t), \sin(2\pi \frac{j}{n} t)$, 其中 $j = 1, 2, \cdots, n/2$,它们有时也被对应简记为 $\mathrm s_j(t) = \sin(2\pi \frac{j}{n} t), \mathrm c_j(t) = \cos(2\pi \frac{j}{n} t)$. 
- 这时我们的分解式变为:
    $$
    Y_t \approx A_0 + \sum_{j=1}^{n/2} \left(A_j \cos(2\pi \frac{j}{n} t) + B_j \sin(2\pi \frac{j}{n} t) \right):= A_0 + \sum_{j=1}^{n/2} (A_j  \mathrm c_j(t) + B_j  \mathrm s_j(t))
    $$
    其中 $A_0, A_1, B_1, \cdots, A_{n/2}, B_{n/2}$ 是待定系数, 或者叫作 Fourier coefficients.

- 这时可能会观察到, 上面的这个分解式在形式上和线性回归的形式非常相似! 不考虑正余弦的特殊性, 我们可以将上式写为:
    $$\begin{aligned}
    Y_t &= A_0 + A_1 \mathrm c_1 + B_1 \mathrm s_1 + \cdots + A_{n/2} \mathrm c_{n/2}\\
    Y_t &= \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_{n} x_{n}
    \end{aligned}$$
    其中 $\beta_0$ 就对应着 $A_0$, $\beta_j$ 对应着 $A_j$ 和 $B_j$, $X_j$ 对应着 $\mathrm c_j$ 和 $\mathrm s_j$! 二者没有任何本质区别. 因此, 我们可以将 Fourier series 的分解看作是一个线性回归问题, 通过 OLS 方法来估计 Fourier coefficients.
    - 注意这里的最后一项没有包括 $\mathrm s_{n/2}(t)$, 因为 $\mathrm s_{n/2}(t) = \sin(2\pi \frac{n/2}{n} t) = \sin(\pi t) = 0$. 所以其实其对应的 Fourier coefficient 也不用估计. 不过这样算上 $A_0$ 一共有 $n/2\times2 -1 +1 = n$ 个 Fourier coefficients. 正好!

- 通过几乎完全相同的步骤, 我们可以得到 Fourier coefficients 的估计公式:
    $$\begin{aligned}
    \widehat A_0 &= \bar{Y}\\
   \widehat A_j &= \frac{2}{n} \sum_{t=1}^n Y_t \cos(2\pi \frac{j}{n} t) = \frac{2}{n} \sum_{t=1}^n Y_t \mathrm c_{j}(t)\\
    \widehat B_j &= \frac{2}{n} \sum_{t=1}^n Y_t \sin(2\pi \frac{j}{n} t) = \frac{2}{n} \sum_{t=1}^n Y_t \mathrm s_{j}(t)
    \end{aligned}$$
    其中 $\bar{Y}$ 是 $Y_t$ 的均值.

- 因此, 我们得到的 Fourier series 的估计值为:
    $$\begin{aligned}
    \widehat Y_t &= \widehat A_0 + \sum_{j=1}^{n/2} (\widehat A_j \mathrm  c_j(t) + \widehat B_j \mathrm s_j(t)) 
    \end{aligned}$$
    - $\widehat A_0 = \bar{Y}$
    - $\widehat A_j = \frac{2}{n} \sum_{t=1}^n Y_t \mathrm c_{j}(t) = \frac{2}{n} \sum_{t=1}^n Y_t \cos(2\pi \frac{j}{n} t)$
    - $\widehat B_j = \frac{2}{n} \sum_{t=1}^n Y_t \mathrm s_{j}(t) = \frac{2}{n} \sum_{t=1}^n Y_t \sin(2\pi \frac{j}{n} t)$

## Periodogram (周期图)

### Periodogram 的原理

- 当我们得到了 Fourier coefficients 之后, 我们就可以有效的将时间序列 $Y_t$ 分解为一系列不同频率的正余弦波的叠加. 一个对这些频率分解进行可视化的方法就是 Periodogram. 
- 一个典型的 Periodogram 如下图所示:
    ![Periodogram](https://yanshuo.quarto.pub/nus-ts-book/09-spectral_analysis_files/figure-html/fig-spectral-employment-periodogram-1.png)
  - Periodogram 的横轴是频率, 也就是我们刚刚设置的 Fourier frequencies ($f_1 = \frac{1}{n}, f_2 = \frac{2}{n}, \cdots, f_{n/2} = \frac{n/2}{n}$) (所以 Periodogram 的横轴是总是从 0 到 0.5).
  - Periodogram 纵轴是在每个x轴对应频率下分解出的一种“强度”: $P_j = \frac{n^2}{4} (\widehat A_j^2 + \widehat B_j^2)$: 
    - 回忆我们刚刚求得 $\widehat Y_t = \widehat A_0 + \sum_{j=1}^{n/2} (\widehat A_j \mathrm  c_j(t) + \widehat B_j \mathrm s_j(t))$, 其中每个$j$就对应了一个频率下的正余弦分解$\widehat A_j \mathrm  c_j(t) + \widehat B_j \mathrm s_j(t)$.
    - 这里的“强度”就是对应的 Fourier coefficients 的平方和, 即 $P_j = \frac{n^2}{4} (\widehat A_j^2 + \widehat B_j^2)$.
- Periodogram 的直观解释是: 
  - 横轴对应频率, 纵轴对应频率对应的正余弦分解的“强度”.
  - 如果在某个频率下的“强度”存在一个峰值, 那么说明在这个频率对应的周期下, 时间序列 $Y_t$ 存在一个明显的周期性变化.
    - 具体来说, **如果数据在高频部分有很高的“强度”, 说明数据存在很多快速波动(高频噪声)**; **如果数据在低频部分有很高的“强度”, 说明数据存在一些长期的周期性变化**.

### Periodogram 的例子

#### Star Magnitude (星等)
![Periodogram of Stars](https://yanshuo.quarto.pub/nus-ts-book/09-spectral_analysis_files/figure-html/fig-spectral-star-periodogram-1.png)

- Periodogram 主要有两个明显的峰值, 一个在约 0.04167 (最高), 一个在约 0.35. 对于 $f = 0.04167$, 对应的周期为 $1/f \approx 24$ (天), 说明星星的亮度大约每 24 天有一个周期性变化; 对于 $f = 0.35$, 对应的周期为 $1/f \approx 2.857$ (天), 说明星星的亮度在约3天内也有一个周期性变化.
- 24 天的周期性变化可能是由于恒星的自转引起的, 而 3 或许和双星系统等的运动有关.

#### Lynx (猞猁皮毛)

![Periodogram of Lynx](https://yanshuo.quarto.pub/nus-ts-book/09-spectral_analysis_files/figure-html/fig-spectral-lynx-periodogram-1.png)

- Periodogram 主要有一个明显的峰值, 在约 0.1. 对应的周期为 $1/f \approx 10$ (年), 说明猞猁皮毛的数量大约每 10 年有一个周期性变化, 可能与种群的繁荣-衰退周期有关. 这可能是由于猞猁的繁殖周期引起的. 有一个次高峰在 0.2, 对应的周期为 5 年 (但比较弱).
- 相对而言, 这个 periodogram 的峰值较为“宽”, 这说明数据的周期性存在, 但不是非常稳定. 也就是说, 种群的周期并不是严格的10年一次, 而会有一些波动.

#### US Retail Employment (美国零售业就业)

![Periodogram of US Retail Employment](https://yanshuo.quarto.pub/nus-ts-book/09-spectral_analysis_files/figure-html/fig-spectral-employment-periodogram-1.png)

- 从这个 periodogram 可以看出有多个峰值, 说明美国零售业就业数据存在多个周期性变化. 例如, 在 $f=0.08$ (约12.5个月, 1年) 是一个最强的峰值,  说明就业数据有一个强烈的年度季节性变化. 另外, 在 $f=0.16$ (约6个月) 也有一个较强的峰值, 说明就业数据有一个半年度的季节性变化 (可能与春/秋的就业波动有关). 在 $f=0.25$ (约4个月) 和 $f=0.33$ (约3个月) 也有一些峰值, 说明就业数据有一些短期的季节性变化.

### Detrending in Spectral Analysis

- 在频谱分析中, 有时我们会遇到一些数据, 其中存在一些趋势性的变化, 这些趋势性的变化可能会影响我们对数据的周期性变化的分析. 因此, 有时我们需要对数据进行去趋势化 (detrending) 处理.
- 对于原始的数据, 往往包含了 Trend, Seasonal 和 Remainder 三个部分. 而 Periodogram 可以帮助我们分析 Seasonal 部分.
- 如果包含了 Trend, 则可能会导致 Periodogram 错误的将 Trend 误认为是一些低频的周期性变化, 引起分析的偏差. 因此, 我们需要对数据进行去趋势化处理.

### Applications of Spectral Analysis

1. 识别周期性变化 (比 ACF / 季节性方法更精准)
   - Periodogram 提供了更“高分辨率”的周期性变化的信息, 可以更精确的识别数据中隐藏的周期性变化,且不依赖于人工的周期长度的选择.
2. 特征提取, 时间序列分类和降噪
   - 当 Periodogram 只在某几个频率下有明显的峰值时, 这些频率就可以作为时间序列的特征, 用于时间序列的分类等 (尤见于某些生物医疗数据的分类).
   - 我们还可以去除某些不重要的频率(即设置这些 Fourier coefficients 为 0), 从而实现时间序列的降噪 (比如可以有针对性的去除某些高频噪声). 从而提取最主要的趋势或波动. 
3. 预测
   - DHR + ARIMA 是一种常见的时间序列预测方法.
   - 传统的 ARIMA 主要处理的是趋势和自回归模式, 但是无法直接处理强周期性的数据. - 而 DHR (即 Fourier transformation 等一系列应用) 更好的提取了数据的周期性变化
   - 二者结合, 可以更好的预测时间序列.

## Appendix

### Some Other Related Concepts
- 通过 FFT (Fast Fourier Transform, 快速傅里叶变换) 算法, 我们可以更快的计算 Fourier coefficients 提高计算效率, 从而更好的进行频谱分析.
- Periodogram 的强度 $P_j$ 事实上也可以通过 ACVF (Auto-Covariance Function) 来计算, 二者恰好是 Fourier transform 的关系. 事实上, Periodogram 是 ACVF 的 Fourier transform.
-  Wavelet analysis (小波分析) 是另一种频谱分析的方法. Periodogram 适用于周期性变化比较稳定的数据, 其计算的是整个时间序列的频谱, 但是不能识别某些随时间变化的频率. 但是在实际数据中, 某些周期性的模式可能只在某些时间段内出现, 这时 Wavelet analysis 就可以更好的识别这些随时间变化的频率.
   -  Wavelet analysis 是一种时域+频域的分析方法, 它可以将时间序列分解为不同频率的小波, 并且可以随时间变化. 通过 Wavelet analysis, 我们可以更好的识别数据中的短期和长期周期性变化. 
   -  其对应的可视化方法是 Scalogram. 这个图的横轴是时间,表示时间的变化. 纵轴是频率, 表示不同的频率成分. 颜色深浅表示不同频率的强度. 通过 Scalogram, 我们可以更好的识别数据中的短期和长期周期性变化.
    ![Scalogram](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250211200052.png)
- Spectral Density (谱密度) 是另一种频谱分析的方法. 它是对 Periodogram 的一种改进, 通过对 Periodogram 进行平滑处理, 从而去除了一些噪声, 使得频谱分析更加稳定. 对于一些平稳时间序列, Spectral Density 可以更好的提取数据的周期性变化.

### * Some Properties and Mathematical Details

(以下内容为一些数学细节的推导, 可以跳过, 而且感觉可能有一点typo, 有待进一步验证)

#### OLS Estimation of Fourier Coefficients

这里对 Fourier coefficients 的 OLS 估计公式进行一些简单的推导.
- 我们刚刚仿照线性回归的形式写出了 Fourier series 的分解式 (为区别 $t$ 和 $j$, 这里暂时将 $t$ 作为变量从下标改为 $Y(t)$):
    $$   Y(t) = A_0 + A_1 \mathrm c_1(t) + B_1 \mathrm s_1(t) + \cdots + A_{n/2} \mathrm c_{n/2}(t) + B_{n/2} \mathrm s_{n/2}(t)  $$
- 对照 $\mathrm Y = \mathrm X \beta + \epsilon$ 的线性回归形式, 我们对应得到:
    $$\begin{aligned}
    \mathrm Y &= \begin{pmatrix} Y(1) \\ Y(2) \\ \vdots \\ Y(n) \end{pmatrix}, \quad \mathrm Z = \begin{pmatrix} \frac{1}{\sqrt{2}} & \mathrm c_1(1) & \mathrm s_1(1) & \cdots & \mathrm c_{n/2}(1)\\ \frac{1}{\sqrt{2}} & \mathrm c_1(2) & \mathrm s_1(2) & \cdots & \mathrm c_{n/2}(2)  \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \frac{1}{\sqrt{2}} & \mathrm c_1(n) & \mathrm s_1(n) & \cdots & \mathrm c_{n/2}(n)  \end{pmatrix}, \quad \mathcal{A} = \begin{pmatrix} A_0 \\ A_1 \\ B_1 \\ \vdots \\ A_{n/2}  \end{pmatrix}
    \end{aligned}$$
    即每一个频率的正余弦分解就对应一个回归变量, 每一个时间点就对应一个观测值. **注意! 这里最后一列没有包括 $\mathrm s_{n/2}(t)$, 因为 $\mathrm s_{n/2}(t) = \sin(2\pi \frac{n/2}{n} t) = \sin(\pi t) = 0$**.
- 因此, 对照 OLS 估计的公式 $\widehat \beta = (\mathrm X^\top \mathrm X)^{-1} \mathrm X^\top \mathrm Y$, 得到:
    $$
    \mathcal{\widehat A} = (\mathrm Z^\top \mathrm Z)^{-1} \mathrm Z^\top \mathrm Y \stackrel{\dagger}{=} \frac{2}{n} \mathrm{Z}^\top \mathrm{Y}  
    $$
    其中 $\dagger$ 是因为 $\mathrm Z$ 是正交矩阵 (具体将在下面证明), 即 $\mathrm Z^\top \mathrm Z = \frac{n}{2} \mathrm I$.
- 而 $\frac{2}{n} \mathrm{Z}^\top \mathrm{Y}$ 进行展开就是 Fourier coefficients 的 OLS 估计公式:
    $$\begin{aligned}
    \frac{2}{n} \mathrm{Z}^\top \mathrm{Y} &= \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & \cdots & \frac{1}{\sqrt{2}} \\ \mathrm c_1(1) & \mathrm c_1(2) & \cdots & \mathrm c_1(n) \\ \mathrm s_1(1) & \mathrm s_1(2) & \cdots & \mathrm s_1(n) \\ \vdots & \vdots & \ddots & \vdots \\ \mathrm c_{n/2}(1) & \mathrm c_{n/2}(2) & \cdots & \mathrm c_{n/2}(n)  \end{pmatrix} \begin{pmatrix} Y(1) \\ Y(2) \\ \vdots \\ Y(n) \end{pmatrix}\\

    &= \begin{pmatrix} \frac{1}{\sqrt{2}} \sum_{t=1}^n Y(t) \\ \frac{1}{\sqrt{2}} \sum_{t=1}^n Y(t) \cos(2\pi \frac{1}{n} t) \\ \frac{1}{\sqrt{2}} \sum_{t=1}^n Y(t) \sin(2\pi \frac{1}{n} t) \\ \vdots \\ \frac{1}{\sqrt{2}} \sum_{t=1}^n Y(t) \cos(2\pi \frac{n/2}{n} t)  \end{pmatrix} = \begin{pmatrix} \bar{Y} \\ \frac{2}{n} \sum_{t=1}^n Y(t) \cos(2\pi \frac{1}{n} t) \\ \frac{2}{n} \sum_{t=1}^n Y(t) \sin(2\pi \frac{1}{n} t) \\ \vdots \\ \frac{2}{n} \sum_{t=1}^n Y(t) \cos(2\pi \frac{n/2}{n} t)  \end{pmatrix}
    \end{aligned}$$

#### Orthogonality of Fourier Frequencies

这里对 Fourier frequencies 的正交性进行一些简单的推导.

对于 
$$\begin{aligned}
\mathrm{Z} = \begin{pmatrix} \frac{1}{\sqrt{2}} & \mathrm c_1(1) & \mathrm s_1(1) & \cdots & \mathrm c_{n/2}(1)  \\ \frac{1}{\sqrt{2}} & \mathrm c_1(2) & \mathrm s_1(2) & \cdots & \mathrm c_{n/2}(2)  \\ \vdots & \vdots  & \ddots & \vdots & \vdots \\ \frac{1}{\sqrt{2}} & \mathrm c_1(n) & \mathrm s_1(n) & \cdots & \mathrm c_{n/2}(n)  \end{pmatrix}
\end{aligned}$$
我们有:
$$\begin{aligned}
\mathrm{Z}^\top \mathrm{Z} &= \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & \cdots & \frac{1}{\sqrt{2}} \\ \mathrm c_1(1) & \mathrm c_1(2) & \cdots & \mathrm c_1(n) \\ \mathrm s_1(1) & \mathrm s_1(2) & \cdots & \mathrm s_1(n) \\ \vdots & \vdots & \ddots & \vdots \\ \mathrm c_{n/2}(1) & \mathrm c_{n/2}(2) & \cdots & \mathrm c_{n/2}(n)  \end{pmatrix} \begin{pmatrix} \frac{1}{\sqrt{2}} & \mathrm c_1(1) & \mathrm s_1(1) & \cdots & \mathrm c_{n/2}(1)  \\ \frac{1}{\sqrt{2}} & \mathrm c_1(2) & \mathrm s_1(2) & \cdots & \mathrm c_{n/2}(2)  \\ \vdots & \vdots  & \ddots & \vdots & \vdots \\ \frac{1}{\sqrt{2}} & \mathrm c_1(n) & \mathrm s_1(n) & \cdots & \mathrm c_{n/2}(n)  \end{pmatrix}\\
&= \begin{pmatrix} \frac{n}{2} &  \frac{1}{\sqrt{2}}\sum_{t=1}^n \mathrm c_1(t) & \frac{1}{\sqrt{2}}\sum_{t=1}^n \mathrm s_1(t) & \cdots & \frac{1}{\sqrt{2}}\sum_{t=1}^n \mathrm c_{n/2}(t)  \\ \frac{1}{\sqrt{2}}\sum_{t=1}^n \mathrm c_1(t) & \sum_{t=1}^n \mathrm c_1^2(t) & \sum_{t=1}^n \mathrm c_1(t)\mathrm s_1(t) & \cdots & \sum_{t=1}^n \mathrm c_1(t)\mathrm c_{n/2}(t)  \\ \frac{1}{\sqrt{2}}\sum_{t=1}^n \mathrm s_1(t) & \sum_{t=1}^n \mathrm c_1(t)\mathrm s_1(t) & \sum_{t=1}^n \mathrm s_1^2(t) & \cdots & \sum_{t=1}^n \mathrm s_1(t)\mathrm c_{n/2}(t)  \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \frac{1}{\sqrt{2}}\sum_{t=1}^n \mathrm c_{n/2}(t) & \sum_{t=1}^n \mathrm c_1(t)\mathrm c_{n/2}(t) & \sum_{t=1}^n \mathrm s_1(t)\mathrm c_{n/2}(t) & \cdots & \sum_{t=1}^n \mathrm c_{n/2}^2(t)  \end{pmatrix}
\end{aligned}$$

- 对于第一行/第一列除了$\frac{n}{2}$ 之外 (细节可由欧拉公式或纯粹初等数学中的三角恒等变换具体得证):
    $$\begin{aligned}
    \frac{1}{\sqrt{2}}\sum_{t=1}^n \mathrm c_1(t) &= \frac{1}{\sqrt{2}}\sum_{t=1}^n \cos(2\pi \frac{1}{n} t)  = 0\\
    \frac{1}{\sqrt{2}}\sum_{t=1}^n \mathrm s_1(t) &= \frac{1}{\sqrt{2}}\sum_{t=1}^n \sin(2\pi \frac{1}{n} t)  = 0
    \end{aligned}$$

-  对于除了$\frac{n}{2}$ 之外的其余对角线元素:
    $$\begin{aligned}
    \sum_{t=1}^n \mathrm c_j^2(t) &= \sum_{t=1}^n \cos^2(2\pi \frac{j}{n} t) = \frac{n}{2}\\
    \sum_{t=1}^n \mathrm s_j^2(t) &= \sum_{t=1}^n \sin^2(2\pi \frac{j}{n} t) = \frac{n}{2}\end{aligned}$$

- 对于除了对角线外的其余元素:
    $$\begin{aligned}
    \sum_{t=1}^n \mathrm c_j(t) \mathrm s_j(t) &= \sum_{t=1}^n \cos(2\pi \frac{j}{n} t) \sin(2\pi \frac{j}{n} t) = 0\\
    \sum_{t=1}^n \mathrm c_j(t) \mathrm c_k(t) &= \sum_{t=1}^n \cos(2\pi \frac{j}{n} t) \cos(2\pi \frac{k}{n} t) = 0\\
    \sum_{t=1}^n \mathrm s_j(t) \mathrm s_k(t) &= \sum_{t=1}^n \sin(2\pi \frac{j}{n} t) \sin(2\pi \frac{k}{n} t) = 0
    \end{aligned}$$

综上所述, 最终我们得到了 $\mathrm Z^\top \mathrm Z = \begin{pmatrix} \frac{n}{2} & 0 & 0 & \cdots & 0 \\ 0 & \frac{n}{2} & 0 & \cdots & 0 \\ 0 & 0 & \frac{n}{2} & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & \frac{n}{2}  \end{pmatrix} = \frac{n}{2} \mathrm I$.