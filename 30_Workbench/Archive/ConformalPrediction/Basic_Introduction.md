# Conformal Prediction

> Ref: Advanced Topics in Statistical Learning, Spring 2023, Ryan Tibshirani, UCB
>
> 注: 本文档中的数学编号和公式均参照 Ryan 的讲义, 以便于对照阅读. 部分公式由于排版整理等原因, 因此会出现编号不连续的情况.

## Introduction

### A Lofty Goal
Conformal prediction, 中文常称为“保形预测”，是一种类似于区间估计/区间预测的统计方法. 这里首先尝试给出该任务的定义和目标.

- 从经典的有监督学习任务出发. 假设有输入-输出对 $(X_i, Y_i)\stackrel{\text{i.i.d.}}{\sim} P$, $\forall i=1, \ldots, n$, 其中 $P$ 是 $\mathcal{X} \times \mathcal{Y}$ 上的联合分布, $\mathcal{X}$ 是输入空间 (如 $\mathcal{X} = \mathbb{R}^d$), $\mathcal{Y}$ 是输出空间 (如 $\mathcal{Y} = \mathbb{R}$). 

- 和区间估计类似, 我们最终希望能给出一个预测区间 (称为**预测带**, prediction band) $\widehat{C}_n: \mathcal{X} \to \{\text{subsets of } \mathcal{Y}\}$. 该预测带保证, 对于一个新的测试点 $(X_{n+1}, Y_{n+1}) \sim P$ (独立于训练数据), 输入特征 $X_{n+1}$ 的条件下, 预测带 $\widehat{C}_n(X_{n+1})$ 将以至少 $1-\alpha$ (给定) 的概率包含真实输出 $Y_{n+1}$, 即
$$\mathbb{P}\{Y_{n+1} \in \widehat{C}_n(X_{n+1}) \} \geq 1 - \alpha \quad (1)$$

- 这个估计的主要特点在于我们不对数据分布 $P$ 做任何假设, 也不进行渐进分析, 而是直接在有限样本下给出一个非渐进的概率保证. 

当然, 我们可以构造平凡预测:
$$\widehat{C}_n(x) = \begin{cases}
\mathcal{Y}, & \text{with probability } 1-\alpha \\
\emptyset, & \text{with probability } \alpha
\end{cases}$$
即随机以 $1-\alpha$ 的概率输出整个输出空间 $\mathcal{Y}$, 以 $\alpha$ 的概率输出空集. 不过我们希望能够找到非平凡的预测带, 即 $\widehat{C}_n(x)$ 能够尽可能小 (即精确) (在某种意义下).

### This is Achievable!

该目标的一个实现思路是, 从一个典型的 point predictor $\widehat{f}_n$ (如回归, 决策树等) 出发, 尝试根据此扩展出一个非平凡的 set predictor $\widehat{C}_n$. 

为进一步说明, 考虑一个更为退化的场景. 
- 退化掉输入 $X$, , 我们只掌握输出 $Y_i \in \mathbb{R}$, $i=1, \ldots, n$. 并且我们给出的 prediction band 就是一个形如 $\widehat{C}_n = (-\infty, \hat{q}_n]$ 的区间, 使得 
$$\mathbb{P}(Y_{n+1} \leq \hat{q}_n) \geq 1-\alpha \quad (3).$$
这里 $\hat{q}_n$ 是一个基于 $Y_1, \ldots, Y_n$ 的统计量. 

- 一个很直觉的做法是, 直接使用经验分布的分位数. 粗略讲即寻找样本中$(1-\alpha)\times 100\%$位置的观测值作为 $\hat{q}_n$. 严格地, 记 $\delta_{Y_i}$ 为 $Y_i$ 的 point mass, 则经验分布即为 $\frac{1}{n} \sum_{i=1}^n \delta_{Y_i}$. 故上述分位数可以定义为
$$\hat{q}_n = \text{Quantile}\left(1-\alpha; \frac{1}{n} \sum_{i=1}^n \delta_{Y_i}\right) $$
- 不过显然, 该分位数确定的预测带只能提供近似的概率保证 $\mathbb{P}(Y_{n+1} \leq \hat{q}_n) \approx 1-\alpha$. 其只能在渐进意义下满足要求. 

#### Key Idea 1: Use Rank to Adjust Quantiles

为了在有限样本下解决这一问题, 我们尝试引入 rank 的信息来对分位数进行调整. 

- 具体地, 由于 $Y_{n+1}$ 独立同分布于 $Y_1, \ldots, Y_n$, 因此其排名在 $Y_1, \ldots, Y_n, Y_{n+1}$ 中是均匀分布的. 换言之, $Y_{n+1}$ 在 $Y_1, \ldots, Y_n, Y_{n+1}$ 中排第 $\lceil (1-\alpha)(n+1) \rceil$ 小位置的概率至少为 $1-\alpha$: 
$$\mathbb{P}\{ Y_{n+1} \text{ is the } \lceil (1-\alpha)(n+1) \rceil\text{ smallest among } Y_1,\cdots,Y_n, Y_{n+1} \} \geq 1-\alpha. \quad(4)$$
而这个陈述等价于, $Y_{n+1}$ 在 $Y_1, \ldots, Y_n$ 中排第 $\lceil (1-\alpha)(n+1) \rceil$ 小位置的概率至少为 $1-\alpha$, 即
$$\mathbb{P}\{ Y_{n+1} \text{ is the } \lceil (1-\alpha)(n+1) \rceil\text{ smallest among } Y_1,\cdots,Y_n \} \geq 1-\alpha \quad (5).$$

- 尽管只有一个观测的差别, 然而后面的这个陈述可以立刻根据此定义一个新的统计量:
$$\widehat{q}_n = \lceil (1-\alpha)(n+1) \rceil\text{-th smallest among } Y_1,\cdots,Y_n \quad (6)$$
即等价地,
$$\widehat{q}_n = \text{Quantile}\left(\frac{\lceil (1-\alpha)(n+1) \rceil}{n}; \frac{1}{n} \sum_{i=1}^n \delta_{Y_i}\right) \quad(7).$$

  - 这相当于给我们先前提出的分位数选择方法做了一个向上的调整. 这是我们在有限样本下的修正, 补充了朴素的经验分位数方法在有限样本下的不足.

#### Exchangeability Is All You Need

- 考虑有序数列 $(Y_1, \ldots, Y_n, Y_{n+1})$, 定义 permeation $\sigma: \{1, \ldots, n+1\} \to \{1, \ldots, n+1\}$ 是一个对 $n+1$ 个元素的重新排列. 则定义 exchangeable (可交换) 为对任意的 permutation $\sigma$, 有
$$(Y_1, \ldots, Y_n, Y_{n+1}) \stackrel{d}{=} (Y_{\sigma(1)}, \ldots, Y_{\sigma(n)}, Y_{\sigma(n+1)}).$$
也就是说, 交换这些变量的顺序并不会改变它们的联合分布.

- Exchangeability 是 i.i.d. 的一个弱化版本. 事实上, 如果 $Y_1, \ldots, Y_{n+1}$ 是 i.i.d., 则它们必然是 exchangeable 的. 但反之则不必然. 

- 后续我们将利用 exchangeability 来推广上述的思路.

#### Coverage Upper Bound When There are No Ties

- 事实上在 $Y_1, \cdots, Y_{n+1}$ 没有相等的观测值 (ties) 或通过合理的随机化消除 ties 的情况下 $(4)$ 可进一步被加强为等式:
$$\mathbb{P}\{ Y_{n+1} \text{ is the } \lceil (1-\alpha)(n+1) \rceil\text{ smallest among } Y_1,\cdots,Y_n, Y_{n+1} \} = \frac{\lceil (1-\alpha)(n+1) \rceil}{n+1} \quad(8).$$

- 对 $(8)$ RHS 进行放缩, 根据 $\frac{(1-\alpha)(n+1)}{n+1} \leq \frac{\lceil (1-\alpha)(n+1) \rceil}{n+1} \leq \frac{(1-\alpha)(n+1) + 1}{n+1} $, 可得
$$(1-\alpha) \leq \mathbb{P}\{ Y_{n+1}\text{ is ... among } Y_1,\cdots,Y_{n+1} \} < (1-\alpha) + \frac{1}{n+1}.$$

- 此外, 我们已经说明 $\mathbb{P}\{ Y_{n+1}\text{ is ... among } Y_1,\cdots,Y_{n+1} \} \equiv \mathbb{P}\{ Y_{n+1} \leq \widehat{q}_n \}$, 因此
$$\mathbb{P}(Y_{n+1} \leq \widehat{q}_n) \in \left[1-\alpha, 1-\alpha + \frac{1}{n+1}\right) \quad(10).$$
  - 该不等式中, LHS 是恒成立之情况, RHS 是我们通过 almost surely (a.s.) no ties 的假设来保证的.

#### Naive Attempt to Lift This Idea to Regression

我们尝试将上述利用 rank 的思路推广到回归问题中. 具体地, 给定 $X_i \in \mathcal{X}$, $Y_i \in \mathbb{R}$, $i=1, \ldots, n$, 以及一个新的测试点 $X_{n+1}, Y_{n+1}$. 我们希望构造一个基于 $X_{n+1}$ 的 prediction set. 此外, 假设我们通过某种方法通过训练 $\{(X_i, Y_i)\}_{i=1}^n$ 得到一个 point predictor $\widehat{f}_n(x)$.  

据此, 一个朴素的构造方法如下. 
- 首先得到估计的绝对值残差 (absolute residuals):
    $$R_i = |Y_i - \widehat{f}_n(X_i)|, \quad i=1, \ldots, n.$$
- 然后基于 $(6)$ 的思路, 计算残差的 $\lceil (1-\alpha)(n+1) \rceil$ 分位数:
    $$\widehat{q}_n = \lceil (1-\alpha)(n+1) \rceil\text{-th smallest among } R_1,\cdots,R_n.$$
- 最后构造预测带 $\widehat{C}_n(X) = \{Y: |Y - \widehat{f}_n(X)| \leq \widehat{q}_n\}$, 即
    $$\widehat{C}_n(X) = [\widehat{f}_n(X) - \widehat{q}_n, \widehat{f}_n(X) + \widehat{q}_n].$$
    以期望达到 $(1)$ 的覆盖率要求

然而, 该构造方法是有问题的: 
- 根据 conformal prediction 的定义, 我们以 $(1-\alpha)$ 的概率希望保证 $Y_{n+1} \in \widehat{C}_n(X_{n+1})$, 即 $|Y_{n+1} - \widehat{f}_n(X_{n+1})| \leq \widehat{q}_n$. 
- 这等价于 $R_{n+1} = |Y_{n+1} - \widehat{f}_n(X_{n+1})| \leq \widehat{q}_n$. 
- 而根据上述的构造, 这等价于 $R_{n+1}$ 在 $R_1, \ldots, R_n$ 中排名第 $\lceil (1-\alpha)(n+1) \rceil$ 小位置.

但最后一个陈述并不成立. 这是因为 $R_1, \ldots, R_n, R_{n+1}$ 并不是 exchangeable 的. 
- 直觉上理解, 由于我们在训练中并没有见过测试样本 $X_{n+1}$, 因此 $R_{n+1}$ 往往随机意义上大于 (stochastically larger than) 训练残差 $R_1, \ldots, R_n$: 对任意阈值 $t$, 有 $\mathbb P(R_{n+1}>t)\ \gtrsim\ \mathbb P(R_i>t)$, $\forall i=1, \ldots, n$. 这就破坏了 exchangeability.
- 一个例子是 OLS 中, 训练的残差之方差为 $\mathrm{Var}(r_i)=\sigma^2(1-h_{ii})\le\sigma^2$. 而测试点的残差之方差为 $\sigma^2(1+x_0^\top(X^\top X)^{-1}x_0)\ge\sigma^2$
- 这里也将这种差异导致的现象称为欠覆盖 **under-coverage**.

## Split Conformal Prediction

上述的朴素方法由于残差不具备 exchangeability 而无法保证覆盖率. 为了解决这个问题, 我们引入 **split conformal prediction** 方法.

#### Key Idea 2: Construct Scores Symmetrically

Split conformal prediction 的核心思路是, 通过对数据进行对称的处理来构造残差, 以保证残差的 exchangeability. 具体地,
- 我们将训练集 $\mathcal D_{\text{train}} = \{(X_i, Y_i)\}_{i=1}^n$ 随机划分为两部分: $\mathcal D_{\text{prop}}$ (proper training set) 和 $\mathcal D_{\text{cal}}$ (calibration set), 且有 $\mathcal D_{\text{prop}} \cap \mathcal D_{\text{cal}} = \emptyset$, $\mathcal D_{\text{prop}} \cup \mathcal D_{\text{cal}} = \mathcal D_{\text{train}}$ 以保证划分之准确. 且记 $|\mathcal D_{\text{prop}}| = n_{\text{prop}}$, $|\mathcal D_{\text{cal}}| = n_{\text{cal}}$, 则有 $n_{\text{prop}} + n_{\text{cal}} = n$. 
- 用 proper training set $\mathcal D_{\text{prop}}$ 训练 point predictor 得到 $\widehat{f}_{\text{prop}}$.
- 接着计算 $f_{\text{prop}}$ 在 calibration set $\mathcal D_{\text{cal}}$ 上的残差:
    $$R^{\text{cal}}_i = |Y_i^{\text{cal}} - \widehat{f}_{\text{prop}}(X^{\text{cal}}_i)|, \quad (X^{\text{cal}}_i, Y^{\text{cal}}_i) \in \mathcal D_{\text{cal}}.$$
- 以及基于同样的思路, 在 calibration 上计算 conformal quantile:
    $$\widehat{q}_{\text{cal}} = \lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil\text{-th smallest among } R^{\text{cal}}_i, (X^{\text{cal}}_i, Y^{\text{cal}}_i) \in \mathcal D_{\text{cal}}.$$
    - 若记 $k_\alpha = \lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil$, 则 $\widehat{q}_{\text{cal}}$ 也可表示为 $R^{\text{cal}}_{(k_\alpha)}$.
- 最后构造预测带 $\widehat{C}_n(X^{\text{test}}) = \{Y: |Y - \widehat{f}_{\text{prop}}(X^{\text{test}})| \leq \widehat{q}_{\text{cal}}\}$, 即
    $$\widehat{C}_n(X^{\text{test}}) = [\widehat{f}_{\text{prop}}(X^{\text{test}}) - \widehat{q}_{\text{cal}}, \widehat{f}_{\text{prop}}(X^{\text{test}}) + \widehat{q}_{\text{cal}}]. \quad(11)$$
    - 也可以理解为 $\widehat{C}_n(X^{\text{test}}) = \{Y: R^{\text{test}} \leq R^{\text{cal}}_{(k_\alpha)}\}$, 其中 $R^{\text{test}} = |Y - \widehat{f}_{\text{prop}}(X^{\text{test}})|$.
- 我们能保证有概率:
    $$\mathbb{P}\left( Y^{\text{test}} \in \widehat{C}_n(X^{\text{test}}) \right) \in \left[1-\alpha, 1-\alpha + \frac{1}{n_{\text{cal}} + 1}\right). \quad(12)$$
    - 其中下界是恒成立的, 上界是基于 calibration set 上 a.s. no ties 的假设. 而该假设只要我们通过合理的设置 (condition on) proper training set, 便可使 calibration set $\mathcal D_{\text{cal}}$ 对应的残差 $R_i, (X_i, Y_i) \in \mathcal D_{\text{cal}}$ 与测试点的残差 $R_{n+1}$ 之间不是 i.i.d. 的, 故不存在 ties 的问题.

#### Any Score Function Will Work

事实上, 上述的 split conformal prediction 方法并不局限于使用绝对值残差 $R^{\text{cal}}_i = |Y_i^{\text{cal}} - \widehat{f}_{\text{prop}}(X^{\text{cal}}_i)|$ 来构造分数 (score), 例如标准化残差 $V(X_i^{\text{cal}}, Y_i^{\text{cal}}) = \frac{|Y_i^{\text{cal}} - \widehat{f}(X_i^{\text{cal}})|}{\widehat{\sigma}(X_i^{\text{cal}})}$ 等. 故下给出一般化的 score function 及其对应的 split conformal prediction 方法.
- 定义 score function $R_i^{\text{cal}} = V(X_i^{\text{cal}}, Y_i^{\text{cal}})$, 其中 $(X_i^{\text{cal}}, Y_i^{\text{cal}}) \in \mathcal D_{\text{cal}}$.
- 则基于该 score function 给出的 conformal 集合为
    $$\widehat{C}_n(X) = \{Y: V(X, Y) \leq \widehat{q}_{\text{cal}}\}.$$
    - 其中 $\widehat{q}_{\text{cal}}$ 是 $\{R_i^{\text{cal}}: (X_i^{\text{cal}}, Y_i^{\text{cal}}) \in \mathcal D_{\text{cal}}\}$ 的 $\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil$ 分位数 (第 $\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil$ 小的值).
- 由于残差具体的定义并不会影响其 exchangeability, 因此 $(12)$ 仍然成立.


#### Quantile and CDF Formulations

这里对 split conformal prediction 的分位数定义进行等价的变形, 以便于后续的推导等. 

回顾, 我们目前的 prediction set 定义为:
$$\widehat{C}_n(X^{\text{test}}) = \{Y: V(X^{\text{test}}, Y) \leq \widehat{q}_{\text{cal}}\},$$
  其中 $\widehat{q}_{\text{cal}}$ 是 $\{R_i^{\text{cal}}: (X_i^{\text{cal}}, Y_i^{\text{cal}}) \in \mathcal D_{\text{cal}}\}$ 的 $\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil$ 分位数 (第 $\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil$ 小的值).
- 对 quantile 的定义进行变形, 可得
$$\widehat{C}_n(X^{\text{test}})  = \left\{ Y: V(X^{\text{test}},Y) \leq \text{Quantile} \left(\frac{\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil}{n_{\text{cal}}}; \frac{1}{n_{\text{cal}}} \sum_{(X_i^{\text{cal}}, Y_i^{\text{cal}}) \in \mathcal D_{\text{cal}}} \delta_{R_i^{\text{cal}}} \right)\right\}.$$
- 进一步, 用 CDF 的形式, 可得
  $$\widehat{C}_n(X^{\text{test}}) = \left\{ Y: \frac{1}{n_{\text{cal}}} \sum_{(X_i^{\text{cal}}, Y_i^{\text{cal}}) \in \mathcal D_{\text{cal}}} \boldsymbol{1}\{R_i^{\text{cal}} < V(X^{\text{test}},Y)\} \leq \frac{\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil}{n_{\text{cal}}} \right\}. \quad (15)$$
    - 进一步明确其含义. $R_i^{\text{cal}}$ 表示 calibration set 上的数据得到的 (广义) 残差得分. $V(X^{\text{test}}, Y)$ 表示测试点 $(X^{\text{test}}, Y)$ 的 (广义) 残差得分. 故 LHS 表示当我们给定一种残差得分后, 测试点的残差得分在 calibration set 上的残差得分中排名占比. 
    - 而 RHS 是我们通过 rank 的方式对分位数进行的调整. 表示在给定的置信水平 $1-\alpha$ 下, 测试点的残差得分在 calibration set 上的残差得分中排名占比的上界.
    - 该不等式整体的含义是, 当给定了置信水平 $1-\alpha$ 后, 我们通过确定 calibration set 就可以确定一个测试点的分位数上界. 根据这个上界, 我们便能推出能够容忍的残差得分的范围, 也就确定了我们预测值能够接受的范围, 即预测带.
> Note: 这里给出 CDF 形式的详细推导. 回忆 CDF 的定义, 对于 calibration residuals $R_i^{\text{cal}}$, 我们有
> $$\widehat{F}_{R^{\text{cal}}}(t) = \frac{1}{n_{\text{cal}}} \sum_{(X_i^{\text{cal}}, Y_i^{\text{cal}}) \in \mathcal D_{\text{cal}}} \boldsymbol{1}\{R_i^{\text{cal}} < t\}.$$
> 又根据 CDF 和 quantile 的关系, 有
> $$\text{Quantile}(p; \widehat{F}_{R^{\text{cal}}}) = \inf\{t: \widehat{F}_{R^{\text{cal}}}(t) \geq p\}.$$
>   - 其中 $p \in [0, 1]$ 是分位数, 用 rank 的形式可表示为 $p = \frac{k}{n_{\text{cal}}}$, $k=1, \ldots, n_{\text{cal}}$.
>
> 我们已有的条件是: $V(X^{\text{test}}, Y) \leq \hat{q}_{\text{cal}}$, 故左右同取 CDF, 可得
> $$\widehat{F}_{R^{\text{cal}}}(V(X^{\text{test}}, Y)) \leq \widehat{F}_{R^{\text{cal}}} (\hat{q}_{\text{cal}}) = \frac{\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil}{n_{\text{cal}}}.$$
>  而左边即为 CDF 的定义, 故可得
> $$\frac{1}{n_{\text{cal}}} \sum_{(X_i^{\text{cal}}, Y_i^{\text{cal}}) \in \mathcal D_{\text{cal}}} \boldsymbol{1}\{R_i^{\text{cal}} \leq V(X^{\text{test}},Y)\} \leq \frac{\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil}{n_{\text{cal}}}.$$

#### Auxiliary Randomization

前面提到, 我们对 split conformal prediction 的覆盖保证为:
$$\mathbb{P}\left( Y^{\text{test}} \in \widehat{C}_n(X^{\text{test}}) \right) \in \left[1-\alpha, 1-\alpha + \frac{1}{n_{\text{cal}} + 1}\right). \quad(12)$$ 而事实上, 我们总可以通过引入辅助的随机化 (auxiliary randomization) 来使得上界等于下界, 即收紧到 $1-\alpha$. 

为了说明这一点, 我们需要进一步理解 $(15)$ 的含义:
  $$\widehat{C}_n(X^{\text{test}}) = \left\{ Y: \frac{1}{n_{\text{cal}}} \sum_{(X_i^{\text{cal}}, Y_i^{\text{cal}}) \in \mathcal D_{\text{cal}}} \boldsymbol{1}\{R_i^{\text{cal}} < V(X^{\text{test}},Y)\} \leq \frac{\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil}{n_{\text{cal}}} \right\}. \quad (15)$$
- 对于 LHS, $\frac{1}{n_{\text{cal}}} \sum_{\mathcal D_{\text{cal}}} \boldsymbol{1}\{R_i^{\text{cal}} < V(X^{\text{test}},Y)\}$ 表示在 calibration set 上, 有多少比例的 score 比测试点的 score 要小. 数学严格意义上说, 这是一个 CDF 的左连续版本 (因为是 $<$ 而不是 $\leq$), 因此严谨地说应当记作 $\widehat{F}^-_{R^{\text{cal}}}(\cdot)$. 
- 对于 RHS, $\frac{\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil}{n_{\text{cal}}}$ 是一个经过矫正的分位数阈值. 
- 我们之所以在 $(12)$ 中只能保证一个区间, 恰恰就是因为这个 CDF 是一个阶梯函数, 每多一个 calibration point, CDF 就会跳跃一次, 且其取值恰为 $\frac{1}{n_{\text{cal}}}$. 

为解决这一问题, 我们进一步将测试分数 $R^{\text{test}} := R_{n_{\text{cal}}+1} = V(X^{\text{test}}, Y^{\text{test}})$ 加入进来, 一共得到 $n_2 +1$ 个分数, 对这些分数一起进行排序:
 $$\small{\widehat{C}_n(X^{\text{test}}) = \left\{ Y: \frac{\boldsymbol{1}\{V(X^{\text{test}},Y) < R^{\text{test}}\}+\sum_{ \mathcal D_{\text{cal}}} \boldsymbol{1}\{R_i^{\text{cal}} < V(X^{\text{test}},Y)\}}{1+n_{\text{cal}}}  \leq \frac{\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil}{n_{\text{cal}}+1} \right\}}$$

对这 $n_{\text{cal}} + 1$ 个分数可以同理定义 CDF:
$$\hat F^-_{n_{\text{cal}} + 1}(t) = \frac{1}{n_{\text{cal}}+1}\sum_{i=1}^{n_{\text{cal}}+1} \mathbf 1\{R_i < t\}$$
- 若代入 $t = R^{\text{test}}$, 则有:
    $$\hat F^-_{n_{\text{cal}} + 1}(R^{\text{test}}) = \frac{\boldsymbol{1}\{V(X^{\text{test}},Y^{\text{test}}) < R^{\text{test}}\}+\sum_{ \mathcal D_{\text{cal}}} \boldsymbol{1}\{R_i^{\text{cal}} < V(X^{\text{test}},Y^{\text{test}})\}}{1+n_{\text{cal}}}.$$

因此有如下等价关系至少以 $1-\alpha$ 的概率成立:
    $$Y^{\text{test}} \in \widehat{C}_n(X^{\text{test}}) \iff \hat F^-_{n_{\text{cal}} + 1}(R^{\text{test}}) \leq \frac{\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil}{n_{\text{cal}}+1}$$


**Lemma (随机化 CDF)** : 一般地, 对任意随机变量 $Z$, 记右连续 CDF (默认) 为 $F_Z(z) = \mathbb P(Z \leq z)$,  左连续 CDF 为 $F^-_Z(z) = \mathbb P(Z < z) \equiv \sup_{\zeta < z} F_Z(\zeta) \equiv \lim_{\zeta \uparrow z} F_Z(\zeta)$. 则对任意与 $Z$ 独立的 $U \sim \text{Uniform}(0, 1)$, 定义随机化 CDF 为
$$\widetilde{F}_Z(z) = F^-_Z(z) + U(F_Z(z) - F^-_Z(z)).$$
则有性质 $\mathbb P\big(F^{*}(Z)\le t\big)=t$ 对任意 $t \in [0, 1]$ 成立, 即 $\widetilde{F}_Z(Z) \sim \text{Uniform}(0, 1)$.
- 具体证明略. 一个直观的说明是, 当 $Z=z$ 恰好落在 CDF 的跳跃点时, 由于 $U$ 的引入, 使得 $\widetilde{F}_Z(z)$ 在 $[F^-_Z(z), F_Z(z)]$ 上均匀分布. 这样便消除了 CDF 跳跃带来的影响.

最终得到的随机化的 split conformal prediction 方法为:
$$\hat C_n^{*}(x)=\Big\{y:\frac{1}{n_{\text{cal}}+1}\sum_{\mathcal{D}_{\text{cal}}}{\boldsymbol{1}}\{R_i<V(X^{\text{test}},Y^{\text{test}})\}+\frac{U}{n_{\text{cal}}+1}\Big(\sum_{i\in D_2}{\boldsymbol 1}\{R_i=V(X^{\text{test}},Y^{\text{test}})\}+1\Big)\ \le\ 1-\alpha\Big\},$$

### Remarks

首先有几点总结与说明:

1. 朴素的 rank 估计 (不进行 split 时) 由于残差不具备 exchangeability 而无法保证覆盖率, 这被称为欠覆盖 (under-coverage). 尤其当点估计出现过拟合时, 该问题尤为严重, 因为此时训练残差往往过小而测试残差过大. 通过 split conformal prediction 方法, 我们将训练数据分为两部分, 一部分用于训练点估计, 另一部分用于计算残差并保证其 exchangeability. 这样便能保证覆盖率.即使在过拟合的情况下, 该方法仍然有效.

2. 在进行 split conformal prediction 时, 传统的 absolute residuals $R_i = |Y_i - \widehat{f}(X_i)|$ 并不是唯一的选择, 也并不是一个很好的选择. 其得到的 $\hat C_n(x)=\{y:\ |y-\hat f(x)|\le \hat q\}=[\,\hat f(x)-\hat q,\ \hat f(x)+\hat q\,]$ 中的 $\hat q$ 是一个全局的常数, 不能反映不同 $X$ 位置的差异 (如异方差). 

3. Conformal prediction 本身对于预测的算法并无偏好. 但是预测的 prediction band $\hat C_n(x)$ 的宽度与 $\hat{f}_{\text{prop}}$ 的拟合质量有关, 即 $\hat{f}_{\text{prop}}$ 越准确, 则 $\hat C_n(x)$ 越窄, 估计的效率越高. 

### Example: Split Conformal with  Smoothing Spline

下图展示了一个 split conformal prediction 的例子. 该例子中, 我们使用 smoothing spline (5 degrees of freedom) 作为点估计 $\hat f_{\text{prop}}$. 

具体地: 
- 将数据一分为二, 一半作为 proper training set (即图中黑色点), 另一半作为 calibration set (即图中蓝色点). 
- 通过 proper training set 训练 smoothing spline 得到 $\hat f_{\text{prop}}$ (即图中黑色曲线). 
- 计算 calibration set 上的残差 (这里采用绝对值残差): $R_i^{\text{cal}} = |Y_i^{\text{cal}} - \widehat{f}_{\text{prop}}(X^{\text{cal}}_i)|$.
- 取得残差的 $\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil$ 分位数 $\hat q_{\text{cal}}$ (这里 $\alpha=0.1$). 得到预测带 $\hat C_n(x) = [\hat f_{\text{prop}}(x) - \hat q_{\text{cal}}, \hat f_{\text{prop}}(x) + \hat q_{\text{cal}}]$ (即图中黄色带).

![Example of split conformal prediction, based on a smoothing spline with 5 degrees of freedom.](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250910053934.png)

对拟合结果有几点观察/说明:
- 只要测试点来自与训练数据相同的分布，测试覆盖率至少 为 $1-\alpha=90\%$. 这与点估计的质量, 数据的分布无关, 且在有限样本下给出的保证. 
- 用绝对残差作分数会得到严格常宽的带 (对所有 $x$ 的半宽都是 $\hat q_{\text{cal}}$), 这反映在图中黄色带的宽度是恒定的. 
- 仔细观察, 发现数据的分布并非均匀. 数据在 $x$  的左侧噪声较小, 右侧噪声较大. 但由于我们使用了绝对残差, 因此无法反映这种异方差的现象. 因此出现左侧过覆盖 (over-coverage), 右侧欠覆盖 (under-coverage) 的现象. 这说明绝对残差并不是一个很好的选择.

### Conditional Coverage Properties?

这里再对这里的随机性/条件概率等进行一些讨论.

回顾我们的作法. 
- 我们将训练数据 $\mathcal D_{\text{train}}$ 随机划分为 proper training set $\mathcal D_{\text{prop}}$ 和 calibration set $\mathcal D_{\text{cal}}$. 
- 当我们给定了 proper training set $\mathcal D_{\text{prop}}$ 后, 也就给定了点预测 $\hat f_{\text{prop}}$. 
- 接着我们基于 calibration set $\mathcal D_{\text{cal}}$ 计算残差 $R_i^{\text{cal}} = |Y_i^{\text{cal}} - \widehat{f}_{\text{prop}}(X^{\text{cal}}_i)|$, 以及分位数 $\hat q_{\text{cal}}$. 根据 calibration score $R_i^{\text{cal}}$ 和测试 score $R^{\text{test}}$ 的 exchangeability, 我们保证了
$$\mathbb{P}\left( Y^{\text{test}} \in \hat{C}_n(X^{\text{test}}) \mid \mathcal D_{\text{prop}} \right) \in \left[1-\alpha, 1-\alpha + \frac{1}{n_{\text{cal}} + 1}\right).$$

对于上式的条件概率, 尝试通过积分的方式去掉对 $\mathcal D_{\text{prop}}$ 的条件, 可得:
$$\mathbb{P}\left( Y^{\text{test}} \in \hat{C}_n(X^{\text{test}}) \right) = \mathbb{E}_{\mathcal D_{\text{prop}}} \left[ \mathbb{P}\left( Y^{\text{test}} \in \hat{C}_n(X^{\text{test}}) \mid \mathcal D_{\text{prop}} \right) \right] \in \left[1-\alpha, 1-\alpha + \frac{1}{n_{\text{cal}} + 1}\right).$$
- 这是符合直觉的. 因为上面的条件概率对任意 $\mathcal D_{\text{prop}}$ 都成立, 因此对 $\mathcal D_{\text{prop}}$ 取期望也必然成立.

如果我们进一步将 $\mathcal D_{\text{prop}}, \mathcal D_{\text{cal}}$ 都给定, 只考虑测试点的随机性, 则覆盖概率
$$\mathbb{P}\left( Y^{\text{test}} \in \hat{C}_n(X^{\text{test}}) \mid \mathcal D_{\text{prop}}, \mathcal D_{\text{cal}} \right)\sim \text{Beta}(k_\alpha, n_{\text{cal}}+1-k_\alpha), ~k_\alpha = \lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil$$
- 这是因为
    - 回忆 $Y^{\text{test}} \in \hat{C}_n(X^{\text{test}}) \iff R^{\text{test}} \leq R^{\text{cal}}_{(k_\alpha)}$, 其中 $R^{\text{cal}}_{(k_\alpha)}$ 是 calibration set 上的第 $k_\alpha$ 小的残差. 
    - 当 $\mathcal D_{\text{prop}}, \mathcal D_{\text{cal}}$ 都给定后, $R^{\text{cal}}_{(k_\alpha)}$ 也是一个常数. 故 :
    $$\mathbb{P}\left( Y^{\text{test}} \in \hat{C}_n(X^{\text{test}}) \mid \mathcal D_{\text{prop}}, \mathcal D_{\text{cal}} \right) = \mathbb{P}(R^{\text{test}} \leq R^{\text{cal}}_{(k_\alpha)}) = F_{R^{\text{test}}}(R^{\text{cal}}_{(k_\alpha)})\sim \text{Uni}(0, 1)$$
      -  其中 $F(\cdot)$ 是 $R^{\text{test}}$ 的 CDF. 
   -  因此 $F(R^{\text{cal}}_{(k_\alpha)})$ 是均匀分布的 $k_\alpha$ 顺序统计量, 其分布为 $\text{Beta}(k_\alpha, n_{\text{cal}}+1-k_\alpha)$.
-  根据 beta 分布的性质, 其均值为 $\mathbb E[\pi]=\frac{k_\alpha}{n_2+1}
=\frac{\lceil(1-\alpha)(n_2+1)\rceil}{n_2+1},\mathrm{Var}[\pi]=\frac{k_\alpha(n_2+1-k_\alpha)}{(n_2+1)^2(n_2+2)}\approx \frac{\alpha(1-\alpha)}{n_2+2}$
    - 这说明, 当训练集总量 $n$ 固定时, $n_{\text{cal}}$ 越大, 则 $\mathrm{Var}[\pi]$ 越小, 即覆盖率越稳定. 但另一方面, $n_{\text{cal}}$ 越大, 则 proper training set 越小, 点估计 $\hat f_{\text{prop}}$ 的质量越差, 预测带 $\hat C_n(x)$ 越宽. 因此 $n_{\text{cal}}$ 的选择需要在稳定性和效率之间进行权衡.

## Full Conformal Prediction

上述的 split conformal prediction 方法通过对数据进行划分来保证残差的 exchangeability. Full conformal prediction 则类似另一个极端, 即不进行数据划分, 而是通过对每一个候选的测试值 $y$ 都进行一次训练来保证残差的 exchangeability. 这是一个计算开销非常大的方法, 但思路非常直接, 在一些特定的情况下也可以进行高效的计算.

Full conformal prediction 同样要对称地对数据进行处理. 给定训练数据 $\mathcal D_{\text{train}} = \{(X_i, Y_i)\}_{i=1}^n$ 和一个新的测试点 $X^{\text{test}} \in \mathcal X$. 我们理论上对每一个可行的测试值 $y^{\text{trial}} \in \mathbb R$ 都进行计算 (称之为 trial 或 query), 以判断 $y^{\text{trial}}$ 是否落在预测带 $\hat C_n(X^{\text{test}})$ 中. 具体地, 
- 对训练数据进行增广, 得到 $\mathcal D_{\text{aug}} = \mathcal D_{\text{train}} \cup (X^{\text{test}}, y^{\text{trial}})$ (共 $n+1$ 个样本). 并根据 $\mathcal D_{\text{aug}}$ 训练点估计 $\hat f_{\text{aug}}$.
- 计算 $\mathcal D_{\text{aug}}$ 上的残差 (同样可以是绝对残差或其他残差):
    $$R^{\text{aug}}_i = |Y_i^{\text{aug}} - \widehat{f}_{\text{aug}}(X^{\text{aug}}_i)|, \quad (X^{\text{aug}}_i, Y^{\text{aug}}_i) \in \mathcal D_{\text{aug}}, i=1, \ldots, n+1.$$
- 计算 $\mathcal D_{\text{aug}}$ 上的分位数 (即第 $k_\alpha = \lceil (1-\alpha)(n+1) \rceil$ 小的残差) $\widehat{q}_{\text{aug}} = R^{\text{aug}}_{(k_\alpha)}$, 便可以得到 prediction set:
    $$\widehat{C}_n(X^{\text{test}}) = \{y^{\text{trial}}: |y^{\text{trial}} - \widehat{f}_{\text{aug}}(X^{\text{test}})| \leq \widehat{q}_{\text{aug}}\}.$$
- 且有概率保证:
    $$\mathbb{P}\left( Y^{\text{test}} \in \widehat{C}_n(X^{\text{test}}) \right) \in \left[1-\alpha, 1-\alpha + \frac{1}{n + 1}\right).$$

当然在实践中, 对每一个 $y^{\text{trial}}$ 遍历是不可能的. 因此实际中只能在有限的网络中进行搜索; 因此常用于小规模等的场景中.

总结一下 full conformal prediction:
1. Full conformal prediction 并不局限于使用绝对残差. 我们可以推广到更一般的打分函数 $V$. 在 full conformal prediction 中, 我们对每个候选 $(X^{\text{test}}, y^{\text{trial}})$ 都将其和训练集合并, 重新训练一次模型, 然后计算残差 $R_i^{\text{aug}} = V(X_i^{\text{aug}}, Y_i^{\text{aug}})$, 以及分位数 $\widehat{q}_{\text{aug}}$. 这样便能保证残差的 exchangeability, 从而保证覆盖率.

2. 同样可以有如下 CDF 的形式:
   $$\begin{aligned}
\widehat{C}_n(X^{\text{test}}) &= \left\{ y^{\text{trial}}: R^{\text{test}} \leq \text{Quantile} \left(\frac{\lceil (1-\alpha)(n + 1) \rceil}{n + 1}; \frac{1}{n + 1} \sum_{i=1}^{n+1} \delta_{R_i^{\text{aug}}} \right)\right\} \quad (19)\\
&= \left\{ y^{\text{trial}}: \frac{1}{n} \sum_{i=1}^{n} \boldsymbol{1}\{R_i^{\text{aug}} < V(X^{\text{test}}, y^{\text{trial}})\} \leq \frac{\lceil (1-\alpha)(n + 1) \rceil}{n+1} \right\}\quad (20) \end{aligned}$$

3. 同样可以运用 auxiliary randomization 来收紧覆盖率的区间.

### Remarks

有几个额外的说明:

1. 上述关于 Split Conformal Prediction 的讨论, 也同样适用于 Full Conformal Prediction.
2. 由于提及的计算开销, Barber et al. (2021) 提出了介于 Split 和 Full 之间的方法, 类似于 Cross-Validation 的思路. 详见 *Predictive Inference with the Jackknife+, Barber et al. (2021)*.
3. 对式 $(20)$, 我们变换一下不等式方向, 经过整理可得:
    $$\widehat{C}_n(X^{\text{test}}) = \left\{ y^{\text{trial}}:\underbrace{ \frac{1}{n} \sum_{i=1}^{n+1} \boldsymbol{1}\{R_i^{\text{aug}} \ge V(X^{\text{test}}, y^{\text{trial}})\}}_{p(y)} \geq \frac{\lfloor \alpha(n + 1) \rfloor}{n} \right\}$$
    - 其中 $p(y)$ 是一个关于 $y$ 的函数, 可以理解为 p-value 阈值的形式. 
      - $p(y) = \frac1n \sum_{i=1}^n \boldsymbol{1}\{R_i^{\text{aug}} \ge R_i^{\text{test}}\}$ 本身最直接的含义是, 在当前 $n$ 个训练之残差中, 不小于测试点残差 $V(X^{\text{test}}, y)$ 的比例. 简单讲, 就是有多少比例的训练分数不优于测试分数.
      - 可以认为是对于假设 $H_0: Y^{\text{test}} = y^{\text{trial}}$ 的一个检验统计量.
      - 这是因为, 假设在 $H_0$ 下, trial 值 $y^{\text{trial}}$ 真的是测试点的真实值, 则增广后重新得到的 $n+1$ 个分数应当是 exchangeable 的. 因此测试点的分数在 $n+1$ 个分数中排名的均匀性 (uniformity of ranks) 是成立的. 因此. $p$ 越大, 说明测试点的分数很小, 因此 $y^{\text{trial}}$ 越有可能是测试点的真实值. 反之, $p$ 越小, 则 $y^{\text{trial}}$ 越不可能是测试点的真实值.

### Example: Full Conformal with  Smoothing Spline

与 split conformal prediction 中的数据相同. 这里选取一个固定点 $x^* = 4.75$ 处, 计算 full conformal prediction 的预测带 $\widehat{C}_n(x^*=4.75)$. Point estimate 仍然使用 smoothing spline (15 degrees of freedom). 具体计算过程如下:
1. 固定 $x^* = 4.75$ (即图中蓝色标记处).
2. 在对应位置选取一系列的 trial 值 $y^{\text{trial}}$ (即图中紫色竖线). 每次都将一个 $(x^*, y^{\text{trial}})$ 和训练数据 $\mathcal D_{\text{train}}$ 进行增广, 并基于增广后的数据重新训练 smoothing spline, 得到 $\hat f_{\text{aug}}$. 
3. 在当前增广下, 计算 $n+1$ 个残差分数, 记测试点的分数为 $R_{n+1}^{(x^*, y^{\text{trial}})}$, 再接着计算一下 $n$ 个训练残差中大于这个分数的比例 (图中为红色的直方图的横坐标, 即为对应的 $y^{\text{trial}}$ 计算得到的 $p$. ):
    $$p(y^{\text{trial}}) = \frac1n \sum_{i=1}^n \boldsymbol{1}\{R_i^{\text{aug}} \ge R_{n+1}^{(x^*, y^{\text{trial}})}\}.$$
4. 依次类推, 保留所有满足 $p(y^{\text{trial}}) \ge \frac{\lfloor \alpha(n + 1) \rfloor}{n}$ 的 trial 值, 即得到预测带 $\widehat{C}_n(x^*)$.

![Example of full conformal prediction, where the prediction algorithm is a smoothing spline with 15 degrees of freedom.](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250910115607.png)




### Imposibility of X-conditional Coverage

需要指出, 在不对数据分布 $P$ 进行任何假设的情况下, 任何给出预测带 $\hat{C}_n$ 的方法 (不局限于 split 或 full) 方法若能够保证 $X$-conditional coverage, 则该方法必然是无效的 (trivial). 这里的 $X$-conditional coverage 指的是:
$$\mathbb{P}\left( Y^{\text{test}} \in \hat{C}_n(X^{\text{test}}) \mid X^{\text{test}} = x \right) \geq 1-\alpha, \quad \text{for almost all } x \in \mathcal X.$$
因为 $X$-conditional coverage 将导致 almost surely (a.s.) 地对每个 non-atomic $x_0 \in \mathcal X$, 都会有:
$$\lim_{\delta \downarrow 0} \sup_{x \in B(x_0, \delta)} \mu(\hat{C}_n(x)) = \infty, \quad \text{a.s.}$$
- $\mu(\cdot)$ 是 Lebesgue measure
- $B(x_0, \delta) = \{x: \|x - x_0\|_2 \leq \delta\}$ 是 $x_0$ 处的 $\delta$-邻域, 即以 $x_0$ 为中心, $\delta$ 为半径的 $\ell_2$ 球.
- non-atomic 指的是 $\lim_{\delta \downarrow 0} P_X(B(x_0, \delta)) = 0$, 即 $x_0$ 处的 $P_X$ 没有原子 (point mass). 这在连续分布中几乎总是成立.

该结论的直观理解是: 如果想要在每个 $x$ 处都保证 $1-\alpha$ 的覆盖率, 且当不对分布 $P$ 进行任何假设时, 则只能通过让预测带 $\hat{C}_n(x)$ 在任意小的邻域内都无限膨胀来实现. 这显然是无效的.


## Improving Local Adaptivity

下面将讨论如何提高 conformal prediction 的局部适应性 (local adaptivity). 这里的局部适应性指的是, 预测带 $\hat{C}_n(x)$ 能够根据 $x$ 的不同而变化, 对于容易预测的 $x$, 预测带应当较窄; 对于难以预测的 $x$, 预测带应当较宽. 这里所有的方法都是在原有的 conformal 的基础上, 通过对分数 (score) 的设计来实现的.

### Studentized Residuals

为方便起见, 这里用 split conformal prediction 框架来说明. 其核心思想是:
- 在 $\mathcal D_{\text{prop}}$ 上不仅估计一个点估计 $\hat f_{\text{prop}}$, 还要估计一个 spread predictor $\hat \sigma_{\text{prop}}$. 这个 spread predictor 并没有具体的限制, 可以是各种回归甚至神经网络, 但是大体上是要估计 $\mathbb{E}\left( ||Y - \hat f_{\text{prop}}(X)|| \mid X \right)$ 之类的量. 这样便可以通过 $\hat \sigma_{\text{prop}}$ 来反映不同 $x$ 位置的难易程度. $\hat \sigma_{\text{prop}}(x)$ 越大, 说明 $x$ 位置越难预测, 预测带应当越宽; 反之亦然.
- 在得到 $\hat f_{\text{prop}}, \hat \sigma_{\text{prop}}$ 后, 我们在 calibration set 上计算 studentized residuals:
    $$R_i^{\text{cal}} = \frac{|Y_i^{\text{cal}} - \hat f_{\text{prop}}(X_i^{\text{cal}})|}{\hat \sigma_{\text{prop}}(X_i^{\text{cal}})}, \quad (X_i^{\text{cal}}, Y_i^{\text{cal}}) \in \mathcal D_{\text{cal}}.$$
- 最后得到区间 $\widehat C_n(X) =  [\hat f_{\text{prop}}(x) - \hat q_{\text{cal}} \hat \sigma_{\text{prop}}(x), \hat f_{\text{prop}}(x) + \hat q_{\text{cal}} \hat \sigma_{\text{prop}}(x)]$. 其概率保证不变. 

![Examples of split conformal prediction with the usual residual score (left panel) and the studentized residual score (right panel). The data comes from the same generative model as in Figures 3 and 5. We can see that the studentized residual adapts to the local hardness of prediction (and delivers something closer to conditional coverage). Credit: Lei et al. (2018).](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250910123147.png)

### Quantile Regression

Studentized residuals 可能会遇到的两个现实问题是:
1. 若 $\hat f_{\text{prop}}$ 很复杂, 那么其可能会过拟合(或者至少拟合的很好), 使得残差过小, 进而使得 $\hat \sigma_{\text{prop}}$ 可供学习的信号过小. 该点可以通过对数据的再划分来缓解, 但会损失效率.
2. 本质上, studentized residuals 关注的是分布的方差, 但是其并不是直接代表了预测的分位宽度, 有些问题里二者可能甚至是相反的. 

因此这里提出另一种思路, 直接通过分位回归 (quantile regression) 来估计分位数.

#### Basic Idea of Quantile Regression

这里简要补充介绍一下分位回归 (quantile regression). 其基本思想是, 对于一个给定的分位数 $\tau \in (0, 1)$, 我们想要估计条件分位数函数:
$$Q_\tau(Y \mid X=x) = \inf\{q: \mathbb{P}(Y \leq q \mid X=x) \geq \tau\}.$$
- 若 $\tau = 0.5$, 则对应中位数回归 (median regression). 对于其他非对称的 $\tau$, 则可以通过 $\tau, 1-\tau$ 来生成上下尾分位, 以确定分位区间. 

为实现这一目标, 首先引入 check loss / pinball loss:
$$\rho_\tau(u) = u(\tau - \boldsymbol{1}\{u < 0\}) = \begin{cases} \tau u, & u \geq 0 \\ (\tau - 1)u, & u < 0 \end{cases}.$$
- 该损失函数的直观理解是, 对于正误差, 其损失为 $\tau$ 倍的误差; 对于负误差, 其损失为 $(1-\tau)$ 倍的误差. 因此, 该损失函数对正负误差的惩罚是不对称的. 这与分位数的定义是一致的, 因为分位数本身就是一个不对称的概念. 
- 若记我们的模型为 $q(x)$ (如在线性模型中为 $q_\tau (x) = x^\top \beta_\tau$), 则分位回归的目标为:
    $$\min_{q} \mathbb{E}[\rho_\tau(Y - q(X))].$$
- 该目标的解恰为 $Q_\tau(Y \mid X=x)$.

$\square$ 

将 QR 应用于 conformal prediction 的思路为 (同样以 split conformal prediction 为例):
- 在 proper training set $\mathcal D_{\text{prop}}$ 上, 我们分别以 $\tau = \alpha/2$ 和 $\tau = 1-\alpha/2$ 训练两个分位回归模型, 得到 $\hat f_{\text{prop, QR}}^{\alpha/2}, \hat f_{\text{prop, QR}}^{1-\alpha/2}$. 这样便得到了两条分位数曲线: $\hat f_{\text{prop, QR}}^{\alpha/2}(x)$ 和 $\hat f_{\text{prop, QR}}^{1-\alpha/2}(x)$.
- 在 calibration set $\mathcal D_{\text{cal}}$ 上, 计算残差:
    $$R_i^{\text{cal}} = \max\{ \hat f_{\text{prop, QR}}^{\alpha/2}(X_i^{\text{cal}}) - Y_i^{\text{cal}}, Y_i^{\text{cal}} - \hat f_{\text{prop, QR}}^{1-\alpha/2}(X_i^{\text{cal}}) \}, \quad (X_i^{\text{cal}}, Y_i^{\text{cal}}) \in \mathcal D_{\text{cal}}.$$ 
- 计算分位数 $\hat q_{\text{cal}}$ (同样为 $\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil$ 分位数). 最终得到预测带:
    $$\widehat C_n(X) = [\hat f_{\text{prop, QR}}^{\alpha/2}(x) - \hat q_{\text{cal}}, \hat f_{\text{prop, QR}}^{1-\alpha/2}(x) + \hat q_{\text{cal}}].$$
- 其概率保证不变.

## Conformal Classification

最后简要对分类问题进行说明. 这里同样以 split conformal prediction 为例. 这里假设分类标签为 $\mathcal Y = \{1, 2, \ldots, K\}$ (共 $K$ 类). 

### Likelihood Scores

1. 同样在 proper training set $\mathcal D_{\text{prop}}$ 上训练一个分类器 $\hat f_{\text{prop}}(x;k) \approx \mathbb{P}(Y=k \mid X=x)$.
2. 在 calibration set $\mathcal D_{\text{cal}}$ 上的每个样本 $(X_i^{\text{cal}}, Y_i^{\text{cal}})$, 取该样本真实类别的预测概率作为分数:
    $$R_i^{\text{cal}} = \hat f_{\text{prop}}(X_i^{\text{cal}}; Y_i^{\text{cal}}).$$
3. 略有不同于回归问题得失, 这里的得分是越大越好. 因此我们取 $1 - \alpha$ 分位数 (即第 $\lfloor \alpha(n_{\text{cal}} + 1) \rfloor$ 小的分数) 作为阈值 $\hat q_{\text{cal}}$. 
4. 最终得到预测集:
    $$\widehat C_n(X^{\text{test}}) = \{k: \hat f_{\text{prop}}(X^{\text{test}}; k) \geq \hat q_{\text{cal}}\}.$$
    - 如果模型非常自信 (某个类别的概率非常大), 那么 $\hat q_{\text{cal}}$ 也会相应地较大, 预测集可能只有一个类别; 反之, 预测集可能包含多个类别. 这和回归问题中预测带的宽度是类似的.

5. 其概率保证不变.

### Cumulative Likelihood

上述的做法是最直接的, 但也有一些问题. 
- 当概率分布的熵较大时, 多数类别的概率都比较”均匀的小”, 这会导致 $\hat q_{\text{cal}}$ 很小 (以确保覆盖概率), 进而导致预测集 $\widehat C_n(X^{\text{test}})$ 包含过多的类别, 甚至所有类别. 这显然是无效的.
- 另一方面, 当分布的熵较小时, 预测集可能只有一个类别, 甚至出现空集之情况. 这也不是我们想要的.

因此我们希望这个预测方法更为 adaptive. 这里提出一种思路, 通过累积概率 (cumulative likelihood) 来实现. 

1. **训练分类器**: 同上, 在 proper training set $\mathcal D_{\text{prop}}$ 上训练一个分类器 $\hat f_{\text{prop}}(x;k) \approx \mathbb{P}(Y=k \mid X=x)$.
2. **计算原始预测概率并排序**: 同样对于 calibration set $\mathcal D_{\text{cal}}$, 计算每个样本 $(X_i^{\text{cal}}, Y_i^{\text{cal}})$ 的预测概率 $\hat f_{\text{prop}}(X_i^{\text{cal}}; k)$, 并将其排序 (从大到小):
    $$\hat f_{\text{prop}}(X_i^{\text{cal}}; \pi_i(1)) \geq \hat f_{\text{prop}}(X_i^{\text{cal}}; \pi_i(2)) \geq \ldots \geq \hat f_{\text{prop}}(X_i^{\text{cal}}; \pi_i(K)).$$
    - 其中 $\pi_i$ 对原始类别 $\{1, 2, \ldots, K\}$ 进行了一次排列 (permutation) 以使其满足上述不等式.
3. **计算累积概率作为 APS 分数**: 对于每个样本 $(X_i^{\text{cal}}, Y_i^{\text{cal}})$, 找到该样本的真实类别 $Y_i^{\text{cal}}$ 在排序中的位置 $k_i$ (即 $\pi_i(k_i) = Y_i^{\text{cal}}$), 并计算累积概率:
    $$R_i^{\text{cal}} = \sum_{j=1}^{k_i} \hat f_{\text{prop}}(X_i^{\text{cal}}; \pi_i(j)).$$
    - 该分数被称为 APS (adaptive prediction set) 分数. 其直观理解是, 该分数反映了模型 “不比真类更不 likely (即 at least equal likely)”  的所有类别的累积概率. 这个分数越小, 说明 at least equal likely 的类别越少, 对应的模型的输出就越自信 (越好); 反之亦然.
4. **计算分位数**: 用上面得到的 APS 分数计算分位数 $\hat q_{\text{cal}}$. 由于这个是一个越小越好的分数, 因此我们取 $\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil$ 分位数 (即第 $\lceil (1-\alpha)(n_{\text{cal}} + 1) \rceil$ 小的分数) 作为阈值 $\hat q_{\text{cal}}$.
5. **构造预测集**: 简而言之, 对于一个新的测试点 $X^{\text{test}}$, 我们同样计算其 APS 分数, 并保留所有 APS 分数不大于 $\hat q_{\text{cal}}$ 的类别. 具体地:
   1. 对于一个新的测试点 $X^{\text{test}}$, 计算其预测概率 $\hat f_{\text{prop}}(X^{\text{test}}; k)$, 并将其排序 (对应类别之概率记为 $\pi_{\text{test}}(k)$):
       $$\pi_{\text{test}}(1) \ge \pi_{\text{test}}(2) \geq \ldots \geq \pi_{\text{test}}(K).$$
    2. 计算累积概率:
        $$C_k^{\text{test}} = \sum_{j=1}^k \pi_{\text{test}}(j), k=1, \ldots, K.$$
        该累积概率是单调不减的, 且 $C_K^{\text{test}} = 1$.
    3. 确定最小的跨越阈值 $\hat q_{\text{cal}}$ 的 $k^*$:
        $$k^* = \min\{k: C_k^{\text{test}} \geq \hat q_{\text{cal}}\}.$$
    4. 最终得到预测集:
        $$\widehat C_n(X^{\text{test}}) = \{\pi_{\text{test}}(1), \pi_{\text{test}}(2), \ldots, \pi_{\text{test}}(k^*)\}.$$
6. 其概率保证不变.