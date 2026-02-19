---
aliases: [广义线性模型, Generalized Linear Models, GLM]
tags:
  - method
  - math/statistics
  - stat/methodology
related_concepts:
  - "[[Exponential_Family]]"
  - "[[Logistic_Regression]]"
  - "[[Deviance]]"
---

# Generalized Linear Models

> Ref: Advanced Categorical Data Analysis

## Lect 4: Model Selection & Fitting Binary Data

### 概要

- 评价一个模型的 goodness of fit 有两种常用指标:
  - Deviance 
  - Pearson's chi-square
- 上述两种分布都服从或者近似服从 $\chi^2$ 分布.
- 用Drop-in-Deviance来比较 Nested Models 的拟合效果.
- 用 R 中的 `anova()` 函数来进行 Deviance Test.
- 范例: Titanic 数据, 用R进行数据的拟合、预测、检验与可视化.

### 1. Deviance  & Pearson's Chi-Square

#### Deviance

对于一个含有$n$个观测的数据集, 我们可以拟合的模型可以从 *NULL Model* (对于全部的$Y$只用$\mu$进行刻画) 直到 *Saturated Model* (对于每一个观测都有一个参数, 即一个 Perfect Fit).

参考似然比检验的思想, 定义*Deviance*为一个目标模型与饱和模型的似然比:

> **[Definition]** (Deviance)
$$
D^*(y,\hat\mu) = -2 \log \frac{L (y,\hat\mu)}{L (y,y)}  = 2\log \ell(y;y) - 2\log \ell(\hat\mu;y) \ge 0 \to \chi^2(n-d)
$$ 
> 其中 $\ell(y;y)$ 是饱和模型的对数似然, $\ell(\hat\mu;y)$ 是目标模型的对数似然. $n$ 是观测数, $d$ 是目标模型的参数数.

Deviance 可以理解为目标模型的拟合效果与饱和模型的拟合效果的差异. 因此 Deviance 越大, 拟合差异越大, 拟合效果越差.

- 可以证明, 在 OLS 下, Deviance 即为 RSS. 因此在整体的理解上可以当作 RSS 的推广. 则显然我们希望 Deviance 越小越好.

**[NOTE]**

- 不同的模型有不同的分布对应着不同的似然函数, 因此也对应着不同的 Deviance. 其中, **Bernoulli** (单次0-1事件) 的 Deviance 函数不适用! 其没有办法反应出模型的拟合效果.
- 其余分布的详细 Deviance 函数可以参考课件 Lec 4 P. 8.


#### Pearson $\chi^2$

Pearson's Chi-Square 是另一种用于检验拟合效果的统计量. 定义为:

> ***[Definition]*** (Pearson's Chi-Square)
> $$
> \chi^2 = \sum_{i=1}^n w_i \frac{(y_i - \hat\mu_i)^2}{\text{Var}(\hat\mu_i)} \sim \chi^2(n-d)
> $$

其中 $w_i$ 是权重, 通常取为1. $\text{Var}(\hat\mu_i)$ 是 $\hat\mu_i$ 的方差.

#### 用 Drop-in-Deviance 比较 Nested Models

在比较两个 Nested Models 时, 我们可以使用 *Drop-in-Deviance* 来比较两个模型的拟合效果. 其中, Nested Models 是指:
> **[Definition]** (Nested Models)
>   称模型 $M_1$ 是模型 $M_2$ 的 Nested Model, 如果 $M_1$ 可以通过令 $M_2$ 的部分参数为0得到.

Drop-in-Deviance 的定义为:
> **[Definition]** (Drop-in-Deviance)
> 给定两个 模型 $M_0$ nested in $M_1$, 则 Drop-in-Deviance 定义为: 
> $$
> D_{M_0}-D_{M_1} \sim \chi^2(d_1-d_0)
> $$
> 其中 $d_1$ 是 $M_1$ 的参数数, $d_0$ 是 $M_0$ 的参数数.

事实上, Drop-in-Deviance 的本质是检验一系列参数 $\beta_i = \cdots = \beta_j = 0$ 是否成立. 这也是为什么 Drop-in-Deviance 只能用于 Nested Models 的比较.

> **[Quiz]** 
> 考虑两个模型: $M_1: \text{logit}(p) = \beta_0 + \beta_1 x_1, M_2: \text{probit}(p) = \beta_0 + \beta_1 x_1 + \beta_2 x_2^2$. 则 $M_1$ 不是 $M_2$ 的 Nested Model.  尽管可以令 $\beta_2 = 0$ 从而使RHS成立, 但LHS的link function 不同, 因此两者不是 Nested Models, 亦不能使用 Drop-in-Deviance 来比较.

另外注意: **尽管Deviance**本身对于Bernoulli分布不适用, 但是**Drop-in-Deviance**(即两个模型的Deviance之差)是适用的. 因此在比较两个模型的拟合效果时, 可以使用Drop-in-Deviance.

---

### 2. Deviance Test in R

***Single Deviance Test***: `pchisq(fm$deviance, fm$df.residual, lower.tail=FALSE)` 

- 对于那些可以使用Deviance进行检验的模型, 假设我们已经拟合了一个模型在R中为`fm`, 则我们可以使用 `pchisq(fm$deviance, fm$df.residual, lower.tail=FALSE)` 来进行检验. 其中 `fm$deviance` 是模型的Deviance, `fm$df.residual` 是模型的自由度. 

- 这个检验是用来对比`fm`与一个饱和模型的拟合效果的. 因此, 如果 $P<0.05$, 则说明模型的拟合效果与饱和模型有显著差异, 即模型不是一个adequate model. 如果 $P>0.05$, 则我们认为模型已经拟合充分, 即 do not reject the hypothesis that current model provides an adequate fit. (但是无法说明这个模型就是最优的, 只能说明这个模型和饱和模型的拟合效果没有显著差异, 是还算可以接受的.)

***Drop-in-Deviance Test for Nested Models Comparison***  `anova(MODELs, test='Chisq')`

- 在 R 中, Drop-in-Deviance 可以通过 `anova(MODELs, test='Chisq')` 来进行检验. 其中 `MODELs` 是一个包含了 Nested Models 的模型列表, `test='Chisq'` 用来输出检验的p-value.其中, 若 `MODELs` 只有一个模型, 则程序会从NULL开始, 逐步增加变量(按照模型列表的顺序)与上一个模型进行Drop-in-Deviance检验. 若 `MODELs` 有多个模型, 则程序会直接进行两两模型的Drop-in-Deviance检验. 

- 若某个模型对应的$P<0.05$, 则说明该模型的拟合效果与上一个模型有显著差异, 即这个模型是一个更好的模型, 这个模型的新增变量是显著不为0的, 是对模型产生了显著影响的, 因此应该保留.

例如:

```R
> anova(fm1, test="Chisq")

Analysis of Deviance Table
Model: binomial, link: logit
Response: Survived
Terms added sequentially (first to last)
Df Deviance Resid. Df Resid. Dev Pr(>Chi)
NULL 755 1025.57
Sex 1 228.929 754 796.64 < 2.2e-16
Age 1 1.058 753 795.59 0.3036
Sex:Age 1 25.030 752 770.56 5.645e-07
```

表示: 从 NULL Model 开始, 其原始的 Deviance 为 1025.57, df=755. 在此基础上引入 `SEX` 后, Deviance 减少为 796.64 (降低了228.929), df=754. 由$P<2.2e-16$ 可以看出, 引入 `SEX` 后的模型显著优于 NULL Model, 因此 `SEX` 是一个显著的变量. 以此类推.
![alt text](image.png)

>  ***[NOTE]***
> 1. Deviance Test 顺序是按照模型列表的顺序进行的, 因此在模型列表中的顺序很重要. 有可能在给定变量$X_1$的基础上, 引入变量$X_2$ 会显著; 但反之在给定变量$X_2$的基础上, 引入变量$X_1$ 会不显著.
> 2. 要满足Hierarchical Principle, 即在引入更高阶的变量时, 对应的低阶的变量不能被删除.


## Lect 5: Binomial Data

### 概要

- Grouped Data 与 Un-grouped Data
- Binomial 模型的拟合过程与系数的解读
- Odds, Odds Ratio, Relative Risk 与 Risk Difference 的含义与计算
- 除了logit link外, 其他link function的含义与应用
- 残差的种类与计算

### 1. Grouped Data 与 Un-grouped Data

对于同样一组数据, 其可以以两种形式出现: Grouped Data 与 Un-grouped Data:

- 对于Grouped Data, 每一行相当于一个类别的集合, 例如:
$$
\begin{array}{|c|c|c|}
\hline
\text{Age} & \text{Frequency} & \text{Survived} \\
\hline
0-10 & 100 & 80 \\
10-20 & 200 & 150 \\
\hline
\end{array}
$$
- 而对于Un-grouped Data, 每一行相当于一个观测, 即一个独立的个体, 例如:
$$
\begin{array}{|c|c|}
\hline
\text{Age} & \text{Survived} \\
\hline
5 & 1 \\
15 & 0 \\
20 & 1 \\
\hline
\end{array}
$$

尽管Grouped Data可以由Un-grouped Data得到, 但两者的建模方式是不同的:

- 在GLM建模中, 我们建模的基本单位都是 ‘行’, 即一个观测. 这导致对于 Group Data, 我们往往认为其服从一个Binomial分布, 其暗含着对于一组的所有个体是同质的, 概率相同的; 但是对于Un-grouped Data, 其针对每个个体是一个单独的Bernoulli分布, 隐含着每个个体都会有一个独立的概率——这可能包含了更多的信息, 但是也增加了模型的复杂度, 或许带来了更多的噪音. 

- 对于Un-grouped Data, 随着样本量的增加, 数据的观测数也会增加, 因此我们没法给出其关于Deviance的渐进性质. 但是对于Grouped Data, 其行数是与组数有关而与样本量无关的, 因此其能够给出关于Deviance的渐进性质. 在这个意义上, Grouped Data 有时候会更加方便.

### 2. Binomial 模型的拟合与解读

本小节将以一个 Skin Cancer 的数据集为例, 重点介绍如何对一个拟合的Binomial模型进行解读. 尤其是在 R 的环境下. 

#### 数据集介绍

Skin Cancer 数据集包含了 15 个观测, 包含变量 `Cases` (患病数), `Town` (城镇名), `Age` (年龄段) 与 `Population` (该地区的总人口). 我们的目标是用 `Age` 与 `Town` 来预测皮肤癌的发病率. 这里假设 $Y_i \sim \text{Binomial}(n_i, p_i)$, 其中 $n_i$ 是该地区的总人口, $p_i$ 是发病率.

在 R 中, 导入数据后, 可以通过 `str()` 来查看数据的结构. 尤其注意其中`Factor`类型的变量, 其第一个水平是基准水平.
``` R
> str(skin)

'data.frame': 15 obs. of 4 variables:
$ Cases : int 1 16 30 71 102 130 133 40 4 38 ...
$ Town : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 2 2 ...
$ Age : Factor w/ 8 levels "15-24","25-34",..: 1 2 3 4 5 6 7 8 1 2 ...
$ Population: int 172675 123065 96216 92051 72159 54722 32185 8328 181343 146207 ...
```

#### 模型拟合

对于 Binomial 的模型, 可以用 `glm(formula = cbind(Cases, Population-Cases) ~ Town + Age, family = binomial, data = skin)` 来进行拟合. 其中 `cbind(Cases, Population-Cases)` 是用来表示每个地区的发病数与未发病数, `family = binomial` 是用来指定模型的分布为Binomial分布.

注意, 在这里`formula`中变量的顺序将影响到后续的 *Deviance Test* 的顺序.

模型最终的拟合结果如下:
```R
> summary(fm)

Call:
glm(formula=y.Bin~ Age + Town, family=binomial, data=skin)
Deviance Residuals:
Min 1Q Median 3Q Max
-1.2830 -0.3355 0.0000 0.3927 1.0820
Coefficients:
Estimate Std. Error z value Pr(>|z|)
(Intercept) -11.69364 0.44923 -26.030 < 2e-16
Age25-34 2.62915 0.46747 5.624 1.86e-08
Age35-44 3.84627 0.45467 8.459 < 2e-16
Age45-54 4.59538 0.45104 10.188 < 2e-16
Age55-64 5.08901 0.45031 11.301 < 2e-16
Age65-74 5.65031 0.44976 12.563 < 2e-16
Age75-84 6.20887 0.45756 13.570 < 2e-16
Age85+ 6.18346 0.45783 13.506 < 2e-16
Town1 0.85492 0.05969 14.322 < 2e-16
(Dispersion parameter for binomial family taken to be 1)
Null deviance: 2330.4637 on 14 degrees of freedom
Residual deviance: 5.1509 on 6 degrees of freedom
AIC: 110.1
Number of Fisher Scoring iterations: 4
```

注意这里的自由度:
- `Null deviance` 的自由度为: $df_{\text{Null}} = n-1_{(\small\text{intercept})} = 15-1 = 14$.
- `Residual deviance` 的自由度为: $df_{\text{Residual}} = n-p-1_{(\small\text{intercept})} = 15-8-1 = 6$.

#### 模型解读

模型的拟合结果如上所示. 其数学表达为 $\text{logit}(\pi_i) = \log \frac{\pi_i}{1-\pi_i} = \mathrm{x}^\top \beta$. 其中 $Y_i$ 是皮肤癌的发病情况, $\pi_i$ 是发病率 (即 $Y_i \sim \text{Binomial}(p_i)$), $\mathrm{x}$ 是模型中的变量, $\beta_0, \cdots, \beta_k$ 是模型的系数.


***截距 Intercept $\beta_0$ 的含义***

- The odds of $Y=1$ when all predictors $\mathrm{x}$ are $\exp(\beta_0)$.
- 在本例中, 当 `Age=15-24` 且 `Town=0` (即基准水平) 时, 皮肤癌的发病odds是 $\exp(-11.69364)$. 

***斜率 Slope $\beta_i$ 的含义***

- Given all other predictors are fixed, $\delta$-unit increase in $X_i$ will result in $\exp(\beta_i \delta)$ times increase in the odds of $Y=1$.
  $$\begin{aligned} \frac{\text{odds}(Y=1|\mathrm{x}_j = A+\delta) }{\text{odds}(Y=1|\mathrm{x}_j = A)} = \exp(\beta_i \delta) ~  \\ 
  \Rightarrow \text{odds}(\mathrm{x}_j = A+\delta) = \exp(\beta_i \delta) \cdot \text{odds}(\mathrm{x}_j = A) \end{aligned}$$
- 在本例中 `Town1: 0.85492`: 对于同样年龄段, 在 `Town=1` 的地区的人, 皮肤癌的发病odds是 `Town=0` (baseline) 的地区的人的 $\exp(0.85492)$ 倍. 
- 在本例中 `Age25-34: 2.62915`: 对于同样地区, 在年龄段 `25-34` 的人, 皮肤癌的发病odds是 `15-24` (baaseline) 的人的 $\exp(2.62915)$ 倍.

***交互项的含义***

- 在本例中, 通过 `anova(fm, fm1, test = "Chisq")` 的检验发现引入交互项 `Age:Town` 并不显著, 这可以被解释为: 
  - Effect of `Town` is homogeneous across all `Age` groups.
  - Effect of `Age` is homogeneous across all `Town` groups.


### 3. Odds Ratio 等指标的计算

- **Odds**: 假设某事件发生的概率为 $p$, 则其 odds 为 $\frac{p}{1-p}$. 可以理解为事件成功的概率是失败的$\frac{p}{1-p}$倍. 若一个事件成功概率很低, 则 $\text{odds}= \frac{p}{1-p} \approx p$.
- **Odds Ratio** (cross-product ratio): 两个事件的 odds 之比: $\text{OR} = \frac{\pi_1/(1-\pi_1)}{\pi_2/(1-\pi_2)} = \frac{\pi_1}{\pi_2} \cdot \frac{1-\pi_2}{1-\pi_1}$.可以理解为两个事件成功的概率之比.
  $$\begin{array}{c|c|c}
  &  Y=1 (\text{Success}) & Y=0 (\text{Failure}) \\
  \hline
  X=1 (\text{Condition} 1) & \pi_1 & 1 - \pi_1 \\
  X=0 (\text{Condition} 2) & \pi_2 & 1 - \pi_2 \\
  \end{array}$$
  - 若 $\text{OR} = 1$ ($\log \text{OR} = 0$), 则两个事件的 odds 相等, 即两个事件的成功概率相等. 或可以认为 $X$ 与 $Y$ 相互独立.
- **Relative Risk**: $\text{RR} = \frac{\pi_1}{\pi_2}$. 可以理解为两个事件成功的概率之比. 当 $X$ 与 $Y$ 相互独立时, $\text{RR} = 1$.
- **Risk Difference**: $\text{RD} = \pi_1 - \pi_2$. 可以理解为两个事件成功的概率之差. 当 $X$ 与 $Y$ 相互独立时, $\text{RD} = 0$.

在GLM中, 一般选择的 `logit` link function 是对应于 odds. 而其他的link function也会对应于不同的指标: 
- `log` link function 对应于 relative risk
  - 其数学表达为: $\log \pi_i = \mathrm{x}^\top \beta \Rightarrow \pi_i = \exp(\mathrm{x}^\top \beta)\in (0, \infty)$. 其暗含的一个问题是: 这个模型并没有直接的约束条件限制概率$\pi$在[0,1]之间, 因此这个模型可能会产生一些不合理的结果. (尽管在本例中, 由于 Skin Cancer 的发病率本身就很低, 因此这个问题并不明显.)
  - 例: 在上述的 Skin Cancer 数据中, 可以通过 `glm(y.Bin ~ Age + Town, family = binomial(link = "log"), data = skin)` 来拟合一个对应于 relative risk 的模型.
    - 若`Town1`的系数为 `0.85`, 则说明: 对于在 `Town=1` 的地区的相应人群, 其患有皮肤癌的Relative Risk是 `Town=0` 的地区的同样年龄段的人的 $\exp(0.85)$ 倍.
- `identity` link function 对应于 risk difference
  - 其数学表达为: $\pi_i = \mathrm{x}^\top$. `identity` link 的使用更加削弱了模型的约束条件. 因此在拟合的过程中可能回产生一系列的问题. 具体的问题处理请参考课件: Lec 5 P. 24. 
  - 但需要意识到, 即使是 `identity` link, 其与 `lm()` (即 OLS) 也并不等价. 因为 `lm()` 是基于$Y$服从正态分布的假设, 而 `glm(link = "identity")` 是基于$Y$服从二项分布的假设. 因此在拟合的过程中, 两者的优化目标是不同的, 也会产生不同的结果.

> 注: 其实在描述上, 上述的指标并没有太大的区别. 在实际运用中, 控制住其他变量不变, 用对应的关系指标进行替换即可. 

### 4. 模型的残差分析

#### 残差的种类

首先回顾, 在传统的OLS中, 残差被定义为: $e_i = y_i - \hat{y}_i$. 其中有一个较强的假设为: $e_i \sim N(0, \sigma^2 I)$, 即残差是同方差的, 即对于不同的观测个体, 其残差的方差是相同(分布)的. 但是在GLM中, 由于我们的假设是 $Y_i \sim \text{Binomial}(n_i, p_i)$, 因此每个观测的方差是不同的($\text{Var}(Y_i) = n_i p_i (1-p_i)$), 因此残差的定义也会有所不同. 

> 注意: 残差是和每个观测都一一对应的, 有多少个样本就有多少项残差. 相当于每个真实的观测值和预测值之间的差异. 因此残差是用来评价模型的拟合效果的一个重要指标.

考虑如下GLM模型: $Y_i \sim \text{Binomial}(m_i, \pi_i)$. 则正常的残差 (raw residual) 定义为: $y_i - \hat{\mu}_i = y_i - m_i \hat{\pi}_i$. 但考虑到上述的方差问题, 我们对这个 raw residual 进行改进,得到下面给出四种GLM中较为常用的残差定义:

- *Pearson Residual*: 对每个观测的残差减去均值除以标准差以得到标准化的残差:
  $$r_{P_i} = \frac{y_i - m_i \hat{\pi}_i}{\sqrt{m_i \hat{\pi}_i (1-\hat{\pi}_i)}}$$
  进一步可知, 一系列的Pearson Residuals的平方和服从 $\chi^2$ 分布:
  $$\sum_{i=1}^n r_{P_i}^2 \sim \chi^2(n-p)$$
- *Deviance Residual*: 用于检验模型的拟合效果, 与Pearson Residual类似, 但是其对于过度离群值的惩罚更大:
  $$r_{D_i} = \text{sign}(y_i - m_i \hat{\pi}_i) \sqrt{2 \left[ y_i \log \left(\frac{y_i}{m_i \hat{\pi}_i}\right) + (m_i - y_i) \log \left(\frac{m_i - y_i}{m_i - m_i \hat{\pi}_i}\right) \right]}$$
  其中, $r_{D_i} = 0$ 时, 表示该观测的拟合效果很好; $r_{D_i} > 0$ 时, 表示该观测的拟合效果较差; $r_{D_i} < 0$ 时, 表示该观测的拟合效果较好.
- *Standard Pearson Residual*: 对于Pearson Residual的进一步修正, $r_{SP_i} = \frac{r_{P_i}}{\sqrt{1-h_{ii}}}$, 其中 $h_{ii}$ 是杠杆值(trivial).
- ***Standard Deviance Residual*** (在GLM中最为常用): $r_{SD_i} = \frac{r_{D_i}}{\sqrt{1-h_{ii}}}$.
  - 近似服从标准正态分布: $r_{SD_i} \sim N(0, 1)$.
  - **一般而言, 若模型的拟合是合适的(adequate), 则应有: $ -2 \leq r_{SD_i} \leq 2$.** 否则, 模型的拟合效果可能存在问题.
  - 在GLM中, 我们常参考的是 *Standard Deviance Residual*. 在`R`中, 可以通过 `boot::glm.diag(fm)$rd` 来获取.

#### 残差图 Residual Plot

通常可以刻画如下三种残差关系图:
- **Residual vs Fitted**: 用于检验残差是否随着拟合值的增加而增加, 从而检验模型是否存在异方差性.
- **Residual vs Explanatory Variable**: 用于检验残差是否随着解释变量的增加而增加, 从而检验模型是否存在非线性.
- **Residual vs index**: 用于检验残差是否随着观测的增加而增加, 从而检验模型是否存在自相关性.

对于这三种残差图, 我们希望看到其残差是随机分布的, 且没有明显的趋势. 且尽量处于 $-2 \leq r_{SD_i} \leq 2$ 的范围内.

> 注意: 与OLS相比, GLM的残差图的解读不仅要看残差的位置, 还要看类别的分布.

#### Partial Residual Plot

在正常的残差分析外, 若模型中含有连续的解释变量(数值型变量), 则我们还可以通过 Partial Residual Plot 来检验模型的拟合效果, 以讨论其是否需要进行一些变换 (如对数变换等 Box-Cox 变换). 具体而言, 对于某个连续型的解释变量 $X_j$, 其 Partial Residual 的定义为在正常的某种残差基础上加回该变量的影响:
> **[Definition]** (Partial Residual)
> $$ r_{\partial j} = \text{Working Residual} + \hat{\beta}_j X_j = g(y) - g(\hat y) = \sum_{k \neq j} \hat{\beta}_k X_k $$

Partial Residual Plot 的纵坐标是 Partial Residual, 横坐标是对应的解释变量. 这个图的分布趋势指示着这个解释变量是否需要进行变换. 若其分布是线性的, 则说明模型的拟合效果较好, 可以直接以线性形式加入模型. 若其分布是非线性的, (例如呈现出对数分布), 则说明该变量可能需要进行变换.


## Lect 6 & 7: Applications for Binary GLM - Bioassay & Epidemiology

### 概要

本小节将主要介绍 GLM 在Bioassay (生物测定) 与 Epidemiology (流行病学) 中的应用. 并以这两个应用场景为背景, 进一步介绍一些统计技巧.

### 1. Bioassay

#### 基本介绍

Bioassay 是一种用于测定生物活性的方法. 通俗而言, 实验者会对生物体施加一定浓度的药物, 然后观察生物体的反应 (如死亡率等). 通过这种方法, 可以得到药物的剂量-反应曲线, 从而得到药物的有效剂量 (ED) 等信息.

在处理 Bioassay 数据时, 有一个重要的概念: **Tolerance Distribution**.  这里假定, 某个生物体对药物的承受上限$U$是一个随机变量, 其分布为 $F(u)$. 则对于某个给定的剂量 $d_i$, 若有 $d_i > U$, 则生物体会死亡. 因此, 死亡率 $\pi_i = \mathbb{P}(U<d_i) = F_X(x)$.


通过下列推导可以发现, 当我们假定$U$服从一些特定的具体分布时, 我们便能得到 probit link 或 logit link 的 GLM 模型.

**在正态分布条件下 $U\sim\mathcal{N}(\mu, \sigma^2):$**

- 死亡率 $\pi_i = \mathbb{P}(U<d_i) = \Phi\left(\frac{d_i-\mu}{\sigma}\right)$, 其中 $\Phi$ 是标准正态分布的分布函数.
- 对应GLM: $\text{probit}(\pi_i) = \Phi^{-1}(\pi_i) =\frac{d_i}{\sigma} - \frac{\mu}{\sigma} = \beta_0 + \beta_1 d_i$. (即 Probit Link 的 GLM 模型)

**在Logistic分布条件下 $U\sim\text{Logistic}(\mu, \sigma):$**
- Logistic 分布的分布pdf为: $f(u) = \frac{\exp\left(\frac{u-\mu}{\sigma}\right)}{\sigma \left(1+\exp\left(\frac{u-\mu}{\sigma}\right)\right)^2}$, 其中 $\mu\in\mathbb{R}, \sigma>0$. 且有 $\mathbb{E}(U) = \mu, \text{Var}(U) = \frac{\pi^2}{3}\sigma^2$.
- 死亡率 $\pi_i = \mathbb{P}(U<d_i) = \frac{\exp\left(\frac{d_i-\mu}{\sigma}\right)}{1+\exp\left(\frac{d_i-\mu}{\sigma}\right)}$, 其中 $\mu$ 是生物体对药物的反应阈值, $\sigma$ 是生物体对药物的反应的敏感度.
- 对应GLM: $\text{logit}(\pi_i) = \log\left(\frac{\pi_i}{1-\pi_i}\right) = \frac{d_i}{\sigma} - \frac{\mu}{\sigma} = \beta_0 + \beta_1 d_i$. (即 Logit Link 的 GLM 模型)

综上, 在 Bioassay 中, Probit Link 是最常见的一种模型. 其分析的核心在于: Dense 的剂量与生物体的response 之间的关系.

#### GLM 拟合流程: 以 Beetles 数据集为例

Beetles 数据集包含了 $n=8$ 个观测, 包含了变量 `dose` (剂量), `death` (死亡数), `total` (总数). 我们的目标是用 `dose` 来预测死亡率. 假设 $Y_i \sim \text{Binomial}(m_i, \pi_i)$, 其中 $m_i$ 是总数, $\pi_i$ 是死亡率.

详细的拟合流程结果见课件 Lec 6 P. 7. 下摘录主要的步骤作为流程的参考.

- 首先利用 logit link 拟合一个一次的无交互项的模型: `fm1 <- glm(cbind(death, total-death) ~ dose, family = binomial(link = "logit"), data = beetles)`.
- 通过 `summary(fm1)` 可以得到模型的拟合结果. 其`Deviance = 11.23, df = 6`. 通过 `pchisq(11.23, 6, lower.tail = FALSE)` 可以得到 $P=0.08 > 0.05$, 因此我们认为这个模型是一个adequate model.
- 通过残差检验以及`Dose`的Partial Residual Plot, 发现残差图具有二次型的趋势, 而partial residual plot也具有二次型的趋势. 因此我们认为 `Dose` 需要引入更高阶的项. 
  - 理论上, 由于 `n=8`, 故可以引入最高阶的项为 `Dose^7`. 
- 记引入直到2阶、3阶的模型为 `fm2`, `fm3`. 通过 `pchiq()` 分别检验各自的拟合是否adequate. 而结果展示这两个模型都是adequate的.
- 通过 `anova(fm1, fm2, fm3, test = "Chisq")` 以进行Drop-in-Deviance检验. 发现二次项对应是显著的, 但三次项对应是不显著的. 因此我们认为直到二次项的`fm2`是一个最优的模型.
  - 这个Drop-in-Deviance检验的结果也可以利用 `drop1(fm, test = "Chisq")` 来进行检验. 其原理基本相同, 但是 `drop1()` 可以直接对模型进行逐步的增减变量的检验.
- 经过残差检验, 发现残差图基本都是随机分布的, 且没有明显的趋势. Partial Residual Plot 也基本是线性的. 因此我们认为模型的拟合效果是较好的.

> **[注]**
> 这里补充一个关于模型的解读问题: 在含有二次型的模型中 (例如Beetles 数据中的最终模型: `y.Bin ~ Dose + I(Dose^2)`), 由于二次项的存在, 无法再像之前那样按照类似’边际效应‘的方式来解释模型的系数 (类似于金融定价中的凸性风险). 因此在这种模型的解读中, 更多考虑的是给定具体`Dose`的取值下的死亡率(或odds), 而不讨论其边际效应.

#### Lethal Dose (LD) 的计算


在 Bioassay 中, 一个重要的指标是 Lethal Dose (LD), 即对应于某个死亡率的剂量. 若给定剂量与死亡率的关系为 $\pi_i = F(d_i)$, 则$LD-X$ 即为使得 $\pi_i = F(FD-X) = X$ 的剂量. 通常我们会关注 $LD_{50}$, 即使得死亡率为50%的剂量.

***点估计***

以二次型为例, 假设模型为 $\zeta = \log\frac{\pi_i}{1-\pi_i} = \beta_0 + \beta_1 d_i + \beta_2 d_i^2$. 则有: 
$$
\log\frac{0.5}{1-0.5} = \hat \beta_0 + \hat\beta_1 LD_{50} +\hat \beta_2 LD_{50}^2
$$
从而可以解出 $LD_{50}$. 对于有多个根的情况, 在求解完后需要验证其是否是落在剂量的取值范围内.

具体求解而言, 更一般地应使用数值方法进行求解. 例如在R中, 可以通过 `uniroot()` 来进行求解. 

不论何种求解方法, 其本质上都是在求解一个以模型的系数为未知数的方程组. 故当模型的拟合完成后, 其各概率的LD值就已经确定了, 即: $LD_X = h(\hat\beta)$. 尽管不一定可以显示表达, 但本质上仍为模型的系数的一个函数. 

***区间估计***

区间估计的一般形式为: $LD_X \in [h(\hat\beta) \pm z_{\alpha/2} \text{SE}(h(\hat\beta))]$. 故重点在于计算 $h(\hat\beta)$ 的标准误差. 

一种统计学上的做法是 **DELTA METHOD**.  

- 假设 $\beta$ 是一维随机变量, 其均值为 $\mu$, 方差为 $\sigma^2$. 则对于可微函数 $g(\beta)$, 有:
$$
\text{Var}(g(\beta)) \approx \left(\frac{\partial g}{\partial \beta}\right)^2 \text{Var}(\beta)
$$
  - 若在大样本等情况下有近似正态分布, 则有:
  $$
  g(\hat \beta) \sim N(g(\mu), (g'(\mu))^2 \sigma^2)
  $$

- 若 $\beta\in\mathbb{R}^p$, 则有:
$$
\text{Var}(g(\beta)) \approx \nabla g(\beta) \text{Var}(\beta) \nabla g(\beta)^\top
$$
其中 $\nabla g(\beta) = \left(\frac{\partial g}{\partial \beta_1}, \cdots, \frac{\partial g}{\partial \beta_p}\right)^\top$.
  - 其大样本下的近似分布为:
  $$
  g(\hat \beta) \sim N(g(\mu), \nabla g(\mu) \text{Var}(\beta) \nabla g(\mu)^\top)
  $$

### 2. Epidemiology

Epidemiology 是研究人群中疾病的发生的因素等问题的学科. 主要有两种研究方法: **Cohort Study** 与 **Case-Control Study**. 

#### Cohort Study

- **Cohort Study** 是一种前瞻性的研究方法. 即在研究开始时, 研究者会选择一组无所研究疾病的人群, 并对其进行长期的追踪. 在足够的时间后, 研究者会统计研究人群中疾病的发生情况, 并与一些因素进行关联分析.

- Cohort Study 由于在研究开始时, 研究者并不知道疾病的发生情况, 因此其研究结果是比较可靠的. **COHORT STUDY 可以用来估计某种因素暴露下的疾病的发生率.** 

- 但其研究周期长, 成本高. 如果一些疾病非常罕见, 则可能需要很长时间才能得到足够的数据.

- 在 Cohort Study 中, 通常会有三种基本的变量:
  - **Exposure Variable**: 暴露变量, 即研究者感兴趣的变量, 如吸烟, 饮食习惯等.
  - **Outcome Variable**: 结果变量, 即研究者感兴趣的结果, 如疾病的发生情况.
  - **Countervailing Variable**: 这类变量较为特殊. 其既与暴露变量相关 (如年龄、性别、社会地位等, 例如社会地位可能会影响到吸烟的习惯), 也与结果变量相关 (如年龄可能会影响到疾病的发生). 但本身不是暴露变量的结果(例如吸烟不会导致年龄增长), 因此其不能作为中介变量. 不过同时由于其与结果也有因果关系, 因此不应该被忽略.

***Framinghan Data 为例***

- Framinghan Data 是一个经典的 Cohort Study 数据集. 其包含了一些变量, 如 `age`, `sex`, `chol` (胆固醇水平), `CHD` (心脏疾病发生情况). 我们的目标是用 `age`, `sex`, `chol` 来预测 `CHD` 的发生情况.

其拟合细节省略, 详见课件 Lec 6 P. 30. 这里强调两个重点方法: **利用AIC 进行变量选择, 对于交互项的模型解读.**

- **AIC for Model Selection**
  - AIC 是一种模型选择的准则. 其定义为: $\text{AIC} = -2\log L + 2p$, 其中 $L$ 是似然函数的最大值, $p$ 是模型的参数个数. **我们倾向于选择 AIC 最小的模型.**
  - 在 `R` 中, 可以通过 `stepAIC()` 来进行模型的逐步选择. 例如 `stepAIC(fm, direction = "both")` 可以进行前向与后向的逐步选择.
    - `direction = "forward"` 表示从一个Null Model 开始, 逐步添加变量.
    - `direction = "backward"` 表示从一个Full Model 开始, 逐步删除变量.
    - `direction = "both"` 表示前向与后向的结合, 在每一步都会考虑添加或删除变量.
  - 对于 `stepAIC()` , 其结果可能会受到初始模型的影响. 因此在使用时, 应该多次尝试不同的初始模型. 并且选择几个最终模型中AIC最小的那个.
  - 对于一些参数变量较多的模型, 可以通过 `stepAIC()` 进行基本的变量选择. 但由于AIC依然倾向于选择参数较多的模型, 因此在AIC选择后, 还可以通过调用 `drop1()` 等函数进行进一步的变量选择 (或`add1()`, `anova()`).
- **Interpretation of Interaction Terms**
  - 在本例中, 最终选择的模型为 `CHD ~ sex + age + chol + age:chol + sex:age`. 其对应的系数为: 
    ``` R
    > coef(fm)
    (Intercept)     sexmale     age50-62 chol>250
    -4.55797135 1.36270131 1.99405104 1.54451751
    chol190-219 chol220-249 age50-62:chol>250 age50-62:chol190-219
    0.05637924 0.92448432 -0.87779164 0.22973705
    age50-62:chol220-249 sexmale:age50-62
    -0.55866803 -0.57195686
    ```
    - 注意到在这个模型中, 有两个交互项: `age:chol` 与 `sex:age`. 这说明, `CHD` 和 `chol` 的关联对于不同年龄段是不同的, 但这种关联对于不同性别是基本相同的 (因为 `sex:chol` 并不显著).
    - 在解释含有交互项的模型时, 要说全所有的变量类别, 并且要注意到交互项“且”的关系. 例如:
      - `chol>250: 1.469` 表示: 对于两个性别相同的个体, 对于胆固醇水平大于250的个体(即`chol>250`), 其发生心脏疾病的odds是胆固醇水平小于250的个体的 $e^{1.469} $ 倍, 如果这两个个体的年龄在30-49岁之间 (base level).
      - `age50-62:chol>250: -0.877` 表示: 对于两个性别相同的个体, 对于胆固醇水平大于250的个体, 其发生心脏疾病的odds是胆固醇水平小于250的个体的 $e^{-0.877 + 1.469} $ 倍, 如果这两个个体的年龄在50-62岁之间.

#### Case-Control Study

***基本介绍***

- **Case-Control Study** 是一种回顾性的研究方法. 即在研究开始时, 研究者会在已经发生疾病的人群中抽取一部分样本(case), 并在没有发生疾病的人群中抽取一部分样本(control). 然后对这两部分样本进行比较.
  - 在抽样时, 是否暴露于某种因素不应该影响研究者的选择. 因此在抽样时, 应该采用一些随机的方法.
- 由于 Case-Control Study 是回顾性的, 因此其结果可能会受到一些偏差的影响. 但由于其研究周期短, 成本低, 因此在一些情况下是比较适用的.
- 但Case-Control Study 的抽样是存在偏差的. 其研究的对象中发生疾病的群体的比例往往远高于在整个人群中的比例. 因此**CASE-CONTROL STUDY 不可以用来估计疾病的发生率.** 
  - Case Control 可以估计: 在给定发病的情况下, 某种因素的暴露率. (例如在肺癌患者中, 吸烟者的比例)
  - Case Control 不可以估计: 在给定暴露的情况下, 某种疾病的发生率. (例如在吸烟者中, 肺癌的发生率)

***Case-Control Study 的研究重点***

- 由于上述所说的原因, 因此**疾病发病的 odds 是不可以直接估计的.** 其具体的数学推导见课件 Lec 7 P. 10. 

- **Case-Control Study 的可行研究对象为 odd ratio (OR) 或 relative risk (RR).**
  - **OR**: $OR = \frac{odds_{\text{exposed}}}{odds_{\text{unexposed}}} = \frac{p_0(x_1) / (1-p_0(x_1))}{p_0(x_0) / (1-p_0(x_0))}$. 其中 $p_0(X)$ 表示在观测到的样本中, 暴露于 $X$ 的个体患病的概率.
  - **RR**: $RR = OR \frac{1 - p(x_1)}{1 - p(x_0)}$. 其中 $p(X)$ 表示在整个人群中, 暴露于 $X$ 的个体患病的概率. 若该疾病的发生率很低, 则有 $RR \approx OR$.

**以 Cerivical Cancer 数据为例**. 假设只考虑是否患病于`age`(<= 或 > 15岁) 之间的关系. 给定最终拟合的模型为: 

``` R
> (fm <- glm(cbind(cases, controls) ~ age, dat, family="binomial"))
Call: glm(formula = cbind(cases, controls) ~ age, family = "binomial",
data = dat)
Coefficients:
(Intercept) age<=15
-2.156 1.383
Degrees of Freedom: 1 Total (i.e. Null); 0 Residual
Null Deviance: 15.39
Residual Deviance: 2.864e-14 AIC: 13.19
11
```

- 其中 `age<=15` 的系数为 `1.383`. 这说明: 在`age<=15`的个体中, 患病的odds是在`age>15`的个体中患病的 $e^{1.383}$ 倍. (虽然这里说的是患病的odds, 但其本质上是在比较odds ratio, 因此这里的解释是合理的.)
- 如果可以认为疾病的发生率很低, 则有 $RR \approx OR = e^{1.383} = 3.98$. 这说明: 在`age<=15`的个体中, 患病的相对风险是在`age>15`的个体中患病的 $3.98$ 倍.

补充: 在本例中, $\hat\beta_j$ 相当于是 log odds ratio. 因此在解释时常通过取指数来得到 odds ratio. 这是点估计的解释. 对于区间估计, 若想求解一个 odds ratio 的区间估计, 则可以先求 log odds ratio (即 $\hat\beta_j$) 的置信区间$CI$, 然后再通过对区间两端取指数来得到 odds ratio 的区间估计: $CI_{OR} = [e^{CI_{\hat\beta_j}}]$.