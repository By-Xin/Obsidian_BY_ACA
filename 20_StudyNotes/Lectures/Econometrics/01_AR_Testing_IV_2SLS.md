

# Econometrics

  

0228

  

---

  

#### AR Model


  

<img src="https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260228145525.png" alt="20260228145525" width="100%" />

  

Decay: ACF 柱子随着 lag 增加有规律地变小. 可能单调变小, 也可能正负交替振荡. 但“形状有连续性”

  

Cutoff: 相关性在某一阶之后突然不再有结构. 后面基本只是随机噪声

  

---

  

#### AR Model

  

![20260228145659](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260228145659.png)

  

---

  

#### Hypothesis Testing

  

<br>

<br><br><br><br><br><br><br><br><br><br><br><br>

  

----

  

#### Hypothesis Testing

  

<br>

<br><br><br><br><br><br><br><br><br><br><br><br>

  

---

  

#### Degree of Freedom

  

1. ANOVA 分解: $Y_i = \beta_0 + \beta_1 X_{i1} + \cdots + \beta_{k} X_{ik} + \epsilon_i$. $\text{TSS} = \text{ESS} + \text{RSS}$, $n-1 = k + (n-k-1)$

- $\text{TSS} = \sum_{i=1}^n (Y_i-\bar{Y})^2, \text{df}_{\text{TSS}}=n-1$.

- $\text{ESS} = \sum_{i=1}^n (\hat{Y}_i-\bar{Y})^2, \text{df}_{\text{ESS}}=k$.

- $\text{RSS} = \sum_{i=1}^n (Y_i-\hat{Y}_i)^2, \text{df}_{\text{RSS}}=n-k-1$.

  

2. $\beta$ 的 t-test: $t = \frac{\hat{\beta}_j - \beta_{j,0}}{SE(\hat{\beta}_j)}$，df = $n-k-1$.

  

3. Generalized t-test: $H_0: a_1\beta_1 + \cdots + a_k\beta_k = c$ vs. $H_1: a_1\beta_1 + \cdots + a_k\beta_k \neq c$.

  

$t = \frac{a_1\hat{\beta}_1 + \cdots + a_k\hat{\beta}_k - c}{SE(a_1\hat{\beta}_1 + \cdots + a_k\hat{\beta}_k)}$, df = $n-k-1$.

  

---

  

#### Degree of Freedom

  

4. 误差残差: $\hat{\epsilon}_i = \frac{\text{RSS}}{n-k-1}$. $\text{SER} = \sqrt{\hat{\epsilon}_i}$.

  

5. F-test: $F = \frac{\text{ESS}/k}{\text{RSS}/(n-k-1)}$，df1 = $k$, df2 = $n-k-1$.

  

6. Generalized F-test: $F = \frac{(RSS_r - RSS_{ur})/q}{RSS_{ur}/(n-k-1)}$，df1 = $q$, df2 = $n-k-1$.

  
  

---

  

#### Exercise 1

  

(c) $w_i = -64 + 32 \log(\text{school}) + e_i$.

  

- schooling 每增加 $1\%$，wage 平均增加 $0.32$。

  

(d) $\log(w_i)=−0.93+1.43\log(\text{school})+e_i$

  

- schooling 每增加 $1\%$，wage 平均增加 $1.43\%$。

- Elasticity of wage with respect to schooling is $1.43$.

  

---

#### 关于 Elasticity 的补充

  

$\text{Elasticity} = \frac{\partial \log(y)}{\partial \log(x)}$.

  

考虑 regression: $\log(y_i) = \beta_0 + \beta_1 \log(x_i) + \beta_2 [\log(x_i)]^2 + \cdots + e_i$.

  

则 $y$ 对 $x$ 的 elasticity 是 $\frac{\partial \log(y)}{\partial \log(x)} = \beta_1 + 2\beta_2 \log(x)$，即 elasticity 可能随 $x$ 的水平而变化.

  

有一种问法: 检验在 $x = c$ 处 elasticity 是否等于某个值, 例如 0, 则 $H_0: \beta_1 + 2\beta_2 \log(c) = 0$ vs. $H_1: \beta_1 + 2\beta_2 \log(c) \neq 0$. 为 t-test, 计算 test statistic: $t = \frac{\hat{\beta}_1 + 2\hat{\beta}_2 \log(c) - 0}{SE(\hat{\beta}_1 + 2\hat{\beta}_2 \log(c))}$, df = $n-k-1$.

  

----


  

#### Exercise 2

  

![20260228150929](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260228150929.png)

  

---

  

#### Exercise 2

  

(c) 检验 $H_0: \beta_1 = 0$ vs. $H_1: \beta_1 \neq 0$. 且 $\hat{\beta}_1 = -0.02619$, $SE(\hat{\beta}_1) = 0.01157$.

  

- p-value: The probability of observing a test statistic ***as extreme as, or more extreme than*** the current situation, under $H_0$.

  

- Given $t_0 = \frac{-0.02619 - 0}{0.01157} \approx -2.262$, $\text{df} = n-k-1 = 186-1-1=184$.

- Under $H_0$, $t \sim t_{184}$

- $p = \mathbb{P}(|t_{184} > t_0|) = 2\mathbb{P}(t_{184} > 2.262) \approx 0.025$.

  

---

  

#### Exercise 2

  

![20260228152057](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260228152057.png)

  

---

  

#### Exercise 2

  

(e) 显著性检验 $H_0: \beta_1 = -0.1$ vs. $H_1: \beta_1 \neq -0.1$.

  

$t=\frac{\hat\beta-\beta_0}{SE(\hat\beta)}=\frac{-0.02619-(-0.1)}{0.01157}=\frac{0.07381}{0.01157}\approx 6.379$

  

$t_{0.975,184}\approx 1.973$. Since $|t|=6.379 > 1.973$, reject $H_0$ at 5% significance level.

  

---

  

#### Exercise 2

  

(f) Power of the test with real $\beta_1=-0.08$.

  

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

  

---

  

#### Exercise 3

  

Key: 在统计学中有这样一个关系 $t_v^2 = F_{1,v}$，即 $t$ 分布的平方服从 $F$ 分布. 这带来了一个检验的等价关系: 如果我们进行 F-test, 但只有一个 restriction ($q=1$, 只有一个等号), 则此时, F-statistic = t-statistic 的平方. 因此, 在这种情况下, F-test 和 t-test 是等价的.

  
  

----

  

#### Endogeneity

  

考虑 Regression: $Y = \beta_0 + \beta_1 X + \beta_2 Z + \epsilon$.

  

- 定义: $X$ 是 endogenous 的，如果 $Cov(X, \epsilon) \neq 0$ 或 $\mathbb{E}[\epsilon|X] \neq 0$. Endogeneity 导致 OLS 估计 $\hat{\beta}_1$ 有偏且不一致.

- 来源:

- Simultaneous causality: $X$ 和 $Y$ 互相影响. 如供给和需求模型.

- Omitted variable bias: $Z$ 影响 $Y$，且与 $X$ 相关，但 $Z$ 没有被包含在回归中. 如能力影响工资，且与教育相关，但能力没有被包含在回归中, 则能力出现在误差项中，导致教育与误差项相关.

- Measurement error: $X$ 的测量误差导致 $X$ 与误差项相关.

  

----

  

#### Endogeneity: IV and 2SLS

  

IV (Instrumental Variable) 是一种解决 endogeneity 的方法. 先考虑只有一个 endogenous regressor $X$ 的情况. 需要一个 instrument $Z$ 满足:

- Relevance: $Cov(Z, X) \neq 0$.

- Exogeneity: $Cov(Z, \epsilon) = 0$.

  

---

  

#### Endogeneity: IV and 2SLS

  

2SLS 是一种实现 IV 的方法. 一般的, 考虑回归 $Y \sim X_1, \cdots, X_k, W_1, \cdots, W_m$，其中 $X_1, \cdots, X_k$ 是 endogenous regressors，$W_1, \cdots, W_m$ 是 exogenous regressors. 则至少需要 $k$ 个 instruments $Z_1, \cdots, Z_k$ 满足 relevance 和 exogeneity 条件. 2SLS 的步骤如下:

1. 第一阶段回归: 对每个 endogenous regressor $X_j$，进行回归 $X_j \sim Z_1, \cdots, Z_k, W_1, \cdots, W_m$，得到 fitted values $\hat{X}_j$

2. 第二阶段回归: 用第一阶段的 fitted values 代替原来的 endogenous regressors，进行回归 $Y \sim \hat{X}_1, \cdots, \hat{X}_k, W_1, \cdots, W_m$，得到 2SLS 估计 $\hat{\beta}_j^{2SLS}$.

  

IV 得到的估计量:

1. 由于 $\hat{X}_j$ 是 $Z$ 的线性组合，且 $Z$ 与 $\epsilon$ 不相关，因此 $\hat{X}_j$ 与 $\epsilon$ 也不相关. 这保证了第二阶段回归中 $\hat{X}_j$ 的 exogeneity. 并且是 consistent 的.

2. 不保证 unbiased, 且估计的方差通常比 OLS 大.

  

---

  

#### Endogeneity: IV and 2SLS

  

在一个简单的情况下, 只考虑 $Y = \beta_0 + \beta_1 X + u$, $X$ 是 endogenous 的, 只有一个 instrument $Z$. 则 2SLS 估计 $\hat{\beta}_1^{2SLS} = \frac{Cov(Z, Y)}{Cov(Z, X)}$.

  

稍微再复杂一点, 考虑 $Y = \beta_0 + \beta_1 X + \beta_2 W + u$, 其中 $X$ 是 endogenous 的, $W$ 是 exogenous 的, 只有一个 instrument $Z$.

  

- 此时, $Var(\hat\beta_{1,OLS})=\frac{\sigma^2}{\sum \tilde X^2}$, $Var(\hat\beta_{IV})=\frac{\sigma^2}{\sum \tilde{\hat X}^2}$.

- 其中 $\tilde X$ 是回归 $X = \pi_0 + \pi_1 W + v$ 的残差, $\tilde{\hat X}$ 是回归$\hat X = \pi_0' + \pi_1' W + v'$ 的残差.

- 二者可以证明有关系: $\sum \tilde{\hat X}^2=R_1^2\sum \tilde X^2$, 其中 $R_1^2$ 2SLS 中第一阶段回归 $X \sim Z, W$ 的 $R^2$.

  

这表示: OLS 用的是 $X$ 的全部 variation 来估计 $\beta_1$, 而 2SLS 只用 $X$ 中与 $Z$ 相关的部分 variation 来估计 $\beta_1$. 因此, 当 instrument 很弱 (即 $R_1^2$ 很小) 时, 2SLS 的方差会很大.

  
  

----

  

#### Endogeneity: IV and 2SLS

  
  

因此需要对 IV 的两个条件进行检验.

- Relevance: 要求 $Cov(Z, X) \neq 0$. 可以通过第一阶段回归 $X \sim Z, W$ 的 F-test 来检验. 经验上, 如果 F-statistic 大于 10，则认为 instrument 是 strong 的. 反之则是 weak IV.

- Weak IV 的后果: IV 和 OLS 一样 biased, SE 大, t 小.

  

---

#### Endogeneity: IV and 2SLS

  

- Exogeneity: 要求 $Cov(Z, \epsilon) = 0$. 可以用 Sargan / Hansen J-test 来检验.

- 该 test 只能在工具变量数量多于 endogenous regressor 数量时进行 (即 overidentified). 核心思想: 如果工具变量是外生的, 那么它们不应该解释 IV 残差.同样考虑 $Y \sim X + W$ 的回归.

- 首先进行 2SLS, 得到 $\hat{\beta}_{1,IV}$, $\hat{\beta}_{2,IV}$. 然后可以计算残差 $\hat{u}_{IV} = Y - \hat{\beta}_{1,IV} X - \hat{\beta}_{2,IV} W$.

- 然后用这个残差对外生的 $W$ 和 工具变量 $Z$ 进行回归 $\hat{u}_{IV} \sim Z, W$. 如果工具变量是外生的, 那么它们不应该解释残差, 即 $Z$ 的系数应该不显著. 故 H_0: 第二步回归不显著, 即外生性成立.

- J-test 的统计量是 $J = n R^2 = mF$ (其中 $m$ 是工具变量的数量, $F,R$ 都是回归 $\hat{u}_{IV} \sim Z, W$ 的统计量). $J\sim \chi^2_{m-k}$, 其中 $m-k$ 是工具变量数量与 endogenous regressor 数量之差. 如果 $J$ 的值很大, 则拒绝 $H_0$, 认为工具变量不外生.

  

---

  

#### Endogenity: Hausman-Wu Test

  

$$Y = \beta_0 + \beta_1 X + \beta_2 W + u$$

  

检验 $H_0:\; Cov(X,u)=0.$ 即是否是内生的.

  

第一步: 和 2SLS 相同, $X\sim Z, W$, 得到残差 $\hat{v}$ 和 估计值 $\hat{X}$. $X = \hat{X} + \hat{v}$.

  

第二步: 用 $Y \sim X, W, \hat{v}$ 进行回归. 然后检验 $\hat{v}$ 的系数是否显著. 如果 $H_0: \beta_{\hat{v}} = 0$ 被拒绝, 则认为 $X$ 是内生的.

  

- 核心: $\hat{X} = \hat{\pi}_0 + \hat{\pi}_1 Z + \hat{\pi}_2 W$ , 是由两个外生变量 $Z, W$ 线性组合而成的, 因此 $Cov(\hat{X}, u) = 0$. 如果 $X$ 是内生的, 则 $Cov(X, u) = Cov(\hat{X} + \hat{v}, u) \neq 0$. 因此如果存在内生性, 一定会反映在 $Cov(\hat{v}, u) \neq 0$ , 对应系数不为零.

---

  

#### Endogenity: Forbidden Regression

  

$$Y=\beta_0+\beta_1 X+\beta_2 X^2+\beta_3 W+u$$

  

- 第一步: 当做有两个内生变量进行回归:

- $X=\pi_0+\pi_1 Z+\pi_2 Z^2+\pi_3 W+v_1\Rightarrow \hat X$

- $X^2=\rho_0+\rho_1 Z+\rho_2 Z^2+\rho_3 W+v_2\Rightarrow \widehat{X^2}$

- 第二步: $Y=\beta_0+\beta_1 \hat X+\beta_2 \widehat{X^2}+\beta_3 W+u$.

  
  
  

----

  
