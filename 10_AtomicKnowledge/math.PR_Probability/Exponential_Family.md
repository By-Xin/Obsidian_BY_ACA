---
aliases: [指数族, Exponential Family, 指数族分布, 指数分布族]
tags:
  - concept
  - math/probability
  - stat/inference
related_concepts:
  - "[[GLM]]"
  - "[[Bernoulli_Distribution]]"
  - "[[Normal_Distribution]]"
  - "[[MLE]]"
  - "[[Sufficient_Statistic]]"
  - "[[Central_Limit_Theorem]]"
  - "[[Logistic_Regression]]"
source: "统计学习; GLM理论"
---

# Exponential Family

## Definition

给出exponential family distributions 的定义：
$$ p(y;\eta) = b(y) \exp(\eta^T T(y) - a(\eta)) $$
其中：
- $\eta$ 是 natural parameter 或者 canonical parameter，是决定分布的一个参数
- $T(y)$ 是标签$y$的一个充分统计量(sufficient statistic)，有时会取$T(y) = y$
- $a(\eta)$ 是 log partition function，是一个归一化因子，使得指数分布族中的分布pdf积分为1

当固定$T$的选择后，不同的$a,b$就会确定不同的分布族，这些分布族都是指数分布族，其分布的参数由$\eta$决定。

事实上，诸如*Gaussian, Bernoulli, Binomial, Poisson, Exponential, Gamma, Beta, Dirichlet*等分布都是指数分布族的一种。

## Bernoulli 分布与指数分布族

已知Bournoulli Distribution：
$$\begin{align*} p(y;\phi) &= \phi^y(1-\phi)^{1-y} \\&= \exp(y\log\phi + (1-y)\log(1-\phi)) \\ &= \exp[(\log\frac{\phi}{1-\phi})y + \log(1-\phi) ] \end{align*}$$



参照GLM的定义，可以发现Bernoulli的分布是令GLM中：
- $T(y) = y$
- $\eta = \log(\frac{\phi}{1-\phi})$  *(有趣的是，其等价于$\phi = \frac{1}{1+e^{-\eta}}$，即为logistic function)*
- $a(\eta) = -\log (1-\phi) = \log(1 + e^{\eta})$
- $b(y) = 1$

## 正态分布与指数分布族

不失一般性，令正态分布的$\sigma^2=1$（因正态分布假定的标准差不会影响参数的取值），则有：
$$\begin{align} p(y;\mu)  &= \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}(y-\mu)^2\right) \\ &= \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}y^2\right)\exp\left(\mu y-\frac{1}{2}\mu^2\right) \end{align}$$

对比GLM的定义，可以发现：
- $\eta = \mu$
- $T(y) = y$
- $a(\eta) = \mu^2/2 = \eta^2/2$
- $b(y) = (1/\sqrt{2\pi}\exp(-y^2/2))$


