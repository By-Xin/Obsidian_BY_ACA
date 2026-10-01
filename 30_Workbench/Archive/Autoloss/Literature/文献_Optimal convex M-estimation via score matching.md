#ResearchNotes 

## 1. 介绍

### 研究背景

- 考虑线性模型 $Y_i = X_i^\top \beta_0 + \varepsilon_i, ~i=1,\cdots, n$, 其中 $\beta_0\in \mathbb{R}^d$, $(X_i, Y_i)$ 为 $\text{i.i.d.}$, $\epsilon_i \perp X_i$ 且 $\epsilon_i \sim p_0$ (pdf). 引用 Gauss-Markov 定理：当误差为高斯且具有有限二阶矩时, OLS 是最佳线性无偏估计 (BLUE). 但在实践中, 误差往往是非高斯的, 甚至未知的, 因此需要更多非线性的估计方法.

- 给定某损失函数 $\ell: \mathbb{R} \to \mathbb{R}$, 我们最小化经验风险 (Empirical Risk Minimization, ERM) 得到的估计为:
$$\hat{\beta} \in \arg\min_{\beta\in\mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n \ell(Y_i - X_i^\top \beta)$$当损失函数可导时, 记 $\psi = - \ell'$, 则 $\hat{\beta}$ 应当满足以下方程:
$$\frac{1}{n} \sum_{i=1}^n X_i \psi(Y_i - X_i^\top \hat{\beta}_\psi) = 0.$$
所有这种形式的估计被称为 M-估计 (M-estimation). 常见的 OLS, LAD, Huber Loss 等都属于 M-估计.

- 进一步, 若有关系 $\mathbb{E}[X_i \psi(\epsilon_i)] = 0$, 则称 $\hat{\beta}_\psi$ 为 **Fisher-consistent** 估计. 而若又满足一些正则性条件, 包括 $\psi$ 是可微的且$\mathbb{E}[X_i X_i^\top]$ 是可逆的, 则估计满足以下渐近分布 (Sandwich Theorem):
$$\sqrt{n}(\hat{\beta}_\psi - \beta_0) \overset{d}{\to} \mathcal{N}_d\left(0, V_{p_0}(\psi) \cdot \left[\mathbb{E}(X_1 X_1^\top)\right]^{-1} \right)$$
  其中 :
	- 核心部分 $$V_{p_0}(\psi) = \frac{\mathbb{E}[\psi^2(\varepsilon_1)]}{\left(\mathbb{E}[\psi'(\varepsilon_1)]\right)^2}$$ 完全由误差分布 $p_0$ 和 $\psi = -\ell'$ 决定,
	- 另一部分 $\mathbb{E}(X_1 X_1^\top)$ 是设计矩阵的协方差矩阵, 完全由数据分布决定.


- 因此当数据确定, $V_{p_0}(\psi)$ 是衡量估计效率中唯一和损失函数相关的部分. 这就引出了一个问题: 给定误差分布 $p_0$, 如何选择 $\psi$ 使得 $V_{p_0}(\psi)$ 最小化? 这就是本文的研究目标. 而 $V_{p_0}(\psi)$ 中由于涉及 $\mathbb{E}[\psi^2(\varepsilon_1)]$ 和 $\mathbb{E}[\psi'(\varepsilon_1)]$, 这两个期望的计算往往是困难的, 因此需要一些特殊的技巧.

### MLE 作为效率基准

引入 MLE 作为参考估计量. 这里假设 $\epsilon_i\sim p_0$ 且该密度函数已知且绝对连续. 则可以构造最大似然的损失函数: $\ell = -\log p_0$, 其导数 (score function) 为:
$$\psi_0(y) = - \ell' = \frac{p_0'(y)}{p_0(y)} \boldsymbol{1}_{\{p_0(y) > 0\}}$$