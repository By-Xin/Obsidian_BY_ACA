---
aliases: [面板数据模型, 固定效应模型, 随机效应模型, Panel Data, Fixed Effect, Random Effect]
tags:
  - concept
  - econ/econometrics
  - stat/regression
related_concepts:
  - "[[Linear_Regression]]"
  - "[[GLS]]"
  - "[[Hypothesis_Testing]]"
---

# Panel Data Models

> Introduction to Panel Data: Fixed Effect Model and Random Effect Model

## Introduction

最基本的模型是 **个体效应模型 (Individual-specific Model)**。它假设所有个体的斜率（边际效应）相同，但截距不同。模型如下给出：
$$
y_{it} = \mathbf{x}_{it}^\top \beta + \mathbf{z}_i^\top\delta + u_{i} + \epsilon_{it}
$$
- $\mathbf{z}_i$ 是个体特有的特征（例如：性别），随时间不变。
- $\mathbf{x}_{it}$ 是随时间变化的特征，随时间改变。
- $u_i+\epsilon_{it}$ 是 **复合误差项 (composite error term)**。
  - $u_i$ 是个体特有误差项（例如：能力、动机），在不同个体间不同但在时间上保持恒定，且可能难以观测。
  - $\epsilon_{it}$ 是 **特异性误差项 (idiosyncratic error term)**，在不同个体和不同时间上都不同。它可以被视为个体的 *运气*。
  - 通常我们假设 $u_i \perp \epsilon_{it}$。

根据 $u_i$ 和 $\mathrm{x}_{it}$ 之间的关系，我们可以定义 **固定效应模型 (Fixed Effect Model)** 和 **随机效应模型 (Random Effect Model)**：
- **固定效应模型**：如果 $u_i$ 与某些 ${x}_{it}$ 或 $z_i$ 相关，那么 OLS 是不一致的。此时我们需要使用固定效应模型来控制 $u_i$。
- **随机效应模型**：如果 $u_i$ 与所有 $(\mathbf{x}_{it}, \mathbf{z}_i)$ 都不相关，那么可以使用随机效应模型。


## 混合回归 (Pooled Regression)

混合回归是估计模型最简单的方法。它将所有数据混合在一起并通过 OLS 进行估计。模型如下给出：
$$
y_{it} = \mathbf{x}_{it}^\top \beta + \mathbf{z}_i^\top\delta + \alpha + \epsilon_{it}
$$

虽然通常我们可以假设每个个体是独立的，但在同一个体内部跨时间通常存在相关性（自相关）。

同一个体跨时间的观测值形成一个 **簇 (cluster)**。同一个簇内的样本通常是相关的。不同簇之间的样本通常是独立的。为了更好的估计，通常我们需要考虑 **聚类稳健标准误 (cluster-robust standard errors)**。

该模型的一个基本假设是：**不存在个体效应**，即所有个体共享相同的截距 $\alpha$。


## 固定效应模型 (Fixed Effect Model)

### 组内估计量 / 去均值模型 (Within Estimator / Demeaned Model)

考虑如下固定效应模型：
$$
y_{it} = \mathbf{x}_{it}^\top \beta + \mathbf{z}_i^\top\delta + u_{i} + \epsilon_{it} \quad \small\text{(1)}
$$
我们要假设 $u_i$ 与 $\mathbf{x}_{it}$ 和 $\mathbf{z}_i$ 相关。因此，我们需要某种方法来控制 $u_i$：

- 给定一个个体 $i$，取该个体随时间变化的所有观测值的平均值：
    $$
    \overline{y}_i = \overline{\mathbf{x}}_i^\top \beta + \mathbf{z}_i^\top\delta + u_{i} + \overline{\epsilon}_i \quad \small\text{(2)}
    $$
- 用 (1) 减去 (2)，我们得到：
    $$
    \tilde{y}_{it} = \tilde{\mathbf{x}}_{it}^\top \beta + \tilde{\epsilon}_{it} \quad \small\text{(3)}
    $$
    其中 $\tilde{y}_{it} = y_{it} - \overline{y}_i$，$\tilde{\mathbf{x}}_{it} = \mathbf{x}_{it} - \overline{\mathbf{x}}_i$，且 $\tilde{\epsilon}_{it} = \epsilon_{it} - \overline{\epsilon}_i$。
    - 只要 $\tilde{\epsilon}_{it}$ 与 $\tilde{\mathbf{x}}_{it}$ 不相关（即 $\mathbb{E}[\epsilon_{it} | \{\mathrm{x}_{it}\}_{t=1}^T = 0]$），我们可以通过 OLS 估计该模型并获得一致估计量，记为 $\hat{\beta}_{FE}$。
    - 由于它主要利用个体内部的变异，因此也被称为 **组内估计量 (Within Estimator)**。

由于可能仍然存在个体内部相关性，我们需要使用 **聚类稳健标准误 (cluster-robust standard errors)** 来获得正确的标准误。

固定效应模型的一个潜在问题是它无法估计随时间不变变量（即 $\delta$）的效应。

### LSDV 估计量 (LSDV Estimator)

估计固定效应模型的另一种等价方法是使用 **最小二乘虚拟变量 (LSDV) 估计量**。依然考虑：
$$
y_{it} = \mathbf{x}_{it}^\top \beta + \mathbf{z}_i^\top\delta + u_{i} + \epsilon_{it} \quad \small\text{(1)}
$$
这里 $u_i$ 可以被视为每个个体的截距。我们可以使用一组虚拟变量来表示 $u_i$：

对于 $n$ 个个体，我们可以使用 $(n-1)$ 个虚拟变量来表示 $u_i$，如下所示：
$$
y_{it} = \alpha + \mathbf{x}_{it}^\top \beta + \mathbf{z}_i^\top\delta + \sum_{j=2}^n \gamma_j D_j + \epsilon_{it} \quad \small\text{(4)}
$$
其中 $D_j = \mathbb{I}(i=j)$ 是个体 $j$ 的虚拟变量，$\gamma_j$ 是 $D_j$ 的系数。

然后我们可以通过 OLS 估计该模型并得到一致估计量，记为 $\hat{\beta}_{LSDV}$。

如果 $n$ 很大，该模型可能会引入过多的虚拟变量。


### 一阶差分估计量 (First-order Difference Estimator)

依然考虑：
$$
y_{it} = \mathbf{x}_{it}^\top \beta + \mathbf{z}_i^\top\delta + u_{i} + \epsilon_{it} \quad \small\text{(1)}
$$
并考虑其一阶差分：
$$
y_{i,t-1}  =  \mathbf{x}_{i,t-1}^\top \beta + \mathbf{z}_i^\top\delta + u_{i} + \epsilon_{i,t-1} \quad \small\text{(5)}
$$ 

取 (1) 和 (5) 之间的差分，我们得到：
$$
y_{it} - y_{i,t-1} = (\mathbf{x}_{it} - \mathbf{x}_{i,t-1})^\top \beta + (\epsilon_{it} - \epsilon_{i,t-1}) \quad \small\text{(6)}
$$
这样的 OLS 估计量被称为 **一阶差分估计量**，记为 $\hat{\beta}_{FD}$。

这里它仅假设 $(\epsilon_{it} - \epsilon_{i,t-1})$ 与 $(\mathbf{x}_{it} - \mathbf{x}_{i,t-1})$ 不相关。这是比组内估计量更宽松的假设。

如果 $T=2$，那么 $\hat{\beta}_{FD} = \hat{\beta}_{FE}$。对于 $T>2$，$\hat{\beta}_{FE}$ 比 $\hat{\beta}_{FD}$ 更有效 (efficient)，前提是 $\epsilon_{it}$ 跨时间独立同分布 (i.i.d.)。


## 随机效应模型 (Random Effect Model)

考虑如下随机效应模型：
$$
y_{it} = \mathbf{x}_{it}^\top \beta + \mathbf{z}_i^\top\delta + u_{i} + \epsilon_{it} \quad \small\text{(1)}
$$
但我们要假设 $u_i$ 与 $\mathbf{x}_{it}$ 和 $\mathbf{z}_i$ 不相关。此时 OLS 是一致的。

然而，由于误差项包含 $(u_i + \epsilon_{it})$，这具有个体内部相关性，因此 OLS 是非有效的 (inefficient)。
  $$ \begin{aligned}
  \text{Cov}(u_i + \epsilon_{it}, u_i + \epsilon_{is}) &= \text{Cov}(u_i, u_i) + \text{Cov}(u_i, \epsilon_{is}) + \text{Cov}(\epsilon_{it}, u_i) + \text{Cov}(\epsilon_{it}, \epsilon_{is}) \\
  &= \text{Var}(u_i) := \sigma^2_u \neq 0 \quad (\ s \neq t)  \\
  \text{Var}(u_i + \epsilon_{it}) &= \text{Var}(u_i) + \text{Var}(\epsilon_{it}) := \sigma^2_u + \sigma^2_{\epsilon}\\
  \text{Corr}(u_i + \epsilon_{it}, u_i + \epsilon_{is}) &= \frac{\text{Cov}(u_i + \epsilon_{it}, u_i + \epsilon_{is})}{\sqrt{\text{Var}(u_i + \epsilon_{it})\text{Var}(u_i + \epsilon_{is})}} = \frac{\sigma^2_u}{\sigma^2_u + \sigma^2_{\epsilon}}
  \end{aligned} $$

我们希望能得到更有效的估计量。


### 随机效应模型的 FGLS 估计 (FGLS for Random Effect Model)

我们希望能消除个体内部相关性。

首先定义：
$$
\theta := 1 - \frac{\sigma^2_\epsilon}{\sqrt{T\sigma^2_u + \sigma^2_{\epsilon}}} = 1 - \sqrt{\frac{\sigma^2_\epsilon}{T\sigma^2_u + \sigma^2_{\epsilon}}} \in [0, 1]
$$
其中 $T$ 是时间周期的数量。

给定原始模型：
$$
y_{it} = \mathbf{x}_{it}^\top \beta + \mathbf{z}_i^\top\delta + u_{i} + \epsilon_{it} \quad \small\text{(1)}
$$

给定个体 $i$，取该个体随时间变化的所有观测值的平均值：
$$
\overline{y}_i = \overline{\mathbf{x}}_i^\top \beta + \mathbf{z}_i^\top\delta + u_{i} + \overline{\epsilon}_i \quad \small\text{(2)}
$$

将 (2) 乘以 $\theta$ 得到：
$$
\theta \overline{y}_i = \theta \overline{\mathbf{x}}_i^\top \beta + \theta \mathbf{z}_i^\top\delta + \theta u_{i} + \theta \overline{\epsilon}_i \quad \small\text{(7)}
$$


用 (1) 减去 (7)，我们得到 **准去均值模型 (Quasi-Demeaned Model)**：
$$
y_{it} - \theta \overline{y}_i = (\mathbf{x}_{it} - \theta \overline{\mathbf{x}}_i)^\top \beta + (1-\theta) \mathbf{z}_i^\top\delta + \left[(1-\theta)u_{i} + \epsilon_{it} - \theta \overline{\epsilon}_i\right] \quad \small\text{(8)}
$$
- 它被称为 *准去均值 (quasi-demeaned)* 是因为它只从原始模型中减去了平均值的 *一部分*。
- 可以证明 $(1-\theta)u_{i} + \epsilon_{it} - \theta \overline{\epsilon}_i$ 不再自相关。（因为 $\theta$ 经过恰当选择，它可以消除个体内部相关性。）

在实践中，我们需要先估计 $\theta$，然后通过 OLS 估计模型 $\small\text{(8)}$。该估计量被称为 **可行广义最小二乘法 (FGLS)** 估计量。具体而言：
$$
\hat{\theta} := 1 - \frac{\hat{\sigma}^2_{\epsilon}}{\sqrt{T\hat{\sigma}^2_u + \hat{\sigma}^2_{\epsilon}}}
$$
其中 $\hat{\sigma}^2_{\epsilon}$ 和 $\hat{\sigma}^2_u$ 分别是 $\epsilon_{it}$ 和 $u_i$ 的估计方差。
- $(\sigma_{\epsilon}^2+\sigma_{u}^2)$ 可以通过原始模型 OLS 估计的残差来估计。
- $\sigma_{\epsilon}^2$ 可以通过原始模型 FE 估计的残差来估计。


## 豪斯曼检验：确定 FE 还是 RE (Hausman Test)

决定性的关键假设：
$$
H_0: u_i \perp \mathbf{x}_{it}, \mathbf{z}_i \quad \text{vs.} \quad H_1: u_i \text{ is correlated with } \mathbf{x}_{it}, \mathbf{z}_i
$$
- 如果 $H_0$ 成立，那么 FE 和 RE 都是一致的。但 RE 更有效 (efficient)。
  - 在大样本下， $(\hat{\beta}_{RE} - \hat{\beta}_{FE}) \xrightarrow{p} 0$
- 如果 $H_1$ 成立，那么 FE 是一致的，但 RE 是不一致的。
  - 如果 FE 和 RE 之间的差异显著，我们要拒绝 $H_0$。

考虑 **豪斯曼检验 (Hausman Test)** 统计量：
$$
\mathcal{H} := (\hat{\beta}_{FE} - \hat{\beta}_{RE})^\top \left[\widehat{\text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE})}\right]^{-1} (\hat{\beta}_{FE} - \hat{\beta}_{RE}) \xrightarrow{d} \chi^2(K)
$$
其中 $k$ 是 $\hat{\beta}_{FE}$ 中需要估计的参数数量，即随时间变化的解释变量的数量。如果 $\mathcal{H}$ 太大，我们就倾向于拒绝 $H_0$。

豪斯曼检验的一个问题是 $\widehat{\text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE})}$ 很难估计。并且它要求误差项具有相同的方差。如果不是这样，我们需要使用 **异方差稳健的豪斯曼检验 (heteroskedasticity-robust Hausman Test)**。

