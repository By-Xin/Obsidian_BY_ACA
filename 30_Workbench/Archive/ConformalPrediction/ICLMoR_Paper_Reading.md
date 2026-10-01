# In-context Learning for Mixture of Linear Regressions: Existence, Generalization and Training Dynamics

## Overview

该论文主要研究 Transformer 是否可以通过 In-context Learning (ICL) 的方式学习 Mixture of Linear Regressions (MoR) 问题. 
- MoR: 混合线性回归, 假设数据由 $K$ 个不同的线性回归模型生成, 但我们不知道每个样本具体属于哪个模型.
- 传统方法: 传统上主要通过 EM 算法进行解决, 但该算法非常依赖 initialization, 容易陷入局部最优.
- 本文方法: 通过 In-context Learning 的方式, 只依赖于提供的数据实例而不更新模型参数, 来解决这个问题. 


## Introduction

### Problem Setting

给定数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$, 其中 $x_i \in \mathbb{R}^d$ 是特征, $y_i \in \mathbb{R}$ 是标签.  在 ICL 的任务中, 这些数据不用于更新模型参数, 而只是作为 prompt 进行输入, 用于生成预测. 类似将 Transformer 模型看作一个 meta-learner, 在 forward 中完成估计+预测. 


这里考虑的任务为 MoR (Mixture of Linear Regressions), 即数据由 $K$ 个不同的线性回归模型生成, 但我们不知道每个样本具体属于哪个模型. 具体地, 这里假设: 
- 特征 $x_i \stackrel{i.i.d}{\sim} \mathcal{N}(0, I_d)$
- 噪声 $v_i \sim \mathcal{N}(0,\sigma^2)$ 且与 $x_i$ 独立. 
- 此外, 引入一个 latent variable $z_i$ 来表示样本 $i$ 属于哪个回归成分: $z_i \in \{1,\dots,K\}$, 且服从 categorical distribution:
      $$\mathbb{P}(z_i = k) = \pi_k^*,\quad \sum_{k=1}^K \pi_k^* = 1.$$
- 给定 $z_i = k$, 则标签 $y_i$ 的生成过程为
    $$y_i = x_i^\top w_k^* + v_i,\quad v_i \sim \mathcal{N}(0,\sigma^2).$$
- 即 $y_i$ 服从:
$$y_i \mid (x_i, z_i = k) \sim \mathcal{N}(x_i^\top w_k^*, \sigma^2).$$

在这里, 能够观测到的是 $\{(x_i, y_i)\}_{i=1}^n$, 无法观测 $\{z_i\}_{i=1}^n$. 求解的目标是最大化 marginal log-likelihood:
$$\ell(\theta)= \sum_{i=1}^n \log\left[\sum_{k=1}^K \pi_k\,\varphi\big(y_i;\, x_i^\top w_k,\sigma^2\big)\right],$$
- $\varphi(\cdot; \mu, \sigma^2)$ 是 $\mathcal{N}(\mu, \sigma^2)$ 的概率密度函数.
- $\theta = \big\{\pi_1,\dots,\pi_K,\; w_1,\dots,w_K,\; \sigma^2\big\}$ 为所求参数集合.
- 这个优化目标是一个 log-sum-exp 的形式, 是非凸的, 很难直接进行 MLE 求解. 

### Classic Solution: Expectation-Maximization (EM)

EM 算法是解决 MoR 问题的经典方法, 其基本思想是通过迭代更新 latent parameters 来逼近最优解. 

EM 的 intuition 是, 若真的能知道 $\{z_i\}$, 则概率:
$$p(y,z;\theta) = \prod_{i=1}^n \left[\pi_{z_i}\varphi(y_i; x_i^\top w_{z_i}, \sigma^2)\right].$$
写成 log-likelihood 为:
$$\ell_c(\theta)= \sum_{i=1}^n \sum_{m=1}^M 1_{\{z_i = m\}}
\Big(\log \pi_m- \frac{(y_i - x_i^\top w_m)^2}{2\sigma^2}- \frac12 \log(2\pi\sigma^2)\Big).$$
事实上即为加权最小二乘, 这是一个 convex 的优化问题. 

因此 EM 算法将分为 E-step 和 M-step, 来交替更新 latent parameters.

***E-step***: 计算 soft lable $\gamma_{ik}$. 既然 $z_i$ 是不可观测的, 则尝试计算其 posterior: 
$$\gamma_{ik}^{(t)} := \mathbb{P}\left(z_i = k \mid x_i, y_i, \theta^{(t)}\right) = \mathbb{E}\left[1_{\{z_i=k\}} \mid x_i,y_i;\theta^{(t)}\right]   \stackrel{\text{Bayes}}{=} \frac{\pi_k^{(t)}\varphi(y_i; x_i^\top w_k^{(t)}, {\sigma^{(t)}}^{2})}{\sum_{k=1}^K \pi_k^{(t)}\varphi(y_i; x_i^\top w_k^{(t)}, {\sigma^{(t)}}^2)}.$$
- $\gamma_{ik}^{(t)}$ 可以认为是在第 $t$ 次迭代时, 样本 $i$ 属于成分 $k$ 的概率. 
- 在每一步 E-step 中, 通过当前的参数 $\theta^{(t)}$ 来更新一次 soft label $\gamma_{ik}^{(t)}$. 

***M-step***: 更新参数 $\theta$. 当有了 soft label $\gamma_{ik}^{(t)}$ 后, 就可以得到一个估计的 $\ell_c(\theta^{(t)})$, 因此 M 步是最大化如下函数:
$$Q(\theta\mid\theta^{(t)})= \sum_{i=1}^n \sum_{k=1}^K \gamma_{ik}^{(t)}\Big(\log \pi_k- \frac{(y_i - x_i^\top w_k)^2}{2\sigma^2}- \frac12 \log(2\pi\sigma^2)\Big).$$
- 这里待求解的是 $\theta^{(t+1)} = (\pi_k^{(t+1)},w_k^{(t+1)},\sigma^{(t+1)})$, $\gamma_{ik}^{(t)}$ 是固定常数. 由于在这里 $\pi, w, \sigma$ 可以分别求解如下:
    - $\pi_k^{(t+1)} = \frac{1}{n}\sum_{i=1}^n \gamma_{ik}^{(t)}$
    - $w_k^{(t+1)}= \big(X^\top W_k^{(t)} X\big)^{-1}\, X^\top W_k^{(t)} y$ (每个成分 $k$ 上, 以权重 $\gamma_{ik}^{(t)}$ 加权的最小二乘) 
    - ${\sigma^{(t+1)}}^2 =  \frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K \gamma_{ik}^{(t)} (y_i - x_i^\top w_k^{(t+1)})^2$ (此处进行了归一化 $\sum_{k}\gamma_{ik}^{(t)} = 1$).

因此用伪代码总结一下 EM 算法的整体流程:

> - 输入: $\{x_i, y_i\}_{i=1}^n$, $K$.
> - 初始化: $\theta^{(0)} = (\pi_k^{(0)},w_k^{(0)},\sigma^{(0)})$.
> - 重复直到收敛:
>     - E-step: 对每个样本 $i$, 成分 $k$ 计算
>       $$\gamma_{ik}^{(t)} =\frac{\pi_k^{(t)} \,\varphi\big(y_i;\, x_i^\top w_k^{(t)},\sigma^{2,(t)}\big)}{\sum_{j=1}^K \pi_j^{(t)} \,\varphi\big(y_i;\, x_i^\top w_j^{(t)},\sigma^{2,(t)}\big)}.$$
>   - M-step: 
>       - 更新混合权重 $$\pi_k^{(t+1)} = \frac{1}{n}\sum_{i=1}^n \gamma_{ik}^{(t)}.$$
>       - 更新回归系数 $$w_k^{(t+1)}= \big(X^\top W_k^{(t)} X\big)^{-1}\, X^\top W_k^{(t)} y.$$
>       - 更新噪声方差 $$\sigma^{(t+1)}= \frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K \gamma_{ik}^{(t)} (y_i - x_i^\top w_k^{(t+1)})^2.$$
> - 输出: $\theta^{(T)} = (\pi_k^{(T)},w_k^{(T)},\sigma^{(T)})$.


另外特别指出, 在本论文中考虑的 EM 是 Gradient EM 算法, 即在 M-step 中不是用 WLS 的闭式解, 而是通过梯度下降来进行优化的. 相对而言 Gradient EM 可以更好的在理论上对 attention 机制进行解释, 其在本质上的优化目标和 EM 是一致的.


### Transformer Architecture

***Self-Attention***

这里主要考虑的是 Transformer 的 Decoder 自回归部分. 记输入为 $H \in \mathbb{R}^{p\times q}$, 



### Notation

- 渐近记号
  - $f(n) = \mathcal{O}(g(n))$ 表示存在常数 $C>0$ 和 $n_0>0$, 使得对所有 $n \ge n_0$, 有 $0\le f(n) \le C g(n)$.
  - $f(n) = \Omega(g(n))$ 表示存在常数 $C>0$ 和 $n_0>0$, 使得对所有 $n \ge n_0$, 有 $f(n) \ge C g(n) \ge 0$.
- 范数
  - 向量 $v \in \mathbb{R}^d$ 的 $\ell_2$ 范数记为 $\|v\|_2 = \sqrt{\sum_{i=1}^d v_i^2}$.
  - 矩阵 $A \in \mathbb{R}^{m\times n}$ 的谱范数记为 $\|A\|_{\text{op}} = \max_{v\neq 0} \frac{\|Av\|_2}{\|v\|_2}$.
- 分布
  - 单个样本 $(x,y)$ 来自于联合分布 $\mathcal{P}_{x,y}$. 边际分布为 $\mathcal{P}_x$.
  - 整条 prompt $(x_1,y_1,\dots,x_n,y_n,x_{n+1})$ 来自于联合分布 $\mathcal{P}$.
- 损失函数
  - 对于任意 ICL 算法 $f: H \mapsto \hat{y} \in \mathbb{R}$, 其 MSE 损失定义为
  $$\text{MSE}(f) = \mathbb{E}_{\mathcal{P}}\big[(f(H) - y_{n+1})^2\big].$$

## Existence of Transformer ICL for MoR

本小节尝试说明: 存在这样一个 Transformer 模型, 其能够在 forward 内部就执行一次 Gradient EM 的过程, 从而在 MoR 问题上做到较好的预测效果.

这里首先定义 Oracle Predictor $w^{\text{orc}}$:
$$w^{\text{orc}} := \arg\min_{w \in \mathbb{R}^d} \mathbb{E}_{\mathcal{P}_{x,y}}\big[(y_{n+1} - x_{n+1}^\top w)^2\big]=\sum_{k=1}^K \pi_k^* w_k^*.$$
- 第二个等号的求解过程如下. 而根据线性模型的结论, 最优的 $w^{\star}$ 满足: $\mathbb{E}[x x^\top] w^{\star} = \mathbb{E}[x y].$ 
  - 由于 $x \sim \mathcal{N}(0, I_d)$, 因此 $\mathbb{E}[x x^\top] = \Sigma_x = I_d$.
  - 另一方面, 由于 $y = x^\top w + v$, 且 $v$ 与 $x$ 独立, $\mathbb{E}[v] = 0$, 因此
    $$\begin{aligned} \mathbb{E}[x y] & = \mathbb{E}_z\big[\mathbb{E} [x y \mid z]\big] = 
    \sum_{k=1}^K \pi_k^* \mathbb{E}\left[x (x^\top w_k^* + v) \mid z = k \right] \\
    & = \sum_{k=1}^K \pi_k^* \mathbb{E} [x x^\top] w_k^* + \mathbb{E}[xv] = \sum_{k=1}^K \pi_k^* w_k^*.
    \end{aligned}$$
- Oracle Predictor 相当于假设我们已经知道了所有的 $w_k^*$ 和 $\pi_k^*$, 直接用它们的加权平均来进行预测即可. 最终得到的 oracle 预测为:
$$\hat{y}_{n+1}^{\text{orc}} = x_{n+1}^\top w^{\text{orc}} = x_{n+1}^\top \left(\sum_{k=1}^K \pi_k^* w_k^*\right).$$

- 因此这里的目标是构造一个 Transformer 模型, 使得其在不知道 $\{w_k^*, \pi_k^*\}_{k=1}^K$ 的情况下, 通过观察样本 $\{(x_i,y_i)\}_{i=1}^n$ 来估计出一个 $\hat{w}$, 使得 $\hat{w} \approx w^{\text{orc}}$, 从而使得预测 $\hat{y}_{n+1} = x_{n+1}^\top \hat{w}$ 能够接近 oracle 预测 $\hat{y}_{n+1}^{\text{orc}}$.

***Theorem 3.1 (Existence of Transformer ICL for MoR)***

给定上述的 MoR 任务及 Prompt $H\in\mathbb{R}^{D\times (n+1)}$, 一定能按照上述结构构造一个 Transformer 模型 $\text{TF}_\theta$, (这里要求其层数为 $L$, 每层 attention heads 数量为 $M^{l}\leq 4$), 使得其能够模拟 $T$ 步的 Gradient EM 算法以输出预测 $\hat{y}_{n+1}$.

特别地, 当层数 $L$ 足够大, 且 prompt 的长度 $n$ 及 SNR $\eta$ 满足如下条件时:
$$n \ge C\max\left\{d\log^2\frac{dK^2}{\delta}, \left(\frac{K^2}{\delta}\right)^{1/3},\frac{d}{\pi_{\min}}\log\frac{K^2}{\delta}\right\},$$
$$\eta \ge C K \rho_\pi \log(K\rho_\pi), \quad {\small{\text{for sufficiently large constant } C>0}},$$
则其预测值和 Oracle Predictor 之间的误差 $\Delta_y := \big|\hat{y}_{n+1} - \hat{y}_{n+1}^{\text{orc}}\big| \le \frac{\|w_{\max}^*\|_2}{\sqrt{n}}$ 可被如下界定:
$$\mathcal{O}\left(\sqrt{\log(d/\delta)}\left(\sqrt{\frac{d K \rho_{\pi}^2}{n} \log^2 \left(\frac{n K^2}{\delta}\right)} + \sqrt{\frac{d K \log\left(\frac{K^2}{\delta}\right)}{n \pi_{\min}}}\right)\right)$$
且至少以 $1-9\delta$ 的概率成立, 其中网络总层数大概是 $L = \mathcal{O}\big(T\log(n/d)\big)$.