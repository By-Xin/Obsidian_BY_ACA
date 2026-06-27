# IPMO: Integrated Prediction and Multi-period Portfolio Optimization   

> **Paper**: [IPMO: Integrated Prediction and Multi-period Portfolio Optimization](https://arxiv.org/abs/2512.11273) 
> 
> #tag: decision_focus_learning, multi-period_portfolio_optimization, 


## Problem Background

### Multi-period Portfolio Optimization (MPO)


在当前时刻 $t$, 投资者当前持有 portfolio 权重 $\boldsymbol{z}_t \in \Omega_t = \{\boldsymbol{z} \in \mathbb{R}^N: \sum_{i=1}^N z_i = 1, z_i \geq 0, i=1,2,\ldots,N\}$  (即为 long-only portfolio), 其中 $N$ 是投资 universe 中的资产数量.  此时, 要规划未来 $H \in\mathbb{N}_+$ 个离散时间点的一系列 portfolio 权重 $\mathbf{z}_{t} = (\boldsymbol{z}_{t+1}, \boldsymbol{z}_{t+2}, \ldots, \boldsymbol{z}_{t+H}) \in \Omega_{t+1} \times \Omega_{t+2} \times \cdots \times \Omega_{t+H} :=\Omega$, 或以矩阵形式表示为
$$
\mathbf{Z}_t = \begin{bmatrix}\boldsymbol{z}_{t+1}^\top \\
\boldsymbol{z}_{t+2}^\top \\
\vdots \\   
\boldsymbol{z}_{t+H}^\top \end{bmatrix} \in \mathbb{R}^{H \times N}.
$$

在当前信息集 $\mathcal{F}_t$ 下, 投资者的目标是最大化未来 $H$ 期的条件期望效用:
$$
\max_{\mathbf{z}_{t} \in \Omega} \mathbb{E}[U(\mathbf{z}_t) | \mathcal{F}_t],
$$
- $\mathcal{F}_t$ 是当前时刻 $t$ 的信息集, 包含了投资者在时刻 $t$ 可用的所有信息.
- $U(\mathbf{z}_t)$ 是整个 allocation path 的效用函数, 其中 $U$ 可能包含 expected portfolio return, portfolio risk, transaction cost 等多个方面的因素.

这里进一步给出 *time-separable structure* 的效用函数的定义, 以便后续的分析. 其将负效用函数(之期望)定义为:
$$
- \mathbb{E}[U(\mathbf{z}_t) | \mathcal{F}_t] = \sum_{s = t+1}^{t+H} [ g_s (\boldsymbol{z}_s) + \lambda h_s(\boldsymbol{z}_s -  \boldsymbol{z}_{s-1}) ],
$$
- $g_s: \mathbb{R}^N \to \mathbb{R}$ 是时刻 $s$ 的 portfolio performance function, 刻画该期静态的资产表现, 例如 expected return, variance 等.
- $h_s: \mathbb{R}^N \times \mathbb{R}^N \to \mathbb{R}$ 是时刻 $s$ 的 dynamic trading function, 刻画根据资产的 portfolio adjustment $\boldsymbol{z}_s -  \boldsymbol{z}_{s-1}$ 所产生的 rebalancing cost, 例如 transaction cost, risk due to trading 等. $\lambda \geq 0$ 是一个权重参数, 其取值越大, 对于 portfolio adjustment 的惩罚越大, 因此投资者在进行 portfolio rebalancing 时会越谨慎.

具体地, 这里将 $g_s$ 定义为经典的 MPC mean-variance objective function, 即
$$
g_s(\boldsymbol{z}_s, \mathcal{F}_t) = \frac{\delta}{2} \boldsymbol{z}_s^\top \widehat{\mathbf{V}}_s \boldsymbol{z}_s - \widehat{\mathbf{y}}_s^\top \boldsymbol{z}_s,
$$
- $\widehat{\mathbf{y}}_s \in \mathbb{R}^N$ 是时刻 $t$ 对未来时刻 $s$ 的资产 return $\mathbf{y}_s$ 的预测.
- $\widehat{\mathbf{V}}_s \in \mathbb{S}_{++}^N$ 是时刻 $t$ 对未来时刻 $s$ 的资产 return covariance matrix $\mathbf{V}_s = \operatorname{Var}(\boldsymbol{z}_s^\top \mathbf{y}_s \mid \mathcal{F}_t)$ 的预测. 其作为 covariance matrix 是 symmetric positive definite 的. 注意到, 由于 $\widehat{\mathbf{V}}_s \succ 0$, 因此 $g_s$ 是一个 strongly convex function. 
- $\delta > 0$ 是 risk aversion parameter, 其取值越大, 投资者对于 portfolio risk 的 aversion 越强.

对应的, 将 $h_s$ 定义为对于 turnover 的 smoothed $\ell_1$ penalty, 即
$$
h_s(\boldsymbol{z}_s -  \boldsymbol{z}_{s-1}) = \rho_\kappa(\boldsymbol{z}_s -  \boldsymbol{z}_{s-1}) = \sum_{i=1}^N \sqrt{(z_{s,i} - z_{s-1,i})^2 + \kappa},
$$
- $\kappa > 0$ 是 smoothing parameter.
  - 当 $|x| \gg \sqrt{\kappa}$ 时, $\rho_\kappa(x) \approx |x| + \frac{\kappa}{2|x|}$; 在 $x = 0$ 附近, $\rho_\kappa(x) = \sqrt{\kappa} + \frac{x^2}{2\sqrt{\kappa}} + O(x^4)$,  故在 $x=0$ 附近, $\rho_\kappa$ 类似一个 quadratic function.  
  - 其导数为 $\rho_\kappa'(x) = \frac{x}{\sqrt{x^2 + \kappa}} \in (-1,1)$, 且 $\rho_\kappa''(x) = \frac{\kappa}{(x^2 + \kappa)^{3/2}} > 0$, 因此 $\rho_\kappa$ 是一个 smooth convex function.
  - $\kappa \to 0^+$ 时, $\rho_\kappa(x) \to |x|$. 整体函数类似于 $\ell_1$ penalty, 不过同时其 curvature 更强, 对应数值 condition 可能更差. 

综上, 完整的优化目标函数就可以写为
$$
\begin{aligned}
\min_{\mathbf{z}_t}\quad  & \quad  \sum_{s = t+1}^{t+H} \left[ \frac{\delta}{2} \boldsymbol{z}_s^\top \widehat{\mathbf{V}}_s \boldsymbol{z}_s - \widehat{\mathbf{y}}_s^\top \boldsymbol{z}_s + \lambda \rho_\kappa(\boldsymbol{z}_s - \boldsymbol{z}_{s-1}) \right] \\
\text{s.t.} \quad & \quad \mathbf{z}_t \in \Omega.
\end{aligned} \tag{P}
$$
其中 $\Omega = \Omega_{t+1} \times \Omega_{t+2} \times \cdots \times \Omega_{t+H}$ 是 portfolio 权重的可行域, $\rho_\kappa (\boldsymbol{x}) = \sum_{i=1}^N \sqrt{x_i^2 + \kappa}$ 是对于 turnover 的 smoothed $\ell_1$ penalty.

### Receding-horizon Model Predictive Control (MPC)

首先展示我们的多期预测与传统的贪心预测的区别. 简而言之, 在当前时刻 $t$ 进行交易的时候, 传统的贪心预测方法只会关心 $\widehat{\mathbf{y}}_{t+1}$ 和 $\widehat{\mathbf{V}}_{t+1}$ 来指导下一期的投资. 然而, 在多期决策的框架下, 我们会同时考虑未来 $H$ 期的预测 $\{\widehat{\mathbf{y}}_{t+1}, \ldots, \widehat{\mathbf{y}}_{t+H}\}$ 和 $\{\widehat{\mathbf{V}}_{t+1}, \ldots, \widehat{\mathbf{V}}_{t+H}\}$ 来指导当前时刻 $t$ 的投资决策. 一个并不严谨的例子, 类似于可以以更长期的预测视野进行"提前布局", 从而在未来的投资过程中获得更好的表现.

不过需要指出, 在 MPC 的框架下, 虽然在时刻 $t$ 会有整个未来 $H$ 期路径的决策:
$$
\boldsymbol{z}_{t+1 \mid \mathcal{F}_t}, \boldsymbol{z}_{t+2 \mid \mathcal{F}_t}, \ldots, \boldsymbol{z}_{t+H \mid \mathcal{F}_t},
$$
然而实际在执行的时候, 只有 $\boldsymbol{z}_{t+1}$ 会被真正交易, 其余的 $\boldsymbol{z}_{t+2}, \ldots, \boldsymbol{z}_{t+H}$ 都是 "虚拟" 的决策. 之后在时刻 $t+1$, 投资者会根据新的信息集 $\mathcal{F}_{t+1}$ 来重新进行 MPC 的优化, 从而得到新的未来 $H$ 期的路径决策:
$$
\boldsymbol{z}_{t+1 \mid \mathcal{F}_{t+1}}, \boldsymbol{z}_{t+2 \mid \mathcal{F}_{t+1}}, \ldots, \boldsymbol{z}_{t+H \mid \mathcal{F}_{t+1}}.
$$
因此也叫做 receding-horizon MPC. 

### Traditional two-stage approach: Predict first, optimize second

下具体考虑一个两阶段的预测与优化的 pipeline. 具体约定符号如下.  定义 $\mathbf{X}_s \in \mathbb{R}^{L \times N}$ 为截止至 $s$ 时刻的过去 $L$ 天的 $N$ 只资产的 return matrix. $\mathbf{Y}_s \in \mathbb{R}^{H\times N}$ 为未来 $H$ 期的 $N$ 只资产的 return matrix. 

首先进行预测. 
- 建立机器学习预测模型 $\widehat{\mathbf{Y}}_s = \phi_\theta(\mathbf{X}_s)$ 来预测未来 $H$ 期的 return matrix $\mathbf{Y}_s$, 通过如下 prediction loss 来训练模型参数 $\theta \in \Theta$:
    $$
    \min_{\theta \in \Theta} L_p(\theta) = \frac{1}{T} \sum_{s=t - T - H + 1}^{t-H} \ell_p(\widehat{\mathbf{Y}}_s, \mathbf{Y}_s)  + \beta R(\theta),
    $$
  - $\ell_p: \mathbb{R}^{H \times N} \times \mathbb{R}^{H \times N} \to \mathbb{R}_+$ 是 prediction loss function, 例如 mean squared error (MSE), mean absolute error (MAE) 等.
  - $R: \Theta \to \mathbb{R}_+$ 是模型的 regularization term, 例如 $\ell_2$ regularization, dropout 等. $\beta \geq 0$ 是 regularization parameter.

- 然后进行决策. 
  - 在日期 $t$, 一次性预测出未来 $H$ 期的 return matrix $\widehat{\mathbf{Y}}_t  = \phi_\theta^*(\mathbf{X}_t)$, 其中 $\theta^* = \arg\min_{\theta \in \Theta} L_p(\theta)$ 是通过上述 prediction loss 训练得到的模型参数. 并估计出 covariance matrix $\widehat{\mathbf{V}}_t$ (例如通过 sample covariance estimator, factor model 等方法).
  - 之后将预测结果 $\widehat{\mathbf{Y}}_t$ 以及 covariance matrix 的预测 $\widehat{\mathbf{V}}_t$ 作为输入, 来求解 MPC 的优化问题 $(\text{P})$, 从而得到未来 $H$ 期的 portfolio 权重路径 $\mathbf{z}^*_t = (\boldsymbol{z}^*_{t+1}, \ldots, \boldsymbol{z}^*_{t+H})$. 之后在时刻 $t$ 进行交易, 只执行 $\boldsymbol{z}^*_{t+1}$. 

这是非常典型的在 decision focus learning 中讨论的两阶段的预测与优化的解耦失配问题. 


## IPMO Learning Framework

### Problem Formulation

对于每个训练时点 $s$ 同样考虑输入 $\mathbf{X}_s \in \mathbb{R}^{L \times N}$ 和输出 $\mathbf{Y}_s \in \mathbb{R}^{H \times N}$, 以及对应的 portfolio 权重路径 $\mathbf{z}_s$ 和预测模型 $\phi_\theta$. 

然而这里的预测模型之输出
$$
\widetilde{\mathbf{Y}}_s(\theta) = \phi_\theta(\mathbf{X}_s)  = \begin{bmatrix}\widetilde{\mathbf{y}}_{s+1}^\top \\
\widetilde{\mathbf{y}}_{s+2}^\top \\ 
\vdots \\
\widetilde{\mathbf{y}}_{s+H}^\top \end{bmatrix} \in \mathbb{R}^{H \times N}
$$
虽然形状和传统的预测模型 $\widehat{\mathbf{Y}}_s$ 一样, 但其并不追求最小化与真实未来 return matrix $\mathbf{Y}_s$ 的 prediction loss. 该预测值不会被直接被评价, 而是进入一个 multi-period portfolio optimization 的优化问题中 (即上述 $(\text{P})$), 从而得到一个关于权重路径 $\mathbf{z}_s$ 和预测 $\widetilde{\mathbf{Y}}_s(\theta)$ 的目标函数 $\widetilde{F}(\mathbf{z}_s, \widetilde{\mathbf{Y}}_s(\theta))$, 得到对应输出:
$$
\mathbf{z}_s^*(\theta) = \arg\min_{\mathbf{z}_s \in \Omega} \widetilde{F}(\mathbf{z}_s, \widetilde{\mathbf{Y}}_s(\theta)) = \arg\min_{\mathbf{z}_s \in \Omega} \widetilde{F}(\mathbf{z}_s, \phi_\theta(\mathbf{X}_s)).
$$
具体地, 这里的 $\widetilde{F}$ 可以定义为平滑后的 MPC objective function, 即
$$
\widetilde{F}(\mathbf{z}_s, \widetilde{\mathbf{Y}}_s(\theta)) = \sum_{k=s+1}^{s+H} \left[ \frac{\delta}{2} \boldsymbol{z}_k^\top \widehat{\mathbf{V}}_k \boldsymbol{z}_k - \widetilde{\mathbf{y}}_k^\top \boldsymbol{z}_k + \lambda \rho_\kappa(\boldsymbol{z}_k - \boldsymbol{z}_{k-1}) \right]. 
$$

在得到 $\mathbf{z}_s^*(\theta)$ 之后, 才会再用 $\mathbf{z}_s^*(\theta)$ 和真实的未来 return matrix $\mathbf{Y}_s$ 来计算一个 decision-focused learning 的 loss:
$$
\frac{1}{T}     \sum_{s=t - T - H + 1}^{t-H} \ell_\text{d}(\mathbf{z}_s^*(\theta), \mathbf{Y}_s)
$$
来衡量最后决策的表现. 

因此, 总的而言, 其可以形式化写作如下 bi-level optimization problem:
$$
\begin{aligned}
& \min_{\theta \in \Theta} \quad \frac{1}{T}     \sum_{s=t - T - H + 1}^{t-H} \ell_\text{d}(\mathbf{z}_s^*(\theta), \mathbf{Y}_s)  \\
& \text{s.t.} \quad \mathbf{z}_s^*(\theta) = \arg\min_{\mathbf{z}_s \in \Omega} \widetilde{F}(\mathbf{z}_s, \phi_\theta(\mathbf{X}_s)), \quad s = t - T - H + 1, \ldots, t-H.
\end{aligned} \tag{IPMO}
$$
或具体化为:
$$\begin{aligned}
\min_\theta\quad
&
\frac1T
\sum_{s=t-T-H+1}^{t-H}
\frac1H
\sum_{k=s+1}^{s+H}
\left[
-
\boldsymbol z_k^*(\theta)^\top y_k
+
\frac{\delta}{2}
\boldsymbol z_k^*(\theta)^\top
V_k
\boldsymbol z_k^*(\theta)
\right]
\\
\text{s.t.}\quad
&
\mathbf z_s^*(\theta)
=
\arg\min_{\mathbf z_s\in\Omega}
\sum_{k=s+1}^{s+H}
\left[
\frac{\delta}{2}
\boldsymbol z_k^\top
\widehat V_k
\boldsymbol z_k
-
\widetilde y_k^\top
\boldsymbol z_k
+
\lambda
\rho(\boldsymbol z_k-\boldsymbol z_{k-1})
\right].
\end{aligned}
\tag{IPMO}
$$


本质上, 传统先预测后决策的方法求解的是
$$
\min_{\theta \in \Theta} \ell_p(\phi_\theta(\mathbf{X}_s), \mathbf{Y}_s),
$$
而 IPMO 则是直接求解
$$
\min_{\theta \in \Theta} \ell_\text{d} \left(\arg\min_{\mathbf{z}_s \in \Omega} \widetilde{F}(\mathbf{z}_s, \phi_\theta(\mathbf{X}_s)), \mathbf{Y}_s \right).
$$

### Bilevel Optimization Problem

对于这样的双层优化问题, 其整体的数据流如下
$$
\mathbf{X}_s \xrightarrow{\phi_\theta}  \widetilde{\mathbf{Y}}_s(\theta) \xrightarrow{\text{S}(\cdot)} \mathbf{z}_s^*(\theta) \xrightarrow{\ell_\text{d}(\cdot, \mathbf{Y}_s)} \mathcal{L}_\text{d}(\mathbf{z}_s^*(\theta), \mathbf{Y}_s).
$$
其中 $\text{S}(\cdot)$ 是求解 MPC 优化问题 $(\text{P})$ 的 solution mapping:
$$
\mathbf{z}_s^*(\theta) =
\text{S}(\widetilde{\mathbf{Y}}_s(\theta)) := \arg\min_{\mathbf{z}_s \in \Omega} \widetilde{F}(\mathbf{z}_s, \widetilde{\mathbf{Y}}_s(\theta)).
$$
因此总的 loss function (forward pass) 可以写为
$$
\mathcal{L}(\theta) = \ell_\text{d}(\text{S}(\phi_\theta(\mathbf{X}_s)), \mathbf{Y}_s).
$$

对应的 backward pass 则为:
$$
\nabla_\theta \mathcal{L}(\theta) = \frac{\partial \ell_\text{d}(\mathbf{z}_s^*(\theta), \mathbf{Y}_s)}{\partial \mathbf{z}_s^*(\theta)} \cdot \frac{\partial \text{S}(\widetilde{\mathbf{Y}}_s(\theta))}{\partial \widetilde{\mathbf{Y}}_s(\theta)} \cdot \frac{\partial \phi_\theta(\mathbf{X}_s)}{\partial \theta}.
$$

对于传统的 Bi-level 优化方法, 通常通过对 lower-level optimization problem 的 KKT condition 来求解 $\frac{\partial \text{S}(\widetilde{\mathbf{Y}}_s(\theta))}{\partial \widetilde{\mathbf{Y}}_s(\theta)}$, 从而得到整个 loss function 的 gradient. 然而这对于一般的矩阵形式是非常复杂的. 

### Mirror Descent 

在本文的 framework 中, 由于需要求解 $\mathbb{z}_s^*(\theta) = \arg\min_{\mathbf{z}_s \in \Omega} \widetilde{F}(\mathbf{z}_s, \widetilde{\mathbf{Y}}_s(\theta))$, 其需要限制 $\mathbf{z}_s \in \Omega = \Omega_{t+1} \times \cdots \times \Omega_{t+H}$, 其中 $\Omega_{t+h} = \{\boldsymbol{z} \in \mathbb{R}^N: \sum_{i=1}^N z_i = 1, z_i \geq 0, i=1,2,\ldots,N\}$ 是 simplex. 

故对于这样的有约束 optimization problem, 可以通过 mirror descent 来求解. 下简要对 MD 进行介绍. 这里暂时不考虑 MPC 的复杂优化函数, 而只单纯考虑一个 general 的含 simplex 约束的 optimization problem, 即
$$
\min_{\mathbf{z} \in \Omega} f(\mathbf{z}).
$$

回顾一般的 Gradient Descent $\mathbf{z}^{(k+1)} = \mathbf{z}^{(k)} - \eta \nabla f(\mathbf{z}^{(k)})$, 其事实上等价于
$$
\mathbf{z}^{(k+1)} = \arg\min_{\mathbf{w}} \left\{ \langle \nabla f(\mathbf{z}^{(k)}), \mathbf{w} - \mathbf{z}^{(k)} \rangle + \frac{1}{2\eta} \|\mathbf{w} - \mathbf{z}^{(k)}\|_2^2 \right\}.
$$

Mirror Descent 则是将上述的 $\ell_2$ distance 换成一个 general 的 Bregman divergence, 其更适合 simplex 这样的约束空间:
$$
D_\psi(\mathbf{w}, \mathbf{z}) = \psi(\mathbf{w}) - \psi(\mathbf{z}) - \langle \nabla \psi(\mathbf{z}), \mathbf{w} - \mathbf{z} \rangle,
$$
其中 $\psi$ 要求是 strictly convex, 其必须在可行域的相对内部可微. 并且希望整体的更新是 closed-form 的, 从而可以高效地求解. 这里定义为 negative entropy function, 即 $\psi(\mathbf{z}) = \sum_{i=1}^N z_i \log z_i$. 其对应的 Bregman divergence 就是 KL divergence, 即
$$
D_\psi(\mathbf{w}, \mathbf{z}) = \operatorname{KL}(\mathbf{w} \| \mathbf{z}) = \sum_{i=1}^N  w_i \log \frac{w_i}{z_i}.
$$
因此最终得到的 MD 的更新为
$$
\mathbf{z}^{(k+1)} = \arg\min_{\mathbf{w} \in \Omega} \left\{ \langle \nabla f(\mathbf{z}^{(k)}), \mathbf{w} - \mathbf{z}^{(k)} \rangle + \frac{1}{\eta} \sum_{i=1}^N w_i \log \frac{w_i}{z_i^{(k)}} \right\}.
$$

下进一步具体求解这个 MD 更新的 closed-form solution. 其等价于优化过程:
$$
\begin{aligned}
\min_{\mathbf{w} \in \mathbb{R}^N} \quad & \langle \nabla f(\mathbf{z}^{(k)}), \mathbf{w} - \mathbf{z}^{(k)} \rangle + \frac{1}{\eta} \sum_{i=1}^N w_i \log \frac{w_i}{z_i^{(k)}}  \\
\text{s.t.} \quad & \sum_{i=1}^N w_i = 1, \\
& w_i \geq 0, \quad i=1,2,\ldots,N.
\end{aligned}
$$
通过求解 Lagrange multiplier, 可以得到其 closed-form solution 为
$$
z_i^{(k+1)} = w_i = \frac{z_i^{(k)} \exp(-\eta \nabla f(\mathbf{z}^{(k)})_i)}{\sum_{j=1}^N z_j^{(k)} \exp(-\eta \nabla f(\mathbf{z}^{(k)})_j)}, \quad i=1,2,\ldots,N.
$$
这相当于进行一个含温度的 softmax 更新, 其中 $1/\eta$ 基本就是 softmax 的 temperature. 

因此, 我们展示, 当我们要处理一个 simplex 约束的 optimization problem 时, 使用由 negative entropy 作为 mirror map 的 MD 方法, 可以得到一个 closed-form 的更新, 并且更新规则恰为一个 softmax 更新, 这样的更新将天然地满足 simplex 约束. 这对于我们求解 MPC 的 optimization problem 是非常有用的.

**Theorem**: 设 $f$ 可微且 strongly convex, 则对 $f$ 的最优解 $\mathbf{z}^\star$ 是 Mirror Descent 的一个固定点, 即 $\mathbf{z}^\star = \arg\min_{\mathbf{w} \in \Omega} \left\{ \langle \nabla f(\mathbf{z}^\star), \mathbf{w} - \mathbf{z}^\star \rangle + \frac{1}{\eta} D_\psi(\mathbf{w}, \mathbf{z}^\star) \right\}$. 
  
*Proof*. 对于全局最优解 $\mathbf{z}^\star = \arg\min_{\mathbf{z} \in \Omega} f(\mathbf{z})$, 其满足 first-order optimality condition, 即 $\langle \nabla f(\mathbf{z}^\star), \mathbf{w} - \mathbf{z}^\star \rangle \geq 0$. 因此考虑 MD 的更新公式
$$
\mathbf{z}^\star = \arg\min_{\mathbf{w} \in \Omega} \left\{ \langle \nabla f(\mathbf{z}^\star), \mathbf{w} - \mathbf{z}^\star \rangle + \frac{1}{\eta} D_\psi(\mathbf{w}, \mathbf{z}^\star) \right\},
$$
其中这里的两项, 都有性质 $\langle \nabla f(\mathbf{z}^\star), \mathbf{w} - \mathbf{z}^\star \rangle \geq 0$ 和 $D_\psi(\mathbf{w}, \mathbf{z}^\star) \geq 0$ 且都在 $\mathbf{w} = \mathbf{z}^\star$ 时取到最小值 $0$. 因此其最小值恰好在 $\mathbf{w} = \mathbf{z}^\star$ 处取得, 从而 $\mathbf{z}^\star$ 是 MD 的一个固定点.

### Implicit Differentiation of MD Fixed Point

在 IPMO 中, 我们要求解的内层优化问题为:
$$
\mathbf{Z}^*(\theta) := \arg\min_{\mathbf{Z} \in \Omega} \tilde{F}(\mathbf{Z}, \phi_\theta(\mathbf{X})).
$$
对这个具体的优化问题应用上述的 MD 方法, 则其更新为
$$
\mathbf{Z}^{(k+1)} = \arg\min_{\mathbf{W} \in \Omega} \left\{ \left\langle \nabla_\mathbf{Z} \tilde{F}(\mathbf{Z}^{(k)}, \phi_\theta(\mathbf{X})), \mathbf{W} - \mathbf{Z}^{(k)} \right\rangle + \frac{1}{\eta} D_\psi(\mathbf{W}, \mathbf{Z}^{(k)}) \right\} := \Phi_\text{MD}(\mathbf{Z}^{(k)}, \theta).
$$
这里我们将整个 MD 的更新过程定义为一个 operator $\Phi_\text{MD}: \Omega \times \Theta \to \Omega$ (回顾 $\Omega \subset \mathbb{R}^{H \times N}$ 是 portfolio 权重的 simplex 约束空间), 其相当于是在完整 $H$ 期 horizon 的完整更新:
$$
\Phi_\text{MD}(\mathbf{Z}, \theta) = \begin{bmatrix}
\Phi_{t+1}(\mathbf{Z}, \theta)^\top \\
\Phi_{t+2}(\mathbf{Z}, \theta)^\top \\
\vdots \\
\Phi_{t+H}(\mathbf{Z}, \theta)^\top
\end{bmatrix} \in \mathbb{R}^{H \times N},
$$
其中, 对于某时间点 $s$, 
$$
\Phi_s(\mathbf{Z}, \theta) = 
\operatorname{Normalize}
\left(
\boldsymbol z_s
\odot
\exp\left(
-\eta
\nabla_{\boldsymbol z_s}
\widetilde F
\bigl(\mathbf Z,\phi_\theta(\mathbf X)\bigr)
\right)
\right) = \begin{bmatrix}
[\Phi_s(\mathbf{Z}, \theta)]_1 \\
[\Phi_s(\mathbf{Z}, \theta)]_2 \\
\vdots \\
[\Phi_s(\mathbf{Z}, \theta)]_N
\end{bmatrix} \in \mathbb{R}^N,
$$
其中对于某资产 $i$, 其更新为
$$
[\Phi_s(\mathbf Z,\theta)]_i
=
\frac{
z_{s,i} \exp(-\eta [\nabla_{z_s} \tilde F(\mathbf Z, \phi_\theta(\mathbf X))]_i)
}{
\sum_{j=1}^N z_{s,j} \exp(-\eta [\nabla_{z_s} \tilde F(\mathbf Z, \phi_\theta(\mathbf X))]_j)
}.
$$
换言之, 在第 $k$ 次迭代中, 对于时刻 $s$ 的资产 $i$, 其权重的更新为
$$
z_{s,i}^{(k+1)}
=
\frac{
z_{s,i}^{(k)}
\exp\left(
-\eta
\left[
\nabla_{\boldsymbol z_s}
\widetilde F
\bigl(\mathbf Z^{(k)},\phi_\theta(\mathbf X)\bigr)
\right]_i
\right)
}{
\sum_{j=1}^N
z_{s,j}^{(k)}
\exp\left(
-\eta
\left[
\nabla_{\boldsymbol z_s}
\widetilde F
\bigl(\mathbf Z^{(k)},\phi_\theta(\mathbf X)\bigr)
\right]_j
\right)
}.
$$

至此, 我们成功地建立起 $\Phi_\text{MD}$ 的 closed-form solution, 并且根据 Fixed Point 定理, 可以得到:
$$
\mathbf{Z}_t^*(\theta) = \Phi_\text{MD}(\mathbf{Z}_t^*(\theta), \theta) \in \mathbb{R}^{H \times N}.
$$
故可以通过 implicit differentiation 来求解 $\frac{\partial \mathbf{Z}_t^*(\theta)}{\partial \theta}$:
$$
\frac{\partial \mathbf{Z}_t^*(\theta)}{\partial \theta} = \left( \mathbf{I} - \frac{\partial \Phi_\text{MD}(\mathbf{Z}_t^*(\theta), \theta)}{\partial \mathbf{Z}_t^*} \right)^{-1} \cdot \frac{\partial \Phi_\text{MD}(\mathbf{Z}_t^*(\theta), \theta)}{\partial \theta}.
$$

进一步, 在实践中, 注意到这里需要求解一个 $HN \times HN$ 的矩阵的 inverse. 若即 $\mathbf{J} := \frac{\partial \Phi_\text{MD}(\mathbf{Z}_t^*(\theta), \theta)}{\partial \mathbf{Z}_t^*}$, 则其 Neumann series expansion 为
$$
\left( \mathbf{I} - \mathbf{J} \right)^{-1} = \mathbf{I} + \mathbf{J} + \mathbf{J}^2 + \cdots.
$$
不过需要满足收敛条件: $\rho(\mathbf{J}) < 1$, 其中 $\rho(\mathbf{J})$ 是 $\mathbf{J}$ 的 spectral radius, 即所有特征值的绝对值的最大值:
$$
\rho(\mathbf{J}) = \max_{\lambda \in \operatorname{eig}(\mathbf{J})} |\lambda|.
$$
在实践中通过控制步长 $\eta$ 不要过大来保证.

因此完整的更新可以写为
$$
\frac{\partial \mathbf{Z}_t^*(\theta)}{\partial \theta} = \left( \mathbf{I} + \mathbf{J} + \mathbf{J}^2 + \cdots \right) \cdot \frac{\partial \Phi_\text{MD}(\mathbf{Z}_t^*(\theta), \theta)}{\partial \theta} \approx \sum_{b = 0}^B \mathbf{J}^b \cdot \frac{\partial \Phi_\text{MD}(\mathbf{Z}_t^*(\theta), \theta)}{\partial \theta}.
$$
其中 $B$ 是一个 predefined 的 truncation level. 并且事实上这样的更新也可以进一步通过反复的 Jacobian-vector product 来高效地求解.


### IPMO Pipeline

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/2a910ee31dc4256f5aef2d921dedb58c.jpg)

下面完整的在开发视角落实 IPMO 的框架. 

- 数据集构建
  - 给定 $N=7$ 支 ETF 资产的日度收益率 $\mathbf{R}\in \mathbb{R}^{T \times N}$, 时间范围为 2011~2024 年. 这里取 2011~2018 年为 in-sample, 用于参数训练与超参数选择. 给定 2018~2024 作为 out-of-sample, 用于模型评估. 
  - 对于每个训练日期 $s = 1, 2, \ldots, T$, 构建一个输入输出对 $(\mathbf{X}_s, \mathbf{Y}_s)$, 其中 $\mathbf{X}_s \in \mathbb{R}^{L \times N}$ 是过去 $L$ 天的日度收益率, $\mathbf{Y}_s \in \mathbb{R}^{H \times N}$ 是未来 $H$ 天的日度收益率. 其中 $L=120$, $H=20$, $T = 250$ 为训练集大小. 因此一个完整的训练集即为
    $$
    \mathcal{D}_t^{(i)} = \{(\mathbf{X}_s, \mathbf{Y}_s)\}_{s = 1}^T,
    $$
    其中 $i$ 是训练集的索引. 这里需要说明, 每个 $\mathcal{D}_t^{(i)}$ 都是一个完整的数据集, 用这 $T$ 期的 $(\mathbf{X}_s, \mathbf{Y}_s)$ 来训练一个预测模型 $\theta_i$. 因此, 对于每个训练集, 我们都能得到最终的一套参数, 以及超参数设置.  
  - 每两个相邻的训练集之间相隔 $20$ 天, 不同的训练集类似于查看在不同 scenario 年景下的模型表现. 在实践中, 真正的作用是通过多期的数据集进行超参数选择. 因此其作用类似于时间序列版的 cross-validation.

- 训练细节
  - 对于每个具体的训练集 $\mathcal{D}_t$, 其将运行 IPMO 的算法. 具体流程如下. 

## Experiments

### 数据集



* **资产**：7 个主要资产类别的 ETF

  * 股票：VTI, IWM
  * 债券：AGG, LQD, MUB
  * 商品：DBC
  * 贵金属：GLD
* **时间段**：

  * 2011–2018 年用于 **in-sample / 超参数调优**
  * 2019–2024 年用于 **out-of-sample 测试**
* **输入特征**：

  * 每日历史收益 $X_s \in \mathbb{R}^{L \times N}$（过去 L=120 天）
  * 目标输出 $Y_s \in \mathbb{R}^{H \times N}$（未来 H 天）
  * 协方差矩阵 $V_s$ 用过去 20 天的 EWMA 简单估计，并加 εI 确保正定

### 预测模型

* **RLinear**：

  * 单层全连接网络，独立建模每个资产
  * 历史 5 天收益做均值以缓解噪声
  * L2 正则化
* **CNN-LSTM**：

  * CNN（64 个 channel，kernel=5，stride=1，ReLU），max-pooling kernel=2
  * LSTM 两层，hidden size=64
  * 所有资产共用池化层
  * Reversible instance normalization 处理分布漂移
* **训练策略**：

  * 预测未来 H 天收益
  * 滚动窗口训练：过去 250 天，每 20 天重新训练
  * 学习率 $\gamma \in \{0.001, 0.002, 0.005, 0.01\}$.
  * Turnover penalty $\lambda \in \{0.0001, 0.0005, 0.001, 0.005, 0.01\}$.
  * Horizon $H \in \{1, 5, 10, 20, 50\}$.
  * 两阶段 (TS) 与 IPMO 框架均使用 Sharpe 和 turnover 选择超参数


### 内层优化

* **IPMO 内层**：

  * 目标：多期均值-方差加交易成本惩罚
  * 平滑交易成本使用 
  * 内层求解器：

    * Forward pass：ADMM 求解多期优化 (2)
    * Backward pass：Mirror-descent fixed-point (MDFP) implicit differentiation
  * Output：$H\times N$ allocation sequence
  * 只执行第一步 $z_{t+1}^*$，其余作为梯度传递辅助


### 外层 loss

* 外层决策损失：
  $$
  \mathcal{L}_d(z_s^*(\theta), Y_s)
  $$
* 用于训练预测模型 θ
* 通过公式 (11) 用链式法则：
  $$
  \nabla_\theta \mathcal{L}_d(\theta) = \frac{1}{T} \sum_s \frac{\partial \mathcal{L}_d}{\partial z_s^*} \frac{\partial z_s^*}{\partial \theta}
  $$


### 实验比较

#### 基线：

* Equal-Weighted (EW)
* Classical Mean-Variance (MV)
* 两阶段 TS + 同样预测器

#### 实验指标：

* 收益：Annualized Return
* 风险：Volatility
* 风险调整指标：

  * Sharpe
  * Calmar
  * Return / avg drawdown
* Turnover
* MSE of prediction
* Portfolio weight trajectories（heatmap & total variation）

#### 核心发现：

* IPMO **在所有指标上优于两阶段 TS**，尤其 Sharpe、Calmar、低交易频率
* RLinear + IPMO 与 CNN-LSTM + IPMO 都优于对应 TS
* IPMO 权重调整平滑，减少了大幅波动，表现出 buy-and-hold 行为
* MSE：

  * RLinear：IPMO MSE 略高，但决策质量提升（决策导向训练）
  * CNN-LSTM：IPMO MSE 较高，但收益与风险指标提升，说明更好捕获多期 intertemporal structure


### 计算效率

* **梯度求解**：MDFP vs BPQP vs CvxpyLayer
* Table 4 & Figure 6：

  * MDFP runtime 几乎不随 H 增长（0.69s 增长从小 horizon 到大 horizon）
  * BPQP 线性增长
  * CvxpyLayer 增长最陡
* MDFP 利用 Neumann series 近似 $(I - \partial_z \Phi)^{-1}$
* 优点：

  * 内存占用低
  * 大 horizon 下 runtime 几乎不变
  * 可复用 elementwise Jacobian-vector products


