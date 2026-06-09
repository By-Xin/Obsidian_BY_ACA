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
\widetilde{\mathbf{Y}}_s = \phi_\theta(\mathbf{X}_s)  = \begin{bmatrix}\widetilde{\mathbf{y}}_{s+1}^\top \\
\widetilde{\mathbf{y}}_{s+2}^\top \\ 
\vdots \\
\widetilde{\mathbf{y}}_{s+H}^\top \end{bmatrix} \in \mathbb{R}^{H \times N}
$$
虽然形状和传统的预测模型 $\widehat{\mathbf{Y}}_s$ 一样, 但其并不追求最小化与真实未来 return matrix $\mathbf{Y}_s$ 的 prediction loss. 该预测值不会被直接被评价, 而是进入一个 multi-period portfolio optimization 的优化问题中 (即上述 $(\text{P})$), 从而得到一个关于权重路径 $\mathbf{z}_s$ 和预测 $\widetilde{\mathbf{Y}}_s$ 的目标函数 $\widetilde{F}(\mathbf{z}_s, \widetilde{\mathbf{Y}}_s)$, 得到对应输出:
$$
\mathbf{z}_s^*(\theta) = \arg\min_{\mathbf{z}_s \in \Omega} \widetilde{F}(\mathbf{z}_s, \widetilde{\mathbf{Y}}_s) = \arg\min_{\mathbf{z}_s \in \Omega} \widehat{F}(\mathbf{z}_s, \phi_\theta(\mathbf{X}_s)).
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
& \text{s.t.} \quad \mathbf{z}_s^*(\theta) = \arg\min_{\mathbf{z}_s \in \Omega} \widehat{F}(\mathbf{z}_s, \phi_\theta(\mathbf{X}_s)), \quad s = t - T - H + 1, \ldots, t-H.
\end{aligned} \tag{IPMO}
$$

