#formulation 
考虑如下场景. 对于一组数据, 特征集 $\mathcal{X}\in \mathbb{R}^{n\times d}$, 标签集 $\mathcal{Y} \in \mathbb{R}^{n}$, 其中 $n$ 是样本数, $d$ 是特征维度. 单个观测样本记为 $\mathbf{x}^{(i)} \in \mathbb{R}^{d}$, 其对应的标签为 $y^{(i)} \in \mathcal{Y}$.

引入 ReLU 和 ReHU 两个函数:

$$ \begin{align*} \text{ReLU}(z) &= \max(0, z) \\ \text{ReHU}_\tau(z) &= \begin{cases} 0 & \text{if } z \leq 0 \\ z^2/2 & \text{if } 0 < z \leq \tau \\ \tau z - \tau^2/2 & \text{if } \tau < z \end{cases} \end{align*} $$

其中 $\tau > 0$ 是一个超参数. ReHU 函数在 $z \leq 0$ 时为 $0$, 在 $(0,\tau]$ 时为二次函数, 在 $z > \tau$ 时为线性函数. 

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250612155727.png)

---

_**Regression**_

在线性回归中, 我们假设标签 $y^{(i)} \in \mathbb{R}$ 与特征之间存在线性关系, 即 $\boldsymbol{\beta} \in \mathbb{R}^{d}$ ：$\hat y^{(i)} = \mathbf{x}^{(i) \top} \boldsymbol{\beta}$ 
则每个观测的残差为: $r^{(i)} = \hat y^{(i)} - \mathbf{x}^{(i) \top} \boldsymbol{\beta}$.

**[定义]**(单笔观测的)损失函数为:

$$ \ell^{(i)}(r^{(i)};\boldsymbol{\beta}) = \sum_{l=1}^L \text{ReLU}(U_l^{(i)} r^{(i)} + V_l^{(i)}) + \sum_{h=1}^H \text{ReHU}_\tau(S_h^{(i)} r^{(i)} + T_h^{(i)}) + \lambda \|\boldsymbol{\beta}\|_2^2 $$

+ 其中 $L,H \in \mathbb{R}$ 为提前指定的超参数, 决定 ReLU 和 ReHU 的个数. 
+ $U_l^{(i)}, V_l^{(i)}, S_h^{(i)}, T_h^{(i)} \in \mathbb{R}$ 是该笔观测对应的损失函数的参数, 用来调整 ReLU 和 ReHU 的形状. 
+ $\lambda \in \mathbb{R}$ 是正则化参数, 用于防止过拟合.
+ $\boldsymbol{\beta}\in\mathbb{R}^{d}$ 是回归系数, 需要通过训练来学习.
+ $r^{(i)} = y^{(i)} - \mathbf{x}^{(i) \top} \boldsymbol{\beta}$ 是残差.

根据 ReHLine 的相关研究:

+ $\text{ReLU}(U_l^{(i)} r^{(i)} + V_l^{(i)})$ 等价于 

$$ \begin{align*} \min_\pi \quad \pi_l^{(i)} &,\\ \text{s.t.}\quad \pi_l^{(i)} &\geq U_l^{(i)} r^{(i)} + V_l^{(i)} = U_l^{(i)} y^{(i)} - U_l^{(i)}\mathbf{x}^{(i) \top} \boldsymbol{\beta} + V_l^{(i)} , \\ \pi_l^{(i)} &\ge 0. \end{align*} $$

+ $\text{ReHU}_\tau(S_h^{(i)} r^{(i)} + T_h^{(i)})$ 等价于

$$ \begin{align*} \min_{\vartheta,\sigma}& \quad \frac{1}{2} (\vartheta_h^{(i)})^2 + \tau \sigma_h^{(i)} ,\\ \text{s.t.}& \quad \vartheta_h^{(i)}+\sigma_h^{(i)} \geq S_h^{(i)} r^{(i)} + T_h^{(i)} = S_h^{(i)} y^{(i)} - S_h^{(i)}\mathbf{x}^{(i) \top} \boldsymbol{\beta} + T_h^{(i)}, \\ &\quad \sigma_h^{(i)} \ge 0. \\ \end{align*} $$

因此, 整体损失函数可以写成如下形式:

$$ \begin{align*} \min_{\boldsymbol{\beta}, \pi, \vartheta, \sigma} &\quad \sum_{i=1}^n \left( \sum_{l=1}^L \pi_l^{(i)} + \sum_{h=1}^H \left(\frac{1}{2} (\vartheta_h^{(i)})^2 + \tau \sigma_h^{(i)}\right) + \lambda \|\boldsymbol{\beta}\|_2^2 \right),\\ \text{s.t.} &\quad \pi_l^{(i)} \geq U_l^{(i)} y^{(i)} - U_l^{(i)}\mathbf{x}^{(i) \top} \boldsymbol{\beta} + V_l^{(i)}, \\ &\quad \pi_l^{(i)} \ge 0, \\ &\quad \vartheta_h^{(i)}+\sigma_h^{(i)} \geq S_h^{(i)} y^{(i)} - S_h^{(i)}\mathbf{x}^{(i) \top} \boldsymbol{\beta} + T_h^{(i)}, \\ &\quad \sigma_h^{(i)} \ge 0, \\ &\quad \forall i = 1, \ldots, n; l = 1, \ldots, L; h = 1, \ldots, H. \end{align*} $$

整理成小于等于的标准形式:

$$ \begin{align*} \min_{\boldsymbol{\beta}, \pi, \vartheta, \sigma} &\quad \sum_{i=1}^n \left( \sum_{l=1}^L \pi_l^{(i)} + \sum_{h=1}^H \left(\frac{1}{2} (\vartheta_h^{(i)})^2 + \tau \sigma_h^{(i)}\right) + \lambda \|\boldsymbol{\beta}\|_2^2 \right),\\ \text{s.t.} \quad -U_l^{(i)}\mathbf{x}^{(i) \top} \boldsymbol{\beta} - \pi_l^{(i)}+0+0 &\leq -U_l^{(i)} y^{(i)} - V_l^{(i)}, \\ \quad 0-\pi_l^{(i)}+0+0 &\le 0, \\ \quad -S_h^{(i)}\mathbf{x}^{(i) \top} \boldsymbol{\beta} +0- \vartheta_h^{(i)} - \sigma_h^{(i)} &\leq -S_h^{(i)} y^{(i)} - T_h^{(i)}, \\ \quad 0+0+0-\sigma_h^{(i)} &\le 0, \\ \quad \forall i = 1, \ldots, n;~ l = 1, &\ldots, L; ~h = 1, \ldots, H. \end{align*} $$

---

_**Binary Classification**_

在二分类问题中, 假设标签 $y^{(i)} \in \{-1, 1\}$, 我们的判别函数为 $\hat y^{(i)} = \text{sign}(\mathbf{x}^{(i) \top} \boldsymbol{\beta})$, 则对应的 margin 为 $m^{(i)} = y^{(i)} \mathbf{x}^{(i) \top} \boldsymbol{\beta}$.

可以相应推导出转化后的损失函数:

$$ \begin{align*} \min_{\boldsymbol{\beta}, \pi, \vartheta, \sigma} &\quad \sum_{i=1}^n \left( \sum_{l=1}^L \pi_l^{(i)} + \sum_{h=1}^H \left(\frac{1}{2} (\vartheta_h^{(i)})^2 + \tau \sigma_h^{(i)}\right) + \lambda \|\boldsymbol{\beta}\|_2^2 \right),\\ \text{s.t.} &\quad U_l^{(i)}y^{(i)}\mathbf{x}^{(i) \top} \boldsymbol{\beta} - \pi_l^{(i)}+0+0 \leq - V_l^{(i)}, \\ &\quad 0-\pi_l^{(i)}+0+0 \le 0, \\ &\quad S_h^{(i)}y^{(i)}\mathbf{x}^{(i) \top} \boldsymbol{\beta} +0- \vartheta_h^{(i)} - \sigma_h^{(i)} \leq - T_h^{(i)}, \\ &\quad 0+0+0-\sigma_h^{(i)} \le 0, \\ &\quad \forall i = 1, \ldots, n;~ l = 1, \ldots, L; ~h = 1, \ldots, H. \end{align*} $$