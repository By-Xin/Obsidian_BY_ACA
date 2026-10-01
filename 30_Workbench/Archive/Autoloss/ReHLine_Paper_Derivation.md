#formulation 
# ReHLine
## Introduction
考虑如下经验风险最小问题：

$$ \min_{\boldsymbol{\beta} \in \mathbb{R}^d} \sum_{i=1}^{n} \mathcal{L}_i(\mathbf{x}_i^\top \boldsymbol{\beta}) + \frac{1}{2} \|\boldsymbol{\beta}\|_2^2, \quad \text{s.t. } \mathbf{A} \boldsymbol{\beta} + \mathbf{b} \geq \mathbf{0} \quad \text{(1)} $$

其中

+ $\mathbf{x}_i \in \mathbb{R}^d$ 是第 $i$ 个样本的特征向量
+ $\boldsymbol{\beta} \in \mathbb{R}^d$ 是待求的参数向量
+ $\mathcal{L}_i$ 是第 $i$ 个样本的损失函数, 这里要求其是 convex PLQ (piece-wise linear quadratic) 的
+ $\mathbf{A} \in \mathbb{R}^{K \times d}, \mathbf{b} \in \mathbb{R}^K$ 是约束条件
+ $\|\boldsymbol{\beta}\|_2^2$ 是参数的 L2 范数平方, 用于正则化
+ $d \ll n, K\ll n$ 是典型的情况

$\diamond$ 

尽管 $(1)$ 是一个 strongly convex + linear constraint 的问题, 其理论上具有唯一解, 但其求解有如下困难:

1. $\mathcal{L}_i$ 是一个非光滑的函数
    - 标准的 gradient-based 方法无法直接应用
    - 需要使用 subgradient 方法, 但收敛速度较慢
2. 约束条件 $\mathbf{A} \boldsymbol{\beta} + \mathbf{b} \geq \mathbf{0}$ 构成了一个多面体 (polyhedron), 其边界是一个高维的凸集
    - 基于投影的算法 (如 projected gradient descent) 计算过于复杂

$\diamond$ 

ReHLine 提出了如下三步求解方法:

1. ReLU-ReHU Decomposition: 所有 convex PLQ 的损失函数都可以分解为 ReLU 和 ReHU (ReLU with Hinge Upper bound) 的组合:

$$ L(z) = \sum_{l=1}^{L} \mathrm{ReLU}(u_l z + v_l) + \sum_{h=1}^{H} \mathrm{ReHU}_{\tau_h}(s_h z + t_h) $$

2. Box-Constrained QP Duality: 通过引入拉格朗日乘子, 将原问题转化为对偶问题, 其对偶问题是一个结构良好的含 box 约束的二次规划问题
3. Primal-Dual Algorithm: 设计一个原始-对偶算法, 通过交替优化原始问题和对偶问题, 逐步逼近最优解.

## ReLU-ReHU Decomposition
_**Definition 1 (Composite ReLU-ReHU Function)**_ 若存在向量 $\mathbf{u},\mathbf{v} \in \mathbb{R}^L$ 和 $\mathbf{s},\mathbf{t},\boldsymbol{\tau} \in \mathbb{R}^H$, 使得   

$$ \mathcal{L}(z) = \sum_{l=1}^{L} \mathrm{ReLU}(u_l z + v_l) + \sum_{h=1}^{H} \mathrm{ReHU}_{\tau_h}(s_h z + t_h) $$

则称 $\mathcal{L}(z)$ 是一个 composite ReLU-ReHU 函数, 其中 

$$ \mathrm{ReLU}(x) = \max(0,x) ,\quad \text{ReHU}_\tau(z) = \begin{cases}0, & z \leq 0 \\ \frac{z^2}{2}, & 0 < z \leq \tau \\ \tau(z - \frac{\tau}{2}), & z > \tau\end{cases}. $$

_**Proposition 1 (Closure Under Affine Transformations)**_ 设 $L(z)$ 是一个 composite ReLU-ReHU 函数, 则对于任意 $\alpha, \beta, \gamma \in \mathbb{R}$, $\alpha L(\beta z + \gamma)$ 也是一个 composite ReLU-ReHU 函数.

_**Theorem 1 (ReLU-ReHU Decomposition)**_ 任意 convex PLQ 函数 $L(z)$ 都可以分解为一个 composite ReLU-ReHU 函数, 反之亦然.

_**Example 1 (ReLU-ReHU Decomposition)**_ 许多常见的损失函数都符合 ReLU-ReHU 的形式, 如 SVM (Hinge Loss):

+ 样本标签 $y_i \in \{-1, 1\}$, 对应预测为 $z_i = \mathbf{x}_i^\top \boldsymbol{\beta}$.
+ 损失函数为 $\mathcal{L}(z_i) = c_i \max(0, 1 - y_i z_i) = \mathrm{ReLU}(- c_iy_i z_i + c_i)$, 其中 $c_i$ 是样本权重. 这是标准的 ReLU 函数形式.

_**Definition 2 (ReHLine Optimization)**_ 正式提出我们最终要求解的问题:

$$ \min_{\boldsymbol{\beta} \in \mathbb{R}^d} \left\{ \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}(u_{li} \mathbf{x}_i^\top \boldsymbol{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H \text{ReHU}_{\tau_{hi}}(s_{hi} \mathbf{x}_i^\top \boldsymbol{\beta} + t_{hi}) + \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 \right\} \\ \text{s.t. } \mathbf{A}\boldsymbol{\beta} + \mathbf{b} \geq 0 $$

其中

+ $\mathbf{U} = (u_{li}) , \mathbf{V} = (v_{li}) \in \mathbb{R}^{L \times n}$ 是 ReLU 部分的系数矩阵
+ $\mathbf{S} = (s_{hi}) , \mathbf{T} = (t_{hi}), \boldsymbol{\tau} = (\tau_{hi}) \in \mathbb{R}^{H \times n}$ 是 ReHU 部分的系数矩阵.

## Primal-Dual Formulation
已知 ReLU 和 ReHU 有如下等价表示: 

+ $\text{ReLU}(z) = \min_{\pi \ge 0} \pi, \text{ s.t. } \pi \geq z.$
+ $\text{ReHU}_\tau(z) = \min_{\sigma \geq 0, \theta} \frac{1}{2} \theta^2 + \tau \sigma ,\text{ s.t. } \theta + \sigma \geq z$

用如上等价表示重写 ReHLine 优化问题, **定义正式的 primal 问题**:

$$ \min_{\boldsymbol{\beta}, \Pi, \Theta, \Sigma} \quad \sum_{i=1}^n \sum_{l=1}^L \pi_{li} + \sum_{i=1}^n \sum_{h=1}^H \left( \frac{1}{2} \theta_{hi}^2 + \tau_{hi} \sigma_{hi} \right) +\frac{1}{2} \|\boldsymbol{\beta}\|_2^2 $$

满足 $\forall i, l, h $:

$$ \begin{aligned} &\mathbf{A} \boldsymbol{\beta} + \mathbf{b} \geq 0 \quad &\text{(原始约束)} \\ &\pi_{li} \geq u_{li} \mathbf{x}_i^\top \boldsymbol{\beta} + v_{li} \quad &\text{(ReLU)} \\ &\pi_{li} \geq 0 \\ &\theta_{hi} + \sigma_{hi} \geq s_{hi} \mathbf{x}_i^\top \boldsymbol{\beta} + t_{hi} \quad &\text{(ReHU)} \\ &\sigma_{hi} \geq 0 \end{aligned} $$

记 $\boldsymbol{\Pi} = (\pi_{li}) \in \mathbb{R}^{L \times n}, \boldsymbol{\Theta} = (\theta_{hi}) \in \mathbb{R}^{H \times n}, \boldsymbol{\Sigma} = (\sigma_{hi}) \in \mathbb{R}^{H \times n}$ 为松弛变量. 

_**下给出 dual problem.**_

首先写出 Primal Lagrangian:

$$ \begin{aligned} \mathcal{L}_{\mathcal P}(\beta, \Pi, \Theta, \Sigma; \xi, \Lambda, \Gamma,\Delta, \Psi) = & \sum_{i=1}^n \sum_{l=1}^L \pi_{li} + \sum_{i=1}^n \sum_{h=1}^H \left( \frac{1}{2} \theta_{hi}^2 + \tau_{hi} \sigma_{hi} \right) + \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 \\ & - {\xi}^\top (\mathbf{A} \boldsymbol{\beta} + \mathbf{b}) \\ & - \sum_{i,l} \lambda_{li} (\pi_{li} - u_{li} \mathbf{x}_i^\top \boldsymbol{\beta} - v_{li}) \\ & - \sum_{i,h} \gamma_{hi} (\theta_{hi} + \sigma_{hi} - s_{hi} \mathbf{x}_i^\top \boldsymbol{\beta} - t_{hi})\\ & - \sum_{i,l} \delta_{li} \pi_{li} - \sum_{i,h} \psi_{hi} \sigma_{hi}\end{aligned} $$

其中 ${\xi} \in \mathbb{R}^K$ 是拉格朗日乘子, ${\Lambda} = (\lambda_{li}) \in \mathbb{R}^{L \times n}$ 和 ${\Gamma} = (\gamma_{hi}) \in \mathbb{R}^{H \times n}$ 分别是 ReLU 和 ReHU 的拉格朗日乘子. $\Delta = (\delta_{li}) \in \mathbb{R}^{L \times n}$ 和 $\Psi = (\psi_{hi}) \in \mathbb{R}^{H \times n}$ 是松弛变量的拉格朗日乘子.

$\diamond$

接着对 Lagrangian 求关于 $(\boldsymbol{\beta}, \boldsymbol{\pi}, \boldsymbol{\theta}, \boldsymbol{\sigma})$ 的最小值, 从而构造 Dual Function $f_D(\xi, \Lambda, \Gamma) := \inf_{\boldsymbol{\beta}, \boldsymbol{\pi}, \boldsymbol{\theta}, \boldsymbol{\sigma}} \mathcal{L}(\boldsymbol{\beta}, \boldsymbol{\pi}, \boldsymbol{\theta}, \boldsymbol{\sigma}; \xi, \Lambda, \Gamma)$. 其求解方式即为分别对 $\boldsymbol{\beta}, \boldsymbol{\pi}, \boldsymbol{\theta}, \boldsymbol{\sigma}$ 求偏导数并令其为零:

_1. 对于 $\boldsymbol{\pi}$, 保留 Lagrangian 中的 $\pi_{li} \ge 0$ 项并求其偏导数:_

$$ \frac{\partial \mathcal{L}}{\partial \pi_{li}} = \frac{\partial}{\partial \pi_{li}} ( \pi_{li} - \lambda_{li} \pi_{li} - \delta_{li} \pi_{li} + \text{others}) = 1 - \lambda_{li} - \delta_{li} = 0 . $$

故 $\lambda_{li} = 1 - \delta_{li}\in [0, 1]$, 其中 $\delta_{li} \ge 0$ 是松弛变量的拉格朗日乘子. 因此有约束: 

$$ \boldsymbol{1}_{L \times n} \ge \boldsymbol{\Lambda} \ge \boldsymbol{0}_{L \times n}. $$

_2. 对于 $\boldsymbol{\theta}$,同理有:_

$$ \frac{\partial \mathcal{L}}{\partial \theta_{hi}} = \frac{\partial}{\partial \theta_{hi}} \left( \frac{1}{2} \theta_{hi}^2 - \gamma_{hi} \theta_{hi} \right) = \theta_{hi} - \gamma_{hi} = 0 ~ \Rightarrow \theta_{hi}^* = \gamma_{hi} . $$

将极小值点代回: $\frac{1}{2} \gamma_{hi}^2 - \gamma_{hi} \cdot \gamma_{hi} = -\frac{1}{2} \gamma_{hi}^2$. 因此对应在 $\mathcal{L}$ 中贡献为: 

$$ -\frac{1}{2} \sum_{i,h} \gamma_{hi}^2= -\frac{1}{2} \|\Gamma\|_F^2 = -\frac{1}{2} \text{vec}(\Gamma)^\top \text{vec}(\Gamma) $$

> + **Frobenius norm**: $\|\Gamma\|_F = \sqrt{\sum_{i,h} \gamma_{hi}^2}$ 是矩阵的 Frobenius 范数, 即矩阵每个元素的平方和开方.
> + **vec**: $\text{vec}(\Gamma)$ 是将矩阵 $\Gamma$ 按列展开成向量的操作, 即 $\text{vec}(\Gamma) = (\gamma_{11}, \gamma_{21}, \ldots, \gamma_{H1}, \gamma_{12}, \ldots, \gamma_{Hn})^\top$.
>

_3. 对于 $\boldsymbol{\sigma}$, 同理有:_

$$ \frac{\partial \mathcal{L}}{\partial \sigma_{hi}} = \frac{\partial}{\partial \sigma_{hi}} \left( \tau_{hi}\sigma_{hi} - \gamma_{hi} \sigma_{hi}- \psi_{hi}\sigma_{hi} \right) = \tau_{hi} - \gamma_{hi} - \psi_{hi} = 0~ \Rightarrow \tau_{hi}^* = \gamma_{hi}^* + \psi_{hi}^*. $$

由于 $\psi_{hi} \geq 0$, 所以 $\tau_{hi}^* \geq \gamma_{hi}^*$, 即

$$ \boldsymbol{\tau} \geq \boldsymbol{\Gamma} \geq \boldsymbol{0}_{H \times n}. $$

_4. 对于 $\boldsymbol{\beta}$_

考虑所有含 $\boldsymbol{\beta}$ 的项:

$$ \begin{aligned} \mathcal{L}_{\mathcal P}^{\boldsymbol{\beta}} &= \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 - \xi^\top A \boldsymbol{\beta} + \sum_{i,l} \lambda_{li} u_{li} \mathbf{x}_i^\top \boldsymbol{\beta} + \sum_{i,h} \gamma_{hi} s_{hi} \mathbf{x}_i^\top \boldsymbol{\beta} \end {aligned} $$

引入 tensor 记号, 记 $\mathsf{\bar{U}} = (u_{lij}):=(u_{li}x_{ij})\in \mathbb{R}^{L\times n\times d}$, $\mathsf{\bar{S}} = (s_{hij}):=(s_{hi}x_{ij})\in \mathbb{R}^{H\times n\times d}$ 分别为 ReLU 和 ReHU 的系数张量, 则其 mode-3 unfolding 分别为 $\mathsf{\bar{U}}_{(3)} \in\mathbb{R}^{d\times nL}$ 和 $\mathsf{\bar{S}}_{(3)} \in \mathbb{R}^{d\times nH}$. 则有如下恒等式:

$$ \sum_{l,i}\lambda_{li}u_{li} \mathbf{x}_i^\top = \mathsf{\bar{U}}_{(3)}\text{vec}(\boldsymbol{\Lambda}) ,\quad \sum_{h,i}\gamma_{hi}s_{hi} \mathbf{x}_i^\top = \mathsf{\bar{S}}_{(3)} \text{vec}(\boldsymbol{\Gamma}) . $$

因此 Lagrangian 中含 $\boldsymbol{\beta}$ 的部分可以写成:

$$ \begin{aligned}\mathcal{L}_{\mathcal P}^{\boldsymbol{\beta}}&= \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 - \xi^\top \mathbf{A} \boldsymbol{\beta} + \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}) \boldsymbol{\beta} + \mathsf{\bar{S}}_{(3)} \text{vec}(\boldsymbol{\Gamma}) \boldsymbol{\beta} \\&= \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 - (\xi^\top \mathbf{A} + \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}) + \mathsf{\bar{S}}_{(3)} \text{vec}(\boldsymbol{\Gamma}))\boldsymbol{\beta} \end{aligned} $$

对 $\boldsymbol{\beta}$ 求偏导数并令其为零:

$$ \boldsymbol{\beta}^* = \mathbf{A}^\top \xi - \sum_{i=1}^n\Bigl(\sum_{l}\lambda_{li}u_{li}\;+\;\sum_{h}\gamma_{hi}s_{hi}\Bigr)\mathbf{x}_i^\top = \mathbf{A}^\top\xi + \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}) + \mathsf{\bar{S}}_{(3)} \text{vec}(\boldsymbol{\Gamma}) $$

再将极小值点代回 $\mathcal{L}_{\mathcal P}^{\boldsymbol{\beta}}$ 中:

$$ \begin{aligned}\mathcal{L}_{\mathcal P}^{\boldsymbol{\beta}^*} &= \frac{1}{2} \|\boldsymbol{\beta}^*\|_2^2 - \boldsymbol{\beta}^{*\top} g^* =-\frac{1}{2} \|\boldsymbol{\beta}^*\|_2^2 \\&= \frac{1}{2} \left\|\mathbf{A}^\top \xi - \sum_{i,l}\lambda_{li}u_{li}\mathbf{x}_i^\top\;+\;\sum_{i,h}\gamma_{hi}s_{hi} \mathbf{x}_i^\top \right\|_2^2 \\&= \frac{1}{2} \left(\xi^\top\mathbf{A}\mathbf{A}^\top\xi + \sum_{l,i}\sum_{l',i'}\lambda_{li}u_{li}\lambda_{l'i'}u_{l'i'}\mathbf{x}_i^\top\mathbf{x}_{i'}+\sum_{h,i}\sum_{h',i'}\gamma_{hi}s_{hi}\gamma_{h'i'}s_{h'i'}\mathbf{x}_i^\top\mathbf{x}_{i'}\right) + \\&\quad\xi^\top\mathbf{A}\sum_{l,i}\lambda_{li}u_{li}\mathbf{x}_i^\top - \xi^\top\mathbf{A}\sum_{h,i}\gamma_{hi}s_{hi}\mathbf{x}_i^\top +\sum_{l,i}\sum_{h',i'}\lambda_{li}u_{li}\gamma_{h'i'}s_{h'i'}\mathbf{x}_i^\top\mathbf{x}_{i'} \\&= \frac{1}{2} \xi^\top\mathbf{A}\mathbf{A}^\top\xi + \frac{1}{2} \text{vec}(\boldsymbol{\Lambda})^\top \mathsf{\bar{U}}_{(3)}^\top \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}) + \frac{1}{2} \text{vec}(\boldsymbol{\Gamma})^\top \mathsf{\bar{S}}_{(3)}^\top \mathsf{\bar{S}}_{(3)} \text{vec}(\boldsymbol{\Gamma})\\&\quad -\xi^\top\mathbf{A}\mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}) - \xi^\top\mathbf{A}\mathsf{\bar{S}}_{(3)} \text{vec}(\boldsymbol{\Gamma}) + \text{vec}(\boldsymbol{\Lambda})^\top \mathsf{\bar{U}}_{(3)}^\top \mathsf{\bar{S}}_{(3)} \text{vec}(\boldsymbol{\Gamma}) \end{aligned} $$

$\diamond$

分别将 $\boldsymbol{\beta}^*$, $\boldsymbol{\theta}^*$ 代回上述 Primal Lagrangian $\mathcal{L}_{\mathcal P}(\boldsymbol{\beta}, \boldsymbol{\pi}, \boldsymbol{\theta}, \boldsymbol{\sigma}; \xi, \Lambda, \Gamma)$ 中, 得到 Dual Lagrangian:

$$ \begin{aligned}\mathcal{L}_D =\ & \frac{1}{2} \left( \sum_{k=1}^K \sum_{k'=1}^K \xi_k \xi_{k'} \mathbf{a}_k^\top \mathbf{a}_{k'} + \sum_{l,i} \sum_{l',i'} \lambda_{li} \lambda_{l'i'} u_{li} u_{l'i'} \mathbf{x}_i^\top \mathbf{x}_{i'} + \sum_{h,i} \sum_{h',i'} \gamma_{hi} \gamma_{h'i'} s_{hi} s_{h'i'} \mathbf{x}_i^\top \mathbf{x}_{i'} \right) \\& - \sum_{k=1}^K \sum_{l,i} \xi_k \lambda_{li} u_{li} \mathbf{a}_k^\top \mathbf{x}_i - \sum_{k=1}^K \sum_{h,i} \xi_k \gamma_{hi} s_{hi} \mathbf{a}_k^\top \mathbf{x}_i + \sum_{l,i} \sum_{h',i'} \lambda_{li} u_{li} \gamma_{h'i'} s_{h'i'} \mathbf{x}_i^\top \mathbf{x}_{i'} \\& - \frac{1}{2} \sum_{h,i} \gamma_{hi}^2 + \sum_{k=1}^K \xi_k b_k - \sum_{l,i} \lambda_{li} v_{li} - \sum_{h,i} \gamma_{hi} t_{hi} \\=\ & \red {\frac{1}{2} \xi^\top \mathbf{A} \mathbf{A}^\top \xi + \frac{1}{2} \mathrm{vec}(\boldsymbol{\Lambda})^\top \bar{\mathsf{U}}_{(3)}^\top \bar{\mathsf{U}}_{(3)} \mathrm{vec}(\boldsymbol{\Lambda}) + \frac{1}{2} \mathrm{vec}(\boldsymbol{\Gamma})^\top \bar{\mathsf{S}}_{(3)}^\top \bar{\mathsf{S}}_{(3)} \mathrm{vec}(\boldsymbol{\Gamma}) +}\\& \red{- \xi^\top \mathbf{A} \bar{\mathsf{U}}_{(3)} \mathrm{vec}(\boldsymbol{\Lambda}) - \xi^\top \mathbf{A} \bar{\mathsf{S}}_{(3)} \mathrm{vec}(\boldsymbol{\Gamma}) + \mathrm{vec}(\boldsymbol{\Lambda})^\top \bar{\mathsf{U}}_{(3)} \bar{\mathsf{S}}_{(3)} \mathrm{vec}(\boldsymbol{\Gamma})} \\& \blue{-\frac{1}{2} \mathrm{vec}(\boldsymbol{\Gamma})^\top\mathrm{vec}(\boldsymbol{\Gamma})}\\& + \xi^\top \mathbf{b} - \mathrm{Tr}(\boldsymbol{\Lambda} \mathbf{V}^\top) - \mathrm{Tr}(\mathbf{T} \mathbf{T}^\top)\end{aligned} $$

其中最后三项 $\xi^\top \mathbf{b} - \mathrm{Tr}(\boldsymbol{\Lambda} \mathbf{V}^\top) - \mathrm{Tr}(\mathbf{T} \mathbf{T}^\top)$ 是关于 primal 变量的无关常数项 (也就是在上述四次求导中本身都没有涉及到的项, 自然地就留在式子中).

且应满足如下约束:

$$ \begin{aligned}&\boldsymbol{1}_{L \times n} \geq \boldsymbol{\Lambda} \geq \boldsymbol{0}_{L \times n} \\&\boldsymbol{\tau} \geq \boldsymbol{\Gamma} \geq \boldsymbol{0}_{H \times n}\\&\boldsymbol{\xi} \geq \boldsymbol{0}_{K \times n} \\\end{aligned} $$

---

整理一下这个优化问题的 KKT 条件. 设 $(\boldsymbol{\beta}^\star,\Pi^\star,\Theta^\star,\Sigma^\star;\, \boldsymbol{\xi}^\star,\boldsymbol{\Lambda}^\star, \boldsymbol{\Delta}^\star,\boldsymbol{\Gamma}^\star,\boldsymbol{\Psi}^\star)$ 是 primal problem 的一组可行点, 则其是最优解当且仅当满足以下 KKT 条件:

_Primal Feasibility_:

$$ \begin{cases}\mathbf{A}\boldsymbol{\beta}^\star+\mathbf{b}\;\ge\;\mathbf{0},\\\pi_{li}^\star\;\ge\;u_{li}\mathbf{x}_i^\top\boldsymbol{\beta}^\star+v_{li},\\\pi_{li}^\star\;\ge\;0,\\\theta_{hi}^\star+\sigma_{hi}^\star\;\ge\;s_{hi}\mathbf{x}_i^\top\boldsymbol{\beta}^\star+t_{hi},\\\sigma_{hi}^\star\;\ge\;0.\end{cases} $$

_Dual Feasibility_:

$$ \boldsymbol{\xi}^\star\ge0,\quad \boldsymbol{\Lambda}^\star\ge0,\quad \boldsymbol{\Delta}^\star\ge0,\quad \boldsymbol{\Gamma}^\star\ge0,\quad \boldsymbol{\Psi}^\star\ge0 $$

_Stationarity_:

$$ \boxed{\begin{aligned}\frac{\partial\mathcal{L}}{\partial\boldsymbol{\beta}} \;=\;& \boldsymbol{\beta}^\star - \mathbf{A}^{\top}\boldsymbol{\xi}^\star +\sum_{i,l}\lambda_{li}^\star u_{li}\mathbf{x}_{i}+\sum_{i,h}\gamma_{hi}^\star s_{hi}\mathbf{x}_{i} \;=\;\mathbf{0};\\[4pt]\frac{\partial\mathcal{L}}{\partial\pi_{li}} \;=\;&1-\lambda_{li}^\star-\delta_{li}^\star \;=\;0\;\;\Longrightarrow\;\; 0\le\lambda_{li}^\star\le1;\\[4pt]\frac{\partial\mathcal{L}}{\partial\theta_{hi}} \;=\;&\theta_{hi}^\star-\gamma_{hi}^\star\;=\;0\;\;\Longrightarrow\;\; \theta_{hi}^\star=\gamma_{hi}^\star;\\[4pt]\frac{\partial\mathcal{L}}{\partial\sigma_{hi}} \;=\;&\tau_{hi}-\gamma_{hi}^\star-\psi_{hi}^\star \;=\;0\;\;\Longrightarrow\;\; 0\le\gamma_{hi}^\star\le\tau_{hi}.\end{aligned}} $$

且有

$$ \boldsymbol{\beta}^\star=\mathbf{A}^{\top}\boldsymbol{\xi}^\star-\sum_{i,l}\lambda_{li}^\star u_{li}\mathbf{x}_{i}-\sum_{i,h}\gamma_{hi}^\star s_{hi}\mathbf{x}_{i} $$

_Complementary Slackness_:

$$ \boxed{\begin{aligned}&\xi_k^\star\,(A\boldsymbol{\beta}^\star+b)_k \;=\;0, &&k=1,\dots,K;\\[2pt]&\lambda_{li}^\star\bigl(\pi_{li}^\star-u_{li}\mathbf{x}_i^\top\boldsymbol{\beta}^\star-v_{li}\bigr)=0,&&l=1,\dots,L,\;i=1,\dots,n;\\[2pt]&\delta_{li}^\star\,\pi_{li}^\star = 0,&&l=1,\dots,L,\;i=1,\dots,n;\\[2pt]&\gamma_{hi}^\star\bigl(\theta_{hi}^\star+\sigma_{hi}^\star-s_{hi}\mathbf{x}_i^\top\boldsymbol{\beta}^\star-t_{hi}\bigr)=0,&&h=1,\dots,H,\;i=1,\dots,n;\\[2pt]&\psi_{hi}^\star\,\sigma_{hi}^\star = 0,&&h=1,\dots,H,\;i=1,\dots,n.\end{aligned}} $$

## Solving the Dual Problem by Coordinate Descent
在成功地构建起 primal 和 dual 问题后, 我们可以通过对偶问题的优化来求解原始问题. 根据优化理论, 在当前强凸的 primal problem 上, dual 问题的最优解等价于 primal 问题的最优解, 即

$$ \min_{\text{primal}} f(\boldsymbol{\beta}) = \max_{\text{dual}} f_D(\xi, \Lambda, \Gamma) $$

更进一步, primal 解 $\boldsymbol{\beta}^*$ 和 dual 解 $(\xi^*, \boldsymbol{\Lambda}^*, \boldsymbol{\Gamma}^*)$ 之间还通过 KKT 条件彼此关联:

$$ \boldsymbol{\beta}^* = \mathbf{A}^\top \xi^* + \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}^*) + \mathsf{\bar{S}}_{(3)} \text{vec}(\boldsymbol{\Gamma}^*) $$

因此理论上, 只要我们求得了 $(\xi^*, \boldsymbol{\Lambda}^*, \boldsymbol{\Gamma}^*) = \arg\max_{\xi, \Lambda, \Gamma} f_D(\xi, \Lambda, \Gamma)$ 就可以直接得到 primal 解 $\boldsymbol{\beta}^*$. 另一个需要注意的地方, 在实践中我们往往取标准意义上 dual problem 的相反数作为目标函数, 因此实践中我们是在进行 $\min_{\xi, \Lambda, \Gamma} -f_D(\xi, \Lambda, \Gamma)$ 的优化.

不过想要直接求解对偶问题 $f_D(\xi, \Lambda, \Gamma)$ 的最大值并不容易, 因此这里引入 coordinate descent 方法, 通过对偶问题的坐标下降来求解. 简单来说, 坐标下降方法就是在每次迭代中, 固定其他变量, 只优化一个变量的值, 直到收敛. 在 ReHLine 中, 我们将对偶问题的三个变量 $\xi$, $\Lambda$, $\Gamma$ 分别作为坐标进行优化; 并且每次在优化一个坐标后, 同步更新 primal 解 $\boldsymbol{\beta}$.

具体流程如下. 

**Requirements**  
首先需要获得数据 $\mathbf{X} \in \mathbb{R}^{n \times d}$, 约束矩阵 $\mathbf{A} \in \mathbb{R}^{K \times d}$ 和偏置向量 $\mathbf{b} \in \mathbb{R}^K$.

**Step 1: Initialization**

设定初始化: $\xi^0 = \mathbf{0}$, $\boldsymbol{\Lambda}^0 = \mathbf{0}$, $\boldsymbol{\Gamma}^0 = \mathbf{0}$.

根据 KKT 条件, 初始化 primal 解 $\boldsymbol{\beta}^0$:

$$ \boldsymbol{\beta}^0 = \mathbf{A}^\top \xi^0 + \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}^0) + \mathsf{\bar{S}}_{(3)} \text{vec}(\boldsymbol{\Gamma}^0) = \mathbf{0}. $$

**Step 2: Update $\xi_k \leftrightarrow \boldsymbol{\beta}$**

在后面的更新中, 都是根据按照 $\text{dual} \leftrightarrow \text{primal}$ 的关系, 先更新 dual 变量, 再更新 primal 变量进行的. 具体地, 论文中的更新顺序为: $(\xi_k \leftrightarrow \boldsymbol{\beta}) \to (\lambda_{li} \leftrightarrow \boldsymbol{\beta}) \to (\gamma_{hi} \leftrightarrow \boldsymbol{\beta})$. 也就是每更新一个最基本的 dual 变量, 都会更新一次 primal 解 $\boldsymbol{\beta}$, 直到当前的 dual 变量更新完毕, 再更新下一个 dual 变量.

这里直接给出 $\xi_k$ 的更新规则. 具体的推导详见 $\lambda_{li}$ 部分. 事实上这三者都是类似的.

$$ \xi_k^{\text{new}} \gets \mathcal{P}_{[0, +\infty)} \left( \xi_k^{\text{old}} - \frac{\nabla_{\xi_k} \mathcal{L}(\xi^{\text{old}})}{\lVert \mathbf{a}_k \rVert_2^2} \right) = \max \left( 0, \xi_k^{\text{old}} - \frac{\mathbf{a}_k^\top \boldsymbol{\beta}^{\text{old}} + b_k}{\lVert \mathbf{a}_k \rVert_2^2} \right) $$

$$ \boldsymbol{\beta}^{\text{new}} \gets \boldsymbol{\beta}^{\text{old}} + (\xi_k^{\text{new}} - \xi_k^{\text{old}})\mathbf{a}_k. $$

**Step 2: Update $\lambda_{li} \leftrightarrow \boldsymbol{\beta}$**  
在每次迭代中, 固定 $\xi$ 和 $\Gamma$, 只优化 $\Lambda$ 的值. 具体来说, 求解如下问题:

$$ \begin{aligned} \boldsymbol{\Lambda}^{\text{new}} &= \arg\min_{\boldsymbol{\Lambda}^{\text{old}}} \left\{ \mathcal L_D(\xi^{\text{old}}, \boldsymbol{\Lambda}^{\text{old}}, \boldsymbol{\Gamma}^{\text{old}}) \right\} \\ \end{aligned} $$

回顾 $\mathcal{L}_D(\xi, \Lambda, \Gamma)$ 的表达式, 保留关于 $\lambda_{li}$ 的项, 记为 $\mathcal{L}_D^{\lambda_{li}}$:

$$ \begin{aligned}\mathcal{L}_D^{\lambda_{li}} &= \frac{1}{2} u_{li}^2 (\mathbf{x}_i^\top \mathbf{x}_i) \lambda_{li}^2 + \sum_{(l',i') \ne (l,i)} \lambda_{l'i'} u_{l'i'} u_{li} (\mathbf{x}_{i'}^\top \mathbf{x}_i) \lambda_{li} \\& - \sum_{k=1}^{K} \xi_k u_{li} (\mathbf{a}_k^\top \mathbf{x}_i) \lambda_{li} + \sum_{h', i'} u_{li} \gamma_{h'i'} s_{h'i'} \mathbf{x}_i^\top \mathbf{x}_{i'} \lambda_{li} - v_{li} \lambda_{li}.\end{aligned} $$

为求最小值, 对 $\mathcal{L}_D^{\lambda_{li}}$ 关于 $\lambda_{li}$ 求导并令其为零:

$$ \begin{aligned} \frac{\partial \mathcal{L}_D^{\lambda_{li}}}{\partial \lambda_{li}} &= u_{li}^2 (\mathbf{x}_i^\top \mathbf{x}_i) \lambda_{li} + \sum_{(l',i') \ne (l,i)} \lambda_{l'i'} u_{l'i'} u_{li} (\mathbf{x}_{i'}^\top \mathbf{x}_i) \\&\quad - \sum_{k=1}^{K} \xi_k u_{li} (\mathbf{a}_k^\top \mathbf{x}_i) + \sum_{h', i'} u_{li} \gamma_{h'i'} s_{h'i'} \mathbf{x}_i^\top \mathbf{x}_{i'} - v_{li} = 0. \end{aligned} $$

这是一个关于 $\lambda_{li}$ 的一次方程, 可以直接求解:

$$ \begin{aligned}\lambda_{li} = \frac{u_{li} \mathbf{x}_i^\top \left( \sum_{k=1}^{K} \xi_k \mathbf{a}_k - \sum_{(l', i') \ne (l, i)} \lambda_{l'i'} u_{l'i'} \mathbf{x}_{i'} - \sum_{h', i'} \gamma_{h'i'} s_{h'i'} \mathbf{x}_{i'} \right) + v_{li}}{ u_{li}^2 \left\| \mathbf{x}_i \right\|_2^2 } \end{aligned} $$

不过考虑到 $\lambda_{li}$ 的约束条件 $0 \leq \lambda_{li} \leq 1$, 我们需要对求解结果进行截断, 引入截断函数: 

$$ \mathcal{P}_{[a,b]}(x) = \min\{b, \max\{a, x\}\} = \begin{cases} a, & x < a \\ x, & a \leq x \leq b \\ b, & x > b \end{cases} $$

则理论上的更新规则为 (不过依然不是最终采用的规则, 后面会通过迭代增量更新进一步减少计算):

$$ \begin{aligned} \lambda_{li}^{\text{new}} &\gets \mathcal{P}_{[0, 1]} \left( \frac{u_{li} \mathbf{x}_i^\top \left( \sum_{k=1}^{K} \xi_k \mathbf{a}_k - \sum_{(l', i') \ne (l, i)} \lambda_{l'i'} u_{l'i'} \mathbf{x}_{i'} - \sum_{h', i'} \gamma_{h'i'} s_{h'i'} \mathbf{x}_{i'} \right) + v_{li}}{u_{li}^2 \left\| \mathbf{x}_i \right\|_2^2} \right) \end{aligned} $$

更进一步地, 我们想要利用 $\boldsymbol{\beta}$ 和 $\lambda_{li}$ 之间的关系, 来减少计算量.

这里记 $\boldsymbol{\mu} := (\boldsymbol{\xi}, \boldsymbol{\Lambda}, \boldsymbol{\Gamma})$ 为当前的对偶变量, 回顾在 KKT 条件下, 有:

$$ \boldsymbol{\beta} := \boldsymbol{\beta}(\boldsymbol{\mu}) = \sum_{k=1}^K \xi_k \mathbf{a}_k - \sum_{(l,i)} \lambda_{li} u_{li} \mathbf{x}_i - \sum_{(h,i)} \gamma_{hi} s_{hi} \mathbf{x}_i. $$

另一方面回顾 $\mathcal{L}_D(\boldsymbol{\mu})$ 关于 $\lambda_{li}$ 的导数:

$$ \begin{aligned}\frac{\partial \mathcal{L}_D^{\lambda_{li}}}{\partial \lambda_{li}} &= u_{li} \mathbf{x}_i^\top \left( \sum_{k=1}^{K} \xi_k \mathbf{a}_k - \sum_{(l', i') \ne (l, i)} \lambda_{l'i'} u_{l'i'} \mathbf{x}_{i'} - \sum_{h', i'} \gamma_{h'i'} s_{h'i'} \mathbf{x}_{i'} \right) + v_{li}-\lambda_{li} u_{li} \|\mathbf{x}_i\|_2^2. \end{aligned} $$

观察二者关系, 该导数实际上可以由 $\boldsymbol{\beta}$ 表示:

$$ \begin{aligned}\frac{\partial \mathcal{L}_D^{\lambda_{li}}}{\partial \lambda_{li}} &= -(u_{li} \mathbf{x}_i^\top \boldsymbol{\beta} + v_{li}) .\end{aligned} $$

因此尽管我们可以用上述理论的 $\lambda$ 更新公式直接进行迭代, 一个更便捷的方式是通过 当前的 $\boldsymbol{\beta}$ 来更新 $\lambda_{li}$. 因此若记 $\boldsymbol{\beta}^\text{old} = \boldsymbol{\beta}(\boldsymbol{\mu}^\text{old}) = \boldsymbol{\beta}(\xi^\text{old}, \boldsymbol{\Lambda}^\text{old}, \boldsymbol{\Gamma}^\text{old})$ 为当前的 primal 解, 则可以直接用以下公式更新 $\lambda_{li}$:

$$ \begin{aligned}\lambda_{li}^{\text{new}} &\gets \mathcal{P}_{[0, 1]} \left( \lambda_{li}^\text{old} + \frac{u_{li} \mathbf{x}_i^\top \boldsymbol{\beta}^\text{old} + v_{li}}{u_{li}^2 \|\mathbf{x}_i\|_2^2} \right) \end{aligned} $$

并且对应地,

$$ \boldsymbol{\beta}^\text{new} \gets \boldsymbol{\beta}^\text{old} +(\lambda_{li}^{\text{new}} - \lambda_{li}^\text{old}) u_{li} \mathbf{x}_i. $$

**Step 3: Update $\gamma_{hi} \leftrightarrow \boldsymbol{\beta}$**

$$ \gamma_{hi}^{\text{new}} \gets \mathcal{P}_{[0, \tau_{hi}]} \left( \gamma_{hi}^{\text{old}} - \frac{\nabla_{\gamma_{hi}} \mathcal{L}(\gamma^{\text{old}})}{s_{hi}^2 \lVert \mathbf{x}_i \rVert_2^2 + 1} \right) = \max\left( 0, \min\left( \tau_{hi}, \gamma_{hi}^{\text{old}} + \frac{s_{hi} \mathbf{x}_i^\top \boldsymbol{\beta}^{\text{old}} + t_{hi} - \gamma_{hi}^{\text{old}}}{s_{hi}^2 \lVert \mathbf{x}_i \rVert_2^2 + 1} \right) \right), $$

$$ \boldsymbol{\beta}^{\text{new}} \gets \boldsymbol{\beta}^{\text{old}} - (\gamma_{hi}^{\text{new}} - \gamma_{hi}^{\text{old}}) s_{hi} \mathbf{x}_i. $$

**Step 4: Convergence Check**

对于该算法的收敛性, 文章给出了如下定理:  
_**Theorem 3**_: 对于当前第 $q$ 次迭代, 得到 $\boldsymbol{\mu}^q = (\xi^q, \boldsymbol{\Lambda}^q, \boldsymbol{\Gamma}^q)$, 假设存在 dual 的一个最优解 $\boldsymbol{\mu}^* = (\xi^*, \boldsymbol{\Lambda}^*, \boldsymbol{\Gamma}^*)$. 则可以断言: 存在常数 $0<\eta<1$ 和 $q_0 \in \mathbb{N}$, 使得对于所有 $q \geq q_0$, 有:

$$ \mathcal{L}_D(\boldsymbol{\mu}^q) - \mathcal{L}_D(\boldsymbol{\mu}^*) \leq \eta \left( \mathcal{L}_D(\boldsymbol{\mu}^0) - \mathcal{L}_D(\boldsymbol{\mu}^*) \right). $$

即具有全局线性收敛速率（global linear convergence rate）

---

综上, 文中给出下图所示的 ReHLine 更新算法:

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250708202850.png)