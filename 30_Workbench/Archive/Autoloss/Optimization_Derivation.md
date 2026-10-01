---
aliases: [Autoloss优化推导, Bilevel Optimization, 双层优化]
tags:
  - proof
  - proj/Autoloss
  - math/optimization
related_concepts:
  - "[[Implicit Function Theorem]]"
  - "[[Lagrange Duality]]"
---

#formulation
# Autoloss 优化部分推导 (diffopt 文档解读)

## 背景

### 原始优化问题

给定一个凸优化问题

$$ \min_{w\in C} f(w;\theta) \tag{1} $$

其目标函数 $f(w;\theta)$ 是关于参数 $w$ 的凸函数, 而 $w$ 本身又将依赖于一些超参数 $\theta$. 因此优化目标 $w^* = w(\theta)$ 是一个关于超参数 $\theta$ 的隐函数 $\min_{w\in C} f(w(\theta);\theta)$ (由于通常我们没有 $w(\theta)$ 的显式表达式, 因此需要通过优化算法来求解).

如果对应在本 Autoloss 问题中, $f$ 对应我们定义的损失函数

$$ \ell(r )=\ell(y - x^\top\boldsymbol{\beta}) = \sum_{l=1}^L \text{ReLU}(U_l r + V_l) + \sum_{h=1}^H \text{ReHU}_\tau(S_h r + T_h) + \lambda_{\text{reg}} \|\boldsymbol{\beta}\|_2^2 $$

+ 其中 $w$ 对应模型参数 (如 $\beta$), 而 $\theta$ 则对应损失函数的超参数 (如 $U,V,S,T$). 对于每组超参数 $\theta = (U,V,S,T)$ 就可以定义出一个损失函数, 也就对应着一个当前损失函数下的最优参数 $w^* (\theta)= \beta^*(U,V,S,T)$.

+ $\lambda_{\text{reg}}$ 是正则化参数, 由于后面的 Lagrangian 中也有同样的符号, 因此需要注意区分.

+ 并且不难看出当前 $\beta$ 虽然是与 $U,V,S,T$ 有关的, 但是一种隐式关系.

### 双层优化系统

目前, 我们通过一个双层优化系统来进行求解:

$$ \min_\theta \mathcal{L}_{\text{val}}\left(w^*(\theta)\right), \quad \text{where } w^*(\theta) = \arg\min_w f(w;\theta) $$

有时候也可以写成:

$$ \min_{\theta} \; \mathcal{L}_{\text{val}}(w^*(\theta)) \quad \text{s.t.} \quad w^*(\theta) = \arg\min_{w} \mathcal{L}_{\text{train}}(w; \theta) $$

+ 具体地, `qpth` 来求解给定 $\theta=(U,V,S,T)$ 时的最优参数 $w^* = \beta^*(U,V,S,T)$, 而再通过定义一个外层的损失函数(例如 MSE) $\mathcal{L}( y, x^\top\beta(\theta))$ 来对超参通过梯度下降进行优化.

$$ \min_{\theta} \;\; \mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{val}}} \, \mathcal L(f_{\hat\beta(\theta)}(x), y) \quad \text{s.t.} \quad \hat\beta(\theta) = \arg\min_{\beta} \sum_{i=1}^{n} \ell(f_\beta(x^{(i)}), y^{(i)}; \theta) $$

+ 这里需要再深入理解一下这个双层优化的结构.
    - 首先, 虽然优化的过程是迭代循环的, 但是并不存在优化逻辑上的死锁. 需要指出, $w$ 是依赖于 $\theta$ 的 (因为一组 $\theta$ 就定义了一个损失函数也就对应了一个最优参数 $w^*$), 而 $\theta$ 的更新依据是当前 $w^*$ 在 validation 的表现.

我们希望这个 $w(\theta)$ 是一个关于 $\theta$ 的可微函数, 可以计算 $\nabla_\theta w(\theta)$, 从而使得我们可以将当前这个求解过程嵌入到神经网络中实现端到端的训练.

## 关键问题

目前我们不仅要解决一个优化问题 $w^*(\theta) = \arg\min_w \mathcal{L}_{\text{train}}(\beta;\theta)$, 更棘手的问题是我们要处理这个隐函数关系 $w^* (\theta) = \beta^*(U,V,S,T)$, 使得我们可以计算 $\frac{\partial \beta^*}{\partial \theta}$ 从而优化 validation 上的终极损失:

$$ \frac{d}{d\theta} \mathcal{L}_{\text{val}}\left(\beta^*(\theta)\right) = \frac{\partial \mathcal{L}_{\text{val}}}{\partial \beta^*} \cdot \frac{\partial \beta^*}{\partial \theta} $$

### 无约束情景

首先回顾 [[Jacobian Matrix]].

_**定理1 ([[隐函数定理]], Implicit Function Theorem)**_

对于自变量 $\theta \in \mathbb{R}^d, w \in \mathbb{R}^n$, 定义函数 $f: \mathbb{R}^d \times \mathbb{R}^n \to \mathbb{R}^n$ 满足如下条件:

1. 函数 $f$ 在某点 $(\bar{\theta}, \bar{w})$ 的邻域内连续可微.
2. 且该点是一个零点, 即 $f(\bar{\theta}, \bar{w}) = 0$.
3. 且函数 $f$ 关于 $w$ 的雅可比矩阵 $J_f(\bar{\theta}, \bar{w})$ 在该点是非奇异的, 即:

$$ \nabla_{w} f(\bar{\theta}, \bar{w}) = \begin{bmatrix} \frac{\partial f_1}{\partial \bar w_1} & \cdots & \frac{\partial f_1}{\partial \bar w_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_n}{\partial \bar w_1} & \cdots & \frac{\partial f_n}{\partial \bar w_n} \end{bmatrix} \in \mathbb{R}^{n\times n}, ~ \det(\nabla_{w} f(\bar{\theta}, \bar{w})) \neq 0 $$

则有:

1. 在 $\bar{\theta}$ 的邻域内, 满足 $f(\theta, w) = 0$ 的解 $w^*$ 是一个 $\theta\in \mathbb{R}^d$ 的隐函数, 可以表示为 $w^* = s(\theta) \in \mathbb{R}^n$.
2. 且有

$$ \underbrace{\nabla_{\theta} s(\theta)}_{\mathbb{R}^{n\times d}} = -\left[ \underbrace{\nabla_{w} f(\theta, s(\theta))}_{\mathbb{R}^{n\times n}} \right]^{-1} \underbrace{\nabla_{\theta} f(\theta, s(\theta))}_{\mathbb{R}^{n\times d}} $$

借助上述定理, 我们可以在不需要显式求解 $s(\theta)$ 的情况下, 直接计算 $\nabla_{\theta} s(\theta)$. 我们只需要知道 $\nabla_{w} f, \nabla_{\theta} f$ 在某点的值, 以及 $\nabla_{w} f$ 的逆矩阵.

$\diamond$

对应在 general 的优化问题 $(1)$ 上:

+ 由于优化目标 $\min_{w\in C} f(w;\theta)$, 其最优点 $w^*(\theta)$ 若存在必满足: $\nabla_w f(w^*;\theta) = 0$
+ 进一步定义这个偏导数为: $F(w, \theta) = \nabla_w f(w;\theta)$
+ 则求解最优解 $w^*(\theta)$ 的过程转化为: $w^* (\theta) = w(\theta), ~\text{s.t. } F(w(\theta), \theta) = 0$

对应在我们的求解问题中,

+ 我们通过某种优化手段在给定 $\theta$ 的情况下在训练集上求解出最优参数 $w^*(\theta)$
+ 接着我们希望在验证集上优化 $\theta$ 使得 $\min_{\theta} \mathcal{L}_{\text{val}}(w^*(\theta))$, 故我们必须求: $\frac{\mathrm{d}}{\mathrm{d}\theta} \mathcal{L}_{\text{val}}(w^*(\theta)) = \frac{\partial \mathcal{L}_{\text{val}}}{\partial w^*} \cdot \frac{\partial w^*}{\partial \theta}$
+ 而这里的 $\frac{\partial w^*}{\partial \theta}$ 就是我们需要求解的隐函数导数, 其可以通过上述定理1直接计算: $\frac{\partial w^*}{\partial \theta} = -\left[ \nabla_w F(w^*(\theta), \theta) \right]^{-1} \cdot \nabla_\theta F(w^*(\theta), \theta)$

### 有约束 QP 情景

#### 引入: 为什么我们需要有约束的 QP?

结合实际场景, 我们在上述的讨论中考虑的 $f$ 是一个无约束的凸函数, 且具有可微等良好性质. 然而在我们目前的构造中, $f$ 对应的是由若干 ReLU 和 ReHU 函数构成的损失函数, 直接优化 $\min_w \sum \text{ReLU}(w) + \sum \text{ReHU}(w)$ 更为困难.

因此我们需要将其转化为一个有约束的凸优化问题, 例如 QP 问题. 由于注意到有如下等价关系:

$$ \begin{align*} \text{ReLU}(z) &= \max(0, z) \equiv \min_{\pi \geq 0, \pi \geq z} \pi\\ \text{ReHU}_\tau(z) &\equiv \min_{\vartheta, \sigma \geq 0} \frac{1}{2}\vartheta^2 + \tau \sigma, \quad \text{s.t. } \vartheta + \sigma \geq z \end{align*} $$

因此我们本质关心的损失函数可以理解为一个 QP 优化问题:

$$ \begin{aligned} \min_{\boldsymbol{\beta}, \{\pi, \vartheta, \sigma\}} \quad & \sum_{l=1}^L \pi_l^{(i)} + \sum_{h=1}^H \left( \frac{1}{2} (\vartheta_h^{(i)})^2 + \tau \sigma_h^{(i)} \right) + \lambda_{\text{reg}} \|\boldsymbol{\beta}\|_2^2 \\ \text{s.t.} \quad & \pi_l^{(i)} \geq U_l^{(i)} r^{(i)} + V_l^{(i)} ,\quad \pi_l^{(i)} \geq 0 \\ & \vartheta_h^{(i)} + \sigma_h^{(i)} \geq S_h^{(i)} r^{(i)} + T_h^{(i)} , \quad \sigma_h^{(i)} \ge 0 \end{aligned} $$

其中 $r^{(i)} = y^{(i)} - \mathbf{x}^{(i) \top} \boldsymbol{\beta}$ 是 $\boldsymbol{\beta}$ 的函数.

$\diamond$

#### 补充: KKT 条件

这里我们需要引入 KKT 条件来处理有约束的 QP 问题. KKT 条件是求解约束优化问题的必要条件, 其作用类似于在无约束优化中使用的梯度为零的条件. 另外, 对于凸优化的满足 Slater条件的凸优化问题, KKT 条件也是充分必要的.

下面给出 KKT 条件的一般形式.

考虑如下含约束问题:

$$ \begin{aligned} \min_{x} \quad & f(x) \\ \text{s.t.} \quad & g_i(x) \le 0,\quad i=1,\dots,m \\ & h_j(x) = 0,\quad j=1,\dots,p \end{aligned} \tag{3} $$

定义 Lagrangian 函数:

$$ \mathcal{L}(x, \lambda_{\text{lgrg}}, \nu) = f(x) + \sum_{i=1}^m \lambda_{\text{lgrg},i} g_i(x) + \sum_{j=1}^p \nu_j h_j(x) $$

其中 $\lambda_{\text{lgrg},i}$ 是对应不等式约束 $g_i(x) \le 0$ 的拉格朗日乘子 (dual variable), $\nu_j$ 是对应等式约束 $h_j(x) = 0$ 的拉格朗日乘子.

KKT 条件包括以下几个部分:

1. **Stationarity** (梯度条件):

$$ \nabla_x \mathcal{L}(x^*, \lambda_{\text{lgrg}}^*, \nu^*) = \nabla f(x^*) + \sum_i \lambda_{\text{lgrg},i}^* \nabla g_i(x^*) + \sum_j \nu_j^* \nabla h_j(x^*) = 0 $$

2. **Primal feasibility** (原始可行性):

$$ \begin{aligned} g_i(x^*) &\le 0, \quad i=1,\dots,m \\ h_j(x^*) &= 0, \quad j=1,\dots,p \end{aligned} $$

3. **Dual feasibility** (对偶可行性):

$$ \lambda_{\text{lgrg},i}^* \ge 0, \quad i=1,\dots,m $$

4. **Complementary slackness** (互补松弛性):

$$ \lambda_{\text{lgrg},i}^* g_i(x^*) = 0, \quad i=1,\dots,m $$

#### 有约束 QP 的一般形式

下面给出含约束的 QP 优化问题的一般形式.

$$ \begin{aligned} \min_{z} \quad & \frac{1}{2} z^\top Q z + q^\top z \\ \text{s.t.} \quad & A z = b \\ & G z \le h \end{aligned} \tag{2} $$

其中:

+ $z \in \mathbb{R}^n$ 是优化变量. (这里我们先考虑这个一般形式, 不考虑其本身 QP 的松弛构造),
+ $Q \in \mathbb{R}^{n\times n}$ 是对称(半)正定矩阵 ($Q \succcurlyeq 0$, 因此是凸的), 定义了二次项.
+ $q \in \mathbb{R}^n$ 是线性项的系数向量.
+ $A \in \mathbb{R}^{m_\text{eq}\times n}, b \in \mathbb{R}^{m_\text{eq}}$ 定义了等式约束.
+ $G \in \mathbb{R}^{m_\text{ineq}\times n}, h \in \mathbb{R}^{m_\text{ineq}}$ 定义了不等式约束.

> 若对应到我们的具体工作中, 则 $G,h$ 等都应当是关于 $\theta$ 的函数 (这里理论上不含等式约束). 另外, 对于优化问题 $(2)$ 的最优解 $z^*$, 其也应当是 $\theta$ 的隐函数, 即 $z^* = z(\theta)$.

这里引入两组 Lagrangian Multiplier:

+ $\nu \in \mathbb{R}^{m_\text{eq}}$ 对应 $Az= b$
+ $\lambda_{\text{lgrg}} \in \mathbb{R}^{m_\text{ineq}}$ 对应 $Gz \le h$

因此我们可以定义 Lagrangian 函数:

$$ \mathcal{L}(z, \nu, \lambda_{\text{lgrg}}) = \frac{1}{2} z^\top Q z + q^\top z + \nu^\top (Az - b) + \lambda_{\text{lgrg}}^\top (Gz - h) $$

对于这个 Lagrangian 函数, KKT 条件如下:

1. **Stationarity**:

$$ \nabla_z \mathcal{L}(z^*, \nu^*, \lambda_{\text{lgrg}}^*) = Q z^* + q + A^\top \nu^* + G^\top \lambda_{\text{lgrg}}^* = 0 $$

2. **Primal feasibility**:

$$ \begin{aligned} Az^* &= b \\ Gz^* &\le h \end{aligned} $$

3. **Dual feasibility**:

$$ \lambda_{\text{lgrg}}^* \ge 0 $$

4. **Complementary slackness**:

$$ \lambda_{\text{lgrg},i}^* (G z^* - h)_i = 0, ~ i=1,\cdots,m_\text{ineq} \Leftrightarrow \text{Diag}(\lambda_{\text{lgrg}}^*) (Gz^* - h) = 0 $$

---

整理一下, 我们在 KKT 条件中得到了如下等式:

$$ \begin{aligned} Qz^\ast + q + A^\top \nu^\ast + G^\top \lambda^\ast &= 0 \\ Az^\ast - b &= 0 \\ D(\lambda^\ast)(Gz^\ast - h) &= 0 \end{aligned} $$

在引入新的方程前, 对符号进行一下声明. 定义 augmented 变量: $\omega^* = (z^*, \lambda_{\text{lgrg}}^*,\nu^*)^\top \in \mathbb{R}^{n + m_\text{eq} + m_\text{ineq}}$. 考虑参数 $Q,q,A,b,G,h$ 其某种意义上都可以看作是超参数 $\theta$ 的某种函数. (事实上, $Q,q$ 作为优化目标并不真的依赖于 $\theta$, 目前我们也没有考虑等式约束 $A,b$ 目前没有引入, 但这都可以看作是退化的情况. )

因此上述的三个 KKT 等式方程可以写成一个向量方程:

$$ F(\omega ^*, \theta) = \begin{bmatrix} Qz^* + q + A^\top \nu^* + G^\top \lambda_{\text{lgrg}}^* \\ Az^* - b \\ D(\lambda_{\text{lgrg}}^*)(Gz^* - h) \end{bmatrix} = 0 $$

#### 从 KKT 等式到隐函数导数

回顾, 我们的本质优化问题是通过双层优化系统来求解的:

$$ \min_{\theta} \; \mathcal{L}_{\text{val}}(w^*(\theta)) \quad \text{s.t.} \quad w^*(\theta) = \arg\min_{w} \mathcal{L}_{\text{train}}(w; \theta) $$

+ 这里 $\mathcal{L}_{\text{train}}$ 是我们目前在苦苦推导的含约束 QP 问题. 通过求解这个内层损失我们得到最优解 $w^*(\theta)$
+ 进而可以在外层损失 $\mathcal{L}_{\text{val}}(w^*(\theta))$ 上进行反向传播到 $\theta$ 上, 使得我们可以通过梯度下降来优化 $\theta$: $\frac{\mathrm d}{\mathrm d\theta} \mathcal{L}_{\text{val}}(w^*(\theta)) = \frac{\partial \mathcal{L}_{\text{val}}}{\partial w^*} \cdot \frac{\partial w^*}{\partial \theta}$. 所以我们需要得到 $\frac{\partial z^*}{\partial \theta}$, 即 $\nabla_\theta w^*(\theta)$.

我们通过上述的 KKT 条件得到了一个向量方程: $F(\omega, \theta) = 0$. 根据_**定理1 (Implicit Function Theorem)**_, 只要 $\nabla_w F(\omega^*, \theta)$ 在某点是非奇异的,则可以得到 $\nabla_\theta \omega^*(\theta)$:

$$ \nabla_\theta \omega^*(\theta) = - \left[ \nabla_\omega F(\omega^*, \theta) \right]^{-1} \cdot \nabla_\theta F(\omega^*, \theta) $$

此时我们的问题就充分转化为求解 $F(\omega^*, \theta)$ 的雅可比矩阵 $\nabla_w F(\omega^*, \theta)$ 和 $\nabla_\theta F(\omega^*, \theta)$.

回顾(此处用下标表示其维度)

$$ F(\omega^*, \theta) = \begin{bmatrix} F_1 \in \mathbb{R}^n:= Q_{(n\times n)}z_{(n)}^* + q_{(n)} + (A_{(m_\text{eq}\times n)})^\top \nu_{(m_\text{eq})}^* + (G_{(m_\text{ineq}\times n)})^\top \lambda_{\text{lgrg} (n_\text{ineq})}^* \\ F_2\in\mathbb{R}^{m_\text{eq}}:=A_{(m_\text{eq}\times n)}z_{(n)}^* - b_{(m_\text{eq})} \\ F_3\in\mathbb{R}^{m_\text{ineq}}:=D(\lambda_{\text{lgrg}}^*)_{(m_\text{ineq})}(G_{(m_\text{ineq}\times n)}z_{(n)}^* - h_{(m_\text{ineq})}) \end{bmatrix} = 0 $$

且 $\omega^* = (z_{(n)}^*, \lambda_{\text{lgrg}(m_\text{ineq})}^*,\nu_{(m_\text{eq})}^*)^\top$.

故对于 $\nabla_\omega F(\omega^*, \theta)$:

$$ \nabla_\omega F(\omega^*, \theta) = \begin{bmatrix} \frac{\partial F_1}{\partial z^*} & \frac{\partial F_1}{\partial \lambda_{\text{lgrg}}^*} & \frac{\partial F_1}{\partial \nu^*} \\ \frac{\partial F_2}{\partial z^*} & \frac{\partial F_2}{\partial \lambda_{\text{lgrg}}^*} & \frac{\partial F_2}{\partial \nu^*} \\ \frac{\partial F_3}{\partial z^*} & \frac{\partial F_3}{\partial \lambda_{\text{lgrg}}^*} & \frac{\partial F_3}{\partial \nu^*} \end{bmatrix} = \begin{bmatrix} Q_{(n\times n)} & (G^\top)_{(n\times m_\text{ineq})} & (A^\top)_{(n\times m_\text{eq})} \\ A_{(m_\text{eq}\times n)} & \boldsymbol{0}_{(m_\text{eq}\times m_\text{eq})} & \boldsymbol{0}_{(m_\text{eq}\times m_\text{ineq})} \\ (\text{Diag}(\lambda_{\text{lgrg}}^*) G)_{(m_\text{ineq}\times n)} & \text{Diag}(Gz^* - h)_{(m_\text{ineq})} & \boldsymbol{0}_{(m_\text{ineq}\times m_\text{eq})} \end{bmatrix} \in \mathbb{R}^{(n + m_\text{eq} + m_\text{ineq})} $$

其实在矩阵上也可以类似微分算子, 得到如下表达式:

$$ \partial F(\omega^*, \theta) = \begin{bmatrix} Q & G^\top & A^\top \\ A & \boldsymbol{0} & \boldsymbol{0} \\ \text{Diag}(\lambda_{\text{lgrg}}^*) G & \text{Diag}(Gz^* - h) & \boldsymbol{0} \end{bmatrix} \begin{bmatrix} \mathrm{d} z^* \\ \mathrm{d} \lambda_{\text{lgrg}}^* \\ \mathrm{d} \nu^* \end{bmatrix} = - \begin{bmatrix} \mathrm{d} Qz^* + \mathrm{d} q + \mathrm{d} G^\top \lambda_{\text{lgrg}}^* + \mathrm{d} A^\top \nu^* \\ \text{Diag}(\lambda_{\text{lgrg}}^*) \mathrm{d} G z^* - \text{Diag}(\lambda_{\text{lgrg}}^*) \mathrm{d} h \\ \mathrm{d} A z^* - \mathrm{d} b \end{bmatrix} $$

注意到, $\omega^* = (z^*, \lambda_{\text{lgrg}}^*,\nu^*)^\top$. 而其中, 在本实验涉及到的具体问题中, $z^* = (\boldsymbol{\beta}^*, \pi^*, \vartheta^*, \sigma^*)^\top$ 其中 $w^* = \boldsymbol{\beta}^*$ 是模型回归参数, $\pi^*, \vartheta^*, \sigma^*$ 是对应 ReLU 和 ReHU 的松弛变量. 因此有关系: $\boldsymbol{\beta}^* = w^* \subset z^* \subset \omega^*$ (这里的 $\subset$ 指不严谨的包含关系(因为其并不是集合而是向量)).

---

因此再次重新整理一下我们的求解目标:

$$ \frac{\mathrm d \mathcal{L}_{\text{val}}}{\mathrm{d}\theta} = \frac{\partial \mathcal{L}_{\text{val}}}{\partial w^*} \cdot \frac{\partial w^*}{\partial \theta} = \frac{\partial \mathcal{L}_{\text{val}}}{\partial \omega^*} \cdot \frac{\partial \omega^*}{\partial \theta} \quad\text{(a)} $$

其中:

$$ \frac{\partial\mathcal{L}_{\text{val}}}{\partial \omega^*} = \frac{\partial \mathcal{L}_{\text{val}}}{\partial [z^*, \lambda_{\text{lgrg}}^*, \nu^*]^\top} = [\frac{\partial \mathcal{L}_{\text{val}}}{\partial z^*}, \boldsymbol{0}_{(m_\text{ineq})}, \boldsymbol{0}_{(m_\text{eq})}]^\top \in \mathbb{R}^{n + m_\text{eq} + m_\text{ineq}} \quad\text{(b)} $$

刚刚又由隐函数定理, 我们求得:

$$ \frac{\partial \omega^*}{\partial \theta} = - \left( \frac{\partial F}{\partial \omega} \right)^{-1} \cdot \frac{\partial F}{\partial \theta}= - \begin{bmatrix} Q & G^\top & A^\top \\ A & \boldsymbol{0} & \boldsymbol{0} \\ \text{Diag}(\lambda_{\text{lgrg}}^*) G & \text{Diag}(Gz^* - h) & \boldsymbol{0} \end{bmatrix}^{-1} \cdot \frac{\partial F}{\partial \theta} \quad{\text{(c)}} $$

将 $\text{(b)},\text{(c)}$ 代入 $\text{(a)}$, 则有:

$$ \begin{aligned} \frac{\mathrm{d}}{\mathrm{d}\theta} \mathcal{L}_{\text{val}} &= \frac{\partial \mathcal{L}_{\text{val}}}{\partial \omega^*} \cdot \left( - \left( \frac{\partial F}{\partial \omega} \right)^{-1} \cdot \frac{\partial F}{\partial \theta} \right) \\ &= \underbrace{- \begin{bmatrix}\frac{\partial \mathcal{L}_{\text{val}}}{\partial z^*} \\ \boldsymbol{0}\\ \boldsymbol{0}\end{bmatrix} \cdot \begin{bmatrix} Q & G^\top & A^\top \\ A & \boldsymbol{0} & \boldsymbol{0} \\ \text{Diag}(\lambda_{\text{lgrg}}^*) G & \text{Diag}(Gz^* - h) & \boldsymbol{0} \end{bmatrix}^{-1} }_{\triangleq D}\cdot \frac{\partial F}{\partial \theta} \quad \diamond \end{aligned} $$

我们不希望直接计算这样一个大矩阵的逆并再进行矩阵乘法. 下面是进一步的推导.

将上述的线性系统记为 $D$, 则:

$$ D^\top = - \begin{bmatrix} Q^\top & A^\top & \text{Diag}(\lambda_{\text{lgrg}}^*) G^\top \\ G & \boldsymbol{0} & \text{Diag}(Gz^* - h) \\ A & \boldsymbol{0} & \boldsymbol{0} \end{bmatrix}^{-1} \cdot \begin{bmatrix} \frac{\partial \mathcal{L}_{\text{val}}}{\partial z^*} \\ \boldsymbol{0} \\ \boldsymbol{0} \end{bmatrix} := \begin{bmatrix} d_{z^*} \\ d_{\lambda_{\text{lgrg}}} \\ d_{\nu} \end{bmatrix} $$

那么

$$ \frac{\mathrm{d}}{\mathrm{d}\theta} \mathcal{L}_{\text{val}} = D^\top \cdot \frac{\partial F}{\partial \theta} $$