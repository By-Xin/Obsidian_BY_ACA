> 本文档是 ReLU KKT 系统的高效隐式求导方法的简化版本. 适用于直接 public 的, 去除详细数学推导说明的公开版本之手稿. 本文档力求符号维度清晰明了, 定义清晰, 简介不冗余. 

## 问题背景

考虑如下 ReLU 模型:
$$\min_{\boldsymbol{\beta}\in\mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}(u_{l,i} \mathbf{x}_i^\top \boldsymbol{\beta} + v_{l,i})+\frac12 \|\boldsymbol{\beta}\|^2$$

- 其中 $U = (u_{l,i}) , V = (v_{l,i}) \in \mathbb{R}^{L \times n}$ 是模型参数, $L$ 是 ReLU 的个数, $n$ 是样本的个数, $\text{ReLU}(x) = \max(0,x)$ 是 ReLU 函数. $\mathbf{x}_i \in \mathbb{R}^d$ 是输入特征, $\boldsymbol{\beta}\in\mathbb{R}^d$ 是对应系数.

根据 ReLU 的等价变换, 进一步定义最终要求解的 Primal 问题为:

$$
\begin{aligned} \min_{\boldsymbol{\beta},\Pi} &\quad \sum_{i=1}^n \sum_{l=1}^L \pi_{l,i} + \frac12 \|\boldsymbol{\beta}\|^2 \\ \text{s.t. } &\quad  u_{l,i} \mathbf{x}_i^\top \boldsymbol{\beta} + v_{l,i} - \pi_{l,i} \geq 0,\quad \forall i,l \\ &\quad  \pi_{l,i} \geq 0,\quad \forall i,l\end{aligned}
$$

- 其中 $\Pi = (\pi_{l,i}) \in \mathbb{R}^{L \times n}$ 是 ReLU 的松弛变量.

因此为了求解这个问题, 我们构建了如下的 Lagrange 方程:

$$
\begin{aligned} \mathcal{L}_{\mathcal P}(\boldsymbol{\beta}, \Pi;  \Lambda, \Delta) = & \sum_{i=1}^n \sum_{l=1}^L \pi_{l,i}  + \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 \\ &  - \sum_{i=1}^n \sum_{l=1}^L \lambda_{l,i} (\pi_{l,i} - u_{l,i} \mathbf{x}_i^\top \boldsymbol{\beta} - v_{l,i}) \\ & - \sum_{i=1}^n \sum_{l=1}^L \delta_{l,i} \pi_{l,i} \end{aligned}
$$

通过求解该 Lagrange 的 KKT 方程, 我们得到如下方程:

$$
\begin{aligned}
F := \begin{bmatrix} F_1 \\ F_2 \\ F_3 \end{bmatrix} :=
\begin{bmatrix}
\boldsymbol{\beta}^* + \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}^*) \\
\text{diag}(\text{vec}(\Lambda^*)) \left(
    \text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)\right) \\
\left(I_{Ln} - \text{diag}(\text{vec}(\Lambda^*))\right) \text{vec}(\Pi^*)
\end{bmatrix} = \begin{bmatrix} \boldsymbol{0}_{d} \\ \mathbf{0}_{Ln} \\ \mathbf{0}_{Ln} \end{bmatrix} \in \mathbb{R}^{(d+Ln+Ln)} 
\end{aligned}
$$

- 其中 $\mathsf{\bar{U}} = (u_{lij}):=(u_{li}x_{ij})\in \mathbb{R}^{L\times n\times d}$, 矩阵 $\bar{\mathsf{U}}_{(3)}\in \mathbb{R}^{d \times (Ln)}$ 是张量 $\bar{\mathsf{U}}$ 在 mode-3 上的展开，本质上是一个 $d \times (Ln)$ 的矩阵.

定义 $\omega := \left[\boldsymbol{\beta}^*;\boldsymbol{\mathrm{vec}(\Pi)}^*;\boldsymbol{\mathrm{vec}(\Lambda)}^*\right]^\top \in \mathbb{R}^{d + Ln + Ln}$,  $\theta:= \left[\,\mathrm{vec}(U);\ \mathrm{vec}(V)\,\right]^\top\in \mathbb{R}^{2Ln}$.  

- 可以求解 $F$ 关于 $\omega$ 的 Jacobian 矩阵:
    $$\nabla_\omega F \;=\;\begin{bmatrix}\boldsymbol{I}_{d \times d} & \boldsymbol{0}_{d \times Ln} & \mathsf{\bar{U}}_{(3)} \\ -\text{diag}(\text{vec}(\Lambda^*)) \mathsf{\bar{U}}_{(3)}^\top & \text{diag}(\text{vec}(\Lambda^*)) & \text{diag}(\text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)) \\ \boldsymbol{0}_{Ln \times d} & I_{Ln} - \text{diag}(\text{vec}(\Lambda^*)) & -\text{diag}(\text{vec}(\Pi^*))\end{bmatrix}\\ :=\begin{bmatrix}I_d & \mathbf{0} & \bar{\mathsf{U}}_{(3)} \\[6pt]-\,\mathcal{D}_{\Lambda^*}\,\bar{\mathsf{U}}_{(3)}^\top & \mathcal{D}_{\Lambda^*} & \mathcal{D}_{2} \\[6pt]\mathbf{0} & I_{Ln} - \mathcal{D}_{\Lambda^*} & -\,\mathcal{D}_{\Pi^*}\end{bmatrix},$$

  - 其中 $\mathcal{D}_{\Lambda^*} = \operatorname{diag}(\mathrm{vec}(\Lambda^*))$、$\mathcal{D}_{\Pi^*} = \operatorname{diag}(\mathrm{vec}(\Pi^*))$、$\mathcal{D}_2 = \operatorname{diag}\bigl(\mathrm{vec}(\Pi^*) - \bar{\mathsf{U}}_{(3)}^\top \beta^* - \mathrm{vec}(V)\bigr)$.

- Jacobian $\nabla_\theta F$ 也具有结构化形式：
    $$\nabla_\theta F \;=\;\begin{bmatrix}\mathcal{X}\,\mathcal{D}_{\Lambda^*} & \mathbf{0}\\-\,\mathcal{D}_{\Lambda^*}\,\mathcal{D}_{(\mathcal {X}^\top \beta^*)} & -\,\mathcal{D}_{\Lambda^*}\\\mathbf{0} & \mathbf{0}\end{bmatrix} \in \mathbb{R}^{(2Ln+d) \times 2Ln},$$

  - 其中 $\mathcal{X} = [X, \dots, X]$（共 $L$ 份），$\mathcal{D}_{(\mathcal{X}^\top \beta^*)}$ 是向量 $\mathcal X^\top \beta^*$ 的对角矩阵。

## 核心求解过程

我们最终求解的目的是进行如下的双层优化系统:

$$
\min_{\theta} \; \mathcal{L}_{\text{val}}(\omega^*(\theta)) \quad \text{s.t.} \quad \omega^*(\theta) = \arg\min_{\omega} \mathcal{L}_{\text{train}}(\omega; \theta)
$$

因此在求解 $\min_\theta\mathcal{L}_{\text{val}}$ 的时候, 我们就希望能够拿到其关于 $\theta$ 的梯度数据, 这样就可以调用基于梯度的优化算法进行求解. 进而, 我们最终的终极求解链式法则为:

$$
\frac{\mathrm d \mathcal{L}_{\text{val}}}{\mathrm{d}\theta}= \frac{\partial \mathcal{L}_{\text{val}}}{\partial \omega^*} \cdot \frac{\partial \omega^*}{\partial \theta}
$$

- 又加之在最优解处，我们假设 $\nabla_\omega F$ 可逆（不满足此条件将导致活动集退化）
  $$\frac{\partial \omega}{\partial \theta} = - \left(\nabla_\omega F\right)^{-1} \nabla_\theta F$$

故:
$$\boxed{
\nabla_\theta \mathcal{L}_{\mathrm{val}} = \nabla_\omega \mathcal{L}_{\mathrm{val}} \cdot \left(- (\nabla_\omega F)^{-1} \cdot \nabla_\theta F \right)} \quad (1)
$$

记 $\boldsymbol{\psi} := \nabla_\omega \mathcal{L}_{\mathrm{val}} \in \mathbb{R}^{1 \times (d+Ln+Ln)}, ~K := \nabla_\omega F \in \mathbb{R}^{(2Ln+d) \times (2Ln+d)}$,  则 $(1)$ 可以写成:

$$
\nabla_\theta \mathcal{L}_{\mathrm{val}} = - {\boldsymbol{\psi}} \cdot {K^{-1} \cdot \nabla_\theta F}\quad (2)
$$

我们的自动微分求解器 $\mathsf{AD}$ 相当于一个函数系统, 给定 $\boldsymbol{\psi}$, 便可以计算出 $\nabla_\theta \mathcal{L}_{\mathrm{val}} = \mathsf{AD}(\boldsymbol{\psi}, K, \nabla_\theta F)$. 这里我们并不关心 $\mathcal{L}_{\mathrm{val}}$ 的具体形式.

---

## 微分推导

对于 $(2)$, 记 $\boldsymbol{v} := (\boldsymbol{\psi} K^{-1})^\top := \begin{bmatrix} v_\beta \\ v_\Pi \\ v_\Lambda \end{bmatrix} \in \mathbb{R}^{d+Ln+Ln}$, 则求解 $\boldsymbol{v}$ 的过程等价于求解方程: $K^\top \boldsymbol{v} = \boldsymbol{\psi}^\top$.

展开具体形式, 等价于求解:
$$\begin{aligned}
I_d v_\beta - \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} v_\Pi &= \psi_\beta \quad \mathrm{(a)}\\
\mathcal{D}_{\Lambda^*} v_\Pi + (I_{Ln} - \mathcal{D}_{\Lambda^*}) v_\Lambda &= \psi_\Pi \quad \mathrm{(b)}\\
\bar{\mathsf{U}}_{(3)}^\top v_\beta + \mathcal{D}_{2} v_\Pi - \mathcal{D}_{\Pi^*} v_\Lambda &= \psi_\Lambda \quad \mathrm{(c)}
\end{aligned}$$

首先求解 $\mathrm{(b)}$ . 

- 经过等价变形, 得到主线性系统
    $$[\mathcal{D}_{\Pi^*} (I - \mathcal{D}_{\Lambda^*})^{-1} + \mathcal{D}_2\mathcal{D}_{\Lambda^*}^{-1}+ \bar{\mathsf{U}}_{(3)}^\top\bar{\mathsf{U}}_{(3)}] v_\Pi = [(I - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1} \mathcal{D}_{\Lambda^*}]^{-1}[\psi_\Pi -  (I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1}(\bar{\mathsf{U}}_{(3)}^\top \psi_\beta - \psi_\Lambda)] := \tilde  c$$

- 令 $D := \mathcal{D}_{\Pi^*} (I - \mathcal{D}_{\Lambda^*})^{-1} + \mathcal{D}_2\mathcal{D}_{\Lambda^*}^{-1}$, 简记 $\bar{\mathsf{U}}_{(3)} = \bar{\mathsf{U}}$, 则上式可以化简为
$$
(D + \bar{U}^\top \bar{U}) v_\Pi := \widetilde{S}v_\Pi =  \tilde c
$$

- 而进一步, $\widetilde{\mathcal{S}}$ 满足 SMW 条件, 因此利用对称的 Woodbury 公式:
$$\widetilde{\mathcal{S}}^{-1}
= D^{-1} - D^{-1} \bar{U}^\top (I + \bar{U} D^{-1} \bar{U}^\top)^{-1} \bar{U} D^{-1}
$$

- 再代回主线性系统中, 有:
  $$\begin{aligned} v_\Pi &= \widetilde{\mathcal{S}}^{-1} \tilde{c} \\ &= D^{-1} \tilde{c} - D^{-1} \bar{U}^\top (I + \bar{U} D^{-1} \bar{U}^\top)^{-1}[ \bar{U} (D^{-1} \tilde{c})] \\ &= D^{-1} \tilde{c} - D^{-1} \bar{U}^\top \text{CholeskySolve}\left(I + \bar{U} D^{-1} \bar{U}^\top, \bar{U} (D^{-1} \tilde{c})\right) \end{aligned}$$
    其中最后一个等式中的求逆是通过 Cholesky 分解来求解的, 其合法性由 $I + \bar{U} D^{-1} \bar{U}^\top$ 的对称正定性保证.

进一步利用其余 $\mathrm{(a)}$ 和 $\mathrm{(c)}$ 方程, 可以得到:
  - $v_\beta = \psi_\beta + \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} v_\Pi$
  - $v_\Lambda = \mathcal{D}_{\Pi^*}^{-1} \left( \bar{\mathsf{U}}_{(3)}^\top \psi_\beta - \psi_\Lambda + ( \bar{\mathsf{U}}_{(3)}^\top \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} + \mathcal{D}_{2}) v_\Pi \right)$

## Package 的任务

希望做出这个自动求导模块, 使得未来用户在进行求解 $\min_{\theta} \; \mathcal{L}_{\text{val}}(\omega^*(\theta)) \quad \text{s.t.} \quad \omega^*(\theta) = \arg\min_{\omega} \mathcal{L}_{\text{train}}(\omega; \theta)$ 的时候, 由于需要 $\frac{\mathrm d \mathcal{L}_{\text{val}}}{\mathrm{d}\theta}= \frac{\partial \mathcal{L}_{\text{val}}}{\partial \omega^*} \cdot \frac{\partial \omega^*}{\partial \theta}$ 因此需要 $\nabla_\theta \mathcal{L}_{\mathrm{val}} = - {\boldsymbol{\psi}} \cdot {K^{-1} \cdot \nabla_\theta F}\quad (2)$ .这个地方本来需要求导求逆,用自动微分很费力, 而我已经通过 Woodbury +cholesky 求出了一个更计算有效的版本, 因此只需要用户提供 psi, 我其余的地方就可以用解方程组的方式直接求出梯度

建一个代码仓库来实现differentiable ReHLine的算法. 可以参照这两个库的结构：https://github.com/google-research/fast-soft-sort/，https://github.com/teddykoker/torchsort (上面这两个库也是自定义的求导模块，可以结合我之前发的资料理解一下PyTorch自定义模块的写法) 。后把ReLU这个版本的代码实现一下，写成一个在PyTorch中能自动微分的组件。

differentiable ReHLine单独开一个repo，专注实现自动微分功能，然后autoloss在它的基础上单独再搭一层。相当于把任务拆解成两个部分，更加模块化

双层优化那里其实就是怎么用ReHLine的输出结果定义下游损失函数，剩下的链式求导PyTorch会自动完成。自动微分模块的核心就是把最终损失对模块输出对象的导数当作一个可变的参数，不需要提前指定，因此具有泛用性. 也就是说，我们实现ReHLine的自动微分，是把d_L_val/d_w当作一个函数参数传进来，然后我们要输出d_L_val/d_theta的结果。至于d_L_val/d_w具体是什么形式，或者怎么计算，是与ReHLine模块本身无关的

但我们并不是显式求出dw/dtheta，而是完成它的Jacobian-vector乘法运算. 相当于我们要表达一个矩阵A，但我们并不给出A的具体取值，而是实现一个函数f(v)，它可以对任意输入v返回$A*v$的结果。从这个意义上讲，f(v)的信息和A本身是完全等价的，只是表达的方式不同 这样的好处是，A本身是一个高维矩阵，但f(v)的输入和输出都是向量，因此实现f(v)可能比直接给出A要计算更高效，占用内存也更小