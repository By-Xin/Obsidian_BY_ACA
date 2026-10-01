## 问题回顾

[[纯ReLU部分推导]]

考虑如下简化 ReHLine 模型:

$$
\min_{\boldsymbol{\beta}\in\mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}(u_{l,i} \mathbf{x}_i^\top \boldsymbol{\beta} + v_{l,i})+\frac12 \|\boldsymbol{\beta}\|^2
$$

- 其中 $U = (u_{l,i}) , V = (v_{l,i}) \in \mathbb{R}^{L \times n}$ 是模型参数, $L$ 是 ReLU 的个数, $n$ 是样本的个数, $\text{ReLU}(x) = \max(0,x)$ 是 ReLU 函数. $\mathbf{x}_i \in \mathbb{R}^d$ 是输入特征, $\boldsymbol{\beta}\in\mathbb{R}^d$ 是对应系数.

根据 ReLU 的等价变换, 进一步定义最终要求解的 Primal 问题为:

$$
\begin{aligned} \min_{\boldsymbol{\beta},\Pi} &\quad \sum_{i=1}^n \sum_{l=1}^L \pi_{l,i} + \frac12 \|\boldsymbol{\beta}\|^2 \\ \text{s.t. } &\quad  u_{l,i} \mathbf{x}_i^\top \boldsymbol{\beta} + v_{l,i} - \pi_{l,i} \leq 0,\quad \forall i,l \\ &\quad  \pi_{l,i} \geq 0,\quad \forall i,l\end{aligned}
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

## Lagrange 中的 Jacobian 结构

在原始的[[纯ReLU部分推导]]部分中, 我们已经根据上述方程 $F=\boldsymbol{0}$ 方程推出了一个隐函数系统. 我们需要从这个隐函数系统出发, 进一步研究其高效计算的方法.

我们考虑 ReLU 优化问题的 KKT 条件，其原始变量包括 $\beta$、$\Pi$，以及对偶乘子 $\Lambda$，如所提供文档中所述。KKT 残差向量 $F(\omega, \theta) = 0$ 定义

$$
\omega := \left[\boldsymbol{\beta}^*;\boldsymbol{\mathrm{vec}(\Pi)}^*;\boldsymbol{\mathrm{vec}(\Lambda)}^*\right]^\top \in \mathbb{R}^{d + Ln + Ln},\quad
\theta:= \left[\,\mathrm{vec}(U);\ \mathrm{vec}(V)\,\right]^\top\in \mathbb{R}^{2Ln},
$$

其 Jacobian $\nabla_\omega F$ 具有特殊的块稀疏结构：

$$
\nabla_\omega F \;=\;\begin{bmatrix}
\boldsymbol{I}_{d \times d} & \boldsymbol{0}_{d \times Ln} & \mathsf{\bar{U}}_{(3)} \\
-\text{diag}(\text{vec}(\Lambda^*)) \mathsf{\bar{U}}_{(3)}^\top & \text{diag}(\text{vec}(\Lambda^*)) & \text{diag}(\text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)) \\
\boldsymbol{0}_{Ln \times d} & I_{Ln} - \text{diag}(\text{vec}(\Lambda^*)) & -\text{diag}(\text{vec}(\Pi^*))
\end{bmatrix}\\ :=\begin{bmatrix}
I_d & \mathbf{0} & \bar{\mathsf{U}}_{(3)} \\[6pt]
-\,\mathcal{D}_{\Lambda^*}\,\bar{\mathsf{U}}_{(3)}^\top & \mathcal{D}_{\Lambda^*} & \mathcal{D}_{2} \\[6pt]
\mathbf{0} & I_{Ln} - \mathcal{D}_{\Lambda^*} & -\,\mathcal{D}_{\Pi^*}
\end{bmatrix},
$$

其中 $\mathcal{D}_{\Lambda^*} = \operatorname{diag}(\mathrm{vec}(\Lambda^*))$、$\mathcal{D}_{\Pi^*} = \operatorname{diag}(\mathrm{vec}(\Pi^*))$、$\mathcal{D}_2 = \operatorname{diag}\bigl(\mathrm{vec}(\Pi^*) - \bar{\mathsf{U}}_{(3)}^\top \beta^* - \mathrm{vec}(V)\bigr)$。矩阵 $\bar{\mathsf{U}}_{(3)}\in \mathbb{R}^{d \times (Ln)}$ 是张量 $\bar{\mathsf{U}}$ 在 mode-3 上的展开，本质上是一个 $d \times (Ln)$ 的矩阵，而 $\mathsf{\bar{U}} = (u_{lij}):=(u_{li}x_{ij})\in \mathbb{R}^{L\times n\times d}$。由于互补松弛性，$\nabla_\omega F$ 中许多块为零或对角形式，具有良好的稀疏性。

在最优解处，我们假设 $\nabla_\omega F$ 可逆（不满足此条件将导致活动集退化）。我们的目标是高效计算灵敏度：

$$
\frac{\partial \omega}{\partial \theta} = - \left(\nabla_\omega F\right)^{-1} \nabla_\theta F,
$$

且尽量避免显式构造或求逆整个 Jacobian。Jacobian $\nabla_\theta F$ 也具有结构化形式：

$$
\nabla_\theta F \;=\;
\begin{bmatrix}
\mathcal{X}\,\mathcal{D}_{\Lambda^*} & \mathbf{0}\\
-\,\mathcal{D}_{\Lambda^*}\,\mathcal{D}_{(\mathcal {X}^\top \beta^*)} & -\,\mathcal{D}_{\Lambda^*}\\
\mathbf{0} & \mathbf{0}
\end{bmatrix} \in \mathbb{R}^{(2Ln+d) \times 2Ln},
$$

其中 $\mathcal{X} = [X, \dots, X]$（共 $L$ 份），$\mathcal{D}_{(\mathcal{X}^\top \beta^*)}$ 是向量 $\mathcal X^\top \beta^*$ 的对角矩阵。

## 双层优化结构

此外, 还需要指出的是, 我们最终求解的目的是进行如下的双层优化系统:

$$
\min_{\theta} \; \mathcal{L}_{\text{val}}(\omega^*(\theta)) \quad \text{s.t.} \quad \omega^*(\theta) = \arg\min_{\omega} \mathcal{L}_{\text{train}}(\omega; \theta)
$$

因此在求解 $\min\mathcal{L}_{\text{val}}$ 的时候, 我们就希望能够拿到其关于 $\theta$ 的梯度数据, 这样就可以调用基于梯度的优化算法进行求解. 进而, 我们最终的终极求解链式法则为:

$$
\frac{\mathrm d \mathcal{L}_{\text{val}}}{\mathrm{d}\theta}= \frac{\partial \mathcal{L}_{\text{val}}}{\partial \omega^*} \cdot \frac{\partial \omega^*}{\partial \theta}
$$

而其中, $\omega^*$ 是 $\mathcal{L}_{\text{val}}$ 的显示函数 (如 MSE 等), 其梯度可以较为直接的算出.因此最终的问题就落在 $\frac{\partial \omega^*}{\partial \theta}$ 上. 而这便也就顺承了上面的隐函数求导部分, 我们有 $\frac{\partial \omega}{\partial \theta} = - \left(\nabla_\omega F\right)^{-1} \nabla_\theta F$, 故:

$$
\nabla_\theta \mathcal{L}_{\mathrm{val}} = \nabla_\omega \mathcal{L}_{\mathrm{val}} \cdot \left(- (\nabla_\omega F)^{-1} \cdot \nabla_\theta F \right) \quad (1)
$$

## 避免显式逆：解线性系统

回顾我们有: $\omega:= [\mathrm{vec}(\Pi)^*;\, \beta^*;\, \mathrm{vec}(\Lambda)^*]^\top \in \mathbb{R}^{2Ln + d}$, $\theta := [\mathrm{vec}(U);\ \mathrm{vec}(V)]^\top \in \mathbb{R}^{2Ln}$, $\nabla_\theta F \in \mathbb{R}^{(2Ln+d) \times 2Ln}$, $\nabla_\omega F \in \mathbb{R}^{(2Ln+d) \times (2Ln+d)}$.

后文中为说话方便, 记 $\boldsymbol{\psi} := \nabla_\omega \mathcal{L}_{\mathrm{val}} \in \mathbb{R}^{1 \times (d+Ln+Ln)}, ~K := \nabla_\omega F \in \mathbb{R}^{(2Ln+d) \times (2Ln+d)}$,  则 $(1)$ 可以写成:

$$
\nabla_\theta \mathcal{L}_{\mathrm{val}} = - {\boldsymbol{\psi}} \cdot {K^{-1} \cdot \nabla_\theta F}\quad (2)
$$

其中:

- $K = \begin{bmatrix}I_d & \mathbf{0} & \bar{\mathsf{U}}_{(3)} \\[6pt]-\,\mathcal{D}_{\Lambda^*}\,\bar{\mathsf{U}}_{(3)}^\top & \mathcal{D}_{\Lambda^*} & \mathcal{D}_{2} \\[6pt]\mathbf{0} & I_{Ln} - \mathcal{D}_{\Lambda^*} & -\,\mathcal{D}_{\Pi^*}\end{bmatrix} \subset\begin{bmatrix}\mathbb{R}^{d \times d} & \mathbb{R}^{d \times (Ln)} & \mathbb{R}^{d \times (Ln)} \\[6pt]\mathbb{R}^{(Ln) \times d} & \mathbb{R}^{(Ln) \times (Ln)} & \mathbb{R}^{(Ln) \times (Ln)} \\[6pt]\mathbb{R}^{(Ln) \times d} & \mathbb{R}^{(Ln) \times (Ln)} & \mathbb{R}^{(Ln) \times (Ln)}\end{bmatrix}$
- $\boldsymbol{\psi} = \frac{\partial \mathcal{L}_{\mathrm{val}}}{\partial \omega} = \begin{bmatrix}  \nabla_{\boldsymbol{\beta}} \mathcal{L}_{\mathrm{val}} &\nabla_{\mathrm{vec}(\Pi)} \mathcal{L}_{\mathrm{val}} & \nabla_{\mathrm{vec}(\Lambda)} \mathcal{L}_{\mathrm{val}} \end{bmatrix} := \begin{bmatrix}  \psi_\beta & \psi_\Pi & \psi_\Lambda \end{bmatrix} \in \mathbb{R}^{1 \times (d+Ln+Ln)}$

在 $(2)$ 中, 我们记 $\boldsymbol{v} := (\boldsymbol{\psi} K^{-1})^\top := \begin{bmatrix} v_\beta \\ v_\Pi \\ v_\Lambda \end{bmatrix} \in \mathbb{R}^{d+Ln+Ln}$, 则我们需要求解的线性系统为:

$$
K^\top \boldsymbol{v} = \boldsymbol{\psi}^\top\\
\Leftrightarrow
\begin{bmatrix}
I_d & (-\bar{\mathsf{U}}_{(3)}\,\mathcal{D}_{\Lambda^*})_{d \times Ln} & \mathbf{0}_{d \times Ln} \\[6pt]
\mathbf{0}_{Ln \times d} & \mathcal{D}_{\Lambda^*} & I_{Ln} - \mathcal{D}_{\Lambda^*} \\[6pt]
(\bar{\mathsf{U}}_{(3)}^\top)_{Ln\times d} & \mathcal{D}_{2} & -\,\mathcal{D}_{\Pi^*}
\end{bmatrix}
\begin{bmatrix} v_{\beta} \\ v_{\Pi} \\ v_{\Lambda} \end{bmatrix} = 
\begin{bmatrix}\psi_\beta \\ \psi_\Pi \\ \psi_\Lambda\end{bmatrix}
$$

当我们解出 $\boldsymbol{v}$ 后, 我们就可以得到最终的梯度:

$$
\nabla_\theta \mathcal{L}_{\mathrm{val}} = - (\nabla_\theta F)^\top \boldsymbol{v}
$$

这就是所谓 Vector-Jacobian 策略 (VJP).

**因此, $(3)$ 的高效求解就是我们目前最核心的问题.**

### 线性系统详细求解过程

我们按照 $(3)$ 中的分块逻辑进行拆解, 得到:

$$
\begin{aligned}
I_d v_\beta - \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} v_\Pi &= \psi_\beta \quad \mathrm{(a)}\\
\mathcal{D}_{\Lambda^*} v_\Pi + (I_{Ln} - \mathcal{D}_{\Lambda^*}) v_\Lambda &= \psi_\Pi \quad \mathrm{(b)}\\
\bar{\mathsf{U}}_{(3)}^\top v_\beta + \mathcal{D}_{2} v_\Pi - \mathcal{D}_{\Pi^*} v_\Lambda &= \psi_\Lambda \quad \mathrm{(c)}
\end{aligned}
$$

- 通过对 $(a)$ 式进行变形, 我们可以得到:

$$
v_\beta = \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} v_\Pi + \psi_\beta \quad \mathrm{(a_1)}
$$

- 将 $(a_1)$ 代入 $(c)$ 式中, 我们可以得到:

$$
\begin{aligned}
\bar{\mathsf{U}}_{(3)}^\top (\mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} v_\Pi + \psi_\beta) + \mathcal{D}_{2} v_\Pi - \mathcal{D}_{\Pi^*} v_\Lambda &= \psi_\Lambda \\
\Leftrightarrow \bar{\mathsf{U}}_{(3)}^\top \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} v_\Pi + \bar{\mathsf{U}}_{(3)}^\top \psi_\beta + \mathcal{D}_{2} v_\Pi - \psi_\Lambda  &=  \mathcal{D}_{\Pi^*} v_\Lambda 
\end{aligned}
$$

又由于 $\mathcal{D}_{\Pi^*}$ 是对角矩阵, 我们可以将其移到等式的右侧, 得到:

$$
\begin{aligned}
v_\Lambda &= \mathcal{D}_{\Pi^*}^{-1} \left( \bar{\mathsf{U}}_{(3)}^\top \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} v_\Pi + \bar{\mathsf{U}}_{(3)}^\top \psi_\beta + \mathcal{D}_{2} v_\Pi - \psi_\Lambda \right)\\
&= \mathcal{D}_{\Pi^*}^{-1} \left( \bar{\mathsf{U}}_{(3)}^\top \psi_\beta - \psi_\Lambda + \left( \bar{\mathsf{U}}_{(3)}^\top \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} + \mathcal{D}_{2} \right) v_\Pi \right) \quad \mathrm{(c{a_1})}
\end{aligned}
$$

- 将 $(c{a_1})$ 代入 $(b)$ 式中, 我们可以得到:

$$
\begin{aligned}
 (I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1} \left( \bar{\mathsf{U}}_{(3)}^\top \psi_\beta - \psi_\Lambda + \underbrace{\left( \bar{\mathsf{U}}_{(3)}^\top \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} + \mathcal{D}_{2} \right)}_{\triangleq \mathcal E} v_\Pi \right) = \psi_\Pi - \mathcal{D}_{\Lambda^*} v_\Pi\\
\end{aligned}
$$

记 $\mathcal{E} = \bar{\mathsf{U}}_{(3)}^\top \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} + \mathcal{D}_{2} \in \mathbb{R}^{Ln \times Ln}$, 则对上式继续变形:

$$
\begin{aligned}
(I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1} \left( \mathcal{E} v_\Pi + \bar{\mathsf{U}}_{(3)}^\top \psi_\beta - \psi_\Lambda \right) = \psi_\Pi - \mathcal{D}_{\Lambda^*} v_\Pi\\
\Leftrightarrow (I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1} \mathcal E v_\Pi + (I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1}(\bar{\mathsf{U}}_{(3)}^\top \psi_\beta - \psi_\Lambda) + \mathcal{D}_{\Lambda^*} v_\Pi = \psi_\Pi \\
\Leftrightarrow \underbrace{\left[(I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1} \mathcal E  + \mathcal{D}_{\Lambda^*}\right]}_{\mathcal S}v_\Pi = \psi_\Pi -  (I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1}(\bar{\mathsf{U}}_{(3)}^\top \psi_\beta - \psi_\Lambda) 
\end{aligned}
$$

记 $\mathcal{S} = (I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1} \mathcal E  + \mathcal{D}_{\Lambda^*} \in \mathbb{R}^{Ln \times Ln}$, 则我们可以得到主线性系统, 这是一个只和一个变量 $v_\Pi$ 相关的线性系统:

$$
\mathcal{S} \cdot v_\Pi = \psi_\Pi -  (I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1}(\bar{\mathsf{U}}_{(3)}^\top \psi_\beta - \psi_\Lambda) \quad \mathrm{(d)}
$$

- $\mathrm{(d)}$ 式中其只含有单一变量 $v_\Pi$, 一旦我们将其求出, 就可以求解其他变量:
  - 利用方程 $\mathrm{(a_1)}$ 求出: $v_\beta = \psi_\beta + \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} v_\Pi$
  - 利用方程 $\mathrm{(c{a_1})}$ 求出: $v_\Lambda = \mathcal{D}_{\Pi^*}^{-1} \left( \bar{\mathsf{U}}_{(3)}^\top \psi_\beta - \psi_\Lambda + \mathcal{E} v_\Pi \right)$

### 借助 Woodbury 公式求解主线性系统

分析我们当前的主线性系统:

$$
\mathcal{S} \cdot v_\Pi = \underbrace{\psi_\Pi -  (I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1}(\bar{\mathsf{U}}_{(3)}^\top \psi_\beta - \psi_\Lambda)}_{\triangleq c}
$$

- 其中 $\mathcal{S} = (I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1} \mathcal{E} + \mathcal{D}_{\Lambda^*} \in\mathbb{R}^{Ln \times Ln}$, $\mathcal{E} = \bar{\mathsf{U}}_{(3)}^\top \mathcal{D}_{\Lambda^*} \bar{\mathsf{U}}_{(3)} + \mathcal{D}_{2} \in \mathbb{R}^{Ln \times Ln}$.
- 我们不能直接求解 $\mathcal{S}$ 的逆, 因为其是一个 $Ln \times Ln$ 的矩阵, 直接求逆代价较高. 但我们可以利用 Woodbury 公式来简化计算.

将 $\mathcal E$ 的具体表达代入, 观察 $\mathcal{S}$ 的结构, 可以发现其本质上是一个**对角矩阵+低秩扰动**的形式:

$$
\mathcal S
=\underbrace{\Bigl[\mathcal D_{\Lambda^*}+
(I-\mathcal D_{\Lambda^*})\mathcal D_{\Pi^*}^{-1}\mathcal D_2\Bigr]}_{=: \mathcal A\;(\text{对角})}
+\underbrace{(I-\mathcal D_{\Lambda^*})\mathcal D_{\Pi^*}^{-1}\bar{\mathsf U}_{(3)}^\top\mathcal D_{\Lambda^*}\bar{\mathsf U}_{(3)}}_{=: \mathcal P\mathcal Q^\top\;(\text{秩}\le d)}
$$

- 其中 $\operatorname{rank}(\mathcal P\mathcal Q^\top)\le d$ 是因为: 由于$\bar{\mathsf U}_{(3)}=\bigl[u_{l,i}\mathbf x_i\bigr]_{(l,i)} \in\mathbb R^{d\times Ln}$, 故 $\operatorname{rank}\bigl(\bar{\mathsf U}_{(3)}\bigr)\le d$. 因此 $\mathcal{P,Q}$ 作为 $\bar{\mathsf U}_{(3)}$ 乘以对角矩阵的相关变换, 依然满足 $\operatorname{rank}(\mathcal P)\le d,\;\operatorname{rank}(\mathcal Q)\le d$. 进而 $\operatorname{rank}(\mathcal P\mathcal Q^\top)\le\min\{\operatorname{rank}(\mathcal P),\,\operatorname{rank}(\mathcal Q)\}\le d$.
- 其恰好满足 Woodbury 公式的形式: 若有矩阵 $\mathcal{S} = \mathcal{A} + \mathcal{P}\mathcal{Q}^\top$, 其中 $\mathcal{A}$ 是一个对角矩阵，$\mathcal{P}$ 和 $\mathcal{Q}$ 分别是 rank-$d$ 的矩阵, 则其逆可以通过 Woodbury 公式高效计算:

$$
\mathcal{S}^{-1} = \mathcal{A}^{-1} - \mathcal{A}^{-1} \mathcal{P} \left(\underbrace{\mathcal{Q}^\top \mathcal{A}^{-1} \mathcal{P} + I_d}_{\triangleq\mathcal{B}}\right)^{-1} \mathcal{Q}^\top \mathcal{A}^{-1} \quad (\star)
$$

    - 其中$\mathcal{A}^{-1}$ 是对角矩阵的逆，计算代价较低.
    - 核心的计算瓶颈现在被控制在了 $d$ 维度空间中 $\mathcal{B}:=(\mathcal{Q}^\top \mathcal{A}^{-1} \mathcal{P}+ I_d)\in\mathbb{R}^{d\times d}$ 的逆矩阵计算上, 主要计算开销为 $\mathcal{O}(d^3)$.

下面我们将 $\mathcal{S}$ 在本任务中的具体形式进行明确, 以便代入 Woodbury 公式 ($\star$).

令:

- $\mathcal P:=(I-\mathcal D_{\Lambda^*})\mathcal D_{\Pi^*}^{-1}\bar{\mathsf U}_{(3)}^\top\mathcal D_{\Lambda^*}^{\frac12}\in\mathbb R^{Ln\times d}$,
- $\mathcal Q:= \bar{\mathsf U}_{(3)}^\top\mathcal D_{\Lambda^*}^{\frac12}\in\mathbb R^{Ln\times d}$,
- $\mathcal A:=\mathcal D_{\Lambda^*}+(I-\mathcal D_{\Lambda^*})\mathcal D_{\Pi^*}^{-1}\mathcal D_2\in\mathbb R^{Ln\times Ln}$.

则可对应 $(\star)$ 中的 $\mathcal{A}, \mathcal{P}, \mathcal{Q}$ 进行带入求解. 其求解细节如下.

1. $\mathcal{A}^{-1}$ 是对角矩阵的逆, 求逆的过程较为简单, 不过需要注意其计算的数值稳定性.
2. 核心问题在于计算第二个部分的逆. 为了求解该部分, 还需进一步定义:

   $$
   begin{aligned}
   \mathcal{B}&:=I_d+\mathcal Q^\top\mathcal A^{-1}\mathcal P \\
   &=I_d+ \bar{\mathsf{U}}_{(3)} \mathcal{D}_{\Lambda^*}^{\frac12} \mathcal{A}^{-1} (I-\mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1} \bar{\mathsf{U}}_{(3)}^\top \mathcal{D}_{\Lambda^*}^{\frac12} \\
   &= I_d + \mathcal{D}_{\Lambda^*}(I-\mathcal{D}_{\Lambda^*}) \mathcal{A}^{-1} \mathcal{D}_{\Pi^*}^{-1} \mathsf{\bar{U}}_{(3)} \mathsf{\bar{U}}_{(3)}^\top \\
   &= I_d + \text{DiagMatrix}\cdot \bar{\mathsf{U}}_{(3)} \mathsf{U}_{(3)}^\top \in \mathbb{R}^{d\times d}
   \end{aligned}
   $$

   - 构造 $\mathcal{B}$ 本身的计算复杂度讨论如下:
     - 计算 $\mathcal{Y} := \mathcal{A}^{-1} \mathcal{P}$ 为 $\mathcal{O}(Ln\cdot d)$
     - 计算 $\mathcal{Q}^\top \mathcal{Y}$ 为 $\mathcal{O}(Ln\cdot d^2)$
   - $\mathcal{B}$ 本身是对称正定的 (SPD) 矩阵. 其对称性与正定性都显然成立.
   - 因此对于 SPD 矩阵 $\mathcal{B}$，我们可以使用 Cholesky 分解来高效求解其逆矩阵 (求解方程组).
     - 为确保计算的数值稳定性, 可以首先进行对角增强: $\mathcal{B} \leftarrow \mathcal{B} + \epsilon I_d$，其中 $\epsilon \sim 10^{-10}$ 是一个小的正数.
     - 在具体实现中, 由于我们需要求解的是主线性系统 $\mathcal{S} v_\Pi = c$, 因此根据前面的 Woodbury 公式 $v_\Pi = \mathcal{S}^{-1} c = \mathcal{A}^{-1} c - \mathcal{A}^{-1} \mathcal{P} \mathcal{B}^{-1} \mathcal{Q}^\top \mathcal{A}^{-1} c := z_0 - \mathcal{Y} \eta$ (其中进一步记 $z_0 := \mathcal{A}^{-1} c \in \mathbb{R}^{Ln}$, $\eta := \mathcal{B}^{-1} \mathcal{Q}^\top z_0 \in \mathbb{R}^{d}$, $\mathcal{Y} := \mathcal{A}^{-1} \mathcal{P} \in \mathbb{R}^{Ln \times d}$), 我们利用 Cholesky 分解 $\mathcal{B} = L L^\top$，其中 $L$ 是下三角矩阵来计算 $\eta$ 的值 (其余部分可以直接计算):
       - 首先解方程 $L y = \mathcal{Q}^\top z_0$，得到 $y$.
       - 然后解方程 $L^\top \eta = y$，得到 $\eta$.

综合上述层层拆解, 我们对于主线性系统 $\mathcal{S} v_\Pi = c$ 的求解过程可以总结为:

$$
\begin{aligned}
v_\Pi &= \mathcal{S}^{-1} c \quad \small \text{(apparently)}\\
&= \mathcal{A}^{-1} c - \mathcal{A}^{-1} \mathcal{P}\mathcal{B}^{-1} \mathcal{Q}^\top \mathcal{A}^{-1} c\quad \small \text{(Woodbury Eq.)}\\
&= \mathcal{A}^{-1} c - \mathcal{A}^{-1} \mathcal{P} (L L^\top)^{-1} \mathcal{Q}^\top \mathcal{A}^{-1} c \quad \small \text{(Cholesky Decomposition)}\\
\end{aligned}
$$

如前所述, 在得到 $v_\Pi$ 后, 我们可以进一步求解 $v_\beta$ 和 $v_\Lambda$. 而一旦求解出这些变量后, 我们便得到了 $\boldsymbol{v} = [v_\Pi^\top, v_\beta^\top, v_\Lambda^\top]^\top$. 然后便可以进一步代回到 $(2)$ 式中, 得到最终的梯度 $\nabla_\theta \mathcal{L}_{\mathrm{val}}$.

### Woodbury 公式的更紧凑版本

回顾我们得到的主线性系统:

$$
\mathcal{S} \cdot v_\Pi = \underbrace{\psi_\Pi -  (I_{Ln} - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1}(\bar{\mathsf{U}}_{(3)}^\top \psi_\beta - \psi_\Lambda)}_{\triangleq c}
$$

其中

$$
\mathcal S
=\underbrace{\Bigl[\mathcal D_{\Lambda^*}+
(I-\mathcal D_{\Lambda^*})\mathcal D_{\Pi^*}^{-1}\mathcal D_2\Bigr]}_{=: \mathcal A\;(\text{对角})}
+\underbrace{(I-\mathcal D_{\Lambda^*})\mathcal D_{\Pi^*}^{-1}\bar{\mathsf U}_{(3)}^\top\mathcal D_{\Lambda^*}\bar{\mathsf U}_{(3)}}_{=: \mathcal P\mathcal Q^\top\;(\text{秩}\le d)}
$$

进一步, 令 $E := (I - \mathcal{D}_{\Lambda^*}) \mathcal{D}_{\Pi^*}^{-1} \mathcal{D}_{\Lambda^*}$, 则:

$$
\begin{aligned}
\mathcal{S}
&= \mathcal{A} + E\, \bar{\mathsf{U}}_{(3)}^\top \bar{\mathsf{U}}_{(3)} \\
&= E \left( E^{-1} \mathcal{A} + \bar{\mathsf{U}}_{(3)}^\top \bar{\mathsf{U}}_{(3)} \right)
=: E \cdot \widetilde{\mathcal{S}}.
\end{aligned}$$ 

称 $\widetilde{\mathcal{S}} = E^{-1} \mathcal{A} + \bar{\mathsf{U}}_{(3)}^\top \bar{\mathsf{U}}_{(3)}$ 为正规化的主线性系统, 其中 $E^{-1}\mathcal{A} = \mathcal{D}_{\Pi^*} (I - \mathcal{D}_{\Lambda^*})^{-1} + \mathcal{D}_2\mathcal{D}_{\Lambda^*}^{-1}=:D$. 故原先的 $\mathcal{S} v_\Pi = c$ 可以改写为 $\widetilde{\mathcal{S}} v_\Pi = E^{-1} c =: \tilde{c}$.

现在我们有:
$$\widetilde{\mathcal{S}} = D + \bar{U}^\top \bar{U},\qquad D := E^{-1} \mathcal{A},\quad \bar{U} := \bar{\mathsf{U}}_{(3)} \in \mathbb{R}^{d \times Ln}$$

上式满足 SMW 条件, 因此利用对称的 Woodbury 公式:
$$\widetilde{\mathcal{S}}^{-1}
= D^{-1} - D^{-1} \bar{U}^\top (I + \bar{U} D^{-1} \bar{U}^\top)^{-1} \bar{U} D^{-1}
$$

再代回主线性系统中, 有:

$$
\begin{aligned}
v_\Pi &= \widetilde{\mathcal{S}}^{-1} \tilde{c} \\
&= D^{-1} \tilde{c} - D^{-1} \bar{U}^\top (I + \bar{U} D^{-1} \bar{U}^\top)^{-1}[ \bar{U} (D^{-1} \tilde{c})] \\
&:= D^{-1} \tilde{c} - D^{-1} \bar{U}^\top (I + \bar{U} D^{-1} \bar{U}^\top)^{-1} w \\
&= D^{-1} \tilde{c} - D^{-1} \bar{U}^\top \text{CholeskySolve}\left(I + \bar{U} D^{-1} \bar{U}^\top, w\right)
\end{aligned}
$$

其中最后一个等式中的求逆是通过 Cholesky 分解来求解的, 其合法性由 $I + \bar{U} D^{-1} \bar{U}^\top$ 的对称正定性保证.
