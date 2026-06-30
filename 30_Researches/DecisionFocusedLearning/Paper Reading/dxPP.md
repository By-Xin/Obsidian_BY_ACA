# A Penalty Approach for Differentiation Through Black-Box Quadratic  Programming Solvers

## Introduction


考虑如下凸二次规划问题
$$
\begin{aligned}
\mathbf{z}^\star(\boldsymbol{\theta}) = \arg\min_{\mathbf{z} \in \mathbb{R}^n} \quad & \frac{1}{2}\mathbf{z}^\top \mathbf{P}(\boldsymbol{\theta})\mathbf{z} + \mathbf{q}(\boldsymbol{\theta})^\top \mathbf{z} \\
\text{s.t.} \quad & \mathbf{A}(\boldsymbol{\theta})\mathbf{z} = \mathbf{b}(\boldsymbol{\theta}) \\
& \mathbf{C}(\boldsymbol{\theta})\mathbf{z} \leq \mathbf{d}(\boldsymbol{\theta})
\end{aligned}
$$
其中 $\mathbf{P}(\boldsymbol{\theta}) \in \mathbb{S}_{++}^n$ 是实对称正定矩阵, $\mathbf{q}(\boldsymbol{\theta}) \in \mathbb{R}^n$ 是线性项, $\mathbf{A}(\boldsymbol{\theta}) \in \mathbb{R}^{p \times n}$, $\mathbf{b}(\boldsymbol{\theta}) \in \mathbb{R}^p$, $\mathbf{C}(\boldsymbol{\theta}) \in \mathbb{R}^{m \times n}$, $\mathbf{d}(\boldsymbol{\theta}) \in \mathbb{R}^m$ 是对应约束系数. $\boldsymbol{\theta}$ 是参数向量, 这里认为是上一层神经网络的输出.

每当给定一个参数 $\boldsymbol{\theta}$ 时, 我们就定义了一个优化问题. 在网络中, 我们希望通过 端到端的方法进行反向传播, 因此需要计算 $\partial_{\boldsymbol{\theta}} \mathbf{z}^\star(\boldsymbol{\theta})$. 传统的方法要通过 KKT 条件构建等式关系, 并使用隐函数定理进行构造. 本文的主要贡献是将前向 forward pass 封装给成熟的 QP solver, 并着重讨论了反向传播的求解. 

## Methodology

### Traditional Differentiation Through KKT Conditions

对于上述 QP 问题, 引入拉格朗日乘子 $\boldsymbol{\nu} \in \mathbb{R}^p$ 和 $\boldsymbol{\mu} \in \mathbb{R}^m$, 故根据 KKT 条件, $(\mathbf{z}^\star, \boldsymbol{\nu}^\star, \boldsymbol{\mu}^\star)$ 是上述问题的 primal-dual optimal solution, 当且仅当满足以下条件:
$$
\begin{aligned}
\mathbf{P}(\boldsymbol{\theta})\mathbf{z}^\star + \mathbf{q}(\boldsymbol{\theta}) + \mathbf{A}(\boldsymbol{\theta})^\top \boldsymbol{\nu}^\star + \mathbf{C}(\boldsymbol{\theta})^\top \boldsymbol{\mu}^\star &= 0 \\
\mathbf{A}(\boldsymbol{\theta})\mathbf{z}^\star - \mathbf{b}(\boldsymbol{\theta}) &= 0 \\
\mathbf{C}(\boldsymbol{\theta})\mathbf{z}^\star - \mathbf{d}(\boldsymbol{\theta}) &\leq 0 \\
\boldsymbol{\mu}^\star &\geq 0 \\
\operatorname{diag}(\boldsymbol{\mu}^\star)(\mathbf{C}(\boldsymbol{\theta})\mathbf{z}^\star - \mathbf{d}(\boldsymbol{\theta})) &= 0
\end{aligned}
$$
因此整理其中全部的等式约束, 可以得到一个非线性方程组:
$$
\mathbf{U}(\mathbf{z}, \boldsymbol{\nu}, \boldsymbol{\mu}; \boldsymbol{\theta}) = 
\begin{bmatrix}
\mathbf{P} \mathbf{z} + \mathbf{q} + \mathbf{A}^\top \boldsymbol{\nu} + \mathbf{C}^\top \boldsymbol{\mu} \\
\mathbf{A}\mathbf{z} - \mathbf{b} \\
\operatorname{diag}(\boldsymbol{\mu})(\mathbf{C}\mathbf{z} - \mathbf{d})
\end{bmatrix} = 0
$$
注意到, 这里省略了参数 $\boldsymbol{\theta}$ 的依赖关系, 然而事实上其中的每一项, 包括 $\mathbf{P}, \mathbf{q}, \mathbf{A}, \mathbf{b}, \mathbf{C}, \mathbf{d}$ 以及 accordingly 得到的最优解 $(\mathbf{z}^\star, \boldsymbol{\nu}^\star, \boldsymbol{\mu}^\star)$ 都是 $\boldsymbol{\theta}$ 的函数. 换言之, 上述 KKT 系统等于零应当是对任意给定的 $\boldsymbol{\theta}$ 都成立的 (当然这里假设我们给出的 $\boldsymbol{\theta}$ 是可行的). 换言之, $\mathbf{U}(\boldsymbol{\theta})\equiv 0$. 

因此求解 $\mathbf{U}$ 关于 $\boldsymbol{\theta}$ 的导数:
$$
\frac{\partial \mathbf{U}}{\partial (\mathbf{z}, \boldsymbol{\nu}, \boldsymbol{\mu})} 
\begin{bmatrix}
\partial_{\boldsymbol{\theta}} \mathbf{z}^\star \\
\partial_{\boldsymbol{\theta}} \boldsymbol{\nu}^\star \\
\partial_{\boldsymbol{\theta}} \boldsymbol{\mu}^\star
\end{bmatrix} +
\frac{\partial \mathbf{U}}{\partial \boldsymbol{\theta}} = \mathbf{0}
$$

化简整理有
$$
\begin{bmatrix}
\mathbf{P} & \mathbf{A}^\top & \mathbf{C}^\top \\
\mathbf{A} & 0 & 0 \\
\operatorname{diag}(\boldsymbol{\mu}^\star)\mathbf{C} & 0 & \operatorname{diag}(\mathbf{C}\mathbf{z}^\star - \mathbf{d})
\end{bmatrix}

\begin{bmatrix}
\partial_{\boldsymbol{\theta}} \mathbf{z}^\star \\
\partial_{\boldsymbol{\theta}} \boldsymbol{\nu}^\star \\
\partial_{\boldsymbol{\theta}} \boldsymbol{\mu}^\star
\end{bmatrix} = -
\begin{bmatrix}
\partial_{\boldsymbol{\theta}} \mathbf{P}\,\mathbf{z}^\star + \partial_{\boldsymbol{\theta}} \mathbf{q} + (\partial_{\boldsymbol{\theta}} \mathbf{A})^\top \boldsymbol{\nu}^\star + (\partial_{\boldsymbol{\theta}} \mathbf{C})^\top \boldsymbol{\mu}^\star \\
\partial_{\boldsymbol{\theta}} \mathbf{A}\,\mathbf{z}^\star - \partial_{\boldsymbol{\theta}} \mathbf{b} \\
\operatorname{diag}(\boldsymbol{\mu}^\star)(\partial_{\boldsymbol{\theta}} \mathbf{C}\,\mathbf{z}^\star - \partial_{\boldsymbol{\theta}} \mathbf{d})
\end{bmatrix}
$$

若在 LICQ 和 strict complementarity 条件下, 上述线性系统是非退化的. 在这种情况下, 考虑 active constraint 构成的 active set:
$$
\mathcal{A} = \{i  \in [m] : \mathbf{C}_i \mathbf{z}^\star - \mathbf{d}_i = 0\}
$$
记对应的分块为
$$
\mathbf{C}_\mathcal{A}  \mathbf{z}^* = \mathbf{d}_\mathcal{A},
$$


进而构成简化后的线性系统:
$$
\begin{bmatrix}
\mathbf{P} & \mathbf{A}^\top & \mathbf{C}_\mathcal{A}^\top \\
\mathbf{A} & 0 & 0 \\
\mathbf{C}_\mathcal{A} & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\partial_{\boldsymbol{\theta}} \mathbf{z}^\star \\
\partial_{\boldsymbol{\theta}} \boldsymbol{\nu}^\star \\
\partial_{\boldsymbol{\theta}} \boldsymbol{\mu}_\mathcal{A}^\star
\end{bmatrix} = -
\begin{bmatrix}
\partial_{\boldsymbol{\theta}} \mathbf{P}\,\mathbf{z}^\star + \partial_{\boldsymbol{\theta}} \mathbf{q} + (\partial_{\boldsymbol{\theta}} \mathbf{A})^\top \boldsymbol{\nu}^\star + (\partial_{\boldsymbol{\theta}} \mathbf{C}_\mathcal{A})^\top \boldsymbol{\mu}_\mathcal{A}^\star \\
\partial_{\boldsymbol{\theta}} \mathbf{A}\,\mathbf{z}^\star - \partial_{\boldsymbol{\theta}} \mathbf{b} \\
\partial_{\boldsymbol{\theta}} \mathbf{C}_\mathcal{A}\,\mathbf{z}^\star - \partial_{\boldsymbol{\theta}} \mathbf{d}_\mathcal{A}
\end{bmatrix}
$$

对于这个系统的求解即完成了 $\partial_{\boldsymbol{\theta}} \mathbf{z}^\star$ 的求解. 

### Penalty Reformulation and Implicit Differentiation

一个 alternative 的方法是, 我们可以将原始的 QP 问题 reformulate 成一个 penalty problem, 从而转化为一个无约束问题, 进而抛弃掉 KKT 条件的求解. 

给定一个 $\boldsymbol{\theta}$, 考虑原问题:
$$
\begin{aligned}
\min_{\mathbf{z} \in \mathbb{R}^n} \quad & f(\mathbf{z}) = \frac{1}{2}\mathbf{z}^\top \mathbf{P} \mathbf{z} + \mathbf{q}^\top \mathbf{z} \\
\text{s.t.} \quad & \mathbf{A}\mathbf{z} = \mathbf{b} \\
& \mathbf{C}\mathbf{z} \leq \mathbf{d}
\end{aligned}
$$

定义 exact penalty objective:
$$
F(\mathbf{z}; \boldsymbol{\theta}, \rho, \alpha) = f(\mathbf{z}) + \rho \|\mathbf{A}\mathbf{z} - \mathbf{b}\|_1 + \alpha \|(\mathbf{C}\mathbf{z} - \mathbf{d})_+\|_1
$$

文中 Proposition 3.1 证明, 当 $\rho \geq \|\boldsymbol{\nu}^\star\|_\infty$ 且 $\alpha \geq \|\boldsymbol{\mu}^\star\|_\infty$ 时, 原问题的最优解 $\mathbf{z}^\star$ 也是 penalty problem 的最优解. 

再进一步, 对于这个 exactly penalty objective, 其由于使用了 $\ell_1$ norm, 因此是 non-smooth 的. 故考虑使用 soft plus 函数进行平滑化:
$$
p_\delta(t) = \delta \log(1 + \exp(t/\delta)), \quad \delta > 0
$$
得到 smoothed penalty objective:
$$
\Phi_\delta(\mathbf{z}; \boldsymbol{\theta}) = f(\mathbf{z}) + \alpha \sum_{i=1}^m p_\delta( (\mathbf{C}\mathbf{z} - \mathbf{d})_i) + \rho \sum_{j=1}^p \left(p_\delta((\mathbf{A}\mathbf{z} - \mathbf{b})_j) + p_\delta((\mathbf{b} - \mathbf{A}\mathbf{z})_j)\right)
$$
- 其中, softplus 是 $(\cdot)_+$ 的平滑化. 因此对于 $|u|$, 其应当分解为 $|u| = (u)_+ + (-u)_+$, 因此得到后面的两项.

很直接地, 该非约束问题的最优解 $\mathbf{z}_\delta^\star(\boldsymbol{\theta})$ 满足一阶最优性条件:
$$
\nabla_\mathbf{z} \Phi_\delta(\mathbf{z}_\delta^\star; \boldsymbol{\theta}) = 0
$$
这便构建出了一个替代的等式关系. 因此根据隐函数定理, 有
$$
\partial_{\boldsymbol{\theta}} \mathbf{z}_\delta^\star(\boldsymbol{\theta}) = - \left(\nabla_{\mathbf{z}\mathbf{z}}^2 \Phi_\delta(\mathbf{z}_\delta^\star; \boldsymbol{\theta})\right)^{-1} \nabla_{\mathbf{z}\boldsymbol{\theta}}^2 \Phi_\delta(\mathbf{z}_\delta^\star; \boldsymbol{\theta})
$$

此时, 原先的对于 KKT 的大线性系统的求解, 转化为了对 smooth penalty 的 Hessian 的求解. 


### Plug-in Sensitivity and Consistency Analysis

上述的 smooth penalty 确实给出了一个绕开 KKT 线性系统的方式. 然而在 forward pass 中, 已经使用 QP solver 求解了原始 QP 的 $\mathbf{z}^\star, \boldsymbol{\nu}^\star, \boldsymbol{\mu}^\star$. 因此在 backward pass 中, 我们期望能够直接复用这些准确的解.  简而言之, 我们要把 $\mathbf{z}^\star$ 直接代入到上面的隐函数定理中, 即:
$$
\partial_{\boldsymbol{\theta}} \mathbf{z}^\star(\boldsymbol{\theta}) = - \left(\nabla_{\mathbf{z}\mathbf{z}}^2 \Phi_\delta(\mathbf{z}^\star; \boldsymbol{\theta})\right)^{-1} \nabla_{\mathbf{z}\boldsymbol{\theta}}^2 \Phi_\delta(\mathbf{z}^\star; \boldsymbol{\theta})
$$
我们接下来要证明这一操作的合理性.

具体的, 在得到 $\mathbf{z}^\star$ 后, 检查各个约束的活跃性, 即检查 $\mathbf{C}\mathbf{z}^\star - \mathbf{d}$ 的符号. 若 $\mathbf{C}_\mathcal{A}\mathbf{z}^\star - \mathbf{d}_\mathcal{A} = 0$, 则认为该约束是 active 的, 否则为 inactive. 进而构建出 active set $\mathcal{A}$. 将所有的 equality constraints 和 active inequalities 整合在一起, 记为:
$$
g(\mathbf{z}; \boldsymbol{\theta}) =
\begin{bmatrix}
\mathbf{A}\mathbf{z} - \mathbf{b} \\
\mathbf{C}_\mathcal{A}\mathbf{z} - \mathbf{d}_\mathcal{A}
\end{bmatrix} \in \mathbb{R}^{p + |\mathcal{A}|}
$$
且注意到在 $\mathbf{z}^\star$ 处, 有
$$
g(\mathbf{z}^\star; \boldsymbol{\theta}) = 0
$$

则 $g$ 关于 $\mathbf{z}$ 的雅可比矩阵:
$$
\nabla_\mathbf{z} g(\mathbf{z}, \boldsymbol{\theta}) =
\begin{bmatrix}
\mathbf{A} \\
\mathbf{C}_\mathcal{A}
\end{bmatrix} \in \mathbb{R}^{(p + |\mathcal{A}|) \times n}
$$


计算 $\nabla_{\mathbf{z}\mathbf{z}}^2 \Phi_\delta(\mathbf{z}^\star; \boldsymbol{\theta})$:
$$
\nabla_{\mathbf{z}\mathbf{z}}^2 \Phi_\delta(\mathbf{z}^\star; \boldsymbol{\theta}) = 
\mathbf{P} + \frac{\rho}{2\delta} \mathbf{A}^\top \mathbf{A}+ \frac{\alpha}{4\delta} \mathbf{C}_\mathcal{A}^\top \mathbf{C}_\mathcal{A} + \mathcal{E}_\delta = \mathbf{P} + \frac{1}{\delta} \begin{bmatrix} \mathbf{A} \\ \mathbf{C}_\mathcal{A} \end{bmatrix}^\top
\begin{bmatrix}
\frac{\rho}{2} \mathbf{I}_p & 0 \\ 0 & \frac{\alpha}{4} \mathbf{I}_{|\mathcal{A}|}
\end{bmatrix}
\begin{bmatrix}
\mathbf{A} \\ \mathbf{C}_\mathcal{A}
\end{bmatrix} + \mathcal{E}_\delta
$$
- 其中:
    $$
    \mathcal{E}_\delta = \alpha \mathbf{C}_{\mathcal{I}}^\top \operatorname{diag}\left(
        p_\delta''\left((\mathbf{C}_\mathcal{I}\mathbf{z}^\star - \mathbf{d}_\mathcal{I}) \right)
    \right) \mathbf{C}_{\mathcal{I}} \to 0, \quad \text{as } \delta \to 0
    $$


计算 $\nabla_{\mathbf{z}\boldsymbol{\theta}}^2 \Phi_\delta(\mathbf{z}^\star; \boldsymbol{\theta})$:
$$\begin{aligned}
\nabla_{\mathbf z\boldsymbol\theta}^2
\Phi_\delta(\mathbf z^\star;\boldsymbol\theta)
=&\;
\nabla_{\mathbf z\boldsymbol\theta}^2
f(\mathbf z^\star;\boldsymbol\theta)
\\
&+
(\partial_{\boldsymbol\theta}\mathbf A)^\top
\left[
\rho\psi_\delta'(\mathbf A\mathbf z^\star-\mathbf b)
\right]
\\
&+
(\partial_{\boldsymbol\theta}\mathbf C_{\mathcal A})^\top
\left[
\alpha p_\delta'(\mathbf C_{\mathcal A}\mathbf z^\star-\mathbf d_{\mathcal A})
\right]
\\
&+
\mathbf A^\top
\operatorname{Diag}
\left[
\rho\psi_\delta''(\mathbf A\mathbf z^\star-\mathbf b)
\right]
\left[
(\partial_{\boldsymbol\theta}\mathbf A)\mathbf z^\star
-
\partial_{\boldsymbol\theta}\mathbf b
\right]
\\
&+
\mathbf C_{\mathcal A}^\top
\operatorname{Diag}
\left[
\alpha p_\delta''(\mathbf C_{\mathcal A}\mathbf z^\star-\mathbf d_{\mathcal A})
\right]
\left[
(\partial_{\boldsymbol\theta}\mathbf C_{\mathcal A})\mathbf z^\star
-
\partial_{\boldsymbol\theta}\mathbf d_{\mathcal A}
\right]
\\
&+
(\partial_{\boldsymbol\theta}\mathbf C_{\mathcal I})^\top
\left[
\alpha p_\delta'(\mathbf C_{\mathcal I}\mathbf z^\star-\mathbf d_{\mathcal I})
\right]
\\
&+
\mathbf C_{\mathcal I}^\top
\operatorname{Diag}
\left[
\alpha p_\delta''(\mathbf C_{\mathcal I}\mathbf z^\star-\mathbf d_{\mathcal I})
\right]
\left[
(\partial_{\boldsymbol\theta}\mathbf C_{\mathcal I})\mathbf z^\star
-
\partial_{\boldsymbol\theta}\mathbf d_{\mathcal I}
\right].
\end{aligned}
$$

进一步进行近似, 得到
$$\boxed{
\begin{aligned}
\nabla_{\mathbf z\boldsymbol\theta}^2
\Phi_\delta(\mathbf z^\star;\boldsymbol\theta)
\approx
&\;
\nabla_{\mathbf z\boldsymbol\theta}^2
f(\mathbf z^\star;\boldsymbol\theta)
+
(\partial_{\boldsymbol\theta}\mathbf A^\top)\boldsymbol\nu^\star
+
(\partial_{\boldsymbol\theta}\mathbf C_{\mathcal A}^\top)
\boldsymbol\mu_{\mathcal A}^\star
\\
&+
\frac{\rho}{2\delta}
\mathbf A^\top
\left[
(\partial_{\boldsymbol\theta}\mathbf A)\mathbf z^\star
-
\partial_{\boldsymbol\theta}\mathbf b
\right]
+
\frac{\alpha}{4\delta}
\mathbf C_{\mathcal A}^\top
\left[
(\partial_{\boldsymbol\theta}\mathbf C_{\mathcal A})\mathbf z^\star
-
\partial_{\boldsymbol\theta}\mathbf d_{\mathcal A}
\right].
\end{aligned}
}$$