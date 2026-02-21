# Optimality Conditions for Constrained Problems

> [!info] References
> - Lecture: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/
> - Reading: 最优化: 建模、算法与理论, 刘浩洋等, 5.5 小节.

## General Constrained Problems (No Convexity Assumption)

### First-Order Optimality Conditions

回顾, 考虑如下一般的含约束的优化问题 (不要求是凸的):
$$\begin{aligned}
& \min_{\mathbf{x}\in \mathbb{R}^n} && f(x) \\
& \text{subject to} && g_i(\mathbf{x}) \leq 0, i\in \mathcal{I} \\
& && h_j(\mathbf{x}) = 0, j\in \mathcal{E}
\end{aligned}$$

其 Lagrangian 函数为:
$$
L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i\in \mathcal{I}} \lambda_i g_i(\mathbf{x}) + \sum_{j\in \mathcal{E}} \nu_j h_j(\mathbf{x})
$$

其 Lagrange Dual Function 为:
$$
g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}\in \mathbb{R}^n} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
$$

#### Optimality Conditions by Tangent Cone

为定义可行域内的一系列点列的极限状态, 引入切向量和切锥的概念. 

***Definition* (Tangent Vector)**: 对于可行域 $\mathcal{X}$ 内的点列 $\{\mathbf{x}_k\}_{k=1}^\infty \subseteq \mathcal{X}\subset \mathbb{R}^n$, 其极限状态为 $\lim_{k\to\infty} \mathbf{x}_k = \mathbf{x}^*\subset \mathcal{X}$ (即该点列逼近 $\mathbf{x}^*$). 若存在向量 $\mathbf{d}\in \mathbb{R}^n$, 以及一个正数标量序列 $\{t_k\}_{k=1}^\infty$ 且 $t_k \to 0$ 使得:
$$
\lim_{k\to\infty} \frac{\mathbf{x}_k - \mathbf{x}^*}{t_k} = \mathbf{d}
$$
则称 $\mathbf{d}$ 为 $\mathbf{x}^*$ 处的切向量.

***Definition* (Tangent Cone)**: 对于上述点 $\mathbf{x}^*$ 处的全部切向量之集合, 称为该点处的切锥, 记作 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*)$.

- 切锥表示从当前位置出发, 所有能够满足约束条件的行动方向之集合. 

> [!example] **切锥的例子**
>
> ![Ref: 最优化: 建模、算法与理论, 刘浩洋等, 5.5 小节](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202602202338194.png)
>
> - 如上图所示. 图中两条曲线分别代表两条约束方程 $g_1(\mathbf{x})$ 和 $g_2(\mathbf{x})$ 的图像. 左侧为不等式约束, 图中深色阴影部分表示两约束方程构成的可行域 $\mathcal{X}$. 右侧为等式约束, 故可行域只有轮廓本身. 
>   - 对于不等式约束, 其切锥为整个深浅阴影区域, 为一个凸锥. 
>   - 对于等式约束, 其切锥只能取在左图轮廓线上, 即图中两条射线. 

***Theorem* (Optimality Conditions by Tangent Cone)**: 设 $x^*$ 是可行域 $\mathcal{X}$ 内的一个局部极小点. 若 $f$ 和 $g_i, h_j (\forall i,j)$ 在 $\mathbf{x}^*$ 处可微, 则有:
$$
\mathbf{d}^\top \nabla f(\mathbf{x}^*) \geq 0, \quad \forall \mathbf{d} \in \mathcal{T}_{\mathcal{X}}(\mathbf{x}^*)
$$

或等价地:
$$
\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) \cap \{\mathbf{d}\in \mathbb{R}^n \mid \mathbf{d}^\top \nabla f(\mathbf{x}^*) = 0\} = \emptyset
$$

- 其直观的理解为, 从最优点 $\mathbf{x}^*$ 出发, 所有能够满足约束条件的行动方向, 其都应当和梯度方向夹角为锐角; 即任何从最优点出发的可行方向都应当是上升方向. 
- *Proof.*
  - 用反证法, 假设在 $\mathbf{x}^*$ 处有 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) \cap \{\mathbf{d}\in \mathbb{R}^n \mid \mathbf{d}^\top \nabla f(\mathbf{x}^*) = 0\} \neq \emptyset$, 则记该集合中的某个向量为 $\mathbf{d}^*$.
  - 根据切向量的定义, 存在 $\{t_k\}_{k=1}^\infty$ 且 $t_k \to 0$ 以及对应的切向量 $\{\mathbf{d}_k\}_{k=1}^\infty$ 使得 $\mathbf{x}^*+ t_k \mathbf{d}_k \in \mathcal{X}$. 
  - 对 $f$ 在 $\mathbf{x}^*$ 进行 Tayler 展开, 有:
    $$
    \begin{aligned}
    f(\mathbf{x}^*+ t_k \mathbf{d}_k) &= f(\mathbf{x}^*) + t_k \mathbf{d}_k^\top \nabla f(\mathbf{x}^*) + o(t_k) \\
    &= f(\mathbf{x}^*) + \underbrace{t_k \mathbf{d}_k^\top \nabla f(\mathbf{x}^*)}_{<0} + \underbrace{t_k (\mathbf{d}_k - \mathbf{d}^*)^\top \nabla f(\mathbf{x}^*)}_{\to 0} + o(t_k)\\
    &< f(\mathbf{x}^*)
    \end{aligned}
    $$
  $\square$

- 该定理在几何上给出了可行域的判定定理, 然而其计算往往是不容易的. 如下我们需要给出更容易计算的可行方向集合之定义. 

#### Optimality Conditions by Linear Feasible Direction Cone