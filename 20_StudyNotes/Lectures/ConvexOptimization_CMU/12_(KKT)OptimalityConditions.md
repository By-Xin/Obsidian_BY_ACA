# Optimality Conditions for Constrained Problems

> [!info] References
> - Lecture: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/
> - Reading: 最优化: 建模、算法与理论, 刘浩洋等, 5.5 小节.

## 1. General Constrained Problems (No Convexity Assumption)

### 1.1. First-Order Optimality Conditions

回顾, 考虑如下一般的含约束的优化问题 (不要求是凸的):
$$\begin{aligned}
& \min_{\mathbf{x}\in \mathbb{R}^n} && f(x) \\
& \text{subject to} && c_i(\mathbf{x}) \leq 0, i\in \mathcal{I} \\
& && c_j(\mathbf{x}) = 0, j\in \mathcal{E}
\end{aligned}$$

其 Lagrangian 函数为:
$$
L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i\in \mathcal{I}} \lambda_i c_i(\mathbf{x}) + \sum_{j\in \mathcal{E}} \nu_j c_j(\mathbf{x})
$$

其 Lagrange Dual Function 为:
$$
g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}\in \mathbb{R}^n} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})
$$

#### 1.1.1. Optimality Conditions by Tangent Cone

为定义可行域内的一系列点列的极限状态, 引入切向量和切锥的概念. 

***Definition* (Tangent Vector)**: 对于可行域 $\mathcal{X}$ 内的点列 $\{\mathbf{x}_k\}_{k=1}^\infty \subseteq \mathcal{X}\subset \mathbb{R}^n$, 其极限状态为 $\lim_{k\to\infty} \mathbf{x}_k = \mathbf{x}^*\subset \mathcal{X}$ (即该点列逼近 $\mathbf{x}^*$). 若存在向量 $\mathbf{d}\in \mathbb{R}^n$, 以及一个正数标量序列 $\{t_k\}_{k=1}^\infty$ 且 $t_k \to 0$ 使得:
$$
\lim_{k\to\infty} \frac{\mathbf{x}_k - \mathbf{x}^*}{t_k} = \mathbf{d}
$$
则称 $\mathbf{d}$ 为 $\mathbf{x}^*$ 处的切向量.

***Definition* (Bouligian Tangent Cone)**: 对于上述点 $\mathbf{x}^*$ 处的全部切向量之集合, 称为该点处的切锥, 记作 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*)$, 其数学表达为:
$$
\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) = \left\{
\mathbf{d} \in \mathbb{R}^n \;\middle|\; 
\exists t_k \downarrow 0, \exists \mathbf{x}_k \in \mathcal{X}, \frac{\mathbf{x}_k - \mathbf{x}^*}{t_k} \to \mathbf{d}
\right\}
$$

- **切锥表示从可行域内某点 $\mathbf{x}^*$ 出发, 所有能够满足约束条件的行动方向之集合.**

> [!example] **切锥的例子**
>
> ![Ref: 最优化: 建模、算法与理论, 刘浩洋等, 5.5 小节](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202602202338194.png)
>
> - 如上图所示. 图中两条曲线分别代表两条约束方程 $c_1(\mathbf{x})$ 和 $c_2(\mathbf{x})$ 的图像. 左侧为不等式约束, 图中深色阴影部分表示两约束方程构成的可行域 $\mathcal{X}$. 右侧为等式约束, 故可行域只有轮廓本身. 
>   - 对于不等式约束, 其切锥为整个深浅阴影区域, 为一个凸锥. 
>   - 对于等式约束, 其切锥只能取在左图轮廓线上, 即图中两条射线. 

***Theorem* (Optimality Conditions by Tangent Cone)**: 设 $x^*$ 是可行域 $\mathcal{X}$ 内的一个局部极小点. 若 $f$ 和 $c_i, c_j (\forall i,j)$ 在 $\mathbf{x}^*$ 处可微, 则有:
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


#### 1.1.2. Optimality Conditions by Linearized Feasible Direction Cone

上述在几何上给出了可行域的判定定理, 然而其计算往往是不容易的. 如下我们需要给出更容易计算的可行方向集合之定义. 

***Definition* (Active Set)**: 对于可行域 $\mathcal{X}$ 内的点 $\mathbf{x}$, 其 active set 定义为:
$$
\mathcal{A}(\mathbf{x}) = \mathcal{E} \cup \{i \in \mathcal{I} \mid c_i(\mathbf{x}) = 0\}
$$

即所有等式约束和所有不等式约束中, 在 $\mathbf{x}$ 处取等号的约束的集合. 

- Active set 是对于当前点 $\mathbf{x}$ 处, 所有真正起到约束作用的约束的集合. 对于所有 $c_i(\mathbf{x}) < 0$ 的约束, 其并没有起到约束作用, 在这些该点的微小领域内, 这些约束仍然可以被满足. 

***Definition* (Linearized Feasible Direction Cone)**: 对于可行域 $\mathcal{X}$ 内的点 $\mathbf{x}$, 其 linearized feasible direction cone 定义为:
$$
\mathcal{F} = \left\{
\mathbf{d} \in \mathbb{R}^n \;\middle|\; \begin{aligned}
\mathbf{d}^\top \nabla c_j(\mathbf{x}) = 0, & \quad \forall j \in \mathcal{E} \\
\mathbf{d}^\top \nabla c_i(\mathbf{x}) \geq 0, & \quad \forall i \in \mathcal{A}(\mathbf{x}) \cap \mathcal{I}
\end{aligned}
\right\}
$$

- 该定义的 intuition 如下: 
  - 我们尝试寻找从当前点 $\mathbf{x}$ 出发, 所有能够满足约束条件的行动方向之集合. 
  - 希望存在微小量 $t > 0$ 使得 $\mathbf{x} + t \mathbf{d} \in \mathcal{X}$. 故需要对每个约束 $c_j, c_i$ 求解一阶 Taylor 近似 (即线性化):
    $$
    c_i(\mathbf{x} + t \mathbf{d}) \approx c_i(\mathbf{x}) + t \nabla c_i(\mathbf{x})^\top \mathbf{d} \quad \forall i
    $$

  - 对于等式约束 $i\in \mathcal{E}$, 要求有 $c_i(\mathbf{x} + t \mathbf{d}) = 0$, 又由于 $c_i(\mathbf{x}) = 0$, 代入上述展开故有:
    $$
    \nabla c_i(\mathbf{x})^\top \mathbf{d} = 0,\quad \forall i \in \mathcal{E}
    $$
    - 站在当前点 $\mathbf{x}$ 处, 只能沿着等式约束的切向量方向移动, 即"沿着等式约束的轮廓线"移动.
  - 对于不等式约束 $i\in \mathcal{A}(\mathbf{x}) \cap \mathcal{I}$, 要求有 $c_i(\mathbf{x} + t \mathbf{d}) \leq 0$, 又由于 $c_i(\mathbf{x}) \leq 0$, 故有:
    $$
    \nabla c_i(\mathbf{x})^\top \mathbf{d} \leq 0,\quad \forall i \in \mathcal{A}(\mathbf{x}) \cap \mathcal{I}
    $$
    - 站在当前点 $\mathbf{x}$ 处, 可行的移动方向必与梯度方向夹角为钝角, 即必往约束的内部移动.
      - *为什么梯度方向的钝角方向为约束内部?* 事实上, 这是因为该约束为 $\leq$ 的不等式约束, 而梯度方向本质上为最陡上升方向, 因此沿梯度方向的分量为上升方向, 而对于 Active set 中的点而言, 任何上升分量都将导致 $\leq$ 的不等式约束不成立. 
    

***Corollary* (Contingent Tangent Cone and Linearized Feasible Direction Cone)**: 若每个 $c_i$ 在 $\mathbf{x}^*$ 附近一阶连续可微, 即 $c_i \in C^1(\mathcal{X})$, 则对于任意可行点 $\mathbf{x}$, 满足
$$
\mathcal{T}_{\mathcal{X}}(\mathbf{x}) \subseteq \mathcal{F}(\mathbf{x})
$$
- 观察下述例子:
  - 考虑问题
    $$
    \begin{aligned}
    \min_{x\in \mathbb{R}} & f(x) = x \\
    \text{s.t.} & (-x+3)^3 \leq 0
    \end{aligned}
    $$
    其可行域为 $\mathcal{X}=[3,\infty)$. 
    - 根据切锥的定义, 可知在 $x = 3$ 处, 其切锥为 $\mathcal{T}_{\mathcal{X}}(3) = \{d \mid d \geq 0\}$. 
    - 又根据线性化可行方向锥的定义, 其梯度(导数)方向为 $c'(x) = 3(-x+3)^2\mid_{x=3} = 0$, 且该点 $x=3$ 处为等式约束, 故其线性化可行方向需满足 $0\cdot d \leq 0$, 即 $d \in \mathbb{R}$. 
    - 故有 $\mathcal{T}_{\mathcal{X}}(3) \subset \mathcal{F}(3)$.
  - 另一方面, 若将约束条件改为
    $$
    \begin{aligned}
    \min_{x\in \mathbb{R}} & f(x) = x \\
    \text{s.t.} & -x+3 \leq 0
    \end{aligned}
    $$
    其可行域仍为 $\mathcal{X}=[3,\infty)$. 
    - 根据定义, 由于可行域没有改变, 故在 $x = 3$ 处, 其切锥仍为 $\mathcal{T}_{\mathcal{X}}(3) = \{d \mid d \geq 0\}$. 
    - 然而, 其导数发生了变化, 此时 $c'(x) = -1\mid_{x=3} = -1$, 故其线性化可行方向需满足 $-1\cdot d \leq 0$, 即 $d \geq 0$. 
    - 此时有 $\mathcal{T}_{\mathcal{X}}(3) = \mathcal{F}(3) = \{d \mid d \geq 0\}$. 
  - 该例子说明, 即使对于同一个可行域, 只是其约束条件的代数表示发生了变化, 其线性化可行方向锥可能发生变化. 本质上, 这是因为线性化可行方向锥的定义是基于一阶 Taylor 近似, 而在高维表述中该一阶信息可能丢失, 从而影响对于可行方向的判定. 


综上, 我们有观察: 
- 线性化可行方向锥易于计算和使用, 但其本身会受到问题的代数表示的影响
- 切锥相对稳健, 然而其计算往往需要计算极限等复杂操作. 

#### 1.1.3. Constraint Qualification

根据上述观察, 引入约束的品性 (Constraint Qualification) 这一概念, 满足该品性的约束往往保证了在最优点 $\mathbf{x}^*$ 处可以有诸如 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}^*) = \mathcal{F}(\mathbf{x}^*)$ 的优秀性质. 

***Definition* (Linear Independence Constraint Qualification)**: 对于可行域 $\mathcal{X}$ 内的点 $\mathbf{x}$, 任意 Active set $\mathcal{A}(\mathbf{x})$ 中的约束 $\nabla c_i(\mathbf{x}), \quad i\in \mathcal{A}(\mathbf{x})$ 线性无关, 则称该约束在 $\mathbf{x}$ 处满足 LICQ.

***Lemma* (LICQ Property)**: 若任意可行点 $\mathbf{x}\in \mathcal{X}$ 满足 LICQ, 则有:
$$
\mathcal{T}_{\mathcal{X}}(\mathbf{x}) = \mathcal{F}(\mathbf{x})
$$

- *Proof.*
  - 不失一般性, 假设 active set $\mathcal{A}(\mathbf{x}) = \mathcal{E} \cup \mathcal{I}$ 且 $|\mathcal{A}(\mathbf{x})| = m$. 记矩阵:
    $$
    \mathbf{A}(\mathbf{x}) = \begin{bmatrix}
    \nabla c_1(\mathbf{x}) & \nabla c_2(\mathbf{x}) & \dots & \nabla c_m(\mathbf{x})
    \end{bmatrix}^\top \in \mathbb{R}^{m\times n}
    $$
  - 由 LICQ 之假设, 各约束之间是线性独立的, 故有 $\text{rank}(\mathbf{A}(\mathbf{x})) = m$. 对应的 $A$ 矩阵的零空间为 $\mathbf{Z}:= \text{Null}(\mathbf{A}(\mathbf{x})) \in \mathbb{R}^{n\times (n-m)}$, 其满足 $\mathbf{Z}^\top \mathbf{A}(\mathbf{x}) = \mathbf{0}$ (即为等式约束的一阶可行方向对应的空间). 此外, 根据代数结论, 有 $\text{rank}(\mathbf{A}(\mathbf{x})) + \text{Null}(\mathbf{A}(\mathbf{x})) = n$ 故有 $\text{rank}(\mathbf{Z}) = n-m$. 
    - $\mathbf{Z}$ 张成的空间即为贴着活跃约束的边界, 沿着该方向移动不会违反任何约束. 
  - 给定任意可行点 $\mathbf{x}_0$ 与 $\mathbf{d}\in \mathcal{F}(\mathbf{x}_0)$, 欲证 $\mathbf{d}\in \mathcal{T}_{\mathcal{X}}(\mathbf{x}_0)$, 即 $\mathcal{F}(\mathbf{x}_0)\subseteq\mathcal{T}_{\mathcal{X}}(\mathbf{x}_0)$. 若命题得证, 再加之 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}_0) \subseteq \mathcal{F}(\mathbf{x}_0)$, 则将有 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}_0) = \mathcal{F}(\mathbf{x}_0)$. 
    - 根据切锥定义, 我们需要构造一串可行点列 $\{\mathbf{z}_k\}_{k=1}^\infty \subseteq \mathcal{X}\subset \mathbb{R}^n$ 和 $t_k \downarrow 0$ 使得 $\dfrac{\mathbf{z}_k - \mathbf{x}_0}{t_k} \to \mathbf{d}$, 或者说 $\mathbf{z}_k = \mathbf{x}_0 + t_k \mathbf{d} + \delta_{t_k}$, 且 $\delta_{t_k} = o(t_k)$. 本质在于说明, 每给定一个步长 $t_k$ 以及想要移动的方向 $\mathbf{d}$, 我们总能找到一个可行点 $\mathbf{z}_k$ (事实上依赖于步长故 $\mathbf{z}(t_k)$) 保证 $\mathbf{z}(t_k) \to \mathbf{x}_0$. 并且这种趋近关系事实上应当是对于所有 $t_k \downarrow 0$ 都成立的. 下面的证明将说明, 我们可以构造出一个线性系统 $R(\mathbf{z},t_k; \mathbf{d}) = \mathbf{0}$ 使得可以从中解出 $\mathbf{z}(t_k)$.
    - 记 $\mathbf{c}(\mathbf{z}) = \begin{bmatrix} c_1(\mathbf{z}) & c_2(\mathbf{z}) & \dots & c_m(\mathbf{z}) \end{bmatrix}^\top \in \mathbb{R}^m$, 是一个所有活跃函数组成的向量函数. 则上述映射 $R: \mathbb{R}^n \times \mathbb{R} \to \mathbb{R}^{m+(n-m)}$ 可以表示为:
      $$
      \begin{aligned}
      R(\mathbf{z},t_k; \mathbf{d}) &=
        \begin{bmatrix}
        \mathbf{c}(\mathbf{z}) - t_k A(\mathbf{x}_0) \mathbf{d} \\
        \mathbf{Z}^\top (\mathbf{z} - \mathbf{x}_0 - t_k \mathbf{d})
        \end{bmatrix} = \mathbf{0}
      \end{aligned}
      $$
      - 第一部分 $\mathbf{c}(\mathbf{z}) - t_k A(\mathbf{x}_0) \mathbf{d} = \mathbf{0}$ 等价于对于任意 $i\in \mathcal{A}(\mathbf{x}_0)$ 有 $c_i(\mathbf{z}) = t_k \mathbf{d}^\top \nabla c_i(\mathbf{x}_0)$. 其 LHS 代表我们想要找出的点的约束取值; 而 RHS 代表在 $\mathcal{F}(\mathbf{x}_0)$ 中的约束状态. 若 RHS $i\in\mathcal{E}$, 则 $\mathbf{d}^\top \nabla c_i(\mathbf{x}_0) = 0$, 对应 LHS $c_i(\mathbf{z}) = 0$; 若 $i\in\mathcal{I}$, 则 $\mathbf{d}^\top \nabla c_i(\mathbf{x}_0) \leq 0$, 对应 LHS $c_i(\mathbf{z}) \leq 0$. 因此通过第一部分, 我们可以很好地控制 $\mathbf{z}$ 仍为可行点.
      - 第二部分 $\mathbf{Z}^\top (\mathbf{z} - \mathbf{x}_0 - t_k \mathbf{d}) = \mathbf{0}$ 相当于说明 $\mathbf{z} - (\mathbf{x}_0 + t_k \mathbf{d}) = \mathbf{z} - \delta_{t_k}$ 与 $\mathbf{Z}$ 的列向量正交, 即只能沿着约束梯度方向进行移动修正. 
    - 在定义 $R(\mathbf{z},t_k; \mathbf{d}) = \mathbf{0}$ 后, 根据隐函数定理, 能够保证对于任意给定充分小的 $t_k > 0$, 存在唯一的 $\mathbf{z}(t_k)$, 使得 $R(\mathbf{z}(t_k),t_k) = \mathbf{0}$.
      - 具体而言, 对于 $R: \mathbb{R}^n \times \mathbb{R} \to \mathbb{R}^n$ 连续可微. 若 (1) $R(\mathbf{x}_0,0) = \mathbf{0}$; (2) $\dfrac{\partial R}{\partial \mathbf{z}}(\mathbf{x}_0,0)$ 可逆, 则定理可以保证: 存在 $\epsilon>0$, 以及唯一的函数 $\mathbf{z}(t)$, 使得 $R(\mathbf{z}(t),t)=\mathbf{0},~ \mathbf{z}(0)=\mathbf{z}$.
    - 具体求解 $\mathbf{z}(t_k)$ 时, 对 $\mathbf{c}(\mathbf{z})$ 在 $\mathbf{x}_0$ 处进行一阶 Taylor 展开, 有:
      $$
      \mathbf{c}(\mathbf{z}_k) = \mathbf{c}(\mathbf{x}_0) + \nabla \mathbf{c}(\mathbf{x}_0)(\mathbf{z}_k - \mathbf{x}_0) + \mathbf{e}_k
      $$
    - 将上表达式进行代入整理, 可得 (具体过程略)
      $$
      \begin{aligned}
      \frac{\mathbf{z}_k - \mathbf{x}_0}{t_k} &= \mathbf{d} + \frac{1}{t_k} \begin{bmatrix} A(\mathbf{x}_0) \\ \mathbf{Z}^\top \end{bmatrix} \begin{bmatrix} \mathbf{e}_k \\\mathbf{0}\end{bmatrix}
      \end{aligned}
      $$
    - 根据 $\|\mathbf{e}_k\|=o(t_k)$, 命题得证.


***Definition* (Mangasarian-Fromovitz Constraint Qualification, MFCQ)** 给定可行点 $\mathbf{x}_0$ 及其 active set $\mathcal{A}(\mathbf{x}_0)$, 若存在一个向量 $\mathbf{w}\in \mathbb{R}^n$ 满足:
$$
\begin{aligned}
\nabla c_i(\mathbf{x}_0)^\top \mathbf{w} &\lt 0, \quad &&\forall i \in \mathcal{A}(\mathbf{x}_0) \cap \mathcal{I} \\
\nabla c_i(\mathbf{x}_0)^\top \mathbf{w} &= 0, \quad &&\forall i \in  \mathcal{E}
\end{aligned}
$$
且等式约束的梯度集合 $\{\nabla c_i(\mathbf{x}_0)\}_{i\in \mathcal{E}}$ 线性无关, 则称该约束在 $\mathbf{x}_0$ 处满足 MFCQ.
- 可以证明，由 LICQ 可以推出 MFCQ, 但反之不然. 
- 若 MFCQ 成立, 同样可知 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}_0) = \mathcal{F}(\mathbf{x}_0)$. 


***Definition* (Linear Restriction Qualification, LRQ)** 若优化问题中的全部约束函数 $c_i(\mathbf{x}), i\in \mathcal{E} \cup \mathcal{I}$ 都是线性的, 则称线性约束品性满足. 
- 对于线性约束品性, 有 $\mathcal{T}_{\mathcal{X}}(\mathbf{x}) = \mathcal{F}(\mathbf{x})$. 
- LP, QP 等优化问题自然满足线性约束品性. 
- LRQ 和 LICQ 直接一般没有必然关联.


#### 1.1.4. Karush-Kuhn-Tucker (KKT) Conditions

回忆约束问题的几何最优性条件. 对于局部最小点 $\mathbf{x}^*$, 若 $f, c_i ~(\forall i\in \mathcal{E} \cup \mathcal{I})$ 在 $\mathbf{x}^*$ 处可微, 则有 $\mathbf{d}^\top \nabla f(\mathbf{x}^*) \ge 0,~\forall \mathbf{d}\in \mathcal{T}_{\mathcal{X}}(\mathbf{x}^*)$. 