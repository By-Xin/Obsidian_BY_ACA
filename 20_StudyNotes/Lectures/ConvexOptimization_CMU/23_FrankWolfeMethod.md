# Frank-Wolfe Method

>[!quote]
>
> - Lecture Reference: 
>   - <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>

## Introduction

***Motivation***

回顾 Projected Gradient Descent 和 Proximal Gradient Descent 方法. 给定如下一般有约束问题:
$$
\min_{x \in \mathcal{D}} f(x)
$$
其中 $\mathcal{D}$ 是一个凸集, $f$ 是一个凸且光滑函数. 

Projected Gradient Descent 方法非常直观: 在每一步迭代中先进行一次 GD, 再投影回 feasible set $\mathcal{D}$:
$$
x_{k+1} = \Pi_{\mathcal{D}}(x_k - \eta_k \nabla f(x_k))$$
其中 $\Pi_{\mathcal{D}}$ 是投影算子, $\eta_k$ 是步长

更一般地, Projected Gradient Descent 方法可以推广到 Proximal Gradient Descent 方法, 其迭代公式为:
$$
x_{k+1} = \Pi_{\mathcal{D}} \left(
    \arg\min_y \langle \nabla f(x_k), y - x_k \rangle + \frac{1}{2\eta_k} \|y - x_k\|^2
\right)
$$
相当于把之前的梯度下降替换为了一个局部的二次近似最小化问题, 然后再投影回 feasible set $\mathcal{D}$.

然而, *投影操作并不总是容易计算的*, 例如在 $\mathcal{D}$ 是一个核范数约束集时, 投影操作就非常困难. Frank-Wolfe 方法, 也称 conditional gradient 方法, 是一种避免投影操作的算法. 

***Frank-Wolfe 方法***

对于上述问题, Frank-Wolfe 方法会首先在当前迭代点 $x^{k-1}$ 处对目标函数 $f$ 进行线性展开, 然后在 feasible set $\mathcal{D}$ 去求解这个线性问题的最优解:
$$
s^{k-1} = \arg\min_{s \in \mathcal{D}} \langle s, \nabla f(x^{k-1}) \rangle
$$
接着再以系数 $\gamma_k$ 在 $x^{k-1}$ 和 $s^{k-1}$ 之间进行凸组合, 得到新的迭代点:
$$
x^k = (1 - \gamma_k) x^{k-1} + \gamma_k s^{k-1}
$$
- 通常默认 $\gamma_k = 2 /(k+1)$. 并且还可以把更新改写为 $x^k = x^{k-1} + \gamma_k (s^{k-1} - x^{k-1})$, 也就是沿着 $s^{k-1} - x^{k-1}$ 的方向以 $\gamma_k$ 的步长进行更新, 且步长逐步减小.

<!-- ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260630170924.png) -->

一个典型的可视化如图. 下面的 $\mathcal{D}$ 是一个 polytope, 蓝色曲面为目标函数 $f$, 红色点为当前迭代点 $x^{k-1}$ 以及展开的 LP 问题的最优解 $s^{k-1}$, 棕色平面为 $f$ 在 $x^{k-1}$ 处的线性展开. 
- 这里特别标注了一个当前迭代点和线性展开中的最小值点之差, 记为 **Frank-Wolfe duality gap**:
    $$
    g(x^{k-1}) = \langle x^{k-1} - s^{k-1}, \nabla f(x^{k-1}) \rangle
    $$
    其衡量了在线性近似意义下还能带来多大的下降, 也可以作为一个收敛的判据. 具体而言, 我们有如下结论:
    $$
    f(x^{k}) - f^\star \leq g(x^{k-1})
    $$
    - *Proof*. 
        根据凸性, 有对于任意 $s$
        $$
        f(s) \geq f(x^{k}) + \langle s - x^{k}, \nabla f(x^{k}) \rangle
        $$
        对该不等式左右两侧在 $s \in \mathcal{D}$ 上取最小值, 得到
        $$
        f^\star \geq f(x^{k}) + \min_{s \in \mathcal{D}} \langle s - x^{k}, \nabla f(x^{k}) \rangle = f(x^{k}) + \langle \nabla f(x^{k}), s^{k} - x^{k} \rangle = f(x^{k}) - g(x^{k})
        $$
        $\square$

    - 之所以称之为 duality gap, 是因为若引入 indicator function $\delta_{\mathcal{D}}(x) = 0$ if $x \in \mathcal{D}$, $\infty$ otherwise, 则原问题可以写为
        $$
        \min_x f(x) + \delta_{\mathcal{D}}(x)
        $$
        而其对偶问题为
        $$
        \max_y -f^*(y) - \delta_{\mathcal{D}}^*(-y)
        $$
        其中 $\delta_{\mathcal{D}}^*(y) = \sup_{x \in \mathcal{D}} \langle x, y \rangle$ 是 $\delta_{\mathcal{D}}$ 的 Fenchel 对偶函数. 

- 此外注意, 如果约束本身 $\mathcal{D}$ 是一个 polytope, 那么 Frank-Wolfe 方法的每一步迭代都是在求解 LP 问题, 因此最优解定会落在 $\mathcal{D}$ 的一个顶点上. 因此在这些场景下, 只需要比较各个顶点的取值即可, 这也是 Frank-Wolfe 方法的一个优势.

***Frank-Wolfe in Norm Constraints***

当 $\mathcal{D}$ 是某种范数球约束时, Frank-Wolfe 方法可以有一个更为优雅的 closed-form solution.

考虑任意范数 $\|\cdot\|$ 约束:
$$
\mathcal{D} = \{x: \|x\| \leq t\}
$$
其中 $t>0$ 是半径常数. 

则此时的 Frank-Wolfe 方法的 LP 子问题为:
$$
\begin{aligned}
s^{k-1} &= \arg\min_{s \in \mathcal{D}} \langle s, \nabla f(x^{k-1}) \rangle \\
&= - t \Bigl( \arg\max_{ \|s\| \leq 1} \langle s, \nabla f(x^{k-1}) \rangle \Bigr) \\
&= - t \cdot \partial \|\nabla f(x^{k-1})\|_*
\end{aligned}
$$

- 其中 $\|\cdot\|_*$ 是范数 $\|\cdot\|$ 的对偶范数, 定义为:
    $$
    \|y\|_* = \sup_{\|x\| \leq 1} \langle x, y \rangle
    $$
    衡量给定输入向量 $y$ 在范数 $\|\cdot\|$ 下的最大投影长度.

***Constrained  and Lagrange forms***

在优化中, 面对约束需求, 通常有两种处理方式:
- Constrained form: $\min_x f(x) ~ s.t. ~ \|x\| \leq t$. 这里 $t$ 是约束的严格程度, 越小则约束越严格.
- Lagrange form: $\min_x f(x) + \lambda \|x\|$. 这里也可以看作是通过引入一个惩罚项来间接地处理约束问题, $\lambda$ 是惩罚系数, 其大小决定了约束的严格程度.

当 $t$ 和 $\lambda$ 分别从 $[0, \infty]$ 取值时, 两种形式给出的解集是等价的. 即对于任意 $t$, 都存在一个 $\lambda$ 使得两种形式的最优解相同, 反之亦然.

因此在解决统计学习等诸多问题时, 可以根据问题的便利程度选择使用 Constrained form 或 Lagrange form, 并可以使用类似 cross-validation 的方法来选择合适的 $t$ 或 $\lambda$.

Frank-Wolfe 方法可以直接处理 Constrained form 的问题, 而类似 Proximal or Projected Gradient Descent 方法则更适合处理 Lagrange form 的问题.

## Examples

### l1 Regularization

对于 l1 范数约束, 即
$$
\min_x f(x) \quad s.t. \quad \|x\|_1 \leq t
$$

可以立刻有
$$
s^{k-1} = - t \partial \|\nabla f(x^{k-1})\|_\infty = - t \cdot \text{sign}(\nabla f(x^{k-1})_i) e_{i^{k-1}}
$$
其中 $i^{k-1} = \arg\max_i |\nabla f(x^{k-1})_i|$, 也就是取梯度中绝对值最大的坐标, 然后在该坐标上取 $-t$ 或 $t$.

此时更新为
$$
x^k = (1 - \gamma_k) x^{k-1} + \gamma_k t \cdot \text{sign}(\nabla f(x^{k-1})_i) e_{i^{k-1}}
$$

其基本等价于一个贪心的 coordinate descent 方法, 每次迭代只更新一个坐标 (贪心地选择梯度最大的坐标), 并且每次迭代的步长逐渐减小.

### lp Regularization

对于 lp 范数约束, 其中 $1 \leq p \leq \infty$, 即
$$
\min_x f(x) \quad s.t. \quad \|x\|_p \leq t,
$$
由对偶关系有:
$$
s_i^{(k-1)} = -\alpha\cdot\text{sign}\big(\nabla_if(x^{(k-1)})\big)\cdot\big|\nabla_if(x^{(k-1)})\big|^{q/p}, \quad i=1,\dots,n
$$
其中 $\alpha = t / \|\nabla f(x^{(k-1)})\|_q^{q/p}$, 且 $\|s^{(k-1)}\|_p = t$, $q$ 是 $p$ 的对偶范数, 满足 $1/p + 1/q = 1$.

### Trace Norm Regularization

进一步扩展到矩阵情景, 考虑
$$
\min_X f(X) \quad s.t. \quad \|X\|_{\text{tr}} \leq t
$$
- 其中 $\|\cdot\|_{\text{tr}}$ 是矩阵的 trace norm, 也称 nuclear norm, 定义为矩阵奇异值的和.

则根据对偶形式, 有
$$
S^{(k-1)} = - t \cdot \partial \|\nabla f(X^{(k-1)})\|_{\text{op}}
$$
- 其中 $\|\cdot\|_{\text{op}}$ 是矩阵的 operator norm, 也就是矩阵的最大奇异值. 

下具体求解该函数的次梯度.
- Lemma1: 给定凸集 $\mathcal{B}$, 设 $\phi(z) = \max_{u \in \mathcal{B}} \langle u, z \rangle$. 若 $u^* \in \arg\max_{u \in \mathcal{B}} \langle u, z_0 \rangle$, 则有 $u^* \in \partial \phi(z_0)$.
  - *Proof*.  次梯度需证 $\phi(z) \geq \phi(z_0) + \langle u^*, z - z_0 \rangle$ 对任意 $z$ 成立. 由 $u^*$ 的最优性, $\phi(z_0) = \langle u^*, z_0 \rangle$. 故 $\phi(z_0) + \langle u^*, z - z_0 \rangle = \langle u^*, z \rangle \leq \max_{u \in \mathcal{B}} \langle u, z \rangle = \phi(z)$, 证毕.
- Lemma 2: Operator norm 是 trace norm 的对偶范数, 即 $\|Z\|_{\text{op}} = \max_{\|U\|_{\text{tr}} \leq 1} \langle U, Z \rangle$. 且该最大值在 $U^* = u_1 v_1^\top$ 处取得, 其中 $u_1, v_1$ 分别是 $Z$ 的最大奇异值对应的左、右奇异向量.
-  综合上述两条 Lemma, 可得
    $$
    S^{(k-1)} = - t \cdot u_1 v_1^\top
    $$

## Convergence Analysis

对于 Frank-Wolfe 方法, 其步长设置为 $\gamma_k = 2/(k+1)$ 时, 有如下收敛性结论:
$$
f(x^k) - f^\star \leq \frac{2 C_f}{k+2}, \quad k = 1, 2, \dots
$$
其中 $C_f$ 是一个与 $f$ 的 curvature 相关的常数, 定义为:
$$
C_f = \sup_{x, s \in \mathcal{D}, \gamma \in [0, 1], y = x + \gamma(s - x)} \frac{2}{\gamma^2} \big(f(y) - f(x) - \langle y - x, \nabla f(x) \rangle\big).
$$


下给出理解和证明. 

- 首先观察 $C_f$, 其衡量了 $f$ 在 $\mathcal{D}$ 上的非线性程度. 注意到 $f(x) + \langle y - x, \nabla f(x) \rangle$ 是 $f$ 在 $x$ 处的线性展开, 故 $f(y) - f(x) - \langle y - x, \nabla f(x) \rangle$ 衡量了 $f$ 在 $y$ 处相对于 $x$ 的线性展开的偏离程度, 称为 Bregman divergence. 而 $C_f$ 则是对该偏离程度在可行域 $\mathcal{D}$ 上的最大值, 并且对 $\gamma$ 进行了归一化. 因此 $C_f$ 越小, 表明 $f$ 越接近线性函数, Frank-Wolfe 方法的收敛速度也就越快.
- 因此对于单步更新 $x^{k+1} = x^k + \gamma_k (s^k - x^k), \quad \gamma_k = \frac{2}{k+2}$, 令 $C_f$ 定义中 $y \leftarrow x^{k+1}, x \leftarrow x^k, s \leftarrow s^k$, 则有
    $$
    C_f \geq \frac{2}{\gamma_k^2} \big(f(x^{k+1}) - f(x^k) - \langle x^{k+1} - x^k, \nabla f(x^k) \rangle\big)
    $$
    整理有
    $$
    \begin{aligned}
    f(x^{k+1}) &\leq f(x^k) + \langle x^{k+1} - x^k, \nabla f(x^k) \rangle + \frac{\gamma_k^2}{2} C_f \\
    &= f(x^k) + \gamma_k \langle s^k - x^k, \nabla f(x^k) \rangle + \frac{\gamma_k^2}{2} C_f 
    \end{aligned}
    $$
- 回顾 Frank-Wolfe 的 Duality gap $g(x^k) = \langle x^k - s^k, \nabla f(x^k) \rangle$, 则有
    $$
    f(x^{k+1}) \leq f(x^k) - \gamma_k g(x^k) + \frac{\gamma_k^2}{2} C_f
    $$
    故
    $$
    f(x^{k+1}) - f^\star \leq f(x^k) - f^\star - \gamma_k g(x^k) + \frac{\gamma_k^2}{2} C_f
    $$
    又根据 dual 的性质 $g(x^k) \geq f(x^k) - f^\star$, 故有
    $$
    f(x^{k+1}) - f^\star \leq (1 - \gamma_k) (f(x^k) - f^\star) + \frac{\gamma_k^2}{2} C_f
    $$
- 现已有递推关系 $f(x^{k+1}) - f^\star \leq (1 - \gamma_k) (f(x^k) - f^\star) + \frac{\gamma_k^2}{2} C_f$, 且 $\gamma_k = \frac{2}{k+2}$. 根据数学归纳法,
    - 可另外验证当 $k=1$ 时, 有 $f(x^1) - f^\star \leq 2C_f/3$, 满足结论.
    - 假设当 $k$ 时, 有 $f(x^k) - f^\star \leq 2C_f/(k+2)$ 成立, 则当 $k+1$ 时, 有
        $$
        f(x^{k+1}) - f^\star \leq \frac{2C_f}{k+3}
        $$
        该命题可通过代数整理放缩得到.

    $\square$


## Properties and Variants

### Affine Invariance

Frank-Wolfe 方法具有仿射不变性.

给定原始空间的约束问题:
$$
\min_{x \in \mathcal{D}} f(x)
$$

以及 affine transformation 关系 $x = A\tilde{x}$ 其中 $A$ 是非奇异的, $\tilde{x}$ 变换后的空间, 对应的约束问题为:
$$
\min_{\tilde{x} \in A^{-1}(\mathcal{D})} f(A\tilde{x}) =: \tilde{f}(\tilde{x})
$$

该性质表明, 在原始空间中进行 Frank-Wolfe 方法的迭代
$$
s^{k-1} = \arg\min_{s \in \mathcal{D}} \langle s, \nabla f(x^{k-1}) \rangle, \quad x^k = (1 - \gamma_k) x^{k-1} + \gamma_k s^{k-1}
$$
与在变换空间中进行 Frank-Wolfe 方法的迭代
$$
\tilde{s}^{k-1} = \arg\min_{\tilde{s} \in A^{-1}(\mathcal{D})} \langle \tilde{s}, \nabla \tilde{f}(\tilde{x}^{k-1}) \rangle, \quad \tilde{x}^k = (1 - \gamma_k) \tilde{x}^{k-1} + \gamma_k \tilde{s}^{k-1}
$$
是完全等价的. 换言之, 对在原空间中进行 Frank-Wolfe 方法的迭代的结果, 通过 affine transformation 可以得到在变换空间中进行 Frank-Wolfe 方法的迭代的结果, 反之亦然.


该性质的作用是: Frank-Wolfe 方法对于问题的 conditioning 不敏感. 只要该问题可以通过 affine transformation 转换为一个 well-conditioned 的问题, 则 Frank-Wolfe 方法在原问题 (即使是 ill-conditioned) 上的收敛速度也不会受到影响. 故对于一些几何形状复杂的约束问题, Frank-Wolfe 方法仍然可以有较好的收敛速度.

### Inexact Updates

Inexact Updates 是指在 Frank-Wolfe 方法中, 允许在每一步迭代中不精确地求解 LP 子问题. 只要近似解的误差在一定范围内, 仍然可以保证 Frank-Wolfe 方法的收敛性. 具体地, 在标准 Frank-Wolfe 方法中, 我们需要最小化:
$$
s^{k-1} = \arg\min_{s \in \mathcal{D}} \langle s, \nabla f(x^{k-1}) \rangle
$$
然而若在实践中, 这样的求解复杂度过高, 则可以允许我们在每一步迭代中只求得一个近似解 $\tilde{s}^{k-1}$, 使得给定一个误差 $\delta$ 比例, 有:
$$
\langle \tilde{s}^{k-1}, \nabla f(x^{k-1}) \rangle \leq \min_{s \in \mathcal{D}} \langle s, \nabla f(x^{k-1}) \rangle + \frac{\gamma_k C_f}{2} \delta, \quad \gamma_k = \frac{2}{k+1}
$$
则仍然能够保证有
$$
f(x^k) - f^\star \leq \frac{2 C_f (1 + \delta)}{k+1}, \quad k = 1, 2, \dots
$$

注意, 这里的误差 $\delta$ 是相对于 curvature $C_f$ 的, 而不是绝对误差. 绝对的误差要随着迭代次数的增加而逐渐减小.


### Some Variants

#### Line Search

Line Search 是指在 Frank-Wolfe 方法的每一步迭代中, 不再使用固定的步长 $\gamma_k = 2/(k+1)$, 而是再次求解一个一维的最优化问题, 来选择最优的步长 $\gamma_k$:
$$
\gamma_k = \arg\min_{\gamma \in [0, 1]} f(x^{k-1} + \gamma (s^{k-1} - x^{k-1}))
$$

如果本身这个一维问题结构良好有闭式解, 则可以直接优化; 否则也可以考虑使用 backtracking, 从一个较大的步长开始, 逐步减小, 直到满足某种条件如 Armijo condition.

可以证明, 其复杂度仍然是 $\mathcal{O}(1/\varepsilon)$.

#### Fully Corrective Frank-Wolfe

标准的 Frank-Wolfe 方法每一步迭代只使用当前的 $s^{k-1}$ 来和上一步的 $x^{k-1}$ 进行凸组合, 而 Fully Corrective Frank-Wolfe 方法则会在每一步迭代中, 将之前所有的 $s^i$ 都纳入考虑, 并重新求解一个凸组合的最优解. 具体地, 在第 $k$ 步迭代中, 首先还是会和标准的 Frank-Wolfe 方法一样, 求解 $s^{k-1}$:
$$
s^{k-1} = \arg\min_{s \in \mathcal{D}} \langle s, \nabla f(x^{k-1}) \rangle
$$
接着, 将之前所有的 $s^i$ 都纳入考虑, 并求解一个凸组合的最优解:
$$
x^k = \arg\min_{x \in \text{conv}\{s^0, s^1, \dots, s^{k-1}\}} f(x)
$$
其中 $\text{conv}\{s^0, s^1, \dots, s^{k-1}\}$ 表示这些点的凸包. 

由于本身 $f$ 也是线性的, 因此这个问题相当于求解单纯形上的一个凸优化问题:
$$
\min_{\lambda \in \Delta^{k}} f\Big(\sum_{i=0}^{k-1} \lambda_i s^i\Big)
$$
这时, 我们将问题进行了极大的简化, 从原先的 $\mathcal{D}$ 上的凸优化问题, 转化为了一个 $K$ 维的 simplex 约束问题. 


可以证明, 其复杂度仍然是 $\mathcal{O}(1/\varepsilon)$.

#### Away Steps

对于标准的 Frank-Wolfe 方法, 每一步迭代都是沿着历史的顶点进行加权. 历史的信息只能被逐渐稀释, 而不能被删除. 而 Away Steps 方法则允许在每一步迭代中, 选择一个历史的顶点, 并沿着该顶点的反方向进行更新, 也就是允许删除历史信息.

具体地, 在第 $k$ 步迭代中, 除了还是和正常的 Frank-Wolfe 方法一样, 求解 $s^{k-1}$ 作为前进方向:
$$
s^{k-1} = \arg\min_{s \in \mathcal{D}} \langle s, \nabla f(x^{k-1}) \rangle
$$
此外还会求解一个历史顶点 $v^{k-1}$ 作为后退方向 (away step):
$$
v^{k-1} = \arg\max_{v \in S} \langle v, \nabla f(x^{k-1}) \rangle
$$
- 其中 $S$ 是当前活跃的顶点集合, 也就是之前迭代中被选中的顶点的集合 (而不是 $\mathcal{D}$ 的所有顶点).

然后在更新中, 每次从如下两个方向中选择一个最优的方向进行更新:
- Forward step: $x^k = x^{k-1} + \gamma_k (s^{k-1} - x^{k-1})$
- Away step: $x^k = x^{k-1} + \gamma_k (x^{k-1} - v^{k-1})$

可以证明, 若约束集 $\mathcal{D}$ 是一个 polytope, 且 $f$ 是一个 strongly convex 函数, 则 Away Steps 方法可以达到线性收敛的速度:
$$
f(x^k) - f^\star \leq \mathcal{O}(\rho^k), \quad \rho < 1
$$

## Path Following

在实践当中, 在面对问题
$$
\min_x f(x) \quad s.t. \quad \|x\| \leq t
$$
往往并不知道最优的参数 $t$ 的选择是什么. 因此有时需要在不同的 $t$ 下, 进行一系列的求解, 也就是所谓的 path following 方法. 如果在求解 Path Following 时, 把每一个 $t$ 的问题都当作一个独立的问题, 则是非常低效的.

一个可能的处理思路如下. 对于任意给定 $t$, 设 $\hat{x}(t)$ 是一个误差不超过 $\varepsilon$ 的近似解. 对 $t \in [0, \infty)$ 进行离散化, 得到一系列的分段 $[t_0, t_1), [t_1, t_2), \dots$, 在每一个分段中都使用同一个 $\hat{x}(t_i)$ 作为近似解. 

因此核心问题是: 如何选择这些分段的端点 $t_i$, 区间内的误差能够充分利用但不超过 $\varepsilon$.  故这里再次引出 dual gap. 回顾其定义为:
$$
g_t(x) = \max_{ \|s\| \leq t} \langle x - s, \nabla f(x) \rangle = \langle x, \nabla f(x) \rangle + \max_{\|s\| \leq t} \langle -s, \nabla f(x) \rangle = \langle x, \nabla f(x) \rangle + t \|\nabla f(x)\|_*
$$

这说明, 给定 $x$, 其 dual gap 是一个关于 $t$ 的线性函数, 且斜率为 $\|\nabla f(x)\|_*$.

因此, 若在 $t_{k-1}$ 处求得的近似解 $\hat{x}(t_{k-1})$ 满足精度要求 $g_{t_{k-1}}(\hat{x}(t_{k-1})) \leq \varepsilon/m$, 且在 $t_k +\Delta t$ 处恰超过 $\varepsilon$, 即:
$$
g_{t_k + \Delta t}(\hat{x}(t_{k-1})) = g_{t_k}(\hat{x}(t_{k-1})) + \Delta t \|\nabla f(\hat{x}(t_{k-1}))\|_* = \varepsilon
$$
从中可以反过来解出, 在
$$
\Delta t = \frac{(1- 1/m)\varepsilon}{\|\nabla f(\hat{x}(t_{k-1}))\|_*}
$$
范围内, 近似解 $\hat{x}(t_{k-1})$ 都可以满足精度要求. 因此可以选择 $t_k = t_{k-1} + \Delta t$ 作为下一个分段的端点, 并在该分段内使用 $\hat{x}(t_{k-1})$ 作为近似解.