# Hölder Smoothness

## 1. Introduction 

### 1.1 [[Hölder Smoothness]] Definition

***Definition* ([[Hölder Smoothness]])** 给定一阶连续可微函数 $f: \Omega \to \mathbb{R}$, 其中 $\Omega \subseteq \mathbb{R}^n$ 是开凸集. 称 $f$ 是 ($L-\rho$) [[Hölder Smoothness]] (或等价地, 其梯度是 ($L-\rho$) [[Holder Continuous]]), 若存在常数 $L > 0$ 和 $\rho \in (0, 1]$, 使得对于任意 $\mathbf{x}, \mathbf{y} \in \Omega$, 有:
$$
\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\|_2 \leq L \|\mathbf{x}-\mathbf{y}\|_2^\rho
$$

特别地, 有如下几种情况:

1. $\rho = 1$ 时, 称为 [[Lipschitz Smoothness]], 即:
   $$
   \|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\|_2 \leq L \|\mathbf{x}-\mathbf{y}\|_2
   $$

2. $\rho \in (0, 1)$ 时, 称为 [[Holder Smoothness]], 其梯度连续但弱于线性情况:
   $$
   \|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\|_2 \leq L \|\mathbf{x}-\mathbf{y}\|_2^\rho
   $$

> [!note] Remark: 
> 
>  当 $\rho=0$ 时, 形式上我们将得到, 对于任意 $\mathbf{x} \neq \mathbf{y}$, 有
> $$
> \|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\|_2 \leq L
> $$
>
> 需要指出, 此时其仍然是可微的, 因此仍然属于光滑函数的范畴. 然而其处在 Holder Smoothness 的边界上. 
>
> ***Proposition* (Degenerate Case of $\rho=0$)** 当 $\rho=0$ 时, $f$ 是 ($L-0$) [[Hölder Smoothness]] 的, 其满足如下性质:
>   1. $\nabla f$ 在 $\Omega$ 上有界;
>   2. $f$ 在 $\Omega$ 上是 Lipschitz 连续的
>   3. 对于任意 $\mathbf{x}\neq \mathbf{y}$, 有:
>    $$
>     f(\mathbf{x}) \leq f(\mathbf{y}) + \nabla f(\mathbf{y})^\top (\mathbf{x}-\mathbf{y}) + L \|\mathbf{x}-\mathbf{y}\|
>    $$


### 1.2 First-Order Upper Bound for [[Hölder Smoothness]]


Hölder Smoothness 可以立刻推出如下上界, 并且注意到在 $\rho=1$ 时, 该结论退化为标准的 Lipschitz Smoothness 的上界.

***Proposition* (Generalized Descent Lemma)**: 对于任意 $\mathbf{x},\mathbf{y} \in \Omega$, 及 $\rho \in (0, 1]$, 有:
$$
f(\mathbf{x}) \leq f(\mathbf{y}) + \nabla f(\mathbf{y})^\top (\mathbf{x}-\mathbf{y}) + \frac{L}{1+\rho} \|\mathbf{x}-\mathbf{y}\|^{1+\rho}
$$

- *Proof.*
  - 构造辅助函数. 考虑从 $\mathbf{y}$ 沿着方向 $\mathbf{d} := \mathbf{x}-\mathbf{y}$ 行进微小步长 $t\in[0,1]$ 得到的函数值 $\phi(t) := f(\mathbf{y} + t\mathbf{d})$. 
    - 由于 $\Omega$ 是开凸集, 线段 $\mathbf{y} + t\mathbf{d} \in \Omega, ~ t\in[0,1]$. 故 $\phi(t)$ 在 $[0,1]$ 上可微, 且 $\phi'(t) = \langle\nabla f(\mathbf{y} + t\mathbf{d}), \mathbf{d}\rangle$. 
  - 由微积分基本定理,
    $$
    \begin{aligned}
    f(\mathbf{x}) - f(\mathbf{y}) &= \phi(1) - \phi(0) \\
    &= \int_0^1 \phi'(t) \mathrm{d}t \\
    &= \int_0^1 \langle\nabla f(\mathbf{y} + t\mathbf{d}), \mathbf{d}\rangle \mathrm{d}t \\
    &= \langle\nabla f(\mathbf{y}), \mathbf{d}\rangle + \int_0^1 \left\langle\nabla f(\mathbf{y} + t\mathbf{d}) - \nabla f(\mathbf{y}), \mathbf{d} \right\rangle ~\mathrm{d}t \\
    \end{aligned}
    $$

  - 对照 lemma 的形式, 只需对余项 $R:=\int_0^1 \left\langle\nabla f(\mathbf{y} + t\mathbf{d}) - \nabla f(\mathbf{y}), \mathbf{d} \right\rangle ~\mathrm{d}t$ 利用 Cauchy-Schwarz 不等式进行放缩. 
    - 根据 Hölder Smoothness 的定义, 有:
      $$
      \|\nabla f(\mathbf{y} + t\mathbf{d}) - \nabla f(\mathbf{y})\| \leq L \|t\mathbf{d}\|^\rho = L t^\rho \|\mathbf{d}\|^\rho
      $$
    - 根据 Cauchy-Schwarz 不等式, 有:
      $$
      \left\langle \nabla f(\mathbf{y} + t\mathbf{d}) - \nabla f(\mathbf{y}), \mathbf{d} \right\rangle \leq \|\nabla f(\mathbf{y} + t\mathbf{d}) - \nabla f(\mathbf{y})\| \cdot \|\mathbf{d}\| \leq L t^\rho \|\mathbf{d}\|^{1+\rho}
      $$
    - 从而对左右两侧同时对 $t$ 积分, 得到:
      $$
      R = \int_0^1 \left\langle \nabla f(\mathbf{y} + t\mathbf{d}) - \nabla f(\mathbf{y}), \mathbf{d} \right\rangle ~\mathrm{d}t \leq \int_0^1 L t^\rho \|\mathbf{d}\|^{1+\rho} ~\mathrm{d}t = \frac{L}{1+\rho} \|\mathbf{d}\|^{1+\rho} = \frac{L}{1+\rho} \|\mathbf{x}-\mathbf{y}\|^{1+\rho}
      $$
    - 从而得到:
      $$
      f(\mathbf{x}) \leq f(\mathbf{y}) + \nabla f(\mathbf{y})^\top (\mathbf{x}-\mathbf{y}) + \frac{L}{1+\rho} \|\mathbf{x}-\mathbf{y}\|^{1+\rho}
      $$
    $\square$


- 在凸且可微的情况下, 由一阶凸性不等式: $f(\mathbf{x}) \geq f(\mathbf{y}) + \langle\nabla f(\mathbf{y}), \mathbf{x}-\mathbf{y}\rangle$, 结合上述结论, 可以得到:
  $$
  0 \leq f(\mathbf{x}) - f(\mathbf{y}) - \langle\nabla f(\mathbf{y}), \mathbf{x}-\mathbf{y}\rangle \leq \frac{L}{1+\rho} \|\mathbf{x}-\mathbf{y}\|^{1+\rho}
  $$

### 1.3 Convergence Baseline 

#### Notations

首先统一规范记号. 考虑如下凸优化问题:
$$
\min_{\mathbf{x} \in Q} f(\mathbf{x})
$$

其中 $Q\subseteq \mathbb{R}^n$ 是非空闭凸集, $f: Q \to \mathbb{R}$ 是凸函数. 记最优值为:
$$
f^\star = \inf_{\mathbf{x} \in Q} f(\mathbf{x})
$$
对应的最优解集合为:
$$
\mathcal{X}^\star = \{\mathbf{x} \in Q: f(\mathbf{x}) = f^\star\}
$$

对于上述优化问题, 及算法产生的点列 $\{\mathbf{x}^{(k)}\}_{k=0}^\infty \subseteq Q$, 在给定精度 $\varepsilon > 0$ 下, 若存在 $k$ 使得:
$$
f(\mathbf{x}^{(k)}) - f^\star \leq \varepsilon
$$
则称算法收敛到最优值 $\mathbf{x}^\star$ 满足精度 $\varepsilon$. 

若 $\mathcal{X}^\star \neq \emptyset$, 记初始点到最优解集的距离为:
$$
R_0 := \text{dist}(\mathbf{x}^{(0)}, \mathcal{X}^\star) = \inf_{\mathbf{x}^\star \in \mathcal{X}^\star} \|\mathbf{x}^{(0)} - \mathbf{x}^\star\| < +\infty
$$

下对算法的收敛性进行初步讨论, 即为了达到 $f(\mathbf{x}^{(k)}) - f^\star \leq \varepsilon$, 需要多少次迭代 $k$.

#### Universal First-Order Method

这里采用 Nesterov 的 universal fast gradient method 作为一阶凸优化方法的复杂度标准基线. 

假设 $f$ 在 $\Omega$ 上是 ($L_\rho-\rho$) [[Hölder Smoothness]], $\rho \in (0,1]$. 则 universal fast gradient method 的复杂度为:

$$
k = \mathcal{O}\left(\left(\frac{L_\rho R_0^{1+\rho}}{\varepsilon}\right)^{\frac{2}{1+3\rho}}\right) = \mathcal{O}(\varepsilon^{\frac{-2}{1+3\rho}})
$$

- 特别地, 代入 $\rho = 1$ 和 $\rho = 0$ 的情况, 可以得到:
  - 当 $\rho = 1$ 时, 有:
    $$
    k = \mathcal{O}(\varepsilon^{\frac{-2}{4}}) = \mathcal{O}(\varepsilon^{-\frac{1}{2}})
    $$
  - 当 $\rho = 0$ 时, 有:
    $$
    k = \mathcal{O}(\varepsilon^{\frac{-2}{1}}) = \mathcal{O}(\varepsilon^{-2})
    $$

#### Goal of our Approach

本工作接下来的目标不是要改进任意 Holder smooth 函数的优化复杂度; 相反,  我们希望文体局有额外的结构, 例如 max-conjugate 形式的表达, 从而通过二次光滑化方法, 将其转化为一个 Holder smooth 的函数, 从而得到更好的优化复杂度.

具体而言, 我们希望在后续的结构化模型中, 最终能够逼近如下复杂度水平:
$$
\mathcal{O}(\varepsilon^{-\frac{1}{1+\rho}})
$$


## 2. A Simple Start : A Canonical Weakly Smooth Model



### 2.1 Notations and Problem Setup

为了说明我们后续的思路, 这里先从一个简单的 weakly smooth 模型开始:

$$
\min_{\mathbf{x} \in \mathbb{R}^n} F(\mathbf{x}), \quad F(\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|_p^p = \sum_{i=1}^m |\mathbf{a}_i^\top \mathbf{x}|^p, \quad p \in (1,2]
$$

其中 $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{a}_i^\top \in \mathbb{R}^{1 \times n}$ 是 $\mathbf{A}$ 的第 $i$ 行.

后文中, 一些关于向量的绝对值, 幂次与符号函数等操作, 都是逐元素操作. 例如对于 $\mathbf{z}  \in \mathbb{R}^m$, $|\mathbf{z}|^{p-2} \odot \mathbf{z} = \begin{pmatrix} |z_1|^{p-2} z_1,& \cdots, & |z_m|^{p-2} z_m \end{pmatrix}^\top$.


首先说明, 这里考虑的这个问题本身是平凡的, 因为对于 $\min_{\mathbf{x} \in \mathbb{R}^n} \|\mathbf{A}\mathbf{x}\|_p^p$, 其最优解 $F(\mathbf{x}) = 0$ 当且仅当 $\mathbf{A}\mathbf{x} = \mathbf{0}$. 因此我们更关注该问题的优化结构, 而非其求解算法本身. 

### 2.2 Weakly Smoothness of $\|\mathbf{A}\mathbf{x}\|_p^p$

#### Gradient Formula

记 $\mathbf{z} := \mathbf{A}\mathbf{x} \in \mathbb{R}^m$, $z_i = \mathbf{a}_i^\top \mathbf{x}, ~ i=1, \cdots, m$.  

由于 $p>1$, 故标量函数 $t \mapsto \frac{1}{p} |t|^p$ 是可微的, 其导数为:
$$
\frac{\mathrm{d}}{\mathrm{d}t} \frac{1}{p} |t|^p = |t|^{p-2} t := \psi(t)
$$

因此由链式法则:
$$
\nabla F(\mathbf{x}) = \mathbf{A}^\top \left(|\mathbf{A}\mathbf{x}|^{p-2} \odot (\mathbf{A}\mathbf{x})\right)
$$

若进一步用 $\psi(\cdot)$ 来表示该函数逐分量作用在向量 $\mathbf{A}\mathbf{x}$ 上的结果, 则可以简写为:
$$
\nabla F(\mathbf{x}) = \mathbf{A}^\top \psi(\mathbf{A}\mathbf{x})
$$

#### Weak Smoothness of $\|\mathbf{A}\mathbf{x}\|_p^p$

本节证明: 当 $p \in (1,2]$ 时, $F(\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|_p^p$ 是 $(p-1)$-[[Hölder Smoothness]] 的. 

***Lemma* (Scalar Hölder Continuity of $\psi$)** 对于 $\psi(t) = |t|^{p-2} t, ~ p\in (1,2]$, 其是 $L_p$-[[Hölder Continuous]] 的, 其中 $L_p := 2^{2-p}$. 即对于任意 $u,v \in \mathbb{R}$, 有:
$$
|\psi(u) - \psi(v)| \leq L_p |u-v|^{p-1}
$$

- *Proof.* 
  - 记 $\alpha = p-1 = (0,1]$, 则 $\psi(t) = |t|^\alpha \cdot \text{sign}(t)$. 下具体分两种情况进行讨论. 
  - 对于 $uv \ge 0$, 此时 $u$ 和 $v$ 同号或至少其一为 $0$.
    - 因此
      $$
      |\psi(u) - \psi(v)| = ||u|^\alpha - |v|^\alpha| 
      $$
    - 由于 $\alpha \in (0,1]$, 对于函数 $t \mapsto t^\alpha$ 有如下代数性质:
      $$
      |a^\alpha - b^\alpha| \leq |a-b|^\alpha, \quad \forall a,b \geq 0
      $$
      因此
      $$
      |\psi(u) - \psi(v)| = ||u|^\alpha - |v|^\alpha| \leq ||u| - |v||^\alpha \leq |u-v|^\alpha
      $$
  - 对于 $uv < 0$, 此时 $u$ 和 $v$ 异号. 因此
      $$
      |\psi(u) - \psi(v)| = |u|^\alpha + |v|^\alpha
       $$
       - 由于 $t \mapsto t^\alpha$ 是凹函数, 根据 Jensen 不等式, 有:
         $$
         \left(\frac{a+b}{2}\right)^\alpha \geq \frac{a^\alpha + b^\alpha}{2}, \quad \forall a,b \geq 0
         $$
      - 从而得到:
        $$
        |\psi(u) - \psi(v)| = |u|^\alpha + |v|^\alpha \leq 2 \left(\frac{|u| + |v|}{2}\right)^\alpha = 2^{1-\alpha} (|u-v|)^\alpha
        $$
        - 其中最后一个等式是由于 $u$ 和 $v$ 异号, 因此 $|u| + |v| = |u-v|$.
  
  $\square$

***Lemma* (Vector Hölder Bound in Euclidean Norm)** 对于 $p\in (1,2]$, 以及逐分量作用的函数 $\psi$, 对于任意 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^m$, 有:
$$
\|\psi(\mathbf{u}) - \psi(\mathbf{v})\|_2 \leq L_p \cdot m^{\frac{2-p}{2}}\cdot\|\mathbf{u} - \mathbf{v}\|_2^{p-1}, \quad L_p = 2^{2-p}
$$
- *Proof.* 
  - 由前一个 lemma, 对于每个分量 $i = 1, \cdots, m$, 有:
    $$
    |\psi(u_i) - \psi(v_i)| \leq L_p |u_i - v_i|^{p-1}
    $$
  - 因此
    $$
    \begin{aligned}
    \|\psi(\mathbf{u}) - \psi(\mathbf{v})\|_2^2 &= \sum_{i=1}^m |\psi(u_i) - \psi(v_i)|^2 \\
    &\leq L_p^2 \sum_{i=1}^m |u_i - v_i|^{2(p-1)} \\
    &\leq L_p^2 m^{\frac{2-p}{2}} \left(\sum_{i=1}^m |u_i - v_i|^2\right)^{p-1} = L_p^2 m^{\frac{2-p}{2}} \|\mathbf{u} - \mathbf{v}\|_2^{2(p-1)}
    \end{aligned}
    $$
    - 其中最后一个不等式是由于对于任意 $\mathbf{w}\in \mathbb{R}^m$, 以及 $\gamma \in (0,2]$, $\ell_1$ 和 $\ell_2$ 范数满足如下关系:
      $$
      \sum_{i=1}^m |w_i|^\gamma = \|\mathbf{w}\|_\gamma^\gamma \leq m^{\frac{2-\gamma}{2}} \|\mathbf{w}\|_2^\gamma
      $$

  $\square$


***Proposition* (Weak Smoothness of $\|\mathbf{A}\mathbf{x}\|_p^p$)** 设函数
$$
F(\mathbf{x}) = \frac{1}{p}\|\mathbf{A}\mathbf{x}\|_p^p, \quad p\in (1,2]
$$

则 $F$ 是 $\mathbb{R}^n$ 上的 ($L_F-\rho$) [[Hölder Smoothness]] 的, 其中 $\rho = p-1 \in (0,1]$, 可取 $L_F = 2^{2-p}\cdot m^{\frac{2-p}{2}}\cdot \|\mathbf{A}\|_2^p$. 即对于任意 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$, 有:
$$
\|\nabla F(\mathbf{x}) - \nabla F(\mathbf{y})\|_2 \leq L_F \|\mathbf{x}-\mathbf{y}\|_2^{p-1}
$$

$\diamond$

- *Proof.* 
  - 由梯度公式, 
    $$
    \nabla F(\mathbf{x}) - \nabla F(\mathbf{y}) = \mathbf{A}^\top \left(\psi(\mathbf{A}\mathbf{x}) - \psi(\mathbf{A}\mathbf{y})\right)
    $$
  - 因此
    $$
    \begin{aligned}
    \|\nabla F(\mathbf{x}) - \nabla F(\mathbf{y})\|_2 &\leq \|\mathbf{A}^\top\|_2 \cdot \|\psi(\mathbf{A}\mathbf{x}) - \psi(\mathbf{A}\mathbf{y})\|_2 \\
    &\leq \|\mathbf{A}\|_2 \cdot L_p \cdot m^{\frac{2-p}{2}}\cdot\|\mathbf{A}\mathbf{x} - \mathbf{A}\mathbf{y}\|_2^{p-1} \\
    &\leq \|\mathbf{A}\|_2^p \cdot L_p \cdot m^{\frac{2-p}{2}}\cdot\|\mathbf{x} - \mathbf{y}\|_2^{p-1}
    \end{aligned}
    $$

  $\square$


### 2.3 Fenchel Conjugate of $\frac{1}{p}\|\mathbf{z}\|_p^p$

对于函数 $h(\mathbf{y}) := \frac{1}{q}\|\mathbf{z}\|_q^q, ~y\in \mathbb{R}^m$, 其 Fenchel 共轭函数$h^*(\mathbf{z})$ 为:
$$
h^*(\mathbf{z}) = \frac{1}{p}\|\mathbf{y}\|_p^p, \quad \frac{1}{p} + \frac{1}{q} = 1
$$

因此回到我们原来的问题, 
$$
F(\mathbf{x}) = \frac{1}{p}\|\mathbf{A}\mathbf{x}\|_p^p = h^*(\mathbf{A}\mathbf{x})
$$

由 Fenchel 共轭的定义, 可以得到:
$$
F(\mathbf{x}) = h^*(\mathbf{A}\mathbf{x}) = \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - h(\mathbf{y})\right) = \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q\right)
$$

#### Explicit Form of the Conjugate Maximizer

首先尝试写出该最大化问题的显式解.

***Proposition* (Explicit Dual Maximizer)** 对于固定的 $\mathbf{z} \in \mathbb{R}^m$, 定义函数:
$$
\varphi(\mathbf{y}; \mathbf{z}) = \langle \mathbf{y}, \mathbf{z} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q
$$

则 $\varphi (\cdot; \mathbf{z})$ 有唯一的最大化解 $\mathbf{y}^\star$, 其满足如下显式表达式:
$$
\max_{\mathbf{y} \in \mathbb{R}^m} \varphi(\mathbf{y}; \mathbf{z}) = \frac{1}{p}\|\mathbf{z}\|_p^p
$$

$\diamond$

- *Proof*
  - $\|\mathbf{y}\|_q^q$ 是严格凸的,  从而 $\varphi(\mathbf{y}; \mathbf{z})$ 是严格凹的. 因此 $\varphi(\cdot; \mathbf{z})$ 的最大化解是唯一的.
  - 因此求解该无约束最大问题, 可以直接对 $\mathbf{y}$ 求梯度, 得到:
    $$
    \nabla_\mathbf{y} \varphi(\mathbf{y}; \mathbf{z}) = \mathbf{z} - |\mathbf{y}|^{q-2} \odot \mathbf{y}
    $$
    根据最优性条件 $\nabla_\mathbf{y} \varphi(\mathbf{y}^\star; \mathbf{z}) = 0$, 可以得到:
    $$
    |\mathbf{y}^\star|^{q-2} \odot \mathbf{y}^\star = \mathbf{z} \iff |y_i^\star|^{q-1}\text{sign}(y_i^\star) = z_i, ~ i=1, \cdots, m
    $$
  - 反解出 $\mathbf{y}^\star$:
    $$
    \begin{aligned}
    & y_i^\star = \text{sign}(z_i) \cdot |z_i|^{1/(q-1)} := \text{sign}(z_i) \cdot |z_i|^{p-1} \iff \mathbf{y}^\star = \text{sign}(\mathbf{z}) \odot |\mathbf{z}|^{p-1}
    \end{aligned}
    $$
  - 将 $\mathbf{y}^\star$ 代入 $\varphi(\mathbf{y}; \mathbf{z})$ 中, 得到:
    $$
    \begin{aligned}
    \varphi(\mathbf{y}^\star; \mathbf{z}) &= \langle \mathbf{y}^\star, \mathbf{z} \rangle - \frac{1}{q}\|\mathbf{y}^\star\|_q^q \\
    &= \sum_{i=1}^m y_i^\star z_i - \frac{1}{q} \sum_{i=1}^m |y_i^\star|^q = \sum_{i=1}^m |z_i|^{p-1} z_i - \frac{1}{q} \sum_{i=1}^m |z_i|^p = \frac{1}{p}\|\mathbf{z}\|_p^p
    \end{aligned}
    $$

  $\square$

***Corollary* (Explicit Dual Maximizer for $\mathbf{z} = \mathbf{A}\mathbf{x}$)** 将 $\varphi(\mathbf{y}; \mathbf{z})$ 中的 $\mathbf{z}$ 替换为 $\mathbf{A}\mathbf{x}$, 则对于每个固定的 $\mathbf{x} \in \mathbb{R}^n$, 有:
$$
F(\mathbf{x}) = \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q\right) = \frac{1}{p}\|\mathbf{A}\mathbf{x}\|_p^p
$$

且对应的唯一最大化解 $\mathbf{y}^\star$ 满足如下显式表达式: 
$$
\mathbf{y}^\star = \text{sign}(\mathbf{A}\mathbf{x}) \odot |\mathbf{A}\mathbf{x}|^{p-1}
$$

### 2.4 Summary

本节说明, 对于 $F(\mathbf{x}) = \frac{1}{p}\|\mathbf{A}\mathbf{x}\|_p^p$, 其是一个 structured but not smooth 的模型. 并且可以从 primal 和 dual 两个角度来理解该模型的结构.

- 从 Primal 角度, $F$ 是一个 weakly smooth 的函数, 其梯度的 Holder smoothness 参数 $\rho = p-1 \in (0,1]$.
- 从 Dual 角度, $F$ 可以写成一个 max-conjugate 的形式, 即 $F(\mathbf{x}) = \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q\right)$, 其中 $\frac{1}{q}\|\mathbf{y}\|_q^q$ 是一个 strongly convex 的函数.

## 3. Traditional Approach: Standard Smooth via Quadratic Approximation

### 3.1 Problem Setup

令 $\mathcal{Y} \subseteq \mathbb{R}^m$ 是紧且凸的. 考虑目标函数
$$
G(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y}} \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle, \quad \mathbf{x} \in \mathbb{R}^n, \mathbf{A} \in \mathbb{R}^{m \times n}
$$

对于该目标函数, 我们有如下几点观察:
- $G$ 是凸函数, 因为其是关于 $\mathbf{x}$ 的线性函数的上确界.
- 由于 $\mathcal{Y}$ 是紧的, 因此 $G$ 是有界的, 故最大值总能够达到. 
- $G$ 一般是不可微的, 因为对于某些 $\mathbf{x}$, 其最大值解可能不是唯一的. 

综上, $G$ 一般是一个 structured but not smooth 的模型. 

### 3.2 Quadratic Smoothing

对于给定的 smoothness 参数 $\mu > 0$, 定义 $G$ 的 quadratic smoothing 近似为:
$$
G_\mu(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y}} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}\|^2\right) 
$$

并且对应的近似点为:
$$
\mathbf{y}_\mu(\mathbf{x}) \in \text{arg} \max_{\mathbf{y} \in \mathcal{Y}} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}\|^2\right) = \arg\min_{\mathbf{y} \in \mathcal{Y}} \left(\frac{\mu}{2}\|\mathbf{y}\|^2 - \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle\right)
$$

此外, 由于 $-\frac{\mu}{2}\|\mathbf{y}\|^2$ 是 $\mu$-strongly concave 的, 而 $\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle$ 是线性的, 因此 $\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}\|^2$ 是 $\mu$-strongly concave 的, 从而其最大值解是唯一的.

### 3.3 Explicit Form of Smoothed Optimizer

首先说明, 上述的 quadratic smoothing 近似方法相当于在 $\mathcal{Y}$ 上进行欧氏投影.

***Proposition* (Euclidean Projection for Smooth Optimizer)** 对于每个给定的 $\mathbf{x}\in \mathbb{R}^n$, 二次光滑函数 $G_\mu(\mathbf{x})$ 的近似点 $\mathbf{y}_\mu(\mathbf{x})$ 可以通过如下欧氏投影得到:
$$
\mathbf{y}_\mu(\mathbf{x}) = \arg\min_{\mathbf{y} \in \mathcal{Y}} \left(\frac{\mu}{2}\|\mathbf{y}\|_2^2 - \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle\right) = \text{Proj}_{\mathcal{Y}}\left(\frac{\mathbf{A}\mathbf{x}}{\mu}\right)
$$
其中 $\text{Proj}_{\mathcal{Y}}(\mathbf{z}) = \arg\min_{\mathbf{y} \in \mathcal{Y}} \|\mathbf{y} - \mathbf{z}\|_2$ 是 $\mathbf{z}$ 在 $\mathcal{Y}$ 上的欧氏投影.

$\diamond$

- *Proof.* 
  - 对于给定的 $\mathbf{x}$, 考虑 $\mathbf{y}_\mu(\mathbf{x}) = \arg\min_{\mathbf{y} \in \mathcal{Y}} \left(\frac{\mu}{2}\|\mathbf{y}\|_2^2 - \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle\right)$.
  - 对目标进行配方, 得到:
    $$
    \begin{aligned}
    &\frac{\mu}{2}\|\mathbf{y}\|_2^2 - \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle \\
    =& \frac{\mu}{2}\|\mathbf{y}\|_2^2 - \langle \mathbf{y}, \mu \cdot \frac{\mathbf{A}\mathbf{x}}{\mu} \rangle \\
    =& \frac{\mu}{2}\|\mathbf{y}\|_2^2 - \langle \mathbf{y}, \mu \cdot \frac{\mathbf{A}\mathbf{x}}{\mu} \rangle + \frac{\mu}{2} \left\|\frac{\mathbf{A}\mathbf{x}}{\mu}\right\|_2^2 -  \frac{\mu}{2} \left\|\frac{\mathbf{A}\mathbf{x}}{\mu}\right\|_2^2\\
    =&  \frac{\mu}{2} \left\|\mathbf{y} -  \frac{\mathbf{A}\mathbf{x}}{\mu}\right\|_2^2 -  \frac{\|\mathbf{A}\mathbf{x}\|_2^2}{2\mu}
    \end{aligned}
    $$
  - 因此
    $$
    \begin{aligned}
    \mathbf{y}_\mu(\mathbf{x}) &= \arg\min_{\mathbf{y} \in \mathcal{Y}} \left(\frac{\mu}{2}\|\mathbf{y}\|_2^2 - \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle\right) \\
    &= \arg\min_{\mathbf{y} \in \mathcal{Y}} \left( \frac{\mu}{2} \left\|\mathbf{y} -  \frac{\mathbf{A}\mathbf{x}}{\mu}\right\|_2^2 -  \frac{\|\mathbf{A}\mathbf{x}\|_2^2}{2\mu} \right) \\
    &= \arg\min_{\mathbf{y} \in \mathcal{Y}} \left\|\mathbf{y} -  \frac{\mathbf{A}\mathbf{x}}{\mu}\right\|_2^2 = \text{Proj}_{\mathcal{Y}}\left(\frac{\mathbf{A}\mathbf{x}}{\mu}\right)
    \end{aligned}
    $$

  $\square$


### 3.4 Lipschitz Continuity of $\mathbf{y}_\mu(\mathbf{x})$

***Lemma* (Non-expansiveness of Projection)** 设 $\mathcal{Y} \subseteq \mathbb{R}^m$ 是非空闭凸集, 则对于任意 $\mathbf{z}_1, \mathbf{z}_2 \in \mathbb{R}^m$, 其在 $\mathcal{Y}$ 上的欧氏投影满足:
$$
\|\text{Proj}_{\mathcal{Y}}(\mathbf{z}_1) - \text{Proj}_{\mathcal{Y}}(\mathbf{z}_2)\|_2 \leq \|\mathbf{z}_1 - \mathbf{z}_2\|_2
$$

$\diamond$

- *Proof.* 
  - 记 $\mathbf{y}_1 = \text{Proj}_{\mathcal{Y}}(\mathbf{z}_1)$ 和 $\mathbf{y}_2 = \text{Proj}_{\mathcal{Y}}(\mathbf{z}_2)$.
  - 根据欧氏投影的最优性条件, 对于任意 $\mathbf{y} \in \mathcal{Y}$, 有:
    $$
    \langle \mathbf{z_1} - \mathbf{y}_1, \mathbf{y} - \mathbf{y}_1 \rangle \leq 0, \quad \langle \mathbf{z_2} - \mathbf{y}_2, \mathbf{y} - \mathbf{y}_2 \rangle \leq 0
    $$
  - 将 $\mathbf{y} = \mathbf{y}_2$ 代入第一个不等式, 将 $\mathbf{y} = \mathbf{y}_1$ 代入第二个不等式, 得到:
    $$
    \langle \mathbf{z_1} - \mathbf{y}_1, \mathbf{y}_2 - \mathbf{y}_1 \rangle \leq 0, \quad \langle \mathbf{z_2} - \mathbf{y}_2, \mathbf{y}_1 - \mathbf{y}_2 \rangle \leq 0
    $$
  - 将上述两个不等式相加, 得到:
    $$
    \begin{aligned}
    &\langle \mathbf{z}_1 - \mathbf{y}_1, \mathbf{y}_2 - \mathbf{y}_1 \rangle + \langle \mathbf{y}_2 - \mathbf{z}_2, \mathbf{y}_2 - \mathbf{y}_1 \rangle = \langle \mathbf{z}_1 - \mathbf{z}_2, \mathbf{y}_2 - \mathbf{y}_1 \rangle - \|\mathbf{y}_2 - \mathbf{y}_1\|_2^2  \leq 0 
    \end{aligned}
    $$
    整理并根据 Cauchy-Schwarz 不等式得到:
    $$
    \|\mathbf{y}_1- \mathbf{y}_2\|_2^2 \leq \langle \mathbf{z}_1 - \mathbf{z}_2, \mathbf{y}_1 - \mathbf{y}_2 \rangle \leq \|\mathbf{z}_1 - \mathbf{z}_2\|_2 \cdot \|\mathbf{y}_2 - \mathbf{y}_1\|_2
    $$
  - 最后, 若 $\mathbf{y}_1 = \mathbf{y}_2$, 则不等式显然成立; 否则两边同时除以 $\|\mathbf{y}_1 - \mathbf{y}_2\|_2$, 得到:
    $$
    \|\mathbf{y}_1- \mathbf{y}_2\|_2 \leq \|\mathbf{z}_1 - \mathbf{z}_2\|_2
    $$

  $\square$

***Corollary* (Lipschitz Continuity of $\mathbf{y}_\mu(\mathbf{x})$)** 对于任意 $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^n$, 有:
$$
\|\mathbf{y}_\mu(\mathbf{x}_1) - \mathbf{y}_\mu(\mathbf{x}_2)\|_2 \leq \frac{\|\mathbf{A}\|_2}{\mu} \|\mathbf{x}_1 - \mathbf{x}_2\|_2
$$

$\diamond$

- *Proof.* 
  - 由前一个 proposition, $\mathbf{y}_\mu(\mathbf{x}) = \text{Proj}_{\mathcal{Y}}\left(\frac{\mathbf{A}\mathbf{x}}{\mu}\right)$. 
  - 因此
    $$
    \begin{aligned}
    \|\mathbf{y}_\mu(\mathbf{x}_1) - \mathbf{y}_\mu(\mathbf{x}_2)\|_2 &= \left\|\text{Proj}_{\mathcal{Y}}\left(\frac{\mathbf{A}\mathbf{x}_1}{\mu}\right) - \text{Proj}_{\mathcal{Y}}\left(\frac{\mathbf{A}\mathbf{x}_2}{\mu}\right)\right\|_2 \\
    &\leq \left\|\frac{\mathbf{A}\mathbf{x}_1}{\mu} - \frac{\mathbf{A}\mathbf{x}_2}{\mu}\right\|_2 = \frac{\|\mathbf{A}(\mathbf{x}_1 - \mathbf{x}_2)\|_2}{\mu} \leq \frac{\|\mathbf{A}\|_2}{\mu} \|\mathbf{x}_1 - \mathbf{x}_2\|_2
    \end{aligned}
    $$

### 3.5 Smoothness of Quadratic Smoothing $G_\mu(\mathbf{x})$

***Proposition* (Smoothness of Quadratic Smoothing)** 设函数
$$
G_\mu(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y} } \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}\|^2\right)
$$
则 $G_\mu$ 在 $\mathbb{R}^n$ 上可微且
$$
\nabla G_\mu(\mathbf{x}) = \mathbf{A}^\top \mathbf{y}_\mu(\mathbf{x})
$$

进一步, $G_\mu$ 是 $\frac{\|\mathbf{A}\|_2^2}{\mu}$-[[Lipschitz Smoothness]] 的. 即对于任意 $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^n$, 有:
$$
\|\nabla G_\mu(\mathbf{x}_1) - \nabla G_\mu(\mathbf{x}_2)\|_2 \leq \frac{\|\mathbf{A}\|_2^2}{\mu} \|\mathbf{x}_1 - \mathbf{x}_2\|_2
$$

$\diamond$

- *Proof*
  - 对于 $G_\mu(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y} } \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}\|^2\right)$ 的 RHS 对应的最大化问题, 其最大化解 $\mathbf{y}_\mu(\mathbf{x})$ 是唯一的. 因此根据 Danskin 定理$^\dagger$, $G_\mu$ 关于 $\mathbf{x}$ 可微, 且其梯度为:
    $$
    \nabla G_\mu(\mathbf x) \overset{\dagger}{=}\nabla_{\mathbf x}\left(\langle \mathbf y,\mathbf A\mathbf x\rangle-\frac{\mu}{2}\|\mathbf y\|_2^2\right)\Bigg|_{\mathbf y=\mathbf y_\mu(\mathbf x)}=\mathbf A^\top \mathbf y_\mu(\mathbf x).
    $$

  - 因此对任意 $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^n$, 有:
    $$
    \begin{aligned}
    &\|\nabla G_\mu(\mathbf{x}_1) - \nabla G_\mu(\mathbf{x}_2)\|_2 = \|\mathbf{A}^\top \mathbf{y}_\mu(\mathbf{x}_1) - \mathbf{A}^\top \mathbf{y}_\mu(\mathbf{x}_2)\|_2 \\
    &\leq \|\mathbf{A}^\top\|_2 \cdot \|\mathbf{y}_\mu(\mathbf{x}_1) - \mathbf{y}_\mu(\mathbf{x}_2)\|_2 \leq \frac{\|\mathbf{A}\|_2^2}{\mu} \|\mathbf{x}_1 - \mathbf{x}_2\|_2
    \end{aligned}
    $$

  $\square$


### 3.6 Uniform Smoothing Error

对于紧集 $\mathcal{Y}$,  定义其的半径为 $R_{\mathcal{Y}} := \max_{\mathbf{y} \in \mathcal{Y}} \|\mathbf{y}\|_2 < +\infty$.  如下定理说明, 当 $\mathcal{Y}$ 有界时, 该平滑误差可以在 $\mathbf{x}$ 上 uniformly 控制.

***Proposition* (Uniform Smoothing Error)** 对于任意 $\mathbf{x} \in \mathbb{R}^n$, 有:
$$
0 \leq G(\mathbf{x}) - G_\mu(\mathbf{x}) \leq \frac{\mu}{2} R_{\mathcal{Y}}^2
$$

- 首先证明左侧不等式. 
  - 对于任意 $\mathbf{y} \in \mathcal{Y}$, 有:
    $$
    \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}\|^2 \leq \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle
    $$
  - 因此
    $$
    G_\mu(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y}} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}\|^2\right) \leq \max_{\mathbf{y} \in \mathcal{Y}} \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle = G(\mathbf{x})
    $$

- 接下来证明右侧不等式. 
  - 对于问题 $G(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y}} \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle$, 取任意 $\mathbf{y}^\star \in \text{arg}\max_{\mathbf{y} \in \mathcal{Y}} \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle$, 则
    $$
    \begin{aligned}
    G_\mu(\mathbf{x}) &= \max_{\mathbf{y} \in \mathcal{Y}} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}\|^2\right) \\
    &\geq \langle \mathbf{y}^\star, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}^\star\|^2 \\
    &= G(\mathbf{x}) - \frac{\mu}{2}\|\mathbf{y}^\star\|^2 \geq G(\mathbf{x}) - \frac{\mu}{2} R_{\mathcal{Y}}^2
    \end{aligned}
    $$

  $\square$


这个地方最终给我们一个启示:
- 对于模型 $G(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y}} \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle$, 我们可以通过如下方法得到二次光滑近似:
    $$
    G_\mu(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y}} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}\|_2^2\right)
    $$
- 并且这个近似本身是依赖 $\mathbf{y}_\mu(\mathbf{x}) =  \arg\max_{\mathbf{y} \in \mathcal{Y}} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}\|_2^2\right)$ 的. 只要这个近似点 $\mathbf{y}_\mu(\mathbf{x})$ 足够稳定 (是 Lipschitz 连续的), 那么我们就可以证明 $G_\mu$ 是一个 smooth 的函数, 有 $\nabla G_\mu(\mathbf{x}) = \mathbf{A}^\top \mathbf{y}_\mu(\mathbf{x})$.

## 4. Structured Smoothing of the Canonical Weakly Smooth Model

### 4.1 Recap and Problem Setup

- 回顾, 在 Section 2 中, 我们说明, 对于函数:
  $$
  F(\mathbf{x}) = \frac{1}{p}\|\mathbf{A}\mathbf{x}\|_p^p, \quad p\in (1,2]
  $$

  其在 primal 上是 $\rho = p-1$-Holder smooth 的. 并且在 dual 上, 可以写成一个 max-conjugate 的形式:
  $$
  F(\mathbf{x}) = \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q\right), \quad \frac{1}{p} + \frac{1}{q} = 1
  $$

  其中, 若记 $h(\mathbf{y}) = \frac{1}{q}\|\mathbf{y}\|_q^q$, 则:
  $$
  F(\mathbf{x}) = h^*(\mathbf{A}\mathbf{x})
  $$

  并且对于每个固定的 $\mathbf{x}$, 其对应的最大化解 $\mathbf{y}^\star$ 满足如下显式表达式:
  $$
  \mathbf{y}_{F}^\star(\mathbf{x})= \arg\max_{\mathbf{y} \in \mathbb{R}^m} F(\mathbf{x}) = \text{sign}(\mathbf{A}\mathbf{x}) \odot |\mathbf{A}\mathbf{x}|^{p-1}
  $$

  且 $F$ 的梯度为 
  $$
  \nabla F(\mathbf{x}) = \mathbf{A}^\top \mathbf{y}^\star(\mathbf{x})
  $$


- 另一方面, 在 Section 3 中, 我们说明, 对于函数
  $$
  G(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y}} \langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle, \quad \mathcal{Y} \subseteq \mathbb{R}^m \text{ is compact convex set}
  $$

  那么其 quadratic smoothing 近似为:
  $$
  G_\mu(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y}} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{\mu}{2}\|\mathbf{y}\|_2^2\right)
  $$

  此时可以证明,  $G_\mu$ 是一个 $\frac{\|\mathbf{A}\|_2^2}{\mu}$ - Lipschitz smooth 的函数, 其梯度为:
  $$
  \nabla G_\mu(\mathbf{x}) = \mathbf{A}^\top \mathbf{y}_\mu(\mathbf{x})
  $$

  并且还可以给出其 uniform smoothing bias:
  $$
  0 \leq G(\mathbf{x}) - G_\mu(\mathbf{x}) \leq \frac{\mu}{2} \max_{\mathbf{y} \in \mathcal{Y}} \|\mathbf{y}\|_2^2.
  $$

---

最终在本章中, 我们希望对于原始模型:
$$
F(\mathbf{x}) = \frac{1}{p}\|\mathbf{A}\mathbf{x}\|_p^p = \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q\right), ~\text{where } p\in (1,2], ~\frac{1}{p} + \frac{1}{q} = 1
$$

给出一个新的 smoothing 近似:
$$
F_\mu(\mathbf{x}) = \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q - \frac{\mu}{2}\|\mathbf{y}\|_2^2\right)
$$

并且得到对应的唯一最大值解:
$$
\mathbf{y}^{\star}_\mu(\mathbf{x}) = \arg\max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q - \frac{\mu}{2}\|\mathbf{y}\|_2^2\right)
$$

***Proposition* (Existence and Uniqueness of Dual Maximizer)** 对于每个固定的 $\mathbf{x} \in \mathbb{R}^n$, 上述的最大化问题有唯一的最大化解 $\mathbf{y}^{\star}_\mu(\mathbf{x})$. 并且进一步有 $\nabla F_\mu(\mathbf{x}) = \mathbf{A}^\top \mathbf{y}_\mu(\mathbf{x})$.

- *Proof.*
  - 由于 $\frac{\mu}{2} \|\mathbf{y}\|_2^2$ 是 $\mu$-strongly convex 的, 而 $\frac{1}{q}\|\mathbf{y}\|_q^q$ 是 convex 的, 因此 $-\frac{\mu}{2}\|\mathbf{y}\|_2^2 - \frac{1}{q}\|\mathbf{y}\|_q^q$ 是 $\mu$-strongly concave 的. 又因为 $\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle$ 是线性的, 因此 $\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q - \frac{\mu}{2}\|\mathbf{y}\|_2^2$ 是 $\mu$-strongly concave 的. 从而其最大化解是唯一的.
  - 同时, 又由于 inner maximizer $\mathbf{y}_\mu(\mathbf{x})$ 是唯一的, 因此根据 Danskin 定理, $F_\mu$ 关于 $\mathbf{x}$ 可微, 且其梯度为:
    $$
    \nabla F_\mu(\mathbf{x}) = \nabla_{\mathbf{x}} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q - \frac{\mu}{2}\|\mathbf{y}\|_2^2\right)\Bigg|_{\mathbf{y} = \mathbf{y}_\mu(\mathbf{x})} = \mathbf{A}^\top \mathbf{y}_\mu(\mathbf{x})
    $$

  $\square$

并且回答如下三个问题:
1. 为什么要在 dual 上进行 quadratic smoothing?
2. $F_\mu$ 的梯度, 误差控制, $\mu$ 的选择等问题.
3. 最终算法的复杂度分析.

### 4.2 Uniform Convexity of the Dual Penalty

对于 Dual Penalty:
$$
h(\mathbf{x}) = \frac{1}{q}\|\mathbf{x}\|_q^q, ~ q\ge 2
$$
其不仅是凸的, 并且有 $q$-uniformly convex 的性质:

***Lemma* (Uniform Convexity of $h(\mathbf{y})$)** 设 $h(\mathbf{y}) = \frac{1}{q}\|\mathbf{y}\|_q^q, ~ q\ge 2$, 则对于任意 $\mathbf{y}_1, \mathbf{y}_2 \in \mathbb{R}^m$, 有:
$$
h(\mathbf{y}_1) \geq h(\mathbf{y}_2) + \langle \nabla h(\mathbf{y}_2), \mathbf{y}_1 - \mathbf{y}_2 \rangle +c_q \|\mathbf{y}_1 - \mathbf{y}_2\|_q^q
$$

其中 $\nabla h(\mathbf{y}) = \text{sign}(\mathbf{y}) \odot |\mathbf{y}|^{q-1}$, 并且 $c_q > 0$ 是一个仅依赖于 $q$ 的正常数.

$\diamond$

***Corollary* (Monotonicity of $\nabla h(\mathbf{y})$)** 设 $h(\mathbf{y}) = \frac{1}{q}\|\mathbf{y}\|_q^q, ~ q\ge 2$, 则对于任意 $\mathbf{y}_1, \mathbf{y}_2 \in \mathbb{R}^m$, 有:
$$
\langle \nabla h(\mathbf{y}_1) - \nabla h(\mathbf{y}_2), \mathbf{y}_1 - \mathbf{y}_2 \rangle \geq c_q \|\mathbf{y}_1 - \mathbf{y}_2\|_q^q
$$

$\diamond$

### 4.3 Estimate of the Smoothed Optimizer $\mathbf{y}_\mu(\mathbf{x})$

***Proposition* (Key Estimate for $\mathbf{y}_\mu(\mathbf{x})$)** 给定 $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^n$, 以及对应的 $\mathbf{y}_1 := \mathbf{y}_\mu(\mathbf{x}_1)$ 和 $\mathbf{y}_2 := \mathbf{y}_\mu(\mathbf{x}_2)$, 记 $\Delta \mathbf{x} = \mathbf{x}_1 - \mathbf{x}_2$ 和 $\Delta \mathbf{y} = \mathbf{y}_1 - \mathbf{y}_2$, 则有如下估计:

$$
\boxed{c_q \|\Delta \mathbf{y}\|_q^q + \mu\|\Delta \mathbf{y}\|_2^2 \leq \langle \mathbf{A}(\Delta \mathbf{x}), \Delta \mathbf{y} \rangle}
$$

$\diamond$

- *Proof.* 
  - 由一阶最优性条件, 对于 $\mathbf{y}_1 = \mathbf{y}_\mu(\mathbf{x}_1)$ 和 $\mathbf{y}_2 = \mathbf{y}_\mu(\mathbf{x}_2)$, 有:
    $$
    \begin{aligned}
    & \nabla h(\mathbf{y}_1) + \mu \mathbf{y}_1 - \mathbf{A}\mathbf{x}_1 = \mathbf{0}\\
    & \nabla h(\mathbf{y}_2) + \mu \mathbf{y}_2 - \mathbf{A}\mathbf{x}_2 = \mathbf{0}
    \end{aligned}
    $$
  - 将上述两个等式相减, 得到:
    $$
    \nabla h(\mathbf{y}_1) - \nabla h(\mathbf{y}_2) + \mu (\mathbf{y}_1 - \mathbf{y}_2) = \mathbf{A}(\mathbf{x}_1 - \mathbf{x}_2)
    $$
  - 因此再与 $\mathbf{y}_1 - \mathbf{y}_2$ 内积, 得到:
    $$
    \begin{aligned}
    &\langle \nabla h(\mathbf{y}_1) - \nabla h(\mathbf{y}_2), \mathbf{y}_1 - \mathbf{y}_2 \rangle + \mu\|\mathbf{y}_1 - \mathbf{y}_2\|_2^2 = \langle \mathbf{A}(\mathbf{x}_1 - \mathbf{x}_2), \mathbf{y}_1 - \mathbf{y}_2 \rangle
    \end{aligned}
    $$
  - 再根据前一个 corollary, 有 $\langle \nabla h(\mathbf{y}_1) - \nabla h(\mathbf{y}_2), \mathbf{y}_1 - \mathbf{y}_2 \rangle \geq c_q \|\mathbf{y}_1 - \mathbf{y}_2\|_q^q$, 从而得到:
    $$
    c_q \|\Delta \mathbf{y}\|_q^q + \mu\|\Delta \mathbf{y}\|_2^2 \leq \langle \mathbf{A}(\Delta \mathbf{x}), \Delta \mathbf{y} \rangle
    $$

  $\square$


### 4.4 Two-Regime Bounds for $\mathbf{y}_\mu(\mathbf{x})$

回顾我们在前文已经定义了:
$$
F_\mu(\mathbf{x})=\max_{\mathbf{y}\in\mathbb{R}^m}\left\{\langle \mathbf{y},\mathbf{A}\mathbf{x}\rangle-\frac{1}{q}\|\mathbf{y}\|_q^q-\frac{\mu}{2}\|\mathbf{y}\|_2^2\right\},\qquad\frac{1}{p}+\frac{1}{q}=1,\quad p\in(1,2].
$$

并且记唯一的最大值解为:
$$
\mathbf{y}^{\star}_\mu(\mathbf{x})\in\arg\max_{\mathbf{y}\in\mathbb{R}^m}\left\{\langle \mathbf{y},\mathbf{A}\mathbf{x}\rangle-\frac{1}{q}\|\mathbf{y}\|_q^q-\frac{\mu}{2}\|\mathbf{y}\|_2^2\right\}.
$$

由于:
$$
\nabla F_\mu(\mathbf{x})=\mathbf{A}^\top \mathbf{y}_\mu^{\star}(\mathbf{x}),
$$
因此要研究 $F_\mu$ 的 smoothness, 关键在于研究 $\mathbf{y}_\mu(\mathbf{x})$ 的有关性质. 

#### Hölder-type Bound for $\mathbf{y}_\mu(\mathbf{x})$

***Proposition* (Hölder-type Bound)** 在上述记号下, 有
$$
\|\Delta \mathbf{y}\|_q \leq c_q^{-\frac{1}{q-1}} \cdot \|\mathbf{A} \Delta \mathbf{x}\|_p^{\frac{1}{q-1}} = c_q^{-\frac{1}{q-1}} \cdot \|\mathbf{A} \Delta \mathbf{x}\|_p^{p-1}
$$

$\diamond$

- *Proof*
  - 由前一个 proposition, 有:
    $$
    c_q \|\Delta \mathbf{y}\|_q^q + \mu\|\Delta \mathbf{y}\|_2^2 \leq \langle \mathbf{A}(\Delta \mathbf{x}), \Delta \mathbf{y} \rangle \implies c_q \|\Delta \mathbf{y}\|_q^q \leq \langle \mathbf{A}(\Delta \mathbf{x}), \Delta \mathbf{y} \rangle
     $$
  - 又根据 Hölder 不等式, 有 $\langle \mathbf{A}(\Delta \mathbf{x}), \Delta \mathbf{y} \rangle \leq \|\mathbf{A} \Delta \mathbf{x}\|_p \cdot \|\Delta \mathbf{y}\|_q$, 从而得到:
    $$
    c_q \|\Delta \mathbf{y}\|_q^q \leq \|\mathbf{A} \Delta \mathbf{x}\|_p \cdot \|\Delta \mathbf{y}\|_q
    $$
  - 最后, 若 $\Delta \mathbf{y} = \mathbf{0}$, 则不等式显然成立; 否则两边同时除以 $\|\Delta \mathbf{y}\|_q$, 得到:
    $$
    \|\Delta \mathbf{y}\|_q^{q-1} \leq c_q^{-1} \cdot \|\mathbf{A} \Delta \mathbf{x}\|_p \implies \|\Delta \mathbf{y}\|_q \leq c_q^{-\frac{1}{q-1}} \cdot \|\mathbf{A} \Delta \mathbf{x}\|_p^{\frac{1}{q-1}} = c_q^{-\frac{1}{q-1}} \cdot \|\mathbf{A} \Delta \mathbf{x}\|_p^{p-1}
    $$

$\square$

***Corollary* (Euclidean Hölder-type Bound)** 在上述记号下, 若统一规范在 Euclidean 范数下, 存在常数 $C_{p,m} := c_q^{-\frac{1}{q-1}} m^{\frac{2-p}{2}}$, 使得
$$
\|\Delta \mathbf{y}\|_2 \leq C_{p,m} \cdot \|\mathbf{A}\|_2^{p-1} \cdot \|\Delta \mathbf{x}\|_2^{p-1} 
$$

$\diamond$

#### Lipschitz-type Bound for $\mathbf{y}_\mu(\mathbf{x})$

***Proposition* (Lipschitz-type Bound)** 在上述记号下, 有
$$
\|\Delta \mathbf{y}\|_2 \leq \frac{1}{\mu} \cdot \|\mathbf{A} \Delta \mathbf{x}\|_2\leq \frac{\|\mathbf{A}\|_2}{\mu} \cdot \|\Delta \mathbf{x}\|_2
$$

$\diamond$

- *Proof.* 
  - 由前一个 proposition, 有:
    $$
    c_q \|\Delta \mathbf{y}\|_q^q + \mu\|\Delta \mathbf{y}\|_2^2 \leq \langle \mathbf{A}(\Delta \mathbf{x}), \Delta \mathbf{y} \rangle \implies \mu\|\Delta \mathbf{y}\|_2^2 \leq \langle \mathbf{A}(\Delta \mathbf{x}), \Delta \mathbf{y} \rangle
    $$
  - 又根据 Cauchy-Schwarz 不等式, 有 $\langle \mathbf{A}(\Delta \mathbf{x}), \Delta \mathbf{y} \rangle \leq \|\mathbf{A}(\Delta \mathbf{x})\|_2 \cdot \|\Delta \mathbf{y}\|_2$, 从而得到:
    $$
    \mu\|\Delta \mathbf{y}\|_2^2 \leq  \|\mathbf{A}(\Delta \mathbf{x})\|_2 \cdot \|\Delta \mathbf{y}\|_2
    $$
  - 最后, 若 $\Delta\mathbf{y} = 0$, 则不等式显然成立; 否则两边同时除以 $\|\Delta\mathbf{y}\|_2$, 得到:
    $$
    \|\Delta\mathbf{y}\|_2\leq\frac{\|\mathbf{A}(\Delta\mathbf{x})\|_2}{\mu}\leq\frac{\|\mathbf{A}\|_2}{\mu}\cdot\|\Delta\mathbf{x}\|_2.
    $$

$\square$

### 4.5 Smoothness of $F_\mu(\mathbf{x})$

根据 $\nabla \mathbf{F}_\mu(\mathbf{x}) = \mathbf{A}^\top \mathbf{y}^{\star}_\mu(\mathbf{x})$, 因此上述关于 $\mathbf{y}^{\star}_\mu(\mathbf{x})$ 的讨论可以转化为 $F_\mu$ 的 smoothness 分析.

***Proposition* (Smoothness of $F_\mu$)** 设 $F_\mu(\mathbf{x}) = \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q - \frac{\mu}{2}\|\mathbf{y}\|_2^2\right)$, 则对于任意 $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^n$, 有:

**Hölder Smoothness:**
$$
\|\nabla F_\mu(\mathbf{x}_1) - \nabla F_\mu(\mathbf{x}_2)\|_2 \leq L_H \cdot \|\mathbf{x}_1 - \mathbf{x}_2\|_2^{p-1}, \quad \text{where } L_H := C_{p,m} \cdot \|\mathbf{A}\|_2^p
$$

**Lipschitz Smoothness:**
$$
\|\nabla F_\mu(\mathbf{x}_1) - \nabla F_\mu(\mathbf{x}_2)\|_2 \leq L_\mu \cdot \|\mathbf{x}_1 - \mathbf{x}_2\|_2, \quad \text{where } L_\mu := \frac{\|\mathbf{A}\|_2^2}{\mu}
$$


$\diamond$

- *Proof.* 
  - 由 $\nabla F_\mu(\mathbf{x}) = \mathbf{A}^\top \mathbf{y}_\mu(\mathbf{x})$, 因此对于任意 $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^n$, 有:
    $$
    \begin{aligned}
    &\|\nabla F_\mu(\mathbf{x}_1) - \nabla F_\mu(\mathbf{x}_2)\|_2 = \|\mathbf{A}^\top \mathbf{y}_\mu(\mathbf{x}_1) - \mathbf{A}^\top \mathbf{y}_\mu(\mathbf{x}_2)\|_2 \\
    &\leq \|\mathbf{A}^\top\|_2 \cdot \|\mathbf{y}_\mu(\mathbf{x}_1) - \mathbf{y}_\mu(\mathbf{x}_2)\|_2
    \end{aligned}
    $$

  - 因此, 将前面两个 proposition 中关于 $\|\Delta \mathbf{y}\|_2$ 的 bound 代入上式, 即可得到 $F_\mu$ 的 Hölder smoothness 和 Lipschitz smoothness.

  $\square$


### 4.6 Pointwise Smoothing Error for $F_\mu(\mathbf{x})$

***Proposition* (Pointwise Smoothing Error)** 对于任意 $\mathbf{x} \in \mathbb{R}^n$, 有:
$$
0 \leq F(\mathbf{x}) - F_\mu(\mathbf{x}) \leq \frac{\mu}{2} m^{\frac{2-p}{p}} \|\mathbf{A}\mathbf{x}\|_2^{2p-2}
$$

或等价地, 根据 $F(\mathbf{x}) = \frac{1}{p}\|\mathbf{A}\mathbf{x}\|_p^p$, 上式也可以写作:
$$
0 \leq F(\mathbf{x}) - F_\mu(\mathbf{x}) \leq D_{p,m} \cdot \mu \cdot F(\mathbf{x})^{\frac{2(p-1)}{p}}, \quad \text{where } D_{p,m} := \frac{1}{2} m^{\frac{2-p}{p}} p^{\frac{2(p-1)}{p}}
$$

$\diamond$

- *Proof*
  - 首先证明左侧不等式. 
    - 根据定义, 有:
      $$
      F_\mu(\mathbf{x}) = \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q - \frac{\mu}{2}\|\mathbf{y}\|_2^2\right) \leq \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q\right) = F(\mathbf{x})
      $$

    - 因此 $F(\mathbf{x}) - F_\mu(\mathbf{x}) \geq 0$.

  - 接下来证明右侧不等式. 
    - 对于任意给定 $\mathbf{x}$, 回顾原问题 $F(\mathbf{x}) = \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q\right)$, 其唯一最大值解为 $\mathbf{y}_F^\star(\mathbf{x}) = \text{sign}(\mathbf{A}\mathbf{x}) \odot |\mathbf{A}\mathbf{x}|^{p-1}$. 因此, 上式等价于 $F(\mathbf{x}) = \langle \mathbf{y}_F^\star(\mathbf{x}), \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}_F^\star(\mathbf{x})\|_q^q$.

    - 因此, 将 $\mathbf{y}_F^\star(\mathbf{x})$ 代入 $F_\mu(\mathbf{x}) = \max_{\mathbf{y} \in \mathbb{R}^m} \left(\langle \mathbf{y}, \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}\|_q^q - \frac{\mu}{2}\|\mathbf{y}\|_2^2\right)$ 的 RHS 中, 即可得到:
      $$
      \begin{aligned}
      F_\mu(\mathbf{x}) &\geq \underbrace{\langle \mathbf{y}_{F}^\star(\mathbf{x}), \mathbf{A}\mathbf{x} \rangle - \frac{1}{q}\|\mathbf{y}_{F}^\star(\mathbf{x})\|_q^q}_{F(\mathbf{x})} - \frac{\mu}{2}\|\mathbf{y}_{F}^\star(\mathbf{x})\|_2^2 \\
      &= F(\mathbf{x}) - \frac{\mu}{2}\|\mathbf{y}_{F}^\star(\mathbf{x})\|_2^2 \\
      \end{aligned}
      $$
      故
      $$
      F(\mathbf{x}) - F_\mu(\mathbf{x}) \leq \frac{\mu}{2}\|\mathbf{y}_{F}^\star(\mathbf{x})\|_2^2
      $$

    - 根据 $\ell_r$ 和 $\ell_p$ 的范数关系, 且根据 $\mathbf{y}_F^\star(\mathbf{x})=\operatorname{sign}(\mathbf{A}\mathbf{x})\odot|\mathbf{A}\mathbf{x}|^{p-1}.$, 有:
      $$
      \begin{aligned}
      \|\mathbf{y}_F^\star(\mathbf{x})\|_2^2  =  m^{\frac{2-p}{p}}\|\mathbf{A}\mathbf{x}\|_p^{2p-2},
      \end{aligned}
      $$
      故
      $$
      F(\mathbf{x}) - F_\mu(\mathbf{x}) \leq \frac{\mu}{2} m^{\frac{2-p}{p}} \|\mathbf{A}\mathbf{x}\|_p^{2p-2} = D_{p,m} \cdot \mu \cdot F(\mathbf{x})^{\frac{2(p-1)}{p}}
      $$
      最后一步是因为 $F(\mathbf{x}) = \frac{1}{p}\|\mathbf{A}\mathbf{x}\|_p^p$, 从而 $\|\mathbf{A}\mathbf{x}\|_p^{2p-2} = p^{\frac{2(p-1)}{p}} \cdot F(\mathbf{x})^{\frac{2(p-1)}{p}}$.

  $\square$


### 4.7 Selection of $\mu$ and Overall Complexity

由上一小节的结果:
$$
0 \leq F(\mathbf{x}) - F_\mu(\mathbf{x}) \leq D_{p,m} \cdot \mu \cdot F(\mathbf{x})^{\frac{2(p-1)}{p}}, \quad \text{where } D_{p,m} := \frac{1}{2} m^{\frac{2-p}{p}} p^{\frac{2(p-1)}{p}}
$$
我们可以进一步讨论对于误差 $\varepsilon>0$, 关于原问题 $F(\mathbf{x}) \leq \varepsilon$ 时, $\mu$ 的选择.

***Proposition* (Accuracy Transfer)** 设 $\varepsilon > 0$. 若 $\mu$ 满足
$$
\mu \leq \frac{1}{2D_{p,m}}\cdot \varepsilon^{\frac{2}{p}-1} 
$$
则对任意 $\mathbf{x} \in \mathbb{R}^n$, 有
$$
F_\mu(\mathbf{x}) \leq \varepsilon/2 \implies F(\mathbf{x}) \leq \varepsilon
$$

- *Proof*
  - 记 $s := 2 - \dfrac{2}{p}>0 \in (0,1]$. 根据上面的讨论, $F(\mathbf{x}) - F_\mu(\mathbf{x}) \leq D_{p,m} \cdot \mu \cdot F(\mathbf{x})^{s}$. 
  - 下证明其逆否命题: 在 $\mu \leq \frac{1}{2D_{p,m}}\cdot \varepsilon^{\frac{2}{p}-1}$ 的前提下, 若 $F(\mathbf{x}) > \varepsilon$, 则 $F_\mu(\mathbf{x}) > \varepsilon/2$.
    - 设 $F(\mathbf{x}) > \varepsilon$, 则:
      $$
      \begin{aligned}
      F(\mathbf{x})^s &= F(\mathbf{x}) \cdot F^{s-1}(\mathbf{x}) \leq F(\mathbf{x}) \cdot \varepsilon^{s-1} \\
      \end{aligned}
      $$
    - 将 $F^s$ 代入 $F(\mathbf{x}) - F_\mu(\mathbf{x}) \leq D_{p,m} \cdot \mu \cdot F(\mathbf{x})^{s}$ 中, 得到:
      $$
      F(\mathbf{x}) - F_\mu(\mathbf{x}) \leq D_{p,m} \cdot \mu \cdot F(\mathbf{x})^{s} \leq D_{p,m} \cdot \mu \cdot F(\mathbf{x}) \cdot \varepsilon^{s-1} = 2 D_{p,m} \cdot \mu \cdot F(\mathbf{x}) \cdot \varepsilon^{\frac{2}{p}-2}
      $$
    - 此时 $\mu \leq \frac{1}{2D_{p,m}}\cdot \varepsilon^{\frac{2}{p}-1}$, 则 $F(\mathbf{x}) - F_\mu(\mathbf{x}) \leq F(\mathbf{x})/2$, 从而 $F_\mu(\mathbf{x}) \geq F(\mathbf{x})/2 > \varepsilon/2$. 
    - 综上, 我们有: 若 $F(\mathbf{x}) > \varepsilon$, 则 $F_\mu(\mathbf{x}) > \varepsilon/2$. 因此, 其逆否命题也成立: 若 $F_\mu(\mathbf{x}) < \varepsilon/2$, 则 $F(\mathbf{x}) < \varepsilon$.

  $\square$

该 Proposition 的意义在于, 只要我们能够选择恰当的 $\mu$ 将其优化至 $\varepsilon/2$ 的精度, 就可以保证原问题 $F(\mathbf{x})$ 的精度达到 $\varepsilon$.