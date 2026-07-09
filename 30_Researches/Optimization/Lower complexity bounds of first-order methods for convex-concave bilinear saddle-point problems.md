# Lower complexity bounds of first-order methods for convex-concave bilinear saddle-point problems

## Introduction

### Intuition 

本文从理论上尝试分析对于大规模 convex-concave bilinear saddle-point problems, 任何 deterministic first-order method 的算法收敛下界. 

回顾基本的一阶算法及其收敛性质. 
- 对于 Projected Gradient Method, 其更新为:
    $$
    \mathbf{x}^{t+1} = \Pi_{\mathcal{X}}(\mathbf{x}^t - \alpha \nabla f(\mathbf{x}^t)),
    $$
    其中 $\Pi_{\mathcal{X}}$ 是投影算子, $\alpha$ 是步长.
- 若 $f$ 是 convex, 且 $L_f$-smooth, 则通过合适的步长选择, Projected Gradient Method 可以达到收敛率
    $$
    f(\mathbf{x}^t) - f(\mathbf{x}^*) = \mathcal{O}\left(\frac{1}{t}\right).
    $$
- 进一步, 若使用 Nesterov's accelerated gradient method, 则可以达到更快的收敛率
    $$
    f(\mathbf{x}^t) - f(\mathbf{x}^*) = \mathcal{O}\left(\frac{1}{t^2}\right).
    $$
    并且已有证明说明, 该收敛率就是最优的.

本文则将考虑如下 bilinear saddle-point problem (SPP):
$$
\min_{\mathbf{x} \in \mathcal{X}} \max_{\mathbf{y} \in \mathcal{Y}}
\mathcal{L}(\mathbf{x}, \mathbf{y}) 
= 
f(\mathbf{x}) + \langle \mathbf{A}\mathbf{x} - \mathbf{b}, \mathbf{y} \rangle - g(\mathbf{y}),
$$
- 其中 $\mathcal{X} \subseteq \mathbb{R}^{n}$ 和 $\mathcal{Y} \subseteq \mathbb{R}^{m}$ 是 closed and convex, $f: \mathbb{R}^n \to \mathbb{R}$ 和 $g: \mathbb{R}^m \to \mathbb{R}$ 是 closed and convex, $\mathbf{A} \in \mathbb{R}^{m\times n}$, $\mathbf{b} \in \mathbb{R}^{m}$. 
- 假设 $f$ 是 $L_f$-smooth:
    $$
    \|\nabla f(\mathbf{x_1}) - \nabla f(\mathbf{x}_2)\| \leq L_f \|\mathbf{x_1} - \mathbf{x}_2\|, \quad \forall \mathbf{x}_1, \mathbf{x}_2 \in \mathcal{X}.
    $$
- 假设 $g$ 是一个较为简单的函数, 其 proximal operator 可以被高效计算.

对于该问题, 有 primal 和 dual 两重视角:
- Primal: 即先对内层 $\mathbf{y}$ 进行最大化, 再对外层 $\mathbf{x}$ 进行最小化, 得到 primal problem:
    $$
    \phi^* := \min_{\mathbf{x} \in \mathcal{X}} \left\{
        \phi(\mathbf{x}) := f(\mathbf{x}) + \max_{\mathbf{y} \in \mathcal{Y}} [\langle \mathbf{A}\mathbf{x} - \mathbf{b}, \mathbf{y} \rangle - g(\mathbf{y})]
        \right\}
    $$

- Dual: 即先对外层 $\mathbf{x}$ 进行最小化, 再对内层 $\mathbf{y}$ 进行最大化, 得到 dual problem:
    $$
    \psi^* := \max_{\mathbf{y} \in \mathcal{Y}} \left\{
        \psi(\mathbf{y}) := -g(\mathbf{y}) + \min_{\mathbf{x} \in \mathcal{X}} [\langle \mathbf{A}\mathbf{x} - \mathbf{b}, \mathbf{y} \rangle + f(\mathbf{x})]
        \right\}
    $$

- 不难得到, 弱对偶性是一直成立的, 即
    $$
    \psi^* \leq \phi^*.
    $$
    因为对任意固定的 $\mathbf{x} \in \mathcal{X}$ 与 $\mathbf{y} \in \mathcal{Y}$, 有
    $$
    \psi = 
    \min_{\mathbf{x}' \in \mathcal{X}} \mathcal{L}(\mathbf{x}', \mathbf{y}) \leq \mathcal{L}(\mathbf{x}, \mathbf{y}) \leq \max_{\mathbf{y}' \in \mathcal{Y}} \mathcal{L}(\mathbf{x}, \mathbf{y}') = \phi.
    $$
    故各自的最优值满足 $\psi^* \leq \phi^*$.


- 在一些温和条件, 如 $\mathcal{X}$ 和 $\mathcal{Y}$ 是 compact, 强对偶性也可以得到, 即 $\psi^* = \phi^*$. 此时有 saddle point $(\mathbf{x}^*, \mathbf{y}^*)$ 满足
    $$
    \mathcal{L}(\mathbf{x}^*, \mathbf{y}) \leq \mathcal{L}(\mathbf{x}^*, \mathbf{y}^*) \leq \mathcal{L}(\mathbf{x}, \mathbf{y}^*), \quad \forall \mathbf{x} \in \mathcal{X}, \forall \mathbf{y} \in \mathcal{Y}.
    $$

事实上, 许多的优化问题都可以整理为 SPP, 例如 affinely constrained smooth convex optimization:
$$
f^* := \min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x}) \quad \text{s.t. } \mathbf{A}\mathbf{x} = \mathbf{b},
$$
- 下说明, 当前这个例子就等价于将标准形式中 $\mathcal{Y} = \mathbb{R}^m$, $g \equiv 0$.  具体地, 此时标准形式为:
    $$
    \min_{\mathbf{x} \in \mathcal{X}}  f(\mathbf{x}) +  \max_{\mathbf{y} \in \mathbb{R}^m} \langle \mathbf{A}\mathbf{x} - \mathbf{b}, \mathbf{y} \rangle.
    $$
    因此对于后面的最大化问题, 当且仅当 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 时, 其最大值才是有限的且为零, 否则其最大值为 $+\infty$.  因此二者等价. 

### Main Goal

在本文的 deterministic first-order method分析中涉及到如下几个概念和问题. 

- Deterministic 表明算法不涉及随机性, 给定相同的初始点和参数, 算法每次迭代都会产生相同的结果.
- First-order method 可以理解为黑箱 Oracle 能够返回的信息仅限于函数值和梯度信息. 对于本文的 SPP, 其 Oracle 可以返回如下信息:
  $$
  O(\mathbf{x}, \mathbf{y}) := (\nabla f(\mathbf{x}), \mathbf{A}\mathbf{x}, \mathbf{A}^\top \mathbf{y}). 
  $$
  - 对 SPP 目标函数求关于 $\mathbf{x}$ 的梯度, 其结果为 $\nabla f(\mathbf{x}) + \mathbf{A}^\top \mathbf{y}$; 求关于 $\mathbf{y}$ 的梯度, 其结果为 $\mathbf{A}\mathbf{x} - \mathbf{b} - \nabla g(\mathbf{y})$. 当然其他的一些信息如 $\mathbf{b}$ 是已知的; 和 $g$ 有关的信息, 文中认为其是简单的, 因此同样不作为 Oracle 的返回信息.
  - 具体的黑箱计算过程如下:
    - 算法的迭代初始点为 $(\mathbf{x}^{(0)}, \mathbf{y}^{(0)}) \in \mathcal{X} \times \mathcal{Y}$, 迭代次数为 $t = 0, 1, 2, \ldots$.
    - 在第 $t$ 次迭代中, 算法会在当前迭代点的 inquiry point $(\mathbf{x}^{(t)}, \mathbf{y}^{(t)})$ 处调用 Oracle, 得到
      $$
      O(\mathbf{x}^{(t)}, \mathbf{y}^{(t)}) = (\nabla f(\mathbf{x}^{(t)}), \mathbf{A}\mathbf{x}^{(t)}, \mathbf{A}^\top \mathbf{y}^{(t)}).
      $$
    - 然后算法根据当前迭代点和 Oracle 返回的信息, 计算出下一次迭代的 inquiry point $(\mathbf{x}^{(t+1)}, \mathbf{y}^{(t+1)})$ 和最终用来返回作为输出的点 $(\bar{\mathbf{x}}^{(t+1)}, \bar{\mathbf{y}}^{(t+1)})$. 迭代过程可以表示为:
      $$
      (\mathbf{x}^{(t+1)}, \mathbf{y}^{(t+1)}, \bar{\mathbf{x}}^{(t+1)}, \bar{\mathbf{y}}^{(t+1)}) = \mathcal{I}_t\left( \boldsymbol{\vartheta}; O(\mathbf{x}^{(0)}, \mathbf{y}^{(0)}), \ldots, O(\mathbf{x}^{(t)}, \mathbf{y}^{(t)})\right),
      $$
      - 这里之所以区分 $(\mathbf{x}^{(t+1)}, \mathbf{y}^{(t+1)})$ 和 $(\bar{\mathbf{x}}^{(t+1)}, \bar{\mathbf{y}}^{(t+1)})$, 是因为有些算法 (例如 Nesterov's accelerated gradient method), 用来查询梯度信息和最终返回作为决策变量的点是不同的. 因此这样表达可以更为一般化.
      - $\boldsymbol{\vartheta}$ 是本身问题包含的所有静态信息, 例如 $\mathbf{A}$, $\mathbf{b}$, $L_f$, $\mathcal{X}$, $\mathcal{Y}$ 等等. 这些信息独立于迭代之外, 是随着问题的定义而固定的. 