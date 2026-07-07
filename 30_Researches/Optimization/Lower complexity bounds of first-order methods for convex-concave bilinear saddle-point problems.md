# Lower complexity bounds of first-order methods for convex-concave bilinear saddle-point problems

## Introduction

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