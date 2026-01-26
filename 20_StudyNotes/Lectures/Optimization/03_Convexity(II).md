# Convexity (II): Optimization Basics

## Optimization Terminology

回顾, 一个凸优化问题具有如下形式:
$$\begin{aligned}
\min_{x\in D} \quad & f(x)\\
\text{s.t.}. \quad & g_i(x) \leq 0, \quad i=1, \ldots, m \\
& Ax = b
\end{aligned}$$
其中 $f, g_i$ 是凸函数, optimization domain $D = \bigcap_{i=1}^m \text{dom}(g_i) \bigcap \text{dom}(f)$ 是凸集.
- $f$ 称作 criterion 或 objective function.
- $g_i$ 称作 inequality constraint functions.
- 对于满足所有约束条件的 $x$ 称作 feasible point, 否则称作 infeasible point.
- 若存在 $x^*$ 是所有 feasible point 中使 $f(x)$ 取得最小值的点, 则称 $x^*$ 为 optimal point 或该问题的 solution, 最小值 $f(x^*)$ 称作 optimal value.
- 若 $x$ 是 feasible point, 且有 $f(x) \leq f^*(x)+\epsilon$, 则称 $x$ 为 $\epsilon$-suboptimal point.
- 若 $x$ 是 feasible point 且 $g_i(x) = 0$ 对某些 $i$ 成立, 则称该约束条件 $g_i$ 在 $x$ 处为 active constraint, 否则称为 inactive constraint.
- 一个凸的最小化问题可以等价于最大化 $-f(x)$ 的问题, 因此有时也称 $f$ 为 cost function.

下假设我们的凸优化问题定有解, 则记所有最优解组成的集合为 $X_{\text{opt}}$:
$$
X_{\text{opt}} = \{x^* \in \argmax f(x) \mid g_i(x) \leq 0, \forall i~; Ax=b\}
$$

关于最优解集, 有两条重要性质:
1. 由于 $f$ 和 $g_i$ 均为凸函数, 且约束条件均为凸集, 因此 $X_{\text{opt}}$ 也是凸集.
   - *Proof.*
     - 假设 $x_1, x_2 \in X_{\text{opt}}$, 则对任意 $\theta \in [0,1]$, 有:
       - $g_i(\theta x_1 + (1-\theta)x_2) \leq \theta g_i(x_1) + (1-\theta)g_i(x_2) \leq 0$, 因此 $\theta x_1 + (1-\theta)x_2$ 满足不等式约束条件. 
       - $A(\theta x_1 + (1-\theta)x_2) = \theta Ax_1 + (1-\theta)Ax_2 = \theta b + (1-\theta)b = b$, 因此 $\theta x_1 + (1-\theta)x_2$ 满足等式约束条件.
       - $f(\theta x_1 + (1-\theta)x_2) \leq \theta f(x_1) + (1-\theta)f(x_2) = f^*$, 因此 $\theta x_1 + (1-\theta)x_2$ 也是最优解.
     - 综上, $\theta x_1 + (1-\theta)x_2 \in X_{\text{opt}}$, 因此 $X_{\text{opt}}$ 是凸集.

2. 若 $f$ 严格凸, 则最优解若存在则唯一.
    - *Proof.*
      - 假设存在 $x_1, x_2 \in X_{\text{opt}}$, 且 $x_1 \neq x_2$, 则对任意 $\theta \in (0,1)$, 有:
         - $f(\theta x_1 + (1-\theta)x_2) < \theta f(x_1) + (1-\theta)f(x_2) = f^*$, 这与 $f^*$ 为最小值矛盾.
      - 因此, 最优解唯一.

***Example* (LASSO)** 给定 $\boldsymbol{y} \in \mathbb{R}^n$, $\boldsymbol{X} \in \mathbb{R}^{n \times p}$, LASSO 问题定义为:
$$\begin{aligned}
\min_{\boldsymbol{\beta} \in \mathbb{R}^p} \quad & \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\|_2^2 \\
\text{s.t.} \quad & \|\boldsymbol{\beta}\|_1 \leq \lambda
\end{aligned}$$

- 若 $n\ge p$ 且 $\boldsymbol{{X}}$ 是列满秩的, 则该问题的解是唯一的. 
  - *Proof.* 考虑 $\nabla^2 \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\|_2^2 = 2\boldsymbol{X}^\top\boldsymbol{X}$, 由于 $\boldsymbol{X}$ 是列满秩的, 因此 $\boldsymbol{X}^\top\boldsymbol{X}$ 是正定矩阵, 故 $\|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\|_2^2$ 是严格凸函数, 因此该问题的解唯一.
- 若 $n < p$ 的高维场景, 则该问题可能存在多个解.

***Example* (SVM)** 考虑如下支持向量机(SVM)的优化问题. 对于给定的训练数据集 $\{(\boldsymbol{x}_i, y_i)\}_{i=1}^n$, 其中 $\boldsymbol{x}_i \in \mathbb{R}^p$ 是样本特征, $y_i \in \{-1, 1\}$ 是样本标签, SVM 的优化问题定义为 (其中 $\boldsymbol{\xi} = (\xi_1, \ldots, \xi_n)^\top$ 是松弛变量, 允许一定程度的分类错误):
$$\begin{aligned}
\min_{\boldsymbol{\beta} \in \mathbb{R}^p,\beta_0 \in \mathbb{R}, \boldsymbol{\xi}\in \mathbb{R}^n} \quad & \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 + C \sum_{i=1}^n \xi_i \\
\text{s.t.} \quad \quad& y_i (\boldsymbol{x}_i^\top \boldsymbol{\beta} + \beta_0) \geq 1 - \xi_i,  \quad i=1, \ldots, n\\
& \xi_i \geq 0, \quad i=1, \ldots, n
\end{aligned}$$