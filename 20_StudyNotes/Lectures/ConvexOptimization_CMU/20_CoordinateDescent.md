# Coordinate Descent

>[!quote]
>
> - Lecture Reference: 
>   - <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>
>   - https://statr.me/teaching/compstat
> - Reading Reference: 《最优化：建模、算法与理论》第 8.4 小节

## Introduction

对于一个一般的非光滑问题, 直接进行全局的优化可能比较困难. 不过在一些具体的情况, 例如一元的场景下, 或许会有一些操作的空间.

***Example* (Lasso Regression)**: 给定数据 $\mathbf{X} \in \mathbb{R}^{n \times p}$ 和 $\mathbf{y} \in \mathbb{R}^n$, Lasso Regression 的优化问题为
$$
\min_{\boldsymbol{\beta}} \ell({\boldsymbol{\beta}}) := \min_{\boldsymbol{\beta}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X} \boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1
$$
直接对于 $\boldsymbol{\beta}$ 进行全局优化比较困难. 不过如果我们固定 $\boldsymbol{\beta}$ 的其他分量, 只考虑其中一个分量 $\beta_j$, 则可以得到一个关于 $\beta_j$ 的一维优化问题, 并推导出 $\beta_j$ 的解析解. 推导如下. 

- 记 $\mathbf{X}_{i} \in \mathbb{R}^n$ 为 $\mathbf{X}$ 的第 $i$ 列, $\mathbf{X}_{-j} \in \mathbb{R}^{n \times (p-1)}$ 为 $\mathbf{X}$ 除了第 $j$ 列后的剩余部分, $\boldsymbol{\beta}_{-j} \in \mathbb{R}^{p-1}$ 为 $\boldsymbol{\beta}$ 除了第 $j$ 个分量后的剩余部分. 
- 则根据矩阵分块即 permutation 的性质, 有
  $$
  \mathbf{X} \boldsymbol{\beta} = \begin{bmatrix} \mathbf{X}_{-j} & \mathbf{X}_j \end{bmatrix} \begin{bmatrix} \boldsymbol{\beta}_{-j} \\ \beta_j \end{bmatrix} = \mathbf{X}_{-j} \boldsymbol{\beta}_{-j} + \beta_j \mathbf{X}_j
  $$
  因此
  $$
  \begin{aligned}
    \ell(\boldsymbol{\beta}) &= \frac{1}{2n} \|\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j} - \beta_j \mathbf{X}_j\|^2 + \lambda |\beta_j| + \lambda \|\boldsymbol{\beta}_{-j}\|_1 \\
    &= \frac{1}{2n} \left(\|\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j}\|^2 - 2 \beta_j \mathbf{X}_j^\top (\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j}) + \beta_j^2 \|\mathbf{X}_j\|^2\right) + \lambda |\beta_j| + \lambda \|\boldsymbol{\beta}_{-j}\|_1 \\
    &= C + \frac{1}{2n} \left(\beta_j^2 \|\mathbf{X}_j\|^2 - 2 \beta_j \mathbf{X}_j^\top (\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j})\right) + \lambda |\beta_j|
  \end{aligned}
  $$
  其中 $C$ 是一个常数, 与 $\beta_j$ 无关. 
- 因此, 考虑 $\tilde{\ell}(\beta_j) \equiv \ell(\boldsymbol{\beta})$ 视作一个关于 $\beta_j$ 的一元函数, 则其 subgradient 为
  $$
  \partial \tilde{\ell}(\beta_j) = \frac{1}{n} \left(\beta_j \|\mathbf{X}_j\|^2 - \mathbf{X}_j^\top (\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j})\right) + \lambda \partial |\beta_j|
  $$  

- 故 $\beta_j$ 的最小值点满足 $0 \in \partial \tilde{\ell}(\beta_j)$. 通过分别讨论 $\beta_j > 0$, $\beta_j < 0$ 和 $\beta_j = 0$ 的情况, 可以得到
  $$
  \beta_j = \begin{cases}
  \frac{1}{\|\mathbf{X}_j\|^2} \left(\mathbf{X}_j^\top (\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j}) - n \lambda\right), & \text{if } \mathbf{X}_j^\top (\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j}) > n \lambda \\
  \frac{1}{\|\mathbf{X}_j\|^2} \left(\mathbf{X}_j^\top (\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j}) + n \lambda\right), & \text{if } \mathbf{X}_j^\top (\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j}) < -n \lambda \\
  0, & \text{if } |\mathbf{X}_j^\top (\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j})| \leq n \lambda
  \end{cases}
  $$
- 该解析解的形式为一个 soft-thresholding operator:
  $$
  \beta_j = \frac{\text{sign}(\mathbf{X}_j^\top (\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j})) \left(|\mathbf{X}_j^\top (\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j})| - n \lambda\right)_+ }{\|\mathbf{X}_j\|^2}  := \frac{S(\mathbf{X}_j^\top (\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j}), n \lambda)}{\|\mathbf{X}_j\|^2} 
  $$

- 综上, 通过固定 $\boldsymbol{\beta}$ 的其他分量, 只考虑其中一个分量 $\beta_j$, 就可以得到一个关于 $\beta_j$ 的一维优化问题, 每次更新 $\beta_j$ 的时候, 都相当于对第 $j$ 个分量 $\mathbf{X}_j$ 与其余分量组成的 partial residual $\mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j}$ 之内积进行 soft-thresholding 操作. 这就是 Coordinate Descent 在 Lasso Regression 中的一个应用.

## Coordinatewise Minima

对于优化问题
$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})
$$

之前讨论的绝大数做法都是每一步更新整个向量 $\mathbf{x}$. 而 Coordinate Descent 则是每一步只更新 $\mathbf{x}$ 的一个坐标分量:
$$
x_i \leftarrow \arg\min_{z} f(x_1, \ldots, x_{i-1}, z, x_{i+1}, \ldots, x_n)
$$

该方法收敛速度较慢, 但随着高维统计机器学习等领域的发展, 逐渐受到重视.


其中一个重要的问题是: 一个 Coordinatewise Minimum 是否能够保证是一个 Local 甚至 Global Minimum? 故首先给出 Coordinatewise Minimum 的定义.

***Definition* (Coordinatewise Minimum)**: $\mathbf{x}^* \in \mathbb{R}^n$ 是 $f$ 的一个 Coordinatewise Minimum, 如果对于任意 $i \in \{1, \ldots, n\}$ 和所有标准基向量 $\mathbf{e}_i = (0, \ldots, 0, 1, 0, \ldots, 0)^\top$ (第 $i$ 个位置为 1, 其他位置为 0), 都有
$$
f(\mathbf{x}^* + \delta \mathbf{e}_i) \geq f(\mathbf{x}^*), \quad \forall \delta \in \mathbb{R}, \forall i \in \{1, \ldots, n\}
$$
则 $\mathbf{x}^*$ 是 $f$ 的一个 Coordinatewise Minimum.
- 直观的讲, Coordinatewise Minimum 是指在每一个坐标方向上, 都无法通过改变该坐标的值来降低函数值.

Coordinatewise Minimum 与 Global Minimum 的关系如下:
- 若 $f$ 是*凸且可微*的, 则 Coordinatewise Minimum 就是 Global Minimum.
  - *Proof*. 定义 $\phi_i(\delta) = f(\mathbf{x}^* + \delta \mathbf{e}_i)$. 根据 coordinatewise minimum 的定义, $\phi_i(\delta) \geq \phi_i(0)$ 对于所有 $\delta$ 都成立, 故为最小值点. 又因为 $f$ 是可微的, $\phi_i'(\delta) = \partial f(\mathbf{x}^* + \delta \mathbf{e}_i) / \partial \delta = 0$ 对所有 $i$ 都成立. 因此 $\nabla f(\mathbf{x}^*) = 0$, 从而又根据 $f$ 的凸性, $\mathbf{x}^*$ 是 $f$ 的 Global Minimum.


- 若 $f$ 是*凸但不可微*的, 则 Coordinatewise Minimum 不一定是 Global Minimum.
  - 说明:
    - 考虑函数 $f$ 的 subgrad: $\partial f(\mathbf{x}) = \{\mathbf{g} \in \mathbb{R}^n: f(\mathbf{y}) \geq f(\mathbf{x}) + \mathbf{g}^\top (\mathbf{y} - \mathbf{x}), \forall \mathbf{y} \in \mathbb{R}^n\}$. 根据 Coordinatewise Minimum 的定义, 对于任意 $i$ 和 $\delta$, 都有 $\phi_i(\delta) := f(\mathbf{x}^* + \delta \mathbf{e}_i) \geq \phi_i(0) = f(\mathbf{x}^*)$. 因此, $0$ 是 $\phi_i$ 的最小值点, $0 \in \partial \phi_i(0)$. 但 $\partial \phi_i(0) = \{\mathbf{g}^\top \mathbf{e}_i: \mathbf{g} \in \partial f(\mathbf{x}^*)\}$ 只是 subgradient 在各个坐标上的投影, 因此存在 $\mathbf{g} \in \partial f(\mathbf{x}^*)$ 使得 $\mathbf{g}^\top \mathbf{e}_i = 0$ 对所有 $i$ 都成立. 但这并不意味着 $0 \in \partial f(\mathbf{x}^*)$, 从而无法保证 $\mathbf{x}^*$ 是 Global Minimum.

- 若 $f$ 是仍然是不可微的, 但是其能够分解为一个凸且可微的函数 $g: \mathbb{R}^n \to \mathbb{R}$ 和若干关于各个坐标分量的一维凸函数 (不要求可微) $h_i: \mathbb{R} \to \mathbb{R}, ~ i = 1, \ldots, n$ 之和, 即
  $$
  f(\mathbf{x}) = g(\mathbf{x}) + \sum_{i=1}^n h_i(x_i)
  $$
  则此时可以说明, Coordinatewise Minimum 就是 Global Minimum.
    - 证明. 定义一个一维函数 $\phi_i(t):=f(x_1^*, \ldots, x_{i-1}^*, t, x_{i+1}^*, \ldots, x_n^*)$ 表示只考虑第 $i$ 个坐标, 将其他坐标都固定在 $\mathbf{x}^*$ 上的函数. 根据 cordinatewise minimum 的定义, $\phi_i(t) \geq \phi_i(x_i^*)$ 对于所有 $t \in \mathbb{R}$ 都成立. 因此 $x_i^*$ 是 $\phi_i$ 的最小值点, $0 \in \partial \phi_i(x_i^*)$. 而又知 $\partial \phi_i(x_i^*) = \nabla g(\mathbf{x}^*)_i + \partial h_i(x_i^*)$, 因此 $0 \in \nabla g(\mathbf{x}^*)_i + \partial h_i(x_i^*)$ 对所有 $i$ 都成立, 故存在某个 $s_i \in \partial h_i(x_i^*)$ 使得 $\nabla g(\mathbf{x}^*)_i + s_i = 0$ 对所有 $i$ 都成立. 即可以拼成 $\mathbf{s} = [s_1, \ldots, s_n]^\top \in \mathbb{R}^n$ 使得 $\mathbf{s} \in \partial h(\mathbf{x}^*)$ 且 $\nabla g(\mathbf{x}^*) + \mathbf{s} = 0$.  因此， 上文的 $0 = \nabla g(\mathbf{x}^*) + \mathbf{s} \in \nabla g(\mathbf{x}^*) + \partial h(\mathbf{x}^*) = \partial f(\mathbf{x}^*)$, 从而 $\mathbf{x}^*$ 是 $f$ 的 Global Minimum.
    - 其整体意思即为, 对于可微分的部分, 我们可以要求其不可分离, 各分量是彼此耦合的, 这没有问题, 因为第一个性质保证了 Coordinatewise Minimum 就是 Global Minimum. 而对于不可微的部分, 我们要求其可分离, 各分量之间没有耦合, 这样各个分量的 subgradient 就是独立的, 从而能够保证 Coordinatewise Minimum 就是 Global Minimum.

## Coordinate Descent Algorithm

### Cyclic Coordinate Descent

给定优化问题
$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})
$$
其中要求其是可分解的, 即
$$
f(\mathbf{x}) = g(\mathbf{x}) + \sum_{i=1}^n h_i(x_i)
$$
其中 $g$ 是一个凸且可微的函数 (事实上可以不要求 $g$ 是凸的), $h_i$ 是一个关于 $x_i$ 的一维凸函数 (不要求可微).

总体的更新过程类似内外循环的形式, 外循环为迭代次数 $k = 1, 2, \ldots$, 一次外循环的迭代将依次完成对于 $x_1, x_2, \ldots, x_n$ 的更新. 内循环为具体对于每一个坐标分量 $x_i$ 的更新. 故对于第 $k$ 次外循环的第 $i$ 个内循环, 定义一个关于 $x_i$ 的一维辅助函数:
$$
g_i^{(k)}(x_i) := g(x_1^{(k)}, \ldots, x_{i-1}^{(k)}, x_i, x_{i+1}^{(k-1)}, \ldots, x_n^{(k-1)}).
$$
其含义为, 整体的迭代过程将依序更新 $x_1, x_2, \ldots, x_n$, 在更新 $x_i$ 时, 将之前已经更新的 $x_1^{(k)}, \ldots, x_{i-1}^{(k)}$ 和之后还未更新的 $x_{i+1}^{(k-1)}, \ldots, x_n^{(k-1)}$ 固定, 只考虑更新 $x_i$ 的函数 $g_i^{(k)}(x_i)$.

在每一次更新中, 通常根据具体情况不同, 会选择如下三种方式中的一种来更新 $x_i$:
$$
x_i^{(k)} = \arg\min_{x_i} g_i^{(k)}(x_i) + h_i(x_i)
$$
$$
x_i^{(k)} = \arg\min_{x_i} \{g_i^{(k)}(x_i) + \frac{L_i^{(k-1)}}{2} \|x_i - x_i^{(k-1)}\|^2 + h_i(x_i)\}
$$
$$
x_i^{(k)} = \arg\min_{x_i} \{\langle \nabla g_i^{(k)}(\hat{x}_i^{(k-1)}), x_i - \hat{x}_i^{(k-1)} \rangle + \frac{L_i^{(k-1)}}{2} \|x_i - \hat{x}_i^{(k-1)}\|^2 + h_i(x_i)\}
$$
其中 $L_i^{(k-1)} > 0$ 是常数, $\hat{x}_i^{(k-1)}$ 是一个外推点, 给定权重 $\omega_i^{(k-1)} \geq 0$, 可以定义为 
$$
\hat{x}_i^{(k-1)} = x_i^{(k-1)} + \omega_i^{(k-1)} (x_i^{(k-1)} - x_i^{(k-2)}).
$$
- 第一种方式为直接更新, 适用于 $g_i^{(k)}(x_i) + h_i(x_i)$ 的最小值点能够通过解析解或者高效的数值方法求解的情况.
- 第二种方式为 Proximal 更新
- 第三种方式为 Proximal 更新的基础上加入 Nesterov 加速.

### Variants of Coordinate Descent

Coordinate Descent 还有许多变种, 例如:
- Randomized CD / Permutation CD: 在第 $k$ 次外循环的第 $i$ 个内循环, 随机等权重抽样一个坐标 $j \in \{1, \ldots, n\}$ 来更新. 二者的区别可以近似理解成一个是有放回, 一个是无放回的抽样. 
- Block CD: 每次内循环中更新的不是一个坐标分量, 而是一个坐标块. 
- Coordinate Proximal Gradient Descent: 每次内循环中通过 Proximal 更新的方式来更新坐标分量.


## Convergence Property

这里我们仍然考虑
$$
f(\mathbf{x}) = g(\mathbf{x}) + \sum_{i=1}^n h_i(x_i)
$$
且 $g$ 是一个凸且可微的函数, $h_i$ 是一个关于 $x_i$ 的一维凸函数.

Paul Tseng [1] 证明, 只要服从上述结构, 并在一些 mild regularity 条件下, CD (包括 Cyclic CD, Randomized CD, Block CD 等) 可以保证 $\mathbf{x}^{(k)}$ 收敛到 $f$ 的一个全局最优解. 

> [1] Paul Tseng (2001). Convergence of a block coordinate descent method for nondifferentiable minimization. Journal of optimization theory and applications.

对于不同的 CD 的收敛性在不同假设下有不同的分析. 一个较为有代表性的结论如下. 假设 $g$ 是依坐标 Lipschitz 光滑的, 即对于任意 $i$ 和 $\mathbf{x}$, 都有
$$
\|\nabla_i g(\mathbf{x} + t \mathbf{e}_i) - \nabla_i g(\mathbf{x})\| \leq L_i |t|, \quad \forall t \in \mathbb{R}
$$
其中 $\nabla_i g(\mathbf{x})$ 表示 $g$ 关于 $x_i$ 的偏导数. 进一步定义在初始函数范围内, 所有点到最优集合 $\mathcal{X}^*$ 的加权距离上界为 $R_0$, 即
$$
R_0 := \sup_{\mathbf{y}} \sup_{\mathbf{x}^* \in \mathcal{X}^*} \left\{\left(\sum_{i=1}^n L_i (y_i - x_i^*)^2\right)^{1/2}: f(\mathbf{y}) \leq f(\mathbf{x}^{(0)})\right\}.
$$
则考虑 Randomized CD. 给定 confidence parameter $\rho \in (0, 1)$, 以及 accuracy lever $\varepsilon < \min \{R_0^2, f(\mathbf{x}^{(0)}) - f^*\}$, 则当迭代次数 $k$ 满足
$$
k \geq \frac{2 n R_0^2}{\varepsilon} \log \frac{f(\mathbf{x}^{(0)}) - f^*}{\rho \varepsilon}.
$$
则有 $f(\mathbf{x}^{(k)}) - f^* \leq \varepsilon$ 以概率至少 $1 - \rho$ 成立. 该结果表明, Randomized CD 在上述条件下具有以 $O(1/\varepsilon)$ 的速率收敛到全局最优解的保证. 

若进一步对 $g$ 增加 $m$-strongly convex 的假设, 并为方便起见假设所有分量的 Lipschitz 常数相同, 即 $L_i = L$ 对所有 $i$ 都成立, 则对于 Cyclic CD, 当迭代次数 $k$ 满足
$$
k \geq \frac{4Ln}{m} \log \frac{f(\mathbf{x}^{(0)}) - f^*}{\varepsilon\rho}
$$
则有 $f(\mathbf{x}^{(k)}) - f^* \leq \varepsilon$ 以概率至少 $1 - \rho$ 成立. 该结果表明, Cyclic CD 在上述条件下具有以 $O(\log(1/\varepsilon))$ 的速率收敛到全局最优解的保证.