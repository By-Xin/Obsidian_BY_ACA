# Coordinate Descent

>[!quote]
>
> - Lecture Reference: <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>
> - Reading Reference: 《最优化：建模、算法与理论》第 8.4 小节

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

## (Block) Coordinate Descent Algorithm

给定优化问题
$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})
$$
其中要求其是可分解的, 即
$$
f(\mathbf{x}) = g(\mathbf{x}) + \sum_{i=1}^n h_i(x_i)
$$
其中 $g$ 是一个凸且可微的函数 (事实上在《最优化：建模、算法与理论》第 8.4 小节中, 可以不要求 $g$ 是凸的), $h_i$ 是一个关于 $x_i$ 的一维凸函数 (不要求可微).

在更新过程中, 记 $k$ 为总体的迭代次数, 定义辅助函数
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

并且事实上, 上述的每一个分量 $x_i$ 也可以是一个 block, 即一个向量, 而不一定是一个标量. 这样就得到了 Block Coordinate Descent Algorithm.