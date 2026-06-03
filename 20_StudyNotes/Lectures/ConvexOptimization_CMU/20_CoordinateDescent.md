# Coordinate Descent

>[!quote]
>
> - Lecture Reference: <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>

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
  - 其本质问题在于, 当 $f$ 是可微函数时, 任意方向导数都是线性的. 然而当 $f$ 是不可微函数时该线性型不能被保证.