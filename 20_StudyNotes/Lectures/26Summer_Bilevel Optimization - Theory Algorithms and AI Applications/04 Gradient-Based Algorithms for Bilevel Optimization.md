# Gradient-Based Algorithms for Bilevel Optimization

## 1. Unconstrained Bilevel Problem and Failure of the Naive Method


考虑如下无约束 BP 问题:
$$
\begin{aligned}
\min_{\mathbf{x} \in \mathcal{X} \subseteq \mathbb{R}^n, \mathbf{y} \in \mathbb{R}^m} & \quad F(\mathbf{x}, \mathbf{y}) \\
\text{s.t.} & \quad \mathbf{y} \in S(\mathbf{x}) := \arg\min_{\mathbf{y}' \in \mathbb{R}^m} f(\mathbf{x}, \mathbf{y}').
\end{aligned}
$$
其中假设 $F: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}$ 和 $f: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}$ 都是连续可微的. 


一个朴素的想法是对于 BP 的上下层问题交替进行优化:
- Lower-level update: 对于固定的 $x_k$, 更新 $y_{k+1} = \arg\min_{y \in \mathbb{R}^m} f(x_k, y)$.
- Upper-level update: 对于固定的 $y_{k+1}$, 更新 $x_{k+1} = \arg\min_{x \in \mathcal{X} } F(x, y_{k+1})$.

然而可以构造许多反例证明该方法在一般情况下是失败的. 例如考虑如下 BP 问题:
$$
\begin{aligned}
\min_{\mathbf{x},\mathbf{y}} & \quad (\mathbf{x}-1)^2+\mathbf{y}^2 \\
\text{s.t.} & \quad \mathbf{y} \in \arg\min_{\mathbf{y}'} (\mathbf{x}-\mathbf{y}')^2.
\end{aligned}
$$
具体迭代过程略, 可以发现交替过程将收敛到点 $(1,1)$, 然而真实的最优解是 $(1/2, 1/2)$.

## 2. Implicit Gradient Method under Strong Convexity

### Strong Convexity Assumption

若额外假设下层问题 $\min_{\mathbf{y} \in \mathbb{R}^m} f(\mathbf{x}, \mathbf{y})$ 对于任意固定的 $\mathbf{x}$ 是 $\sigma$-强凸的, 则有性质:
1. **唯一性**: 对于任意固定的 $\mathbf{x}$, 下层问题解集 $S(\mathbf{x}) = \{\mathbf{y}^\star(\mathbf{x})\}$ 是单点集, 此时原问题完全等价于单层问题
    $$
    \min_{\mathbf{x} \in \mathcal{X}} \quad \{\Phi(\mathbf{x}) := F(\mathbf{x}, \mathbf{y}^\star(\mathbf{x}))\}.
    $$
   - 称 $\Phi(\mathbf{x})$ 为 BP 的 hyper-objective function, $\nabla \Phi(\mathbf{x})$ 为 hyper-gradient.

2. **可微性**: 由隐函数定理保证 $\mathbf{y}^\star(\mathbf{x})$ 是连续可微的. 
   - 根据 $\sigma$-强凸性, $\nabla^2_{\mathbf{y}\mathbf{y}} f(\mathbf{x}, \mathbf{y}) \succeq \sigma \mathbf{I}_m$, 等价于该 Hessian 矩阵 $\nabla^2_{\mathbf{y}\mathbf{y}} f(\mathbf{x}, \mathbf{y}) \in \mathbb{S}_{++}^m$ 的所有特征值都大于等于 $\sigma > 0$. 且有 $\|[\nabla^2_{\mathbf{y}\mathbf{y}} f(\mathbf{x}, \mathbf{y})]^{-1}\| \leq 1/\sigma$.
   - 由于 $f$ 是凸且可微, 因此 $\mathbf{y}^\star(\mathbf{x})$ 是下层最优解等价于 $(\mathbf{x}, \mathbf{y}^\star(\mathbf{x}))$ 满足 $\nabla_{\mathbf{y}} f(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})) = 0$, 换言之 $\mathbf{y}^\star(\mathbf{x})$ 是 $\nabla_{\mathbf{y}} f(\mathbf{x}, \mathbf{y}) = 0$ 的解. 由解的唯一性及隐函数定理, 最终有:
      $$
      \mathrm{D} \mathbf{y}^\star(\mathbf{x}) = -[\nabla^2_{\mathbf{y}\mathbf{y}} f(\mathbf{x}, \mathbf{y}^\star(\mathbf{x}))]^{-1} \nabla^2_{\mathbf{y}\mathbf{x}} f(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})).
      $$
      其中 $\mathrm{D} \mathbf{y}^\star(\mathbf{x}) \in \mathbb{R}^{m \times n}$ 是 $\mathbf{y}^\star(\mathbf{x})$ 对 $\mathbf{x}$ 的 Jacobian: $[\mathrm{D}\mathbf{y}^\star(\mathbf{x})]_{ij} = \frac{\partial [\mathbf{y}^\star(\mathbf{x})]_i}{\partial [\mathbf{x}]_j}$.

### Hyper-gradient of Single-level Objective Function and Double-loop Structure

因此, 可以用一阶方法对单层优化问题目标函数 $\Phi(\mathbf{x})$ 进行优化, 其梯度为:
$$
\begin{aligned}
\nabla \Phi(\mathbf{x}) &= \nabla_{\mathbf{x}} F(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})) + \mathrm{D} \mathbf{y}^\star(\mathbf{x})^\top \nabla_{\mathbf{y}} F(\mathbf{x}, \mathbf{y}^\star(\mathbf{x}))  
\end{aligned}
$$
- 注意到, 上述 hypgradient 中的第二项, $\mathrm{D} \mathbf{y}^\star(\mathbf{x})$ 即衡量了下层最优解 $\mathbf{y}^\star(\mathbf{x})$ 对上层变量 $\mathbf{x}$ 的敏感性, 也称为 implicit gradient. 
- 在 naive method 中, 该项被忽略了 (只考虑了 $\nabla_{\mathbf{x}} F(\mathbf{x}, \mathbf{y}^\star(\mathbf{x}))$), 因此导致了 naive method 的失败.

若再代入具体的 $\mathrm{D} \mathbf{y}^\star(\mathbf{x})$ 的表达式, 则有:
$$
\nabla \Phi(\mathbf{x}) 
= 
\nabla_{\mathbf{x}} F(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})) - \nabla^2_{\mathbf{y}\mathbf{x}} f(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})) [\nabla^2_{\mathbf{y}\mathbf{y}} f(\mathbf{x}, \mathbf{y}^\star(\mathbf{x}))]^{-1} \nabla_{\mathbf{y}} F(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})).
$$
然而下欲说明, 这里的每一项都是不好计算的:
- $\mathbf{y}^\star(\mathbf{x})$ 本身并没有统一的 closed-form 的表达式. 即使用梯度下降等算法进行迭代, 最终也只是用 $T$ 次迭代后的结果 $\mathbf{y}_T(\mathbf{x})$ 来近似 $\mathbf{y}^\star(\mathbf{x})$ (近似关系记为 $\mathbf{y}^\star(\mathbf{x})\leadsto \mathbf{y}_T$), 且引入近似误差:
  $$
  \|\mathbf{y}_T(\mathbf{x}) - \mathbf{y}^\star(\mathbf{x})\| 
  $$ 
- $\nabla^2_{\mathbf{y}\mathbf{y}} f \in \mathbb{R}^{m \times m}$ 是一个 $m \times m$ 的矩阵, 在机器学习中 $m$ 通常对应模型的参数, 可能非常大. 显然不能显式求逆, 而是尝试通过求解线性系统
  $$
  \mathbf{H} \mathbf{v}^\star = \mathbf{b}, \qquad {\small \text{where }}~ \mathbf{H} := \nabla^2_{\mathbf{y}\mathbf{y}} f(\mathbf{x}, \mathbf{y}^\star(\mathbf{x})), \quad \mathbf{b} := \nabla_{\mathbf{y}} F(\mathbf{x}, \mathbf{y}^\star(\mathbf{x}))
  $$
  来间接计算 $\mathbf{v}^\star = \mathbf{H}^{-1} \mathbf{b}$. 然而该线性系统本身的求解往往也是通过迭代方法近似计算的, 这里的迭代次数记为 $N$, 最终得到近似关系 $\mathbf{v}^\star = \mathbf{H}^{-1} \mathbf{b} \leadsto \mathbf{v}_N$.
  且引入近似误差:
  $$
      \|\mathbf{v}_N - \mathbf{v}^\star\|
  $$

最终得到近似的 hyper-gradient 及对应梯度迭代:
$$
\widehat{\nabla} \Phi(\mathbf{x})
= \nabla_{\mathbf{x}} F(\mathbf{x}, \mathbf{y}_T(\mathbf{x})) - \nabla^2_{\mathbf{y}\mathbf{x}} f(\mathbf{x}, \mathbf{y}_T(\mathbf{x})) \mathbf{v}_N \implies \mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \widehat{\nabla} \Phi(\mathbf{x}_k).
$$
- 而为了得到外层一次的梯度更新, 每次都需要有如下的内循环 (这显然是昂贵的). 称这样的结构为 **nested/double-loop**:
    - $T$ 次迭代来近似求解 $\mathbf{y}^\star(\mathbf{x})$;
    - $N$ 次迭代来近似求解 $\mathbf{v}
- 并且事实上这里得到的 hypergradient 的估计同时还是有偏的. 关于偏差的分析将在稍后给出. 

## 3. Single-loop Implicit Gradient Method

### From Double-loop to Single-loop

