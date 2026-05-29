# Quasi-Newton Methods

>[!quote]
>
> - Lecture Reference: <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>
> - Readings: 
>    - 刘浩洋, 最优化: 建模、算法与理论, Chapter 6.5

## 1. Secant Equation and Quasi-Newton Methods

牛顿法是一个高效的二阶优化方法. 然而在迭代过程中, 计算 Hessian 矩阵 $\nabla^2 f(\mathbf{x}^{k+1})$ 可能非常昂贵. Quasi-Newton 方法通过构建一个近似的 Hessian 矩阵来避免直接计算 Hessian, 从而提高了算法的效率.

回顾牛顿法. 对于一个二阶可微的目标函数 $f: \mathbb{R}^n \to \mathbb{R}$, 其梯度 $\nabla f(\mathbf{x})$ 在 $\mathbf{x}^{k+1}$ 处的泰勒展开为
$$
\nabla f(\mathbf{x}) = \nabla f(\mathbf{x}^{k+1}) + \nabla^2 f(\mathbf{x}^{k+1})(\mathbf{x} - \mathbf{x}^{k+1}) + \mathcal{O}(\|\mathbf{x} - \mathbf{x}^{k+1}\|^2).
$$

令 $\mathbf{x} = \mathbf{x}^k$, 并忽略高阶项, 可得
$$
\nabla f(\mathbf{x}^k) \approx \nabla f(\mathbf{x}^{k+1}) + \nabla^2 f(\mathbf{x}^{k+1})(\mathbf{x}^k - \mathbf{x}^{k+1}).
$$
因此, Hessian 矩阵 $\nabla^2 f(\mathbf{x}^{k+1})$ 可以近似为
$$
({\mathbf{x}^{k+1} - \mathbf{x}^k})\nabla^2 f(\mathbf{x}^{k+1}) \approx {\nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k)}
$$
- 这样的估计也是非常自然的, 我们通过一阶导数的割线的斜率来近似二阶的曲率. 

因此, 若我们能够构建一个满足上述近似关系的矩阵 $B^{k+1}$:
$$
\boxed{B^{k+1}(\mathbf{x}^{k+1} - \mathbf{x}^k) = \nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k)},
$$
则我们可以使用 $B^{k+1}$ 来代替 Hessian 矩阵 $\nabla^2 f(\mathbf{x}^{k+1})$ 进行优化, 从而避免直接计算 Hessian. 我们同时也称上述方程为 **secant equation** (割线方程).

有时更进一步, 由于牛顿法在计算时事实上是在使用 Hessian 的逆 ($\mathbf{x}^{k+1} = \mathbf{x}^k - [\nabla^2 f(\mathbf{x}^k)]^{-1} \nabla f(\mathbf{x}^k)$), 因此有时我们也直接构建一个满足以下关系的 inverse Hessian 的近似矩阵 $H^{k+1}$:
$$
H^{k+1}(\nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k)) = \mathbf{x}^{k+1} - \mathbf{x}^k.
$$
来直接进行更新:
$$
\mathbf{x}^{k+1} = \mathbf{x}^k - H^{k+1} \nabla f(\mathbf{x}^k).
$$
以 inverse Hessian 的近似 Quasi-Newton 方法在实践中往往更为常见, 不过往往基于 $B^{k+1}$ 的方法在理论分析中更为方便. 



> [!note] 割线方程的另一种解读视角
>
> 上述割线方程就是整个 Quasi-Newton 方法的核心. 其还可以从另一个角度完整的推导得到. 
> 
> 回顾牛顿法的核心思想, 我们是在迭代点 $\mathbf{x}^{k}$ 展开二阶近似原函数 $f(\mathbf{x})$:
> $$
>   f(\mathbf{x}) \approx m_k(\mathbf{x}):= f(\mathbf{x}^{k}) + \nabla f(\mathbf{x}^{k})^\top (\mathbf{x} - \mathbf{x}^{k}) + \frac{1}{2} (\mathbf{x} - \mathbf{x}^{k})^\top \nabla^2 f(\mathbf{x}^{k}) (\mathbf{x} - \mathbf{x}^{k})
> $$
> 因此, 自然地会要求这个近似的结果在 $\mathbf{x}^{k}$ 和 $\mathbf{x}^{k+1}$ 处的梯度与原函数 $f(\mathbf{x})$ 的梯度相匹:
> $$
>  \nabla m_k(\mathbf{x}^{k+1}) = \nabla f(\mathbf{x}^{k+1}), \quad \nabla m_k(\mathbf{x}^{k}) = \nabla f(\mathbf{x}^{k}).
> $$
>
> 求解 $m_k(\mathbf{x})$ 的梯度:
> $$
>  \nabla m_k(\mathbf{x}) = \nabla f(\mathbf{x}^{k}) + \nabla^2 f(\mathbf{x}^{k})(\mathbf{x} - \mathbf{x}^{k}).
> $$
>
> 因此, 不难发现 $\nabla m_k(\mathbf{x}^{k} ) \equiv \nabla f(\mathbf{x}^{k})$ 是天然满足的. 故而我们只需要保证 $\nabla m_k(\mathbf{x}^{k+1}) = \nabla f(\mathbf{x}^{k+1})$ 即可. 代入上式, 可得
> $$
> \nabla f(\mathbf{x}^{k}) + \nabla^2 f(\mathbf{x}^{k})(\mathbf{x}^{k+1} - \mathbf{x}^{k}) = \nabla f(\mathbf{x}^{k+1}),
> $$
> 故在 Quasi-Newton 方法中, 用 $B^{k}$ 来近似 $\nabla^2 f(\mathbf{x}^{k})$, 就自然地得到了割线方程.

另一方面, 除了割线方程之外, 另一个重要的设计原则是保持 Hessian 近似矩阵 $B^{k}$ (或 $H^{k}$) 的正定性.  理由有如下几个:
1. 本身 Hessian 矩阵 $\nabla^2 f(\mathbf{x})$ 在凸优化问题中是正定的, 因此我们希望近似矩阵 $B^{k}$ 也能保持这一性质, 从而更好地捕捉原函数的曲率信息.
2. 保持 $B^{k}$ 的正定性可以确保搜索方向 $\mathbf{d}^k = -(B^k)^{-1} \nabla f(\mathbf{x}^k)$ 是一个下降方向, 从而保证算法的收敛性. 
   - 这是因为, 由 Newton 方法的更新方向: $\mathbf{d}^k = -[\nabla^2 f(\mathbf{x}^k)]^{-1} \nabla f(\mathbf{x}^k)$ 可知, 为使得 $\mathbf{d}^k$ 是一个下降方向, 需要 $\langle \nabla f(\mathbf{x}^k), \mathbf{d}^k \rangle < 0$. 这就要求 $-\nabla f^\top(\mathbf{x}^k) [\nabla^2 f(\mathbf{x}^k)]^{-1} \nabla f(\mathbf{x}^k) < 0$. 这在 $\nabla^2 f(\mathbf{x}^k)$ 是正定的情况下是天然满足的. 
3. 保持 $B^{k}$ 的正定性还可以确保近似模型 $m_k(\mathbf{x})$ 是一个严格凸函数, 从而保证每次迭代都能找到一个唯一的最优解.

因此, 若综合考虑上述正定性和割线方程的要求, 可以立刻得到如下必要条件(曲率条件). 在 Quasi-Newton 迭代过程中, 我们需要该条件在每次迭代中都得到满足:
$$
(\mathbf{x}^{k+1} - \mathbf{x}^k)^\top (\nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k)) > 0.
$$
事实上, 在步长搜索中, 若使用 Wolfe 条件, 则可以保证上述曲率条件的满足.
 - 对 Wolfe 条件的曲率条件 $\nabla f(\mathbf{x}^k + \alpha_k \mathbf{d}^k)^\top \mathbf{d}^k \geq c_2 \nabla f(\mathbf{x}^k)^\top \mathbf{d}^k$  左右两侧同时减去 $\nabla f(\mathbf{x}^k)^\top \mathbf{d}^k$, 可得
    $$
    (\nabla f(\mathbf{x}^k + \alpha_k \mathbf{d}^k) - \nabla f(\mathbf{x}^k))^\top \mathbf{d}^k \geq (c_2 - 1) \nabla f(\mathbf{x}^k)^\top \mathbf{d}^k.
    $$
    观察 RHS 的符号, 由于 $c_2 \in (c_1, 1)$, 因此 $c_2 - 1 < 0$. 同时, 由于 $\mathbf{d}^k$ 是一个下降方向, 因此 $\nabla f(\mathbf{x}^k)^\top \mathbf{d}^k < 0$. 因此, RHS 的符号是正的. 因此自动得到了曲率条件的满足.

> [!note] Wolfe 条件
>
> 对于一个目标函数 $f(\mathbf{x})$, 给定在 $\mathbf{x}^k$ 处的搜索方向 $\mathbf{d}^k$, Wolfe 条件要求步长 $\alpha_k$ 满足以下两个条件:
> 1. **Armijo 条件**: $f(\mathbf{x}^k + \alpha_k \mathbf{d}^k) \leq f(\mathbf{x}^k) + c_1 \alpha_k \nabla f(\mathbf{x}^k)^\top \mathbf{d}^k$, 其中 $c_1 \in (0, 1)$ 是一个小的常数.
> 2. **曲率条件**: $\nabla f(\mathbf{x}^k + \alpha_k \mathbf{d}^k)^\top \mathbf{d}^k \geq c_2 \nabla f(\mathbf{x}^k)^\top \mathbf{d}^k$, 其中 $c_2 \in (c_1, 1)$ 是另一个常数.

总的而言, 一个一般的 Quasi-Newton 方法的迭代步骤可以总结如下:
- **INPUT**: 初始点 $\mathbf{x}^0 \in \mathbb{R}^n$, 初始 Hessian 近似 $B^0 \in \mathbb{R}^{n \times n}$ (或 $H^0 \in \mathbb{R}^{n \times n}$). 
- **FOR** $k = 0, 1, 2, \ldots$ **DO**
    1. 计算搜索方向 $\mathbf{d}^k = -(B^k)^{-1} \nabla f(\mathbf{x}^k)$ (或 $\mathbf{d}^k = -H^k \nabla f(\mathbf{x}^k)$).
    2. 进行线搜索以确定步长 $\alpha_k > 0$.
    3. 更新参数: $\mathbf{x}^{k+1} = \mathbf{x}^k + \alpha_k \mathbf{d}^k$.
    4. 更新 Hessian 近似 $B^{k+1}$ (或 $H^{k+1}$) 满足 secant equation.