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

## 2. Update Formulas: SR1, DFP and BFGS

在 Quasi-Newton 方法中, 关键的一步是如何更新 Hessian 近似矩阵 $B^k$ (或 $H^k$) 以满足 secant equation. 常见的更新公式包括 SR1、DFP 和 BFGS 等.

### SR1 更新公式

SR1 (Symmetric Rank-One) 更新公式的形式如下. 假设 $B^k$ 是第 $k$ 轮迭代的 Hessian 近似矩阵, 我们假设存在一个 rank-1 的更新方法得到 $B^{k+1}$ 以满足 secant equation:
$$
B^{k+1} = B^k + a \mathbf{u} \mathbf{u}^\top,
$$
其中 $a \in \mathbb{R}$ 是一个标量, $\mathbf{u} \in \mathbb{R}^n$ 是一个向量. 我们可以通过代入 secant equation ($B^{k+1}(\mathbf{x}^{k+1} - \mathbf{x}^k) = \nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k)$) 来通过待定系数法求解 $a$ 和 $\mathbf{u}$:
$$
(B^k + a \mathbf{u} \mathbf{u}^\top)(\mathbf{x}^{k+1} - \mathbf{x}^k) = \nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k),
$$
整理有:
$$
(a \mathbf{u}^\top (\mathbf{x}^{k+1} - \mathbf{x}^k)) \mathbf{u} = \nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k) - B^k (\mathbf{x}^{k+1} - \mathbf{x}^k).
$$
注意到这里 $a \mathbf{u}^\top (\mathbf{x}^{k+1} - \mathbf{x}^k)$ 是一个标量, 故 $\mathbf{u}$ 的方向应该与 $\nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k) - B^k (\mathbf{x}^{k+1} - \mathbf{x}^k)$ 的方向一致. 故不妨令 $\mathbf{u} = \nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k) - B^k (\mathbf{x}^{k+1} - \mathbf{x}^k)$. 则对应:
$$
a \mathbf{u}^\top (\mathbf{x}^{k+1} - \mathbf{x}^k) = 1 \implies a = \frac{1}{\mathbf{u}^\top (\mathbf{x}^{k+1} - \mathbf{x}^k)} = \frac{1}{(\nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k) - B^k (\mathbf{x}^{k+1} - \mathbf{x}^k))^\top (\mathbf{x}^{k+1} - \mathbf{x}^k)}.
$$

因此, SR1 更新公式可以总结为:
$$
B^{k+1} = B^k + \frac{(\nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k) - B^k (\mathbf{x}^{k+1} - \mathbf{x}^k)) (\nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k) - B^k (\mathbf{x}^{k+1} - \mathbf{x}^k))^\top}{(\nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k) - B^k (\mathbf{x}^{k+1} - \mathbf{x}^k))^\top (\mathbf{x}^{k+1} - \mathbf{x}^k)}.
$$

若记 $\mathbf{y}^k = \nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k)$ 和 $\mathbf{s}^k = \mathbf{x}^{k+1} - \mathbf{x}^k$, 则 SR1 更新公式可以简化为:
$$
B^{k+1} = B^k + \frac{(\mathbf{y}^k - B^k \mathbf{s}^k)(\mathbf{y}^k - B^k \mathbf{s}^k)^\top}{(\mathbf{y}^k - B^k \mathbf{s}^k)^\top \mathbf{s}^k}.
$$


同样的过程, 我们也可以得到 inverse Hessian 的 SR1 更新公式:
$$
H^{k+1} = H^k + \frac{(\mathbf{s}^k - H^k \mathbf{y}^k)(\mathbf{s}^k - H^k \mathbf{y}^k)^\top}{(\mathbf{s}^k - H^k \mathbf{y}^k)^\top \mathbf{y}^k}.
$$


SR1 的方法虽然简单, 但是发现其在迭代过程中，并不能保证 Hessian 近似矩阵 $B^k$ 的正定性, 因此在实践中并不常用.

### BFGS 更新公式

为了保证 Hessian 近似矩阵 $B^k$ 的正定性, BFGS (Broyden-Fletcher-Goldfarb-Shanno) 更新公式被提出. BFGS 更新公式采用 rank-2 的更新方式, 其形式如下:
$$
B^{k+1} = B^k + a \mathbf{u} \mathbf{u}^\top + b \mathbf{v} \mathbf{v}^\top,
$$

同样代入 secant equation, 整理后可以得到:
$$
(a \mathbf{u}^\top (\mathbf{x}^{k+1} - \mathbf{x}^k)) \mathbf{u} + (b \mathbf{v}^\top (\mathbf{x}^{k+1} - \mathbf{x}^k)) \mathbf{v} = \nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k) - B^k (\mathbf{x}^{k+1} - \mathbf{x}^k).
$$
用同样的记号可以简化为:
$$
(a \mathbf{u}^\top \mathbf{s}^k) \mathbf{u} + (b \mathbf{v}^\top \mathbf{s}^k) \mathbf{v} = \mathbf{y}^k - B^k \mathbf{s}^k.
$$

事实上, 在这个方程中, 具体待定系数的求解是欠定的. 一个较为直接的方法就是依序对应元素完全相等, 即:
$$
(a \mathbf{u}^\top \mathbf{s}^k) = 1, \quad \mathbf{u} = \mathbf{y}^k, \quad (b \mathbf{v}^\top \mathbf{s}^k) = -1, \quad \mathbf{v} = B^k \mathbf{s}^k.
$$

因此, BFGS 更新公式可以总结为:
$$
B^{k+1} = B^k + \frac{\mathbf{y}^k (\mathbf{y}^k)^\top}{(\mathbf{y}^k)^\top \mathbf{s}^k} - \frac{B^k \mathbf{s}^k (B^k \mathbf{s}^k)^\top}{\mathbf{s}^{k\top} B^k \mathbf{s}^k}.
$$

根据 Sherman-Morrison-Woodbury 公式, BFGS 的 inverse Hessian 的更新公式可以求出:
$$
H^{k+1} = \left(I - \frac{\mathbf{y}^k (\mathbf{s}^k)^\top}{(\mathbf{y}^k)^\top \mathbf{s}^k}\right)^\top H^k \left(I - \frac{\mathbf{y}^k (\mathbf{s}^k)^\top}{(\mathbf{y}^k)^\top \mathbf{s}^k}\right) + \frac{\mathbf{s}^k (\mathbf{s}^k)^\top}{(\mathbf{y}^k)^\top \mathbf{s}^k}.
$$
- 为使得 BFGS 曲率条件 $(\mathbf{y}^k)^\top \mathbf{s}^k > 0$ 得到满足, 在实践中, 当求出 $H^k$ 以确定搜索方向 $\mathbf{d}^k = -H^k \nabla f(\mathbf{x}^k)$ 后, 通常会配合 Wolfe 条件来进行线搜索, 从而保证曲率条件的满足.

上述的 $H_k$ 的更新公式还可以通过下面的视角进行理解. 事实上, 这样定义的 $H^{k}$ 是如下优化问题的解:
$$
\begin{aligned}
& \min_{H \in \mathbb{R}^{n \times n}}\quad &&\|H - H^k\|_W^2 \\
& \text{subject to.} \quad &&H = H^\top, \\
&&& H \mathbf{y}^k = \mathbf{s}^k.
\end{aligned}
$$
其中 $W$ 是任意满足割线方程 $\nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k) = W (\mathbf{x}^{k+1} - \mathbf{x}^k)$ 的矩阵. 通过这样的 W-norm, 相当于对 $H$ 进行了尺度归一化, 相当于用一个更符合 Hessian 曲率的距离度量来衡量 $H$ 和 $H^k$ 之间的距离. 通过额外地对于对称性, 以及 $H$ 的割线方程的约束, 我们解出的 $H^{k+1}$ 就是 BFGS 的更新公式. 


### DFP 更新公式

BFGS 公式是利用割线方程对 Hessian 近似矩阵 $B^k$ 进行 rank-2 更新的. 而 DFP (Davidon-Fletcher-Powell) 更新公式则是直接对 inverse Hessian 近似矩阵 $H^k$ 进行 rank-2 更新的. DFP 的更新公式如下:
$$
H^{k+1} = H^k + \frac{\mathbf{s}^k (\mathbf{s}^k)^\top}{(\mathbf{y}^k)^\top \mathbf{s}^k} - \frac{H^k \mathbf{y}^k (H^k \mathbf{y}^k)^\top}{\mathbf{y}^{k\top} H^k \mathbf{y}^k}.
$$

DFP 公式与 BFGS 分别呈现对偶关系. 但是在实践中, BFGS 的表现往往优于 DFP, 因此 BFGS 更为常用. 


## 3. Convergence of Quasi-Newton Methods

首先重申记号. 
- 设 $B^k \approx \nabla^2 f(\mathbf{x}^k)$ 是第 $k$ 轮迭代的 Hessian 近似矩阵. 对应的搜索方向为 $\mathbf{d}^k = -(B^k)^{-1} \nabla f(\mathbf{x}^k)$. 另记 $\mathbf{y}^k = \nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k)$ 和 $\mathbf{s}^k = \mathbf{x}^{k+1} - \mathbf{x}^k$. 因此 BFGS 的更新公式为:
   $$ 
   B^{k+1} = B^k + \frac{\mathbf{y}^k (\mathbf{y}^k)^\top}{(\mathbf{y}^k)^\top \mathbf{s}^k} - \frac{B^k \mathbf{s}^k (B^k \mathbf{s}^k)^\top}{\mathbf{s}^{k\top} B^k \mathbf{s}^k}.
   $$
- 在确定更新方向后, 通过线搜索来确定步长 $\alpha_k$ 以满足 Wolfe 条件:
  1. **Armijo 条件**: $f(\mathbf{x}^k + \alpha_k \mathbf{d}^k) \leq f(\mathbf{x}^k) + c_1 \alpha_k \nabla f(\mathbf{x}^k)^\top \mathbf{d}^k$, 其中 $c_1 \in (0, 1)$ 是一个小的常数.
  2. **曲率条件**: $\nabla f(\mathbf{x}^k + \alpha_k \mathbf{d}^k)^\top \mathbf{d}^k \geq c_2 \nabla f(\mathbf{x}^k)^\top \mathbf{d}^k$, 其中 $c_2 \in (c_1, 1)$ 是另一个常数.
- 算法以满足上述条件的 $\alpha_k$ 来更新参数: $\mathbf{x}^{k+1} = \mathbf{x}^k + \alpha_k \mathbf{d}^k$.

如下定理表明, 在一些合理的假设条件下, BFGS 方法是全局收敛的. 也就是说, 迭代点 $\mathbf{x}^k$ 不论其初始值如何, 都会收敛到一个全局最优解. 在给出定理之前, 首先需要一个重要的引理. 该引理适用于任意 Wolfe line search 下降方法. 

***Lemma* (Zoutendijk's Theorem)**: 设 $f: \mathbb{R}^n \to \mathbb{R}$ 是一阶连续可微, 有下界, 且梯度 Lipschitz 连续
$$
\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|, \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^n.
$$
设 $\{\mathbf{x}^k\}$ 是通过满足 Wolfe 条件的线搜索方法生成的迭代点序列, 考虑负梯度方向和搜索方向之间的夹角 $\theta_k$ 之余弦
$$
\cos \theta_k = \frac{-\nabla f(\mathbf{x}^k)^\top \mathbf{d}^k}{\|\nabla f(\mathbf{x}^k)\| \|\mathbf{d}^k\|},
$$
则定有:
$$
\sum_{k=0}^\infty \cos^2 \theta_k \cdot \|\nabla f(\mathbf{x}^k)\|^2 < +\infty.
$$
也称上述级数的收敛为 Zoutendijk's condition. 
- Zoutendijk's condition 的直观理解如下. 考虑 $\cos \theta_k \|\nabla f(\mathbf{x}^k)\|$, 其相当于负梯度在搜索方向上的投影强度, 相当于沿着该方向的下降率的大小. 根据 Wolfe 条件, 如果该量足够大, 则 Wolfe 条件定会使得该次迭代产生足够的下降. 然而又因为 $f$ 是有下界的, 因此可行的下降总量是有限的. 因此, Zoutendijk's condition 就是一个定量化的表达, 说明了沿着搜索方向的下降率的平方和是有限的
- 不过 Zoutendijk's condition 本身并不能直接说明 $\|\nabla f(\mathbf{x}^k)\| \to 0$, 因为其可能包含了两种情况: 1. $\|\nabla f(\mathbf{x}^k)\| \to 0$, 这个是我们希望的情况, 其表明算法收敛到了一个 stationary point; 2. $\cos \theta_k \to 0$, 这是不期望发生的, 其表明搜索方向与梯度夹角接近正交, 发生退化. 

***Theorem* (Global Convergence of BFGS)**: 给定一个二阶可微的目标函数 $f: \mathbb{R}^n \to \mathbb{R}$, 以及一个 level set $\mathcal{L} = \{\mathbf{x} \in \mathbb{R}^n: f(\mathbf{x}) \leq f(\mathbf{x}^0)\}$. 做如下假设:

1. $\mathcal{L}$ 是凸的. 
2. 初始矩阵 $B^0$ 是对称正定的.
3. 存在正常数 $m > 0$ 和 $M > 0$, 使得对于任意 $\mathbf{x} \in \mathcal{L}$, 都有
   $$
   m I \preceq \nabla^2 f(\mathbf{x}) \preceq M I,
   $$
   或等价地, 对于任意 $\mathbf{z} \in \mathbb{R}^n$, 都有
   $$
   m \|\mathbf{z}\|^2 \leq \mathbf{z}^\top \nabla^2 f(\mathbf{x}) \mathbf{z} \leq M \|\mathbf{z}\|^2.
   $$
则通过上述的 BFGS 更新公式和 Wolfe 条件的线搜索, 迭代点 $\mathbf{x}^k$ 将全局收敛到 $f$ 的一个最优解. 


*Analysis Sketch*: 
- 定理的假设主要想说明, 在算法的活动范围 $\mathcal{L}$ 内, 目标函数曲率有界, 既不会太平坦也不会过于陡峭. 并且由于 Wolfe line search 的设计, 可以保证函数值不断下降, 一直都在 $\mathcal{L}$ 内. 
- 根据 Zoutendijk's condition, 可知 $\sum_{k=0}^\infty \cos^2 \theta_k \cdot \|\nabla f(\mathbf{x}^k)\|^2 < +\infty$. 因此, 只要能够保证 $\cos \theta_k$ 不会退化到 0, 就可以得出 $\|\nabla f(\mathbf{x}^k)\| \to 0$, 从而得出 $\mathbf{x}^k$ 收敛到一个 stationary point. 
- 最后, 由于 $f$ 是凸的, 因此该 stationary point 就是一个全局最优解.

*Proof*:
- 首先定义 $m_k := \langle \mathbf{y}^k, \mathbf{s}^k \rangle / \|\mathbf{s}^k\|^2$ 和 $M_k := \|\mathbf{y}^k\|^2 / \langle \mathbf{y}^k, \mathbf{s}^k \rangle$. 由近似关系 $\mathbf{y}^k \approx \nabla^2 f(\mathbf{x}^k) \mathbf{s}^k$, 其同时求关于 $\mathbf{s}^k$ 的内积, 有: $\langle \mathbf{y}^k, \mathbf{s}^k \rangle \approx \mathbf{s}^{k\top} \nabla^2 f(\mathbf{x}^k) \mathbf{s}^k$. 故 $\nabla^2 f(\mathbf{x}^k) \approx (\langle \mathbf{y}^k, \mathbf{s}^k \rangle / \|\mathbf{s}^k\|^2) I = m_k I$. 

## 4. Limited-memory BFGS (L-BFGS)

BFGS 方法虽然克服了牛顿法中 Hessian 计算的昂贵问题, 但是其在每次迭代中都需要存储和更新一个 $B^k \in \mathbb{R}^{n \times n}$ 的矩阵 (或 $H^k$), 这在高维问题中可能会导致巨大的内存开销. 为了解决这个问题, Limited-memory BFGS (L-BFGS) 方法被提出. L-BFGS 通过迭代展开的方式, 用一个较小的历史信息来近似 Hessian 矩阵, 从而大幅降低了内存需求.

首先为推导方便, 这里采用 $H^k$ 来近似 Hessian 的逆. 并且将迭代公式整理如下:
$$
H^{k+1} = (V^k)^\top H^k V^k + \rho_k \mathbf{s}^k (\mathbf{s}^k)^\top,
$$
其中 
$$
\rho_k = \frac{1}{(\mathbf{y}^k)^\top \mathbf{s}^k}, \quad V^k = I - \rho_k \mathbf{y}^k (\mathbf{s}^k)^\top, \quad \mathbf{y}^k = \nabla f(\mathbf{x}^{k+1}) - \nabla f(\mathbf{x}^k), \quad \mathbf{s}^k = \mathbf{x}^{k+1} - \mathbf{x}^k.
$$

观察到, 上述的 $H^{k+1}$ 的计算需要 $H^k$ 呈现递归的关系. 因此, 对该公式进行递归展开 $m$ 次, 可以得到如下的展开式 (可以直接通过迭代展开来验证):
$$
\begin{aligned}
H^{k} &= (V^{k-1})^\top H^{k-1} V^{k-1} + \rho_{k-1} \mathbf{s}^{k-1} (\mathbf{s}^{k-1})^\top \\
& = \qquad \cdots \\
&= (V^{k-1})^\top (V^{k-2})^\top \cdots (V^{k-m})^\top H^{k-m} V^{k-m} \cdots V^{k-2} V^{k-1} +\\
&\quad  \rho_{k-m} \left(V^{k-m+1}V^{k-m+2} \cdots V^{k-1}\right)^\top \mathbf{s}^{k-m} (\mathbf{s}^{k-m})^\top \left(V^{k-m+1}V^{k-m+2} \cdots V^{k-1}\right) +\\
&\quad  \rho_{k-m+1} \left(V^{k-m+2}V^{k-m+3} \cdots V^{k-1}\right)^\top \mathbf{s}^{k-m+1} (\mathbf{s}^{k-m+1})^\top \left(V^{k-m+2}V^{k-m+3} \cdots V^{k-1}\right) +\\
&\quad \cdots + \\
&\quad  \rho_{k-1} \mathbf{s}^{k-1} (\mathbf{s}^{k-1})^\top.
\end{aligned}
$$

更进一步, 在实际计算中, 我们事实上也并不是直接显式地计算 $H^k$ 的矩阵形式, 而是要通过 $H^k$ 来计算搜索方向 $\mathbf{d}^k = -H^k \nabla f(\mathbf{x}^k)$. 因此, 在上展开式左右两侧再同时乘以 $-\nabla f(\mathbf{x}^k)$ (为方便起见, 进一步简化 $\mathbf{g}^k := \nabla f(\mathbf{x}^k)$), 可以得到如下的计算搜索方向的公式:
$$
\begin{aligned}
H^k \mathbf{g}^k &= (V^{k-1})^\top (V^{k-2})^\top \cdots (V^{k-m})^\top H^{k-m} \underline{V^{k-m} \cdots V^{k-2} V^{k-1} \mathbf{g}^k} +\\
&\quad  \rho_{k-m} \left(V^{k-m+1}V^{k-m+2} \cdots V^{k-1}\right)^\top \mathbf{s}^{k-m} (\mathbf{s}^{k-m})^\top \underline{\left(V^{k-m+1}V^{k-m+2} \cdots V^{k-1}\right) \mathbf{g}^k} +\\
&\quad  \rho_{k-m+1} \left(V^{k-m+2}V^{k-m+3} \cdots V^{k-1}\right)^\top \mathbf{s}^{k-m+1} (\mathbf{s}^{k-m+1})^\top \underline{\left(V^{k-m+2}V^{k-m+3} \cdots V^{k-1}\right) \mathbf{g}^k} +\\
&\quad \cdots + \\
&\quad  \rho_{k-1} \mathbf{s}^{k-1} (\mathbf{s}^{k-1})^\top \mathbf{g}^k.
\end{aligned}
$$

这时便可以观察到较强的规律性了.  上式每一个加法项中, 不妨从右向左进行乘法计算. 此时, 都要依次计算 $V^{k-1} \mathbf{g}^k,$ $V^{k-2} (V^{k-1} \mathbf{g}^k)$, 以此类推 (见下划线部分). 并且注意到, 该系列不断左乘 $V^{j}$ 的矩阵乘法一直是 $\mathbb{R}^{n \times n} \times \mathbb{R}^n \to \mathbb{R}^n$ 的形式, 形状保持不变. 因此, 若抽象地概括, 令初始值 $\mathbf{q}_k := \mathbf{g}^k$, 则上述一系列乘法可以通过如下的递归关系来计算:
$$
\begin{aligned}
\mathbf{q}_k &:= \mathbf{g}^k, \\
\mathbf{q}_{j} &= V^{j} \mathbf{q}_{j+1} , \quad j = k-1, k-2, \cdots, k-m, \\
\end{aligned}
$$

下进行具体地代数展开. 

- 首先进行第一轮迭代. 考虑 $V^{k-1} \mathbf{g}^k$ 的计算, 根据定义, $V^{k-1} = I - \rho_{k-1} \mathbf{y}^{k-1} (\mathbf{s}^{k-1})^\top$, 因此
   $$
   \begin{aligned}
   \mathbf{q}_{k-1} &= V^{k-1} \mathbf{g}^k \\&= I\mathbf{g}^k - \rho_{k-1} \mathbf{y}^{k-1} (\mathbf{s}^{k-1})^\top \mathbf{g}^k \\
   &= \mathbf{g}^k - \underbrace{\rho_{k-1}\langle \mathbf{s}^{k-1}, \mathbf{g}^k \rangle}_{\alpha_{k-1}} \cdot \mathbf{y}^{k-1} \\&:= \mathbf{g}^k - \alpha_{k-1} \mathbf{y}^{k-1}.
   \end{aligned}
   $$
   - 故实际在第一轮迭代中:
     - 首先计算 $\alpha_{k-1} = \rho_{k-1}\langle \mathbf{s}^{k-1}, \mathbf{g}^k \rangle \in \mathbb{R}$.
     - 然后由此更新 $\mathbf{q}_{k-1} = \mathbf{g}^k - \alpha_{k-1} \mathbf{y}^{k-1} \in \mathbb{R}^n$.

- 然后进行第二轮迭代. 考虑 $\mathbf{q}_{k-2} = V^{k-2} \mathbf{q}_{k-1} = V^{k-2} V^{k-1} \mathbf{g}^k$ 的计算. 同样根据定义展开:
   $$
   \begin{aligned}
   \mathbf{q}_{k-2} &= V^{k-2} \mathbf{q}_{k-1} \\&= I\mathbf{q}_{k-1} - \rho_{k-2} \mathbf{y}^{k-2} (\mathbf{s}^{k-2})^\top \mathbf{q}_{k-1} \\
   &= \mathbf{q}_{k-1} - \underbrace{\rho_{k-2}\langle \mathbf{s}^{k-2}, \mathbf{q}_{k-1} \rangle}_{\alpha_{k-2}} \cdot \mathbf{y}^{k-2} \\&:= \mathbf{q}_{k-1} - \alpha_{k-2} \mathbf{y}^{k-2}.
   \end{aligned}
   $$
   - 故实际在第二轮迭代中:
     - 首先计算 $\alpha_{k-2} = \rho_{k-2}\langle \mathbf{s}^{k-2}, \mathbf{q}_{k-1} \rangle \in \mathbb{R}$,
     - 然后由此更新 $\mathbf{q}_{k-2} = \mathbf{q}_{k-1} - \alpha_{k-2} \mathbf{y}^{k-2} \in \mathbb{R}^n$.

- 以此类推, 在第 $j$ 轮迭代中, 首先计算 
   $$
   \alpha_j = \rho_j \langle \mathbf{s}^j, \mathbf{q}_{j+1} \rangle \in \mathbb{R},
   $$
   然后更新 
   $$
   \mathbf{q}_j = \mathbf{q}_{j+1} - \alpha_j \mathbf{y}^j = V^j V^{j+1} \cdots V^{k-1} \mathbf{g}^k\in \mathbb{R}^n.
   $$

因此, 整理 $m$ 轮迭代的结果, 我们最终得到 $\mathbf{q}_{k-1}, \alpha_{k-1}, \mathbf{q}_{k-2}, \alpha_{k-2}, \cdots, \mathbf{q}_{k-m}, \alpha_{k-m}$. 此时再次列出最开始的递推公式:
$$
\begin{aligned}
H^k \mathbf{g}^k &= (V^{k-1})^\top (V^{k-2})^\top \cdots (V^{k-m})^\top H^{k-m} {V^{k-m} \cdots V^{k-2} V^{k-1} \mathbf{g}^k} +\\
&\quad \blue{\rho_{k-m}} \left(V^{k-m+1}V^{k-m+2} \cdots V^{k-1}\right)^\top \mathbf{s}^{k-m} \blue{(\mathbf{s}^{k-m})^\top {\left(V^{k-m+1}V^{k-m+2} \cdots V^{k-1}\right) \mathbf{g}^k}} +\\
&\quad  \blue{\rho_{k-m+1}} \left(V^{k-m+2}V^{k-m+3} \cdots V^{k-1}\right)^\top \mathbf{s}^{k-m+1} \blue{(\mathbf{s}^{k-m+1})^\top {\left(V^{k-m+2}V^{k-m+3} \cdots V^{k-1}\right) \mathbf{g}^k}} +\\
&\quad \cdots + \\
&\quad  \blue{\rho_{k-1}} \mathbf{s}^{k-1} \blue{(\mathbf{s}^{k-1})^\top \mathbf{g}^k}.
\end{aligned}
$$
- 回顾, 在迭代中, 我们算出 $\alpha_j = \rho_j \langle \mathbf{s}^j, \mathbf{q}_{j+1} \rangle$ , $\mathbf{q}_j = \mathbf{q}_{j+1} - \alpha_j \mathbf{y}^j$ . 且本身 $\mathbf{q}_j = V^j V^{j+1} \cdots V^{k-1} \mathbf{g}^k$. 因此, 将这个迭代展开式代回 $\alpha_j = \rho_j \langle \mathbf{s}^j, \mathbf{q}_{j+1} \rangle$ 中, 则有:
   $$
   \blue{\alpha_j} = \rho_j \langle \mathbf{s}^j, \mathbf{q}_{j+1} \rangle = \blue{\rho_j \langle \mathbf{s}^j, V^{j+1} \cdots V^{k-1} \mathbf{g}^k \rangle}, \quad j = k-1, k-2, \cdots, k-m.
   $$
- 上面的完整递推公式中每一个加法项中的蓝色部分 $\rho_j \langle \mathbf{s}^j, V^{j+1} \cdots V^{k-1} \mathbf{g}^k \rangle$ 就是对应 $\alpha_j$ 的值. 


进而, 在第一轮循环结束后, 我们可以利用 $\alpha_j$ 来进一步简化原始的递推公式:
$$
\begin{aligned}
H^k \mathbf{g}^k &= (V^{k-m} V^{k-m+1} \cdots V^{k-1})^\top H^{k-m} \mathbf{q}_{k-m} +\\
&\qquad (V^{k-m+1}V^{k-m+2} \cdots V^{k-1})^\top \mathbf{s}^{k-m} \alpha_{k-m} +\\
&\qquad  (V^{k-m+2}V^{k-m+3} \cdots V^{k-1})^\top \mathbf{s}^{k-m+1} \alpha_{k-m+1} +\\
&\qquad \cdots +  V^{k-1\top} \mathbf{s}^{k-2} \alpha_{k-2} + \mathbf{s}^{k-1} \alpha_{k-1}.
\end{aligned}
$$
下面开始第二轮循环合并. 观察到, 上式子的各项加法中有大量的公共因子, 因此进行合并. 
- 第一步整理前两项. 
  - 展开 transpose, 提取前两项的绿色部分公共因子, 并将 $V^{k-m \top} = I - \rho_{k-m} \mathbf{s}^{k-m} (\mathbf{y}^{k-m})^\top$ 代入第一项中, 可得
     $$
     \begin{aligned}
     H^k \mathbf{g}^k   &= \green{V^{k-1\top} V^{k-2\top} \cdots V^{k-m+2 \top} V^{k-m+1\top}} V^{k-m\top} H^{k-m} \mathbf{q}_{k-m} +\\
     &\qquad  \green{V^{k-1\top} V^{k-2\top} \cdots V^{k-m+2 \top} V^{k-m+1\top}}\mathbf{s}^{k-m} \alpha_{k-m} +\\
     &\qquad  V^{k-1\top} V^{k-2\top} \cdots V^{k-m+2\top} \mathbf{s}^{k-m+1} \alpha_{k-m+1} +\\
     &\qquad \cdots +  V^{k-1\top} \mathbf{s}^{k-2} \alpha_{k-2} + \mathbf{s}^{k-1} \alpha_{k-1} \\
     &= \green{V^{k-1\top} V^{k-2\top} \cdots V^{k-m+2 \top} V^{k-m+1\top}} \left[ \left(I - \rho_{k-m} \mathbf{s}^{k-m} (\mathbf{y}^{k-m})^\top\right) H^{k-m} \mathbf{q}_{k-m} + \mathbf{s}^{k-m} \alpha_{k-m} \right] + \\
     &\qquad  V^{k-1\top} V^{k-2\top} \cdots V^{k-m+2\top} \mathbf{s}^{k-m+1} \alpha_{k-m+1} +\\
     &\qquad \cdots +  V^{k-1\top} \mathbf{s}^{k-2} \alpha_{k-2} + \mathbf{s}^{k-1} \alpha_{k-1} \\
     &= \green{V^{k-1\top} V^{k-2\top} \cdots V^{k-m+2 \top} V^{k-m+1\top}} \left[ \underbrace{H^{k-m} \mathbf{q}_{k-m}}_{\mathbf{r}_{k-m}} + \mathbf{s}^{k-m} (\alpha_{k-m} - \rho_{k-m} (\mathbf{y}^{k-m})^\top \mathbf{r}_{k-m})
     \right] +
     \\
     &\qquad  V^{k-1\top} V^{k-2\top} \cdots V^{k-m+2\top} \mathbf{s}^{k-m+1} \alpha_{k-m+1} +\\
     &\qquad \cdots +  V^{k-1\top} \mathbf{s}^{k-2} \alpha_{k-2} + \mathbf{s}^{k-1} \alpha_{k-1} \\
     \end{aligned}
     $$
   - 为方便, 引入两个变量: $\mathbf{r}_{k-m} := H^{k-m} \mathbf{q}_{k-m} \in \mathbb{R}^n$ 和 $\beta_{k-m} :=\rho_{k-m} (\mathbf{y}^{k-m})^\top \mathbf{r}_{k-m} \in \mathbb{R}$. 则上式可以简化为
     $$
     \begin{aligned}
     H^k \mathbf{g}^k   &= {V^{k-1\top} V^{k-2\top} \cdots V^{k-m+2 \top} V^{k-m+1\top}} \left[\underbrace{\mathbf{r}_{k-m} + \mathbf{s}^{k-m} (\alpha_{k-m} - \beta_{k-m})}_{\mathbf{r}_{k-m+1}}
     \right] +
     \\
     &\qquad  V^{k-1\top} V^{k-2\top} \cdots V^{k-m+2\top} \mathbf{s}^{k-m+1} \alpha_{k-m+1} +\\
     &\qquad \cdots +  V^{k-1\top} \mathbf{s}^{k-2} \alpha_{k-2} + \mathbf{s}^{k-1} \alpha_{k-1},
     \end{aligned}
     $$
     其中
      $$
      \begin{aligned}
      \mathbf{r}_{k-m+1} &:= \mathbf{r}_{k-m} + \mathbf{s}^{k-m} (\alpha_{k-m} - \beta_{k-m}), \\
      \beta_{k-m} &:= \rho_{k-m} (\mathbf{y}^{k-m})^\top \mathbf{r}_{k-m}.
      \end{aligned}
       $$
     根据结构的相似性, 又可以递归地定义 $\mathbf{r}_{k-m+1} := \mathbf{r}_{k-m} + \mathbf{s}^{k-m} (\alpha_{k-m} - \beta_{k-m}) \in \mathbb{R}^n$, 从而得到
     $$
       \begin{aligned}
       H^k \mathbf{g}^k   &= \purple{V^{k-1\top} V^{k-2\top} \cdots V^{k-m+2 \top}} V^{k-m+1\top} \mathbf{r}_{k-m+1} +
         \\
           &\qquad  \purple{V^{k-1\top} V^{k-2\top} \cdots V^{k-m+2\top}} \mathbf{s}^{k-m+1} \alpha_{k-m+1} +\\
         &\qquad \cdots +  V^{k-1\top} \mathbf{s}^{k-2} \alpha_{k-2} + \mathbf{s}^{k-1} \alpha_{k-1}.
       \end{aligned}
     $$

- 因此第二步又可以进行合并.
  - 所有展开的过程完全相同, 最终的结果为:
      $$
      \begin{aligned}
      H^k \mathbf{g}^k   &= V^{k-1\top} V^{k-2\top} \cdots V^{k-m+3 \top} V^{k-m+2\top} \mathbf{r}_{k-m+1} +\\
      &\qquad  V^{k-1\top} V^{k-2\top} \cdots V^{k-m+3\top} \mathbf{s}^{k-m+1} \alpha_{k-m+2} +\\
      &\qquad \cdots +  V^{k-1\top} \mathbf{s}^{k-2} \alpha_{k-2} + \mathbf{s}^{k-1} \alpha_{k-1}.
      \end{aligned}
      $$
      其中
      $$
      \begin{aligned}
      \mathbf{r}_{k-m+2} &:= \mathbf{r}_{k-m+1} + \mathbf{s}^{k-m+1} (\alpha_{k-m+1} - \beta_{k-m+1}), \\
      \beta_{k-m+1} &:= \rho_{k-m+1} (\mathbf{y}^{k-m+1})^\top \mathbf{r}_{k-m+1}.
      \end{aligned}  
      $$
  - 以此类推, 最终可以得到在第 $j \in \{k-1, k-2, \cdots, k-m\}$ 步, 一般的迭代公式为
      $$
      \begin{aligned}
      H^k \mathbf{g}^k   &= V^{k-1\top} V^{k-2\top} \cdots V^{j+1 \top} V^{j\top} \mathbf{r}_{j+1} +\\
      &\qquad  V^{k-1\top} V^{k-2\top} \cdots V^{j+1\top} \mathbf{s}^{j} \alpha_{j+1} +\\
      &\qquad \cdots +  V^{k-1\top} \mathbf{s}^{k-2} \alpha_{k-2} + \mathbf{s}^{k-1} \alpha_{k-1},
      \end{aligned}
      $$
      其中
       $$
       \begin{aligned}
       \mathbf{r}_{j+1} &:= \mathbf{r}_{j} + \mathbf{s}^{j} (\alpha_{j} - \beta_{j}), \\
       \beta_{j} &:= \rho_{j} (\mathbf{y}^{j})^\top \mathbf{r}_{j}.
       \end{aligned}
        $$

  - 最后, 在完成 $j = k-1$ 的迭代后, 就得到了最终的搜索方向 $\mathbf{d}^k = -H^k \mathbf{g}^k$ 的计算公式:
      $$
      \begin{aligned}
      \mathbf{d}^k = -H^k \mathbf{g}^k   &= -\mathbf{r}_{k-1} - \mathbf{s}^{k-1} \alpha_{k-1} \\
      &= -\mathbf{r}_{k-2} - \mathbf{s}^{k-2} \alpha_{k-2} - \mathbf{s}^{k-1} \alpha_{k-1} \\
      &\qquad \cdots \\
      &= -\mathbf{r}_{k-m} - \sum_{j=k-m}^{k-1} \mathbf{s}^{j} \alpha_j.
      \end{aligned}
      $$