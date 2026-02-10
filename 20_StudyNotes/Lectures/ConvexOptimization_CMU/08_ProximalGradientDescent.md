# Proximal Gradient Descent

> - Lecture Reference: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/
> - Reading Reference: 最优化：建模、算法与理论，刘浩洋等，第 8.1 节

## Proximal Gradient Descent

### Decomposable Functions

在统计建模中, 对于目标函数经常能够分解成两个部分的和, 其中一个部分是光滑的, 另一个部分是非光滑的. 例如, Lasso 回归中的目标函数可以分解为平方损失函数(光滑)和 L1 正则化项(非光滑). 形式化地, (暂时只考虑凸函数) 我们可以将目标函数表示为:
$$
F(x) = \phi(x) + h(x)
$$
- $\phi$ 是光滑凸函数, 不妨令 $\text{dom}(\phi) = \mathbb{R}^n$.
- $h$ 是非光滑凸函数, 但具有简单的结构, 使得我们能够高效地计算其**近端算子**(proximal operator).

在处理这样的优化问题时传统的梯度下降方法可能无法直接应用, 因为 $h$ 的非光滑性会导致梯度不存在; 而若直接使用次梯度方法, 则可能会导致收敛速度较慢. 

因此我们需要一种新的优化方法来处理这种分解结构的目标函数, 这就是**近端梯度下降**(Proximal Gradient Descent) 方法. 其对于光滑的部分 $\phi$ 使用梯度信息, 对于非光滑的部分 $h$ 则利用其近端算子来进行优化.

回顾一下, 传统的梯度下降方法在每一步迭代中更新参数 $x$ 的方式为: $x^+ = x - t \Delta x$. 我们选择 $\Delta x = \nabla f(x)$ 是因为想要最小化一个近似的目标函数:
$$
\min_{z} f(x) \approx f(x) + \nabla f(x)^\top (z-x) + \frac{1}{2t} \|z-x\|^2
$$
- 其中, 二次项是相当于将泰勒展开中的二次项 $\frac{1}{2} (z-x)^\top \nabla^2 f(x) (z-x)$ 用 $\frac{1}{t} I \approx \nabla^2 f(x)$ 来近似的结果.
- 而 $\Delta x = \nabla f(x)$ 的选择使得 $x^+$ 成为上述近似目标函数的最小值: $x^+ = \arg\min_{z} f(x) + \nabla f(x)^\top (z-x) + \frac{1}{2t} \|z-x\|^2$.

在当前的 Composable Functions 的设置中, 类似地有:
$$\begin{aligned}
x^+ &= \arg\min_{z} \left[\phi(z)+h(z)\right] \\
&\approx \arg\min_{z} \left[\phi(x) + \nabla \phi(x)^\top (z-x) + \frac{1}{2t} \|z-x\|^2  + h(z)\right] \\
&= \arg\min_{z} \left[\frac{1}{2t} \left\|z - (x - t \nabla \phi(x))\right\|^2 + h(z)\right]
\end{aligned}$$
- 其中最后一个等式就是将最小化中与 $z$ 无关项丢弃后并配方得到的纯代数整理结果.
- 这个更新步骤的核心思想是: 在每次迭代中, 首先对光滑部分 $\phi$ 进行梯度下降的更新, 得到一个临时变量 $x' = x - t \nabla \phi(x)$; 然后通过最小化一个包含非光滑部分 $h$ 的二次近似目标函数来得到新的参数 $x^+$.

### Proximal Operator

事实上, 上述推导的最后一步, 若将 $x-t \nabla \phi(x)$ 当作一个新的输入, 那么这就是一个在给定非光滑函数 $h$ 的情况下, 进行近端映射的过程. 即:
$$x^+ := \text{prox}_{t h}(x - t \nabla \phi(x)) = \arg\min_{z} \left[\frac{1}{2t} \|z - (x - t \nabla \phi(x))\|^2 + h(z)\right]$$

***Definition* (近端算子):** 对于一个凸函数 $h: \mathbb{R}^n \to \mathbb{R}$, 其近端算子 $\text{prox}_{h}: \mathbb{R}^n \to \mathbb{R}^n$ 定义为:
$$\text{prox}_{h}(x) = \arg\min_{z} \left[\frac{1}{2} \|z - x\|^2 + h(z)\right]$$
- 其中, $x$ 是输入向量, $z$ 是优化变量, 与 $x$ 在同一空间中. 该算子相当于在 $x$ 的基础上进行一个平滑的调整, 使得调整后的点能够在最小化非光滑函数 $h$ 的同时尽可能接近 $x$.

其具有如下性质: 对于适当的闭凸函数 $h$:
- **对任意 $x\in\mathbb{R}^n$, $\text{prox}_{h}(x)$ 都是存在且唯一的.**
  - 这保证了近端梯度下降方法在每次迭代中都能够得到一个明确的更新结果.
- **$u = \text{prox}_{h}(x)$ 等价于 $x-u \in \partial h(u)$, 其中 $\partial h(u)$ 是 $h$ 在 $u$ 处的次微分.**
  - 近端算子所得到的点 $u$ 与输入点 $x$ 之间的差异正好对应于 $h$ 在 $u$ 处的次梯度信息.
  - *Proof*
    - 若已知 $u = \text{prox}_{h}(x)$, 则 $u$ 是以下优化问题的最小值: $ \min_{v} \phi(v) := \frac{1}{2} \|v - x\|^2 + h(v)$. 因此 $\mathbf{0} \in \partial \phi(u) = u - x + \partial h(u)$, 从而 $x-u \in \partial h(u)$
    - 若 $x-u \in \partial h(u)$, 则由次梯度定义可知 $h(v) \geq h(u) + (x-u)^\top (v-u)$, 从而 $h(v)+ \frac{1}{2} \|v-x\|^2 \geq h(u) + (x-u)^\top (v-u) + \frac{1}{2} \|v-x\|^2 \ge h(u) + \frac{1}{2} \|u-x\|^2$, 从而 $u = \text{prox}_{h}(x)$ (其中最后一个不等式是通过将$\|v-x\|^2 = \|v-u+u-x\|^2$ 展开后进行配方整理得到的结果).
    $\square$

进一步, 用 $t>0$ 来缩放函数 $h$, 则有:
***Definition* (缩放的近端算子):** 对于一个凸函数 $h: \mathbb{R}^n \to \mathbb{R}$ 以及一个正标量 $t>0$, 其缩放的近端算子 $\text{prox}_{t h}: \mathbb{R}^n \to \mathbb{R}^n$ 定义为:
$$\text{prox}_{t h}(x) = \arg\min_{z} \left[\frac{1}{2} \|z - x\|^2 + t h(z)\right]$$
- 该算子与未缩放的近端算子类似, 但在优化目标中对函数 $h$ 进行了缩放, 这在实际应用中可以调整近端映射的强度.
- 上述和次梯度的关系同样成立: $u = \text{prox}_{t h}(x)$ 等价于 $x-u \in t \partial h(u)$.

***Example* ($\ell_1$ 的 prox 算子):** 对于 $h(x) = \|x\|_1$ 其中 $x\in\mathbb{R}^n$ 及 $t>0$, 其 prox 算子 $u = \text{prox}_{t h}(x)$ 的计算结果为:
$$\text{prox}_{t h}(x) = \text{sign}(x) \odot \max\{|x| - t, 0\}$$

### Proximal Gradient Descent Algorithm

对于可分解的凸优化问题:
$$\min_{x\in\mathbb{R}^n} f(x) = \phi(x) + h(x)$$
- 其中 $\phi$ 是光滑凸函数, $h$ 是非光滑凸函数. 我们可以通过近端梯度下降算法来求解该问题. 
- 事实上, 对于含约束的优化问题同样也可以令 $h(x) = \delta_{\mathcal{C}}(x)$ 来将约束条件隐式地包含在非光滑函数中, 从而使得近端梯度下降算法同样适用.

Proximal Gradient Descent 的迭代更新步骤如下:
1. **初始化**: 选择一个初始点 $x^{(0)} \in \mathbb{R}^n$ 和一个步长参数 $t_0> 0$.
2. **迭代更新**: 对于 $k = 0, 1, 2, \ldots$ 进行以下更新:
   $$\boxed{x^{(k+1)} = \text{prox}_{t_k h}(x^{(k)} - t_k \nabla \phi(x^{(k)}))}$$
   - 其中, $x^{(k)}$ 是当前迭代的参数, $\nabla \phi(x^{(k)}$ 是光滑部分 $\phi$ 在 $x^{(k)}$ 处的梯度, $t_k$ 是步长参数且同样可以设置为常数或通过线搜索等方法自适应调整, $\text{prox}_{t_k h}$ 是非光滑部分 $h$ 的缩放近端算子.

- 当 $h(x) = 0$ 时, 该算法退化为传统的梯度下降方法:
    $$x^{(k+1)} = x^{(k)} - t_k \nabla \phi(x^{(k)})$$
- 当 $h(x) = \delta_{\mathcal{C}}(x)$ 时, 该算法退化为投影梯度下降方法:
    $$x^{(k+1)} = \text{Proj}_{\mathcal{C}}(x^{(k)} - t_k \nabla \phi(x^{(k)}))$$

观察上述迭代更新, 其还可以等价表述为:
$$\begin{aligned}
x^{(k+1)} &= \text{prox}_{t_k h}\left(x^{(k)} - t_k \nabla \phi(x^{(k)})\right) \\
&= x^{(k)} - t_k \cdot \frac{x^{(k)} - \text{prox}_{t_k h}(x^{(k)} - t_k \nabla \phi(x^{(k)}))}{t_k} := x^{(k)} - t_k G_{t_k}(x^{(k)})\\
&= x^{(k)} - t_k  \nabla \phi(x^{(k)}) - t_k g^{(k)}
\end{aligned}$$
- 其中第二个等式中 $G_{t_k}(x^{(k)}) =  \left[x^{(k)} - \text{prox}_{t_k h}[x^{(k)} - t_k \nabla \phi(x^{(k)})]\right]/t_k$ 被称为**近端梯度映射**(proximal gradient mapping), 其在某种意义上可以看作是一个综合了光滑部分梯度信息和非光滑部分次梯度信息的复合梯度.
- 第三个等式也显式地展示了近端梯度下降相当于对光滑部分进行梯度下降的同时, 对非光滑部分进行隐式梯度下降. 推导如下: 
  - 根据更新规则 $x^{(k+1)} = \text{prox}_{t_k h}(x^{(k)} - t_k \nabla \phi(x^{(k)})):=\text{prox}_{t_k h}(x')$ 以及 prox 算子与次梯度的关系, 可知: $x' - x^{(k+1)} \in t_k \partial h(x^{(k+1)})$
  - 从而存在 $g^{(k)} \in \partial h(x^{(k+1)})$ 使得 $x' - x^{(k+1)} = t_k g^{(k)}$, 即 
    $$x^{(k+1)} = x^{(k)} - t_k \nabla \phi(x^{(k)}) - t_k g^{(k)}$$

对于步长的选择:
- 若 $\phi$ 的梯度是 $L$-Lipschitz 连续的, 选择 $t_k = t \leq 1/L$.
- 当 $L$ 不可知时, 可采用如下线搜索方法来选择步长 $t_k$:
   $$\phi(x^{(k+1)}) \leq \phi(x^{(k)}) +(\nabla \phi(x^{(k)}))^\top (x^{(k+1)} - x^{(k)}) + \frac{1}{2 t_k} \|x^{(k+1)} - x^{(k)}\|^2$$

### Examples 

#### ISTA for Lasso Regression

给定 $y\in\mathbb{R}^n$ 和 $X\in\mathbb{R}^{n\times p}$, Lasso 回归的目标函数为:
$$\min_{\beta\in\mathbb{R}^p} \underbrace{\frac{1}{2} \|y - X\beta\|_2^2}_{\phi(\beta)} + \underbrace{\lambda \|\beta\|_1}_{h(\beta)}$$
- 其中 $\phi(\beta) = \frac{1}{2} \|y - X\beta\|_2^2$ 是光滑凸函数, $h(\beta) = \lambda \|\beta\|_1$ 是非光滑凸函数.
- 利用 proximal gradient descent 方法, 首先分别求解梯度与近端算子:
  - $\nabla \phi(\beta) = X^\top (X\beta - y)$
  - $\text{prox}_{t h}(\beta) = \text{sign}(\beta) \odot \max\{|\beta| - t \lambda, 0\}:= S_{t \lambda}(\beta)$, 其中 $S_{t \lambda}(\cdot)$ 是**软阈值函数**(soft-thresholding function).
- 因此, proximal gradient descent 的迭代更新步骤为:
    $$\beta^{(k+1)} = S_{t \lambda}(\beta^{(k)} - t X^\top (X\beta^{(k)} - y))$$    
    该步骤也可以化作如下分步进行的形式:
    - 首先计算一个临时变量 $\beta' = \beta^{(k)} - t X^\top (X\beta^{(k)} - y)$, 这相当于对光滑部分 $\phi$ 进行梯度下降的更新.
    - 然后对 $\beta'$ 进行软阈值处理 $\beta^{(k+1)} = S_{t \lambda}(\beta') = \text{sign}(\beta') \odot \max\{|\beta'| - t \lambda, 0\}$, 这相当于对非光滑部分 $h$ 进行近端映射的处理.
- 该算法被称为**迭代软阈值算法**(Iterative Soft-Thresholding Algorithm, ISTA), 是一种求解 Lasso 回归问题的经典方法.

#### Low-rank Matrix Completion

给定一个矩阵 $M \in \mathbb{R}^{m\times n}$ 以及一个索引集合 $\Omega \subseteq \{1, \ldots, m\} \times \{1, \ldots, n\}$ 表示已知的矩阵元素索引, 低秩矩阵补全的目标函数为:
$$\begin{aligned} &\min_{X\in\mathbb{R}^{m\times n}} && \text{rank}(X) \\ &\text{subject to} && X_{ij} = M_{ij}, \forall (i,j) \in \Omega \end{aligned}$$

该优化问题进一步可以通过 Nuclear Norm 的松弛来转化为如下形式:
$$\begin{aligned} &\min_{X\in\mathbb{R}^{m\times n}} && \|X\|_* \\ &\text{subject to} && X_{ij} = M_{ij}, \forall (i,j) \in \Omega  \end{aligned}$$
- 可以证明该形式是一个凸优化问题. 

若进一步考虑到观测数据中可能存在噪声, 则可以将约束条件松弛为一个正则项, 从而得到如下优化问题:
$$\min_{X\in\mathbb{R}^{m\times n}} {\mu \|X\|_*} + {\frac{1}{2} \sum_{(i,j)\in\Omega} (X_{ij} - M_{ij})^2}=\min_{X\in\mathbb{R}^{m\times n}} \underbrace{\mu \|X\|_*}_{h(X)} + \underbrace{\frac{1}{2} \|P\odot (X - M)\|_F^2}_{\phi(X)}$$
- 其中 $\mu>0$ 是一个正则化参数, 用于平衡核范数正则项与数据拟合项之间的权重. $P_{ij} = 1$ 当 $(i,j) \in \Omega$ 时, 否则 $P_{ij} = 0$. $\odot$ 表示元素级乘法. $\|\cdot\|_F$ 是 Frobenius 范数, 定义为 $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$.
- 该优化问题同样可以通过 proximal gradient descent 方法来求解:
  - $\nabla \phi(X) = P \odot (X - M)$
  - $\text{prox}_{t_k h}(X) = U \cdot \text{Diag}(\max\{|d| - t_k \mu, 0\}) \cdot V^\top$, 其中 $X = U \cdot \text{Diag}(d) \cdot V^\top$ 是 $X$ 的奇异值分解(SVD).
- 因此, proximal gradient descent 的迭代更新步骤为:
$$X^{(k+1)} = U \cdot \text{Diag}(\max\{|d - t_k P \odot (X^{(k)} - M)| - t_k \mu, 0\}) \cdot V^\top$$

## Convergence Analysis


## Special Cases

## Acceleration

