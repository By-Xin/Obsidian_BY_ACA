# Proximal Gradient Descent

> - Lecture Reference: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/

## Proximal Gradient Descent

- *Reading Reference: 最优化：建模、算法与理论，刘浩洋等，第 8.1 节*

### Decomposable Functions

在统计建模中, 对于目标函数经常能够分解成两个部分的和, 其中一个部分是光滑的, 另一个部分是非光滑的. 例如, Lasso 回归中的目标函数可以分解为平方损失函数(光滑)和 L1 正则化项(非光滑). 形式化地, (暂时只考虑凸函数) 我们可以将目标函数表示为:
$$
F(x) = \phi(x) + h(x)
$$
- $\phi$ 是光滑凸函数, 不妨令 $\text{dom}(\phi) = \mathbb{R}^n$.(事实上该条件可以放宽为只要求可微性存在即可)
- $h$ 是非光滑的凸函数, 但具有简单的结构, 使得我们能够高效地计算其**近端算子**(proximal operator).

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
&= x^{(k)} - t_k \cdot \frac{x^{(k)} - \text{prox}_{t_k h}(x^{(k)} - t_k \nabla \phi(x^{(k)}))}{t_k} \\&:= \boxed{x^{(k)} - t_k G_{t_k}(x^{(k)})}\\
&= \boxed{x^{(k)} - t_k  \nabla \phi(x^{(k)}) - t_k g^{(k)}}
\end{aligned}$$
- 其中第三个等式中 
  $$G_{t_k}(x^{(k)}) =  \frac{x^{(k)} - \text{prox}_{t_k h}[x^{(k)} - t_k \nabla \phi(x^{(k)})]}{t_k}$$ 
  被称为**近端梯度映射(proximal gradient mapping)**.
  - 其在某种意义上可以看作是一个综合了光滑部分梯度信息和非光滑部分次梯度信息的复合梯度, 其作用也相当于传统梯度下降法中的搜索方向.
  - Proximal Gradient Mapping 与次梯度的关系为: 
    $$\boxed{G_{t_k}(x^{(k)}) - \nabla \phi(x^{(k)}) \in \partial h(x^{(k+1)})=\partial h(x - t_k G_{t_k}(x^{(k)})))} \quad (\dagger)$$
  - 此外 $G_{t_k}(x^{(k)})$ 还具有如下性质: $G_{t_k}(x^{(k)}) = \mathbf{0}$ 当且仅当 $x^{(k)}$ 是 $F(x) = \phi(x) + h(x)$ 的一个最优解.
- 第四个等式也显式地展示了近端梯度下降相当于对光滑部分进行梯度下降的同时, 对非光滑部分进行隐式梯度下降. 推导如下: 
  - 根据更新规则 $x^{(k+1)} = \text{prox}_{t_k h}(x^{(k)} - t_k \nabla \phi(x^{(k)})):=\text{prox}_{t_k h}(x')$ 以及 prox 算子与次梯度的关系, 可知: $x' - x^{(k+1)} \in t_k \partial h(x^{(k+1)})$
  - 从而存在 $g^{(k)} \in \partial h(x^{(k+1)})$ 使得 $x' - x^{(k+1)} = t_k g^{(k)}$, 即 
    $$x^{(k+1)} = x^{(k)} - t_k \nabla \phi(x^{(k)}) - t_k g^{(k)}$$

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

### Backtracking Line Search for Proximal Gradient Descent

对于 proximal gradient descent 方法, backtracking line search 同样成立, 只不过其搜索对象是分解后光滑部分 $\phi$ 的梯度.

整体而言, 给定 $\beta \in (0,1)$ 及初始化 $t_0 > 0$, backtracking line search 的步骤如下:
- 若 $\phi(x - t \cdot G_t(x)) > \phi(x) + (\nabla \phi(x))^\top (-t \cdot G_t(x)) + \frac{1}{2t} \|t \cdot G_t(x)\|^2$, 则令 $t = \beta t$ 并重复上述条件的检查, 直到满足条件为止.
- 其中 $G_t(x) = \frac{1}{t} (x - \text{prox}_{t h}(x - t \nabla \phi(x)))$ 是 proximal gradient mapping.

在进行搜索后, 对非光滑部分 $h$ 的近端映射仍然保持不变.

不过还需要额外指出, 在 proximal gradient descent 中, 存在比 backtracking line search 更加高效的步长选择方法.

## Convergence Analysis

- *Reading Reference: 最优化：建模、算法与理论，刘浩洋等，第 8.1 节*

### Algorithm and Assumptions

回顾完整的 proximal gradient descent 算法. 考虑如下优化问题:
$$\min_{x\in\mathbb{R}^n} F(x) = \phi(x) + h(x)$$
- 其中 $\phi$ 是光滑凸函数, $h$ 是非光滑的凸函数. 
  
Proximal gradient descent 的迭代更新步骤为:
$$x^{(k+1)} = \text{prox}_{t_k h}(x^{(k)} - t_k \nabla \phi(x^{(k)})):= x^{(k)} - t_k G_{t_k}(x^{(k)})$$
- 其中 $G_{t_k}(x^{(k)}) = \frac{1}{t_k} (x^{(k)} - \text{prox}_{t_k h}(x^{(k)} - t_k \nabla \phi(x^{(k)})))$ 是 proximal gradient mapping.

在进行收敛性分析之前, 明确如下假设:
1. $\phi$ 在定义域 $\mathbb{R}^n$ 上是凸且其梯度 $\nabla \phi$ 是 $L$-Lipschitz 连续的, 即对于任意 $x,y \in \mathbb{R}^n$, 都有 $\|\nabla \phi(x) - \nabla \phi(y)\| \leq L \|x-y\|$.
2. $h$ 是一个适当的闭凸函数, 使得其近端算子 $\text{prox}_{t h}$ 是良好定义的.
3. 目标函数 $F(x) = \phi(x) + h(x)$ 的最小值 $F^* = F(x^*)$ 是可达且有限的, 并在某个点 $x^*$ 处达到. 不过这里并不要求 $x^*$ 是唯一的.

### Convergence Rate

在上述假设条件下, proximal gradient descent 方法的收敛性由以下定理保证:
***Theorem* (Proximal Gradient Descent 的收敛率):** 在满足上述条件, 并给定步长 $t_k = t \in (0, 1/L]$ 的情况下, 迭代序列 $\{x^{(k)}\}$ 满足
$$F(x^{(k)}) - F^* \leq \frac{\|x^{(0)} - x^*\|^2}{2 t k}$$
即迭代点 $x^{(k)}$ 的函数值以 $\mathcal{O}(1/k)$ 的速率收敛到最优值 $F^*$.

- *Proof*
  - 根据假设中的 $L$-Lipschitz 连续性, 对 $\phi$ 进行二阶泰勒展开的上界估计, 可得对于任意 $x,y \in \mathbb{R}^n$, 都有:
    $$\phi(y) \leq \phi(x) + \nabla \phi(x)^\top (y-x) + \frac{L}{2} \|y-x\|^2$$
    - 令此处的 $y = x - t G_t(x)$, 则有:
    $$\phi(x - t G_t(x)) \leq \phi(x) - t \nabla \phi(x)^\top G_t(x) + \frac{L t^2}{2} \|G_t(x)\|^2$$
    - 根据步长假设 $t \leq 1/L$, 可得 $\frac{L t^2}{2} \|G_t(x)\|^2 \leq \frac{t}{2} \|G_t(x)\|^2$. 从而:
    $$\phi(x - t G_t(x)) \leq \phi(x) - t \nabla \phi(x)^\top G_t(x) + \frac{t}{2} \|G_t(x)\|^2 ,\quad(1)$$
  - 另一方面, 根据假设 $\phi(x), h(x)$ 均为凸函数, 对于任意 $z\in\text{dom}(F)$, 都有:
    - $\phi(x) \leq \phi(z)  - \nabla \phi(x)^\top (z-x), \quad (2)$
    - $h(x') \leq h(z) - g^\top (z-x')$ 其中 $g \in \partial h(x'),$, $x' = x-t G_t(x)$. 从而若将 $x'$ 代入 $h$ 的不等式中, 则有
    $$h(x - t G_t(x)) \leq h(z) - (G_t(x) - \nabla \phi(x))^\top (z - x + t G_t(x)),\quad(3)$$
      - 其中 $g = G_t(x) - \nabla \phi(x) \in \partial h(x-t G_t(x))$ 是根据 prox 算子与次梯度的关系得到的结果, 见 $(\dagger)$.
  - 将 $(1), (2), (3)$ 三个不等式相加, 并根据 decomposable function 的定义 $F(x) = \phi(x) + h(x)$, 经整理化简, 对任意 $z\in\text{dom}(F)$ 都有:
    $$\begin{aligned}
    F(x - t G_t(x)) &\leq F(z) +G_t(x)^\top (x-z) - \frac{t}{2} \|G_t(x)\|^2 \\
    \end{aligned}$$
    若另记 $x^+ = x - t G_t(x)$, 则上式可以化作如下形式:
  $$F(x^+) \leq F(z) + G_t(x)^\top (x-z) - \frac{t}{2} \|G_t(x)\|^2$$
  - 令 $z=x$, 则有:
    $$F(x^+) \leq F(x) - \frac{t}{2} \|G_t(x)\|^2$$
     - 这表明每次迭代都会使得函数值至少下降 $\frac{t}{2} \|G_t(x)\|^2$, 从而保证了函数值的单调不增.
  - 特别地, 令 $z = x^*$, 则有:
    $$\begin{aligned}
    F(x^+) &\leq F(x^*) + G_t(x)^\top (x-x^*) - \frac{t}{2} \|G_t(x)\|^2 \\
    &= \frac{1}{2t} (\|x-x^*\|^2 - \|x - x^*- t G_t(x)\|^2) \\
    &= \frac{1}{2t} (\|x-x^*\|^2 - \|x^+ - x^*\|^2) 
    \end{aligned}$$
    - 其中第二个等式是通过单纯的代数整理得到的: $v^\top u - \frac{t}{2} \|v\|^2 = \frac{1}{2t} (\|u\|^2 - \|u - t v\|^2)$.
  - 因此从 $x^{(0)}$ 开始迭代, 可以得到如下递推关系:
    $$F(x^{(k+1)}) - F^* \leq \frac{1}{2t} (\|x^{(k)}-x^*\|^2 - \|x^{(k+1)} - x^*\|^2)$$
    将上述不等式两边同时求和, 则有:
    $$\sum_{i=0}^{k-1} (F(x^{(i+1)}) - F^*) \leq \frac{1}{2t} \|x^{(0)} - x^*\|^2$$
    从而由于 $F(x^{(i+1)})$ 是单调不增的, 可得:
    $$k (F(x^{(k)}) - F^*) \leq \sum_{i=0}^{k-1} (F(x^{(i+1)}) - F^*) \leq \frac{1}{2t} \|x^{(0)} - x^*\|^2$$
    从而得到最终的收敛率结果:
    $$F(x^{(k)}) - F^* \leq \frac{\|x^{(0)} - x^*\|^2}{2 t k}$$
$\square$

如果我们使用 backtracking line search 来选择步长, 我们可以从某个 $t = t_0>0$ 开始, 通过不断缩小 $t \leftarrow \beta t$ 来不断回溯, 直到满足条件:
$$\phi(x - t G_t(x)) \leq \phi(x) - t \nabla \phi(x)^\top G_t(x) + \frac{t}{2} \|G_t(x)\|^2$$
并且可以由类似的分析过程来证明, 在满足上述条件的情况下, 其收敛情况为:
$$F(x^{(k)}) - F^* \leq \frac{\|x^{(0)} - x^*\|^2}{2 k \min\{t_0, \beta/L\}}$$



## Special Cases

根据 Composable Functions 的部分不同, 我们还有一些特殊的优化算法可以看作是 Proximal Gradient Descent 的特例.

### Projected Gradient Descent

由 Proximal Gradient Descent 的定义可知, 当非光滑部分 $h$ 是一个指示函数 $\delta_{\mathcal{C}}$ 时, 其近端算子 $\text{prox}_{t h}$ 就退化为一个投影算子 $\text{Proj}_{\mathcal{C}}$. 因此, 在这种特殊情况下, Proximal Gradient Descent 就退化为传统的**投影梯度下降**(Projected Gradient Descent) 方法. 其迭代更新步骤为:
$$x^{(k+1)} = \text{Proj}_{\mathcal{C}}(x^{(k)} - t_k \nabla \phi(x^{(k)}))$$
- 其中 $\text{Proj}_{\mathcal{C}}(x) = \arg\min_{z\in\mathcal{C}} \|z-x\|$ 是将 $x$ 投影到集合 $\mathcal{C}$ 上的操作.
- 其含义即为: 在每次迭代中, 首先对光滑部分 $\phi$ 进行梯度下降的更新, 得到一个临时变量 $x' = x^{(k)} - t_k \nabla \phi(x^{(k)})$; 然后将 $x'$ 投影到约束集合 $\mathcal{C}$ 上, 从而得到新的参数 $x^{(k+1)}$.

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260214003311.png)

### Proximal Point Algorithm 

- *Reading Reference: 最优化：建模、算法与理论，刘浩洋等，第 8.3 节*

当光滑部分 $\phi$ 恒为零时, 考虑如下优化问题:
$$\min_{x\in\mathbb{R}^n} F(x)$$
- 其中 $F$ 是一个适当的闭凸函数, 并不要求其具有可微性.

对于上述优化问题, Proximal Gradient Descent 的迭代更新步骤退化为:
$$x^{(k+1)} = \text{prox}_{t_k F}(x^{(k)}) = \arg\min_{z} \left[\frac{1}{2t_k} \|z - x^{(k)}\|^2 + F(z)\right]$$

## Acceleration: Nesterov's Accelerated Gradient Method

对于优化问题
$$\min_{x\in\mathbb{R}^n} F(x) = \phi(x) + h(x)$$
使用 Proximal Gradient Descent, 如果光滑部分的函数 $\phi$ 是 $L$-smooth 的, 则目标函数的收敛速度为 $\mathcal{O}(1/k)$. 但是, 通过一些加速技巧, 可以将收敛速度提升到 $\mathcal{O}(1/k^2)$. Nesterov 在 1983, 1988, 2005 年提出了三种改进的一阶算法. Beck 和 Teboulle 在 2008 年给出了 Nesterov 1983 算法的 Proximal Gradient 版本, 被称为 **FISTA**(Fast Iterative Shrinkage-Thresholding Algorithm).

### FISTA Algorithm

FISTA 的算法步骤如下:
1. **初始化**: 选择一个初始点 $x^{(0)} \in \mathbb{R}^n$, 并令 $x^{(-1)} = x^{(0)}$ 
2. **迭代更新**: 对于 $k = 1, 2, \ldots$ 进行以下更新直到满足收敛条件:
   - $v^{(k)} = x^{(k-1)} + \frac{k-2}{k+1} (x^{(k-1)} - x^{(k-2)})$
   - $x^{(k)} = \text{prox}_{t_k h}(v^{(k)} - t_k \nabla \phi(v^{(k)}))$

其还有一种表述形式:
1. **初始化**: 选择一个初始点 $x^{(0)} \in \mathbb{R}^n$, 并令 $v^{(0)} = x^{(0)}$. 选定加速参数 $\gamma_k$.
2. **迭代更新**: 对于 $k = 1, 2, \ldots$ 进行以下更新直到满足收敛条件:
   - 计算 $y^{(k)} = (1-\gamma_k)x^{(k-1)} + \gamma_k v^{(k-1)}$
   - 选择 $t_k > 0$, 计算 $x^{(k)} = \text{prox}_{t_k h}(y^{(k)} - t_k \nabla \phi(y^{(k)}))$
   - 计算 $v^{(k)} = x^{(k-1)} + \frac{1}{\gamma_k} (x^{(k)} - x^{(k-1)})$

当 $\gamma_k = \frac{2}{k+1}$ 且步长固定时, 上述两种表述形式是等价的.

对于 FISTA 算法, 在步长 $t_k$ 和加速参数 $\gamma_k$ 满足如下条件的情况下 (此处 notation 以第二种表达形式为准), 其收敛率为 $\mathcal{O}(1/k^2)$:
1. $f(x^{(k)}) \leq f(y^{(k)}) + \langle \nabla f(y^{(k)}), x^{(k)} - y^{(k)} \rangle + \frac{1}{2 t_k} \|x^{(k)} - y^{(k)}\|^2_2$
2. $\gamma_1=1$; 对于 $j \geq 2$, $\dfrac{(1-\gamma_j)t_j}{\gamma_j^2} \leq \dfrac{t_{j-1}}{\gamma_{j-1}^2}$
3. $\dfrac{\gamma_k^2}{t_k} = \mathcal{O}(1/k^2)$.

在满足上述假设条件, 并在固定步长 $t_k = t \in (0, 1/L]$ 的情况下, FISTA 迭代序列 $\{x^{(k)}\}$ 满足
$$F(x^{(k)}) - F^* \leq \frac{2 L \|x^{(0)} - x^*\|^2}{(k+1)^2}$$

#### Line Search for FISTA

对于 FISTA 算法, 其步长 $t_k$ 的选择同样可以通过 backtracking line search 来进行调整. 

最基础的一个版本的 line search 过程如下:
1. **初始化**: 选择一个初始点 $x^{(0)} \in \mathbb{R}^n$, 并令 $v^{(0)} = x^{(0)}$. 选定加速参数 $\gamma_k$.
2. **迭代更新**: 对于 $k = 1, 2, \ldots$ 进行以下更新直到满足收敛条件:
   - 计算 $y^{(k)} = (1-\gamma_k)x^{(k-1)} + \gamma_k v^{(k-1)}$
   - 通过 line search 来选择 $t_k$ 并更新 $x^{(k)}$ (给定迭代起始搜索步长 $t_k = t_{k-1}>0$, 以及缩放因子 $\rho \in (0,1)$):
     - 计算 $x^{(k)} = \text{prox}_{t_k h}(y^{(k)} - t_k \nabla \phi(y^{(k)}))$
     - 若 $\phi(x^{(k)}) > \phi(y^{(k)}) + \langle \nabla \phi(y^{(k)}), x^{(k)} - y^{(k)} \rangle + \frac{1}{2 t_k} \|x^{(k)} - y^{(k)}\|^2_2$, 则
       - 令 $t_k \leftarrow \rho t_k$ ,
       - 重复上述计算 $x^{(k)}$ 和条件检查, 直到满足条件为止.
     - 返回满足条件的 $t_k$ 和对应的 $x^{(k)}$.
   - 计算 $v^{(k)} = x^{(k-1)} + \frac{1}{\gamma_k} (x^{(k)} - x^{(k-1)})$

其问题在于: 对于第 $k$ 次迭代, 其 line search 的初始步长 $t_k$ 的选择总是取为 $t_{k-1}$, 这导致其在迭代过程中是不断缩小步长的, 从而可能会导致步长过小, 进而影响算法的收敛速度.

通过其他一些改进的 line search 方法, 例如通过求解关于 $\gamma$ 的方程 $t_{k-1} \gamma^2 = t_k\gamma_{k-1}^2(1-\gamma_k)$ 来动态调整 $\gamma_k$ 和 $t_k$, 可以在保证满足 line search 条件的同时, 使得步长 $t_k$ 不会过快地缩小, 从而提升算法的收敛效率. 

#### Serveral Remarks on FISTA

加速方法并不是适用于所有问题的. 
- 例如在 warm start 的情况下 (例如在求解 Lasso 路径问题 $\min_{\beta} \frac{1}{2} \|y - X\beta\|^2 + \lambda \|\beta\|_1$, 其中 $\lambda_1 > \lambda_2 > \ldots > \lambda_m > 0$ 是一系列递减的正则化参数), 由于每次迭代的初始点 $x^{(0)}$ 已经非常接近最优解, 因此加速方法可能会导致过度震荡, 从而反而降低收敛效率. 

- 另外还比如在矩阵补全问题中, prox 的计算涉及到奇异值分解(SVD), 其计算复杂度较高. 因此在类似 prox 会涉及到核范数, SVD, 或者其他高复杂度计算的情况下, 加速方法可能会导致每次迭代的计算成本过高, 从而降低整体的效率.
