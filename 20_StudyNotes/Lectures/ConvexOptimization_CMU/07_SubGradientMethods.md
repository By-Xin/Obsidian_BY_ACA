# Subgradient Methods

> - Lecture Reference: https://www.stat.cmu.edu/~ryantibs/convexopt-F18/
> - Reading Reference: https://stanford.edu/class/ee364b/lectures/subgrad_method_slides.pdf

## Introduction & Motivation

- 对于无约束凸优化问题 $\min_{x \in \mathbb{R}^n} f(x)$, 如果 $f$ 是可微的, 我们可以用梯度下降法: $x^{(k+1)} = x^{(k)} - t_k \nabla f(x^{(k)})$ 进行优化. 若 $\nabla f(x)$ 是 Lipschitz 连续的, 则可以保证其收敛速率为 $\mathcal{O}(1/k)$.  

- 当 $f$ 不可微时, 我们可以使用次梯度方法 (subgradient method) 来进行优化: 类似地, 首先确定初始点 $x^{(0)}$, 然后迭代更新: 
    $$x^{(k+1)} = x^{(k)} - t_k \cdot g^{(k)}, \quad g^{(k)} \in \partial f(x^{(k)})$$
    其中 $\partial f(x)$ 是 $f$ 在 $x$ 处的 subdifferential.

- 然而, 由于 subgradient 更新并不一定必然导致函数值的下降, 因此需要对每次的更新值进行追踪, 并记录迄今为止的最优值 $f(x_{best}) = \min_{0 \leq i \leq k} f(x^{(i)})$.

## Subgradient Algorithm 

### Step Size Selection

在更一般的非光滑优化问题中, 选择一个能同时保证收敛又能有效步进的步长往往是十分困难的, 因此往往是通过经验提前设定的, 常见的步长选择方法有:
- 固定步长: $t_k = t$.
- 固定步进: $t_k  = \gamma / \|g^{(k)}\|_2$, 其中 $\gamma$ 是一个常数, 这样保证每次更新的步长 $\|x^{(k+1)} - x^{(k)}\|_2 = \gamma$ 是固定的.
- 平方收敛步长: 使得步长满足 $\sum_{k=1}^\infty t_k = \infty$ 和 $\sum_{k=1}^\infty t_k^2 < \infty$, 例如 $t_k = \frac{1}{\sqrt{k}}$.
- 极限收敛步长: 使得步长满足 $\lim_{k \to \infty} t_k = 0$ 和 $\sum_{k=1}^\infty t_k = \infty$, 例如 $t_k = \frac{1}{k}$.

### Convergence Analysis

#### Assumptions

为分析方便, 额外添加如下假设:
- $f$ 是凸的, 且 $\text{dom}(f) = \mathbb{R}^n$. 
- 最优值是有限的 ($f^* = \min_{x \in \mathbb{R}^n} f(x) > -\infty$) 且可达的 (存在 $x^* \in \mathbb{R}^n$ 使得 $f(x^*) = f^*$).
- 初始点 $x^{(0)}$ 与最优点 $x^*$ 之间的距离是有限的, 即 $\|x^{(0)} - x^*\|_2 \leq R$.
- $f$ 是 $G$-Lipschitz 连续的, 即 $\|g\|_2 \leq G$ 对于所有 $g \in \partial f(x)$ 和 $x \in \mathbb{R}^n$ 都成立, 或等价地 $|f(x) - f(y)| \leq G \|x - y\|_2$ 对于所有 $x, y \in \mathbb{R}^n$ 都成立.
  - *Proof*: 下给出两种表述的等价性证明.
    - 由 $\|g\|_2 \leq G$ 推出 $|f(x) - f(y)| \leq G \|x - y\|_2$.
      - 由于 $g \in \partial f(x)$, 根据 subgradient 的定义, 对于任意 $y \in \mathbb{R}^n$, 都有 $f(y) \geq f(x) + g^\top (y - x)$, 从而 $f(x)-f(y) \leq g^\top (x - y)$
      - 由 Cauchy-Schwarz 不等式, $g^\top (x - y) \leq \|g\|_2 \|x - y\|_2 \leq G \|x - y\|_2$, 从而 $f(x)-f(y) \leq G \|x - y\|_2$. 同理由对称性, 我们也可以得到 $f(y)-f(x) \leq G \|y - x\|_2$, 从而 $|f(x) - f(y)| \leq G \|x - y\|_2$.
    - 由 $|f(x) - f(y)| \leq G \|x - y\|_2$ 推出 $\|g\|_2 \leq G$.
      - 由 subgradient 的定义, 取单位向量 $u$, 令 $y = x + tu$, $t>0$, 则 $f(x + tu) \geq f(x) + g^\top (tu)$, 从而 
        $$\frac{f(x + tu) - f(x)}{t} \geq g^\top u$$ 
      - 由 Lipschitz 连续性, $\frac{f(x + tu) - f(x)}{t} \leq G \|u\|_2 = G$. 因此 $g^\top u \leq G$ 对于任意单位向量 $u$ 都成立.
      - 根据事实: $\|g\|_2 = \sup_{\|u\|_2 = 1} g^\top u$, 可得 $\|g\|_2 \leq G$.
    $\square$

#### Basic Inequality & Convergence Analysis

首先给出如下基本不等式, 以便后续分析. 记 $f(x_\text{best}^{(k)}) = \min_{0 \leq i \leq k} f(x^{(i)})$ 为历史最优值, $R$ 为初始点与最优点之间的距离, $G$ 为 $f$ 的 Lipschitz 常数, $t_k$ 为第 $k$ 次迭代的步长, $f^* = f(x^*)$ 为最优值, 则对于任意 $k \geq 0$, 都有:
$$\boxed{
    f(x_\text{best}^{(k)}) - f(x^*) \leq \frac{R^2 + G^2 \sum_{i=0}^k t_i^2}{2 \sum_{i=0}^k t_i}
}$$
- *Proof*: 
  - 首先证明 $\|x^{(k+1)} - x^*\|_2^2 \leq \|x^{(k)} - x^*\|_2^2 - 2t_k [f(x^{(k)}) - f(x^*)] + t_k^2 \|g^{(k)}\|_2^2$.
    - 由 subgradient method 的更新公式, 有 $\|x^{(k+1)} - x^*\|_2^2 = \|x^{(k)} - t_k g^{(k)} - x^*\|_2^2 = \|x^{(k)} - x^*\|_2^2 - 2t_k (g^{(k)})^\top (x^{(k)} - x^*) + t_k^2 \|g^{(k)}\|_2^2$.
    - 由  subgradient 的定义, $f(x^*) \geq f(x^{(k)}) + (g^{(k)})^\top (x^* - x^{(k)})$, 从而 $(g^{(k)})^\top (x^{(k)} - x^*) \geq f(x^{(k)}) - f(x^*)$. 因此上不等式最终可以化为:
        $$\|x^{(k+1)} - x^*\|_2^2 \leq \|x^{(k)} - x^*\|_2^2 - 2t_k [f(x^{(k)}) - f(x^*)] + t_k^2 \|g^{(k)}\|_2^2$$
  - 接下来, 将上述不等式$^{(1)}$进行迭代展开 , 并根据 $^{(2)}$ $f(x_\text{best}^{(k)}) = \min_{0 \leq i \leq k} f(x^{(i)})$, 因此 $f(x^{(i)}) - f(x^*) \geq f(x_\text{best}^{(k)}) - f(x^*)$ 对于所有 $0 \leq i \leq k$ 都成立 ,以及$^{(3)}$ $\|x^{(0)} - x^*\|_2 \leq R$ 的假设, 可以得到:
    $$\begin{aligned}
        \|x^{(k+1)} - x^*\|_2^2 & \stackrel{(1)}{\leq} \|x^{(0)} - x^*\|_2^2 - 2 \sum_{i=0}^k t_i [f(x^{(i)}) - f(x^*)] + \sum_{i=0}^k t_i^2 \|g^{(i)}\|_2^2 \\
        & \stackrel{(2)}{\leq} \|x^{(0)} - x^*\|_2^2 - 2 [f(x_\text{best}^{(k)}) - f(x^*)] \sum_{i=0}^k t_i + \sum_{i=0}^k t_i^2 \|g^{(i)}\|_2^2 \\
        & \stackrel{(3)}{\leq} R^2 - 2 [f(x_\text{best}^{(k)}) - f(x^*)] \sum_{i=0}^k t_i + \sum_{i=0}^k t_i^2 \|g^{(i)}\|_2^2
    \end{aligned}$$
  - 进而整理得到:
    $$f(x_\text{best}^{(k)}) - f(x^*) \leq \frac{R^2 + \sum_{i=0}^k t_i^2 \|g^{(i)}\|_2^2}{2 \sum_{i=0}^k t_i}$$
  - 若进一步利用 $f$ 的 Lipschitz 连续性, 即 $\|g^{(i)}\|_2 \leq G$ 对于所有 $i$, 则可以得到:
    $$f(x_\text{best}^{(k)}) - f(x^*) \leq \frac{R^2 + G^2 \sum_{i=0}^k t_i^2}{2 \sum_{i=0}^k t_i}$$
    $\square$

- 对于不同的步长策略, 上不等式可以进一步进行化简整理.
  - 对于固定步长 $t_k = t$, 上式可以化简为:
    $$f(x_\text{best}^{(k)}) - f(x^*) \leq \frac{R^2+G^2 t^2 (k+1)}{2 t (k+1)}\stackrel{k\to\infty}{\longrightarrow} \frac{G^2 t}{2}$$
  - 对于固定步进 $t_k = \gamma / \|g^{(k)}\|_2$, 将其带入 Lipschitz 连续性化简前的不等式, 可以得到:
    $$\begin{aligned}
        f(x_\text{best}^{(k)}) - f(x^*) &\leq \frac{R^2 + \sum_{i=0}^k \left(\frac{\gamma}{\|g^{(i)}\|_2}\right)^2 \|g^{(i)}\|_2^2}{2 \sum_{i=0}^k \frac{\gamma}{\|g^{(i)}\|_2}} \\
        &= \frac{R^2 + (k+1) \gamma^2}{2 \gamma \sum_{i=0}^k 1/\|g^{(i)}\|_2} \\
        &\leq \frac{R^2 + (k+1) \gamma^2}{2 \gamma (k+1)/G}  \stackrel{k\to\infty}{\longrightarrow} \frac{G \gamma}{2}
    \end{aligned}$$
  - 对于平方收敛步长 $t: \sum t_k^2 < \infty$ 和 $\sum t_k = \infty$, 上式可以化简为:
    $$f(x_\text{best}^{(k)}) - f(x^*) \leq \frac{R^2 + G^2 \sum_{i=0}^k t_i^2}{2 \sum_{i=0}^k t_i} \stackrel{k\to\infty}{\longrightarrow} 0$$

- 总的而言, 其一般的收敛速率为 $\mathcal{O}(\frac{1}{\epsilon^2})$, 这一速率是非常缓慢的. 如果观测其收敛上界 $\frac{R^2 + G^2 \sum_{i=0}^k t_i^2}{2 \sum_{i=0}^k t_i}$, 由 Cauchy-Schwarz 不等式可以给出理论的最快收敛情况在等步长 $t_k = \frac{R}{G \sqrt{k+1}}$ 时达到, 此时上界为 $\frac{RG}{\sqrt{k}}$, 从而可以得到 $\mathcal{O}(\frac{1}{\epsilon^2})$ 的收敛速率. 此外, 这个停止条件依赖 $R,G,f^*$ 之类的全局常数, 现实里通常不知道, 而且即便知道, 它也只给出一个极其保守的最坏情况保证.

***Example***  考虑如下的非光滑优化问题:
$$\min f(x) = \max_{1 \leq i \leq m} a_i^\top x + b_i$$ 
对于该函数, 在之前的章节中提供了其 subgradient 的计算方法, 这里记之为 $g \in \partial f(x)$. 下分别考虑通过固定步进和几种衰退步长的 subgradient method 来进行优化, 其迭代更新公式如下:
- 固定步进: $x^{(k+1)} = x^{(k)} - \frac{\gamma}{\|g^{(k)}\|_2} g^{(k)}$, 分别取 $\gamma = 0.05, 0.01, 0.005$.
  - 观察到初期下降速度很快, 但是后面会进入平台期. 并且根据 $\gamma$ 的不同, 平台期的水平也不同. 由于 $\lim\sup_{k\to\infty}\left(f^{(k)}_{\text{best}}-f^*\right)\le \frac{G\gamma}{2},$ 因此 $\gamma$ 越小, 平台期的水平越低, 但下降速度也越慢.
    ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260209162612.png)
- 衰退步长: $x^{(k+1)} = x^{(k)} - \alpha_k g^{(k)}$, 分别取 $\alpha_k = 0.1/\sqrt{k}, 1/\sqrt{k}, 1/k, 10/k$.
  - 观察到不同常数系数差异巨大: 系数过大前期抖动更明显, 过小则整体太慢. 这也说明 subgradient 对步长非常敏感, 调参是主要成本. 此外, 衰减型相比于固定步进, 下降速度更慢, 但没有明显的平台期, 这也符合理论分析.
   ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260209162644.png)


#### Polyak: Optimal Step Size when $f^*$ is Known

在理想分析中, $f^*$ 是已知的, 因此可以选择如下的 Polyak 步长:
$$t_{k+1} = \frac{f(x^{(k)}) - f^*}{\|g^{(k)}\|_2^2}, \quad k=0,1,2,\cdots$$

- Polyak 的更新思想很简单: 利用当前的函数值和最优值之间的差距来动态调整步长, 使得每次迭代都能最大程度地减少当前点和最优点之间的距离. 然而, 由于 $f^*$ 在实际问题中通常是未知的, 因此 Polyak 的步长虽然在理论上是最优的, 但在实践中很难直接使用. 

- 回顾在之前的基本不等式中, 利用 subgradient 的定义, 可以得到:
    $$\|x^{(k+1)} - x^*\|_2^2 \leq \|x^{(k)} - x^*\|_2^2 - 2t_k [f(x^{(k)}) - f(x^*)] + t_k^2 \|g^{(k)}\|_2^2$$
    而 Polyak 的步长恰好为使得 RHS 最小的步长, 即其最大程度控制了下一次迭代位置和最优点之间的距离. 

- 将该最优步长代入, 可得不等式:
    $$\|x^{(k+1)} - x^*\|_2^2 \leq \|x^{(k)} - x^*\|_2^2 - \frac{[f(x^{(k)}) - f(x^*)]^2}{\|g^{(k)}\|_2^2}$$
    - 该式立刻保证 $\|x^{(k+1)} - x^*\|_2 \leq \|x^{(k)} - x^*\|_2$, 从而保证了每次迭代都不会使得当前点和最优点之间的距离增加.
    - 令 $k = 0,1,2,\cdots$, 将上述不等式进行迭代展开, 可以得到:
        $$\sum_{i=0}^{k}\frac{[f(x^{(i)}) - f(x^*)]^2}{\|g^{(i)}\|_2^2} \leq \|x^{(0)} - x^*\|_2^2 $$
    - 再分别利用 $f$ 的 Lipschitz 连续性 $\|g^{(i)}\|_2 \leq G$ 和 $\|x^{(0)} - x^*\|_2 \leq R$ 的假设, 可以得到:
        $$\sum_{i=0}^{k} [f(x^{(i)}) - f(x^*)]^2 \leq G^2 R^2$$
        即证 $f(x^{(k)}) \stackrel{k\to\infty}{\longrightarrow} f(x^*)$.

在实践中, 有时可以通过 Estimated Polyak 步长来近似 Polyak 步长. 具体地, 用目前为止观测到的最优值 $f(x_\text{best}^{(k)})$ 来近似 $f^*$:
$$f(x_\text{best}^{(k)}) = \min_{0 \leq i \leq k} f(x^{(i)}) \implies \hat{f}^*_k := f(x_\text{best}^{(k)}) - \gamma_k.$$ 
- 其中 $\gamma_k$ 是一个小的正数避免止步长为零. $\gamma_k$ 往往也会选择满足 Robbins-Monro 条件的衰减步长 (即 $\sum_{k=0}^\infty \gamma_k = \infty$ 和 $\sum_{k=0}^\infty \gamma_k^2 < \infty$), 以保证其在理论上能够收敛到最优值, 例如 $\gamma_k = \frac{c}{k+1}$.
- 可以证明, 估计的历史最优值会逐渐逼近最优值. 

#### General Convergence Result

对于一般的一阶非光滑算法:
$$x^{(k)} \in x^{(0)} + \mathrm{span}\{g^{(0)}, g^{(1)}, \dots, g^{(k-1)}\},$$
对于 weak oracle (查询一次, 只能得到函数值 $f(x)$ 和一阶 $\partial f(x)$ 这类有限信息, 没有更强的二阶信息或全局结构), 此时有定理保证对于任何 $k\leq n-1$ 及任意初始点 $x^{(0)}$, 都存在一个目标函数使得 
$$f(x^{(k)}) - f^* \geq \frac{G R}{2(1+\sqrt{k})}$$
- 该定理表明, 在最坏的情况下, 任何一阶非光滑优化算法在 $k$ 次迭代后都无法保证其函数值与最优值之间的差距小于 $\frac{G R}{2(1+\sqrt{k})}$. 这也说明了 subgradient method 的 $\mathcal{O}(\frac{1}{\sqrt{k}})$ 的收敛速率是最优的.

不过, 该定理实在说明在一般情况下的收敛水平. 对于具有特殊结构的非光滑优化问题, 可能存在更快的算法. 例如考虑如下的 composite 结构:
$$\min_x f(x) = g(x) + h(x)$$   
- $g$ 是一个凸且可微 (通常还进一步假设其梯度是 Lipschitz 连续的) 的函数
- $h$ 是一个凸但不可微的函数, 但其相对容易计算的
此时可以通过利用这些特殊结构将效率提升为 $\mathcal{O}(\frac{1}{k})$, 例如 Proximal Gradient Method.

### Further Applications of Subgradient Methods

#### Alternating Projections

考虑求解凸集交的问题. 这里将说明此类问题的对应解决方法 (即 alternating projections) 是 subgradient method 的一个特殊实例.

- 问题的叙述为: 
    $$\text{find } x \in \mathcal{C} = \bigcap_{i=1}^m \mathcal{C}_i$$
    其中 $\mathcal{C}_i$ 是 $\mathbb{R}^n$ 中的凸集.

- 可通过如下方法将问题转化为一个非光滑优化问题:
  - 定义投影点 $\text{Proj}_{\mathcal{C}_i}(x):=\arg\min_{y\in\mathcal{C}_i}\|x-y\|_2$. 定义点 $x$ 到集合 $\mathcal{C}_i$ 的距离为 $f_i(x) = \text{dist}(x, \mathcal{C}_i) := \min_{y \in \mathcal{C}_i} \|x - y\|_2 = \|x - \text{Proj}_{\mathcal{C}_i}(x)\|_2$.
  - 注意到 $x \in \mathcal{C}$ 当且仅当 $f_i(x) = 0$ 对于所有 $i$ 都成立, 因此可以将原问题转化为如下的非光滑优化问题:
     $$\min_x f(x) = \max_{1 \leq i \leq m} f_i(x)$$

- 由于 $f_i$ 是 convex 的, 因此 $f$ 也是 convex 的. 因此可以通过 subgradient method 来进行优化.
  - 计算每个分量的 subgradient: 对于集合 $\mathcal{C}_j$ 及点 $x$, 其距离为 $d_j(x):=\mathrm{dist}(x,\mathcal{C}_j)=\|x-\text{Proj}_{\mathcal{C}_j}(x)\|_2$. 
    - 当 $x\notin \mathcal{C}_j$ 时, $d_j(x) > 0$, 此时 $g_j = \frac{x-\text{Proj}_{\mathcal{C}_j}(x)}{\|x-\text{Proj}_{\mathcal{C}_j}(x)\|_2}$ 是 $f_j$ 在 $x$ 处的 subgradient. 
    - 当 $x \in \mathcal{C}_j$ 时, $d_j(x) = 0$, 此时 $g_j = 0$ 是 $f_j$ 在 $x$ 处的 subgradient. 
  - 因此对于 $f(x) = \max_{1 \leq i \leq m} f_i(x)$, 记 $I(x) = \{i: f_i(x) = f(x)\}$ 为最大值对应的索引集合, 则 
        $$\partial f(x) = \text{conv}\left(\bigcup_{i \in I(x)} \partial f_i(x)\right)$$
- 进而 subgradient method 的更新公式为:
    $$\begin{aligned}
    x^{(k+1)}
    &=x^{(k)}-f(x^{(k)})\frac{x^{(k)}-\text{Proj}_{\mathcal{C}_j}(x^{(k)})}{\|x^{(k)}-\text{Proj}_{\mathcal{C}_j}(x^{(k)})\|_2}\\
    &=x^{(k)}-\|x^{(k)}-\text{Proj}_{\mathcal{C}_j}(x^{(k)})\|_2\frac{x^{(k)}-\text{Proj}_{\mathcal{C}_j}(x^{(k)})}{\|x^{(k)}-\text{Proj}_{\mathcal{C}_j}(x^{(k)})\|_2}\\
    &=x^{(k)}-\big(x^{(k)}-\text{Proj}_{\mathcal{C}_j}(x^{(k)})\big)\\
    &=\text{Proj}_{\mathcal{C}_j}(x^{(k)}).
    \end{aligned}$$

因此, 在每次迭代中, subgradient method 都会将当前点 $x^{(k)}$ 投影到距离其最远的集合 $\mathcal{C}_j$ 上. 这就是 alternating projections 的核心思想. 通过不断地交替投影, 可以逐渐逼近交集 $\mathcal{C}$ 中的一个点.

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260209185847.png)


#### Projected Subgradient Method

考虑非光滑凸有约束优化问题:
$$\min_x f(x) \quad \text{s.t. } x \in C,$$

由于约束条件的存在, 直接进行 subgradient method 的更新可能会导致迭代点 $x^{(k)}$ 不满足约束条件. 因此, 可以通过在每次迭代后进行投影来保证迭代点始终满足约束条件. 具体地, 其更新公式为:
$$x^{(k+1)} = \text{Proj}_C\left(x^{(k)} - t_k g^{(k)}\right), \quad g^{(k)} \in \partial f(x^{(k)})$$
其中 $\text{Proj}_C(z) = \arg\min_{y \in C} \|y - z\|_2$ 是将点 $z$ 投影到集合 $C$ 上的操作.
- 只要这个投影是便于计算的, 那么 projected subgradient method 就是一个非常实用的算法. 其步长的选择规则等也与之前的无约束情况类似, 例如固定步长、衰减步长等.

