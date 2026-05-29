# Poisson Process

## Introduction to Poisson Process

Poisson Process 是一个连续时间, 离散状态的随机过程, 用来描述在单位时间内发生某事件的次数, 或称为事件的计数过程. 

对于连续时间 $t \geq 0$, 设 $N(t)$ 表示在时间 $[0, t]$ 内发生的事件数. 最基本地, 我们希望 $N(t)$ 满足以下几个性质:
1. **独立增量 (Independent Increments)**: 对于任意的 $0 \leq t_1 < t_2  \leq t_3 < t_4$, 增量 $N(t_2) - N(t_1)$ 和 $N(t_4) - N(t_3)$ 是相互独立的.
2. **平稳增量 (Stationary Increments)**: 对于任意的 $s, t \geq 0$, 增量 $N(t+s) - N(s)$ 的分布仅依赖于跨度 $t$, 与具体时刻 $s$ 无关.
3. **稀疏性 (Sparsity)**: 在这里, 我们定义稀疏性为:
    $$
    \lim_{\Delta t \to 0} \frac{\mathbb{P}(N(\Delta t) \geq 2)}{\mathbb{P}(N(\Delta t) = 1)} = 0
    $$
    即在短时间间隔内, 发生两个及以上事件的概率相对于发生一个事件的概率是一个高阶小量.

下面我们希望能够根据上述的定性描述来描述清楚 $N(t)$ 的统计分布规律. 
- 引入随机变量 $X$ 的矩母函数 $G_X(z) = \mathbb{E}[z^{X}]$. 则计算 $N(t)$ 的矩母函数如下:
    $$
    G_{N(t)}(z) = \mathbb{E}[z^{N(t)}] = \sum_{k=0}^{\infty} \mathbb{P}(N(t) = k) z^k
    $$
- 构建增量在短时间隔内的矩母函数:
    $$
    \begin{align*}
    G_{N(t+\Delta t)}(z) - G_{N(t)}(z) 
    &= \mathbb{E}[z^{N(t)}(z^{N(t+\Delta t) - N(t)} - 1)]\\
    &= \mathbb{E}[z^{N(t)}]\mathbb{E}[z^{N(t+\Delta t) - N(t)}- 1] \quad \text{\small(根据独立性)}\\
    &= \mathbb{E}[z^{N(t)}]\left(\mathbb{E}[z^{N(\Delta t)}- 1] \right) \quad \text{\small (根据平稳增量性)}\\
    &= \mathbb{E}[z^{N(t)}]\left(\sum_{k=0}^{\infty} \mathbb{P}(N(\Delta t) = k) z^k - 1\right)\\
    &= \mathbb{E}[z^{N(t)}]\left(\mathbb{P}(N(\Delta t) = 0) z^0  + \mathbb{P}(N(\Delta t) = 1) z^1 + \sum_{k=2}^{\infty} \mathbb{P}(N(\Delta t) = k) z^k - 1\right)\\
    \end{align*}
    $$
- 具体考察 $N(\Delta t)$ 的概率分布. 
    $$
    \begin{align*}
    \mathbb{P}(N(t) = 0) 
    &= \mathbb{P}(N(s) = 0, N(t) - N(s) = 0) \quad (\forall s < t) \\
    &= \mathbb{P}(N(s) = 0) \cdot \mathbb{P}(N(t) - N(s) = 0) \\
    &= \mathbb{P}(N(s) = 0) \cdot \mathbb{P}(N(t-s) = 0) \\
    \end{align*}
    $$
  - 形式上, 我们得到的概率关系为:
      $$
      P(t) = P(s) \cdot P(t-s) \iff P(t+s) = P(t) \cdot P(s), ~\forall s < t
      $$
      其中 $P(t) := \mathbb{P}(N(t) = 0)$. 即 $P$ 是满足线性性的. 而根据分析性质, 符合该关系的唯一函数为指数函数, 即存在参数 $\lambda > 0$, 使得:
      $$
      P(t) = \exp(-\lambda t) \implies \mathbb{P}(N(\Delta t) = 0) = \exp(-\lambda \Delta t)
      $$

- 进一步利用稀疏性, 对高阶项进行分析.
    $$
    \begin{align*}
    \sum_{k=2}^{\infty} \mathbb{P}(N(\Delta t) = k) z^k 
    &= \mathbb{P}(N(\Delta t) = 1) \sum_{k=2}^{\infty} z^k \frac{\mathbb{P}(N(\Delta t) = k)}{\mathbb{P}(N(\Delta t) = 1)} \to 0.
    \end{align*}
    $$
    其中最后一个极限利用了 Z-Transform 的相关性质放缩得到.

- 根据概率的归一化, 得到:
    $$
    1 - \mathbb{P}(N(\Delta t) = 0) = \mathbb{P}(N(\Delta t) = 1) + \mathbb{P}(N(\Delta t) = 1) \frac{\mathbb{P}(N(\Delta t) \geq 2)}{\mathbb{P}(N(\Delta t) = 1)} 
    $$
    即:
    $$
    \frac{1}{\Delta t} \left[1 - \exp(-\lambda \Delta t)\right] = \frac{1}{\Delta t} \mathbb{P}(N(\Delta t) = 1) + o(\Delta t)
    $$
    令 $\Delta t \to 0$, 得到:
    $$
    \lim_{\Delta t \to 0} \frac{1}{\Delta t} \mathbb{P}(N(\Delta t) = 1) = \lambda. 
    $$
    
- 最终, 有:
    $$
    \frac{1}{\Delta t} \mathbb{E}[z^{N(\Delta t)} - 1] \to - \lambda + \lambda z
    $$
    即有 ODE 方程:
    $$
    \frac{\mathrm{d}}{\mathrm{d}t} G_{N(t)}(z) = G_{N(t)}(z) \lambda (z-1), \quad G_{N(0)}(z) = \mathbb{E}[z^{N(0)}] = 1.
    $$
    解得:
    $$
    G_{N(t)}(z) = \exp(\lambda (z-1) t), \quad G_{N(t)}(1) = \exp(\lambda t).
    $$
    即:
    $$
    \mathbb{P}(N(t) = k) = \frac{(\lambda t)^k}{k!} \exp(-\lambda t)
    $$

因此可以得到 Poisson Process 的正式定义. 

***Definition* (Poisson Process):** 对于一个计数过程 $\{N(t), t \geq 0\}$, 若对于给定 $\lambda > 0$, 满足以下条件:
1. $N(0) = 0$
2. $\{N(t), t \geq 0\}$ 具有独立增量
3. $\mathbb{P}(N(t+h) - N(t) = 1) = \lambda h + o(h)$
4. $\mathbb{P}(N(t+h) - N(t) \geq 2) = o(h)$
则称 $\{N(t), t \geq 0\}$ 为参数为 $\lambda$ 的 Poisson Process.


立刻有如下结论:
- $\mathbb{P}(N(t) = k) = \frac{(\lambda t)^k}{k!} \exp(-\lambda t)$
- $\mathbb{P}(N(t) - N(s) = k) = \frac{(\lambda (t-s))^k}{k!} \exp(-\lambda (t-s))$
- $\mathbb{E}[N(t)] = \lambda t \iff \lambda = \mathbb{E}[N(t)] / t$, 即单位时间间隔内事件发生的平均次数, 或事件发生的强度. 

## Properties of Poisson Process

1. **事件发生间隔服从指数分布:** 考虑第 $i$ 次事件发生的时间间隔 $T_i$. 对于第一次事件发生的时间 $T_1$, 有
    $$
    F_{T_1}(t) = \mathbb{P}(T_1 \leq t) = 1 - \mathbb{P}(T_1 > t) = 1 - \mathbb{P}(N(t) = 0) = 1 - \exp(-\lambda t)
    $$
    即 $T_1$ 服从参数为 $\lambda$ 的指数分布. **事实上, 所有间隔 $T_i$ 之间都是独立同分布的指数分布.** 
    - 指数分布是无记忆性的:
        $$
        \mathbb{P}(T_1 > t + s | T_1 > t) = \mathbb{P}(T_1 > s)
        $$

2. **事件的发生时刻服从 Gamma 分布:** 考虑第 $k$ 次事件发生的时间 $S_k := \sum_{i=1}^{k} T_i$. 则有:
    $$
    f_{S_k}(t) = \frac{\lambda(\lambda t)^{k-1}}{(k-1)!} \exp(-\lambda t)
    $$
    即 $S_k$ 服从参数为 $\lambda$ 和 $k$ 的 Gamma 分布. 

