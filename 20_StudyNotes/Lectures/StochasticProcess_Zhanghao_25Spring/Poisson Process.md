# Poisson Process

## Introduction to Poisson Process

Poisson Process 是一个连续时间, 离散状态的随机过程, 用来描述在单位时间内发生某事件的次数, 或称为事件的计数过程. 

对于连续时间 $t \geq 0$, 设 $N(t)$ 表示在时间 $[0, t]$ 内发生的事件数. 最基本地, 我们希望 $N(t)$ 满足以下两个性质:
1. **独立增量 (Independent Increments)**: 对于任意的 $0 \leq t_1 < t_2  \leq t_3 < t_4$, 增量 $N(t_2) - N(t_1)$ 和 $N(t_4) - N(t_3)$ 是相互独立的.
2. **平稳增量 (Stationary Increments)**: 对于任意的 $s, t \geq 0$, 增量 $N(t+s) - N(s)$ 的分布仅依赖于跨度 $t$, 与具体时刻 $s$ 无关.

下尝试定义一个满足上述性质的随机过程.
- 首先引入 Moment Generating Function (MGF) 的概念. 对于一个随机变量 $X$, 其 MGF 定义为
    $$
    M_X(\theta) = \mathbb{E}[e^{\theta X}]
    $$
- 下尝试使用 MGF 来定义 $N(t)$ 的分布并根据独立增量和平稳增量的性质来建立微分方程. 
  - 首先考虑差分:
    $$
    \begin{aligned}
    \frac{1}{\Delta t} \left( M_{N(t+\Delta t)}(\theta) - M_{N(t)}(\theta) \right) &= \frac{1}{\Delta t} \left( \mathbb{E}[e^{\theta N(t+\Delta t)}] - \mathbb{E}[e^{\theta N(t)}] \right) \\
    &= \frac{1}{\Delta t} \left( \mathbb{E}[e^{\theta (N(t) + (N(t+\Delta t) - N(t)))}] - \mathbb{E}[e^{\theta N(t)}] \right) \\
    &= \frac{1}{\Delta t} \left( \mathbb{E}[e^{\theta N(t)}] \mathbb{E}[e^{\theta (N(t+\Delta t) - N(t))}] - \mathbb{E}[e^{\theta N(t)}] \right) \quad \text{\small(由独立增量)} \\
    &= M_{N(t)}(\theta) \cdot \frac{1}{\Delta t} \left( M_{N(\Delta t)}(\theta) - 1 \right) \quad \text{\small(由平稳增量)} \\
    \end{aligned}
    $$
  - 下仔细考虑 $M_{N(\Delta t)}(\theta) - 1$ 的行为.
    $$
    \begin{aligned}
    M_{N(\Delta t)}(\theta) - 1 &= \mathbb{E}[e^{\theta N(\Delta t)}] - 1 \\
    &= \sum_{k=0}^{\infty} e^{\theta k} \cdot \mathbb{P}(N(\Delta t) = k) - 1 \\
    &= \mathbb{P} (N(\Delta t) = 0) - 1  + e^{\theta} \mathbb{P}(N(\Delta t) = 1) + \sum_{k=2}^{\infty} e^{\theta k} \mathbb{P}(N(\Delta t) = k) 
    \end{aligned}
    $$
    - 而其中, 对于 $s \in (0, \Delta t)$, $N(\Delta t) = 0$ 当且仅当 $N(s) = 0$ 且 $N(\Delta t) - N(s) = 0$. 因此,
        $$
        \begin{aligned}
        \mathbb{P}(N(\Delta t) = 0) &= \mathbb{P}(N(s) = 0, N(\Delta t) - N(s) = 0) \\
        &= \mathbb{P}(N(s) = 0) \cdot \mathbb{P}(N(\Delta t) - N(s) = 0) \quad \text{\small(由独立增量)} \\
        &= \mathbb{P}(N(s) = 0) \cdot \mathbb{P}(N(\Delta t - s) = 0) \quad \text{\small(由平稳增量)} \\
        \end{aligned}
        $$ 
        上述分析表明, 若令 $f(t) = \mathbb{P}(N(t) = 0)$, 则 $f$ 满足:
        $$
        f(t+s) = f(t) \cdot f(s)
        $$
        可以证明, 满足上述函数方程的函数 $f$ 必然是指数函数. 因此, 存在 $\lambda > 0$, 使得
        $$
        f(t) = e^{-\lambda t}
        $$
        - *Proof*.  
          - 首先说明 $f(t) > 0$. 根据构造, $f(t) = (f(\frac{t}{2}))^2$ 表明 $f(t) \geq 0$. 若存在 $t_0$ 使得 $f(t_0) = 0$, 则 $f(\frac{t_0}{2}) = 0$, 以此类推, 可得 $f(\frac{t_0}{2^n}) = 0$ 对任意 $n \in \mathbb{N}$ 成立. 因此, 由连续性可得 $\lim_{n \to \infty} f(\frac{t_0}{2^n}) = f(0) = 0$. 因此当且仅当 $f \equiv 0$ 时, $f(t)$ 才可能取到 $0$. 否则, $f(t) > 0$ 对任意 $t \geq 0$ 成立. 
          - 接着说明 $f$ 是指数函数. 由于 $f(t) > 0$, 可定义 $g(t) = \ln f(t)$. 则 $g$ 满足
            $$
            g(t+s) = g(t) + g(s), \quad \forall t, s \in \mathbb{R}
            $$
            若 $t = 0$, 则 $g(0) = g(0) + g(0)$, 从而 $g(0) = 0$.  若 $t \in \mathbb{Z}_+$, 则 $g(t) = g(t-1) + g(1) = \cdots = t g(1)$. 这可以看作是以 $g(1)$ 为斜率的线性函数在正整数上的取值. 若 $t \in \mathbb{Z}_-$, 则 $g(t) = g(0) - g(-t) = 0 - (-t) g(1) = t g(1)$, 这也与上述形式一致. 若 $t \in \mathbb{Q}$, 令 $t = \frac{m}{n}$, 则 $g(1) = n g(\frac{1}{n})$ 从而 $g(1/n) = g(1)/n$. 因此, $g(t) = g(m/n) = m g(1/n) = m g(1)/n = t g(1)$. 若 $t \in \mathbb{R}$, 根据实数理论, 存在 $\{q_n\} \subset \mathbb{Q}$ 使得 $\lim_{n \to \infty} q_n = t$. 因此, $g(t) = g(\lim_{n \to \infty} q_n) = \lim_{n \to \infty} g(q_n) = \lim_{n \to \infty} q_n g(1) = t g(1)$ (其中由连续性保证了极限与函数值的交换).  