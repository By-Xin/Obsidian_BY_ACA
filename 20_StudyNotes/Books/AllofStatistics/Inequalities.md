---
aliases: [不等式, Concentration Inequalities, 集中不等式, Markov不等式, Chebyshev不等式, Hoeffding不等式]
tags:
  - concept
  - math/probability
  - ml/stats
source: "All of Statistics, Larry Wasserman"
related_concepts:
  - "[[Probability Theory]]"
  - "[[Tail Bounds]]"
---

# Inequalities (Concentrations)

> Ref: All of Statistics by Larry Wasserman

- 统计学中的一个基础的话题就叫作 concentration, 其研究的核心问题就是随机变量在其期望值附近"集中"的程度如何, 换言之, 随机变量偏离其期望的尾部概率有多大. 
- Concentration Inequalities 可以认为是一系列从弱到强的不等式, 通过进行更多的假设/给定信息, 我们可以逐渐收紧不等式, 从而得到更精确的尾部概率估计.

## The Polynomial Era

### Markov's Inequality 

首先是 Markov's Inequality, 它是后续不等式的基础.

***THEOREM:*** 令 $X$ 为非负随机变量且期望 $\mathbb{E}[X]$ 存在. 则对任意 $\epsilon>0$,
$$
\mathbb{P}[X\geq \epsilon]\leq \frac{\mathbb{E}[X]}{\epsilon}
$$

**Proof**

$$\mathbb{E}(X) = \int_{0}^\infty x f(x) \mathrm{d}x = \int_{0}^\epsilon f(x)x \mathrm{d}x + \int_{\epsilon}^\infty f(x)x\mathrm{d}x \ge \epsilon\int_{\epsilon}^\infty f(x)\mathrm{d}x \ge \epsilon\cdot\mathbb{P}[X\geq \epsilon].$$

$\square$

**Remark:**
- 对于非负随机变量, 尾部概率受期望值约束. 
    - 很直观的例子, 如果平均分只有 60 分, 那就不可能全班 90% 的人都考 100 分.
- 衰减速度为 $\mathcal{O}(\frac{1}{\epsilon})$ (线性衰减).

### Chebyshev's Inequality

Chebyshev's  Inequality 可以看作是 Markov's Inequality 的推广, 它对随机变量的分布没有限制, 但引入了二阶矩的信息. 其适用于所有方差有界的情况.  

***THEOREM:*** 对于任意随机变量 $X$, 令期望方差均存在, 且分别记为 $\mu = \mathbb{E}[X], \sigma^2 = \mathbb{Var}[X]$, 则对任意 $\epsilon>0$,
$$
\mathbb{P}\big(|X-\mu|\geq \epsilon\big)\leq \frac{\sigma^2}{\epsilon^2}
$$

- *Proof*
    $$
    \frac{\sigma^2}{\epsilon^2}  = \frac{\mathbb{E}\big[(X-\mu)^2\big]}{\epsilon^2} \stackrel{\text{Markov Ineq.}}{\ge} \mathbb{P}\big((X-\mu)^2\geq \epsilon^2\big) = \mathbb{P}\big(|X-\mu|\geq \epsilon\big).
    $$

$\square$

**Remark:**
- 由于引入了方差信息, Chebyshev's Inequality 的衰减速度为 $\mathcal{O}(\frac{1}{\epsilon^2})$ (二次衰减).
- 这个不等式也是自然的, 因为方差越大,  数据的分布越分散, 其定义的尾部概率上界就越松; 反之则越紧.

上述不等式也可以等价的写作如下标准化形式 (令 $k = \frac{\epsilon}{\sigma}$):
$$
\mathbb{P}\big(|X-\mu|\geq k\sigma\big)\leq \frac{1}{k^2}
$$
**Remark:**
- 若取 $k=2,3,10$, 则相当于给出了在任意分布下的 $2\sigma, 3\sigma, 10\sigma$ 原则, 其表明偏离中心 $\sigma, 2\sigma, 3\sigma$ 的概率分别小于 $1/4, 1/9, 1/100$.
- 一般而言, 也表明任意随机变量偏离均值的 $k$ 个标准差的概率小于 $\frac{1}{k^2}$.


***COROLLARY (Weak Law of Large Numbers):*** 统计推断中, $X_1, \ldots, X_n$ 为 i.i.d. 样本, 假设 $\mu = \mathbb{E}[X_i], \sigma^2 = \mathbb{Var}[X_i]$ 均存在. 故样本均值  $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ 满足: $\mathbb{E}[\bar{X}] = \mu$ 且 $\mathbb{Var}[\bar{X}] = \frac{\sigma^2}{n}$. 故根据 Chebyshev's Inequality, 对于任意 $\epsilon>0$, 
$$
\mathbb{P}\big(|\bar{X}-\mu|\geq \epsilon\big)\leq \frac{\sigma^2}{n\epsilon^2}
$$
**Remark:**
- 此即弱大数定律  (WLLN) 的有限样本版本. 

## The Exponential Era

### Mill's Inequality

Mill's Inequality 是后续不等式的一个 baseline 水平, 其提供了标准正态分布的尾部厚度. 后续的一系列不等式很多都是与之对比的.

***THEOREM (Mill's Inequality):*** 设 $X\sim \mathcal{N}(0,1)$, 则
$$
\mathbb{P}\left(|X|> t\right)\leq \sqrt{\frac{2}{\pi}}\frac{\exp(-t^2/2)}{t}\sim\frac{\exp(-t^2/2)}{t}
$$
- *Proof*
    - 给定 $\phi_X(x) = \frac{1}{\sqrt{2\pi}}\exp(-x^2/2)$, 则
    $$\begin{aligned}
    \mathbb{P}\left(|X|> t\right) &= \frac{2}{\sqrt{2\pi}}\int_{t}^\infty \exp\left(-\frac{x^2}{2}\right)\mathrm{d}x \\&= \sqrt{\frac{2}{\pi}}\frac{1}{t}\int_{t}^\infty t\exp\left(-\frac{x^2}{2}\right)\mathrm{d}x \\ &\leq \sqrt{\frac{2}{\pi}}\frac{1}{t}\int_{t}^\infty x\exp\left(-\frac{x^2}{2}\right)\mathrm{d}x \\ &= \sqrt{\frac{2}{\pi}}\frac{1}{t}\int_{t^2/2}^\infty \exp(-u)\mathrm{d}u \\ &= \sqrt{\frac{2}{\pi}}\frac{\exp(-t^2/2)}{t}
    \end{aligned}$$

$\square$
    

### Chernoff Inequality and Chernoff Method

从 Markov (一阶矩) 到 Chebyshev (二阶矩) Inequality, 我们通过引入更多的信息不断收紧不等式.  因此一个启发是我们是否能够继续推广到更高阶矩, 或者说所有矩的信息? Chernoff Bound 从这一 motivation 出发, 通过引入矩母函数 (MGF) 的信息, 得到了一个更一般的不等式.

***THEOREM (Chernoff Bound):***  设 $X$ 是随机变量. 对于任意 $\epsilon\in\mathbb{R}$ 和 $\lambda > 0$,  有:
$$
\mathbb{P}\big(X\ge \epsilon\big)\leq \inf_{\lambda>0} \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda \epsilon}} = e^{-\lambda \epsilon}  M_X(\epsilon),
$$

其中 $M_X(\epsilon)$是 MGF. 类似地, 对于 $\lambda < 0$, 有:
$$
\mathbb{P}\big(X\le \epsilon\big)\leq \inf_{\lambda<0} \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda \epsilon}} = e^{-\lambda \epsilon}  M_X(\epsilon).
$$

有时为了方便确定 RHS 的  $\inf$, 往往会使用其对数的等价形式 (理由和 MLE 中的对数似然类似):
$$
\log \mathbb{P}\big(X\ge \epsilon\big)   \leq  \inf_{\lambda>0} \big(\log \mathbb{E}[e^{\lambda X}] - \lambda \epsilon\big) := \inf_{\lambda >0}  \big(\psi_X(\lambda) - \lambda \epsilon\big).
$$

- *Proof*

    $$
    \mathbb{P}\big(X\ge \epsilon\big) \stackrel{\lambda>0}{=} \mathbb{P}\big(\lambda X \ge \lambda \epsilon\big) = \mathbb{P}\big(e^{\lambda X} \ge e^{\lambda \epsilon}\big)\stackrel{\text{Markov}}{\leq} \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda \epsilon}} ~(\forall \lambda>0)
    $$
    由 $\lambda >  0$ 的任意性, $\mathbb{P}\big(e^{\lambda X} \ge e^{\lambda \epsilon}\big) \leq \inf_{\lambda>0} \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda \epsilon}}$.

$\square$

**Remark:**
- 在具体问题中, 往往通过求解 $\frac{\mathrm{d}}{\mathrm{d}\lambda} \big[\psi_X(\lambda) - \lambda \epsilon\big] = 0$ 来确定 $\inf$ 对应的 $\lambda^*$.
- Chernoff 将衰减速度增加到了指数级别 $\mathcal{O}(e^{-cn})$

***Methodology (Chernoff Method):*** 事实上, Chernoff Inequality 证明过程的核心思想 (把难算的尾部概率转化为好算的凸优化问题) 在概率问题中可以被抽象成一套通用的方法论, 即 Chernoff Method, 其核心步骤分三步:
1. **Exponentialize:** 如果 $\mathbb{P}(X\ge \epsilon)$ 难算, 则考虑引入一个参数 $\lambda > 0$, 根据指数函数的单调性将事件转为 $\mathbb{P}(e^{\lambda X}\ge e^{\lambda \epsilon})$.
2. **Markov's Inequality:** 利用 Markov Inequality, 有 $\mathbb{P}(e^{\lambda X}\ge e^{\lambda \epsilon})\leq \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda \epsilon}}$. *在这一步, 我们利用 Markov Inequality 引入了期望, 因此指数+参数+期望, 我们就可以将难以计算的概率转化为了相对容易计算的 MGF.*.
3. **Optimization:** 通过优化 RHS 的 $\inf$ 来得到 Chernoff Bound. 这是这个方法最精髓的一步. 既然上述的不等式对于引入的任意 $\lambda>0$ 均成立, 那我们就可以通过调节 $\lambda$ 来得到最紧的上界 ($\inf$).


***EXAMPLE (Chernoff  Bound of Normal Distribution)***  设 $X\sim\mathcal{N}(\mu,\sigma^2)$,  则  $\mathbb{P}(X\ge \epsilon)$ 的 Chernoff Bound 求解如下. 

- 已知正态分布的 MGF 为 $M_X(\lambda) = \exp[\mu\lambda+\sigma^2\lambda^2/2].$ 故 $\psi_X(\lambda) = \mu\lambda+\sigma^2\lambda^2/2.$
- 对对数部分 RHS 求导并令其为 $0$, 有 $\mu+ \sigma^2\lambda = \epsilon$, 从而 $\lambda^* = \frac{\epsilon-\mu}{\sigma^2}$.
- 由 Chernoff Bound 知, 
    $$\mathbb{P}(X \geq \epsilon) \leq \inf_{\lambda > 0} e^{\mu\lambda + \frac{\sigma^2\lambda^2}{2} - \lambda \epsilon} =  e^{-\frac{(\epsilon-\mu)^2}{2\sigma^2}}.$$
    
***PROPOSITION $\star$ (Chernoff Bound of Binomial Distribution)*** 对于 $n$ 重 Bernoulli 实验 $X\sim\text{Binomial}(n,p)$ (即 $X = \sum_{i=1}^n X_i, X_i\sim \text{Bernoulli(p)}$), 其 Chernoff Bound 为
$$\mathbb{P}(X\ge k) \leq \exp\left[-n \cdot D_{KL}\left( \frac{k}{n} \| p\right)\right].$$
其中  $D_{KL}(q \| p) = q \log \frac{q}{p} + (1-q) \log \frac{1-q}{1-p}$ 恰为 Kullback-Leibler (KL)  散度 (相对熵).

- *Proof*
    - 对于 $X_i \sim \text{Bernoulli(p)}$, 其 MGF 为 $M_{X_i}(\lambda) = \mathbb{E}[e^{\lambda X_i}] = p e^{\lambda} + (1-p)$ ($\forall \lambda >0$).
    - 根据 $X_i$ 的独立性, 由 MGF 的性质 (独立的随机变量和的 MGF 相当于各自 MGF 的乘积),  有:
    $$
    M_X(\lambda) = \mathbb{E}\left[e^{\lambda X}\right] = \mathbb{E}\left[\prod_{i=1}^n e^{\lambda X_i}\right] = \prod_{i=1}^n \mathbb{E}\left[e^{\lambda X_i}\right] = \prod_{i=1}^n M_{X_i}(\lambda) = \left[p e^{\lambda} + (1-p)\right]^n.
    $$
    - 故 $\psi_X(\lambda) = \log M_X(\lambda) = n\log[p e^{\lambda} + (1-p)].$
    - 构造 Chernoff Inequality, $\forall k$:
        $$\mathbb{P}(X \geq k) \leq \inf_{\lambda > 0} \exp\{n\log[p e^\lambda + (1-p)] - \lambda k\}$$
    - 令 $\mathrm{d}\psi_X / \mathrm{d}\lambda^* = k$, 可解 (其中 $q= k/n$):
        $$\lambda^* = \log \frac{k/n(1-p)}{(1-k/n)p} := \log \frac{q (1-p)}{(1-q)p}
        $$
    - 将 $\lambda^*$  代入 Chernoff Inequality, 经过化简整理, 有:
        $$\mathbb{P}(X\ge k)\leq\exp\left[-n\left(q \log \frac{q}{p} + (1-q) \log \frac{1-q}{1-p}\right)\right] := \exp\left[-nD_{KL}\left( \frac{k}{n} \| p\right)\right].$$

- **Remark:**
    - KL 散度相当于两个分布的比较. 这在当前 Binomial 的语境下是非常明显的. $P = \text{Bernoulli}(p)$ 是期望分布, 而 $Q = \text{Bernoulli}(q), q= k/n$ 是实际观察得到的分布. 因此 $D_{KL}(q\| p)$ 就是在衡量 *我们实际观察到的比例 $q$ 而不是期望 $p$ 的 "意外程度"*.
    - KL  散度还有等价定义为 **似然比的对数期望**: $D(Q \| P) = \mathbb{E}_Q\left[\log \frac{Q}{P}\right]$. 这也衡量了 $Q$ 分布相比  $P$ 分布的平均"意外"程度.
    - 从信息论角度, KL 散度作为 relative entropy, 还可以理解为如果我们用按照 $P$ 设计的编码去 encode 来自 $Q$ 分布的数据, 我们需要的额外信息量恰为 $n\cdot D(q\| p)$.

***COROLLARY:*** 该不等式还可以理解为,  对于独立同分布的 $X_1,...,X_n \sim \text{Bernoulli}(p)$, 其均值 $\bar{X}$ 大于某个阈值 $q, (q>p)$ 的概率上界:
$$\mathbb{P}(\bar{X} \ge q) \le e^{-n \cdot D_{KL}(q || p)}.$$

- 特别地, 令 $k = (1+\delta)np ~(\delta>0)$  (即 $q=(1+\delta)p$) , 则该不等式还可以进一步简化为:
$$\mathbb{P}(X \geq (1+\delta)np) \leq \left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^{np}.$$

#### Chernoff Bound 与 Legendre-Fenchel 变换与 Large Deviation Theory *

在凸分析中, Legendre-Fenchel 变换 (或 convex conjugate) 是一种重要的对偶变换. 给定函数 $f: \mathbb{R}^d \to \mathbb{R}\cup\{+\infty\}$, 其 Legendre-Fenchel 变换定义为:
$$f^*(y) = \sup_{x\in\mathbb{R}^d} \{\langle x, y\rangle - f(x)\}.$$
其中 $x,y \in \mathbb{R}^d, \langle x, y\rangle$ 表示内积. 其具有如下关键性质:
1. 自对偶性 (Fenchel-Moreau 定理): 若 $f$ 是凸且下半连续的, 则 $f^{**}(x) =  f(x)$.
2. 凸性: $f^*$ 总是凸的, 无论 $f$ 是否是凸的. 
3. Young Inequatlity: $\forall x,y \in \mathbb{R}^d$, $\langle x, y\rangle \leq f(x) + f^*(y)$. 当且仅当 $y\in\partial f(x)$ 时取等.
4. 若 $f$ 可微且严格凸, 则 $y = \nabla f(x) \Leftrightarrow x = \nabla f^*(y)$ 且 $f^*(\nabla f(x)) = x^\top \nabla f(x) - f(x)$.

回顾, 我们在处理 Chernoff Bound 的时候, 引入了对数矩母函数 $\psi(t) = \log M(t) = \log \mathbb{E}[e^{tX}]$, 其也称为 Cumulant Generating  Function, CGF. 其有如下良好性质:
1. Convexity: $\psi(t)$ 在非退化场景下是严格凸函数;
2. 矩生成: $\psi'(0) = \mathbb{E}[X] = \mu, \psi''(0)  = \mathbb{Var}[X]$.
3. Exponential Family:  在 GLM 中, $\psi(t)$ 还是决定分布性质的那个函数 (log-partition function).


并且我们最终得到的 Chernoff Bound 为:
$$\log\mathbb{P}(X\ge t) \leq \inf_{\lambda>0} [\psi(\lambda)- \lambda t] = -\sup_{\lambda>0}[\lambda t-\psi(\lambda)]:=-\psi^*(t)$$

即这里的 RHS $\psi^*(t)$ 恰符合 $\psi(\lambda)$ 的 Legndre 变换之形式. 

在 Large Deviation Theory 中, 也会将 $\psi^*$ 称为 Cramér 变换或率函数 (rate function), 记为 $I(t)$, 其严格定义为:对于 $X_1, X_2, ... \stackrel{i.i.d}{\sim} \mathcal{P}(\mu)$, 记 $I(t)$ 为对数 MGF 的 Legendre 变换:
$$I(t) = \sup_\lambda [\lambda t - \log \mathbb{E}[e^{\lambda X}]] = \sup_\lambda [\lambda t - \psi(\lambda)].$$

## The Workhorses

这一部分的不等式更近一步, 对变量的分布本身给出了有界的限制. 这在真实数据中也是合理的. 

### Hoeffding Inequality

Hoeffding 可以看作是在 Chernoff 的基础上再强化条件得到的结果. 既然 Chernoff 已经相当于刻画了所有阶矩的情况, 因此需要考虑其他的补充条件. 在这里给定的是随机变量分布的取值范围. 其已知数据分布有界, 但并不假设方差. Hoeffding Inequality 是机器学习泛化误差的基础. 

***THEOREM (Hoeffding Inequality)*** 对于**独立**随机变量 $X_1,...,X_n$ (注意并不要求同分布), 且每个变量 $X_i$ almost surely 有界 (即 $\mathbb{P}(a_i\leq X_i \leq b_i) = 1, \forall i$). 记 $S_n= \sum_{i=1}^n X_i$. 则对于任意 $\epsilon>0$, 以下不等式成立:
- One-sided Upper Tail:
    $$\mathbb{P}\left( S_n - \mathbb{E}[S_n] \ge \epsilon \right) \le \exp\left( - \frac{2\epsilon^2}{\sum_{i=1}^n (b_i - a_i)^2} \right),$$
- One-sided Lower Tail:
    $$\mathbb{P}\left( S_n - \mathbb{E}[S_n] \le -\epsilon \right) \le \exp\left( - \frac{2\epsilon^2}{\sum_{i=1}^n (b_i - a_i)^2} \right),$$
- Two-sided:
    $$\mathbb{P}\left( |S_n - \mathbb{E}[S_n]| \ge \epsilon \right) \le 2 \exp\left( - \frac{2\epsilon^2}{\sum_{i=1}^n (b_i - a_i)^2} \right).$$

- *Proof* (这里给出 One-sided Upper Tail 的证明, Two-sided 可以类似证明):
    - **0. Centering:** W.L.O.G. 我们可以将 $X_i$ 中心化, 令 $Y_i = X_i - \mathbb{E}[X_i]$, 则 $\mathbb{E}[Y_i] = 0$. 此时 $Y_i$ 仍满足上述所有条件, 并且保证 $Y_i \in [a'_i, b'_i]$, a.s., 且区间的长度不变. 因此此时证明的目标转化为证明:
        $$\mathbb{P}\left(\sum_{i=1}^n Y_i \ge \epsilon \right) \le \exp\left( - \frac{2\epsilon^2}{\sum_{i=1}^n (b'_i - a'_i)^2} \right).$$

    - **1. Chernoff Method:** 对于任意 $\lambda>0$, 根据 Chernoff Method, 有:
        $$ \mathbb{P}\left(\sum_{i=1}^n Y_i \ge \epsilon\right) = \mathbb{P}\left(e^{\lambda \sum_{i=1}^n Y_i} \ge e^{\lambda \epsilon}\right) \leq \frac{\mathbb{E}[e^{\lambda \sum Y_i}]}{e^{\lambda \epsilon}}.$$

    - **2. Independence** 由于 $X_i$ 独立, $Y_i$ 亦独立, 因此 $\mathbb{E}[e^{\lambda \sum_{i=1}^n Y_i}] = \prod_{i=1}^n \mathbb{E}[e^{\lambda Y_i}]$

    - **3. Hoeffding's Lemma** (这是该不等式证明的核心支撑引理, 这里先直接给出): 由该 Lemma 可以保证, 对于中心化的 $Y_i \in [a_i', b_i']$, a.s., 有
        $$\mathbb{E}[e^{\lambda Y_i}] \le \exp\left( \frac{\lambda^2 (b_i' - a_i')^2}{8} \right).$$
        代入 step 2 的乘积中, 可得
        $$\prod_{i=1}^n \mathbb{E}[e^{\lambda Y_i}] \le \prod_{i=1}^n \exp\left( \frac{\lambda^2 (b_i' - a_i')^2}{8} \right) = \exp\left( \frac{\lambda^2}{8} \sum_{i=1}^n (b_i' - a_i')^2 \right).$$
        再将其代回到 step 1 RHS 的分子, 有
        $$\mathbb{P}\left(\sum_{i=1}^n Y_i \ge \epsilon \right) \le \exp\left( \frac{\lambda^2}{8} \sum_{i=1}^n (b_i' - a_i')^2 - \lambda \epsilon \right).$$

    - **4. Optimization**: 下面要寻找 RHS 的 $\inf$. 为简洁, 记 $K = \sum_{i=1}^n (b_i' - a_i')^2$. 则有 $g(\lambda) = \frac{K}{8}\lambda^2 - \lambda \epsilon$. 通过求导 $g'(\lambda) = \frac{K}{4}\lambda - \epsilon = 0$ 可得 $\lambda^* = \frac{4\epsilon}{K}$ (检查其二阶导 $g''(\lambda) = \frac{K}{4}$ 确保为最小值).  因此将 $\lambda^*$ 代入 $g(\lambda)$ 中, 可得
        $$g(\lambda^*) = \frac{K}{8}\left(\frac{4\epsilon}{K}\right)^2 - \frac{4\epsilon}{K}\epsilon = -\frac{2\epsilon^2}{K}.$$
    
    - **5. Conclusion**: 因此将 $g(\lambda^*)$ 代入 RHS, 可得
        $$\mathbb{P}\left(\sum_{i=1}^n Y_i \ge \epsilon \right) \le \exp\left( -\frac{2\epsilon^2}{K} \right) = \exp\left( -\frac{2\epsilon^2}{\sum_{i=1}^n (b_i' - a_i')^2} \right).$$
$\square$



**Remark:**
- 在 i.i.d. 形式中, 不等式上界仍服从 $e^{-nC}$ 的形式. 这也体现了指数集中的特性: 随着样本量 $n$ 线性增加, 出错的概率指数下降. 
- Hoeffding 中的独立性保证 $X_i$ 之间不会合谋一起偏离; 其有界性保证 $X_i$ 的波动有限不会有异常极端值.

***COROLLARY (Hoeffding Inequality for i.i.d.):***  对于独立同分布的 $X_1,...,X_n$, 且 $\forall X_i \in [a,b]$, 记 $\bar{X} = \frac{1}{n} S_n$, $\mu = \mathbb{E}[X]$. 对于任意误差容忍度 $\epsilon>0$, 将上式中 $\epsilon = n\varepsilon$ (因为总偏差 $\epsilon$ 相当于 $n$ 倍均值偏差 $\varepsilon$), 代入化简可得:
$$\mathbb{P}\left( |\bar{X} - \mu| \ge \epsilon \right) \le 2 \exp\left( - \frac{2n\varepsilon^2}{(b-a)^2} \right).$$
- 特别地， 若 $X_i$ 服从的是 $\text{Bernoulli}(p)$, 则 $\mu = p$, $a=0$, $b=1$, 代入上式可得
$$\mathbb{P}\left( |\bar{X} - p| \ge \epsilon \right) \le 2 \exp\left( - \frac{2n\varepsilon^2}{1^2} \right) = 2 \exp\left( - 2n\varepsilon^2 \right).$$
- Hoeffding Inequality 还提供了一种构造置信区间的方法. 以 Bernoulli 为例, 令 $\mathbb{P}\left( |\bar{X} - \mu| \ge \epsilon \right) \leq 2e^{-2n\varepsilon^2} = \alpha$, 从中解得 $\varepsilon_n$, 则可以得到置信区间为:
$$\bar{X}_n\pm \epsilon_n = \left[ \bar{X} - \sqrt{\frac{\ln(2/\alpha)}{2n}}, \bar{X} + \sqrt{\frac{\ln(2/\alpha)}{2n}} \right].$$


#### Hoeffding Lemma

***LEMMA (Hoeffding's Lemma):*** 这里补充该 Lemma 的详细证明.  设 $X$ 是一个均值为 $0$ 的随机变量, $X\in[a,b]$ a.s.. 我们要证明:
$$\mathbb{E}[e^{\lambda X}] \le \exp\left( \frac{\lambda^2 (b-a)^2}{8} \right).$$

- **Proof:** 
    - **Step 1 凸性放缩**: 由于 $x\in[a,b]$ a.s., 故可以用连接 $(a,e^{\lambda a})$ 和 $(b,e^{\lambda b})$ 的弦进行线性上界估计. 
        -   具体地, 可以将 $x$ 表示为 $a,b$ 的凸组合: $x = \frac{b-x}{b-a}a+ (1-\frac{b-x}{b-a})b$. 根据指数函数的凸性 ($\forall \gamma\in[0,1], f(\gamma x_1 + (1-\gamma) x_2) \leq \gamma f(x_1) + (1-\gamma) f(x_2)$), 有
            $$\exp(\lambda x)  = \exp\left( \frac{b-x}{b-a}\lambda a+\frac{x-a}{b-a}\lambda b\right) \le \frac{b-x}{b-a}\exp(\lambda a)+\frac{x-a}{b-a}\exp(\lambda b).$$
        - 再对该式左右两侧同取期望, 由于 $\mathbb{E}[X]=0$, 可得
            $$\mathbb{E}[\exp(\lambda X)] \le \frac{b-X}{b-a}\exp(\lambda a)+\frac{X-a}{b-a}\exp(\lambda b) = \frac{b}{b-a}\exp(\lambda a)+\frac{-a}{b-a}\exp(\lambda b).$$
        - 观察上式的 RHS, 记 $p = \frac{-a}{b-a}$, $L = \lambda(b-a)$, 则 $\frac{b}{b-a} = 1-p$, $a = -pL, b = (1-p)L$. 上式可以认为是一个随机变量 $Z$ 的 MGF ($\mathbb{E}[e^{\lambda Z}]$), 这个随机变量只在 $\{a,b\}$ 上取值, 具体为 $\mathbb{P}(Z=b) = p, \mathbb{P}(Z=a)=1-p$, 且 $\mathbb{E}(Z) = 0$. 因此上式可以化简为
            $$\mathbb{E}[e^{\lambda X}] \le \mathbb{E}[e^{\lambda Z}] = (1-p)e^{\lambda a} + pe^{\lambda b} = e^{-pL}(1-p+pe^L) := \phi(L).$$
    - **Step 2 Taylor Expansion:** 对我们得到的特殊的两点分布的 log-MGF 进行二阶泰勒展开.
        - 定义 $Z$ 的 log-MGF 为 
            $$\phi(L) = \log \mathbb{E}[e^{\lambda Z}] = \log \left( e^{-pL}(1-p+pe^L) \right) = -pL + \log(1-p+pe^L).$$
        - 对 $\phi(L)$ 在 $L=0$ 进行二阶泰勒展开, 则定存在某 $\xi\in[0,L]$ 使得
            $$\phi(L) = \phi(0) + \phi'(0) L + \frac{1}{2}\phi''(\xi) L^2 = \frac{L^2}{2}\phi''(\xi).$$
            - 这也是符合直觉的, 因为我们已经说明 $Z$ 的期望为 0, 故对于中心化的 log-MGF 进行求导就是其各阶矩.
    - **Step 3 二阶导数上界:** 最后我们利用两点分布的方差对该二阶导数进行 bound. 
        - 事实上, 观察 $\phi''(L)$ 的表达式, 可以发现其符合 logistics sigmoid 的形式. 若另记 $u = \frac{pe^L}{1-p+pe^L}$, 则 $\phi''(L) = u(1-u)$, $u\in(0,1)$. 因此 $\forall \xi,  \phi''(\xi)\le \frac{1}{4}$.
    - **Step 4 Conclusion:** 
        - 将 $\phi''(\xi)$ 的上界代入上式, 可得
            $$\log\mathbb{E}[e^{\lambda X}] \le \frac{L^2}{2}\times \frac{1}{4} = \frac{L^2}{8}.$$
        - 再代入 $L = \lambda(b-a)$, 可得
            $$\mathbb{E}[e^{\lambda X}] \le \exp\left( \frac{\lambda^2 (b-a)^2}{8} \right).$$
$\square$

**Remark:**
- Hoeffding Lemma 的证明的直觉相当于, 若 $X$ 是被限制在 $[a,b]$ 的随机变量, 那么波动最大的情况无外乎变成只在端点两点取值的 Rademacher 随机变量 (该分布是所有有界分布中方差最大的). 因此若我们能够用 Gaussian MGF 控制住它, 那么我们就控制住了其他所有的有界分布. 


#### Hoeffding Inequality and Subgaussianity

Sub-Gaussianity 是高维概率中的一个重要概念. 直觉上, 如果一个随机变量的 tail decay 至少和 Gaussian 一样快, 那么就称其服从 sub-Gaussian 的.

***DEFINITION (Sub-Gaussian):*** 称随机变量 $X$ 为 $\sigma^2$-sub-Gaussian 的, 如果存在常数 $\sigma>0$ 及任意 $\lambda \in \mathbb{R}$, 均有:
$$\mathbb{E}[e^{\lambda (X-\mathbb{E}[X])}] \le \exp\left( \frac{\lambda^2 \sigma^2}{2} \right).$$

**Remark:**
- 这里的 $\sigma^2$ 可以看作是 proxy variance, 用于度量 $X$ 的 tail decay speed. Hoeffding Lemma 说明: **任何有界的随机变量都是 sub-Gaussian 的**. 在尾部概率控制上, 任意有界分布都可以看作是一个方差为 $(b-a)^2/4$ 的 sub-Gaussian.
- 对比正态分布 $Z\sim \mathcal{N}(0,\sigma^2)$ 的 MGF, 可以发现 $\mathbb{E}[e^{\lambda Z}] = \exp\left( \frac{\lambda^2 \sigma^2}{2} \right).$ 因此 Sub-Gaussianity 的 sub 即表示 $X$ 的 MGF 被 Gaussian 的 MGF 限制住了. 同时由 Chernoff Trick, 若一个随机变量 $X$ 的 MGF 被 Gaussian MGF 控制, 那么立马可得其尾部概率同样被 Gaussian 的 tail probability 控制. 即, 若有 $\mathbb{E}[e^{\lambda X}] \le \exp\left( \frac{\lambda^2 \sigma^2}{2} \right)$, 则有 $\forall \lambda>0$, 
$$\mathbb{P}(X > \epsilon) = \mathbb{P}(e^{\lambda X} > e^{\lambda \epsilon}) \stackrel{\text{Markov Ineq.}}{\le} \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda \epsilon}} \le \inf_\lambda \exp\left( \frac{\lambda^2 \sigma^2}{2} - \lambda \epsilon \right) = \exp\left( -\frac{\epsilon^2}{2\sigma^2} \right).$$
- MGF 和 尾部概率 (tail probability) 类似硬币的一体两面:
    - Chernoff Trick 是沟通二者的桥梁. 其核心规则即为 Legendre 规则.
    - 这本身也是符合直觉的. MGF 作为 $X$ 的所有高阶矩的加权和, 若其能够被控制住, 这就说明 $X$ 取大值的概率必须非常非常小, 否则 $e^{\lambda X}$ 会指数级爆炸. 故反过来说明高阶矩的增长速度并么有超过 Gaussian. 
- 在统计推断中, 我们往往对分布有如下分类:
    - Bounded: 如 Bernoulli, Uniform 分布. 其本身随机变量的取值就是有界的. 故必然 Sub-Gaussian. 
    - Sub-Gaussian: 尾部不厚于 Gaussian, 可能有界, 可能无界. 如 Gaussian, Rademacher 分布.
    - Sub-Exponential: 尾部厚于 Gaussian, 但不厚于多项式分布. 如 Poission, Exponential 分布. 此时 Hoeffding Inequality 不再适用, 但 Bernstein Inequality 可以适用.
    - Heavy-tailed: 尾部厚于多项式分布. 只有有限阶矩, 甚至没有矩. 如 Cauchy, t 分布.


### Bernstein Inequality

Hoeffding 的一个问题在于其进行的放缩过于"悲观": 对于任意的有界变量, 其都是使用随机变量的上下界以 Rademacher 随机变量的 MGF 进行放缩. 然而对于方差较小的变量, 这种放缩没有很好的利用这一方差信息. 因此 Bernstein Inequality 通过显式的引入方差来更好的控制放缩. 其兼具 Hoeffding 的指数衰减形式和 Chebyshev 的方差敏感特性. 

***THEOREM (Bernstein Inequality):***  *(该叙述为 Bernstein Inequality 的常用有界形式)* 设 $X_1,\cdots, X_n$ 是独立的零均值随机变量, 且满足 1. $\forall i, |X_i|\le M$, a.s.; 2. 方差有界, 即 $\forall i, \mathbb{E}[X_i^2]\le \sigma_i^2$. 令 $S_n = \sum_{i=1}^n X_i$, 则对于任意 $\epsilon>0$, 有
$$\mathbb{P}(S_n \ge \epsilon) \le \exp\left( -\frac{\frac12\epsilon^2}{\sum_{i=1}^n \sigma_i^2 + \frac13M\epsilon} \right).$$

- *Proof Sketch:* 
    - **Step 1 Taylor Expansion of Moment:** 这里有别于 Hoeffding Lemma 只使用二阶导, 此处我们使用 $X$ 的所有阶矩的信息. 由于 $|X|\leq M$, 定有如下 Bernstein 条件成立:
        $$\mathbb{E}[|X|^k] \le \frac{k!}{2} \sigma^2 M^{k-2}$$
    - **Step 2 MGF Upper Bound:** 通过指数函数的 Taylor Expansion 及 Step 1 中的矩限制, 通过级数求和等方法, 我们会推得
        $$\mathbb{E}[e^{\lambda X}] \le \exp\left( \frac{\lambda^2 \sigma^2}{2(1 - \lambda M/3)} \right)$$
    - **Step 3 Chernoff Optimization**.

**Remark:**
- 关注 Bernstein Inequality 的 Bound 的分母 $\sum\sigma^2+\frac13M\epsilon$, 其直观上反映了随机变量在不同尺度下的 trade-off:
    - 小偏差: $\epsilon$ 较小或方差很大时, 有 $\sum\sigma^2\gg \frac13M\epsilon$, 故忽略 $M\epsilon$, 得到的 RHS 即为 $\exp\left(-\frac{t^2}{2\sum\sigma^2}\right)$. 这也就是说, 当偏差较小时, Bernstein 就变成了 CLT 的结果, 一个利用方差的更精准版本的 Hoeffding ($\sigma^2$ 而不是 $(b-a)^2$). **在中心附近（正常波动），它表现得像高斯分布.**
    - 大偏差: $\epsilon$ 较大或方差很小时, 例如 Poisson / Exponential 分布, 此时忽略方差项, bound 变成了 $\exp\left(-\frac{3\epsilon}{2M}\right)$. 这时指数部分从平方变成了线性. 此事为 Sub-Exponential 衰减. 这说明, 当偏差极大时, 尾部概率的衰减速度变慢了. **在极远处（极端波动），它表现得像 指数/泊松分布**


## Appendix: Two Expectation Inequalities

最后补充两个在前面经常使用的不等式: Cauchy-Schwarz Inequality 和 Jensen Inequality. 其直接衡量的是数据的矩之间的集合约束. 

***THEOREM (Cauchy-Schwarz Inequality):*** 对于随机变量 $X,Y$, 有
$$|\mathbb{E}[XY]| \le \sqrt{\mathbb{E}[X^2]\mathbb{E}[Y^2]}.$$
- *Proof*: 引入任意实数 $t$, 考虑 $\mathbb{E}[(tX+Y)^2]$
    - 显然 $(tX+Y)^2 \ge 0$, 故 $\mathbb{E}[(tX+Y)^2] \ge 0$. 
    - 将期望中的平方项展开, 根据期望的线性性, 有: $\mathbb{E}[(tX+Y)^2] = t^2\mathbb{E}[X^2]+2t\mathbb{E}[XY]+\mathbb{E}[Y^2] \ge 0$. 
    - 上式可看作是一个关于 $t$ 的二次函数, 为使得其关于任意 $t$ 均成立, 其判别式必须小于等于 0, 即 $\Delta = (2\mathbb{E}[XY])^2-4\mathbb{E}[X^2]\mathbb{E}[Y^2] \le 0$. 
    - 从而有 $\mathbb{E}[XY] \le \sqrt{\mathbb{E}[X^2]\mathbb{E}[Y^2]}$.

$\square$

***THEOREM (Jensen Inequality):*** 对于凸函数 $f$, 有 $\mathbb{E}[f(X)] \ge f(\mathbb{E}[X])$. 对于凹函数 $g$, 有 $\mathbb{E}[g(X)] \le g(\mathbb{E}[X])$.
- *Proof*: 
    - 对于凸函数 $f(x), 作其在 $x=\mathbb{E}[X]$ 处的支撑线 (supporting line, 由于考虑到 $f$ 不一定处处可导, 因此 supporting line 使用的是 sub-gradient 的概念) $L(x) = a + bx$ 且满足 $L(\mathbb{E}[X]) = f(\mathbb{E}[X])$. 
    - 由于 $f$ 是凸的, 因此定有 $f(x) \ge L(x)$, 即 $\mathbb{E}[f(X)] \ge \mathbb{E}[L(X)]$.
    - 故:
        $$\mathbb{E}[f(X)] \ge \mathbb{E}[L(X)] = L(\mathbb{E}[X]) = f(\mathbb{E}[X]).$$
    - 对于凹函数 $g$, 同理可证.

$\square$

上述的证明均较为简单. 但我们可以通过内积的角度更好理解这两个不等式. 

首先从简单的有限离散的随机变量开始. 
- 假设随机变量 $X,Y$ 只有 $n$ 种等可能的取值, 则可以把各自所有可能的取值写成两个向量: $\mathbf{x} = (x_1,\cdots,x_n)^\top, \mathbf{y} = (y_1,\cdots,y_n)^\top$. 
- 此时根据期望的定义:
$$\mathbb{E}[XY] = \sum_{i=1}^n x_i y_i \frac{1}{n} = \frac{1}{n} \langle \mathbf{x}, \mathbf{y} \rangle.$$
- 同理二阶矩可以写成
$$\mathbb{E}[X^2] = \sum_{i=1}^n x_i^2 \frac{1}{n} = \frac{1}{n} \| \mathbf{x} \|^2.$$
- 此时 Cauchy-Schwarz Inequality 可以写成 
    $$\left| \langle \mathbf{x}, \mathbf{y} \rangle \right| \le \| \mathbf{x} \| \| \mathbf{y} \|.$$
    - 几何上, 两个向量的点积永远不会超过它们长度的乘积.

如果 $X$ 变成了连续随机变量或无穷维的随机变量, 此时无法再列出其向量坐标. 
- 但是我们发现期望仍然符合内积的所有性质:
    - 线性性: $\mathbb{E}[aX+bY] = a\mathbb{E}[X]+b\mathbb{E}[Y]$
    - 对称性: $\mathbb{E}[XY] = \mathbb{E}[YX]$
    - 正定性: $\mathbb{E}[X^2] \ge 0$
- 因为 $\mathbb{E}[XY]$ 满足内积的性质, 事实上随机变量空间 (具有有限方差的) 构成了一个 Hilbert Space ($L^2$ Space). 此时期望就相当于是测量两个随机变量夹角和投影的工具.

上述的叙述很好的描述了两个随机变量之间的关系, 而这在对于单一随机变量的描述也是成立的. 此时我们需要引入单位向量 $\boldsymbol{1}$, 其具有性质 $\mathbb{P}(\boldsymbol{1}) = 1$. 故 $\mathbb{E}[X] = \mathbb{E}[X \cdot 1] = \langle X, \boldsymbol{1} \rangle$. 其相当于表示 $X$ 在确定性方向上的投影长度.