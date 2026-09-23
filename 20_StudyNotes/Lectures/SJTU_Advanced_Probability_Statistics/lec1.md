# 概率空间、随机变量与分布

> References: 概率论引论, 谢践生

## 随机事件

- 事件 $A$ 总是对应于某个特定空间 $\Omega$ (所有可能结果之全体, 称为样本空间) 的某些子集 $A \subset \Omega$. $\omega \in\Omega$ 是样本空间的基本单元, 我们认为 $A$ 事件发生等价于 $\omega \in A$. 
- 记 $\Omega$ 所有可能的子集 (即所有可能的事件) 的构成的集合为 $2^\Omega$, 称为 power set. 理论上, power set 是我们最终想要研究概率论的对象, 但从数学上可以说明, 我们并不总能良好的定义这样的集合. 
- 在现代的概率论中, 我们不期望能够对 $2^\Omega$ 定义概率测度. 而是寻找一个足够丰富, 有恰好够用 (对概率操作封闭) 的集合族, 即下文的  $\sigma$-algebra. 

## 测度与公理化概率论

$\sigma$-algebra 是现代概率论中我们可以 '合法' 地谈论概率的范围. 

***Definition* ($\sigma$-algebra)**. 设 $\Omega$ 非空, $\mathcal{F}$ 是 $\Omega$ 的一些子集构成的 collection. 称 $\mathcal{F}$ 是一个 sigma-algebra, 若:
1. $\Omega \in \mathcal{F}$;
2. 对于补集封闭: 若 $E \in \mathcal{F}$, 则 $E^c \in \mathcal{F}$;
3. 对于可列并封闭: 若 $E_1, E_2, \cdots$, 则 $\bigcup_{n=1}^\infty E_n \in \mathcal{F}$.



关于 sigma-algebra 有如下说明:
- 这三点定义是自然的. 其要求: (1) 样本空间本身当然应该能谈概率; (2) 如果事件*发生*能够定义概率, 那么事件*不发生*也应该能定义概率; (3) 如果每个事件 $A_n$ 都能定义概率, 那么*至少*有一个事件发生也应该能定义规律 (在可列个意义下).
- 根据定义可以自然推出许多性质, 例如: 空集属于 $\mathcal{F}$; 可列交封闭; 差集封闭, 等等. 
- 最大的 $\sigma$-algebra 是 $2^\Omega$, 最小的 $\sigma$-algebra 是 $\{\Omega, \emptyset\}$ (或称 trivial $\sigma$-algebra). 
- 直觉上, $\sigma$-algebra 是一种信息集合, $\sigma$-algebra 越大, 能够判断, 区分, 讨论概率的事件信息就越多. 

> *若 $\mathcal{F}$ 是定义在 $\Omega$ 上的 $\sigma$-algebra, 则称 $(\Omega, \mathcal{F})$ 为一个 可测空间 (measureable space).* 

***Definition* (Measure)**. 对于 measure space $(\Omega, \mathcal{F})$, 称映射 $\mu: \mathcal{F} \to [0, \infty]$ 为一个 measure, 若:
1. $\mu(\emptyset) = 0$;
2. 可列可加性: 若 $E_1, E_2, \cdots \in \mathcal{F}$, 则 $\mu(\bigcup_{n=1}^\infty E_n) = \sum_{n=1}^\infty \mu(E_n)$.

> *称 $(\Omega, \mathcal{F}, \mu)$ 为一个测度空间 (measure space).* 

测度是一种抽象的用来描述集合大小的概念. 它具有以下性质:
- 有限可加性: 若 $E_1, E_2, \cdots, E_n \in \mathcal{F}$, 则 $\mu(\bigcup_{i=1}^n E_i) = \sum_{i=1}^n \mu(E_i)$.
- 单调性: 若 $E_1, E_2 \in \mathcal{F}$, 且 $E_1 \subset E_2$, 则 $\mu(E_1) \leq \mu(E_2)$.
- 差集测度: 若 $E_1, E_2 \in \mathcal{F}$, 且 $\mu(E_2) < \infty$, 则 $\mu(E_1 \setminus E_2) = \mu(E_1) - \mu(E_1 \cap E_2)$.
- 交并测度: 若 $E_1, E_2 \in \mathcal{F}$, 则 $\mu(E_1 \cup E_2) = \mu(E_1) + \mu(E_2) - \mu(E_1 \cap E_2)$.


概率 $\mathbb{P}$ 是一种特殊的测度, 其要求 total mass 为 1, 即 $\mathbb{P}(\Omega) = 1$. 故总结为如下公理化定义:

***Definition* (Probability axiom)**. 设 $\Omega$ 是一个非空集合, $\mathcal{F}$ 是 $\Omega$ 上的 $\sigma$-algebra, $\mathbb{P}: \mathcal{F} \to [0, 1]$ 是一个函数. 若 $(\Omega, \mathcal{F}, \mathbb{P})$ 满足以下公理:
1. 非负性: $\mathbb{P}(A) \geq 0$, 对于任意 $A \in \mathcal{F}$;
2. 归一化: $\mathbb{P}(\Omega) = 1$;
3. 可列可加性: 若 $E_1, E_2, \cdots \in \mathcal{F}$ 两两互不相交, 则 $\mathbb{P}(\bigcup_{n=1}^\infty E_n) = \sum_{n=1}^\infty \mathbb{P}(E_n)$.

> *则称 $\mathbb{P}$ 为一个概率 (probability). 称 $(\Omega, \mathcal{F}, \mathbb{P})$ 为一个概率空间 (probability space). 称 $\omega \in \Omega$ 为样本点, $\mathcal{F}$ 为事件域, $A \in \mathcal{F}$ 为事件.*

根据概率的公理化定义, 可以自然推出许多性质:
- 非负性: $\mathbb{P}(A) \geq 0$, 对于任意 $A \in \mathcal{F}$;
- 归一化: $\mathbb{P}(\Omega) = 1$;
- 可列可加性: 若 $E_1, E_2, \cdots \in \mathcal{F}$ 两两互不相交, 则 $\mathbb{P}(\bigcup_{n=1}^\infty E_n) = \sum_{n=1}^\infty \mathbb{P}(E_n)$.
- 单调性: 若 $E_1, E_2 \in \mathcal{F}$, 且 $E_1 \subset E_2$, 则 $\mathbb{P}(E_1) \leq \mathbb{P}(E_2)$.
- 差集概率: 若 $E_1, E_2 \in \mathcal{F}$, 且 $\mathbb{P}(E_2) < \infty$, 则 $\mathbb{P}(E_1 \setminus E_2) = \mathbb{P}(E_1) - \mathbb{P}(E_1 \cap E_2)$.
- 交并概率: 若 $E_1, E_2 \in \mathcal{F}$, 则 $\mathbb{P}(E_1 \cup E_2) = \mathbb{P}(E_1) + \mathbb{P}(E_2) - \mathbb{P}(E_1 \cap E_2)$.
- 次可列可加性: 对于任意可列个事件 $E_1, E_2, \cdots \in \mathcal{F}$, $\mathbb{P}(\bigcup_{n=1}^\infty E_n) \leq \sum_{n=1}^\infty \mathbb{P}(E_n)$.


## Borel $\sigma$-algebra 与随机变量

到上面的论述为止, 我们已经完成了对概率的公理化定义. 然而当我们考虑随机变量时, 还需要更多的分析工具:
- 随机变量粗略讲 $X$ 是一个从样本空间 $\Omega$ 到实数集 $\mathbb{R}$ 的映射: $X(\omega) \in \mathbb{R}$, 对于任意 $\omega \in \Omega$.
- 我们也非常习惯的用随机变量来描述事件并且讨论对应的概率, 比如对 $X$ 讨论 $\mathbb{P}(X = x)$, $\mathbb{P}(X \in A)$, 等等, 这并没有什么问题. 
- 然而在前面的公理化定义中, 概率 $\mathbb{P}: \mathcal{F} \to [0, 1]$ 是一个定义在 $\sigma$-algebra $\mathcal{F}$ 上的函数, 其本质的 input 是一些关于 $\Omega$ 的子集的信息. 然而诸如 $X \in A$ 这样的表达式并不是一个直接可以输入到 $\mathbb{P}$ 中的信息. 
- 换言之, 我们习以为常的讨论例如 $\mathbb{P}(X \in A)$ 其实更严谨地说, 是 $\mathbb{P}(\{ \omega \in \Omega: X(\omega) \in A \}) \equiv \mathbb{P}(X^{-1}(A))$, 即那些使得 $X(\omega) \in A$ 的样本点 $\omega$ 的概率测度. 故要使得 $\mathbb{P}(X \in A)$ 有意义, 我们需要 $X^{-1}(A) \in \mathcal{F}$.

因此为了能够严谨地讨论随机变量, 我们需要引入 Borel $\sigma$-algebra. 

***Definition* (Borel $\sigma$-algebra)**. 对于实数集 $\mathbb{R}$, 其所有开集生成的 $\sigma$-algebra 称为 Borel $\sigma$-algebra, 记作 $\mathcal{B}(\mathbb{R})$. 这里的生成是指从所有开集出发, 通过可列并, 可列交, 补集运算得到的最小 $\sigma$-algebra.

***Definition* (Measurable function)**. 给定两个可测空间 $(\Omega_1, \mathcal{F}_1)$ 和 $(\Omega_2, \mathcal{F}_2)$, 称映射 $f: \Omega_1 \to \Omega_2$ 为可测函数 (measurable function), 若对于任意 $A \in \mathcal{F}_2$, 其原象 $f^{-1}(A) \in \mathcal{F}_1$.
- 可测函数在说: codomain (值域) 上的可测集合, 通过 $f$ 的原象映射回 domain (定义域) 上, 仍然是可测集合.

***Definition* (Borel measurable function)**. 给定函数 $f: \mathbb{R} \to \mathbb{R}$, 称 $f$ 是 Borel measurable, 若对于任意 Borel 集中的元素 $A \in \mathcal{B}(\mathbb{R})$, 其原象 $f^{-1}(A) \in \mathcal{B}(\mathbb{R})$. 等价地, $\{t: f(t) \leq x\} \in \mathcal{B}(\mathbb{R})$ 对于任意 $x \in \mathbb{R}$.
- 直观上, Borel measurable function 是指函数 $f$ 的原象映射能够将 Borel 集映射回 Borel 集. 这保证了我们可以在实数轴上讨论 $f$ 的概率事件.

- 通过 Borel set 构造出的 $\sigma$-algebra 是符合我们平时的实数结构和自然直觉的. 任意的区间, 单点, 及其通过可列交并差集运算得到的集合都属于 Borel $\sigma$-algebra. 一般的连续函数, 单调函数等也都是 Borel measurable function.

因此严谨地定义随机变量需要如下. 

***Definition* (Random variable)**. 对于概率空间 $(\Omega, \mathcal{F}, \mathbb{P})$, 称映射 $X: \Omega \to \mathbb{R}$ 为随机变量, 若其满足 $\{X \leq x\} \in \mathcal{F}, \forall x \in \mathbb{R}$. *通俗地讲: 对于所有实数 $x$, 事件 $\{X \leq x\}$ 都能谈论概率*. 
- 由刚才 Borel measurable function 的定义可知, 随机变量是 Borel measurable function. 
- 同时记 $\sigma(X) := \sigma\{X^{-1}(B): B \in \mathcal{B}(\mathbb{R})\}$ **为 $X$ 生成的 $\sigma$-algebra**
  - 表示: 能用 $X$ 描写 (谈论概率) 的所有事件的集合. 换言之: 如果我们只能观察到随机变量 $X$, 那么我们能够分辨的事件 (的全体) 就是 $\sigma(X)$.
  - 例如, $\Omega = \{A,B,C\}$, $X(A) = 1$, $X(B) =  X(C) = 0$, 则 $\sigma(X) = \{\emptyset, \{A\}, \{B,C\}, \Omega\}$.
  - $\sigma(X)$ 是 $\mathcal{F}$ 的子集, 是在 $\Omega$ 上所有 $X$ 可测的最小 $\sigma$-algebra. 
- **Borel set 的 intuition**: $\mathcal{B}(\mathbb{R})$ 是定义在实数轴上的 $\sigma$-algebra, 即一些实数集合. 在这些实数集上能够良好谈论诸如 $X \in B \in \mathcal{B}(\mathbb{R})$ 这样的概率事件. 

随机向量的定义是随机变量的自然推广. 当 $X_1, X_2, \cdots, X_n$ 是 $n$ 个随机变量时, 称 $\mathbf{X} = (X_1, X_2, \cdots, X_n)^\top: \Omega \to \mathbb{R}^n$ 为随机(列)向量. 此时 $\mathbf{X}$ 满足 $\mathbf{X}^{-1}(\mathcal{B}(\mathbb{R}^n)) \in \mathcal{F}$.

*Example*. 考虑投掷均匀硬币, 一般的建模方法是取 $(\Omega, \mathcal{F}, \mathbb{P}) := (\Sigma, 2^\Sigma, \mathbb{P})$, 其中 $\Omega = \{H, T\}$, $\Sigma = \{ 0,1\}$, $\mathbb{P}(\{0\}) = \mathbb{P}(\{1\}) = 1/2$. 


## 分布律

现有概率空间 $(\Omega, \mathcal{F}, \mathbb{P})$, 随机变量 $X: (\Omega, \mathcal{F}) \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$. 我们自然希望直接在实数轴上讨论 $X$ 落入某个集合 $B$ 的概率. 

在以往的初等概率论中, 我们对随机变量 $X$ 的分布函数的概念已经比较熟悉. 

***Definition* (Cumulative Distribution Function)**. 对于随机变量 $X$, 称函数 $F(x) := \mathbb{P}(X \leq x)$ 对于任意 $x \in \mathbb{R}$ 为 $X$ 的分布函数. 若 $\mathbf{X} \in \mathbb{R}^n$ 是随机向量, 称函数 $F(x_1, x_2, \cdots, x_n) := \mathbb{P}(X_1 \leq x_1, X_2 \leq x_2, \cdots, X_n \leq x_n)$ 对于任意 $x_1, x_2, \cdots, x_n \in \mathbb{R}$ 为 $\mathbf{X}$ 的联合分布函数.

分布函数有如下性质:
- $F$ 是单调不减函数
- $F$ 是右连续函数: $\lim_{x \to x_0^+} F(x) = F(x_0)$.
- $\lim_{x \to -\infty} F(x) = 0$, $\lim_{x \to \infty} F(x) = 1$.

基于测度, 在上述分布函数的基础上, 还可以进一步定义**分布测度**. 

***Definition* (Distribution measure)**. 对于随机变量 $X: (\Omega, \mathcal{F}) \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$, 称映射 $\mu_X: \mathcal{B}(\mathbb{R}) \to [0, 1]$ 为 $X$ 的分布测度, 若 $\mu_X(B) = \mathbb{P}(X \in B) = \mathbb{P}(X^{-1}(B)) $ 对于任意 $B \in \mathcal{B}(\mathbb{R})$.


**说明** [关于随机变量, 分布律, 分布函数与分布测度]:
- 在初等概率论的讨论中, 分布函数与分布测度常常未做区分. 然而事实上, $\mathbb{P}$ 是 $\Omega$ 上的概率测度 (输入是各种事件), 而 $\mu_X$ 是 $\mathbb{R}$ 上的概率测度 (输入是实数集合). 
- 因此根据随机变量 $X$, 其将所有在抽象样本空间 $\Omega$ 上的讨论外推 (push forward) 到了实数轴 $\mathbb{R}$ 上:
    $$
    (\Omega, \mathcal{F}, \mathbb{P}) \xrightarrow{X} (\mathbb{R}, \mathcal{B}(\mathbb{R}), \mu_X)
    $$
- 进而, 一般的 $\mathbb{P}(X \in A)$ 是一种简写, 严格说应该是 $\mathbb{P}(X^{-1}(A))$. 而一旦定义了分布测度 $\mu_X$, 反而应该使用 $\mu_X(A)$ 来表示 $X$ 落入集合 $A$ 的概率. 

- 对于每个随机变量 $X$, 都唯一对应一个分布函数 $F$ 和一个分布测度 $\mu_X$. 因为, 给定分布函数 $F$, 可以定义
    $$
    \mu_X((a,b]):=F(b)-F(a), \forall a, b \in \mathbb{R}, a < b.
    $$
    反过来, 给定分布测度 $\mu_X$, 可以定义分布函数 $F(x) = \mu_X((-\infty, x])$.

- 然而一个分布函数 $F$ 并不一定对应一个随机变量 $X$. 例如考虑两点样本空间 $\Omega = \{\omega_1, \omega_2\}$, $X(\omega_1) = 0$, $X(\omega_2) = 1$, 而 $Y(\omega_1) = 1$, $Y(\omega_2) = 0$, 则 $X \stackrel{d}{=} Y$ 但 $X \neq Y$. 故有逻辑关系:
    $$
    X \longrightarrow F_X \longleftrightarrow \mu_X
    $$

---

## 总结

- **$\sigma$-algebra**: 现代概率论中我们可以 '合法' 地谈论概率的范围. 是一些集合的 collection, 满足 1. $\Omega$ 在其中; 2. 对于补集封闭; 3. 对于可列并封闭.
    - $\mathcal{F}$: 定义在 $\Omega$ 上的 $\sigma$-algebra, 是由 $\Omega$ 上所有可能事件的集合构成的集合族. 
    - **可测空间**: $(\Omega, \mathcal{F})$ 称为可测空间 (measureable space).
- **测度**: 从 $\mu: \mathcal{F} \to [0, \infty]$ 的映射, 满足 1. $\mu(\emptyset) = 0$; 2. 可列可加性.
- **概率**: 是一种特殊的测度, 其要求 total mass 为 1, 即 $\mathbb{P}(\Omega) = 1$. $(\Omega, \mathcal{F}, \mathbb{P})$ 称为概率空间 (probability space).
- **Borel $\sigma$-algebra**: 实数集上的 sigma-field. 对于实数集 $\mathbb{R}$, 其所有开集生成的 $\sigma$-algebra 称为 Borel $\sigma$-algebra, 记作 $\mathcal{B}(\mathbb{R})$.
- **可测函数**: 给定两个可测空间 $(\Omega_1, \mathcal{F}_1)$ 和 $(\Omega_2, \mathcal{F}_2)$, 称映射 $f: \Omega_1 \to \Omega_2$ 为可测函数 (measurable function), 若对于任意 $A \in \mathcal{F}_2$, 其原象 $f^{-1}(A) \in \mathcal{F}_1$.
- **Borel measurable function**: 给定函数 $f: \mathbb{R} \to \mathbb{R}$, 称 $f$ 是 Borel measurable, 若对于任意 Borel 集中的元素 $A \in \mathcal{B}(\mathbb{R})$, 其原象 $f^{-1}(A) \in \mathcal{B}(\mathbb{R})$.
- **随机变量**: 对于概率空间 $(\Omega, \mathcal{F}, \mathbb{P})$, 称映射 $X: \Omega \to \mathbb{R}$ 为随机变量, 若其满足 $\{X \leq x\} \in \mathcal{F}, \forall x \in \mathbb{R}$.
- $X$ 生成的 $\sigma$-algebra: $\sigma(X) := \sigma\{X^{-1}(B): B \in \mathcal{B}(\mathbb{R})\}$, 表示能用 $X$ 描写 (谈论概率) 的所有事件的集合. $\sigma$ 是在 $\Omega$ 上所有 $X$ 可测的最小 $\sigma$-algebra.
- **分布函数**: 对于随机变量 $X$, 称函数 $F(x) := \mathbb{P}(X \leq x)$ 对于任意 $x \in \mathbb{R}$ 为 $X$ 的分布函数.
- **分布测度**: 对于随机变量 $X: (\Omega, \mathcal{F}) \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$, 称映射 $\mu_X: \mathcal{B}(\mathbb{R}) \to [0, 1]$ 为 $X$ 的分布测度, 若  $\mu_X(B) = \mathbb{P}(X \in B) = \mathbb{P}(X^{-1}(B))$ 对于任意 $B \in \mathcal{B}(\mathbb{R})$.