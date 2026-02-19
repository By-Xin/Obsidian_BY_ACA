## 引入

我们尝试从概率论的视角引入 Lebesgue测度的概念.  Lebesgue 测度类似于一个值域定义在 $[0, +\infty]$ 上的概率.
- 回顾**概率的公理化定义**:
  - 给定样本空间 $\Omega$
  - $\mathcal{F}$ 是 $\Omega$ 上的 $\sigma$-代数, 其是由 $\Omega$ 的某些子集组成的集合类 (其中 $\mathcal{F}$ 中的每个元素表示一个事件). 统称 $(\Omega, \mathcal{F})$ 为概率空间/可测空间. 
  - 定义在可测空间上满足**非负性, 可列可加性, 正则性**的函数称为概率 $\mathbb{P}$. 定义 $(\Omega, \mathcal{F}, \mathbb{P})$ 为概率空间.

- 进一步回顾 **$\sigma$-代数** 的概念:
  - 给定集合 $X$, $\mathcal{F}$ 是 $X$ 上的 $\sigma$-代数, 如果满足:
    - $\emptyset \in \mathcal{F}$
    - 对任意 $A \in \mathcal{F}$, 有 $A^c \in \mathcal{F}$
    - 对任意可列集族 $\{A_n\}_{n=1}^\infty \subseteq \mathcal{F}$, 有 $\bigcup_{n=1}^\infty A_n \in \mathcal{F}$

- 相仿地, 尝试定义一个欧式空间上的 Lebesgue 测度. 以 $\mathbb{R}$ 为例. 
  - 在 $\mathbb{R}$ 上会存在其某些子集构成的集合类 $\mathcal{B}$.
  - 二元组 $(\mathbb{R}, \mathcal{B})$ 组成了一个可测空间. 
  - 定义在可测空间上一个满足非负, 可列可加, 正则的函数被称为 Lebesgue 测度 $\mu$. 即:
    - 非负性: $\mu(A) \geq 0$ 对任意 $A \in \mathcal{B}$
    - 可列可加性: 对任意 $\{A_n\}_{n=1}^\infty \subseteq \mathcal{B}$, 有 $\mu(\bigcup_{n=1}^\infty A_n) = \sum_{n=1}^\infty \mu(A_n)$
    - 正则性: $\mu([a, b]) = b - a$ 对任意闭区间 $[a, b] \subseteq \mathbb{R}$

下面类似于积分中的 Darboux 和的思想, 尝试给出 Lebesgue 测度的构造.
- 对于任意集合 $E \in \mathbb{R}$, 尝试找出一些区间 $A_1, A_2, \ldots$,
  - 用外包法, 使得 $E \subseteq \bigcup_{n=1}^\infty A_n$. 我们将这些区间的体积之和的下确界称为 $E$ 的外测度: $\mu^*(E) = \inf \left\{ \sum_{n=1}^\infty |A_n| : E \subseteq \bigcup_{n=1}^\infty A_n \right\}$.
  - 用内填充法, 使得 $E \supseteq \bigcap_{n=1}^\infty A_n$. 我们将这些区间的体积之和的上确界称为 $E$ 的内测度: $\mu_*(E) = \sup \left\{ \sum_{n=1}^\infty |A_n| : E \supseteq \bigcap_{n=1}^\infty A_n\right\}$.
  - 若外包内填得到的外测度和内测度相等, 即 $\mu^*(E) = \mu_*(E)$, 则称 $E$ 是 Lebesgue 可测的, 且 Lebesgue 测度 $\mu(E) = \mu^*(E) = \mu_*(E)$.

- 从技术上讲, 处理外测度往往更为容易. Carathéodory 定理给出了相应条件, 使得在满足该条件的情况下, 外测度可以被视为 Lebesgue 测度. 我们后续的研究也多是从外测度视角出发.


## 外测度

下面给出 Lebesgue 外测度的定义和性质.

***定义 (Lebesgue 外测度)***:
- 对于 $\mathbb{R}^n$ 上的任意集合 $E$, 总能找到开区间序列 $I_1, I_2, \ldots$ 使得 $E \subset \bigcup_{k=1}^\infty I_k$. 计算出这些开区间的体积之和的下确界, 即为该集合的 Lebesgue 外测度:
$$\mu^*(E) = \inf \left\{ \sum_{k=1}^\infty |I_k| : E \subset \bigcup_{k=1}^\infty I_k \right\}$$

> 区别该定义与有限开覆盖定理. 后者也认为一定存在 $\bigcup_{k=1}^\infty I_k$ 使得 $E \subset \bigcup_{k=1}^\infty I_k$. 但后者要求 $E$ 必为闭区间, 然而 Lebesgue 外测度则会对 $\mathbb{R}^n$ 上的任意集合 $E$ 都成立.

对于外测度, 显然有如下性质成立:
1. $\mu^*(E) \ge 0, \mu^*(\emptyset) = 0$
2. 若 $A\subset B$, 则 $\mu^*(A) \leq \mu^*(B)$
3. $\mu^*(\cup_{k=1}^\infty A_k) \leq \sum_{k=1}^\infty \mu^*(A_k)$ (**次可加性**)

思考如下集合的外测度. 
- 对于 $A=[0,1]$, $\mu^*(A) = 1$. 下给出证明: $\mu^*(A) = |A|$. 
  - **首先说明 $\mu^*(A) \leq |A| = 1 $**. 对任意 $\epsilon>0$, 一定存在一个开区间 $I_0$ 使得 $A\subset I$ 且 $|A| + \epsilon > |I_0|$. 例如 $I_0 = (-\epsilon/3, 1+\epsilon/3)$. 在此基础上再构造一个足够小的开区间序列 $I_1, I_2, \cdots$ 使其长度之级数和足够小 (例如: $I_1 = (a, a+\frac{\epsilon}{100}), I_2 = (a, a+\frac{\epsilon}{2\times100}), \cdots$, 此时 $\sum_{k=1}^\infty |I_k| = \frac{\epsilon}{100}\sum_{k=1}^{\infty} \frac{1}{2^{k-1}} = \frac{\epsilon}{50}$), 此时我们便得到了一个开区间序列 $I_0, I_1, I_2, \cdots$, 满足 $A\subset I_0 \subseteq \cup_{k=0}^\infty I_k$. 而这个序列有: $ \sum_{k=0}^\infty |I_k| = 1 +\frac{2\epsilon}{3} + \frac{\epsilon}{50}< 1 +\epsilon$. 故由 $\epsilon$ 的任意性, $\inf  \sum_{k=0}^\infty |I_k| < 1 + \epsilon$. 又 $\mu^*(A)\triangleq\inf\sum_{k=0}^\infty |I_k|$, 故 $\mu^*(A) < 1+\epsilon = |A|+\epsilon$, 即 $\mu^*(A)\leq|A|$. 
  - **再说明 $\mu^*(A) \ge |A|$**. 对于闭区间 $A=[0,1]$, 由有限覆盖定理能够保证我们定能找到有限开覆盖 $\cup_{k=1}^n I_k$ 使得 $A\subset \cup_{k=1}^n I_k$, 并可立即推出 $A = A\cap (\cup_{k=1}^n I_k)$. 再根据集合的运算, $A = \cup_{k=1}^n (A\cap I_k)$, 故 $|A| = | \cup_{k=1}^n (A\cap I_k)| \leq \sum_{k=1}^n | A\cap I_k| \leq \sum_{k=1}^n|I_k|$. 而其中最后一个有限项的数列和也可以看作是外测度定义中的级数和之退化情形, 并由外测度之定义, $\mu^*(A) = \inf\sum_{k=1}^\infty|I_k| =\inf\sum_{k=1}^{n}|I_k|$, 故对任意 $\epsilon>0$, $\sum_{k=1}^n|I_k|  <\inf\sum|I_k|+ \epsilon = \mu^*(A)+ \epsilon$. 故 $|A| \leq \mu^*(A)$.
  - 综上, $\mu^*(A) = |A| = 1$. 
- 有理数集 $\mathbb{Q} \cap [0,1]$ 的外测度为 $0$. 
  - 事实上, 我们可以将 $\mathbb{Q} \cap [0,1]$ 中的有理数按从小到大的顺序排列为 $\{r_1, r_2, \ldots\}$. 对于任意 $\epsilon > 0$, 我们可以构造开区间序列 $I_k = (r_k - \frac{\epsilon}{2^{k+1}}, r_k + \frac{\epsilon}{2^{k+1}})$, 使得 $\mathbb{Q} \cap [0,1] \subseteq \bigcup_{k=1}^\infty I_k$. 此时这些开区间的长度之和为 $\sum_{k=1}^\infty |I_k| = \sum_{k=1}^\infty \frac{\epsilon}{2^k} = \epsilon$, 即 $\mu^*(\mathbb{Q} \cap [0,1]) = \inf \left\{ \sum_{k=1}^\infty |I_k| : \mathbb{Q} \cap [0,1] \subseteq \bigcup_{k=1}^\infty I_k \right\} \leq \epsilon$. 由 $\epsilon$ 的任意性, 可知 $\mu^*(\mathbb{Q} \cap [0,1]) = 0$.
- 事实上, 任意可数集的外测度均为 $0$. 
  

## Lebesgue 可测集

在上一节, 我们成功地定义了 Lebesgue 外测度. 其好处是, 外测度对于任意集合均有定义. 然而根据前面的探讨, 一个集合当且仅当其外测度与内测度均存在且相等时, 才能被确定其 Lebesgue 测度. 这就引出了 Lebesgue 可测集的概念. 而 Carathéodory 定理给出了一个更为简洁的判定 Lebesgue 可测集的条件.

***Theorem (Carathéodory 定理)***:
对于 $\mathbb{R}^n$ 上的任意集合 $A$, 我们称 $A$ 为 Lebesgue 可测的, 如果对于 $\mathbb{R}^n$ 上的任意集合 $E$, 以及外测度 $\mu^*$, 都有
$$\mu^*(E) = \mu^*(E \cap A) + \mu^*(E \cap A^c).$$
这时, 记 $\mu(A) = \mu^*(A)$, 为 $A$ 的 Lebesgue 测度.