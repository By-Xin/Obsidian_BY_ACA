
# Markov Chain

## 1. Introduction: Markov Assumption

- 对于一组随机变量 $X_1, X_2, \cdots, X_n$，其最完整的认知是其联合分布 $\mathbb{P}(X_1, X_2, \cdots, X_n)$. 然而对于大规模的随机变量集合, 其计算和存储都非常困难. 
- 通过条件概率, 我们可以将联合分布分解为条件分布的乘积:
    $$
    \mathbb{P}(X_1, X_2, \cdots, X_n) \equiv \mathbb{P}(X_1) \cdot \mathbb{P}(X_2 | X_1) \cdot \mathbb{P}(X_3 | X_1, X_2) \cdots \mathbb{P}(X_n | X_1, X_2, \cdots, X_{n-1})
    $$
    - 然而这是一个恒等变换, 说明了联合分布和条件分布之间的关系, 但并没有简化问题. 这里的条件相当于是一组复杂的 constraint, 需要考虑前面所有的随机变量.

- Markov Assumption: 通过引入 Markov 假设, 我们可以大大简化条件分布的计算. 其假设是当前状态 $X_n$ 只依赖于最近的前一个状态 $X_{n-1}$, 而与之前的状态无关. 
    $$
    \mathbb{P}(X_n | X_1, X_2, \cdots, X_{n-1}) = \mathbb{P}(X_n | X_{n-1})
    $$
    则此时联合分布可以简化为:
    $$
    \mathbb{P}(X_1, X_2, \cdots, X_n) = \prod_{i=1}^n \mathbb{P}(X_i | X_{i-1})
    $$

- 因此满足 Markov 假设的随机变量序列被称为 Markov Chain. 


- Markov 性质可以抽象地概括为:
    $$
    \mathbb{P}(\text{Future} | \text{Present}, \text{Past}) = \mathbb{P}(\text{Future} | \text{Present})
    $$
    且等价于 
    $$
    \mathbb{P}(\text{Past}, \text{Future} | \text{Present}) = \mathbb{P}(\text{Past} | \text{Present}) \cdot \mathbb{P}(\text{Future} | \text{Present})
    $$
    即在给定当前状态的条件下, 过去和未来是条件独立的.

$\quad$

> *抓住当下, 未来就与过去无关.*

## 2. Discrete-Time Markov Chain

### 2.1 Basic Definitions

***Definition* (Discrete-Time Markov Chain)** 给定离散时间, 离散过程的随机变量序列 $\{X_n\}$, 若满足
$$
\mathbb{P}(X_n | X_1, X_2, \cdots, X_{n-1}) = \mathbb{P}(X_n | X_{n-1}), \quad \forall n \geq 2
$$
则称 $\{X_n\}$ 是一个离散时间 Markov Chain.


- 为了简化符号, 我们将离散 Markov 的状态空间中的每一个状态用一个整数来表示. 因此状态空间可以表示为:
    $$
    \mathcal{S} = \{1, 2, \cdots, n, \cdots\}
    $$


***Definition* (Transition Probability)** 对于 Markov Chain $\{X_n\}$, 定义转移概率为:
$$
P_{ij} (m,n) = \mathbb{P}(X_n = j | X_m = i), \quad m < n
$$
即 *在时间 $m$ 处于状态 $i$ 的条件下, 在时间 $n$ 处于状态 $j$ 的概率*.



***Definition* (Stationary Markov Chain)** 若对于任意时刻 $m < n$, 转移概率 $P_{ij}(m,n)$ 仅依赖于时间差 $n-m$, 则称 Markov Chain 是平稳的. 即存在一个转移矩阵 $P$ 使得:
$$
P_{ij}(m,n) = P_{ij}(n-m) := P_{ij}
$$ 
若无特殊说明, 后续的 Markov Chain 均指平稳 Markov Chain.

- Markov Chain 是一个 chain, 其可以直观地理解成一个 Directed Graph (即有向图), 其中每一个节点代表一个状态, 每一条有向边代表一个转移概率. 
- Stationary Markov Chain 则意味着这个 Directed Graph 是一个 *Time-Homogeneous* 的图, 即转移概率不随时间变化. 也就是一旦这个 Directed Graph 确定了, 那么和具体到底什么时刻发生转移无关, 只要知道当前状态, 就可以根据这个 Directed Graph 来计算未来的状态分布.

### 2.2 $n$-step Transition Probability and Chapman-Kolmogorov Equation
 
Markov Chain 的一个最基本任务是计算转移概率:
$$
P_{ij}(n)
$$
即从状态 $i$ 出发经过 $n$ 步转移到状态 $j$ 的概率.


***Theorem* (Chapman-Kolmogorov Equation)** 对于离散平稳 Markov Chain $\{X_n\}$, 任意 $n, m \geq 0$, 有
$$
P_{ij}(m+n) = \sum_{k} P_{ik}(m) P_{kj}(n)
$$

- 该定理的 intuition 相当于是将不同的轨迹**根据空间进行分解**, 进行了分组: 
  - 将所有经过 $m$ 步, 能够从状态 $i$ 转移到状态 $k$ 的轨迹分为一组, 其概率为 $P_{ik}(m)$; 
  - 将所有经过 $n$ 步, 能够从状态 $k$ 转移到状态 $j$ 的轨迹分为一组, 其概率为 $P_{kj}(n)$. 
  - 因此对于每一个中间状态 $k$, 从状态 $i$ 出发经过 $m+n$ 步转移到状态 $j$ 的概率可以表示为 $P_{ik}(m) P_{kj}(n)$, 
  - 最后对所有的中间状态进行求和即可得到最终的转移概率.


- *Proof*. 
  - 根据平稳性及转移概率定义, LHS 相当于在求解:
    $$
    P_{ij}(m+n) = \mathbb{P}(X_{m+n} = j | X_0 = i) = \sum_{k} \mathbb{P}(X_{m+n} = j, X_m = k | X_0 = i)
    $$
    - 其中第二个表达式是通过全概率公式得到的, 其将所有可能的中间状态 $k$ 进行了分组.
  - 根据条件概率的性质, $\mathbb{P}(X_{m+n} = j, X_m = k | X_0 = i) \equiv \mathbb{P}(X_{m+n} = j | X_m = k, X_0 = i) \cdot \mathbb{P}(X_m = k | X_0 = i)$. 故上式可以继续化简为:
    $$
    P_{ij}(m+n) = \sum_{k} \mathbb{P}(X_{m+n} = j | X_m = k, X_0 = i) \cdot \mathbb{P}(X_m = k | X_0 = i)
    $$

  - 再根据 Markov 假设, $\mathbb{P}(X_{m+n} = j | X_m = k, X_0 = i) \equiv \mathbb{P}(X_{m+n} = j | X_m = k)$. 故上式可以继续化简为:
    $$
    P_{ij}(m+n) = \sum_{k} \mathbb{P}(X_{m+n} = j | X_m = k) \cdot \mathbb{P}(X_m = k | X_0 = i) = \sum_{k} P_{ik}(m) P_{kj}(n)
    $$


进一步观察到, 上述 C-K 方程组的表达方式与矩阵乘法的表达方式非常相似. 故我们可以将转移概率 $P_{ij}(n)$ 以矩阵的形式进行表示:
$$
\mathbf{P}(n) = \begin{bmatrix}
P_{11}(n) & P_{12}(n) & \cdots & P_{1k}(n) & \cdots \\
P_{21}(n) & P_{22}(n) & \cdots & P_{2k}(n) & \cdots \\
\vdots & \vdots & \ddots & \vdots & \cdots
\end{bmatrix} \in [0,1]^{|\mathcal{S}| \times |\mathcal{S}|}
$$
- 其中 $\mathbf{P}(n)$ 的第 $i$ 行第 $j$ 列的元素即为从状态 $i$ 出发经过 $n$ 步转移到状态 $j$ 的概率 $P_{ij}(n)$.
- $|\mathcal{S}|$ 表示状态空间的大小, 即 Markov Chain 中状态的数量.

则 C-K 方程组可以简化为矩阵乘法的形式:
$$
\mathbf{P}(m+n) = \mathbf{P}(m) \cdot \mathbf{P}(n)
$$

故利用这个方程组, 可以得到:
$$
\mathbf{P}(n) = \mathbf{P}(n-1) \cdot \mathbf{P}(1) = \mathbf{P}(n-2) \cdot \mathbf{P}(1) \cdot \mathbf{P}(1) = \cdots = \mathbf{P}(1)^n := \mathbf{P}^n
$$

- 这就完美地给出了对于 $P_{ij}(n)$ 的计算方法.  即 **$n$ 步转移概率矩阵 $\mathbf{P}(n)$ 可以通过 $1$ 步转移概率矩阵 $\mathbf{P}(1)$ 的 $n$ 次幂来计算得到.**

再仔细观察一下一步转移概率矩阵:
$$
\mathbf{P}(1) := \mathbf{P} = \begin{bmatrix}
P_{11} & P_{12} & \cdots & P_{1k} & \cdots \\
P_{21} & P_{22} & \cdots & P_{2k} & \cdots \\
\vdots & \vdots & \ddots & \vdots & \cdots
\end{bmatrix}
$$

- $\sum_{j} P_{ij} = 1, \forall i$. 即矩阵的行和为 $1$. 相当于从状态 $i$ 出发, 转移到所有可能的状态 $j$ 的概率之和为 $1$.
- 其行类似于一个"输入", 列类似于一个"输出".

### 2.3 Classification of States

我们并没有满足于仅仅知道 $n$ 步转移概率的计算方法, 还想知道当 $n$ 趋近于无穷大时, $n$ 步转移概率的极限行为. 特别地, 我们想知道当 $n \to \infty$ 时, $P_{ij}(n)$ 是否可能和一个与 $i$ 无关的常数 $P_j$ 收敛.  此时, 一个随机过程退化成了一个随机变量 (一个分布). 在渐进意义上, 虽然其本身是一个随机过程, 但其在 "in the long run" 上在每个阶段 (状态) 出现的概率是一个常数, 与时间无关.


首先对 Markov Chain 的状态进行分类. 对于 Markov Chain $\{X_n\}$, 定义状态 $i$ 和状态 $j$ 之间的关系如下:

#### Reachability and Communication

***Definition* (Reachability):** 状态 $i$ 可以到达状态 $j$ (记为 $i \to j$), 如果存在一个正整数 $n$ 使得 $P_{ij}(n) > 0$. 即存在一个正整数 $n$ 使得从状态 $i$ 出发经过 $n$ 步转移到状态 $j$ 的概率大于零.
- Reachability 是非对称的关系, 即 $i \to j$ 不一定意味着 $j \to i$.
- Reachability 是传递的关系, 即如果 $i \to j$ 且 $j \to k$, 则 $i \to k$.

***Definition* (Communicative):** 状态 $i$ 和状态 $j$ 相通 (记为 $i \leftrightarrow j$), 如果 $i \to j$ 且 $j \to i$. 即状态 $i$ 可以到达状态 $j$, 同时状态 $j$ 也可以到达状态 $i$.
- Communication 是对称的关系, 即如果 $i \leftrightarrow j$, 则 $j \leftrightarrow i$.
- Communication 是传递的关系, 即如果 $i \leftrightarrow j$ 且 $j \leftrightarrow k$, 则 $i \leftrightarrow k$.

#### Closed Set and Irreducibility

***Definition* (Closed State Set):**  对于状态集 $C \subseteq \mathcal{S}$, 称 $C$ 是 close 的, 当且仅当 $\forall i \in C, j \notin C$, $P_{ij}(n) = 0, \forall n \geq 0$. 即对于状态集 $C$ 中的任意状态 $i$, 都无法转移到状态集 $C$ 外的任意状态 $j$.
- Closed 的定义意味着, 一旦进入了 closed 集合中的某个状态, 就无法再离开这个 closed 集合.
- Closed 集合提供了一个 Markov Chain 的子集, 在这个子集中, Markov Chain 的行为是独立于外部状态, 且还是一个完整的 Markov Chain.


***Definition* (Irreducibility):** Markov Chain $\{X_n\}$ 是不可约的, 当且仅当 Markov Chain 中没有闭的真子集. 即对于 Markov Chain $\{X_n\}$ 中的任意非空真子集 $C \subset \mathcal{S}$, 都存在 $i \in C$ 和 $j \notin C$ 使得 $P_{ij}(n) > 0$ 对于某个 $n \geq 0$ 成立.
- **Markov Chain 不可约  $\iff$ Markov Chain 中所有状态都是相通的.**
- *Proof*
    - ($\Leftarrow$) 这是显然的.
    - ($\Rightarrow$) 其总的证明思路如下: 对于 $\mathcal{S}$ 中的任意状态 $i$, 定义所有从 $i$ 出发能够到达的状态集合为 $C_i = \{j \in \mathcal{S} | i \to j\}$. 若能证明 $C_i$ 是 closed 的, 则由于已知 $\mathcal{S}$ 是 irreducible 的, 故不存在闭的真子集, 则 $C_i$ 必须等于 $\mathcal{S}$, 而根据 closed 的定义, $C_i$ 中的任意状态 $j$ 都可以到达状态 $i$, 则 $i \leftrightarrow j$. 从而证明了 Markov Chain 中所有状态都是相通的.
    - 其 closeness 的证明如下: 对于任意 $j \in C_i$ 和任意 $k \notin C_i$, 若 $C_i$ 是 closed 的, 则 $j \to k$ 不成立. 反证法, 假设 $j \to k$ 成立, 则根据 reachability 的传递性, $i \to j$ (由于 $j \in C_i$)  且 $j \to k$ 则 $i \to k$, 从而 $k \in C_i$, 矛盾. 故 $C_i$ 是 closed 的.

$\square$

- 对于一个可约的 Markov Chain, 其一步转移概率矩阵 $\mathbf{P}$ 可以通过适当的行列变换 (其实相当于只是换了一下 $\mathcal{S}$ 中状态的顺序), 则总可以以如下的形式进行分块表示:
$$
\mathbf{P} = \begin{bmatrix}
\mathsf{P} & \mathsf{R} \\
\mathsf{0} & \mathsf{Q}
\end{bmatrix}
$$
- 左下角的 $\mathsf{0}$, 则表示从下面的对应的状态集合出发, 无法转移到上面的对应的状态集合. (Recall 一步转移矩阵的行相当于"起点", 列相当于"终点", 因此 $\mathsf{0}$ 的存在意味着从下面的状态集合出发, 无法转移到上面的状态集合.)
- 若递归地, $\mathsf{P}, \mathsf{Q}$ 仍然是可约的, 则可以继续进行分块, 直到所有的分块都是不可约的. 从而对于一个可约的 Markov Chain, 其一步转移概率矩阵 $\mathbf{P}$ 可以通过适当的行列变换, 呈现出一个阶梯状的分块结构, 其左下角的分块全为 $\mathsf{0}$, 其余的分块都是不可约的.

#### Transient State and Recurrent State

常返性 (Recurrence) 是 Markov Chain 中一个非常重要的概念. 其描述了 Markov Chain 中状态的长期行为. 其在数学上有几个等价命题. 对于其中一种叙述方式, 首先引入首达概率之概念:

***Definition* (First Passage Probability)** 对于 Markov Chain $\{X_n\}$, 定义从状态 $i$ 出发, 经 $n$ 步首次到达状态 $j$ 的概率为:
$$
f_{ij}(n) = \mathbb{P}(X_n = j, X_k \neq j, \forall k < n | X_0 = i)
$$
- 首达概率满足性质
    $$
    0 \leq \sum_{n=1}^{\infty} f_{ij}(n) \leq 1
    $$
    这是因为首达的限制使得这里的事件是互斥的, 因此其概率之和不超过 $1$. 但与之相对应的, 转移概率则没有首达的限制, 因此其概率之和可以超过 $1$:
    $$
    0 \leq \sum_{n=1}^{\infty} P_{ij}(n) \leq \infty
    $$

***Definition* (Recurrent State):** 状态 $i$ 是 recurrent 的, 当且仅当从状态 $i$ 出发, 首次回到状态 $i$ 的概率之级数和为 $1$. 即
$$
\text{State } i \text{ is recurrent} \iff \sum_{n=1}^{\infty} f_{ii}(n) = 1
$$
 - 直观来看, 其取到了首达概率之和的上界, 即 asymptotically, 从状态 $i$ 出发, 首次回到状态 $i$ 的概率之和为 $1$, 即几乎必然会回到状态 $i$.

- 定义随机变量 $\tau_i$ 为从状态 $i$ 出发, 返回 $i$ 的次数. 则 $\tau_i$ 的期望值可以表示为:
    $$
    \begin{aligned}
    \mathbb{E}[\tau_i] & = \mathbb{E}\left[\sum_{n=1}^{\infty} \mathbf{1}_{\{X_n = i\}} \mid X_0 = i\right] \\
    & = \sum_{n=1}^{\infty} \mathbb{E}[\mathbf{1}_{\{X_n = i\}} | X_0 = i] \\
    & = \sum_{n=1}^{\infty} \mathbb{P}(X_n = i | X_0 = i) \\
    & = \sum_{n=1}^{\infty} P_{ii}(n) 
    \end{aligned}
    $$
  - 因此, 若状态 $i$ 是 recurrent 的, 则 $\sum_{n=1}^{\infty} P_{ii}(n) = \infty$, 即从状态 $i$ 出发, 返回 $i$ 的次数的期望值为无穷大. 这也是 recurrent 状态的一个重要性质.

- 另一方面, $\tau_i$ 的期望还可以计算为:
    $$
    \begin{aligned}
        \mathbb{E}[\tau_i] & = \sum_{k=1}^{\infty} k \cdot \mathbb{P}(\tau_i = k) \\
        &= \sum_{k=1}^{\infty} k \cdot \left(\sum_{n=1}^{\infty} f_{ii}(n)\right)^k \cdot \left(1 - \sum_{n=1}^{\infty} f_{ii}(n)\right) \\
        &:= \sum_{k=1}^{\infty} k\cdot  f_{ii}^k \cdot (1 - f_{ii}) \\
        &= \frac{f_{ii}}{1 - f_{ii}} 
    \end{aligned}
    $$
    - 其中 $\sum_{n=1}^{\infty} f_{ii}(n)$ 表示从状态 $i$ 出发, 经过若干步首次回到状态 $i$ 的概率. 又根据本身作为 Markov Chain 的无记忆性, 每次回到状态 $i$ 后, 都相当于重新开始了一个新的 Markov Chain 的过程, 因此每次回到状态 $i$ 的概率都是 $\sum_{n=1}^{\infty} f_{ii}(n)$, 每次不回到状态 $i$ 的概率都是 $1 - \sum_{n=1}^{\infty} f_{ii}(n)$. 从而 $\tau_i$ 的分布可以表示为一个几何分布 (Geometric Distribution), 其参数为 $f_{ii} = \sum_{n=1}^{\infty} f_{ii}(n)$.
    - 也同样根据 $f_{ii}$ 在 recurrent 状态和 transient 状态的定义, 可以得到 $\mathbb{E}[\tau_i]$ 的取值为:
        $$
        \mathbb{E}[\tau_i] = \begin{cases}
        \infty, & \text{if } f_{ii} = 1 \iff i \text{ is recurrent} \\
        < \infty, & \text{if } f_{ii} < 1 \iff i \text{ is transient}
        \end{cases}
        $$

- 再记 $g_{ii}(n)$ 为从状态 $i$ 出发回到 $i$, 至少经过 $n$ 步的概率. 故
    $$
    \begin{aligned}
    g_{ii}(n) & = \mathbb{P}(\tau_i \geq n) = \sum_{m=1}^\infty f_{ii}(m) \cdot \mathbb{P}(\tau_i \geq n-1) = f_{ii} \cdot g_{ii}(n-1) \\
    \end{aligned}
    $$
    - 即从状态 $i$ 出发回到 $i$, 至少经过 $n$ 步的概率, 可以表示为从状态 $i$ 出发, somehow ever 第一次回到 $i$ 之后, 再至少用 $n-1$ 步回到 $i$ 的概率. 因此 $g_{ii}(n)$ 满足一个递归关系, 其解为 $g_{ii}(n) = f_{ii}^n$. 从而可以得到 $\mathbb{P}(\tau_i \geq n) = f_{ii}^n$. 这也说明了 $\tau_i$ 的分布是一个几何分布.
    - 进一步, 令 $n \to \infty$, 则 $\mathbb{P}(\tau_i = \infty) = \lim_{n \to \infty} \mathbb{P}(\tau_i \geq n) = \lim_{n \to \infty} f_{ii}^n$. 因此当 $f_{ii} < 1$ 时, $\mathbb{P}(\tau_i = \infty) = 0$, 即从状态 $i$ 出发, 最终回到状态 $i$ 的概率为 $1$; 当 $f_{ii} = 1$ 时, $\mathbb{P}(\tau_i = \infty) = 1$, 即从状态 $i$ 出发, 最终回到状态 $i$ 的概率为 $0$. 从而可以得到如下结论:
        $$
        \mathbb{P}(\tau_i = \infty) = \begin{cases}
        0, & \text{if } f_{ii} < 1 \iff i \text{ is transient} \\
        1, & \text{if } f_{ii} = 1 \iff i \text{ is recurrent}
        \end{cases}
        $$

- 为了进一步加深对 recurrent 状态的理解, 考虑如下方程:
    $$
    P_{ij}(n) = \sum_{k=1}^n f_{ij}(k) P_{jj}(n-k)
    $$
    - 直观含义为: 从 $i$ 出发经过 $n$ 步转移到 $j$ 的概率, 相当于在某一中间时刻 $k$ 首次到达 $j$ , 再用剩余的 $n-k$ 步从 $j$ 出发转移到 $j$ 的概率进行乘积, 最后对所有可能的中间时刻 $k$ 进行求和. 其相当于对所有可能的轨迹进行了时间的分解. 
    - *Proof*
        - 定义随机变量 $T_{ij}$ 为 First Passage Time, 即从状态 $i$ 出发首次到达状态 $j$ 所需的步数. 则从 $i$ 出发经过 $n$ 步转移到 $j$ 的事件, 可以表示为 $T_{ij} = k$ (即在第 $k$ 步首次到达 $j$) 且在剩余的 $n-k$ 步中从 $j$ 出发转移到 $j$. 因此可以得到:
        $$
        P_{ij}(n) = \mathbb{P}(X_n = j | X_0 = i) = \sum_{k=1}^n \mathbb{P}(X_n = j, T_{ij} = k | X_0 = i)
        $$
        - 再根据条件概率的性质, $\mathbb{P}(X_n = j, T_{ij} = k | X_0 = i) \equiv \mathbb{P}(X_n = j | T_{ij} = k, X_0 = i) \cdot \mathbb{P}(T_{ij} = k | X_0 = i)$. 故上式可以继续化简为:
        $$
        P_{ij}(n) = \sum_{k=1}^n \mathbb{P}(X_n = j | T_{ij} = k, X_0 = i) \cdot \mathbb{P}(T_{ij} = k | X_0 = i)
        $$
        - 观察 $\{T_{ij} = k\}$ 的定义, 这个事件等价于 $\{X_k = j\}$ 且 $\{X_1, X_2, \cdots, X_{k-1} \neq j\}$. 因此 
        $$
        \mathbb{P}(X_n = j | T_{ij} = k, X_0 = i) \equiv \mathbb{P}(X_n = j | X_k = j, X_1, X_2, \cdots, X_{k-1} \neq j, X_0 = i).
        $$
        再根据 Markov 假设, $\mathbb{P}(X_n = j | X_k = j, X_1, X_2, \cdots, X_{k-1} \neq j, X_0 = i) \equiv \mathbb{P}(X_n = j | X_k = j)$. 故上式可以继续化简为:
        $$
        P_{ij}(n) = \sum_{k=1}^n \mathbb{P}(X_n = j | X_k = j) \cdot \mathbb{P}(T_{ij} = k | X_0 = i)
        $$
        而 $\mathbb{P}(X_n = j | X_k = j) \equiv P_{jj}(n-k)$, $\mathbb{P}(T_{ij} = k | X_0 = i) \equiv f_{ij}(k)$. 从而得到:
        $$
        P_{ij}(n) = \sum_{k=1}^n f_{ij}(k) P_{jj}(n-k)
        $$

        $\square$

    - 观察这个方程, 发现这是一个卷积 (convolution) 的方程. 根据 $z$ 变换的性质, 我们得到如下结论:
        $$
        \sum_{n=0}^{\infty} P_{ii}(n) = \frac{1}{1 - \sum_{n=1}^{\infty} f_{ii}(n)}
        $$
        因此, 可以总结出如下结论:
        $$
        \boxed{
        \sum_{n=0}^{\infty} P_{ii}(n) = \begin{cases}
        \infty, & \text{if } \sum_{n=1}^{\infty} f_{ii}(n) = 1 \iff i \text{ is recurrent} \\
        < \infty, & \text{if } \sum_{n=1}^{\infty} f_{ii}(n) < 1 \iff i \text{ is transient}
        \end{cases}}
        $$
        这将 Recurrent State 的判断和转移概率的级数求和的敛散性联系了起来. 



***Example* (One-Dimensional Random Walk)** 考虑一个一维的随机游走 (One-Dimensional Random Walk), 其状态空间为 $\mathcal{S} = \mathbb{Z}$, 即整数集合. 其转移概率如下:
$$
P_{i,i+1} = p, \quad P_{i,i-1} = 1-p, \quad \forall i \in \mathbb{Z}
$$
下讨论该随机游走中状态 $0$ 的常返性.

- *Solution*: 考虑 $P_{00}(n)$, 即从状态 $0$ 出发经过 $n$ 步转移回状态 $0$ 的概率. 
  - 首先若 $n = 2k + 1$ 是奇数, 则 $P_{00}(n) = 0$, 因为在奇数步时, 无论如何转移, 都无法回到状态 $0$.
  - 其次若 $n = 2k$ 是偶数, 则
    $$
    \mathbb{P}_{00}(2k) = \binom{2k}{k} p^k (1-p)^k
    $$
    - 故讨论如下级数的敛散性:
        $$
        \sum_{k=0}^{\infty} \binom{2k}{k} p^k (1-p)^k
        $$
        - 根据 Stirling's approximation, $n! \sim \left(\dfrac{n}{e}\right)^n \sqrt{2 \pi n}$:
            $$
            \binom{2k}{k} p^k (1-p)^k \sim \frac{1}{\sqrt{ k}} (4p(1-p))^k
            $$
        - 因此当 $p \neq \frac{1}{2}$ 时, $4p(1-p) < 1$, 则级数收敛; 当 $p = \frac{1}{2}$ 时, $4p(1-p) = 1$, 则级数发散. 从而状态 $0$ 在 $p \neq \frac{1}{2}$ 时是 transient 的, 在 $p = \frac{1}{2}$ 时是 recurrent 的.

***Example* (Two-Dimensional Random Walk)** 考虑一个二维的随机游走 (Two-Dimensional Random Walk), 其状态空间为 $\mathcal{S} = \mathbb{Z}^2$, 即二维整数集合. 考虑平衡的随机游走, 即在每一个时刻, 以相同的概率 $1/4$ 向四个方向 (上、下、左、右) 转移. 下讨论该随机游走中状态 $(0,0)$ 的常返性. 
- *Solution*:
  - 相似地, 考虑偶数 $2n$ 步转移回状态 $(0,0)$ 的概率 $P_{(0,0)(0,0)}(2n)$.
  - 对于 $P_{(0,0)(0,0)}(2n)$, 其相当于在 $2n$ 步中, 向上转移的步数与向下转移的步数相同, 向左转移的步数与向右转移的步数相同. 因此可以将 $2n$ 步中的 $n$ 步分为两类: 向上或向下转移的步数 (记为 $k$), 向左或向右转移的步数 (记为 $n-k$). 则
    $$
    P_{(0,0)(0,0)}(2n) = \sum_{k=0}^n \frac{(2n)!}{k! k! (n-k)! (n-k)!} \left(\frac{1}{4}\right)^{2n}  = \binom{2n}{n}^2 16^{-n} \sim \frac{1}{\pi n}.
    $$
  - 因此讨论如下级数的敛散性:
    $$
    \sum_{n=0}^{\infty} P_{(0,0)(0,0)}(2n) \sim \sum_{n=0}^{\infty} \frac{1}{\pi n} = \infty
    $$
  - 从而状态 $(0,0)$ 是 recurrent 的.


***Example* (Three-Dimensional Random Walk)** 考虑一个三维的随机游走 (Three-Dimensional Random Walk). 事实上, 即使是一个平衡的三维随机游走, 其状态 $(0,0,0)$ 也是 transient 的:
$$
P_{ii}(2n) \sim n^{-3/2}
$$
故高于二维的随机游走, 其状态 $(0,0,\cdots,0)$ 就是 transient 的.

---

下面对 Recurrent State 和 Transient State 进行一些进一步的讨论. 

1. 若状态 $j$ 是 transient 的, 则定有 $\sum_{n=0}^{\infty} P_{jj}(n) < \infty$. 从而推出:
    $$
    \lim_{n \to \infty} P_{ij}(n) = 0, \qquad\forall j
    $$
    无论起点状态 $i$ 是什么, transient 状态 $j$ 在长期上出现的概率都趋近于 $0$. 这也是 transient 状态的一个重要性质.


2. 若状态 $i$ 和状态 $j$ 是相通的, 则 $i$ 和 $j$ 的常返性是相同的 (但反之不必然). 
    - *Proof*:  由于 $i$ 和 $j$ 是相通的, 则存在 $m, n \geq 0$ 使得 $P_{ij}(m) > 0$ 且 $P_{ji}(n) > 0$. 因此对于任意 $k \geq 0$, 
        $$
        P_{ii}(m+n+k) \geq P_{ij}(m) P_{jj}(k) P_{ji}(n)
        $$
        因此
        $$
        \sum_{k=0}^{\infty} P_{ii}(m+n+k) \geq P_{ij}(m) P_{ji}(n) \sum_{k=0}^{\infty} P_{jj}(k).
        $$
        从而如果 $j$ 是 recurrent 的, 则 $\sum_{k=0}^{\infty} P_{jj}(k) = \infty$, 则 $\sum_{k=0}^{\infty} P_{ii}(m+n+k) = \infty$, 则 $i$ 也是 recurrent 的. 同理, 如果 $i$ 是 recurrent 的, 则 $j$ 也是 recurrent 的.

      $\square$

    - 进而对于一组 irreducible 的状态, 其常返性是相同的. 

3. 对于有限状态 Markov Chain, 其至少存在一个 recurrent 状态.   
    - *Proof*. 用反证法. 假设一个有限状态 $\mathcal{S}$ 的 Markov Chain 中所有状态都是 transient 的. 则对于任意状态 $i$, $\sum_{n=0}^{\infty} P_{ii}(n) < \infty$. 从而 $\lim_{n \to \infty} P_{ii}(n) = 0$. 对任意步长 $n$, 考虑 $P_{ij} (n)$. 则定有 $1 = \sum_{j} P_{ij}(n)$ 对任意状态 $i$ 成立. 因此
        $$
        \lim_{n \to \infty} \sum_{j = 1}^{|\mathcal{S}|} P_{ij}(n) = \sum_{j = 1}^{|\mathcal{S}|} \lim_{n \to \infty} P_{ij}(n) = 1
        $$
        这与全部状态都是 transient 的假设矛盾. 从而至少存在一个 recurrent 状态.

        $\square$

   -  进一步, 有限状态 Markov Chain 若是 irreducible 的, 则其所有状态都是 recurrent 的. 

4. Recurrent State 只访问 recurrent state, 不访问 transient state.  即: 若 $i$ 是 recurrent 的, 则对于任意 $j$ 使得 $i \to j$, 定有 $j \to i$. 
    - *Proof*. 用 $g_{ji}$ 表示从状态 $j$ 出发至少经过无穷多步回到状态 $i$ 的概率. 根据 $i \to j$ 的定义, 定有 $P_{ij}(n) > 0$ 对于某个 $n \geq 0$ 成立. 因此
        $$
        1 = g_{ii} = \sum_{k \in \mathcal{S}} P_{ik}(n) g_{ki} 
        $$
        由于 $i$ 是 recurrent 的, 则 $g_{ii} = 1$. 又由于 $\sum_{k \in \mathcal{S}} P_{ik}(n) \equiv 1$, 则
        $$
        1 = \sum_{k \in \mathcal{S}} P_{ik}(n) g_{ki} = \sum_{k \in \mathcal{S}} P_{ik}(n) \cdot 1 = 1
        $$
        其等价于
        $$
        \sum_{k \in \mathcal{S}} P_{ik}(n) (g_{ki} - 1) = 0
        $$
        又因为 $P_{ik}(n) \geq 0$ 且 $g_{ki} - 1 \leq 0$, 且 $P_{ij}(n) > 0$, 则
        $$
        g_{ji} = 1
        $$

        $\square$

    - 这事实上是一个更强的结论, 说明 $j$ 是 almost surely 无穷次回到 $i$ 的. 从而 $j$ 也是 recurrent 的.


### 2.4 Long-run Properties of Markov Chain

接下来我们想要重点考察 Markov Chain 的长期行为. 也即当 $n \to \infty$ 时, $P_{ij}(n)$ 的极限行为. 首先对于其极限值的存在性进行讨论.

#### Existence of Limiting Distribution

考虑转移概率的 Cesaro Sum:
$$
\lim_{n \to \infty} \frac{1}{n} \sum_{k=1}^n P_{ij}(k)
$$

***Theorem* (Weak Ergodic Theorem):** 对于 Markov Chain $\{X_n\}$, 若其是 **irreducible 且 recurrent** 的, 则对于任意状态 $i, j$, 转移概率的 Cesaro Sum 的极限存在, 且与 $i$ 无关. 即
$$
\lim_{n \to \infty} \frac{1}{n} \sum_{k=1}^n P_{ij}(k) := \frac{1}{\mu_j} := \tilde{\pi_j} > 0
$$

- 其中 $\mu_j = \sum_{n=1}^{\infty} n f_{jj}(n)$ 表示从状态 $j$ 出发, 首次回到状态 $j$ 的期望时间. 
  - 若 $\mu_j < \infty \iff \tilde{\pi_j} > 0$, 则称状态 $j$ 是*正常返 (Positive Recurrent)* 的
  - 若 $\mu_j = \infty \iff \tilde{\pi_j} = 0$, 则称状态 $j$ 是*零返 (Null Recurrent)* 的

***Definition* (Periodicity):** Markov Chain $\{X_n\}$ 中的状态 $i$ 的周期 (period) 定义为 $d_i := \operatorname{gcd}\{n \geq 1 | P_{ii}(n) > 0\}$, 即从状态 $i$ 出发, 转移回状态 $i$ 的步数的最大公约数.
- 若 $d_i = 1$, 则称状态 $i$ 是 aperiodic 的. 即从状态 $i$ 出发, 转移回状态 $i$ 的步数没有周期性.
  - 在 aperiodic 的状态 $i$ 中, 定存在某 $N$ 使得对于任意 $n \geq N$, $P_{ii}(n) > 0$. 即从状态 $i$ 出发, 转移回状态 $i$ 的步数在某个时刻之后的每一步都大于零.
- 若两个状态 $i$ 和 $j$ 是相通的, 则 $d_i = d_j$. 

***Proposition*:** 对于 Markov Chain $\{X_n\}$, 若其是 **irreducible 且 aperiodic** 的, 则对于任意状态 $i, j$, 转移概率的极限存在, 且与 $i$ 无关. 
$$
\lim_{n \to \infty} P_{ij}(n) = \pi_j > 0
$$
或者用矩阵的形式表示为
$$
\lim_{n \to \infty} \mathbf{P}^n = \Pi = \begin{bmatrix}
\pi_1 & \pi_2 & \cdots & \pi_{|\mathcal{S}|} \\
\pi_1 & \pi_2 & \cdots & \pi_{|\mathcal{S}|} \\
\vdots & \vdots & \ddots & \vdots \\
\pi_1 & \pi_2 & \cdots & \pi_{|\mathcal{S}|}
\end{bmatrix}
$$

---

***Chapman-Kolmogorov Stationary Equation:*** 在对上述存在性进行刻画后, 下面试讨论其极限值的具体形式. 考虑 C-K 方程:
$$
\begin{aligned}
\mathbf{P}(n) &= \mathbf{P}^n \\ 
&= \mathbf{P}(n-1) \cdot \mathbf{P} \quad \text{(Forward Equation)} \\
&= \mathbf{P} \cdot \mathbf{P}(n-1) \quad \text{(Backward Equation)}
\end{aligned}
$$

在 Forward Equation 中, 对于左右两边同时取极限, 则
$$
\Pi = \Pi \cdot \mathbf{P}
$$
其等价于
$$
\begin{bmatrix}
\pi_1 & \pi_2 & \cdots & \pi_{|\mathcal{S}|} \\
\pi_1 & \pi_2 & \cdots & \pi_{|\mathcal{S}|} \\
\vdots & \vdots & \ddots & \vdots \\
\pi_1 & \pi_2 & \cdots & \pi_{|\mathcal{S}|}
\end{bmatrix} = \begin{bmatrix}
\pi_1 & \pi_2 & \cdots & \pi_{|\mathcal{S}|} \\
\pi_1 & \pi_2 & \cdots & \pi_{|\mathcal{S}|} \\
\vdots & \vdots & \ddots & \vdots \\
\pi_1 & \pi_2 & \cdots & \pi_{|\mathcal{S}|}
\end{bmatrix} \cdot \begin{bmatrix}
P_{11} & P_{12} & \cdots & P_{1|\mathcal{S}|} \\
P_{21} & P_{22} & \cdots & P_{2|\mathcal{S}|} \\
\vdots & \vdots & \ddots & \vdots \\
P_{|\mathcal{S}|1} & P_{|\mathcal{S}|2} & \cdots & P_{|\mathcal{S}||\mathcal{S}|}
\end{bmatrix}
$$
从而可以得到:
$$
 \begin{bmatrix}
\pi_1 & \pi_2 & \cdots & \pi_{|\mathcal{S}|}
\end{bmatrix} = \begin{bmatrix}
\pi_1 & \pi_2 & \cdots & \pi_{|\mathcal{S}|}
\end{bmatrix} \cdot \begin{bmatrix}
P_{11} & P_{12} & \cdots & P_{1|\mathcal{S}|} \\
P_{21} & P_{22} & \cdots & P_{2|\mathcal{S}|} \\
\vdots & \vdots & \ddots & \vdots \\
P_{|\mathcal{S}|1} & P_{|\mathcal{S}|2} & \cdots & P_{|\mathcal{S}||\mathcal{S}|}
\end{bmatrix} \iff \boldsymbol{\pi} = \boldsymbol{\pi} \cdot \mathbf{P}
$$
或等价地
$$
\begin{cases}
\pi_1 = \sum_{i=1}^{|\mathcal{S}|} \pi_i P_{i1} \\
\pi_2  = \sum_{i=1}^{|\mathcal{S}|} \pi_i P_{i2} \\
\vdots \\
\pi_{|\mathcal{S}|} = \sum_{i=1}^{|\mathcal{S}|} \pi_i P_{i|\mathcal{S}|}
\end{cases}
$$
从而求解 $\pi_j$ 的值.

对于该方程, 有如下说明:
1. 这是一个 Left-hand Equation, 即 $\boldsymbol{\pi}$ 是一个行向量, 其左乘 $\mathbf{P}$ 得到 $\boldsymbol{\pi}$ 本身 ($\boldsymbol{\pi} \cdot \mathbf{P} = \boldsymbol{\pi}$). 这与我们在 Linear Algebra 中常见的 Right-hand Equation 略有差异.
2. Stationary Distribution 还以为这, 若 $X_0\sim \boldsymbol{\pi}$, 则对于任意 $n \geq 0$, $X_n \sim \boldsymbol{\pi}$. 即如果初始状态 $X_0$ 的分布是 $\boldsymbol{\pi}$, 则在经过任意步转移后, 状态 $X_n$ 的分布仍然是 $\boldsymbol{\pi}$. 
3. 对于有限状态 Markov Chain, 该方程总有非零解, 而和对应的 Markov Chain 的具体性质, 如 recurrence, periodicity 等无关. 甚至对于没有极限的 Markov Chain 也同样适用. 只不过此时求得的是 Markov Chain 的一个 stationary distribution, 而非 limiting distribution. 这也是该方程的一个重要性质.


***Definition* (Stationary Distribution):** 对于 Markov Chain $\{X_n\}$, 定义 $\boldsymbol{\pi} = (\pi_1, \pi_2, \cdots, \pi_{|\mathcal{S}|})$ 为其 stationary distribution, 若其满足 $\boldsymbol{\pi} = \boldsymbol{\pi} \cdot \mathbf{P}$ 且 $\sum_{j=1}^{|\mathcal{S}|} \pi_j = 1$. 
 - 直观的看, 其相当于是说给定当前的一个分布 $\boldsymbol{\pi}$, 则在经过一步转移后, 其分布仍然是 $\boldsymbol{\pi}$, 即 $\boldsymbol{\pi}$ 是一个不变的分布. 因此称其为 stationary distribution. 当然可以递推地得到, 其在经过任意步转移后, 其分布仍然是 $\boldsymbol{\pi}$.



***Proposition* (Detailed Balance Condition):** 设 $\mathbf{P}$ 是一个 Markov Chain 的一步转移概率矩阵. 若存在一个概率分布 $\boldsymbol{\pi}$ 满足对于任意状态 $i, j \in \mathcal{S}$, 
$$
\pi_i P_{ij} = \pi_j P_{ji}
$$
则 $\boldsymbol{\pi}$ 就是 $\mathbf{P}$ 的一个 stationary distribution, 即定有
$$
\boldsymbol{\pi} = \boldsymbol{\pi} \cdot \mathbf{P}
$$
这是 stationary distribution 的一个充分条件.


***Example 1*** 考虑一个 Markov Chain $\{X_n\}$, 其状态空间为 $\mathcal{S} = \{1, 2, 3\}$, 任意两个状态之间的转移概率相同均为 $1/2$. 即
$$
\mathbf{P} = \begin{bmatrix}
0 & 1/2 & 1/2 \\
1/2 & 0 & 1/2 \\
1/2 & 1/2 & 0
\end{bmatrix}
$$
故
$$
\begin{bmatrix}\pi_1 & \pi_2 & \pi_3
\end{bmatrix} = \begin{bmatrix}\pi_1 & \pi_2 & \pi_3
\end{bmatrix} \cdot \begin{bmatrix}
0 & 1/2 & 1/2 \\
1/2 & 0 & 1/2 \\
1/2 & 1/2 & 0
\end{bmatrix}
$$
立即有
$$
\begin{bmatrix}\pi_1 & \pi_2 & \pi_3
\end{bmatrix} = \begin{bmatrix}1/3 & 1/3 & 1/3
\end{bmatrix}
$$

***Example 2* (Ehrenfest Model)** 考虑 Ehrenfest Model: 有一密闭容器, 内部分为两部分. 初始时刻, 左侧有 $N$ 个分子, 右侧为真空. 随后将容器内部隔板打开, 分子进行扩散直到达到平衡状态. 这一过程可以建模为一个 Markov Chain $\{X_n\}$. 
- 状态空间: $\mathcal{S} = \{0, 1, 2, \cdots, N\}$, 其中状态 $i$ 表示右侧有 $i$ 个分子, 左侧有 $N-i$ 个分子.
- 假设时间是离散的, 且每个时刻有且仅有一个分子进行转移 (即从左侧转移到右侧, 或者从右侧转移到左侧). 并且每个分子被选中的概率相同均为 $1/N$. 
- 可以写出其转移概率矩阵 $\mathbf{P}$:
    $$
    \mathbf{P} = \begin{bmatrix}
    0 & 1 & 0 & 0 & \cdots & 0 & 0 \\
    1/N & 0 & 1 - 1/N & 0& \cdots & 0 & 0 \\
    0 & 2/N & 0  & 1 - 2/N  & \cdots & 0 & 0  \\
    \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & 0 & 0 & \cdots & 1 & 0 \\
    \end{bmatrix}
    $$
    - 对于第 $i$ 行, 其表示当前右侧有 $i$ 个分子, 则下一步可能的状态为 $i-1$ (即从右侧转移到左侧) 或 $i+1$ (即从左侧转移到右侧). 且从右侧转移到左侧的概率为 $i/N$, 从左侧转移到右侧的概率为 $(N-i)/N$.

- 该 Markov Chain 显然是全部状态相通的, 因此是 irreducible 的. 又由于其状态空间是有限的, 因此所有状态都是 recurrent 的. 另外, 其 periodicity 为 $2$.  

- 故
    $$
    \boldsymbol{\pi} = \boldsymbol{\pi} \cdot \mathbf{P} \implies
    \begin{cases}
    \pi_0 = \pi_1 / N \\
    \cdots \\
    \pi_k = \pi_{k-1} \cdot (N-k+1)/N + \pi_{k+1} \cdot (k+1)/N, \\
    \cdots \\
    \pi_N = \pi_{N-1} / N
    \end{cases}
    $$

  - 可解得 $\pi_k = \binom{N}{k} 2^{-N}, \qquad k = 0, 1, 2, \cdots, N$
    - 这也是 Ehrenfest Model 的一个重要性质, 即在平衡状态下, 右侧有 $k$ 个分子的概率服从二项分布 (Binomial Distribution) $\operatorname{Binomial}(N, 1/2)$.
    - 由于其对于任意状态 $k$, $\pi_k > 0$, 因此所有状态都是正常返 (Positive Recurrent) 的.

- 另外, 根据公式 $\pi_i = (\sum_{n=1}^{\infty} n f_{ii}(n))^{-1}$, 故
    $$
    \pi_0 = \left(\sum_{n=1}^{\infty} n f_{00}(n)\right)^{-1} = 2^{-N} \implies \sum_{n=1}^{\infty} n f_{00}(n) = 2^N
    $$
    而取 $N = N_A \equiv 6.022 \times 10^{23}$, 则 $\sum_{n=1}^{\infty} n f_{00}(n) = 2^{N_A}$. 即, 若平均观察足够长的时间 (例如 $2^{6.022 \times 10^{23}}$ 单位时间), 则将回到初始状态 (即右侧没有分子) .


***Example 3* (Random Walk on a Finite Graph)** 考虑任意无向 finite graph $G = (V, E)$, 其顶点集合为 $V$, 边集合为 $E$, 且不存在孤立点. 定义一个 Markov Chain $\{X_n\}$ 在该图上进行随机游走 (Random Walk), 即在每个时刻, 从当前所在的顶点出发, 以相同的概率转移到其邻接的顶点. 下讨论该 Markov Chain 的长期行为.
- 记节点 $i$ 的度 (degree, 即与节点 $i$ 相邻的边的数量) 为 $d_i$. 则从节点 $i$ 转移到其邻接节点 $j$ 的概率为 $P_{ij} = 1/d_i$ (如果 $i$ 和 $j$ 是相邻的), 否则为 $0$.
- 合理猜测, 其 stationary distribution $\boldsymbol{\pi}$ 的第 $i$ 个分量 $\pi_i$ 与节点 $i$ 的度 $d_i$ 成正比. 即 $\pi_i = d_i / \sum_{j \in V} d_j$. 
    - *Proof*: 记节点 $i$ 的邻接节点集合为 $\mathcal{N}(i)$.  则
        $$
        \sum_i \pi_i P_{ij} = \sum_{i \in \mathcal{N}(j)} \frac{d_i}{\sum_{k \in V} d_k} \cdot \frac{1}{d_i} = \frac{1}{\sum_{k \in V} d_k} \sum_{i \in \mathcal{N}(j)} 1 = \frac{d_j}{\sum_{k \in V} d_k} = \pi_j
        $$
        即 $\boldsymbol{\pi} = \boldsymbol{\pi} \cdot \mathbf{P}$ 成立, 从而 $\boldsymbol{\pi}$ 是一个 stationary distribution.


- Markov Chain 在图上的典型应用之一是 PageRank 算法, 其核心思想就是将网页之间的链接关系建模为一个 Graph, 而一个网页的被链接指向数量 (即其度) 就是其重要程度的一个指标. 通过计算该 Graph 上的 stationary distribution, 就可以得到每个网页的重要程度. 这也是 PageRank 算法的一个重要性质.


### 2.5 Transient Behavior of Markov Chain

对于 Transient State, 其长期行为是趋近于 $0$ 的. 对于非常返的状态, 主要关注如下两个问题:

- Absorption Time: 从状态 $i$ 出发, 首次被吸收在集合 $A$ 中的时间. 记为 $T_i^A$:
    $$
    T_i^A := \min\{ n: X_n \in A \mid X_0 = i \}
    $$
    - 考察其期望值 $\mathbb{E}[T_i^A]$. 根据空间进行分解, 即从状态 $i$ 出发首先转移到状态 $j$, 然后从状态 $j$ 出发吸收到集合 $A$ 中的期望时间. 从而可以得到如下递推关系:
        $$
        \mathbb{E}[T_i^A] = 1 + \sum_{j \in \mathcal{S}} P_{ij} \mathbb{E}[T_j^A]
        $$

- Absorbing Probability: 从状态 $i$ 出发, 最终被吸收在集合 $A$ 中的概率. 记为 $P_i^A$:
    $$
    P_i^A := \mathbb{P}(T_i^A < \infty)
    $$
    - 对 $P_i^A$ 根据空间进行分解, 即从状态 $i$ 出发吸收到集合 $A$ 中的概率, 可以分解为从状态 $i$ 出发首先转移到状态 $j$, 然后从状态 $j$ 出发吸收到集合 $A$ 中的概率. 从而可以得到如下递推关系:
        $$
        P_i^A = \sum_{j \in \mathcal{S}} P_{ij} P_j^A
        $$
    - 若记 $\boldsymbol{\pi}^A = (P_0^A, P_1^A, \cdots, P_{|\mathcal{S}|}^A)$, 则 $\boldsymbol{\pi}^A$ 满足:
        $$
        \boldsymbol{\pi}^A = \mathbf{P} \cdot \boldsymbol{\pi}^A  
        $$
        这恰恰和前面的 forward equation 相对称, 是一个 right-hand equation.


***Example* (Gambler's Ruin Problem)** 考虑一个赌博者, 其初始资金为 $n$ 元. 在每一轮赌博中, 其以 $p$ 的概率赢得 $1$ 元, 以 $1-p$ 的概率输掉 $1$ 元. 当其资金达到 $0$ 元时, 赌博者破产; 但资金上不封顶. 下讨论赌博者最终破产的概率 $P_n^0$:
$$
P_n^0 = (1-p) P_{n-1}^0 + p P_{n+1}^0, \qquad P_0^0 = 1.
$$
因此根据递推关系:
$$
P_n^0 = \begin{cases}
1, & p \leq 1/2 \\
\left(\frac{1-p}{p}\right)^n, & p > 1/2
\end{cases}
$$
因此当 $p \leq 1/2$ 时, 赌博者最终破产. 注意, 只要赌徒不占优, 即使是公平的赌博 (即 $p = 1/2$), 赌博者最终也会 almost surely 破产. 这也是赌博的一个重要性质, 主要是由于赌博者的资金上不封顶, 因此总会几乎必然破产. 

不过, 若赌徒本身有一个退出机制, 即当其资金达到某个上限 $N$ 时就退出赌博, 则其最终破产的概率为
$$
P_n^0 = \begin{cases}
\frac{r^n - r^N}{1 - r^N}, & p \neq 1/2 \\
\frac{N-n}{N}, & p = 1/2
\end{cases}
$$
其中 $r = (1-p)/p$ 为赌徒的输赢比 (odds ratio). 故, 当有退出上限时, 不管 $p$ 多大, 最终一定会到达 $0$ 或 $N$ 之一. 而 $P_n^0$ 就代表了先到达 $0$ 的概率. 

上述问题都体现了一个 **first-step analysis** 的思想, 即对于一个问题, 可以根据其第一步的转移情况进行空间分解, 从而得到一个递推关系. 通过求解该递推关系, 就可以得到问题的答案. 这也是 Markov Chain 的一个重要分析工具.



### 2.6 Applications of Markov Chain


#### **PageRank Algorithm** (Brin and Page, 1998)

- PageRank 算法将网站之间的链接关系建模为一个有向图, 其节点表示网站, 边表示链接关系, 通过每个节点的入度 (即被其他网站链接的数量) 来衡量网站的重要性, 以对搜索结果进行排序等. 
- 然而单纯的考虑入度是不够的. 更为合理的思路是考虑其极限分布, 即在该有向图上进行一个随机游走 (在网络上进行充分浏览后) 得到的概率分布, 以此来衡量每个节点的重要性. 
- 在实践中, 这样的 Graph 是非常巨大的, 对于其周期性, 可约性的传统判断是非常困难的. 因此 PageRank 算法引入了一个 damping factor $\alpha \in (0, 1)$:
    $$
    \tilde{\mathbf{P}} = (1-\alpha) \mathbf{P} + \frac{\alpha}{|\mathcal{S}|} \mathbf{1}
    $$
    - 其中 $\mathbf{P}$ 是原始的转移概率矩阵, $\mathbf{1}$ 是一个全 $1$ 的矩阵, $\alpha/|\mathcal{S}|$ 是为了保证 $\tilde{\mathbf{P}}$ 是一个转移概率矩阵 (即每行的元素之和为 $1$).  
    - 此时 $\tilde{\mathbf{P}}$ 是一个 irreducible 且 aperiodic 的转移概率矩阵. 从而 $\tilde{\mathbf{P}}$ 的极限分布 $\boldsymbol{\pi}$ 是存在的.
    - 在实践当中, 这样的加总也是合理的. 也就是其将网页的访问刻画为两个过程: 一个是按照原始的链接关系进行访问 (即 $\mathbf{P}$), 另一个是通过输入网址等方法直接访问 (即 $\mathbf{1}$). 通过加权平均的方式, 就可以得到一个更为合理的访问模型. 这也是 PageRank 算法的一个重要性质.

- 故通过求解
    $$
    \boldsymbol{\pi} = \boldsymbol{\pi} \cdot \tilde{\mathbf{P}}
    $$
    即求解
    $\tilde{\mathbf{P}}$ 的 left eigenvector 即可求解 $\boldsymbol{\pi}$ 的值. 从而得到每个节点的重要程度. 


#### **Markov Chain Monte Carlo (MCMC)** (Metropolis et al., 1953; Hastings, 1970)

Monte Carlo 方法或统计模拟方法是一种通过随机采样来近似计算复杂分布性质的数值方法. 而 Markov Chain Monte Carlo (MCMC) 方法则是 Monte Carlo 方法的一种重要实现方式, 其中一个最经典的 MCMC 算法是 Metropolis-Hastings 算法, 由 Metropolis 等人在 1953 年提出, 后由 Hastings 在 1970 年推广.

首先简单介绍 Monte Carlo 方法.

- Monte Carlo 方法常需要应对一个复杂分布 $\mathcal{F}$, 其解析性质难以获得. 但是可以通过近似伪随机采样的方式获得足够多的样本 $\{X_1, X_2, \cdots, X_n\}$ 来近似计算 $\mathcal{F}$ 的一些性质 (例如其均值, 方差等). 从而将解析结构转为对样本的统计分析. 因此 MC 的重点变为如何获得服从 $\mathcal{F}$ 的足够样本, 并且我们期望这样的方法是 universal 的, 即对于任意分布 $\mathcal{F}$ 都适用的. 

- 一个经典的方法是 *Acceptance-Rejection Sampling*. 
  - 给定随机变量 $X \in \mathcal{X}$, 其概率密度为 $f$, 目标是生成一系列服从该分布的样本, 但直接从 $f(x)$ 进行采样可能是困难的. 因此引入一个 proposal distribution 其概率密度为 $g(x)$ 且通常较为简单易采样. 通过引入一个常数 $M > 0$ 确保 $f(x) \leq M g(x)$ 对于所有 $x \in \mathcal{X}$ 都成立, 即
    $$
    0 \leq \frac{f(x)}{M g(x)} \leq 1, \qquad \forall x \in \mathcal{X}
    $$

  - 其算法过程如下:
    1. 从 proposal distribution $g$ 中采样得到 $Y \sim g$.
    2. 在 $[0, 1]$ 上均匀采样得到 $U \sim \operatorname{Uniform}(0, 1)$.
    3. 若 $U \leq f(Y) / (M g(Y))$, 则接受 $Y$ 作为一个样本; 否则拒绝 $Y$ 并返回步骤 1.
    
  - 这是由于, 考虑微小区间 $[x, x + \mathrm{d}x]$, 则 $Y$ 落在该区间的概率为 $g(x) \mathrm{d}x$, 且接着被接受的概率为 $f(x) / (M g(x))$, 从而总的接受某个样本点的概率为 $f(x) \mathrm{d}x / M$. 因此被接受的样本点的概率密度为 $f(x) / M$. 即就是 $f(x)$ 的一个缩放版本, 从而被接受的样本点的分布就是 $f(x)$.

接着考虑 MCMC.

- MCMC 即提供了一个这样的 Universal Pseudo-random Sampling Method. 对于离散 Markov Chain $\{X_n\}$, 其转移概率矩阵为 $\mathbf{\tilde{P}}$, 其 stationary distribution 为 $\boldsymbol{\pi}$. 并若其满足 irreducible 和 aperiodic 的条件, 则其极限分布 $\boldsymbol{\pi}$ 是存在的, 满足:
    $$
    \boldsymbol{\pi} = \boldsymbol{\pi} \cdot \mathbf{\tilde{P}}
    $$
    - MCMC 假设极限分布 $\boldsymbol{\pi}$ 就是我们想要近似的分布 $\mathcal{F}$ (当然这里处理的是离散分布 $\mathcal{F}$; 对于连续分布, 也同样可以用类似的连续时间 Markov Chain 来进行类似分析). 从而尝试求解 $\mathbf{\tilde{P}}$. 
    - 一旦真的得到了这样的 $\mathbf{\tilde{P}}$, 则就可以通过在 $\mathbf{\tilde{P}}$ 上进行随机游走来获得服从 $\boldsymbol{\pi}$ 的样本. 因为 $\boldsymbol{\pi}$ 是 $\mathbf{\tilde{P}}$ 的极限分布, 从而在 $\mathbf{\tilde{P}}$ 上进行随机游走足够长的时间后, 每次进行一次一步转移, 就可以得到一个服从 $\boldsymbol{\pi}$ 的样本. 

- 显然, $\mathbf{\tilde{P}}$ 的求解直观上是欠定的, 其求解的思路如下.
  - 回顾, MC 具有如下 Detailed Balance Condition 作为 $\boldsymbol{\pi} = \boldsymbol{\pi} \cdot \mathbf{P}$ 的一个充分条件:
    $$
    \pi_i P_{ij} = \pi_j P_{ji}, \qquad \forall i, j
    $$

  - 下构造一个满足 Detailed Balance Condition 的转移概率矩阵 $\mathbf{\tilde{P}}$. 首先任取一个 proposal 的转移概率矩阵 $\mathbf{P}$. 对这个 proposal 进行如下修正:
    $$
    \tilde{P}_{ij} = P_{ij} \cdot \min\left\{1, \frac{\pi_j P_{ji}}{\pi_i P_{ij}}\right\}, \qquad \forall i, j
    $$
    对称地,
    $$
    \tilde{P}_{ji} = P_{ji} \cdot \min\left\{1, \frac{\pi_i P_{ij}}{\pi_j P_{ji}}\right\}, \qquad \forall i, j
    $$
    - 可以立即验证, $\mathbf{\tilde{P}}$ 满足 Detailed Balance Condition:
        $$
        \pi_i \tilde{P}_{ij} = \pi_i P_{ij} \cdot \min\left\{1, \frac{\pi_j P_{ji}}{\pi_i P_{ij}}\right\} \equiv \pi_j P_{ji} \cdot \min\left\{1, \frac{\pi_i P_{ij}}{\pi_j P_{ji}}\right\} = \pi_j \tilde{P}_{ji}
        $$
        从而 $\mathbf{\tilde{P}}$ 的极限分布就是 $\boldsymbol{\pi}$. 


  - 通过不同的 proposal 的选择, 可以得到不同的 MCMC 算法等, 以得到更好的算法收敛等进一步的改进优化. 

- 在实作当中, 当我们求得了一个满足 Detailed Balance Condition 的 $\mathbf{\tilde{P}}$, 我们整体的算法过程如下. 其中假设状态空间为 $\mathcal{S} = \{1, 2, \cdots, K\}$. 则每一步 $\boldsymbol{\pi} = (\pi_1, \pi_2, \cdots, \pi_K) \in \mathbb{R}^{1\times K}$, 对应的 $\mathbf{X} \in \mathcal{S}$.
  - 首先任取一个初始状态, 例如 $X_0 \in \mathcal{S}$. 
  - 随后对于每一步 $n \geq 1$,  取其前一步的状态 $X_{n-1}$, 根据 $\mathbf{\tilde{P}}$ 的第 $X_{n-1}$ 行的转移概率进行随机游走 (即以 $\mathbf{\tilde{P}}$ 的第 $X_{n-1}$ 的概率分布进行随机抽样), 得到 $X_n$.
  - 通过上述过程, 就可以得到一个样本序列 $\{X_0, X_1, \cdots, X_n\}$, 丢弃前面的一部分 (成为 burn-in period), 就可以得到一个服从 $\boldsymbol{\pi}$ 的样本序列 $\{X_{n_0}, X_{n_0+1}, \cdots, X_n\}$.

另外还可以从统计计算的角度来理解 MCMC, 并且扩展到连续时间 Markov Chain 的情形. 
- 考虑目标的概率分布 $\boldsymbol{\pi}(x)$, 定义连续状态下的转移核 (即状态转移的概率密度) 为
    $$
    K(x, A) = \mathbb{P}(X_{n+1} \in A \mid X_n = x)
    $$
    同样, 给定一个 proposal $q(y \mid x)$, 即在当前状态 $x$ 的条件下, 转移到状态 $y$ 的概率密度. 通常比如可以选择一个对称的 proposal 如 Gaussian 等.
- 此时类似地, 也有接受概率
    $$
    \alpha(x, y) = \min\left\{1, \frac{\boldsymbol{\pi}(y) q(x \mid y)}{\boldsymbol{\pi}(x) q(y \mid x)}\right\}
    $$
- 从而完整的抽样流程为
    1. 给定当前的状态 $X_n$, 从 proposal $q(y \mid x)$ 中采样得到 $Y \sim q(\cdot \mid X_n)$.
    2. 在 $[0, 1]$ 上均匀采样得到 $U \sim \operatorname{Uniform}(0, 1)$.
    3. 若 $U \leq \alpha(X_n, Y)$, 则接受 $Y$ 作为下一个状态 $X_{n+1}$; 否则拒绝 $Y$ 并令 $X_{n+1} = X_n$.

<!-- #### **Hidden Markov Model (HMM)** 

- HMM 是一个 Markov Chain 的扩展, 其在每个时刻除了有一个隐状态 (hidden state) $S_n$ 之外, 还有一个可观测的输出 (observable output) $O_n$. 其中隐状态 $S_n$ 满足 Markov Chain 的性质, 但是不可直接观测到; 可观测的输出 $O_n$ 则是由隐状态 $S_n$ 生成的, 其具有一定的随机性.
- HMM 有如下两大重要假设:
    -  齐次 Markov 性: 当前的状态是满足 Markov 性质, 并且与任意观测状态无关   
        $$
        \mathbb{P}(S_n \mid S_{n-1}, O_{n-1}, \cdots, S_0, O_0) = \mathbb{P}(S_n \mid S_{n-1})
        $$
    -  观测独立: 任意时刻的观测只依赖于当前的隐状态, 而与其他隐状态和观测状态无关. 即
        $$
        \mathbb{P}(O_n \mid S_n, S_{n-1}, O_{n-1}, \cdots, S_0, O_0) = \mathbb{P}(O_n \mid S_n)
        $$ -->

#### **Markov Decision Process (MDP)**

MDP 是强化学习中的一个重要概念. 其本身的构建是基于 Markov Chain 的. 在正式引入 MDP 之前, 首先介绍一个相关的概念, 即 Markov Reward Process (MRP).
- 一个 Markov Reward Process (MRP) 是一个四元组 $\langle \mathcal{S}, \mathbf{P}, r, \gamma \rangle$
  -  $\mathcal{S}$ 是有限状态集合, $\mathbf{P}$ 是一个转移概率矩阵, 二者共同定义了一个 Markov Chain.
  -  $r: \mathcal{S} \to \mathbb{R}$ 是一个 reward function, 表示当前处在状态 $s$ 时, 转移到下一个状态时获得的 reward 的期望值. 其定义为
        $$
        r(s) = \mathbb{E}[R_{t+1} \mid S_t = s]
        $$
        - 其中 $R_{t+1}$ 是一个随机变量, 表示智能体在 $t$ 处在状态 $S_t$ 时 (并进行了一个 action $A_t$ 后), 环境在下一个时刻反馈给智能体的即时 reward. 整体流程为 $S_t \xrightarrow{A_t} S_{t+1}, R_{t+1}$.
        - 而 $r(s)$ 是在当前处于状态 $s$ 时, 转移到下一个状态时获得的 reward 的期望值, 是一个确定性的函数(条件期望).
  - $\gamma \in [0, 1)$ 是一个 discount factor, 作为对 reward 进行累积时的权重衰减因子. 其强调了 reward 的时间价值,对于远期的 reward 给予更小的权重. $\gamma$ 越接近 $1$, 则越重视远期的 reward; $\gamma$ 越接近 $0$, 则越重视近期的 reward. 
- 对于所有未来 reward 的累计权重级数和即为 return:
    $$
    G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
    $$
    - 由于 $\gamma < 1$, 因此该级数是收敛的, 从而 $G_t$ 是一个 well-defined 的随机变量. 


- 对于一个状态的期望汇报, 即 $G_t$ 在给定当前状态 $S_t = s$ 的条件期望, 为 value function:
    $$
    V(s) = \mathbb{E}[G_t \mid S_t = s]
    $$
    且若代入 $G_t$ 的定义, 可以整理得到如下的方程, namely **Bellman Equation**:
    $$
    \begin{aligned}
    V(s) &= \mathbb{E}\left[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t = s\right] \\
    &= \mathbb{E} \left[R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \cdots) \mid S_t = s\right] ~ \text{\scriptsize (代入 $G_{t+1}$ 的定义)} \\
    &= \mathbb{E} \left[R_{t+1}\mid S_t = s\right] + \gamma \mathbb{E} \left[G_{t+1} \mid S_t = s\right] \\
    &= r(s) + \gamma \sum_{s' \in \mathcal{S}}\mathbb{P} (S_{t+1} = s' \mid S_t = s) \cdot \mathbb{E}[G_{t+1} \mid S_{t+1} = s', S_t = s]  ~ \text{\scriptsize (由 $r(s)$ 定义及全期望公式)} \\
    &= r(s) + \gamma \sum_{s' \in \mathcal{S}} \mathbb{P}(S_{t+1} = s' \mid S_t = s) \cdot V(s') ~ \text{\scriptsize (由 Markov 性质及 $V(s)$ 定义)} \\
    &:= r(s) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s) V(s')  ~ \text{\scriptsize (定义 $P(s' \mid s)$)} \\
    &= r(s) + \gamma \mathbb{E} \left[V(s') \mid S_t = s\right] ~ \text{\scriptsize (由条件期望定义)}
    \end{aligned}
    $$


- 上述方程对于所有状态 $s \in \mathcal{S} = \{s_1, s_2, \ldots, s_n\}$ 都成立, 因此可以写成一个矩阵形式的方程:
    $$
    \mathbf{V} = \mathbf{R} + \gamma \mathbf{P} \cdot \mathbf{V}
    $$
    展开形式为
    $$
    \begin{bmatrix}
    V(s_1) \\ V(s_2) \\ \vdots \\ V(s_n)
    \end{bmatrix} = \begin{bmatrix}
    r(s_1) \\ r(s_2) \\ \vdots \\ r(s_n)
    \end{bmatrix} + \gamma \begin{bmatrix}
    P(s_1 \mid s_1) & P(s_2 \mid s_1) & \cdots & P(s_n \mid s_1) \\
    P(s_1 \mid s_2) & P(s_2 \mid s_2) & \cdots & P(s_n \mid s_2) \\
    \vdots & \vdots & \ddots & \vdots \\
    P(s_1 \mid s_n) & P(s_2 \mid s_n) & \cdots & P(s_n \mid s_n)
    \end{bmatrix} \cdot \begin{bmatrix}
    V(s_1) \\ V(s_2) \\ \vdots \\ V(s_n)
    \end{bmatrix}
    $$
    - 该方程本身可以直接给出 closed-form 的解, 即
        $$
        \mathbf{V} = (\mathbf{I} - \gamma \mathbf{P})^{-1} \cdot \mathbf{R}
        $$
        然而解析计算本身的复杂度为 $\mathcal{O}(n^3)$, 这在往往较大规模的状态空间下是不可行的. 因此在实践中， 通常使用诸如动态规划、蒙特卡洛或时序差分等方法具体求解 Bellman Equation 的数值解.


对 MRP 进一步扩展, 就可以得到 Markov Decision Process (MDP). 在 MRP 中, 状态的转移时随机的, 智能体没有任何的控制权. 而在 MDP 中, 智能体可以通过选择不同的 action 来影响状态的转移. 因此 MDP 是一个五元组 $\langle \mathcal{S}, \mathcal{A}, \mathbf{P}, r, \gamma \rangle$:
- $\mathcal{S}$ 是有限状态集合, $\mathcal{A}$ 是有限 action 集合. ${P}$ 是一个转移概率, 其中 $P(s' \mid s, a)$ 表示在状态 $s$ 下采取 action $a$ 后转移到状态 $s'$ 的概率:
    $$
    P(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)
    $$
    有时也简记为 $P_{ij}^a = P(s_j \mid s_i, a)$. 此时往往不再使用矩阵方式进行表示, 是因为在引入 action 之后, 转移概率本身是一个三维的 tensor 结构. 
- 对于奖励函数 $r$, 其有两种常见的定义方式:
    - $r(s, a)$: 表示在状态 $s$ 下采取 action $a$ 后获得的 reward 的期望值. 其定义为
        $$
        r(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]
        $$
    - $r(s, a, s')$: 表示在状态 $s$ 下采取 action $a$ 后转移到状态 $s'$ 时获得的 reward 的期望值. 其定义为
        $$
        r(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']
        $$
    - 二者有关系:
        $$
        r(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) r(s, a, s')
        $$

- 智能体此时的交互流程为, 在时刻 $t$, 其处在状态 $S_t$, 选择 action $A_t$, 环境根据转移概率 $P(s' \mid s, a)$ 转移到下一个状态 $S_{t+1}$, 并且根据奖励函数 $r(s, a)$ 或 $r(s, a, s')$ 反馈给智能体一个 reward $R_{t+1}$. 整体流程为
    $$
    \cdots \rightarrow  S_t \xrightarrow{A_t} S_{t+1}, R_{t+1} \rightarrow \cdots
    $$

- 通常通过引入一个 policy $\pi: \mathcal{S} \to \mathcal{A}$ 来指导智能体每一步 action 的选择. 其中 $\pi(a \mid s)$ 表示在状态 $s$ 下选择 action $a$ 的概率. 其定义为
    $$
    \pi(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)
    $$

- 类似地, 可以引入 **State-Value Function** 来描述当智能体在 $\pi$ 指导下, 从状态 $s$ 出发的期望 return:
    $$
    V^{\pi}(s) = \mathbb{E}_{\pi}[G_t \mid S_t = s]
    $$
    以及 **Action-Value Function** 来描述当智能体在 $\pi$ 指导下, 从状态 $s$ 出发且给定采取的 action $a$ 时的期望 return:
    $$
    Q^{\pi}(s, a) = \mathbb{E}_{\pi}[G_t \mid S_t = s, A_t = a]
    $$
    - 因此状态价值函数可以通过动作价值函数进行表达, 即
        $$
        V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^{\pi}(s, a) \quad (1)
        $$
        即在 policy $\pi$ 下, 从状态 $s$ 出发的期望就是在状态 $s$ 下采取 action $a$ 的期望的加权平均, 其中权重就是 policy $\pi$ 在状态 $s$ 下选择 action $a$ 的概率. 
    - 反过来, 动作价值函数也可以通过状态价值函数进行表达, 即
        $$
        Q^{\pi}(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^{\pi}(s') \quad (2)
        $$
        即在状态 $s$ 下采取 action $a$ 的期望 return 就等于在状态 $s$ 下采取 action $a$ 后立即获得的即时 reward $r(s, a)$ 加上在状态 $s$ 下采取 action $a$ 后所有可能转移到的下一个状态 $s'$ 的期望 return 的加权平均, 其中权重就是在状态 $s$ 下采取 action $a$ 后转移到状态 $s'$ 的概率.
    - 若再分别将 $(1)$ 的 $V^{\pi}(s)$ 的具体表达代入 $(2)$, 及反过来将 $(2)$ 的 $Q^{\pi}(s, a)$ 的具体表达代入 $(1)$, 就可以得到如下的 **Bellman Expectation Equation** 的两种不同的递推形式:
        $$
        \begin{aligned}
        V^{\pi}(s) &= \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^{\pi}(s')\right] \\
        Q^{\pi}(s, a) &= r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^{\pi}(s', a')
        \end{aligned}
        $$


事实上, 给定一个 MDP 和一个固定策略 $\pi$ (策略本身可以还是一个随机策略, 依概率选择 action, 但本身的概率分布需要是固定的), 就可以将其转化为一个 MRP, 即通过 marginalize 的方法对所有可能的 action 进行加权平均.
- 在 MDP 中, reward 依赖状态和动作 $r(s,a)$, 但若 $\pi$ 是一个固定的策略, 则在状态 $s$ 下采取 action $a$ 的概率为 $\pi(a \mid s)$, 从而可以通过加权平均的方式得到在状态 $s$ 下的 reward 的期望值:
    $$
    r^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) r(s, a)
    $$
- 同样地, 在 MDP 中, 转移概率依赖状态和动作 $P(s' \mid s, a)$, 也可以通过对所有可能的 action 进行加权平均的方式得到在状态 $s$ 下转移到状态 $s'$ 的概率:
    $$
    P^{\pi}(s' \mid s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) P(s' \mid s, a)
    $$
- 因此可以看作一个新的 MRP $\langle \mathcal{S}, \mathbf{P}^{\pi}, r^{\pi}, \gamma \rangle$, 且其 Bellman Equation 可以写作:
    $$
    \begin{aligned}
    V^{\pi}(s) &= \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^{\pi}(s')\right] \\
    &= r^{\pi}(s) + \gamma \sum_{s' \in \mathcal{S}} P^{\pi}(s' \mid s) V^{\pi}(s')
    \end{aligned}
    $$
    其形式上和 MRP 的 Bellman Equation 是完全一样的, 只是其中的 reward 和转移概率都是通过对 action 进行加权平均诱导得到的. 

- 故同样的也可以写作如下的矩阵形式:
    $$
    \mathbf{V}^{\pi} = \mathbf{R}^{\pi} + \gamma \mathbf{P}^{\pi} \cdot \mathbf{V}^{\pi}
    $$
    从而可以得到 closed-form 的解为
    $$
    \mathbf{V}^{\pi} = (\mathbf{I} - \gamma \mathbf{P}^{\pi})^{-1} \cdot \mathbf{R}^{\pi}
    $$


最后, MDP 的一个重要问题是如何找到一个最优策略 $\pi^*$, 使得其对应的 value function $V^{\pi^*}$ 是所有策略中最大的. 
- 首先定义策略的好坏.   
  - 对于任意两个策略 $\pi$ 和 $\pi'$, 若对于所有状态 $s \in \mathcal{S}$ 都满足 $V^{\pi}(s) \geq V^{\pi'}(s)$, 则称 $\pi$ 优于 $\pi'$, 记作 $\pi \succeq \pi'$. 
  - 对应的, 如果存在一个策略 $\pi^*$, 使得对于所有策略 $\pi$, 都满足 $\pi^* \succeq \pi$, 则称 $\pi^*$ 是一个 optimal policy.

- 定义 optimal value function 为
    $$
    V^*(s) = \max_{\pi} V^{\pi}(s)
    $$
    即对于每个状态 $s$, 其 optimal value function 就是所有策略在该状态下的 value function 的最大值.  以及 optimal action-value function 为
    $$
    Q^*(s, a) = \max_{\pi} Q^{\pi}(s, a)
    $$
    即在状态 $s$ 下给定已经采取了 action $a$, 在此之后的最优行动的 optimal value function 的最大值. 
    
    
- 故二者有关系
    $$
    Q^*(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s')
    $$
    - 即表示在状态 $s$ 下采取 action $a$ 的 optimal action-value function 就等于在状态 $s$ 下采取 action $a$ 后立即获得的 reward 加上在状态 $s$ 下采取 action $a$ 后所有可能转移到的下一个状态 $s'$ 的 optimal value function 的加权平均, 其中权重就是在状态 $s$ 下采取 action $a$ 后转移到状态 $s'$ 的概率.
- 另一方面, 还有
    $$
    V^*(s) = \max_{a \in \mathcal{A}} Q^*(s, a)
    $$


- 根据二者的这两重关系, 可以得到如下的 **Bellman Optimality Equation**:
    $$
    \begin{aligned}
    V^*(s) &= \max_{a \in \mathcal{A}} \left[r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s')\right] \\
    Q^*(s, a) &= r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a' \in \mathcal{A}} Q^*(s', a')
    \end{aligned}
    $$


- 因此, 最优策略 $\pi^*$ 可以通过 optimal action-value function 来得到, 即
    $$
    \pi^*(s) \in \arg\max_{a \in \mathcal{A}} Q^*(s, a).
    $$