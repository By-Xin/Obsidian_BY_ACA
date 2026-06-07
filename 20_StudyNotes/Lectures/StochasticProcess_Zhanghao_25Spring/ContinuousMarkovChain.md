# Continuous Time Markov Chain

## Introduction to CTMC

***Definition* (Continuous Time Markov Chain)**: 考虑随机过程 $\{X(t), t \geq 0\}$, 其中 $t \in [0, \infty)$ 是连续时间参数, $X(t) \in S$ 是至多可数的离散状态空间,  通常 $S = \{0, 1, 2, \ldots\}$. 对于 CTMC, 其满足马尔可夫性质: 
$$
\mathbb{P}(X(t+s) = j \mid X(s) = i, \{X(u), 0 \leq u < s\}) = \mathbb{P}(X(t+s) = j \mid  X(s) = i) .
$$
- 通常假设 CTMC 是时间齐次 (time-homogeneous) 的, 即转移概率只依赖于时间间隔 $t$, 而与具体的起始时间 $s$ 无关:
    $$
    \mathbb{P}(X(t+s) = j \mid X(s) = i) = \mathbb{P}(X(t) = j \mid X(0) = i) = P_{ij}(t).
    $$

- 定义转移概率 
    $$
    p_{ij}(t) := \mathbb{P}(X(s+t) = j \mid X(s) = i) = \mathbb{P}(X(t) = j \mid X(0) = i) 
    $$
    则 CTMC 的转移概率矩阵为 $P(t) = \begin{bmatrix} p_{ij}(t) \end{bmatrix}_{i,j \in S}$.

另一方面, 还可以通过 **holding time** + **jump** 的方式来刻画 CTMC 的演化过程:
- 对于每个状态 $i \in S$, 定义在状态 $i$ 的停留时间 (holding time) 为一个随机变量 $H_i$, 表示 CTMC 在状态 $i$ 上停留的时间长度. 
  - 由于 CTMC 满足马尔可夫性质, 故当 CTMC 已经在停留了 $t$ 时间后, 其剩余的停留时间仍不依赖于已经停留的时间 $t$, 即 $H_i$ 具有无记忆性:
    $$
    \mathbb{P}(H_i > s + t \mid H_i > s) = \mathbb{P}(H_i > t).
    $$
    而可以证明, 在非负连续随机变量中, 只有指数分布具有无记忆性. 因此 $H_i \sim \text{Exp}(v_i)$, 其中 $v_i > 0$ 称为状态 $i$ 的 **total exit rate**. 
  - 由指数分布的性质:
      - $\mathbb{E}[H_i] = 1/v_i$, 即 CTMC 在状态 $i$ 上的平均停留时间为 $1/v_i$.
      - 对应的 **survival function** 为 $\mathbb{P}(H_i > t) = e^{-v_i t}$, 即 CTMC 在状态 $i$ 上停留超过 $t$ 时间的概率为 $e^{-v_i t}$. 若考虑一个很短的时间 $h$, 对其进行 Taylor 展开, 则
        $$
        \mathbb{P}(H_i > h) = e^{-v_i h} = 1 - v_i h + o(h).
        $$
        故**在一个很短的时间 $h$ 内, CTMC 从状态 $i$ 转移到其他状态的概率约为 $v_i h$, 而停留在状态 $i$ 的概率约为 $1 - v_i h$.**
- 当 CTMC 从状态 $i$ 转移到另一个状态 $j \neq i$ 时, 就称为一次 **jump**. 对于 jump 的选择仍然和离散时间 Markov chain 的处理类似: 定义一个 CTMC 的 **Embedded DTMC**: 记 CTMC 的 jump 发生在 $0 \leq T_0 < T_1 < T_2 < \ldots$ 时刻, 且定义 $n$ 次 jump 之后的状态为 $Y_n := X(T_n)$, 则 $\{Y_n, n \geq 0\}$ 就是 CTMC 的 embedded DTMC. 对于这个 DTMC, 记其转移概率为 
    $$
    a_{ij} := \mathbb{P}(Y_{n+1} = j \mid Y_n = i).
    $$
    - 通常规定 $a_{ii} = 0$ (即不考虑自己跳转到自己), $\sum_{j \neq i} a_{ij} = 1$ (即从状态 $i$ 跳转到其他状态的概率和为 1). 故有 Discrete Time Transition Matrix $\mathbf{A} = \begin{bmatrix} a_{ij} \end{bmatrix}_{i,j \in S}$.   其描述的是具体的每一次 jump 的转移情况, 而暂时不考虑具体的连续时间参数.

因此, CTMC 的演化可以看作是一个由 DTMC 的 jump + 指数分布的 holding time 组成的过程. 
- 进入状态 $i$ 后, CTMC 在该状态停留一个指数分布的时间 $H_i$, 然后根据 jump 的转移概率从状态 $i$ 转移到另一个状态. 
- 因此, 只要知道了每个状态的 total exit rate $v_i$ 和 embedded DTMC 的转移概率 $\mathbf{A} = (a_{ij})$, 就可以完全刻画 CTMC 的演化过程. 

### Generator Matrix

下根据上述分解思路给出进一步的详细推导. 考虑事件: 一个 CTMC 处在状态 $i$ 上, 在短时间 $h$ 内发生 jump 转移到 $j \neq i$ 上. 记 $H_i$ 为 CTMC 在状态 $i$ 上的停留时间, $Y_n = X(T_n)$ 为 CTMC 的 embedded DTMC, 则上述事件的概率为:
$$
\begin{aligned}
\mathbb{P} (H_i \leq h, Y_1 = j \mid Y_0 = i) & = \mathbb{P} (H_i \leq h \mid Y_0 = i) \cdot \mathbb{P}(Y_1 = j \mid H_i \leq h, Y_0 = i) \\
& = (1 - e^{-v_i h}) \cdot a_{ij} \\
& = v_i a_{ij} h + o(h) := q_{ij} h + o(h).
\end{aligned}
$$

这里, $q_{ij} := v_i a_{ij}, i \neq j$ 就是 CTMC 从状态 $i$ 跳转到状态 $j$ 的 **transition rate**, 表示 CTMC 在状态 $i$ 时, 以 $q_{ij}$ 的速率跳转到状态 $j$. 另外规定对角元 $q_{ii} := - \sum_{j \neq i} q_{ij}$ (该规定的合理性将在稍后说明). 则得到 CTMC 的 **transition rate matrix**, 或称 **generator matrix**:
$$
\mathbf{Q} = \begin{bmatrix} q_{ij} \end{bmatrix}_{i,j \in S}.
$$
- 注意到有如下性质:
  - $\sum_{j \neq i} q_{ij} = v_i$, 即 CTMC 从状态 $i$ 跳转到其他状态的 transition rate 之和等于 total exit rate $v_i$. 这可以由 $\sum_{j \neq i} a_{ij} = 1$ 和 $q_{ij} = v_i a_{ij}$ 立刻得到.
  - $\sum_{j} q_{ij} = 0$, 即 transition rate matrix / generator matrix 的每行行和为 0. 某种意义上, 这体现了转移状态的守恒, 即从 $i$ 跳转到其他状态的速率之和等于从其他状态跳转到 $i$ 的速率之和.

下分析 generator matrix 与 transition probability matrix 之间的关系. 考虑短时间 $h$ 的转移情况. 根据 holding time $H_i$ 的性质
$$
    \mathbb{P}(H_i > h) = e^{-v_i h} = 1 - v_i h + O(h^2). 
$$
因此这说明, 在短时间 $h$ 内, 进行 $0$ 次 jump 的概率约为 $1 - O(h)$, 进行 $1$ 次 jump 的概率约为 $O(h)$, 进行至少 $2$ 次 jump 的概率约为 $O(h^2)$. 
- 考虑 $p_{ii}(h)$. 这说明在 $h$ 时间内, 要么 CTMC 在状态 $i$ 上停留超过 $h$ 时间, 要么在 $h$ 时间内发生 jump 但又跳转回状态 $i$, 即至少发生 $2$ 次 jump. 因此
    $$
    p_{ii}(h) = 1 - v_i h + o(h) = 1 + q_{ii} h + o(h).
    $$

- 考虑 $p_{ij}(h), j \neq i$. 这说明在 $h$ 时间内, CTMC 从状态 $i$ 跳转到状态 $j$. 其同样可能是在 $h$ 时间内发生 $1$ 次 jump 且直接跳转到状态 $j$, 也可能是在 $h$ 时间内发生至少 $2$ 次 jump, 但最终停在状态 $j$. 因此
    $$
    p_{ij}(h) = (1-e^{-v_i h}) a_{ij} + o(h) = q_{ij} h + o(h).
    $$
综上所述, 对于任意 $i,j \in S$, 都有
$$
p_{ij}(h) = \delta_{ij} + q_{ij} h + o(h), \quad \text{其中} \quad \delta_{ij} = \bold{1}_{\{i=j\}}.
$$
故写成矩阵形式, 就得到了 generator matrix $\mathbf{Q}$ 与 transition probability matrix $P(h)$ 之间的关系:
$$
P(h) = \mathbf{I} + \mathbf{Q} h + o(h).
$$
或等价地
$$
\mathbf{Q} = \lim_{h \downarrow 0} \frac{P(h) - \mathbf{I}}{h}.
$$