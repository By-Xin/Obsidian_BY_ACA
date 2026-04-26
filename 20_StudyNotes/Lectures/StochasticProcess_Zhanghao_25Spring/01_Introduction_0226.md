# Introduction to Stochastic Process

## Introduction

随机过程的研究核心是多个随机变量 $X_1, X_2, \ldots$ 的关联. 在统计学的研究中, 对于多个随机变量, 往往考虑是独立同分布的. 然而, 在随机过程中, 这多个随机变量之间是有相关性的. 其中, 对于相关性的刻画, 有如下视角:
- 线性相关性:
  - Correlation: 从时域的角度来刻画相关性 $\leftrightarrow$ Spectral density: 从频域的角度来刻画相关性.
  - 一个典型的随机过程是 Gaussian process. 几乎所有的能够进行相关性的系统分析的随机过程都是基于 Gaussian 的. 
- Markov property: 
  - 对于多个随机变量的完整刻画 (联合分布) $\mathbb{P}(X_1, \ldots, X_n)$, 其计算是非常复杂的. 而 Markov property 的核心思想在于, 而首先考虑条件概率, 其能够让我们缩小研究的范围, 条件住一部分随机变量的随机性, 从而简化联合分布的计算.
    $$
    \mathbb{P}(X_1, \ldots, X_n) = \mathbb{P}(X_1) \mathbb{P}(X_2 | X_1) \cdots \mathbb{P}(X_n | X_{n-1}, \ldots, X_1)
    $$
    不过仅仅是这样还是不够的, 其只是相当于将问题转化为了约束. 

  - 为了进一步解决这个问题, 依靠 Markov property 的假定, 其认为 $\mathbb{P}(X_n | X_{n-1}, \ldots, X_1) = \mathbb{P}(X_n | X_{n-1})$. 这样就大大简化了联合分布的计算:
    $$
    \mathbb{P}(X_1, \ldots, X_n) = \mathbb{P}(X_n | X_{n-1}) \cdots \mathbb{P}(X_2 | X_1) \mathbb{P}(X_1)
    $$

  - Markov Chain 的一个典型的随机过程是 Poisson process. 并且 Markov Chain 在当前的前沿研究中也是非常重要的一个工具. 例如在强化学习中, Markov Decision Process 是一个非常重要的模型.


## Relationship between Random Variables

给定两个随机变量 $X$ 和 $Y$, 其关联性的一个直观刻画是它们的距离. 首先考虑其均方距离:
$$
d^2(X, Y) = \mathbb{E}\left( |X - Y|^2 \right) = \mathbb{E}\left( |X|^2 \right) + \mathbb{E}\left( |Y|^2 \right) - 2 \mathbb{E}\left( XY \right)
$$
而其中最值得注意的就是 $\mathbb{E}\left( XY \right)$ 这个项. 其刻画了 $X$ 和 $Y$ 的相关性. 而其本质即为 $X$ 和 $Y$ 在 Hilbert space 中的内积. 其中每个随机变量都等效于空间中的一个向量. 