---
aliases: [几何分布随机数生成, Generating Geometric Random Variable]
tags:
  - method
  - math/probability
  - stat/computational
related_concepts:
  - "[[Geometric_Distribution]]"
  - "[[Generating_Discrete_Random_Variables]]"
  - "[[Generating_Exponential_Random_Variable]]"
source: "计算统计学"
---

# 从 U(0,1) 生成几何分布随机变量

#ComputationalStatistics 

## 目标

生成随机变量$X$满足参数为$p$的几何分布，即：
$$\begin{aligned}
  P\{X=i\}=p q^{i-1}, \quad i \geq 1, \quad \text { where } q=1-p
  \end{aligned}$$
**解：**
由几何分布的含义，$X$可认为是$n$次独立实验中首次成功的时间，且每次实验的成功概为$p$，故有：$$
\begin{aligned}
\sum_{i=1}^{j-1} P\{X=i\} &=1-P\{X>j-1\} \\
&=1-P\{\text { first } j-1 \text { trials are all failures }\} \\
&=1-q^{j-1}, \quad j \geq 1
\end{aligned}
$$
故可以生成随机数$U$并令：
$$
1-q^{j-1} \leq U<1-q^j ~~ \Rightarrow ~~q^j<1-U \leq q^{j-1}
$$
因此$X$为：
$$
X=\operatorname{Min}\left\{j: q^j<1-U\right\}~~\cdots(\star)
$$
下需解出$j$的具体数值。由对数函数的单调性，对$\star$式集合中不等式两侧求对数，有：
$$
  \begin{aligned}
  X &=\operatorname{Min}\{j: j \log (q)<\log (1-U)\} \\
  &=\operatorname{Min}\left\{j: j>\frac{\log (1-U)}{\log (q)}\right\}
  \end{aligned}
  $$
若用记号$\operatorname{Int}(x)$表示“不大于$x$的最大整数”，则有：
$$
X=\operatorname{Int}\left(\frac{\log (1-U)}{\log (q)}\right)+1
$$
其等价于：
$$
\boxed{X \equiv \operatorname{Int}\left(\frac{\log (U)}{\log (q)}\right)+1}
$$