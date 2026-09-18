# Lecture 1

## 随机事件

- 事件 $A$ 总是对应于某个特定空间 $\Omega$ (所有可能结果之全体, 称为样本空间) 的某些子集 $A \subset \Omega$. $\omega \in\Omega$ 是样本空间的基本单元, 我们认为 $A$ 事件发生等价于 $\omega \in A$. 
- 记 $\Omega$ 所有可能的子集 (即所有可能的事件) 的构成的集合为 $2^\Omega$, 称为 power set. 理论上, power set 是我们最终想要研究概率论的对象, 但从数学上可以说明, 我们并不总能良好的定义这样的集合. 
- 在现代的概率论中, 我们不期望能够对 $2^\Omega$ 定义概率测度. 而是寻找一个足够丰富, 有恰好够用 (对概率操作封闭) 的集合族, 即下文的  $\sigma$-algebra. 

## 公理化概率论

$\sigma$-algebra 是现代概率论中我们可以 '合法' 地谈论概率的范围. 

***Definition* ($\sigma$-algebra)**. 设 $\Omega$ 非空, $\mathcal{F}$ 是 $\Omega$ 的一些子集构成的 collection. 称 $\mathcal{F}$ 是一个 sigma-algebra, 若:
1. $\Omega \in \mathcal{F}$;
2. 对于补集封闭: 若 $E \in \mathcal{F}$, 则 $E^c \in \mathcal{F}$;
3. 对于可列并封闭: 若 $E_1, E_2, \cdots$, 则 $\bigcup_{n=1}^\infty E_n \in \mathcal{F}$.

称 $(\Omega, \mathcal{F})$ 为一个可测空间. 

这三点定义是自然的. 其要求: (1) 样本空间本身当然应该能谈概率; (2) 如果事件*发生*能够定义概率, 那么事件*不发生*也应该能定义概率; (3) 如果每个事件 $A_n$ 都能定义概率, 那么*至少*有一个事件发生也应该能定义规律 (在可列个意义下).

从上述三点定义, 可以自然推出许多性质. 若 $\mathcal{F}$ 是 $\sigma$-algebra, 则 $\emptyset \in \mathcal{F}$, 