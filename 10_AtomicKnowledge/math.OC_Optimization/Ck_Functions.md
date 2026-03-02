---
aliases:
  - "C^k Functions"
  - "C^k 函数"
  - "C^{k, alpha} Functions"
  - "C^{k,alpha} 函数"
  - "C1 Functions"
tags:
  - concept
  - math/optimization
related_concepts:
  - "[[Holder_Continuous]]"
  - "[[Lipschitz_Continuous]]"
---

# $C^k$ Functions

***Definition* ($C^k$ Functions)** 一个函数 $f: \mathbb{R}^n \to \mathbb{R}$ 属于 $C^k$ 类, 若其 $k$ 阶导数存在且连续.

- 例如: $C^0$ 表示函数本身连续, $C^1$ 表示函数一阶导数存在且连续, $C^\infty$ 表示函数无穷阶导数存在且连续.

***Definition* ($C^{k, \alpha}$ Functions)** 一个函数 $f: \mathbb{R}^n \to \mathbb{R}$ 属于 $C^{k, \alpha}$ 类, 若函数是 $C^k$ 类且其 $k$ 阶导数满足 $\alpha$-Hölder 条件, 即:
$$
\|\nabla^k f(x) - \nabla^k f(y)\| \leq L \|x-y\|^\alpha
$$
其中 $L > 0$ 是常数, $\alpha \in (0, 1]$.

例如, $C^{0,\alpha}$ 表示函数本身是 $\alpha$-[[Holder_Continuous]]的. 
$$
| f(x) - f(y) | \leq L \|x-y\|^\alpha, ~\alpha \in (0, 1]
$$
特别地, 当 $\alpha = 1$ 时, 称为 [[Lipschitz_Continuous]]:
$$
| f(x) - f(y) | \leq L \|x-y\| 
$$

> [!note]
>
> 可以证明, 在有限维空间中, 任意 Norm 都是等价的. 因此尽管默认为 $\ell_2$ Norm, 但也可以使用其他 Norm.


此外, 有如下推导关系存在: 若函数是 $C^{k+1}$ 类 ($k+1$ 阶导数存在且连续), 则第 $k$ 阶导数必然是 [[Lipschitz_Continuous]], 则其必然是 [[Holder_Continuous]]的. 即:
$$
C^{k+1} \Rightarrow C^{k, 1} \Rightarrow C^{k, \alpha} \Rightarrow C^{k}
$$
其中 $\alpha \in (0, 1]$.
