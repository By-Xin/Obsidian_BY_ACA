---
aliases:
  - "Lipschitz 连续"
  - "Lipschitz Continuous"
tags:
  - concept
  - math/optimization
related_concepts:
  - "[[Holder_Continuous]]"
  - "[[Ck_Functions]]"
---

# Lipschitz Continuous

***Definition* (Lipschitz Continuous)** 一个函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是 Lipschitz 连续的 (Lipschitz continuous) 若存在常数 $L > 0$, 使得对于任意 $x,y \in \mathbb{R}^n$, 有:
$$
|f(x) - f(y)| \leq L \|x-y\|_2
$$

对于凸函数, Lipschitz 连续性等价于其梯度有界: 对所有 $g \in \partial f(x)$, 有 $\|g\|_2 \leq L$.
