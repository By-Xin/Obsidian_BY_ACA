---
aliases:
  - "Hölder 连续"
  - "Hölder Continuous"
  - "Holder Continuous"
tags:
  - concept
  - math/optimization
related_concepts:
  - "[[Lipschitz_Continuous]]"
  - "[[Ck_Functions]]"
---

# Hölder Continuous

***Definition* (Hölder Continuous)** 一个函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是 Hölder 连续的 (Hölder continuous) 若存在常数 $L > 0$, 使得对于任意 $x,y \in \mathbb{R}^n$, 有:
$$
|f(x) - f(y)| \leq L \|x-y\|^\alpha, ~\alpha \in (0, 1]
$$

对比连续的定义: 对于任意 $\epsilon > 0$, 存在 $\delta > 0$, 使得对于任意 $x,y \in \text{dom}(f)$, 有:
$$
|f(x) - f(y)| < \epsilon
$$


连续性本身只对函数的距离有要求, 但没有给出相近程度的上界. 而 Hölder 连续性则要求连续的上界是一个明确的关于 $\|x-y\|$ 的幂次.
