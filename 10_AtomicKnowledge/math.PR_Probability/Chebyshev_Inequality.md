---
aliases: [切比雪夫不等式, Chebyshev Inequality, 集中不等式, Concentration Inequality]
tags:
  - theorem
  - math/probability
related_concepts:
  - "[[Convergence_in_Statistics]]"
  - "[[Law_of_Large_Numbers]]"
  - "[[Central_Limit_Theorem]]"
  - "[[Markov_Inequality]]"
source: "概率论基础"
---

# 切比雪夫不等式 (Chebyshev Inequality)

#Probability

## 定理陈述

**Chebyshev Inequality** (concentration) $\forall \epsilon > 0$, we have
$$
\mathbb{P}(|X| \geq \epsilon) \leq \frac{\mathbb{E}[X^2]}{\epsilon^2}.
$$

## 证明

$$\mathbb{E}|X|^2 = \int_{\{|X| \geq \epsilon\}~ \cup~ \{|X| < \epsilon\}} |x|^2 f_X(x) dx \geq \int_{|X| \geq \epsilon} x^2 f_X(x) dx \geq  \int_{|X| \geq \epsilon} \epsilon^2 f_X(x) dx = \epsilon^2 \mathbb{P}(|X| \geq \epsilon).$$

## 推论

$$\mathbb{P}(|X-\mathbb{E}[X]| \geq \epsilon) \leq \frac{\operatorname{Var}[X]}{\epsilon^2}.$$

## 相关概念

- [[Convergence_in_Statistics]] - 切比雪夫不等式用于证明大数定律
- [[Law_of_Large_Numbers]] - 弱大数定律的证明依赖本不等式