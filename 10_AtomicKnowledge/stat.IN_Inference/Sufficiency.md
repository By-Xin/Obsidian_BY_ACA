---
aliases: [充分性, Sufficiency, Sufficient Statistic, 充分统计量]
tags:
  - concept
  - theorem
  - math/statistics
  - stat/inference
related_concepts:
  - "[[UMVUE]]"
  - "[[Neyman_Factorization_Theorem]]"
  - "[[Fisher_Information]]"
---

# Sufficiency

## Introduction
R.A. Fisher (1922) proposed the concept of sufficiency, which is defined as follows:
Given a random sample $\mathbf{X} = (X_1, \cdots, X_n)^T$ from a distribution with pdf $f(\mathbf{x},\theta)$, a statistic $S(\mathbf{X})$ is called a sufficient statistic for $\theta$ if $f(\mathbf{x},\theta| S(\mathbf{X}))$ is independent of $\theta$.

*[Example]*
Consider $X_1, \cdots, X_n \sim_{i.i.d.} \text{Bernoulli}(p)$. The likelihood function is
$$
L(p) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i} = p^{\sum_{i=1}^n x_i}(1-p)^{n-\sum_{i=1}^n x_i}
$$ The conditional distribution of $\mathbf{X}$ given $S(\mathbf{X}) = \sum_{i=1}^n X_i$ is
$$
f(\mathbf{x}| S(\mathbf{X})) = \frac{\mathbb{P}(X_1 = x_1, \cdots, X_n = x_n)}{\mathbb{P}(S(\mathbf{X}) = \sum_{i=1}^n x_i = t)} = \frac{p^t(1-p)^{n-t}}{\binom{n}{t} p^t (1-p)^{n-t}} = \binom{n}{t}^{-1}
$$ which is a function of $S(\mathbf{X})$ only without $p$. Therefore, $S(\mathbf{X})$ is a sufficient statistic for $p$.

> This actually is natural regarding the distribution. As long as $\sum_{i=1}^n X_i$ is given, the total number of $X_i = 1$ and $X_i = 0$ are determined, and thus the probability $p$ is determined. The randomness about $\{\sum X_k = t\}$ here is the permutation, or the order of those $0$s and $1$s (whose probability is $\binom{n}{t}^{-1}$, a uniform distribution), which is irrelevant to $p$.


## [[Neyman_Factorization_Theorem]]

*[Theorem]* **(Neyman Factorization Theorem)**
A statistic $S(\mathbf{X})$ is a sufficient statistic for $\theta$ if and only if the probability density function can be written as
$$
f(\mathbf{x}, \theta) = h(\mathbf{x}) g(S(\mathbf{x}), \theta)
$$ for some functions $h(\mathbf{x})$ and $g(S(\mathbf{x}), \theta)$.

