---
aliases: [收敛性, Convergence, 随机变量的收敛, Convergence of Random Variables, 依概率收敛, 依分布收敛]
tags:
  - concept
  - math/probability
  - math/analysis
related_concepts:
  - "[[Central_Limit_Theorem]]"
  - "[[Law_of_Large_Numbers]]"
  - "[[Slutsky_Theorem]]"
  - "[[Chebyshev_Inequality]]"
  - "[[Uniformly_Continuous]]"
  - "[[MLE]]"
source: "概率论与数理统计基础"
---

# Convergence in Statistics

## Introduction

### Convergence of Sequences

Convergence is not a new concept in mathematics. In calculus, we define convergence of a sequence $X_n \to X~ (n \to \infty)$ as:

$$
\forall \epsilon > 0, \exists N \in \mathbb{N},~~\\ \text{s.t.}~ \forall n \geq N, |X_n - X| < \epsilon
$$

Or in vector form, we can say that a sequence of vectors $X_n \to X~ (n \to \infty)$ if:

$$
\forall \epsilon > 0, \exists N \in \mathbb{N},~~\\ \text{s.t.}~ \forall n \geq N, ||X_n - X|| < \epsilon.
$$

Here, when we talk about *CONVERGENCE*, we are talking about *limits*, and when we talk about *limits*, we are talking about *approximations*. Thus as a result, we need a *distance metric* to measure the *closeness* of two points. And this can be shown by $|X_n - X|$ or $||X_n - X||$ (actually they serve as the Euclidean distance).

### Convergence of Functions

Furthermore, recall that, in stochastic calculus, $X_n$ is a random variable, and random variables are functions from $\Omega \to \mathbb{R}$. Thus, convergence of random variables is essentially the limit of a sequence of functions, which is also familiar to us in calculus.

Assume that $\{f_n(x)\}$ is a sequence of functions, and $f(x)$ is a function, which all map from $A\subseteq \mathbb{R} \to B\subseteq \mathbb{R}$. Then we can define the convergence of functions as:

***Pointwise Convergence***

$$
\forall x \in A, \forall \epsilon > 0, \exists N \in \mathbb{N},~~\\ {\textbf{s.t.}}~ {\forall n \geq N}, |f_n(x) - f(x)| < \epsilon.
$$

*[Example]* $f_n(x) = x^n$ converges pointwise to $f(x) = 0$ on $[0,1)$.

***Uniform Convergence***

$$
\forall \epsilon > 0, \exists N \in \mathbb{N},~~\\ \textbf{s.t.}~ \forall x \in A, \forall n>N, |f_n(x) - f(x)| < \epsilon.
$$

Notice the difference between pointwise and uniform convergence is the position of the quantifiers(i.e., $\forall x$ and $\forall n$). In pointwise convergence, $\exists N_{x,\epsilon}$ is dependent on $x$ and $\epsilon$, while in uniform convergence, $\exists N_{\epsilon}$ is independent of $x$. It means for uniform convergence, the rate of convergence is the same for all $x$.

***Intergal Convergence***

$$
||f_n - f||=\int_A |f_n(x) - f(x)|dx \to 0,~ \text{as}~ n \to \infty.
$$

$\square$

In this note, we will introduce 4 types of convergence in probability theory as listed below. And essential differences between them is the way how we measure the distance between random variables.

- Mean Square Convergence
- Convergence in Probability
- Almost Sure Convergence
- Convergence in Distribution

## Convergence of Random Variables

### Mean Square Convergence

In Mean Square Convergence, we measure the distance between two random variables by the mean square error, which is defined as:
    $$
    d(X,Y) = \sqrt{\mathbb{E}|X-Y|^2}.
    $$
*[Definition]* $X_n$ converges to $X$ in mean square ($X_n \xrightarrow{ms} X$) if:
    $$
    \lim_{n \to \infty} \mathbb{E}|X_n - X|^2 = 0.
    \\
    (\forall \epsilon > 0, \exists N \in \mathbb{N},~~ \text{s.t.}~ \forall n \geq N, \mathbb{E}|X_n - X|^2 < \epsilon).
    $$

*[Properties]* (Given $X_n \xrightarrow{ms} X$ and $Y_n \xrightarrow{ms} Y$):

  - $X_n+Y_n \xrightarrow{ms} X+Y ~~(n \to \infty)$
    *[Proof]*
  $$
    d(X_n+Y_n, X+Y) \leq d(X_n, X) + d(Y_n, Y) \xrightarrow{ms} 0
  $$

  - $X_nY_n \xrightarrow{ms} XY ~~(n \to \infty)$
    *[Proof]*
  $$
  ||X_nY_n - XY|| = ||X_nY_n - X_nY + X_nY - XY||\\ \leq ||X_n||\cdot||Y_n - Y|| + ||Y||\cdot||X_n - X|| \\ \leq ||X_n||\cdot||Y_n - Y|| + ||Y||\cdot||X_n - X|| \xrightarrow{ms} 0
  $$

  - Cauchy Criterion: $||X_n - X_m|| \xrightarrow{ms} 0 ~~(n,m \to \infty) \Rightarrow \exists X, X_n \xrightarrow{ms} X$.

  
Note that, in statistics, consistency is a concept similar to mean square convergence. If a sequence of estimators $\hat{\theta}_n$ converges to $\theta$ in mean square, then we say $\hat{\theta}_n$ is a consistent estimator of $\theta$.

> Actually, convergence in mean square is a special case of *Convergence in Lp* (with $p=2$), which is generally defined as:
> $$
> \lim_{n \to \infty} \mathbb{E}|X_n - X|^p = 0.
> $$
> Thus sometimes we also note it as $X_n \xrightarrow{L_2} X$.

### Almost Surely Convergence

*[Definition]* A sequence of random variables $X_1, X_2, \cdots$, coverges almost surely to a random variable $X$ (denote as $X_n \xrightarrow{a.s.} X$) if, for every $\epsilon > 0$:
    $$
    \mathbb{P}(\lim_{n \to \infty} |X_n - X| < \epsilon) = 1.
    $$
    Or equivalently,
    $$
    \mathbb{P}\{\omega \in \Omega: X_n(\omega) \to X(\omega)\}= 1.
    $$
    
- Here, *almost surely* means $X_n$ converges to $X$  for *almost all* $\omega \in \Omega$, except for perhaps some trivial points $\omega$, where $\omega \in N \subseteq \Omega$, with $\mathbb{P}(N) = 0$. 
- It focuses on those sample points that are supportingly making the random variables converge, and ignroes the zero-probability set. Thus, $ \text{a.s.} \approx \text{pointwise} - \text{zero-probability set}$.

*[Example 1]* Let sample space $\Omega = [0,1]$ with uniform distribution. Define $X_n(\omega) = \omega + \omega^n$, and $X(\omega) = \omega$. Then for any $\omega \in [0,1)$, $\omega^n \to 0$ as $n \to \infty$. Thus, $X_n \to \omega = X$ for almost all $\omega \in \Omega$. The only exception is $X_n(1) = 2$, but it doesn't matter since the convergence occurs on $[0,1)$, and $\mathbb{P}\left([0,1)\right) = 1, \mathbb{P}(\{1\}) = 0$.

*[Property]* **Almost surely convergence and Mean Square Convergence has no direct relationship either way**, i.e. $X_n \xrightarrow{a.s.} X \nLeftrightarrow
X_n \xrightarrow{ms} X$. 

*[Theorem]* **(Strong Law of Large Numbers)** **Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with $\mathbb{E}[X_i]=\mu, \operatorname{Var}[X_i]=\sigma^2 < \infty$. Denote $\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i$. Then, for any $\epsilon > 0, \mathbb{P}(\lim_{n \to \infty} |\bar{X}_n - \mu| < \epsilon) = 1$.**

### Convergence in Probability

*[Definition]* A sequence of random variables $X_1, X_2, \cdots$, converges in probability to a random variable $X$ (denote as $X_n \xrightarrow{p} X$) if, for every $\epsilon > 0$:
$$
\lim_{n \to \infty} \mathbb{P}(|X_n - X| < \epsilon) = 1.
$$
Or equivalently,
$$
\lim_{n \to \infty} \mathbb{P}(|X_n - X| \geq \epsilon) = 0.
$$
- It means that those outliers $\omega$ that trying to make $X_n(\omega)$ far away from $X(\omega)$ are getting less and less as $n \to \infty$.

*[Theorem]* **If $X_n \xrightarrow{a.s.} X$, then $X_n \xrightarrow{p} X$.**

*[Theorem]* **Suppose $X_1, X_2, \cdots$ converges to $X$ in probability, with $h(\cdot)$ being a continuous function. Then $h(X_1), h(X_2), \cdots$ converges to $h(X)$ in probability.**

*[Theorem]* **(Weak Law of Large Numbers)** **Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with $\mathbb{E}[X_i]=\mu, \operatorname{Var}[X_i]=\sigma^2 < \infty$. Denote $\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i$. Then, for any $\epsilon > 0, \lim_{n \to \infty} \mathbb{P}(|\bar{X}_n - \mu| < \epsilon) = 1$.**

### Convergence in Distribution / Weak Convergence

*[Definition]* A sequence of random variables $X_1, X_2, \cdots$, converges in distribution to a random variable $X$ if, (denote as $X_n \xrightarrow{d/L} X$, or $F_n(x) \xrightarrow{w} F_X(x)$) if:
for every $x \in \mathbb{R}$ where $F_X(x)$ is continuous at $x$:
$$
\lim_{n \to \infty} F_{X_n}(x) = F_X(x).
$$
Or equivalently, if $X_n \sim F_{X_n}$ and $X \sim F_X$, then $X_n \xrightarrow{d} X$ iif $F_{X_n}(x) \xrightarrow{n \to \infty} F_X(x)$ for all $x$ where $F_X(x)$ is continuous.

*[Theorem]* **(Slutsky's Theorem)** Let $X_n \xrightarrow{d} X, Y_n \xrightarrow{p} c$(a constant). Then:
- $X_n + Y_n \xrightarrow{d} X + c$.
- $X_nY_n \xrightarrow{d} cX$.

*[Theorem]* **If $X_n \xrightarrow{p} X$, then $X_n \xrightarrow{d} X$.**

*[Corollary]* **If $X_n \xrightarrow{a.s.} X$, then $X_n \xrightarrow{d} X$.**

> **Note**: Convergence in Distribution is essentially the convergence of the CDFs, rather than the random variables themselves. It is different from the other two types of convergence, yet both can imply Convergence in Distribution.

*[Theorem]* **Random variables $X_n$ converges in probability to some *constant* $\mu$ *if and only if* $X_n$ converges in distribution to $\mu$** ($X_n \xrightarrow{p} \mu \Leftrightarrow X_n \xrightarrow{d} \mu$)

## Relationship Summary

$$
\begin{align*}
&\text{Almost Sure Convergence} \Rightarrow \text{Convergence in Probability} \Rightarrow \text{Convergence in Distribution}\\
&\text{Mean Square Convergence} \Rightarrow \text{Convergence in Probability}
\end{align*}
$$

However, there is no direct relationship between Almost Sure Convergence and Mean Square Convergence.


