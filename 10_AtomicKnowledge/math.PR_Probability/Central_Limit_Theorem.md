---
aliases: [中心极限定理, Central Limit Theorem, CLT, 林德伯格-列维定理, Lindeberg-Levy CLT]
tags:
  - theorem
  - math/probability
  - stat/inference
related_concepts:
  - "[[Law_of_Large_Numbers]]"
  - "[[Convergence_in_Statistics]]"
  - "[[Normal_Distribution]]"
  - "[[Chebyshev_Inequality]]"
  - "[[Exponential_Family]]"
  - "[[Hypothesis_Testing]]"
source: "概率论与数理统计基础"
---

# Central Limit Theorem

## Central Limit Theorem

Central Limit Theorem wants to show the limit distribution of $Y_n = X_1 + X_2 + \cdots + X_n$. Note that, to constrain the mean and variance as $n$ grows, we need to standardize it as $Y_n^* = \frac{Y_n - \mathbb{E}[Y_n]}{\sqrt{\operatorname{Var}[Y_n]}}$. 

### CLT with i.i.d. condition


- (Simplified Ver.) **Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with $\mathbb{E}[X_i]=0, \operatorname{Var}[X_i]=1$.Then, as $n \to \infty$, $\frac{1}{\sqrt{n}}\sum_{i=1}^n X_i \xrightarrow{d} \mathcal{N}(0,1)$.**
- (General Ver.) **Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables, and there exists MGF $M_X(t)$ for some neighborhood of 0 (i.e. $\exists \delta > 0, s.t. \forall t \in (-\delta, \delta), M_{X_i}(t)$ exists). Furthermore, $\mathbb{E}[X_i]=\mu, \operatorname{Var}[X_i]=\sigma^2 < \infty$, denote $\bar{X}_{(n)} = \frac{1}{n} \sum_{i=1}^n X_i$. Define $G_n(x)$ as the CDF of $\sqrt{n}\frac{\bar{X}_{(n)} - \mu}{\sigma}$. Then, $\forall x \in \mathbb{R}$, 
$$\lim_{n \to \infty} G_n(x) = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}} e^{-t^2/2} dt.$$**
- (Lindeberg-Levy CLT, strong CLT) **Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with $\mathbb{E}[X_i]=\mu, \operatorname{Var}[X_i]=\sigma^2 < \infty$. Then, denote $Y_n^* = \frac{\sum_{i=1}^n (X_i - \mu)}{\sigma \sqrt{n}}$. Then, $\forall x \in \mathbb{R}$,
$$\lim_{n \to \infty} \mathbb{P}(Y_n^* \leq y) = \Phi(y) = \int_{-\infty}^y \frac{1}{\sqrt{2\pi}} e^{-t^2/2} dt.$$**
  - *[Proof]* To prove the last equation, we only need to prove that $Y_n^* \xrightarrow{d} \mathcal{N}(0,1)$, and it is equivalent to show $\varphi_{Y_n^*}(t) \to e^{-t^2/2}$, which is the CF of $\mathcal{N}(0,1)$.
  Denote the CF of $X_n-\mu$ as $\varphi(t)$, then the CF of $Y_n^*$ is $\varphi_{Y_n^*}(t) = \varphi(\frac{t}{\sigma \sqrt{n}})^n$. Given that $\mathbb{E}[X_i]=\mu, \operatorname{Var}[X_i]=\sigma^2$, we can do the Taylor expansion of $\varphi(t)$ as
  $$\varphi(t) = \varphi(0) + \varphi'(0)t + \frac{\varphi''(0)}{2}t^2 + o(t^2) = 1 {- \frac{1}{2}\sigma^2 t^2} + o(t^2).$$Then, we have $\varphi_{Y_n^*}(t) = (1 - \frac{1}{2}\frac{t^2}{n} + o(\frac{t^2}{n}))^n \to e^{-t^2/2}$ as $n \to \infty$.
  $\square$


> CLT shows that, regardless of the original distribution of $X_i$, as long as the $n$ is sufficiently large (and i.i.d & variance exists), the distribution of $\bar{X}_{(n)}$ (or $\sum_{i=1}^n X_i$) will be approximately normal. This also indicates why the measurement error is often assumed to be normally distributed, since it is the sum of many small errors.

*[Example]* **(Generate $\mathcal{{N(0,1)}}$ from $\mathcal{U(0,1)}$ by Lindeberg-Levy CLT)** 
- Generate 12 i.i.d. $\mathcal{U(0,1)}$ random variables $X_1, X_2, \ldots, X_{12}$. Then, $\mathbb{E}[X_i] = \frac{1}{2}, \operatorname{Var}[X_i] = \frac{1}{12}$.
- Calculate $y = \sum_{i=1}^{12} X_i - 6 ~\dot\sim ~\mathcal{N}(0,1)$.
- Transform $y$ to $z = \sigma y + \mu = y ~\dot\sim \mathcal{N}(\mu, \sigma^2)$.
- Repeat the above steps for $n$ times, and we can get $n$ samples from $\mathcal{N}(\mu, \sigma^2)$.

### CLT with independent but not identical condition

In practice, the i.i.d. condition is often too strict. In many cases, the random variables are independent but not identical. Here, we still want to show that, even if the random variables $X_i$ are not identically distributed, given some conditions, the sum of them $Y_n = \sum_{i=1}^n X_i$ will be approximately normal.

#### Lindeberg-Feller CLT

In Lindeberg-Feller CLT, the key is to make each term $X_i$ "uniformly small" compared to the sum of all terms. Now we will derive the *Lindeberg-Feller condition*, which can be proved that, if the condition is satisfied, then the sum of independent but not identical random variables will be approximately normal.

***Lindeberg-Feller Condition*** 

- Let $X_1, X_2, \ldots, X_n$ be independent random variables with finite mean and variance: $\mathbb{E}[X_i] = \mu_i, \operatorname{Var}[X_i] = \sigma_i^2$. Denote $Y_n = \sum_{i=1}^n X_i$. Then, $\mathbb{E}[Y_n] = \sum_{i=1}^n \mu_i, \sigma(Y_n) = \sqrt{\operatorname{Var}[Y_n]} = \sqrt{\sum_{i=1}^n \sigma_i^2}:=B_n$. Then $Y_n$ can be standardized as $Y_n^* = \frac{Y_n - \mathbb{E}[Y_n]}{\sigma(Y_n)} = \frac{Y_n - \sum_{i=1}^n \mu_i}{\sqrt{\sum_{i=1}^n \sigma_i^2}} = \sum_{i=1}^n \frac{X_i - \mu_i}{B_n}$.
- To make each term of $Y_n^*$ "uniformly small", we can constrain the probability of event $\{\frac{X_i - \mu_i}{B_n} \ge \gamma,~ \small{\forall \gamma>0} \}$ to converge to 0, i.e.
$$
\lim_{n \to \infty} \underbrace{\mathbb{P}\left(\max_{1\le i \le n} |X_i - \mu_i| > \gamma {B_n}\right)}_{(:=\dagger)} = 0, ~\small{\forall \gamma > 0}.
$$
By probability theory, we can further derive:
$$
(\dagger) = \mathbb{P}\left(\bigcup_{i=1}^n \{|X_i - \mu_i| > \gamma B_n\}\right) \le \sum_{i=1}^n \mathbb{P}(|X_i - \mu_i| > \gamma B_n) $$

And the last term can be further derived as:
$$
\begin{aligned}
RHS &= \sum_{i=1}^n \int_{|x-\mu_i| > \gamma B_n} f_{X_i}(x) dx\\
&\leq \frac{1}{\gamma^2 B_n^2} \sum_{i=1}^n \int_{|x-\mu_i| > \gamma B_n} (x-\mu_i)^2 f_{X_i}(x) dx
\end{aligned}
$$

Then, we can derive the Lindeberg-Feller condition as, for any $\gamma > 0$:
$$
\lim_{n \to \infty} \frac{1}{\gamma^2 B_n^2} \sum_{i=1}^n \int_{|x-\mu_i| > \gamma B_n} (x-\mu_i)^2 f_{X_i}(x) dx = 0. \quad (\star)
$$where this $(\star)$ is the Lindeberg-Feller condition.

***[Lindeberg-Feller CLT]***

For random variables $X_1, X_2, \ldots, X_n$ satisfying the Lindeberg-Feller condition, then for any $x \in \mathbb{R}$, we have:
$$
\lim_{n \to \infty} \mathbb{P}(\frac{1}{B_n} \sum_{i=1}^n (X_i - \mu_i) \le x) = \Phi(x).
$$

It can be further proved that, if $\{X_i\}$ are i.i.d with finite variance, then the Lindeberg-Feller condition is satisfied.


#### Lyapunov CLT

Let $\{X_n\}$ be a sequence of independent random variables. If there exists a $\delta > 0$ such that $\forall x$:
$$
\lim_{n \to \infty} \frac{1}{B_n^{2+\delta}} \sum_{i=1}^n \mathbb{E}[|X_i - \mu_i|^{2+\delta}] = 0,
$$then we have the Lyapunov CLT:
$$
\lim_{n \to \infty} \mathbb{P}(\frac{1}{B_n} \sum_{i=1}^n (X_i - \mu_i) \le x) = \Phi(x).
$$



## Applications Using CLT

### Error Analysis

- In numerical analysis, there is a kind of error called **round-off error**, which is caused by the finite precision of computer arithmetic. For example, when we calculate $\pi$, we can only use a finite number of digits to represent it like $\pi'\dot=3.14159$. 
- More generally, if we calculate summation of such $n$ numbers $S_n = \sum_{i=1}^n X_i$, approximating it as $S'_n = \sum_{i=1}^n X'_i$, and denote the error as $\varepsilon_n = X_n - X'_n$, then we have the overall error as $S_n - S'_n = \sum_{i=1}^n \varepsilon_i$.
- By the property of rounding, we can see that if the approximated value $X'_i = \overline{x_0.x_1x_2\cdots x_k}$, then the real value should be in the interval $[x_0.x_1x_2\cdots (x_k-1)5, x_0.x_1x_2\cdots x_k4]$ (e.g. if $X'_i = 0.1234$, then the real value should be in $[0.12335, 0.12344]$). And thus the error of a $k$-digit approximation can be regarded as a uniform distribution in $[-0.5 \times 10^{-k}, 0.5 \times 10^{-k}]$.
- Now use CLT to analyze the error of $S_n - S'_n$. Since the error of each term $\varepsilon_i$ is uniformly distributed in $[-0.5 \times 10^{-k}, 0.5 \times 10^{-k}]$, then $\mathbb{E}[\varepsilon_i] = 0, \operatorname{Var}[\varepsilon_i] = \frac{1}{12} \times 10^{-2k}$. Then, by CLT, we have $\sum_{i=1}^n \varepsilon_i \xrightarrow{d} \mathcal{N}(0, \frac{n}{12} \times 10^{-2k})$.

### Normal Approximation

#### Normal Approximation with unknown $\sigma^2$

From CLT above, we know that $\sqrt{n}(\bar{X}_{(n)} - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$. However, in practice, we often do not know the value of $\sigma^2$. In this case, we can use the sample variance $S_n^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X}_{(n)})^2$ to estimate $\sigma^2$. Then, Slutsky's Theorem can guarantee that:
$$
\frac{\sqrt{n}(\bar{X}_{(n)} - \mu)}{S_n} \xrightarrow{d} \mathcal{N}(0,1).
$$

#### Normal Approximation of Negative-Binomial Distribution

Assume $X_1, X_2, \ldots, X_n$ are i.i.d. $\operatorname{NegBin}(r,p)$, then we have: $\mathbb{E}[X_i] = \frac{r(1-p)}{p}, \operatorname{Var}[X_i] = \frac{r(1-p)}{p^2}$. Then, by CLT, we have:
$$
\frac{\sum_{i=1}^n X_i - n\frac{r(1-p)}{p}}{\sqrt{n\frac{r(1-p)}{p^2}}} = \frac{\sqrt{n}(\bar{X}_{n} - \frac{r(1-p)}{p})}{\sqrt{\frac{r(1-p)}{p}}} \xrightarrow{d} \mathcal{N}(0,1).
$$


#### Normal Approximation of Bernoulli Distribution (De Moivre-Laplace CLT)

> De Moivre-Laplace CLT is a special case of CLT (and actually the first one to be proved), which shows that the binomial distribution can be approximated by normal distribution when $n$ is large.

- Assume in $n$ trials Bernoulli with $\mathbb{P}(A) = p$, denote $S_n$ as the number of $A$ happened. Then, $\mathbb{E}[S_n] = np, \operatorname{Var}[S_n] = np(1-p)$. Then, by CLT, we have **De Moivre-Laplace CLT**: 
$$\frac{S_n - np}{\sqrt{np(1-p)}} \xrightarrow{d} \mathcal{N}(0,1)$$
  


***[Several Notes]***

- **Normal approximation v.s. Poisson approximation**
  Empirically, when $np > 5, n(1-p) > 5$, the normal approximation is better than Poisson approximation. Else if $p$ is small, Poisson approximation is better.
- **Laplace's Correction**
  In De Moivre-Laplace CLT, we are using a continuous distribution (normal distribution) to approximate a discrete distribution (binomial distribution). To make the approximation more accurate, we can use Laplace's correction. Specifically, if $S_n \sim \operatorname{Bin}(n,p)$ and $Y \sim \mathcal{N}(np, np(1-p))$, then we have:
    $$\mathbb{P}(S_n\leq x) = \mathbb{P}(Y \leq x + 0.5)\\
    \mathbb{P}(S_n\geq x) = \mathbb{P}(Y \geq x - 0.5)\\
    \mathbb{P}(k_1 \leq S_n \leq k_2) = \mathbb{P}(k_1 - 0.5 \leq Y \leq k_2 + 0.5)$$ especially when $k_1, k_2$ are integers.

$\square$

***[Applications of CLT]***

Using CLT, we can easily approximate:
$$
\mathbb{P}(Y^*_n = \frac{S_n-np}{\sqrt{npq}} \leq y) \approx \mathbb{P}(Z \leq y) = \Phi(y).
$$where $Z \sim \mathcal{N}(0,1)$, and $\Phi(y)$ is the CDF of $\mathcal{N}(0,1)$, $S_n \sim \operatorname{Bin}(n,p)$.


