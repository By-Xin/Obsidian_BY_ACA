#Probability

The Law of Large Numbers states that the sample mean converges to the population expectation as the sample size increases. In detail, given $X_1, X_2, \ldots, X_n$ are i.i.d. random variables, then 
$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{n \to \infty} \mathbb{E}[X_i]
$$
where $\bar{X}_n$ is the sample mean of $X_1, X_2, \ldots, X_n$.

According to the different convergence types, the Law of Large Numbers can be divided into two types: weak law of large numbers (convergence in probability) and strong law of large numbers (almost sure convergence).

## Mean Square LLN

Before we introduce the weak and strong LLN, first consider convergence in mean square, i.e. $\frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{m.s.} \mathbb{E}[X_i]$:
$$\begin{aligned}
\mathbb{E}| \frac{1}{n} \sum_{i=1}^n X_i - \mathbb{E}[X_i] |^2 &= \frac{1}{n^2} \mathbb{E} \left| \sum_{i=1}^n X_i - n \mathbb{E}[X_i] \right|^2 \\
&= \frac{1}{n^2} \mathbb{E} \left[ \left( \sum_{i=1}^n X_i - n \mathbb{E}[X_i] \right)^2 \right] \\
&= \frac{1}{n^2} \sum_{i=1}^n \mathbb{E} \left[ (X_i - \mathbb{E}[X_i])^2 \right] + \frac{2}{n^2} \sum_{i \neq j} \mathbb{E}[(X_i - \mathbb{E}[X_i])(X_j - \mathbb{E}[X_j])] \\
&= \frac{1}{n} \operatorname{Var}[X_i].
\end{aligned}$$

If $\operatorname{Var}[X_i] < \infty$, then $\frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{m.s.} \mathbb{E}[X_i]$.

## Weak Law of Large Numbers

Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with $\mathbb{E}[X_i]=\mu, \operatorname{Var}[X_i]=\sigma^2 < \infty$. Denote $\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i$. Then, for any $\epsilon > 0$,
$$
\lim_{n \to \infty} \mathbb{P}(|\bar{X}_n - \mu| < \epsilon) = 1.
$$

*[Lemma]* (**[[Chebyshev Inequality]]**, concentration) $\forall \epsilon > 0$, we have
$$
\mathbb{P}(|X| \geq \epsilon) \leq \frac{\mathbb{E}[X^2]}{\epsilon^2}.
$$
- [*Proof*]: 
  $$\mathbb{E}|X|^2 = \int_{\{|X| \geq \epsilon\}~ \cup~ \{|X| < \epsilon\}} |x|^2 f_X(x) dx \geq \int_{|X| \geq \epsilon} x^2 f_X(x) dx \geq  \int_{|X| \geq \epsilon} \epsilon^2 f_X(x) dx = \epsilon^2 \mathbb{P}(|X| \geq \epsilon).$$
- Fruthermore, $$\mathbb{P}(|X-\mathbb{E}[X]| \geq \epsilon) \leq \frac{\operatorname{Var}[X]}{\epsilon^2}.$$
$\square$

- *[Proof of Weak LLN]*:
From Chebyshev Inequality, we have
$$\begin{aligned}
\mathbb{P}(|\bar{X}_n - \mu| \geq \epsilon) &\leq \frac{\operatorname{Var}[\bar{X}_n]}{\epsilon^2} \\
&= \frac{1}{n\epsilon^2} \operatorname{Var}[\sum_{i=1}^n X_i] \\
&\xrightarrow{n \to \infty} 0.
\end{aligned}$$
$\square$

*[Theorem]* (**Bernoulli Weak LLN**): Let $S_n = X_1 + X_2 + \ldots + X_n$ be the count of successes in $n$ Bernoulli trials with success probability $p$. Then we can generate a specific form of weak LLN, namely Bernoulli Weak LLN:
$$
\lim_{n \to \infty} \mathbb{P} \left( \left| \frac{S_n}{n} - p \right| < \epsilon \right) = 1.
$$ 
- Bernoulli Weak LLN gives a mathematical proof of the intuition that the sample proportion of successes converges to the true probability of success as the sample size increases, or in short, *frequency converges to probability*.

*[Application]* (**Monte Carlo Simulation for Integration (I)**): Consider function $0\le f(x) \le 1$ and we want to calculate $J = \int_0^1 f(x) dx$. We can generate uniformly random variables in region $[0,1]\times[0,1]$ and calculate the proportion of points that fall under the curve $f(x)$, i.e. $\frac{S_n}{n}$. Theoretically, the probability of falling under the curve is $\mathbf{p} = \mathbb{P}(Y \leq f(X)) = J$. Then, by Bernoulli Weak LLN, we have $\frac{S_n}{n} \xrightarrow{p} \mathbf{p} = J$.
- More generally, if $x\in[a,b], f(x) \in [c,d]$, we can transform the region to $[0,1]\times[0,1]$ by $u = \frac{x-a}{b-a}, g(u) = \frac{f(a+u(b-a))-c}{d-c}$, and calculate the integral by $\int_a^b f(x) dx = (b-a)(d-c)\int_0^1 g(u) du + c(b-a)$.

> **NOTE** Note that the proof of weak LLN does not actually need the assumption of i.i.d. random variables (especially the identical distribution part). Actually, by using the Chebyshev Inequality, we **only need the assumption of finite variance**. And the conclusion of weak LLN can be generalized to the following form:
> - **Chebyshev Weak LLN**: $X_1, X_2, \ldots, X_n$ are *independent* random variables with $\mathbb{E}[X_i]=\mu_i$. Also, the variance of all $X_i$ exists, with common upper bound $c$, i.e. $\operatorname{Var}[X_i] \leq c$. Then, $\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{p} \frac{1}{n} \sum_{i=1}^n \mu_i$.
> - **Markov Weak LLN**: As long as for $X_1, X_2, \ldots, X_n$, $\frac{1}{n^2}\operatorname{Var}(\sum_{i=1}^n X_i) \to 0$, then $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{p} \frac{1}{n}\sum_{i=1}^n \mathbb{E}[X_i]$.
> Here, the Markov Weak LLN is a more general form of weak LLN, which does not require any assumption of i.i.d. random variables.

*[Theorem]* (**Khintchine Weak LLN**): $X_1, X_2, \ldots, X_n$ are i.i.d. random variables with $\mathbb{E}[X_i]=\mu$. Then, $\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{p} \mu$.
*[Collorary]*: If $\{X_i\}$ are i.i.d. , $\mathbb{E}|X_i|^k < \infty, k\in \mathbb{N}$, then $\frac{1}{n} \sum_{i=1}^n X_i^k \xrightarrow{p} \mathbb{E}[X_i^k]$.
- The proof of Khintchine Weak LLN is based on the lemma that $F_{{X}_n}(x)\xrightarrow{w} F_{X}(x) \Leftrightarrow \varphi_{{X}_n}(t)\to \varphi_{X}(t)$, where $\varphi_X(t)$ is the characteristic function of $X$.
  - *[Proof]*: 
    Denote $Y = \frac{1}{n} \sum_{i=1}^n X_i$, then $\varphi_Y(t) =[ \varphi_{X}(t/n)]^n$. 
    By Taylor Expansion, we have:
    $$
    \varphi_{X_i/n}(t) = \varphi_{X}(t/n) = 1 + \varphi_{X}'(0)\frac{t}{n} + o(\frac{1}{n}).
    $$where $\varphi_{X}'(0) = i\mathbb{E}[X]$. 
    Thus, $\varphi_Y(t) = [1 + i\mathbb{E}[X]\frac{t}{n} + o(\frac{1}{n})]^n \to e^{it\mathbb{E}[X]}$, which is the characteristic function of $\mathbb{P}(X=\mathbb{E}[X])=1$, i.e. $Y \xrightarrow{L} \mathbb{E}[X]$.
    By the lemma, we have $Y \xrightarrow{p} \mathbb{E}[X]$.

- *Chebyshev Weak LLN* and *Markov Weak LLN* are more general forms of weak LLN by loosening the assumption of i.i.d. random variables and keeping the assumption of finite variance. While *Khintchine Weak LLN* is trying to **require i.i.d and finite expectation (but no assumption on variance)**.
- Khintchine Weak LLN gives a guarantee that approximating the expectation of $\mathbb{E}(x)$ by observing a random variable independently $n$ times and taking its sample mean is reasonable as long as a sufficient large $n$, regardless of the distribution of $X$. (Actually by the collorary, we can also approximate the higher moments $\mathbb{E}(x^k)$ by observing $\frac{1}{n}\sum_{i=1}^n X_i^k$.)

*[Application]* (**Monte Carlo Simulation for Integration (II)**): Consider the integral $J = \int_a^b f(x) dx$. Let $X\sim\mathcal{U}(0,1)$, then
$$
\mathbb{E}[f(X)] = \int_0^1 f(x) dx = J.
$$
Thus, we can use Monte Carlo Simulation to estimate the integral by estimating $\mathbb{E}[f(X)]$. By Khintchine Weak LLN, 
$$
J \approx \frac{1}{n} \sum_{i=1}^n f(X_i).
$$ Similar transformation can be applied to the case where $[a,b]$ is not $[0,1]$.


## Strong Law of Large Numbers

Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with $\mathbb{E}[X_i]=\mu, \operatorname{Var}[X_i]=\sigma^2 < \infty$. Denote $\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i$. Then, for any $\epsilon > 0$,
$$
\mathbb{P}(\lim_{n \to \infty} |\bar{X}_n - \mu| < \epsilon) = 1.
$$

*[Proof of Strong LLN]*:
- If $\{\bar{X}_n\}$ converges to $\mu$ almost surely, it is equivalent to say that $\mathbb{P} \left( \lim_{n \to \infty} |\bar{X}_n - \mu| \ge \epsilon \right) = 0$ for any $\epsilon > 0$. Thus, we can prove SLLN by showing the set that makes sequence diverge (i.e. $\exists \delta > 0,\textit{s.t. } \forall n, \exists k > n, |\bar{X}_k - \mu| > \delta$) is of measure zero.
- Denote all such $\{\bar{X}_k\}$ that diverge as:
    $$A_\delta = \bigcap_{n=1}^\infty \bigcup_{k=n}^\infty \{ \omega: |\bar{X}_k(\omega) - \mu| > \delta \}.$$
- Clearly that the probability of $A_\delta$ has upper bound if we 'remove' $\cap_{n=1}^\infty$:
    $$\begin{aligned}
    \mathbb{P}(A_\delta) &\leq \mathbb{P} \left( \bigcup_{k=n}^\infty \{ \omega: |\bar{X}_k(\omega) - \mu| > \delta \} \right) \\
    &\leq \sum_{k=n}^\infty \mathbb{P} \left( |\bar{X}_k - \mu| > \delta \right) \\
    &= \sum_{k=1}^\infty \mathbb{P} \left( |\bar{X}_1 - \mu| > \delta \right)\\
    &\le 2\sum_{k=n}^\infty c_k, 0 < c_k <1 
    \end{aligned}$$
- The last inequality is shown in Statistical Inference, Casella, 2nd Ed., Chapter 5. If acknowledge the last inequality, we can conclude that $\mathbb{P}(A_\delta) \leq 2\sum_{k=n}^\infty c_k = 2\frac{c_n}{1-c} \xrightarrow{n \to \infty} 0$. Thus, $\mathbb{P}(A_\delta) = 0$ for any $\delta > 0$, which means $\{\bar{X}_n\}$ converges to $\mu$ almost surely. 
  $\square$