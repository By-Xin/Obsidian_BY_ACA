---
aliases: ['平稳随机过程', 'Stationary Process']
tags:
  - concept
  - math/probability
  - time-series
related_concepts:
  - [[Auto-Correlation_Function]]
  - [[Stochastic_Process]]
---

#StochasticProcess

## Definition of Wide Sense Stationary Process

Given a stochastic process \(X(t)\), we say that it is wide sense stationary if the following conditions are satisfied:

1. The mean of the process is constant for all $t$: $\mathbb{E}[X(t)] ≡ \mu (\text{constant})$.
2. The ACF of the process is a function of the time difference between two time points: $R_X(t+\tau, s+\tau) = R_X(t, s) = R_X(|t-s|)$.

We can immediately conclude the following properties of ACF for a wide sense stationary process:

1. $R_X(-t) = R_X(t)$.
2. $R_X(0) \ge 0$.
3. $|R_X(t)| \le R_X(0), \forall t$.


## Examples of w.s.s. 

### Modulated Signal

Define a modulated signal as $X(t) = A(t) \cos(2\pi f_0 t + \theta)$, where $A(t)$ is a random amplitude (w.s.s.) and is independent of the phase $\theta$, $f_0$ is a constant frequency.

The mean of the process is:
$$
\begin{aligned}
\mathbb{E}[X(t)] &= \mathbb{E}[A(t) \cos(2\pi f_0 t + \theta)] \\
&= \mathbb{E}[A(t)] \mathbb{E}[\cos(2\pi f_0 t + \theta)] ~\small\text{(by independence)} \\
&= \mathbb{E}[A(t)] \left( \int_0^{2\pi} \cos(2\pi f_0 t + \theta) \frac{1}{2\pi} d\theta \right) \\
&= 0.
\end{aligned}
$$

The ACF of the process is:
$$
\begin{aligned}
R_X(t, s) &= \mathbb{E}[X(t)X(s)] \\
&= \mathbb{E}[A(t)A(s) \cos(2\pi f_0 t + \theta) \cos(2\pi f_0 s + \theta)] \\
&= R_A(t, s) \underbrace{\mathbb{E}[\cos(2\pi f_0 t + \theta) \cos(2\pi f_0 s + \theta)]}_{\star} \\
\end{aligned}
$$

The term $\star$ can be simplified as:
$$
\begin{aligned}
\star &= \frac{1}{2\pi} \int_0^{2\pi} \cos(2\pi f_0 t + \theta) \cos(2\pi f_0 s + \theta) d\theta \\
&= \frac{1}{2\pi} \int_0^{2\pi} \frac{1}{2} \left( \cos(2\pi f_0 (t+s) + 2\theta) + \cos(2\pi f_0 (t-s)) \right) d\theta \\
&= \frac{1}{2} \cos(2\pi f_0 (t-s)).
\end{aligned}
$$
This is a function of the time difference $|t-s|$, hence the process is wide sense stationary.

### Random Telegrah Signal

Define a random telegraph signal as $X(t) = 1 \text{ or } -1$. Assume until time $s$, the total times of switching is $N(s) \sim \text{Poisson}(\lambda s)$, i.e., $P(N_s = k) = \frac{(\lambda s)^k}{k!} e^{-\lambda s}$.

The mean of the process is:
$$
\begin{aligned}
\mathbb{E}[X(t)] &= \sum_{k=0,1} k \cdot \textrm{Pr}\{X(t) = k\} \\
&= 1\cdot \textrm{Pr}\{X(t) = 1\} + (-1)\cdot \textrm{Pr}\{X(t) = -1\} \\
\end{aligned}
$$
where 
$$
\begin{aligned}
\textrm{Pr}\{X(t) = 1\} &= \textrm{Pr}\{ X(t) = 1 | X(0) = 1\} \cdot \textrm{Pr}\{X(0) = 1\} \\ &\quad + \textrm{Pr}\{ X(t) = 1 | X(0) = -1\} \cdot \textrm{Pr}\{X(0) = -1\} \\
\end{aligned}
$$
where 
$$
\begin{aligned}
\textrm{Pr}\{ X(t) = 1 | X(0) = -1\} &= \textrm{Pr}\{ N(t)~\text{is odd}\} \\
&= \sum_{k=1,3,5,\ldots} \frac{(\lambda t)^k}{k!} e^{-\lambda t}\\
&=^\dagger \frac12 \left( e^{\lambda t} - e^{-\lambda t} \right) e^{-\lambda t} \\
\end{aligned}
$$
where $\dagger$ is due to the fact that, according to Taylor expansion of $e^x$:
$$
\begin{aligned}
&e^{\lambda t} = \sum_{k=0} \frac{(\lambda t)^k}{k!} \\ 
&e^{-\lambda t} = \sum_{k=0} \frac{(-\lambda t)^k}{k!} = \sum_{k=0} (-1)^k \frac{(\lambda t)^k}{k!} \\
\Rightarrow ~&e^{\lambda t} - e^{-\lambda t} = \sum_{k=1,3,\cdots} \frac{(\lambda t)^k}{k!}
\end{aligned}
$$

Symmetrically, we have:
$$
\begin{aligned}
\textrm{Pr}\{ N(t)~\text{is even}\} &= 1 - \textrm{Pr}\{ N(t)~\text{is odd}\} \\
&= \frac12 \left( 1+ e^{-2\lambda t} \right).
\end{aligned}
$$

Therefore,
$$
\textrm{Pr}\{X(t) = 1\} = \textrm{Pr}\{X(t) = -1\} = \frac12.
\\
\Rightarrow \mathbb{E}[X(t)] = 0. 
$$


