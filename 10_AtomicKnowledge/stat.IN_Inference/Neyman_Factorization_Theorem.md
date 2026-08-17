#Statistics #sufficient_statistics


## Introduction

*[Theorem]* **(Neyman Factorization Theorem)**
A statistic $S(\mathbf{X})$ is a sufficient statistic for $\theta$, if and only if the probability density function can be written as
$$
f(\mathbf{x}, \theta) = h(\mathbf{x}) g(S(\mathbf{x}), \theta)
$$ for some functions $h(\mathbf{x})$ and $g(S(\mathbf{x}), \theta)$.

*[Example 1]* **(Sufficient Statistic for Bernoulli Distribution)**
Consider $X_1, \cdots, X_n \sim_{i.i.d.} \text{Bernoulli}(p)$. The likelihood function is
$$
L(p) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i} = p^{\sum_{i=1}^n x_i}(1-p)^{n-\sum_{i=1}^n x_i}
$$ Let $S(\mathbf{X}) = \sum_{i=1}^n X_i$. Then, the likelihood function can be written as
$$
L(p) = p^S(\mathbf{X})(1-p)^{n-S(\mathbf{X})} = \left(p^{S(\mathbf{X})}\right) \left((1-p)^{n-S(\mathbf{X})}\right)
$$ which is a product of two functions: one is a function of $S(\mathbf{X})$ only, and the other is a function of $\mathbf{X}$ only. Therefore, by the Neyman Factorization Theorem, $S(\mathbf{X})$ is a sufficient statistic for $p$.

*[Example 2]* **(Sufficient Statistic for Poisson Distribution)**
Consider $X_1, \cdots, X_n \sim_{i.i.d.} \text{Poisson}(\lambda)$. The likelihood function is
$$
L(\lambda) = \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!} = \frac{\lambda^{\sum_{i=1}^n x_i} e^{-n\lambda}}{\prod_{i=1}^n x_i!}
$$ Let $S(\mathbf{X}) = \sum_{i=1}^n X_i$. Then, the likelihood function can be written as
$$
L(\lambda) = \lambda^{S(\mathbf{X})} e^{-n\lambda}\cdot\frac{1}{\prod_{i=1}^n x_i!} := g(S(\mathbf{X}), \lambda)\cdot h(\mathbf{X})
$$  Therefore, by the Neyman Factorization Theorem, $S(\mathbf{X})$ is a sufficient statistic for $\lambda$.

*[Example 3]* **(Sufficient Statistic for Normal Distribution, with $\sigma^2$ known)**
Consider $X_1, \cdots, X_n \sim_{i.i.d.} \text{Normal}(\mu, \sigma^2)$ with $\sigma^2$ known. The likelihood function is
$$
L(\mu) = \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^n \exp\left\{-\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2\right\}
$$
Let $S(\mathbf{X}) = \sum_{i=1}^n X_i$. Then, the likelihood function can be written as
$$\begin{aligned}
L(\mu) &= \underbrace{\left(\sqrt{2\pi\sigma^2}\right)^{-n} \cdot \exp\left\{-\frac{n\mu^2}{2\sigma^2}\right\} \cdot \exp \left\{ \frac{\mu\sum_{i=1}^n x_i}{\sigma^2} \right\}}_{g(\sum \textbf{X},\mu) ~(\text{with } \sigma,n \text{ known})}\cdot \underbrace{\exp\left\{\frac{\sum_{i=1}^n x_i^2}{2\sigma^2}\right\}}_{h(\mathbf{X}) ~(\text{function of } \mathbf{X})} \\
&= g(S(\mathbf{X}), \mu) \cdot h(\mathbf{X})
\end{aligned}
$$ Therefore, by the Neyman Factorization Theorem, $S(\mathbf{X}) = \sum_{i=1}^n X_i$ is a sufficient statistic for $\mu$.

*[Example 4]* **(Sufficient Statistic for Normal Distribution, with $\sigma^2$ unknown)**
Consider $X_1, \cdots, X_n \sim_{i.i.d.} \text{Normal}(\mu, \sigma^2)$ with both $\mu$ and $\sigma^2$ unknown. The likelihood function is
$$
L(\mu, \sigma^2) = \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^n \exp\left\{-\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2\right\}
$$ Let $S(\mathbf{X}) = \left(\sum_{i=1}^n x_i, \sum_{i=1}^n x_i^2\right)$. Then, the likelihood function can be written as
$$\begin{aligned}
L(\mu) &= \underbrace{\left(\sqrt{2\pi\sigma^2}\right)^{-n} \cdot \exp\left\{-\frac{n\mu^2}{2\sigma^2}\right\} \cdot \exp \left\{ \frac{\mu\sum_{i=1}^n x_i}{\sigma^2} \right\}\cdot \exp\left\{\frac{\sum_{i=1}^n x_i^2}{2\sigma^2}\right\}}_{g(\sum x_i,\sum x_i^2\mu)} \\
&= g(S(\mathbf{X}), \mu) \cdot 1
\end{aligned}
$$
> In the last case, we can see that the sufficient statistic may not be one-dimensional.