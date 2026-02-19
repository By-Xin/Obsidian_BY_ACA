#Statistics #UMVUE 

## Intuition

Starting from Ancillary Statistic, we can derive a better and better estimator by Rao-Blackwell Procedure. Is there a limit for the quality of an estimator? The answer is yes, and it is the Cramér-Rao Lower Bound.

## Cramér-Rao Lower Bound

Here, we first give a simple version of Cramér-Rao Lower Bound, where we only focus on the parameter $\theta$ (rather than a function of $\theta$).

*[Theorem]* **(Cramér-Rao Lower Bound, Simple Version)**
Given a random sample $\mathbf{X} = (X_1, \cdots, X_n)^T$ from a distribution with pdf $f(\mathbf{x}|\theta)$. Then for any unbiased estimator $\hat{\theta}(\mathbf{X})$, we have:
$$
\text{Var}(\hat{\theta}(\mathbf{X})) \geq \frac{1}{nI(\theta)}
$$where $I(\theta)$ is the [[Fisher Information]] of $\theta$:
$$
I(\theta) = \mathbb{E}_{\theta}\left[\left(\frac{\partial}{\partial\theta}\log f(\mathbf{X}|\theta)\right)^2\right] = -\mathbb{E}_{\theta}\left[\frac{\partial^2}{\partial\theta^2}\log f(\mathbf{X}|\theta)\right].
$$

> [**NOTE**]
>
> Note that there are several ways to define the Fisher Information. 
> - The first one is as shown above, which is derived directly from the pdf of a distribution (of the population distribution)
> - The second way is to view it in a sampling perspective, then the $f(\mathbf{x}|\theta)$ is replaced by the *likelihood function* $L(\theta|\mathbf{x})$. Or it can be regarded as the pdf of the *Joint Distribution*.
> And the Fisher Information is defined as:
    $$\tilde I(\theta) = \mathbb{E}_{\theta}\left[\left(\frac{\partial}{\partial\theta}\log L(\theta|\mathbf{X})\right)^2\right] = -\mathbb{E}_{\theta}\left[\frac{\partial^2}{\partial\theta^2}\log L(\theta|\mathbf{X})\right].
    $$
> - The two definitions are essentially equivalent, as the likelihood function is just the pdf of the sample:
    $$
    \frac{\partial \log L(\theta|\mathbf{X})}{\partial\theta} = \sum_{i=1}^n \frac{\partial \log f(X_i|\theta)}{\partial\theta}
    $$
    Thus for the Fisher Information, 
    $$
    \tilde I(\theta) = nI(\theta).
    $$
    Yet we do have to pay attention to the dimension or meaning of the function when calculating the Fisher Information, and we have to be consistent. (Usually, the log-likelihood function is used in practice.)

*[Proof]*

First, as $\hat{\theta}(\mathbf{X})$ is unbiased, we have:
$$\begin{aligned}
\theta = \mathbb{E}[\hat{\theta}(\mathbf{X})] &= \int_\mathbb{R} \hat{\theta}(\mathbf{x}) f(\mathbf{x}; \theta)\,  \mathrm{d}\mathbf{x} \\
\end{aligned}$$ Taking the derivative with respect to $\theta$ on both sides:
$$\begin{aligned}
\frac{\partial}{\partial\theta}\theta &= \frac{\partial}{\partial\theta}\int_\mathbb{R} \hat{\theta}(\mathbf{x}) f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x} \\
\Leftrightarrow 1 &= \int_\mathbb{R} \hat{\theta}(\mathbf{x}) \frac{\partial}{\partial\theta} f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x}  \quad \cdots \small\text{(1)}
\end{aligned}$$

Meanwhile, consider the pdf $\int_\mathbb{R} f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x} =1$ and take the derivative with respect to $\theta$ on both sides:
$$\begin{aligned}
0 &= \int_\mathbb{R} \frac{\partial}{\partial\theta} f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x}  \\
\Leftrightarrow 0\cdot\theta &= 0 = \int_\mathbb{R} \theta \frac{\partial}{\partial\theta} f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x}  \quad \cdots \small\text{(2)}
\end{aligned}$$

Subtracting (2) from (1):
$$\begin{aligned}
(1) - (2):1 &= \int_\mathbb{R} (\hat{\theta}(\mathbf{x}) - \theta) \frac{\partial}{\partial\theta} f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x} \\
&= \int_\mathbb{R} (\hat{\theta}(\mathbf{x})-\theta)\left( \frac{\partial}{\partial\theta}\log f(\mathbf{x}; \theta)\right) f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x} \quad \small\text{(Fisher Trick)}\\
& = \int_\mathbb{R} (\hat{\theta}(\mathbf{x})-\theta) \sqrt{f(\mathbf{x}; \theta)}\frac{\partial}{\partial\theta}\log f(\mathbf{x}; \theta)\sqrt{f(\mathbf{x}; \theta)} \,  \mathrm{d}\mathbf{x} \\
& \leq \sqrt{\int_\mathbb{R} (\hat{\theta}(\mathbf{x})-\theta)^2 f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x}} \sqrt{ \int_\mathbb{R} \left( \frac{\partial}{\partial\theta}\log f(\mathbf{x}; \theta)\right)^2 f(\mathbf{x}; \theta)\,  \mathrm{d}\mathbf{x}} \quad \small\text{(Cauchy-Schwarz Inequality)}\\
 \Leftrightarrow 1 &\leq \int_\mathbb{R} (\hat{\theta}(\mathbf{x})-\theta)^2 f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x} \int_\mathbb{R} \left( \frac{\partial}{\partial\theta}\log f(\mathbf{x}; \theta)\right)^2 f(\mathbf{x}; \theta)\,  \mathrm{d}\mathbf{x} \\
 & = \mathbb{E}\left[(\hat{\theta}(\mathbf{X})-\theta)^2\right] \mathbb{E}\left[\left( \frac{\partial}{\partial\theta}\log f(\mathbf{X}; \theta)\right)^2\right] \\
    & = \text{Var}(\hat{\theta}(\mathbf{X})) I(\theta)
\end{aligned}$$
Thus, we have:
$$
\text{Var}(\hat{\theta}(\mathbf{X})) \geq \frac{1}{I(\theta)}.
$$

$\square$

*[Theorem]* **(Cramér-Rao Lower Bound, General Version)**

If we generalize the estimator from $\mathbb{E}(\hat{\theta}) =  \theta$ to a function of the parameter $\mathbb{E}(\hat{g}(\theta)) = g(\theta) = \int_\mathbb{R} \hat g(\theta) f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x}$, then its derivative with respect to $\theta$ is $g'(\theta) = \int_\mathbb{R} \hat g(\theta) \frac{\partial}{\partial\theta} f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x}$. Following the same procedure of the proof, we have:
$$
\text{Var}(\hat{g}(\theta)) \geq \frac{g'(\theta)^2}{\mathcal{I}(\theta)}
$$
Here, $\mathcal{I}(\theta)$ is the Fisher Information defined from the joint distribution.

*[Theorem]* **(Cramér-Rao Lower Bound, Multivariate Version)**



---

*[Example 1]*
Let $X_1, \cdots, X_n \sim_{i.i.d.} \mathcal{N}(\mu, \sigma^2_{\text{known}})$, try to find the Cramér-Rao Lower Bound for $\theta = \mu$.

1. **Clearify the Joint distribution -> Log-Likelihood Function**
    $$ \begin{aligned}
    f(\mathbf{x}; \mu)& = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)  \\
    \log f(\mathbf{x}; \mu) & = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2
    \end{aligned}$$

2. **Take the derivative**
    $$\begin{aligned}
    \frac{\partial}{\partial\mu}\log f(\mathbf{x}; \mu) & = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i-\mu) \\
    \frac{\partial^2}{\partial\mu^2}\log f(\mathbf{x}; \mu) & = -\frac{n}{\sigma^2}
    \end{aligned}$$

3. **Calculate the Fisher Information**
    $$\begin{aligned}
    I(\mu) & = -\mathbb{E}_{\mu}\left[\frac{\partial^2}{\partial\mu^2}\log f(\mathbf{X}; \mu)\right] = \frac{n}{\sigma^2} \quad \small \text{or equivalently} \\
    I(\mu) & = \mathbb{E}_{\mu}\left[\left(\frac{\partial}{\partial\mu}\log f(\mathbf{X}; \mu)\right)^2\right] = \mathbb{E}_{\mu}\left[ \frac{n}{\sigma^2}\left(\sum_{i=1}^n \frac{X_i}{n} - \mu\right)\right]= \frac{n}{\sigma^2}
    \end{aligned}$$


4. **Give the Cramér-Rao Lower Bound**
    $$\begin{aligned}
    \text{Var}(\hat{\mu}(\mathbf{X})) & \geq \frac{1}{I(\mu)} = \sigma^2/n
    \end{aligned}$$



*[Example 2]*
Let $X_1, \cdots, X_n \sim_{i.i.d.} \text{Poisson}(\lambda)$, try to find the Cramér-Rao Lower Bound for $\theta = \lambda$.

1. **Clearify the Joint distribution -> Log-Likelihood Function**
    $$ \begin{aligned}  
    f(\mathbf{x}; \lambda)& = \prod_{i=1}^n \frac{\lambda^{x_i}}{x_i!} \exp(-\lambda)  \\
    \log f(\mathbf{x}; \lambda) & = \sum_{i=1}^n x_i\log(\lambda) - n \lambda - \sum_{i=1}^n \log(x_i!)
    \end{aligned}$$

2. **Take the derivative**
    $$\begin{aligned}
    \frac{\partial}{\partial\lambda}\log f(\mathbf{x}; \lambda) & = \frac{1}{\lambda}\sum_{i=1}^n x_i - n \\
    \frac{\partial^2}{\partial\lambda^2}\log f(\mathbf{x}; \lambda) & = -\frac{\sum_{i=1}^n x_i}{\lambda^2}
    \end{aligned}$$

3. **Calculate the Fisher Information**
    $$\begin{aligned}
    I(\lambda) & = -\mathbb{E}_{\lambda}\left[\frac{\partial^2}{\partial\lambda^2}\log f(\mathbf{X}; \lambda)\right] = \frac{n}{\lambda} 
    \end{aligned}$$

4. **Give the Cramér-Rao Lower Bound**
    $$\begin{aligned}
    \text{Var}(\hat{\lambda}(\mathbf{X})) & \geq \frac{1}{I(\lambda)} = \frac{\lambda}{n}
    \end{aligned}$$


## Conditions for Equality of Cramér-Rao Lower Bound

The Cramér-Rao Lower Bound is a lower bound for the variance of any unbiased estimator. When the equality holds, the estimator is called an efficient estimator. Now we give the conditions for the equality of the Cramér-Rao Lower Bound.

Recall the proof of the Cramér-Rao Lower Bound, by the Cauchy-Schwarz Inequality:
$$\begin{aligned}
1 &\leq \int_\mathbb{R} (\hat{\theta}(\mathbf{x})-\theta)^2 f(\mathbf{x}; \theta) \,  \mathrm{d}\mathbf{x} \int_\mathbb{R} \left( \frac{\partial}{\partial\theta}\log f(\mathbf{x}; \theta)\right)^2 f(\mathbf{x}; \theta)\,  \mathrm{d}\mathbf{x} 
\end{aligned}$$
The equality holds if and only if the two functions are linearly dependent, i.e. there exists $c(\theta)$ such that:
$$
c(\theta) \left(\hat{\theta}(\mathbf{x}) - \theta\right) =  \frac{\partial}{\partial\theta}\log f(\mathbf{x}; \theta)
$$

Actually it can be shown that the equality holds if and only if the distribution has the form of :
$$
f(\mathbf{x}; \theta) = \exp\left(A(\theta)\hat\theta(\mathbf{x}) \right) \exp\left(B(\theta)\right) \exp\left(h(\mathbf{x})\right)
$$ which is the form of the exponential family distribution.

**As long as Cramér-Rao Lower Bound is achieved, it must be from an exponential family distribution, and vice versa.**

It can also be shown that such $c(\theta) \equiv  \mathcal{I}(\theta)$, where $\mathcal{I}(\theta)$ is the Fisher Information of $\theta$.

