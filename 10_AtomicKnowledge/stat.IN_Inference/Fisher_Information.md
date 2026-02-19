---
aliases: [Fisher信息, Fisher Information, 费舍尔信息量]
tags:
  - concept
  - math/statistics
  - stat/inference
related_concepts:
  - "[[UMVUE]]"
  - "[[Cramér-Rao Lower Bound]]"
  - "[[Sufficiency]]"
---

# Fisher Information

*[Definition]* **(Fisher Information)**

Given a random sample $\mathbf{X} = (X_1, \cdots, X_n)^T$ from a distribution with pdf $f(\mathbf{x}|\theta)$. Accordingly, calculate the joint pdf  / likelihood function and take the logriathm to get the log-likelihood function:
$$
l(\theta) = \log f(\mathbf{X}|\theta) = \sum_{i=1}^n \log f(X_i|\theta).
$$
Then the Fisher Information of $\theta$ is defined as:
$$
\mathcal{I}(\theta) = \mathbb{E}_{\theta}\left[\left(\frac{\partial}{\partial\theta}\log l(\mathbf{X}|\theta)\right)^2\right] = -\mathbb{E}_{\theta}\left[\frac{\partial^2}{\partial\theta^2}\log l(\mathbf{X}|\theta)\right].
$$

*[Proof of the last equation]*
For pdf of $x$, $\int_\mathbb{R} f({x}; \theta),\mathrm{d}{x} \equiv 1$, take the derivative with respect to $\theta$ on both sides:
$$\begin{aligned}
0 &= \int_\mathbb{R} \frac{\partial}{\partial\theta} f({x}; \theta) \,  \mathrm{d}{x}  \\ 
&= \int_\mathbb{R} \left(\frac{\partial}{\partial\theta}\log f({x}; \theta)\right) f({x}; \theta) \,  \mathrm{d}{x} \quad \small\text{(Fisher Trick)}
\end{aligned}$$
Take the derivative with respect to $\theta$ again on both sides:
$$\begin{aligned}
0 &= \int_\mathbb{R} \frac{\partial}{\partial\theta} \left(\frac{\partial \log f({x}; \theta)}{\partial\theta}\cdot f({x}; \theta) \right)\,  \mathrm{d}{x}  \\
&= \int_\mathbb{R} \frac{\partial^2}{\partial\theta^2}\log f({x}; \theta) f({x}; \theta) \,  \mathrm{d}{x} + \int_\mathbb{R} \left(\frac{\partial}{\partial\theta}\log f({x}; \theta)\right) \cdot \left(\frac{\partial}{\partial\theta} f({x}; \theta)\right) \,  \mathrm{d}{x} \\
& = \mathbb{E}\left[\frac{\partial^2}{\partial\theta^2}\log f(\mathbf{X}; \theta)\right] + \underbrace{\int_\mathbb{R} \left(\frac{\partial}{\partial\theta}\log f({x}; \theta)\right) \cdot \left(\frac{\partial}{\partial\theta} f({x}; \theta)\right) \,  \mathrm{d}{x}}_\triangle \\
\end{aligned}$$
Recall the Fisher Trick: $\frac{\partial}{\partial\theta} f({x}; \theta) = \frac{\partial}{\partial\theta}\log f({x}; \theta) \cdot f({x}; \theta)$, then the term $\triangle$ can be written as:   
$$
\begin{aligned}
\triangle &= \int_\mathbb{R} \left(\frac{\partial}{\partial\theta}\log f({x}; \theta)\right) \cdot \left(\frac{\partial}{\partial\theta}\log f({x}; \theta)\cdot f({x}; \theta)\right) \,  \mathrm{d}{x} \\
&= \mathbb{E}\left[\left(\frac{\partial}{\partial\theta}\log f(\mathbf{X}; \theta)\right)^2\right]
\end{aligned}
$$
Thus, we have:
$$
0 = \mathbb{E}\left[\frac{\partial^2}{\partial\theta^2}\log f(\mathbf{X}; \theta)\right] + \mathbb{E}\left[\left(\frac{\partial}{\partial\theta}\log f(\mathbf{X}; \theta)\right)^2\right]
$$
$\square$




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

