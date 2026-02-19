#Statistics


For linear transformation, we can easily infer the expectation and variance of the transformed random variable. However, for non-linear transformation, it is hard to infer the distribution of the transformed random variable. Delta Method is a method to estimate the distribution of a function of a random variable. It is widely used in the inference of MLE estimators.

*[Theorem]* **(Taylor Expansion for Uni-variate Function)**
- For a function $g(x)$, if the $k$-th derivative of $g(x)$ exists around $x=a$, then for any constant $a$, we have the *Taylor Polynomial of order $r$ about $a$*:
$$
T_r(x) = g(a) + g'(a)(x-a) + \frac{g''(a)}{2!}(x-a)^2 + \cdots + \frac{g^{(r)}(a)}{r!}(x-a)^r
$$ where $g^{(r)}(a)$ is the $r$-th derivative of $g(x)$ at $x=a$.

- For the remainder term $R_r(x)$, we have:
$$
\lim_{x \to a} \frac{g(x) - T_r(x)}{(x-a)^r} = 0
$$

*[Theorem]* **(Taylor Expansion for Multi-random-variate Function)**
- Let $\mathbf{T} = (T_1, T_2, \cdots, T_k)$ be random variables with means $\boldsymbol{\mu} = (\mu_1, \mu_2, \cdots, \mu_k)$. Suppose $g(\mathbf{T})$ is a differentiable function, and is the function we want to estimate the expectation and variance. 
- Define the derivatives of $g(\mathbf{T})$ about $\boldsymbol{\mu}$ (in scalar form) as:
  $$
  g'_i(\boldsymbol{\mu}) = \frac{\partial g(\boldsymbol{\mu})}{\partial \mu_i}
    $$
- Then, the Taylor expansion of $g(\mathbf{T})$ about $\boldsymbol{\mu}$ is:
    $$
    g(\mathbf{T}) \approx g(\boldsymbol{\mu}) + \sum_{i=1}^k g'_i(\boldsymbol{\mu})(T_i - \mu_i)
    $$
- The expectation and variance of $g(\mathbf{T})$ can be estimated as:
    $$
    \begin{aligned}
    E[g(\mathbf{T})] &\approx g(\boldsymbol{\mu}) + \sum_{i=1}^k g'_i(\boldsymbol{\mu})E[T_i - \mu_i] = g(\boldsymbol{\mu}) \\
    Var[g(\mathbf{T})] &\approx \mathbb{E}[(g(\mathbf{T}) - g(\boldsymbol{\mu}))^2] \approx\mathbb{E}\left[\left(\sum_{i=1}^k g'_i(\boldsymbol{\mu})(T_i - \mu_i)\right)^2\right] \\
    &= \sum_{i=1}^k [g'_i(\boldsymbol{\mu})]^2 Var[T_i] + 2\sum_{i>j} g'_i(\boldsymbol{\mu})g'_j(\boldsymbol{\mu})Cov(T_i, T_j)
    \end{aligned}
    $$


---
Actually only the linear transformation is easy to infer. Thus, for any non-linear transformation $h(\beta)$, we can use Taylor expansion to approximate it.

*[Theorem]* **(Univariate Delta Method)**
- Suppose we have a random variable $X$ with mean $\mu$ and variance $\sigma^2$, and we are interested in the function $Y = h(X)$, then we can expand $h(X)$ around $\mu$: $$

h(X) \approx h(\mu) + h'(\mu)(X-\mu)

$$  where $h(\mu)$ and $h'(\mu)$ are non-random values.

- Then we can immediately conclude that: $$\begin{aligned}

\mathbb{E}(Y) &\approx h(\mu) \quad \\

\text{Var}(Y) &\approx h'(\mu)^2\text{Var}(X) \\

\end{aligned}$$ or $$\begin{aligned}

\boxed{\mathbb{E}[h(X)] \approx h(\mathbb{E}[X]) \quad \text{Var}[h(X)]\approx h'(\mathbb{E}[X])^2\text{Var}[X]}

\end{aligned}$$
 -  Especially, let $h(X) = 1/X$, then we can have:$$\begin{aligned}
	
	\mathbb{E}[1/X] &\approx \frac{1}{\mathbb{E}[X]} =\frac{1}{\mu} \\
	
	\text{Var}[1/X] &\approx \frac{1}{\mathbb{E}[X]^4}\text{Var}[X] = \frac{\sigma^2}{\mu^4}
	
	\end{aligned}$$

   - If $X\sim \mathcal{N}(\mu, \sigma^2)$, then we can have:$$
	
	h(X) \dot\sim \mathcal{N}(h(\mu), h'(\mu)^2\sigma^2)
	
	$$

  
*[Theorem]* **(Multivariate Delta Method)**  

- Let $\mathrm{X} = (X_1, X_2, \cdots, X_d)^T$ be a random vector with mean $\mu = (\mu_1, \mu_2, \cdots, \mu_d)^T$ and covariance matrix $\Sigma$. Given a function $h(\mathrm{X}): \mathbb{R}^d \to \mathbb{R}$, we are interested in the mean and variance of $Y = h(\mathrm{X})$.

- By Taylor expansion, we can have: $$

h(\mathrm{X}) \approx h(\mu) + \nabla h(\mu)^T(X-\mu)

$$where $\nabla h(\mu)$ is the gradient of $h$ at $\mu$: $\nabla h(\mu) = (\frac{\partial h}{\partial x_1}(\mu), \frac{\partial h}{\partial x_2}(\mu), \cdots, \frac{\partial h}{\partial x_d}(\mu))^T$.

- Then $$\begin{aligned}

\mathbb{E}[Y] &\approx h(\mu) \\

\text{Var}[Y] &\approx \nabla h(\mu)^T \Sigma\nabla h(\mu)

		\end{aligned}$$$$\begin{aligned}

\boxed{\mathbb{E}[h(\mathrm{X})] \approx h( \mu) \quad \text{Var}[h(\mathrm{X})]\approx \nabla h( \mu)^T \Sigma\nabla h( \mu)}

\end{aligned}$$

##### Delta Method for MLE Inference

- For MLE inference, we know that the MLE estimator has the asymptotic normality, i.e. given MLE estimator $\hat\beta\in \mathbb{R}^d$, for large sample size, we have: $$

 {\hat\beta} \dot\sim \mathcal{N}( \beta, I^{-1}( \beta))

$$where $I( \beta)$ is the Fisher information matrix.

  

- Then by Delta Method, we can infer for $h( {\hat\beta}): \mathbb{R}^d \to \mathbb{R}$:$$

h( {\hat\beta}) ~\dot\sim~ \mathcal{N}\left(h( \beta), \nabla h( \beta)^TI^{-1}( \beta)\nabla h( \beta)\right)

$$

- And equivalently the confidence interval for $h( \beta)$ is:

$$

h( {\hat\beta}) \pm z_{\alpha/2}\sqrt{\nabla h( \beta)^TI^{-1}( \beta)\nabla h( \beta)}

$$