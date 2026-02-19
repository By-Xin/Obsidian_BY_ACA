#Statistics #UMVUE 

*[Theorem]* **(Lehmann-Scheffé Theorem)**
Let $\mathbf{X} = (X_1, \cdots, X_n)^T$ be a random sample from a distribution with pdf $f(\mathbf{x} | \theta)$. Suppose $T(\mathbf{X})$ is a complete and sufficient statistic for $\theta$, and $h(T(\mathbf{X}))$ is unbiased for $\theta$. Then $h(T(\mathbf{X}))$ is the unique UMVUE of $\theta$.

*[Proof]*

Given such $T$, we have to show that $\forall \hat{\theta} \text{ s.t. } \mathbb{E}[\hat{\theta}] = \theta$, we have $\text{MSE} (h(T)) \leq \text{MSE}(\hat{\theta})$.

According to Rao-Blackwell Theorem, we have
$$
\hat{\theta}_{RB} = \mathbb{E}[\hat{\theta} | T] = \tilde{\theta}(T)
$$
Then it is a better estimator than $\hat{\theta}$:
$$
\text{MSE}(\hat{\theta}_{RB}) = \text{MSE}(\tilde{\theta}(T)) \leq \text{MSE}(\hat{\theta})
$$

Moreover, both $\hat{\theta}_{RB}$ and $h(T)$ are unbiased for $\theta$:
$$
\mathbb{E}[h(T) - \tilde{\theta}(T)] = 0.
$$
Since $T$ is complete and $h(T) - \tilde{\theta}(T)$ is a function of $T$, we have $h(T) = \tilde{\theta}(T)$ almost everywhere. 

Therefore, $h(T)$ is the unique UMVUE of $\theta$, 
$$
\text{MSE}(h(T)) = \text{MSE}(\tilde{\theta}(T)) \leq \text{MSE}(\hat{\theta}).
$$
$\square$
