#Statistics #UMVUE

*[Theorem]* **(Rao-Blackwell Theorem)**
If $\hat{\theta}(\mathbf{X})$ is an estimator for $\theta$ which is unbiased ($\mathbb{E}_{\mathbf{X}}[\hat{\theta}(\mathbf{X})] = \theta$), for any sufficient statistic $S(\mathbf{X})$, let
$$
\hat{\theta}_{RB}(\mathbf{X}) = \mathbb{E}_{\mathbf{X}}[\hat{\theta}(\mathbf{X})|S(\mathbf{X})]:= g(S)
$$ then $\hat{\theta}_{RB}(\mathbf{X})$ is a better estimator of $\theta$ than $\hat{\theta}(\mathbf{X})$, as
$$
\text{Var}(\hat{\theta}_{RB}(\mathbf{X})) \leq \text{Var}(\hat{\theta}(\mathbf{X}))
$$ or equivalently,
$$
\text{MSE}(\hat{\theta}_{RB}(\mathbf{X})) \leq \text{MSE}(\hat{\theta}(\mathbf{X})).
$$

*[Proof]*
First it is easy to verify that $\hat{\theta}_{RB}(\mathbf{X})$ is unbiased:
$$
\mathbb{E}_{\mathbf{X}}[\hat{\theta}_{RB}(\mathbf{X})] = \mathbb{E}_{\mathbf{X}}[\mathbb{E}_{\mathbf{X}}[\hat{\theta}(\mathbf{X})|S(\mathbf{X})]] = \mathbb{E}_{\mathbf{X}}[\hat{\theta}(\mathbf{X})] = \theta.
$$

Then,
$$
\begin{aligned}
\text{Var}(\hat{\theta}) &= \text{Var}(\mathbb{E}[\hat{\theta}|\mathbf{S}]) + \mathbb{E}[\text{Var}(\hat{\theta}|\mathbf{S})] \\
&= \text{Var}(\hat{\theta}_{RB}) + \mathbb{E}[\text{Var}(\hat{\theta}|\mathbf{S})] \quad \small\text{(by def)} \\
&\geq \text{Var}(\hat{\theta}_{RB}) \quad \small\text{(as $\text{Var}(\hat{\theta}|\mathbf{S}) \geq 0$)}
\end{aligned}
$$
Moreover, given both $\hat{\theta}(\mathbf{X})$ and $\hat{\theta}_{RB}(\mathbf{X})$ are unbiased,
$$
\text{MSE}(\hat{\theta}_{RB}) = \text{Var}(\hat{\theta}_{RB}) \leq \text{Var}(\hat{\theta}) = \text{MSE}(\hat{\theta}).
$$
$\square$

*[Example]* 
Consider $X_1, \cdots, X_n \sim \mathcal{N}(\mu, \sigma^2_{\text{known}})$. Let $S = \sum_{i=1}^n X_i$. Then $S$ is a sufficient statistic for $\mu$. Consider a naive estimator $\hat{\mu}(\mathbf{X}) = X_1$. Then applying the Rao-Blackwell theorem, let $\hat{\mu}_{RB}(\mathbf{X}) = \mathbb{E}_{\mathbf{X}}[\hat{\mu}(\mathbf{X})|S(\mathbf{X})]$. 

Now try to solve $ \mathbb{E}_{\mathbf{X}}[X_1|S(\mathbf{X})]$. Note that, for any $i,j$,
$$
\mathbb{E}(X_i|S) = \mathbb{E}(X_j|S)
$$
Thus,
$$
\mathbb{E}(X_1|S) = \frac{1}{n} \sum_{i=1}^n \mathbb{E}(X_i|S) = \frac{1}{n} \mathbb{E}\left(\sum_{i=1}^n X_i|\sum_{k=1}^n X_k\right) = \frac{1}{n} \sum_{k=1}^n X_k
$$
We can see that
$$\begin{aligned}
\text{MSE}(\hat{\mu}_{RB}) &= \text{Var}(\hat{\mu}_{RB}) =  \frac{\sigma^2}{n} \\
\text{MSE}(\hat{\mu}) &= \text{Var}(\hat{\mu}) = \sigma^2
\end{aligned}
$$
$\square$