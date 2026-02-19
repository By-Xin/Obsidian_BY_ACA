#StatisticalLearning

## Intuition: 1D Gaussian Mixture Model (GMM)

Assume $X\in \mathbb{R}$ is a random variable with a Gaussian Mixture Model (GMM) distribution, and assume that we know the number of Gaussian components $K$ in the mixture model. However, for each data point $x_i$, we do not know which Gaussian component it belongs to. 

Suppose there's a latent (hidden / unobserved) random variable $Z \in \{1, 2, \ldots, K\}, Z \sim \text{Multinomial}(\pi_1, \pi_2, \ldots, \pi_K)$, where $\pi_k$ is the probability that $Z=k$. The distribution of $(X^(i), Z^(i))$ is given by:
$$
\begin{align*}
p(X^{(i)}, Z^{(i)} ) = p(X^{(i)} \mid Z^{(i)} ) p(Z^{(i)} ) 
\end{align*}
$$ and $X^{(i)} \mid Z^{(i)} = k \sim \mathcal{N}(\mu_k, \sigma_k^2)$, where $\mu_k$ and $\sigma_k^2$ are the mean and variance of the $k$-th Gaussian component, respectively.

If only we know $Z^{(i)}$, then we can easily estimate the parameters $\mu_k$ and $\sigma_k^2$ of the Gaussian components by the Maximum Likelihood Estimation (MLE) method:
$$
\begin{align*}
\pi_k &= \frac{1}{N} \sum_{i=1}^N \mathbb{I}(Z^{(i)} = k) \\
\mu_k &= \frac{\sum_{i=1}^N \mathbb{I}(Z^{(i)} = k) X^{(i)}}{\sum_{i=1}^N \mathbb{I}(Z^{(i)} = k)} \\
\sigma_k^2 &= \frac{\sum_{i=1}^N \mathbb{I}(Z^{(i)} = k) (X^{(i)} - \mu_k)^2}{\sum_{i=1}^N \mathbb{I}(Z^{(i)} = k)}
\end{align*}
$$ (but sadly we don't know the exact value of $Z^{(i)}$).

Thus, we will use the Expectation-Maximization (EM) algorithm to estimate the value of $Z^{(i)}$ . The EM algorithm is an iterative method that alternates between the Expectation (E) step and the Maximization (M) step.

## Expectation-Maximization (EM) Algorithm for GMM

- **E-step**: Estimate the posterior probability by setting:
    $$
    \begin{align*}
    w_k^{(i)} &= p(Z^{(i)} = k \mid X^{(i)}; \mu_k, \sigma_k^2, \pi_k) \\
    &= \frac{p(X^{(i)} \mid Z^{(i)} = k; \mu_k, \sigma_k^2) p(Z^{(i)} = k; \pi_k)}{\sum_{j=1}^K p(X^{(i)} \mid Z^{(i)} = j; \mu_j, \sigma_j^2) p(Z^{(i)} = j; \pi_j)} \\
    &= \frac{\varphi_{\mid \mu_k, \sigma_k^2}(X^{(i)}) \pi_k}{\sum_{j=1}^K \varphi_{\mid \mu_j, \sigma_j^2}(X^{(i)}) \pi_j}
    \end{align*}
    $$ 
    where $\varphi_{\mid \mu_k, \sigma_k^2}(X^{(i)})$ is the probability density function of the Gaussian distribution with mean $\mu_k$ and variance $\sigma_k^2$ evaluated at $X^{(i)}$:
    $$
    \begin{align*}
    \varphi_{\mid \mu_k, \sigma_k^2}(X^{(i)}) = \frac{1}{\sqrt{2\pi \sigma_k^2}} \exp\left(-\frac{(X^{(i)} - \mu_k)^2}{2\sigma_k^2}\right)
    \end{align*}
    $$
    Here $w_k^{(i)}$ is the posterior probability that $X^{(i)}$ belongs to the $k$-th Gaussian component.  


- **M-step**: Update the parameters by maximizing the expected log-likelihood:
    $$
    \begin{align*}
    \pi_k &= \frac{1}{N} \sum_{i=1}^N w_k^{(i)} \\
    \mu_k &= \frac{\sum_{i=1}^N w_k^{(i)} X^{(i)}}{\sum_{i=1}^N w_k^{(i)}} \\
    \sigma_k^2 &= \frac{\sum_{i=1}^N w_k^{(i)} (X^{(i)} - \mu_k)^2}{\sum_{i=1}^N w_k^{(i)}}
    \end{align*}
    $$

---

*[Lemma]*: **(Jensen's Inequality)**: For a convex function $f(x)$ (e.g. $f(x) = x^2$), let $X$ be a random variable, then:
$$
\begin{align*}
\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])
\end{align*}
$$ where the equality holds if and only if $X$ is a constant with probability 1 (i.e. $\mathbb{P}(X = c) = 1$).

*[Proof of E-M algorithm]*:
Assume we have a model $p(X, Z; \theta)$, where $\theta$ is the parameter of the model. The observed data is $X = \{X^{(1)}, X^{(2)}, \ldots, X^{(m)}\}$, and the latent variable is $Z = \{Z^{(1)}, Z^{(2)}, \ldots, Z^{(m)}\}$. The log-likelihood of the observed data is:
$$
\begin{align*}
\sum_{i=1}^m \log p(X^{(i)}; \theta) &= \sum_{i=1}^m \log \sum_{Z^{(i)}} p(X^{(i)}, Z^{(i)}; \theta)  \\
\end{align*}
$$ and the object is to maximize the log-likelihood, which is a concave function of $\theta$. 

> The E-step in the EM algorithm can actually be viewed as finding a lower bound of the log-likelihood, while the value at the point $\theta^{(t)}$ is the same as the log-likelihood. The M-step is to find the maximum of that lower bound found in the E-step, and take the maximum as the new value of $\theta^{(t+1)}$. And another round of E-step is to find a new lower bound of the log-likelihood where the value at $\theta^{(t+1)}$ is the same as the log-likelihood.

Construct a probability distribution $Q(Z^{(i)})$ ($\sum Q_i(Z^{(i)}) = 1$) over the latent variable $Z^{(i)}$ for each data point $X^{(i)}$. Then we have:
$$
\begin{align*}
 \sum_{i=1}^m \log p(X^{(i)}; \theta) &=  \sum_{i=1}^m \log \sum_{Z^{(i)}} p(X^{(i)}, Z^{(i)}; \theta)\\
&=  \sum_{i=1}^m \log \sum_{Z^{(i)}} Q_i(Z^{(i)}) \frac{p(X^{(i)}, Z^{(i)}; \theta)}{Q_i(Z^{(i)})} \\
&=  \sum_{i=1}^m \log \mathbb{E}_{Z^{(i)} \sim Q_i} \left[ \frac{p(X^{(i)}, Z^{(i)}; \theta)}{Q_i(Z^{(i)})} \right] \\
&\geq  \sum_{i=1}^m \mathbb{E}_{Z^{(i)} \sim Q_i} \left[ \log \frac{p(X^{(i)}, Z^{(i)}; \theta)}{Q_i(Z^{(i)})} \right] \\
&= \boxed{\sum_{i=1}^m \sum_{Z^{(i)}} Q_i(Z^{(i)}) \log \frac{p(X^{(i)}, Z^{(i)}; \theta)}{Q_i(Z^{(i)})} }\\
\end{align*}
$$ where the inequality is due to Jensen's inequality.  And here, the last term is the lower bound of the log-likelihood.

Now check the equality condition. On a given iteration $t$ with parameter $\theta^{(t)}$, we want the equality to hold, i.e.:
$$
\begin{align*}
\log\mathbb{E}_{Z^{(i)} \sim Q_i} \left[ \frac{p(X^{(i)}, Z^{(i)}; \theta)}{Q_i(Z^{(i)})} \right] &= \mathbb{E}_{Z^{(i)} \sim Q_i} \left[ \log \frac{p(X^{(i)}, Z^{(i)}; \theta)}{Q_i(Z^{(i)})} \right] \\
\end{align*}
$$ which implies that $\frac{p(X^{(i)}, Z^{(i)}; \theta)}{Q_i(Z^{(i)})}$ is a constant with probability 1. Thus, set $Q_i(Z^{(i)}) \propto p(X^{(i)}, Z^{(i)}; \theta)$, and we have:
$$
\begin{align*}
Q_i(Z^{(i)}) &= \frac{p(X^{(i)}, Z^{(i)}; \theta)}{\sum_{Z^{(i)}} p(X^{(i)}, Z^{(i)}; \theta)} \\
&= \frac{p(X^{(i)}, Z^{(i)}; \theta)}{p(X^{(i)}; \theta)} \\
&= p(Z^{(i)} \mid X^{(i)}; \theta)
\end{align*}
$$ which is the posterior probability of the latent variable $Z^{(i)}$ given the observed data $X^{(i)}$.

To sum up, in E-step, we set $Q_i(Z^{(i)}) = p(Z^{(i)} \mid X^{(i)}; \theta):=w_k^{(i)}$, and in M-step, we maximize the expected log-likelihood with respect to $\theta$: $\theta^{(t+1)} = \arg\max_{\theta} \sum_{i=1}^m \sum_{Z^{(i)}} Q_i(Z^{(i)}) \log \frac{p(X^{(i)}, Z^{(i)}; \theta)}{Q_i(Z^{(i)})}$. 