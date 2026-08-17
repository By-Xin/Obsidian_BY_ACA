#StochasticProcess 

## Binary Relation

Assume we have random variables $X$ and $Y$ with joint distribution $f_{X,Y}(x,y)$. Consider the following 4 binary relation examples:

![Linear Correlation](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20240827110259_Linear_Correlation.png)

In order to measure the strength of the relationship between $X$ and $Y$, here introduce ***distance*** ($L_2$ norm):
$$
d(X,Y) = \mathbb{E}[(X-Y)^2] = \mathbb{E}[X^2] + \mathbb{E}[Y^2] - 2\mathbb{E}[XY].
$$ Here, $\mathbb{E}[X^2]$ and $\mathbb{E}[Y^2]$ are the 2nd moments information of $X$ and $Y$ itself, yet $\mathbb{E}[XY]$ is the key measure of the relationship between $X$ and $Y$.

Actually, $\mathbb{E}[XY]$ essentially is the ***inner product*** or ***angle*** of $X$ and $Y$ in the *Hilbert space*.

Here are some important properties of $\mathbb{E}[XY]$:

1. ***Independece can conclude Uncorrelation, but Uncorrelation can not conclude Independence.***
*[Example]* Let $\theta \in [0,2\pi)$, $X = \cos(\theta)$, $Y = \sin(\theta)$. Apparently, $X$ and $Y$ are not independent, as $X^2 + Y^2 = 1$. However, $\mathbb{E}[X] = \int \cos(\theta) \frac{1}{2\pi} d\theta = 0$, $\mathbb{E}[Y] = \int_{0}^{2\pi} \sin(\theta) \frac{1}{2\pi} d\theta = 0$, $\mathbb{E}[XY] = \int_{0}^{2\pi} \cos(\theta)\sin(\theta) \frac{1}{2\pi} d\theta = 0$. Thus, $X$ and $Y$ are uncorrelated.

2. ***Cauchy Inequality***: $|\mathbb{E}[XY]| \leq (\mathbb{E}[X^2]\mathbb{E}[Y^2])^{1/2}$.
*[Proof]* $\left|\langle X,Y \rangle\right| \leq \left(\langle X,X \rangle \langle Y,Y \rangle\right)^{1/2} $.
Define $g(\alpha) = \langle \alpha X+Y, \alpha X+Y \rangle = \alpha^2 \langle X,X \rangle + 2\alpha \langle X,Y \rangle + \langle Y,Y \rangle \geq 0$. Thus, $\Delta = 4\langle X,Y \rangle^2 - 4\langle X,X \rangle \langle Y,Y \rangle \leq 0$.Thus, $|\langle X,Y \rangle| \leq (\langle X,X \rangle \langle Y,Y \rangle)^{1/2}$.

3. ***Correlation Matrix***: For a random vector $\mathbf{X} = (X_1, X_2, \cdots, X_n)$, the correlation matrix is defined as: $\mathbb{E}(\mathbf{X}\mathbf{X}^T) := \Sigma_{\mathbf{X}}$ 
   - $\Sigma_{\mathbf{X}}$ is a symmetric matrix.
   - $\Sigma_{\mathbf{X}}$ is positive semi-definite, i.e. $\forall z \in \mathbb{R}^n$, $z^T\Sigma_{\mathbf{X}}z \geq 0$.
   *[Proof]* $z^T \Sigma_{\mathbf{X}} z = z^T \mathbb{E}(\mathbf{X}\mathbf{X}^T) z = \mathbb{E}(z^T\mathbf{X}\mathbf{X}^T z) = \mathbb{E}((\mathbf{X}^T z)^2) \geq 0.$ 
