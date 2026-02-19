5. Gaussian processes

Consider a random vector $(X_1,X_2,X_3,X_4)$ that has the joint Gaussian distribution

$$
\begin{bmatrix} X_1 \\ X_2 \\ X_3 \\ X_4\end{bmatrix} \sim N\left(\begin{bmatrix} -0.5 \\ 0 \\ 0.5 \\ 2\end{bmatrix}, \begin{bmatrix} 3 & 1 & 1 & 1 \\ 1 & 3 & 1 & 1 \\ 1 & 1 & 3 & 2 \\ 1 & 1 & 2 & 3\end{bmatrix} \right)
$$

What is the marginal variance of $X_4$?

What is the conditional variance of $X_4$, conditioned on the observations $X_1=1,X_2=1,X_3=1$. Does it depend on these particular values? Hint: The following code snippet creates matrix in R. You may use solve() to find the inverse of a matrix.

Denote $\mathrm{X}_{1:3} = (X_1,X_2,X_3)$. And thus the variance-covariance matrix of can be partitioned as:
$$
\begin{bmatrix} \Sigma_{1:3,1:3} & \Sigma_{1:3,4} \\ \Sigma_{4,1:3} & \Sigma_{4,4} \end{bmatrix}
$$
where $\Sigma_{1:3,1:3} \in \mathbb{R}^{3 \times 3}$, $\Sigma_{1:3,4} \in \mathbb{R}^{3 \times 1}$, $\Sigma_{4,1:3} \in \mathbb{R}^{1 \times 3}$, and $\Sigma_{4,4} \in \mathbb{R}^{1 \times 1}$.
Then the conditional variance of $X_4$ given $\mathrm{X}_{1:3}$ is:
$$
\begin{aligned}
\mathrm{Var}(X_4|\mathrm{X}_{1:3}) &= \Sigma_{4,4} - \Sigma_{4,1:3}\Sigma_{1:3,1:3}^{-1}\Sigma_{1:3,4} \\
&= 3 - \begin{bmatrix} 1 & 1 & 2 \end{bmatrix} \begin{bmatrix} 3 & 1 & 1 \\ 1 & 3 & 1 \\ 1 & 1 & 3 \end{bmatrix}^{-1} \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} \\
&=
\end{aligned}
$$

```R
A <- matrix(c(3, 1, 1, 1,
              1, 3, 1, 1,
              1, 1, 3, 2,
              1, 1, 2, 3),
            nrow = 4, ncol = 4, byrow = TRUE)

A_11 <- A[1:3, 1:3]
A_12 <- A[1:3, 4]
A_21 <- A[4, 1:3]
A_22 <- A[4, 4]

Var_cond <- A_22 - A_21 %*% solve(A_11) %*% A_12
```

What is the conditional mean of $X_4$, conditioned on the observations $X_1=1,X_2=1,X_3=1$?

The conditional mean of $X_4$ given $\mathrm{X}_{1:3}$ is:
$$
\begin{aligned}
\mathrm{E}(X_4|\mathrm{X}_{1:3}) &= \mu_4 + \Sigma_{4,1:3}\Sigma_{1:3,1:3}^{-1}(\mathrm{X}_{1:3} - \mu_{1:3}) \\
&= 2 + \begin{bmatrix} 1 & 1 & 2 \end{bmatrix} \begin{bmatrix} 3 & 1 & 1 \\ 1 & 3 & 1 \\ 1 & 1 & 3 \end{bmatrix}^{-1}
\left( \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} - \begin{bmatrix} -0.5 \\ 0 \\ 0.5 \end{bmatrix} \right) \\
\\&=
\end{aligned}
$$

```R
mu <- c(-0.5, 0, 0.5, 2)
mu_1_3 <- mu[1:3]
mu_4 <- mu[4]

Expect_cond <- mu_4 + A_21 %*% solve(A_11) %*% (c(1, 1, 1) - mu_1_3)
```




Write a level 95% prediction interval for $X_4$ given these observations.

```R
A <- matrix(c(3, 1, 1, 1,
              1, 3, 1, 1,
              1, 1, 3, 2,
              1, 1, 2, 3),
            nrow = 4, ncol = 4, byrow = TRUE)
```

A 95% prediction interval for $X_4$ given $\mathrm{X}_{1:3}$ is:
$$
\begin{aligned}
\mathrm{E}(X_4|\mathrm{X}_{1:3}) \pm \mathcal{Z}_{0.975} \sqrt{\mathrm{Var}(X_4|\mathrm{X}_{1:3})}
\end{aligned}
$$

```R
Z_975 <- qnorm(0.975)
Var_cond <- A_22 - A_21 %*% solve(A_11) %*% A_12
Expect_cond <- mu_4 + A_21 %*% solve(A_11) %*% (c(1, 1, 1) - mu_1_3)
Prediction_interval <- c(Expect_cond - Z_975 * sqrt(Var_cond), Expect_cond + Z_975 * sqrt(Var_cond))
```

