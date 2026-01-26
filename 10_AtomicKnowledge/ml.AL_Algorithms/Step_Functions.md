#StatisticalLearning 

***Introduction***

- **Polynomial regression** supposes a *global* relationship between $Y$ and $X$; while **step functions** avoids such assumption.

- **Step functions** breaks the range of $X$ into bins, and fits a different constant in each bin.

***Specifically,*** assume that we create cutpoints $c_1,...,c_K$ over $X$, and thus obtain $K+1$ new variables:
$$\begin{aligned}
    C_0(X) &= I( C_ < c_1) \\ C_1(X) &= I(c_1 \leq X < c_2) \\ &... \\ C_{K-1}(X) &= I(c_{K-1} \leq X < c_K) \\ C_K(X) &= I(c_K \leq X)
\end{aligned}
$$
where $I(\cdot)$ is the indicator function.

Then we can fit the model:
$$
y_i = \beta_0 + \beta_1C_1(X_i) + ... + \beta_KC_K(X_i) + \epsilon_i
$$
where $C_0(X)$ is excluded given that $\sum_{i=0}^k C_i(X) = 1$, and is the same reason as dummy variable traps. 
Note that, for a given $X$, at most one of the $C_i(X)$ can be none zero. Moreover, if $X<c_1$, then $\forall ~C_i = 0$, thus $\beta_0$ is the *mean value of $Y$ for $X<c_1$.*

- The selection of breakpoints will influence the outcome, and thus even would miss out some important trends (as the curve between the break points are flat)
- Yet, it is popular among biostatistics and epidmiology. 