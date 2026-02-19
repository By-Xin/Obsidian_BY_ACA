#StatisticalLearning 
## 1. Piecewise Polynomials

- **Piecewise polynomials** would fit low-degree polynomials over different regions of $X$.

### Knots

- Knots are the cut points that divide the range of $X$ into distinct regions.

    *E.g. **[A piecewise cubic polynomial with one knot at $c$]***
    $$
    \begin{cases}
    y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \beta_3 x_i^3 + \epsilon_i & \text{if } x_i < c \\
    y_i = \tilde{\beta}_0 + \tilde{\beta}_1 x_i + \tilde{\beta}_2 x_i^2 + \tilde{\beta}_3 x_i^3 + \epsilon_i & \text{if } x_i \geq c
    \end{cases}
    $$   $\diamond$

- More knots would bring more flexibility.
- A piecewise constant *[[Step Functions]]* are polynomials of degree 0.

![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202401230026293.png)

## 2. Constraints and Splines

- To avoid overfitting, we can *constrain* the piecewise polynomials to be *continuous* and *smooth* at the knots.
- Each additional constraint would reduce one degree of freedom.
- If we constrain the piecewise polynomials (e.g. cubic) to be:
     a. **Continuous** at the knots
     b. **Continuous for the first derivative** at the knots
     c. **Continuous for the second derivative** at the knots
    we then would call it a ***cubic spline***.

*E.g. **[Cubic Spline]***

- In this case (as is shown above), each region has 4 parameters (thus 8 parameters in total). We have 3 constraints (continuity, continuity of the first derivative, continuity of the second derivative), thus 3 degrees of freedom are reduced.
- In general, if we have $K$ knots, the degree of freedom would be $K+4$ .

### Degree-d Spline

- We define a *degree-d spline* as:
  - A piecewise polynomial of degree $d$
  - Constrained to be continuous up to the $(d-1)$th derivative at the knots.

## 3. The Spline Basis Representation

- *Spline Basis Representation* would help with fitting the model under the constraints.

A cubic spline with $K$ knots can be represented as:
$$
\begin{aligned}
y_i &= \beta_0 + \beta_1 b_1(x_i) + \beta_2 b_2(x_i) + \cdots + \beta_{K+3} b_{K+3}(x_i) + \epsilon_i
\end{aligned}
$$ where $b_j(x)$ are the basis functions.

### The Choice of basis functions

- The most common choice is the *cubic spline* with **truncated power basis** function per knot
- The truncated power basis function per knot is defined as:
  $$
  h(x, \xi) = (x - \xi)_+^3 = \begin{cases}
  (x - \xi)^3 & \text{if } x > \xi \\
  0 & \text{if } x \leq \xi
  \end{cases}
  $$    where $\xi$ is the knot.

- It can be easily proved that the truncated power basis function per knot will remain continuous up to the second derivative at the knot.

- Thus to fit a data set with $K$ knots, we would perform LSE with $K+4$ predictors (including the intercept): $X, X^2, X^3, h(X, \xi_1), h(X, \xi_2), \cdots, h(X, \xi_K), \beta_0$.

![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202401230130691.png)

### Natural Splines

- Splines have high variance at the outer range of the predictors (as is shown above).
- ***Natural Spline*** is a regression spline with additional constraints : **the function is linear at the boundary** (i.e. in the region where $X$ is smaller than the smallest knot, or larger than the largest knot, the function is linear).
- *Natural Splines* generally produce more stable estimates at the boundaries (smaller variance).

## 4. Choosing the Number and Locations of the Knots

The regression spline is *most flexible* in regions.

***Locating the knots***

- One option is to place more knots where we feel the function might vary most rapidly, and to place fewer knots where it seems more stable.
- Another option is to place knots in a uniform fashion, i.e. to specify the desired **degrees of freedom** of the spline, and then have the software automatically place the corresponding number of knots at uniform quantiles of the data.

***Choosing the number of knots***

- One option is to try out different numbers of knots, and then choose the model with the **smallest RSS or the largest $R^2$**.
- Another option is to use **cross-validation**.
  1. Remove a portion of the data (say 10%)
  2. Fit a spline with a certain number of knots to the remaining data
  3. Use the spline to make predictions for held-out portion
  4. Repeat the above steps until each observation has been left out once
  5. Compute the overall cross-validation error (e.g. MSE)


## 5. Comparison with Polynomial Regression