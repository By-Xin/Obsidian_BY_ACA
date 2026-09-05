# From Proximal to Primal-Dual

It is a simple study note of PDHG series work. But all the way back to the proximal algorithms. 

## Proximal Algorithms

### Proximal Operator

The proximal operator of a convex function $h$ is defined as:
$$
\operatorname{prox}_{h}(x) = \arg\min_{u \in \operatorname{dom}(h)} \left\{ h(u) + \frac{1}{2} \|u - x\|^2 \right\}.
$$

- It can be interpreted as a general projection operator. It receives a point $x$ and returns a point not too far from $x$. Such a $u$ is the optimal neighbor of $x$ that relatively minimizes $h$. Theorem can guarantee the existence and uniqueness of the proximal operator, for some proper $h$.

Proximal operator is closely related to the subgradient of $h$. Acutually, for some proper $h$:
$$
u = \operatorname{prox}_{h}(x) \iff x - u \in \partial h(u). \qquad {\text{(1)}}
$$
- Intuitively, $x-u$ is the gradient of $\frac{1}{2} \|u - x\|^2$ at $u$. Thus, $x-u \in \partial h(u)$ means that the gradient (or *force*) of $h$ at $u$ balances that of $\frac{1}{2} \|u - x\|^2$. And thus reach the optimal.

More frequently, we use the following form:
$$
\operatorname{prox}_{\lambda h}(x) = \arg\min_{u \in \operatorname{dom}(h)} \left\{ h(u) + \frac{1}{2\lambda} \|u - x\|^2 \right\}.
$$
- $\lambda \in \mathbb{R}_{+}$ is a scaling factor to control the strength of the regularization term $\frac{1}{2\lambda} \|u - x\|^2$. 
- It actually is to replace $h$ by $\lambda h$ in the original definition. Yet $\arg\min \{ \lambda h(u) + \frac{1}{2} \|u - x\|^2 \} = \arg\min \{ h(u) + \frac{1}{2\lambda} \|u - x\|^2 \}$, so. 


Calculation of proximal operator is bascially solving a convex optimization problem, though it is not always easy. 
- $\ell_1$ norm: $\operatorname{prox}_{\lambda \| \cdot \|_1}(x) = \operatorname{sign}(x) \odot \max(|x| - \lambda, 0)$, which is the soft-thresholding operator.
- $\ell_2$ norm: $\operatorname{prox}_{\lambda \| \cdot \|_2}(x) = \frac{x}{\|x\|_2} \max(\|x\|_2 - \lambda, 0)$.
- Convex set indicator function: $\operatorname{prox}_{\iota_C}(x) = \Pi_C(x)$, which is the projection onto the convex set $C$. 


### Proximal Gradient Descent

Consider
$$
\min \{ \psi(x) = f(x) + h(x) \},
$$
where $f$ is differentiable with $\operatorname{dom} f= \mathbb{R}^n$ and $h$ convex, maybe non-differentiable (with friendly proximal operator). 

Then the general idea is that: GD the smooth part $f$, and use proximal operator to handle the non-smooth part $h$:
$$
x^{k+1} = \operatorname{prox}_{\lambda_k h} \left( x^k - \lambda_k \nabla f(x^k) \right), \qquad \lambda_k > 0.
$$
- $\lambda_k > 0$ is the step size, can be fixed or given by line search.
  - If $f$ is $L$-smooth, then $\lambda_k = t \leq \frac{1}{L}$. If $L$ unknown, linesearch till satisfy Lipschitz condition:
    $$
    f(x^{k+1}) \leq f(x^k) + \langle \nabla f(x^k), x^{k+1} - x^k \rangle + \frac{1}{2\lambda_k} \|x^{k+1} - x^k\|^2.
    $$
- When $h=0$, it reduces to GD; when $h = \iota_C$, it reduces to projected GD.

***Interpretation***. According to the definition of proximal operator, the above update can be rewritten as:
$$
\begin{aligned}
x^{k+1} &= \arg\min_{u \in \operatorname{dom}(h)} \left\{ h(u) + \frac{1}{2\lambda_k} \|u - (x^k - \lambda_k \nabla f(x^k))\|^2 \right\}\\
&= \arg\min_{u \in \operatorname{dom}(h)} \left\{ \underbrace{h(u)}_{\text{kept unchanged}} + \underbrace{f(x^k) + \langle \nabla f(x^k), u - x^k \rangle}_{\text{linear approx. of } f} + \underbrace{\frac{1}{2\lambda_k} \|u - x^k\|^2}_{u \text{ not too far from } x^k} \right\}.
\end{aligned}
$$
- So it can be seen as: for one, use GD to decrease the smooth part $f$; for another, use proximal operator to minimize the non-smooth part $h$ while not too far *(so that the Taylor expansion is valid)* from the current point $x^k$.

Moreover, by $\text{(1)}$, $x^{k+1} = \operatorname{prox}_{\lambda_k h} \left( x^k - \lambda_k \nabla f(x^k) \right)$ is equivalent to:
$$
\frac{x^k - x^{k+1}}{\lambda_k} - \nabla f(x^k) \in \partial h(x^{k+1}),
$$
or equivalently,
$$
x^{k+1} = x^k - \lambda_k \left( \nabla f(x^k) + g^{k} \right), \qquad g^{k} \in \partial h(x^{k+1}),
$$
- Here, $g^k$ is a subgradient of $h$ at $x^{k+1}$ (note that it's $k+1$ not $k$). So it is also called *forward-backward splitting* or *explicit-implicit GD*. The update of $h$ is like a subgrad-GD which is true, but it cannot be explicitly computed, since the next point $x^{k+1}$ is unknown.



## From Proximal-GD to Mirror Descent via Projected-GD

### Projected Gradient Descent Revisited

Consider
$$
\min_{x \in \mathcal{X}}  f(x) ,
$$
and given its subgradient at $x^k$ as $g^k \in \partial f(x^k)$. Then the classical projected GD is:
$$
x^{k+1} = \Pi_{\mathcal{X}}(x^k - \eta_k g^k) = \arg\min_{x \in \mathcal{X}} \frac{1}{2} \|x - (x^k - \eta_k g^k)\|^2
, \qquad \text{(2)}
$$
and we've already shown that it is a special case of proximal-GD.

Yet, $\text{(2)}$ can be rewritten as:
$$
x^{k+1} = \arg\min_{x \in \mathcal{X}} \left\{ \langle g^k, x - x^k \rangle + \frac{1}{2\eta_k} \|x - x^k\|^2 \right\},
$$
- Similar to the interpretation of proximal-GD, though it's simpler. Its general form is still: *linear approximation* + *proximity control*, i.e., walk along the linear extension to find a better point, but not too far.

### Mirror Descent

> [!quote]
> References
> - Nemirovski & Yudin (1983), Problem Complexity and Method Efficiency in Optimization
> - Beck & Teboulle (2003), Mirror Descent and Nonlinear Projected Subgradient Methods for Convex Optimization

Mirror descent basically follows the same idea, but it challenges the Euclidean geometry of the space: why not use a more general geometry to measure 'not too far'? It replaces the Euclidean distance $\frac{1}{2} \|x - x^k\|^2$ by a more general Bregman divergence $D_{\phi}(x, x^k)$, and the general update is:
$$
x^{k+1} = \arg\min_{x \in \mathcal{X}} \left\{ \eta_k \langle g^k, x - x^k \rangle + D_{\phi}(x, x^k) \right\}, \qquad \text{(3)}
$$
where $\phi$ is a strongly convex function as a auxiliary function to define the geometry of the space, and the Bregman divergence is defined as:
$$
D_{\phi}(x, y) = \phi(x) - \phi(y) - \langle \nabla \phi(y), x - y \rangle.
$$
- If $\phi(x) = \frac{1}{2} \|x\|^2$, then $D_{\phi}(x, y) = \frac{1}{2} \|x - y\|^2$, and mirror descent reduces to projected GD.
- For example, if $\mathcal{X} = \Delta_n$ (the probability simplex), then we can choose entropy $\phi(x) = \sum_{i=1}^n x_i \log x_i$, and the Bregman divergence is the KL divergence:
    $$
    D_{\phi}(x, y) = \sum_{i=1}^n x_i \log \frac{x_i}{y_i}.
    $$

Expanding the Bregman divergence, we can rewrite $\text{(3)}$ as:
$$
x^{k+1} = \arg\min_{x \in \mathcal{X}} \left\{ \phi(x) - \langle \nabla \phi(x^k) - \eta_k g^k, x \rangle \right\} \qquad \text{(4)}
$$
The optimality condition of this minimization problem $\text{(4)}$ is (here we first assume $\mathcal{X} = \mathbb{R}^n$ for simplicity):
$$
\nabla \phi(x^{k+1}) = \nabla \phi(x^k) - \eta_k g^k, 
$$
- Interesting, it's like Mirror Descent is a *gradient descent to $\nabla \phi(x)$*, and then find back the corresponding $x^{k+1}$ by $(\nabla \phi)^{-1}$. 
- Another great thing is that, for some proper strongly convex $\phi$, $(\nabla \phi)^{-1} = \nabla (\phi^*)$, where
    $$
    \phi^*(y) := \sup_{x \in \mathbb{R}^n} \{ \langle x, y \rangle - \phi(x) \}
    $$
    is the convex conjugate of $\phi$. 

Therefore, denote $y^k = \nabla \phi(x^k)$, and utilize property $(\nabla \phi)^{-1} = \nabla \phi^*$, we can formulate the mirror descent update as:
$$
\begin{aligned}
x^{k}&= \nabla \phi^*(y^k),\\
y^{k+1} &= y^k - \eta_k g^k,\\
x^{k+1} &= \nabla \phi^*(y^{k+1}).
\end{aligned}
$$

***Formal Definition of Mirror Descent***. 


***Rethinking Mirror Descent Beyond the Euclidean Geometry***. 

Euclidean geometry is too special a space to conduct optimization. 

- For Mirror Descent, take $\phi(x) = \frac{1}{2} \|x\|^2$, then $\nabla \phi(x) = x$, and $\nabla \phi^*(y) = y$. Thus, all the mappings are identity, and the update reduces to the classical GD:
    $$
    x^{k+1} = x^k - \eta_k g^k.
    $$

- However, if we look closer to $x^k - \eta_k g^k$, we are adding $x^k$ with $g^k$, where $x$ is the *primal variable* and $g$ is the *linear functional* in the dual-space objective. It is the special structure of Euclidean space that allows us to add them together. 

Mirror Descent considers a more general geometry, where the primal space and dual space are not necessarily the same. Thus, it requires to first map the primal variable $x^k$ to the dual space by $\nabla \phi$, then add the linear functional $g^k$ in the dual space, and finally map back to the primal space by $\nabla \phi^*$.

By introducing the Bregman divergence, Mirror Descent generalizes the projection to nonlinear/non-orthogonal projection.