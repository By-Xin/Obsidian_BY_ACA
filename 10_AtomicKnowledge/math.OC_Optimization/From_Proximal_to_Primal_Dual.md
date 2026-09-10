# From Proximal to Primal-Dual

It is a simple study note of PDHG series work. But all the way back to the proximal algorithms. 

## Proximal Algorithms

### Proximal Operator

The proximal operator of a convex function $h$ is defined as:
$$
\operatorname{prox}_{h}(\bm{x}) = \arg\min_{\bm{u} \in \operatorname{dom}(h)} \left\{ h(\bm{u}) + \frac{1}{2} \|\bm{u} - \bm{x}\|^2 \right\}.
$$

- It can be interpreted as a general projection operator. It receives a point $\bm{x}$ and returns a point not too far from $\bm{x}$. Such a $\bm{u}$ is the optimal neighbor of $\bm{x}$ that relatively minimizes $h$. Theorem can guarantee the existence and uniqueness of the proximal operator, for some proper $h$.

Proximal operator is closely related to the subgradient of $h$. Acutually, for some proper $h$:
$$
\bm{u} = \operatorname{prox}_{h}(\bm{x}) \iff \bm{x} - \bm{u} \in \partial h(\bm{u}). \qquad {\text{(1)}}
$$
- Intuitively, $\bm{x}-\bm{u}$ is the gradient of $\frac{1}{2} \|\bm{u} - \bm{x}\|^2$ at $\bm{u}$. Thus, $\bm{x}-\bm{u} \in \partial h(\bm{u})$ means that the gradient (or *force*) of $h$ at $\bm{u}$ balances that of $\frac{1}{2} \|\bm{u} - \bm{x}\|^2$. And thus reach the optimal.

More frequently, we use the following form:
$$
\operatorname{prox}_{\lambda h}(\bm{x}) = \arg\min_{\bm{u} \in \operatorname{dom}(h)} \left\{ h(\bm{u}) + \frac{1}{2\lambda} \|\bm{u} - \bm{x}\|^2 \right\}.
$$
- $\lambda \in \mathbb{R}_{+}$ is a scaling factor to control the strength of the regularization term $\frac{1}{2\lambda} \|\bm{u} - \bm{x}\|^2$. 
- It actually is to replace $h$ by $\lambda h$ in the original definition. Yet $\arg\min \{ \lambda h(\bm{u}) + \frac{1}{2} \|\bm{u} - \bm{x}\|^2 \} = \arg\min \{ h(\bm{u}) + \frac{1}{2\lambda} \|\bm{u} - \bm{x}\|^2 \}$, so. 


Calculation of proximal operator is bascially solving a convex optimization problem, though it is not always easy. 
- $\ell_1$ norm: $\operatorname{prox}_{\lambda \| \cdot \|_1}(\bm{x}) = \operatorname{sign}(\bm{x}) \odot \max(|\bm{x}| - \lambda, 0)$, which is the soft-thresholding operator.
- $\ell_2$ norm: $\operatorname{prox}_{\lambda \| \cdot \|_2}(\bm{x}) = \frac{\bm{x}}{\|\bm{x}\|_2} \max(\|\bm{x}\|_2 - \lambda, 0)$.
- Convex set indicator function: $\operatorname{prox}_{\iota_C}(\bm{x}) = \Pi_C(\bm{x})$, which is the projection onto the convex set $C$. 


### Proximal Gradient Descent

Consider
$$
\min \{ \psi(\bm{x}) = f(\bm{x}) + h(\bm{x}) \},
$$
where $f$ is differentiable with $\operatorname{dom} f= \mathbb{R}^n$ and $h$ convex, maybe non-differentiable (with friendly proximal operator). 

Then the general idea is that: GD the smooth part $f$, and use proximal operator to handle the non-smooth part $h$:
$$
\bm{x}^{k+1} = \operatorname{prox}_{\lambda_k h} \left( \bm{x}^k - \lambda_k \nabla f(\bm{x}^k) \right), \qquad \lambda_k > 0.
$$
- $\lambda_k > 0$ is the step size, can be fixed or given by line search.
  - If $f$ is $L$-smooth, then $\lambda_k = t \leq \frac{1}{L}$. If $L$ unknown, linesearch till satisfy Lipschitz condition:
    $$
    f(\bm{x}^{k+1}) \leq f(\bm{x}^k) + \left\langle \nabla f(\bm{x}^k), \bm{x}^{k+1} - \bm{x}^k \right\rangle + \frac{1}{2\lambda_k} \|\bm{x}^{k+1} - \bm{x}^k\|^2.
    $$
- When $h=0$, it reduces to GD; when $h = \iota_C$, it reduces to projected GD.

***Interpretation***. According to the definition of proximal operator, the above update can be rewritten as:
$$
\begin{aligned}
\bm{x}^{k+1} &= \arg\min_{\bm{u} \in \operatorname{dom}(h)} \left\{ h(\bm{u}) + \frac{1}{2\lambda_k} \left\|\bm{u} - (\bm{x}^k - \lambda_k \nabla f(\bm{x}^k))\right\|^2 \right\}\\
&= \arg\min_{\bm{u} \in \operatorname{dom}(h)} \left\{ \underbrace{h(\bm{u})}_{\text{kept unchanged}} + \underbrace{f(\bm{x}^k) + \left\langle \nabla f(\bm{x}^k), \bm{u} - \bm{x}^k \right\rangle}_{\text{linear approx. of } f} + \underbrace{\frac{1}{2\lambda_k} \|\bm{u} - \bm{x}^k\|^2}_{\bm{u} \text{ not too far from } \bm{x}^k} \right\}.
\end{aligned}
$$
- So it can be seen as: for one, use GD to decrease the smooth part $f$; for another, use proximal operator to minimize the non-smooth part $h$ while not too far *(so that the Taylor expansion is valid)* from the current point $\bm{x}^k$.

Moreover, by $\text{(1)}$, $\bm{x}^{k+1} = \operatorname{prox}_{\lambda_k h} \left( \bm{x}^k - \lambda_k \nabla f(\bm{x}^k) \right)$ is equivalent to:
$$
\frac{\bm{x}^k - \bm{x}^{k+1}}{\lambda_k} - \nabla f(\bm{x}^k) \in \partial h(\bm{x}^{k+1}),
$$
or equivalently,
$$
\bm{x}^{k+1} = \bm{x}^k - \lambda_k \left( \nabla f(\bm{x}^k) + \bm{g}^{k} \right), \qquad \bm{g}^{k} \in \partial h(\bm{x}^{k+1}),
$$
- Here, $\bm{g}^k$ is a subgradient of $h$ at $\bm{x}^{k+1}$ (note that it's $k+1$ not $k$). So it is also called *forward-backward splitting* or *explicit-implicit GD*. The update of $h$ is like a subgrad-GD which is true, but it cannot be explicitly computed, since the next point $\bm{x}^{k+1}$ is unknown.



## From Proximal-GD to Mirror Descent via Projected-GD

### Projected Gradient Descent Revisited

Consider
$$
\min_{\bm{x} \in \mathcal{X}}  f(\bm{x}) ,
$$
and given its subgradient at $\bm{x}^k$ as $\bm{g}^k \in \partial f(\bm{x}^k)$. Then the classical projected GD is:
$$
\bm{x}^{k+1} = \Pi_{\mathcal{X}}(\bm{x}^k - \eta_k \bm{g}^k) = \arg\min_{\bm{x} \in \mathcal{X}} \frac{1}{2} \|\bm{x} - (\bm{x}^k - \eta_k \bm{g}^k)\|^2
, \qquad \text{(2)}
$$
and we've already shown that it is a special case of proximal-GD.

Yet, $\text{(2)}$ can be rewritten as:
$$
\bm{x}^{k+1} = \arg\min_{\bm{x} \in \mathcal{X}} \left\{ \left\langle \bm{g}^k, \bm{x} - \bm{x}^k \right\rangle + \frac{1}{2\eta_k} \|\bm{x} - \bm{x}^k\|^2 \right\},
$$
- Similar to the interpretation of proximal-GD, though it's simpler. Its general form is still: *linear approximation* + *proximity control*, i.e., walk along the linear extension to find a better point, but not too far.

### Mirror Descent

> [!quote]
> References
> - Nemirovski & Yudin (1983), Problem Complexity and Method Efficiency in Optimization
> - Beck & Teboulle (2003), Mirror Descent and Nonlinear Projected Subgradient Methods for Convex Optimization

Mirror descent basically follows the same idea, but it challenges the Euclidean geometry of the space: why not use a more general geometry to measure 'not too far'? It replaces the Euclidean distance $\frac{1}{2} \|\bm{x} - \bm{x}^k\|^2$ by a more general Bregman divergence $D_{\phi}(\bm{x}, \bm{x}^k)$, and the general update is:
$$
\bm{x}^{k+1} = \arg\min_{\bm{x} \in \mathcal{X}} \left\{ \eta_k \left\langle \bm{g}^k, \bm{x} - \bm{x}^k \right\rangle + D_{\phi}(\bm{x}, \bm{x}^k) \right\}, \qquad \text{(3)}
$$
where $\phi$ is a strongly convex function as a auxiliary function to define the geometry of the space, and the Bregman divergence is defined as:
$$
D_{\phi}(\bm{x}, \bm{y}) = \phi(\bm{x}) - \phi(\bm{y}) - \left\langle \nabla \phi(\bm{y}), \bm{x} - \bm{y} \right\rangle.
$$
- If $\phi(\bm{x}) = \frac{1}{2} \|\bm{x}\|^2$, then $D_{\phi}(\bm{x}, \bm{y}) = \frac{1}{2} \|\bm{x} - \bm{y}\|^2$, and mirror descent reduces to projected GD.
- For example, if $\mathcal{X} = \Delta_n$ (the probability simplex), then we can choose entropy $\phi(\bm{x}) = \sum_{i=1}^n x_i \log x_i$, and the Bregman divergence is the KL divergence:
    $$
    D_{\phi}(\bm{x}, \bm{y}) = \sum_{i=1}^n x_i \log \frac{x_i}{y_i}.
    $$

Expanding the Bregman divergence, we can rewrite $\text{(3)}$ as:
$$
\boxed{\bm{x}^{k+1} = \arg\min_{\bm{x} \in \mathcal{X}} \left\{ \phi(\bm{x}) - \left\langle \nabla \phi(\bm{x}^k) - \eta_k \bm{g}^k, \bm{x} \right\rangle \right\}} \qquad \text{(4)}
$$
which is the most fundamental form of mirror descent. 

In specific, consider the optimality condition to further understand the update $\text{(4)}$.

- The optimality condition of this minimization problem $\text{(4)}$ is (here we first assume $\mathcal{X} = \mathbb{R}^n$ for simplicity):
    $$
    \nabla \phi(\bm{x}^{k+1}) = \nabla \phi(\bm{x}^k) - \eta_k \bm{g}^k, \quad \text{(5)}
    $$
  If more generally consider $\mathcal{X} \subset \mathbb{R}^n$, then the optimality condition of $\text{(4)}$ is:
    $$
    \nabla \phi(\bm{x}^{k}) - \eta_k \bm{g}^k - \nabla \phi(\bm{x}^{k+1}) \in N_{\mathcal{X}}(\bm{x}^{k+1})\\ \implies \nabla \phi(\bm{x}^{k+1}) = \nabla \phi(\bm{x}^k) - \eta_k \bm{g}^k + \bm{\nu}^k, \quad \bm{\nu}^k \in N_{\mathcal{X}}(\bm{x}^{k+1}) \quad \text{(6)}
    $$
    where $N_{\mathcal{X}}(\bm{x})$ is the normal cone of $\mathcal{X}$ at $\bm{x}$. 

- We first focus on $\text{(5)}$ (the unconstrained case), which is more intuitive. 
  - It shows that, to solve $\text{(4)}$, it suffices to find the $\bm{x}^{k+1}$ such that $\text{(5)}$ holds. It's equivalent to: first conduct a gradient descent to $\nabla \phi(\bm{x})$, and then find back the corresponding $\bm{x}^{k+1}$ by $(\nabla \phi)^{-1}$.
  - Moreover, denote $\phi^*(\bm{y}) := \sup_{\bm{x} \in \mathbb{R}^n} \{ \langle \bm{x}, \bm{y} \rangle - \phi(\bm{x}) \}$ as the convex conjugate of $\phi$, then we have $(\partial \phi)^{-1} = \nabla (\phi^*)$. And then the strong convexity of $\phi$ guarantees $(\nabla \phi)^{-1} = \nabla \phi^*$ .
  - Therefore, denote $\bm{y}^k = \nabla \phi(\bm{x}^k)$, and utilize property $(\nabla \phi)^{-1} = \nabla \phi^*$, we can formulate the mirror descent update as:
      $$
      \begin{aligned}
      \bm{y}^{k+1} &= \nabla \phi(\bm{x}^k) - \eta_k \bm{g}^k, \\
      \bm{x}^{k+1} &= \nabla \phi^*(\bm{y}^{k+1})
      \end{aligned} \quad (\star)
      $$

- More generally, for constrained case $\text{(6)}$,
  -  The objective is actually $F(\bm{x}) := \phi(\bm{x}) + \iota_{\mathcal{X}}(\bm{x})$, thus natually $\partial F(\bm{x}) = \nabla \phi(\bm{x}) + N_{\mathcal{X}}(\bm{x})$, and thus
    $$
    \nabla F^* = (\partial F)^{-1} = (\nabla \phi + N_{\mathcal{X}})^{-1} 
    $$
  - Thus, even for the closed form , the $(\star)$ update basically still holds, but we need to replace $\nabla \phi^*$ by $\nabla F^*$, and denote the *mirror back-mapping* as $Q(\bm{y}) := \arg\max_{\bm{x} \in \mathcal{X}} \{ \langle \bm{x}, \bm{y} \rangle - \phi(\bm{x}) \}$, then the update is:
      $$
      \begin{aligned}
      \bm{y}^{k+1} &= \nabla \phi(\bm{x}^k) - \eta_k \bm{g}^k, \\
      \bm{x}^{k+1} &= Q(\bm{y}^{k+1})
      \end{aligned}
      $$

***Rethinking Mirror Descent Beyond the Euclidean Geometry***. 

Euclidean geometry is too special a space to conduct optimization. 

- For Mirror Descent, take $\phi(\bm{x}) = \frac{1}{2} \|\bm{x}\|^2$, then $\nabla \phi(\bm{x}) = \bm{x}$, and $\nabla \phi^*(\bm{y}) = \bm{y}$. Thus, all the mappings are identity, and the update reduces to the classical GD:
    $$
    \bm{x}^{k+1} = \bm{x}^k - \eta_k \bm{g}^k.
    $$

- However, if we look closer to $\bm{x}^k - \eta_k \bm{g}^k$, we are adding $\bm{x}^k$ with $\bm{g}^k$, where $\bm{x}$ is the *primal variable* and $\bm{g}$ is the *linear functional* in the dual-space objective. It is the special structure of Euclidean space that allows us to add them together. 

Mirror Descent considers a more general geometry, where the primal space and dual space are not necessarily the same. Thus, it requires to first map the primal variable $\bm{x}^k$ to the dual space by $\nabla \phi$, then add the linear functional $\bm{g}^k$ in the dual space, and finally map back to the primal space by $\nabla \phi^*$.

By introducing the Bregman divergence, Mirror Descent generalizes the projection to nonlinear/non-orthogonal projection.


***Example: Mirror Descent on Probability Simplex***.

Let $\mathcal{X} = \Delta_n := \left\{x_i \geq 0, \sum_{i=1}^n x_i = 1 \right\}$ be the probability simplex, and choose $\phi(\bm{x}) = \sum_{i=1}^n x_i \log x_i$ as the auxiliary function. Then the Mirror Descent update is:
$$
\begin{aligned}
y_i^{k+1} &= \nabla \phi(\bm{x}^k)_i - \eta_k g_i^k = 1 + \log x_i^k - \eta_k g_i^k, \\
x_i^{k+1} &= Q(\bm{y}^{k+1})_i = \frac{\exp(y_i^{k+1} - 1)}{\sum_{j=1}^n \exp(y_j^{k+1} - 1)} = \frac{x_i^k \exp(-\eta_k g_i^k)}{\sum_{j=1}^n x_j^k \exp(-\eta_k g_j^k)}.
\end{aligned}
$$

## Augmented Lagrangian Method (ALM) and its Connection to Proximal Algorithms

> [!quote]
> References
> - Deng, K., Wang, R., Zhu, Z., Zhang, J., & Wen, Z. (2025). The Augmented Lagrangian Methods: Overview and Recent Advances. arXiv preprint arXiv:2510.16827.

### One Intuition of ALM

ALM is a method to solve constrained optimization problems. There are many aspects to understand it. 

Given a simple constrained optimization problem:
$$
\min f(\bm{x}) \quad \text{s.t.} \quad \bm{c}(\bm{x}) = \bm{0}.
$$
There are two natural ways to solve it:
- By *penalty method*, we can convert it to an unconstrained problem:
    $$
    \min f(\bm{x}) + \frac{\rho}{2} \|\bm{c}(\bm{x})\|^2,
    $$
    where $\rho > 0$ is a penalty parameter. However, theoretically, only as $\rho \to \infty$, the constraint $\bm{c}(\bm{x}_\rho) \to \bm{0}$ can be guaranteed. But then the problem becomes ill-conditioned and hard to solve.

- By *Lagrangian multiplier method*, we consider its Lagrangian function:
    $$
    \mathcal{L}(\bm{x}, \bm{\lambda}) = f(\bm{x}) + \bm{\lambda}^\top \bm{c}(\bm{x}),
    $$
    and the KKT condition gives
    $$
    \nabla f(\bm{x}^\star) + \bm{J}_c(\bm{x}^\star)^\top \bm{\lambda}^\star = \bm{0}, \quad \bm{c}(\bm{x}^\star) = \bm{0},
    $$
    where $\bm{J}_c(\bm{x})$ is the Jacobian of $\bm{c}(\bm{x})$. However, to minimize $\mathcal{L}(\bm{x}, \bm{\lambda})$, it's kind of like giving the constraint a linear penalty, which is not strong enough to enforce the constraint.

A natural idea is to combine the two.

### Classical Augmented Lagrangian Method

Consider the classical Powell-Hestenes-Rockafellar (PHR) form of ALM for constrained optimization.  Still consider the simple case with only the equality constraint $\bm{c}(\bm{x}) = \bm{0}$. Then, the AL function is defined as:
$$
\mathcal{L}_\rho(\bm{x}, \bm{\lambda}) = f(\bm{x}) + \bm{\lambda}^\top \bm{c}(\bm{x}) + \frac{\rho}{2} \|\bm{c}(\bm{x})\|^2,
$$
which, by completing the square, can be rewritten as
$\mathcal{L}_\rho(\bm{x}, \bm{\lambda}) = f(\bm{x}) + \frac{\rho}{2} \left\| \bm{c}(\bm{x}) + \frac{\bm{\lambda}}{\rho} \right\|^2 - \frac{1}{2\rho} \|\bm{\lambda}\|^2$, and then minimizing $\mathcal{L}_\rho(\bm{x}, \bm{\lambda})$ w.r.t. $\bm{x}$ is equivalent to minimizing
$$
\min_{\bm{x}} \left\{ f(\bm{x}) + \frac{\rho}{2} \left\| \bm{c}(\bm{x}) + \frac{\bm{\lambda}}{\rho} \right\|^2 \right\}.
$$


## Primal-Dual Series

### Primal-Dual Hybrid Gradient (PDHG)

PDHG is a primal-dual algorithm to solve the saddle point problem. We start from maybe the most simple case, the LP problem:
$$
(\text{LP}) \qquad 
\min_{\bm{x} \in \mathbb{R}^n}   \bm{c}^\top \bm{x} , \quad \text{s.t. } A\bm{x} = \bm{b}, \quad \bm{x} \geq 0,
$$
where $A \in \mathbb{R}^{m \times n}$, $\bm{b} \in \mathbb{R}^m$, $\bm{c} \in \mathbb{R}^n$.

The Lagrangian function of this problem is:
$$
\mathcal{L}(\bm{x}, \bm{y}) = \bm{c}^\top \bm{x} - \bm{y}^\top (A\bm{x} - \bm{b}),
$$
and thus the primal problem is equivalent to the saddle point problem:
$$
\min_{\bm{x} \geq 0} \max_{\bm{y}} \mathcal{L}(\bm{x}, \bm{y}) = \min_{\bm{x} \geq 0} \max_{\bm{y}} \left\{ \bm{c}^\top \bm{x} - \bm{y}^\top (A\bm{x} - \bm{b}) \right\}.
$$

***Classical PDHG***. The naive PDHG is to update the primal variable $\bm{x}$ and dual variable $\bm{y}$ alternatively by GD with proximal regularization:
$$
\begin{aligned}
\bm{x}^{k+1} &= \arg\min_{\bm{x} \geq 0} \left\{ \bm{c}^\top \bm{x} - (\bm{y}^k)^\top (A\bm{x} - \bm{b}) + \frac{1}{2\tau} \|\bm{x} - \bm{x}^k\|^2 \right\} = \left[ \bm{x}^k - \tau (\bm{c} - A^\top \bm{y}^k) \right]_+, \\
\bm{y}^{k+1} &= \arg\max_{\bm{y}} \left\{ \bm{b}^\top \bm{y} - \bm{y}^\top A \bm{x}^{k+1} - \frac{1}{2\sigma} \|\bm{y} - \bm{y}^k\|^2
\right\} = \bm{y}^k - \sigma (A \bm{x}^{k+1} - \bm{b}).
\end{aligned}
$$

- Intuitively, given Lagrangian multipliers $\bm{y}^k$, we calculate the $\min_{\bm{x} \geq 0} \mathcal{L}(\bm{x}, \bm{y}^k)$; then given the primal variable $\bm{x}^{k+1}$, we calculate the $\max_{\bm{y}} \mathcal{L}(\bm{x}^{k+1}, \bm{y})$. Proximal regularization is added to stabilize the update.
- However, since the primal and dual variables are tightly coupled, such saddle point dynamics may cause oscillation and divergence. 

***Chambolle-Pock PDHG***. To stabilize the update, Chambolle and Pock proposed to add an extrapolation step to the primal variable $\bm{x}$, denote an extrapolated variable $\bar{\bm{x}}^{k+1} = \bm{x}^{k+1} + (\bm{x}^{k+1} - \bm{x}^k)$, and then update the dual variable $\bm{y}$ by $\bar{\bm{x}}^{k+1}$ instead of $\bm{x}^{k+1}$:
$$
\begin{aligned}
\bm{x}^{k+1} &= \arg\min_{\bm{x} \geq 0} \left\{ \bm{c}^\top \bm{x} - (\bm{y}^k)^\top (A\bm{x} - \bm{b}) + \frac{1}{2\tau} \|\bm{x} - \bm{x}^k\|^2 \right\} = \left[ \bm{x}^k - \tau (\bm{c} - A^\top \bm{y}^k) \right]_+, \\
\bar{\bm{x}}^{k+1} &= \bm{x}^{k+1} + (\bm{x}^{k+1} - \bm{x}^k), \\
\bm{y}^{k+1} &= \arg\max_{\bm{y}} \left\{ \bm{b}^\top \bm{y} - \bm{y}^\top A \bar{\bm{x}}^{k+1} - \frac{1}{2\sigma} \|\bm{y} - \bm{y}^k\|^2
\right\} = \bm{y}^k - \sigma (A \bar{\bm{x}}^{k+1} - \bm{b}).
\end{aligned}
$$
