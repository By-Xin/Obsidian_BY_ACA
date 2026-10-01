# Proximal three-point identity

## Proximal operator

Consider the following optimization problem:
$$
\min_{\boldsymbol{x} \in \mathbb{R}^n} \{F(\boldsymbol{x}) := f(\boldsymbol{x}) + \omega(\boldsymbol{x})\},
$$
where $f: \mathbb{R}^n \to \mathbb{R}$ is a differentiable convex function and $\omega: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ is a proper closed convex function as a regularizer, which may be non-differentiable. 

At iteration $t$, we have current point $\boldsymbol{x}_t$ and a first-order direction $\boldsymbol{g}_t$. Then the next iterate $\boldsymbol{x}_{t+1}$ is obtained by
$$
\boldsymbol{x}_{t+1} = \arg\min_{\boldsymbol{x} \in \mathbb{R}^n} \left\{
    \underbrace{\langle\boldsymbol{g}_t, \boldsymbol{x}\rangle}_{\text{1-order direction}} + 
    \underbrace{\omega(\boldsymbol{x})}_{\text{regularizer}}
     + 
     \underbrace{\frac{1}{2\eta_t}\|\boldsymbol{x}-\boldsymbol{x}_t\|_2^2}_{\text{proximal term}}
\right\},~ \eta_t > 0,
$$

- $\boldsymbol{g}_t$ could be any first-order direction (e.g., gradient, subgradient, stochastic gradient, or even just some arbitrary direction by some first-order oracle). 

## The three-point identitys

The first-order optimality condition of the above problem is
$\boldsymbol{0} \in \boldsymbol{g}_t + \partial \omega(\boldsymbol{x}_{t+1}) + \frac{1}{\eta_t}(\boldsymbol{x}_{t+1} - \boldsymbol{x}_t)$,
which is equivalent to, for some $\boldsymbol{s}_{t+1} \in \partial \omega(\boldsymbol{x}_{t+1})$,
$$
\boldsymbol{g}_t + \boldsymbol{s}_{t+1} = -\frac{1}{\eta_t}(\boldsymbol{x}_{t+1} - \boldsymbol{x}_t) \tag{1}.
$$

Given any reference point $\bar{\boldsymbol{x}} \in \mathbb{R}^n$, since $\omega$ is convex, we have $\omega(\bar{\boldsymbol{x}}) \geq \omega(\boldsymbol{x}_{t+1}) + \langle \boldsymbol{s}_{t+1}, \bar{\boldsymbol{x}} - \boldsymbol{x}_{t+1} \rangle$, then adding $\langle \boldsymbol{g}_t, \bar{\boldsymbol{x}} - \boldsymbol{x}_{t+1} \rangle$ to both sides, we have 
$\langle \boldsymbol{g}_t,  \boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}}\rangle + \omega(\boldsymbol{x}_{t+1}) - \omega(\bar{\boldsymbol{x}}) \leq \langle \boldsymbol{s}_{t+1} +\boldsymbol{g}_t, \boldsymbol{x}_{t+1} -\boldsymbol{\bar{x}} \rangle$.
$$
\left\langle \boldsymbol{g}_t,  \boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}}\right\rangle + \omega(\boldsymbol{x}_{t+1}) - \omega(\bar{\boldsymbol{x}}) \leq \frac{1}{\eta_t}\langle \boldsymbol{x}_t - \boldsymbol{x}_{t+1}, \boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}} \rangle.
$$

A useful identity is that for any $\boldsymbol{a}, \boldsymbol{b}, \boldsymbol{c} \in \mathbb{R}^n$,
$$
2\langle \boldsymbol{a} - \boldsymbol{b}, \boldsymbol{b} - \boldsymbol{c} \rangle = \|\boldsymbol{a} - \boldsymbol{c}\|_2^2 - \|\boldsymbol{a} - \boldsymbol{b}\|_2^2 - \|\boldsymbol{b} - \boldsymbol{c}\|_2^2.
$$
Let $\boldsymbol{a} = \boldsymbol{x}_t, \boldsymbol{b} = \boldsymbol{x}_{t+1}, \boldsymbol{c} = \bar{\boldsymbol{x}}$, we have $2\langle \boldsymbol{x}_t - \boldsymbol{x}_{t+1}, \boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}} \rangle = \|\boldsymbol{x}_t - \bar{\boldsymbol{x}}\|_2^2 - \|\boldsymbol{x}_t - \boldsymbol{x}_{t+1}\|_2^2 - \|\boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}}\|_2^2$.
Thus, we have the following three-point identity:
$$
\left\langle \boldsymbol{g}_t,  \boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}}\right\rangle + \omega(\boldsymbol{x}_{t+1}) - \omega(\bar{\boldsymbol{x}}) \leq \frac{1}{2\eta_t}\left(\|\boldsymbol{x}_t - \bar{\boldsymbol{x}}\|_2^2 - \|\boldsymbol{x}_t - \boldsymbol{x}_{t+1}\|_2^2 - \|\boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}}\|_2^2\right).
$$

Several notes:
- The inequality is due to the convexity of $\omega$. If $\omega \equiv 0$, then $\boldsymbol{s}_{t+1} = 0$ and the inequality becomes an equality. 
- The three-point identity transforms the first-order quantity to distance-based quantities, which is useful in convergence analysis, especially $\| \boldsymbol{x}_{t}  - \bar{\boldsymbol{x}} \|_2^2 - \|\boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}}\|_2^2$ can be telescoped over iterations.

## Mirror Descent version of the three-point identity

In a more general sence, we could consider the Mirror Descent method, where the Euclidean distance is replaced by a Bregman divergence $D_h(\boldsymbol{x}, \boldsymbol{y}) = h(\boldsymbol{x}) - h(\boldsymbol{y}) - \langle \nabla h(\boldsymbol{y}), \boldsymbol{x} - \boldsymbol{y} \rangle$ for some strongly convex function $h$. Then the mirror step is (we first temporarily ignore the regularizer $\omega$ and $\eta_t$ is the step size):
$$
\boldsymbol{x}_{t+1} = \arg\min_{\boldsymbol{x} \in \mathbb{R}^n} \left\{
   \eta_t \langle \boldsymbol{g}_t, \boldsymbol{x} \rangle + D_h(\boldsymbol{x}, \boldsymbol{x}_t)
\right\}, \qquad \eta_t > 0.
$$
- It still follows the *first-order direction* + *proximal geometry* structure.

Similar to the Euclidean case, we have the following basic equality: for any $\boldsymbol{a}, \boldsymbol{b}, \boldsymbol{c} \in \mathbb{R}^n$,
$$
\langle \nabla h(\boldsymbol{b}) - \nabla h(\boldsymbol{c}), \boldsymbol{a} - \boldsymbol{b} \rangle = D_h(\boldsymbol{a}, \boldsymbol{c}) - D_h(\boldsymbol{a}, \boldsymbol{b}) - D_h(\boldsymbol{b}, \boldsymbol{c}).
$$

*Proof.* (Basically directly obtained) by the definition of Bregman divergence, we have
$$
\begin{aligned}
D_h(\boldsymbol{a}, \boldsymbol{c}) - D_h(\boldsymbol{a}, \boldsymbol{b}) - D_h(\boldsymbol{b}, \boldsymbol{c}) &=- \langle \nabla h(\boldsymbol{c}), \boldsymbol{a} - \boldsymbol{c} \rangle + \langle \nabla h(\boldsymbol{b}), \boldsymbol{a} - \boldsymbol{b} \rangle + \langle \nabla h(\boldsymbol{c}), \boldsymbol{b} - \boldsymbol{c} \rangle\\
&= \langle \nabla h(\boldsymbol{b}) - \nabla h(\boldsymbol{c}), \boldsymbol{a} - \boldsymbol{b} \rangle.
\end{aligned}
$$

$\square$

Then, let $\boldsymbol{a} = \bar{\boldsymbol{x}}, \boldsymbol{b} = \boldsymbol{x}_{t+1}, \boldsymbol{c} = \boldsymbol{x}_t$, we have
$$
\langle \nabla h(\boldsymbol{x}_{t+1}) - \nabla h(\boldsymbol{x}_t), \bar{\boldsymbol{x}} - \boldsymbol{x}_{t+1} \rangle = D_h(\bar{\boldsymbol{x}}, \boldsymbol{x}_t) - D_h(\bar{\boldsymbol{x}}, \boldsymbol{x}_{t+1}) - D_h(\boldsymbol{x}_{t+1}, \boldsymbol{x}_t).
$$

For Mirror Descent, given that $\nabla_{\boldsymbol{x}} D_h(\boldsymbol{x}, \boldsymbol{x}_t) = \nabla h(\boldsymbol{x}) - \nabla h(\boldsymbol{x}_t)$, the first-order optimality condition is
$$
\langle \eta_t \boldsymbol{g}_t + \nabla h(\boldsymbol{x}_{t+1}) - \nabla h(\boldsymbol{x}_t), \boldsymbol{x} - \boldsymbol{x}_{t+1} \rangle \geq 0,~\forall \boldsymbol{x} \in \mathcal{X},
$$
Then plugin the three-point equality, we have the following three-point identity for Mirror Descent:
$$
\eta_t \langle \boldsymbol{g}_t, \boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}} \rangle \leq D_h(\bar{\boldsymbol{x}}, \boldsymbol{x}_t) - D_h(\bar{\boldsymbol{x}}, \boldsymbol{x}_{t+1}) - D_h(\boldsymbol{x}_{t+1}, \boldsymbol{x}_t).
$$
- If let $h(\boldsymbol{x}) = \frac{1}{2}\|\boldsymbol{x}\|_2^2$, then $D_h(\boldsymbol{x}, \boldsymbol{y}) = \frac{1}{2}\|\boldsymbol{x} - \boldsymbol{y}\|_2^2$, and the Mirror Descent three-point identity reduces to the Euclidean three-point identity.

We could further consider the Mirror Descent with a regularizer $\omega$, then the composite Mirror Descent step is
$$
\boldsymbol{x}_{t+1} = \arg\min_{\boldsymbol{x} \in \mathcal{X}} \left\{
   \eta_t \langle \boldsymbol{g}_t, \boldsymbol{x} \rangle + D_h(\boldsymbol{x}, \boldsymbol{x}_t) + \eta_t \omega(\boldsymbol{x})
\right\}, \qquad \eta_t > 0.
$$
Let $\boldsymbol{s}_{t+1} \in \partial \omega(\boldsymbol{x}_{t+1})$, then the first-order optimality condition is
$$
\langle \eta_t \boldsymbol{g}_t + \eta_t \boldsymbol{s}_{t+1} + \nabla h(\boldsymbol{x}_{t+1}) - \nabla h(\boldsymbol{x}_t), \boldsymbol{x} - \boldsymbol{x}_{t+1} \rangle \geq 0, ~\forall \boldsymbol{x} \in \mathcal{X}, 
$$
which gives
$$
\eta_t \langle \boldsymbol{g}_t+\boldsymbol{s}_{t+1}, \boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}} \rangle \leq D_h(\bar{\boldsymbol{x}}, \boldsymbol{x}_t) - D_h(\bar{\boldsymbol{x}}, \boldsymbol{x}_{t+1}) - D_h(\boldsymbol{x}_{t+1}, \boldsymbol{x}_t).
$$
By the convexity of $\omega$, we have
$$
\omega(\boldsymbol{x}_{t+1}) - \omega(\bar{\boldsymbol{x}}) \leq \langle \boldsymbol{s}_{t+1}, \boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}} \rangle,
$$
combining the above two inequalities, we have the three-point identity for composite Mirror Descent:
$$
\eta_t \langle \boldsymbol{g}_t, \boldsymbol{x}_{t+1} - \bar{\boldsymbol{x}} \rangle + \eta_t (\omega(\boldsymbol{x}_{t+1}) - \omega(\bar{\boldsymbol{x}})) \leq D_h(\bar{\boldsymbol{x}}, \boldsymbol{x}_t) - D_h(\bar{\boldsymbol{x}}, \boldsymbol{x}_{t+1}) - D_h(\boldsymbol{x}_{t+1}, \boldsymbol{x}_t).
$$