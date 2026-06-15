# Multi-stage Stochastic Programming and Stochastic Dual Dynamic Programming (SDDP)

>-  https://www.youtube.com/watch?v=jCkHtvTe7tU&t=42s
> 
> - Joaquim Dias Garcia (  / joaquim-dias-garcia  ) Guest Lecture for the Optimal Control & Learning Course (https://github.com/LearningToOptimize....
> 
> - The lecture covered Multistage Stochastic Programming and Stochastic Dual Dynamic Programming (SDDP).

## Basic Startups: Linear Programming

Consider the following linear programming problem:
$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} & \quad \mathbf{c}^\top \mathbf{x} \\
\text{s.t.} & \quad \mathbf{A} \mathbf{x} = \mathbf{b}, \\
& \quad \mathbf{x} \geq 0,
\end{aligned}
$$
where $\mathbf{c} \in \mathbb{R}^n$ is the cost vector, $\mathbf{A} \in \mathbb{R}^{m \times n}$ is the constraint matrix, and $\mathbf{b} \in \mathbb{R}^m$ is the right-hand side vector.

The duality in LP gives:
$$
\text{Primal:} \quad
\begin{aligned}
\mathbf{z}^\star = \min_{\mathbf{x} \in \mathbb{R}^n} & \quad \mathbf{c}^\top \mathbf{x} \\
\text{s.t.} & \quad \mathbf{A} \mathbf{x} = \mathbf{b}, \\
& \quad \mathbf{x} \geq 0,
\end{aligned}
\qquad \text{Dual:} \quad
\begin{aligned}\mathbf{d}^* = \max_{\boldsymbol{\pi} \in \mathbb{R}^m} & \quad \boldsymbol{\pi}^\top \mathbf{b} \\
\text{s.t.} & \quad \boldsymbol{\pi}^\top \mathbf{A} \leq \mathbf{c}.
\end{aligned}
$$

- Weak duality: $\mathbf{c}^\top \mathbf{x} \geq \boldsymbol{\pi}^\top \mathbf{b}$ for any primal feasible solutions

- Strong duality: $\mathbf{z}^\star = \mathbf{c}^\top \mathbf{x}^\star = \boldsymbol{\pi}^{\star\top} \mathbf{b} = \mathbf{d}^\star$ for any primal and dual optimal solutions.

Plus, at the optimality, 
$$
\frac{\partial \mathbf{z}^\star}{\partial \mathbf{b}} = \boldsymbol{\pi}^\star,
$$
which means that the optimal dual variable $\boldsymbol{\pi}^\star$ can be interpreted as the sensitivity of the optimal value with respect to the right-hand side vector $\mathbf{b}$.