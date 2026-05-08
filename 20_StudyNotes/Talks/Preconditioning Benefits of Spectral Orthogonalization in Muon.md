# Preconditioning Benefits of Spectral Orthogonalization in Muon 

Speaker: MA Jianhao

## Introduction to Muon

For $t = 0, 1, \cdots$:

$$
\begin{aligned}
B_t &= \nabla f(\theta_t) + \mu B_{t-1}
&& \text{(gradient + momentum)}, \\
\theta_{t+1} &= \theta_t - \eta_t \operatorname{msign}(B_t)
&& \text{(spectral orthogonalization)}.
\end{aligned}
$$

The matrix sign function is defined as

$$
\operatorname{msign}(Z) := UV^\top,
\quad \text{if } Z = U\Sigma V^\top \text{ is the SVD of } Z.
$$
