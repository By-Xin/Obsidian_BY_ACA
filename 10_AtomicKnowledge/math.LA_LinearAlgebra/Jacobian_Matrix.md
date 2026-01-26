---
aliases: [雅可比矩阵, Jacobian Matrix, Jacobian, 雅可比]
tags:
  - concept
  - math/linear-algebra
  - math/real-analysis
related_concepts:
  - "[[Determinant]]"
  - "[[Matrix]]"
  - "[[Implicit_Function_Theorem]]"
  - "[[Probability_Distribution_Change_of_Variables]]"
source: "多元微积分"
---

# 雅可比矩阵 (Jacobian Matrix)

#LinearAlgebra

Assume $z = (z_1, z_2, \dots, z_n)$, $x = (x_1, x_2, \dots, x_n)$, $f: \mathbb{R}^n \rightarrow \mathbb{R}^n, f(z) = x$

$$
J_f = \frac{\partial f}{\partial z} = \begin{bmatrix}
\frac{\partial f_1}{\partial z_1} & \frac{\partial f_1}{\partial z_2} & \dots & \frac{\partial f_1}{\partial z_n}\\
\frac{\partial f_2}{\partial z_1} & \frac{\partial f_2}{\partial z_2} & \dots & \frac{\partial f_2}{\partial z_n}\\
\vdots & \vdots & \ddots & \vdots\\
\frac{\partial f_n}{\partial z_1} & \frac{\partial f_n}{\partial z_2} & \dots & \frac{\partial f_n}{\partial z_n}\\
\end{bmatrix}
$$

$$
J_{f^{-1}} = \frac{\partial z}{\partial f} = \begin{bmatrix}
\frac{\partial z_1}{\partial f_1} & \frac{\partial z_1}{\partial f_2} & \dots & \frac{\partial z_1}{\partial f_n}\\
\frac{\partial z_2}{\partial f_1} & \frac{\partial z_2}{\partial f_2} & \dots & \frac{\partial z_2}{\partial f_n}\\
\vdots & \vdots & \ddots & \vdots\\
\frac{\partial z_n}{\partial f_1} & \frac{\partial z_n}{\partial f_2} & \dots & \frac{\partial z_n}{\partial f_n}\\
\end{bmatrix}
$$

We can proof that:
$$J_{f^{-1}} J_f = I$$

***E.g.***

$$
f(z) = \begin{bmatrix}
f_1(z_1, z_2)\\
f_2(z_1, z_2)
\end{bmatrix} = \begin{bmatrix}
z_1 + z_2\\
2z_1
\end{bmatrix}
$$

$$
J_f = \begin{bmatrix}
\frac{\partial f_1}{\partial z_1} & \frac{\partial f_1}{\partial z_2}\\
\frac{\partial f_2}{\partial z_1} & \frac{\partial f_2}{\partial z_2}
\end{bmatrix} = \begin{bmatrix}
1 & 1\\
2 & 0
\end{bmatrix}
$$
