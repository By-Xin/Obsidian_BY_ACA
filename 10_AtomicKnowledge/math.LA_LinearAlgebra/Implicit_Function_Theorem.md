---
aliases: [隐函数定理, Implicit Function Theorem, IFT]
tags:
  - concept
  - math/real-analysis
  - math/optimization
related_concepts:
  - "[[Jacobian_Matrix]]"
  - "[[Determinant]]"
  - "[[Matrix]]"
  - "[[Lagrange_Duality]]"
source: "多元微积分; 优化理论"
---

# Implicit Function Theorem

设 $\mathbf{F}: U \to \mathbb{R}^m$ 是一个从 $\mathbb{R}^{n+m}$ 中开集 $U$ 到 $\mathbb{R}^m$ 的函数，其中
$\mathbf{F}(\mathbf{x}, \mathbf{y}) = (F_1(\mathbf{x}, \mathbf{y}), \dots, F_m(\mathbf{x}, \mathbf{y}))^T$。

- $\mathbf{x} = (x_1, \dots, x_n)^\top \in \mathbb{R}^n$：自变量
- $\mathbf{y} = (y_1, \dots, y_m)^\top \in \mathbb{R}^m$：因变量

## Assumptions

1. $\mathbf{F}$ 在点 $(\mathbf{x}_0, \mathbf{y}_0) \in U$ 的某个邻域内 **连续可微**（$\mathbf{F} \in C^1(U)$）。
2. $\mathbf{F}(\mathbf{x}_0, \mathbf{y}_0) = \mathbf{0}$。
3. $\mathbf{F}$ 关于 $\mathbf{y}$ 的偏导数矩阵（[[Jacobian_Matrix]]）在 $(\mathbf{x}_0, \mathbf{y}_0)$ 处 **可逆**。记
   $$
   J_{\mathbf{y}}\mathbf{F}(\mathbf{x}, \mathbf{y})
   = \frac{\partial \mathbf{F}}{\partial \mathbf{y}}(\mathbf{x}, \mathbf{y})
   =
   \begin{pmatrix}
   \frac{\partial F_1}{\partial y_1} & \cdots & \frac{\partial F_1}{\partial y_m} \\
   \vdots & \ddots & \vdots \\
   \frac{\partial F_m}{\partial y_1} & \cdots & \frac{\partial F_m}{\partial y_m}
   \end{pmatrix}.
   $$
   因而 $\det(J_{\mathbf{y}}\mathbf{F}(\mathbf{x}_0, \mathbf{y}_0)) \neq 0$。

## Conclusion

存在 $\mathbf{x}_0$ 的开邻域 $U_0 \subseteq \mathbb{R}^n$ 和 $\mathbf{y}_0$ 的开邻域 $V_0 \subseteq \mathbb{R}^m$，使得对任意 $\mathbf{x} \in U_0$，存在唯一 $\mathbf{y} \in V_0$ 满足 $\mathbf{F}(\mathbf{x}, \mathbf{y}) = \mathbf{0}$。

定义 $\mathbf{h}: U_0 \to V_0$ 使得 $\mathbf{y} = \mathbf{h}(\mathbf{x})$，则：

1. $\mathbf{h}$ 连续可微（$\mathbf{h}\in C^1(U_0)$）。
2. 对所有 $\mathbf{x}\in U_0$，$\mathbf{F}(\mathbf{x}, \mathbf{h}(\mathbf{x}))=\mathbf{0}$。
3. $\mathbf{h}$ 的雅可比矩阵满足：
   $$
   J_{\mathbf{x}}\mathbf{h}(\mathbf{x})
   = \frac{\partial \mathbf{h}}{\partial \mathbf{x}}(\mathbf{x})
   = - \left( \frac{\partial \mathbf{F}}{\partial \mathbf{y}}(\mathbf{x}, \mathbf{h}(\mathbf{x})) \right)^{-1}
   \frac{\partial \mathbf{F}}{\partial \mathbf{x}}(\mathbf{x}, \mathbf{h}(\mathbf{x})).
   $$

其中
$$
\frac{\partial \mathbf{F}}{\partial \mathbf{x}}(\mathbf{x}, \mathbf{y})
=
\begin{pmatrix}
\frac{\partial F_1}{\partial x_1} & \cdots & \frac{\partial F_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial F_m}{\partial x_1} & \cdots & \frac{\partial F_m}{\partial x_n}
\end{pmatrix}.
$$
