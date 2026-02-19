---
aliases: [列空间, Column Space, 矩阵列空间, Image Space, 像空间]
tags:
  - concept
  - math/linear-algebra
related_concepts:
  - "[[Matrix]]"
  - "[[Linear_Space]]"
  - "[[Rank_and_Eigenvalue]]"
source: "MIT 18.065"
---

# 矩阵的列空间 (Column Space of Matrix)

#MIT18065 #Matrix

Let's begin by a matrix multiplied by a vector:
$$
\begin{bmatrix}
2 & 1 & 3 \\
3 & 1 & 4 \\
5 & 7 & 12\\
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
\end{bmatrix}
:= A\mathbf{x}
$$

There are actually two ways to think of this multiplication:

- **Dot product**: this is the most common way to think of matrix multiplication:
    $$
    \begin{bmatrix}
    2x_1 + x_2 + 3x_3\\
    3x_1 + x_2 + 4x_3\\
    5x_1 + 7x_2 + 12x_3\\
    \end{bmatrix}
    $$
- **Column linear combination**: we can also regard this multiplication as a linear combination of the columns of the matrix $A$ weighted by the components of the vector $\mathbf{x}$:
    $$
    x_1\begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix} + x_2\begin{bmatrix} 1 \\ 1 \\ 7 \end{bmatrix} + x_3\begin{bmatrix} 3 \\ 4 \\ 12 \end{bmatrix}
    $$ 
    and this is a recommended way as it shows the column space.

---

If we expand this example and consider all possible vectors $\mathbf{x}$, we get the **column space** of the matrix $A$.

In this example, the column space of $A = \text{span}\left\{ \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix}, \begin{bmatrix} 1 \\ 1 \\ 7 \end{bmatrix}, \begin{bmatrix} 3 \\ 4 \\ 12 \end{bmatrix} \right\}$. But it is a 2-dimensional subspace of $\mathbb{R}^3$, as the third vector is a linear combination of the first two. In other words, $\text{rank}(A) = 2$. From this, we can see that **rank** is the dimension of the column space (or the number of independent columns). And the two independent vectors are the **basis** of the column space.

---

If we denote the column space of $A$ as $C(A)$, then in this example:
$$
C(A) = \begin{bmatrix} 2 & 1 \\ 3 & 1 \\ 5 & 7 \end{bmatrix}
$$

And then we may try to represent the original matrix $A$ by the product of two matrices:
$$\begin{aligned}
A &= C(A) R(A) \\
&= \begin{bmatrix} 2 & 1 \\ 3 & 1 \\ 5 & 7 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1  \\ 0 & 1 & 1 \end{bmatrix} \\
&= \begin{bmatrix} 2 & 1 & 3 \\ 3 & 1 & 4 \\ 5 & 7 & 12 \end{bmatrix}
\end{aligned}$$

  - A way to calculate the matrix $R(A)$ is also to regard it as a linear combination of the columns of $C(A)$. Denote $R(A) = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ r_{21} & r_{22} & r_{23} \end{bmatrix}$, then we have:
    $$
    \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix} = r_{11}\begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix} + r_{21}\begin{bmatrix} 1 \\ 1 \\ 7 \end{bmatrix}
    $$

   - In other words, in the form $A = C(A) R(A)$, the first column of $A$ is equal to the **linear combinition of all columns of $C(A)$** weighted by the first column of $R(A)$:
        $$\begin{aligned}
        \begin{bmatrix} \cdots & \alpha_k & \cdots \end{bmatrix} &= \begin{bmatrix} c_1 & c_2  \end{bmatrix} \begin{bmatrix} \cdots r_{k1} \cdots  \\ \cdots r_{k2} \cdots \end{bmatrix} = c_1 r_{k1} + c_2 r_{k2}
        \end{aligned}$$

  - We also call this space as the **row space** of the matrix $A$, and denote it as $R(A)$. BUT, it can also be regarded as:$R(A) = C(A^T)$.  

And we've define the dimension of the row space as the **row rank**. And it clearly shows that: **Row rank is equal to Column rank!**


---

Finally, consider the multiplication of two matrices $A$ and $B$:
$$
AB = \text{Col}_1(A) \text{Row}_1(B) + \text{Col}_2(A) \text{Row}_2(B) + \cdots + \text{Col}_n(A) \text{Row}_n(B)
$$

Given a $A\in\mathbb{R}^{m\times n}, B\in\mathbb{R}^{n\times p}$, the time complexity of this multiplication is $\mathcal{O}(mnp)$.

