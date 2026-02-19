#RealAnalysis

***Definition (Metric Space)***: A **metric space** $(X,d)$ is a set $X$ equipped with a **metric** $d: X\times X \to \mathbb{R}$, which satisfies the following properties for all $x,y,z\in X$:
1. **Non-negativity**: $d(x,y)\geq 0$. $d(x,y) = 0$ if and only if $x=y$.
2. **Symmetry**: $d(x,y) = d(y,x)$.
3. **Triangle Inequality**: $d(x,y) + d(y,z) \geq d(x,z)$.


Here are some examples of metric spaces that we are already familiar with:
- *Example 1*: $\mathbb{R}$ is a metric space with $d(x,y) = |x-y|$.
  
- *Example 2*: $\mathbb{R}^n$ is a metric space with $d(x,y) = \sqrt{\sum_{i=1}^n (x_i-y_i)^2}$. 
  - The triangle inequality is equivalent to the **Cauchy-Schwarz Inequality**.

There are other ways to define metric, such as:
- $d_1(x,y) = \sum_{i=1}^n |x_i-y_i|$ is called the **Manhattan metric**.
- $d_\infty(x,y) = \max_{i=1}^n |x_i-y_i|$ is called the **maximum metric**.

---

***Definition (Euclidean Space)***: An **Euclidean space** $\mathbb{R}^n$ is a metric space with the Euclidean metric $d(x,y) = \sqrt{\sum_{i=1}^n (x_i-y_i)^2}$.


***Definition (Subspace)***: Given a metric space $(X,d)$, a subset $Y\notin\emptyset, Y \subset X$ is called a **subspace** of $X$ if the metric $d$ is restricted to $Y$.

***Definition (Distance between subspaces)***: Given two subspaces $Y_1, Y_2$ of a metric space $X$, the **distance** between $Y_1$ and $Y_2$ is defined as:
$$d(Y_1,Y_2) = \inf_{y_1\in Y_1, y_2\in Y_2} d(y_1,y_2).$$

***Definition (Diameter of a set)***: Given a subset $A$ of a metric space $X$, the **diameter** of $A$ is defined as:
$$\text{diam}(A) = \sup_{x,y\in A} d(x,y).$$

***Definition (Bounded set)***: A subset $A$ of a metric space $X$ is called **bounded** if $\text{diam}(A) < \infty$.

- ***Claim***: In $\mathbb{R}^n$, a subset $A$ is bounded as long as its distance to the origin is bounded, i.e. $\exists M>0$ such that $d(x,0) < M$ for all $x\in A$.

---

***Definition ($\delta$-neighborhood)***: Given a center $P_0$ and a radius $\delta>0$, the **$\delta$-neighborhood** of $P_0$ is defined as:
$$U_\delta(P_0) = \{P\in X | d(P,P_0) < \delta\}.$$

- ***Property 1***: $P\in U_\delta(P)$
- ***Property 2***: $\forall \delta_1, \delta_2>0$, there always exists $\delta_3>0$ such that $U_{\delta_3}(P) \subset \left(U_{\delta_1}(P) \cap U_{\delta_2}(P)\right)$.
- ***Property 3***: $\forall Q \in U_\delta(P)$, there exists $U_\epsilon(Q)$ such that $U_\epsilon(Q) \subset U_\delta(P)$.
- ***Property 4***: If $P\neq Q$, then there must exist $U_\delta(P) \cap U_\epsilon(Q) = \emptyset$.

---

With the help of these definitions, we can now redefine the concept of **convergence** in $\mathbb{R}^n$.

- Traditionally, we say a sequence $\{x_n\}$ in $\mathbb{R}^n$ converges to $x_0\in\mathbb{R}^n$ if for any $\epsilon>0$, there exists $N\in\mathbb{N}$ such that $\forall n>N$, $|x_n-x_0|<\epsilon$.

- It can be rephrased as: $\lim_{n\to\infty} d(x_n,x_0) = 0$.


Then we can generalize the concept of convergence to metric spaces:

***Definition (Convergence in Metric Space)***: A sequence $\{P_n\}$ in a metric space $(X,d)$ converges to $P_0\in X$ if for $\lim_{n\to\infty} d(P_n,P_0) = 0$.
- Or equivalently, $\lim_{n\to\infty} P_n = P_0$ i.i.f.  for all $\delta$-neighborhood $U_\delta(P_0)$, there exists $N\in\mathbb{N}$ such that $\forall n>N$, $P_n\in U_\delta(P_0)$.