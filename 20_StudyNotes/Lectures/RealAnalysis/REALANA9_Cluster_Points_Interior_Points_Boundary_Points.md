#RealAnalysis

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202501071542914.png)

First consider the points intuitively. Given a set $S$, we can classify the points into three categories:

- **Interior Points**: The neighborhood of the point is always in $S$.
- **Boundary Points**: The neighborhood of the point contains both points in $S$ and points not in $S$.
- **Exterior Points**: The neighborhood of the point is always not in $S$.

Try to describe these points in a more formal way:
- ***Definition: (Interior Points)***: There exists a $\delta$-neighborhood $U_\delta(P_0)$ such that $U_\delta(P_0)\subset S$.
- ***Definition: Exterior Points***: There exists a $\delta$-neighborhood $U_\delta(P_0)$ such that $U_\delta(P_0)\cap S = \emptyset$.
- ***Definition: Boundary Points***: For all $\delta>0$, $U_\delta(P_0)\cap S \neq \emptyset$ and $U_\delta(P_0)\cap S^c \neq \emptyset$.

It is easy to conclude that: if $P_0$ is an interior point of $S$, then $P_0$ is the exterior point of $S^c$. 

---

Further define:
- ***Definition: (Open Kernel)***: The set of all interior points of $S$ is called the **open kernel** of $S$, denoted as $S^\circ$.
- ***Definition: (Boundary)***: The set of all boundary points of $S$ is called the **boundary** of $S$, denoted as $\partial S$.

---

Another way to consider the points:
- ***Definition: (Cluster Points)***: A point $P_0$ is a **cluster point** of $S$ if for all $\delta>0$, $U_\delta(P_0)\cap S\neq \emptyset$. (For any neighborhood of $P_0$, there are always points different from $P_0$ and in $S$.)
- ***Definition: (Isolated Points)***: A point $P_0$ is an **isolated point** of $S$ if there exists a $\delta>0$ such that $U_\delta(P_0)\cap S = \{P_0\}$. (There is a neighborhood of $P_0$ that only contains $P_0$.)

Then claim that: for a boundary point $P_0$, it can be either a cluster point or an isolated point.

---

Claim that, the following three statements are equivalent in a metric space $(X,d)$:
1. $P_0$ is a cluster point of $S$.
2. For any $\delta>0$, $U_\delta(P_0)$ contains infinitely many points in $S$.
3. There exists a distinct sequence $\{P_n\}$ in $S$ such that $\lim_{n\to\infty} P_n = P_0$.

*Proof*:

- $(1)\Rightarrow(2)$: By contradiction, given $P_0$ is a cluster point of $S$, assume there exists a $\delta>0$ such that $U_\delta(P_0)$ contains only finitely many points in $S$, denoting these points: $U_\delta(P_0)\cap S = \{P_1, P_2, \cdots, P_K\}$. As $P_0$ is a cluster point, then $\{P_1, P_2, \cdots, P_K\} - \{P_0\} \neq \emptyset$. W.L.O.G., assume all these points are different from $P_0$. Set a sufficiently small $\epsilon = \frac12 \min\{\delta, d(P_1,P_0), d(P_2,P_0), \cdots, d(P_K,P_0)\}$. Then $\forall P\in U_\epsilon(P_0)-\{P_0\}, P\notin \{P_0, P_1, P_2, \cdots, P_K\}$. This implies that $U_\epsilon(P_0)\cap S = \emptyset$, which contradicts the assumption that $P_0$ is a cluster point of $S$.
- $(2)\Rightarrow(3)$: By induction, construct a sequence $\{P_n\}$ as follows: Let $P_1$ be any point in $U_1(P_0)\cap S$ that is different from $P_0$ (as $U_1(P_0)$ contains infinitely many points in $S$). Assume we have constructed $P_1, P_2, \cdots, P_n$ such that $P_i\in (U_{\frac1i}(P_0) - \{P_0\})$, and $P_i\neq P_j$ for $i\neq j$. Then let $\delta = \frac12 \min \{\frac1{n+1}, d(P_1,P_0), d(P_2,P_0), \cdots, d(P_n,P_0)\}$, and choose $P_{n+1}\in (U_\delta(P_0)-\{P_0\})\cap S$. Then $\{P_n\}$ is a sequence in $S$ that converges to $P_0$: for any $\epsilon>0$, there exists $N\in\mathbb{N}$ such that $N>\frac1\epsilon$, then $\forall n>N$, $d(P_n,P_0)<\frac1n<\epsilon$, which implies $\lim_{n\to\infty} P_n = P_0$.
- $(3)\Rightarrow(1)$: For any $\delta>0$, as $\lim_{n\to\infty} P_n = P_0$, there exists $N\in\mathbb{N}$ such that $\forall n>N$, $d(P_n,P_0)<\delta$. Since $\{P_n\}$ is distinct, then $\exist m>N$ such that $P_m\neq P_0$. Moreover, as $P_m\in (U_\delta(P_0)\cap S - \{P_0\})$, then $U_\delta(P_0)\cap S\neq \emptyset$. 

---

There are another two important concepts:
- ***Definition (Derived Set)***: Denote the set of all cluster points of $S$ as $S'$. Then $S'$ is called the **derived set** of $S$, denoted as $S'$.
- ***Definition(Closure)***: Define $S\cup S'$ as the **closure** of $S$, denoted as $\overline{S}$.
    - $\bar S$ is the closed region that sequence in $S$ can never 'escape' from.
Remark:
- $\bar S = S\cup \partial S = S^\circ \cup \partial S = S' \cup \{\text{isolated points of } S\}$.

***Propertiess***:
- $(E^\circ)^c = \overline{E^c}$.
- $(\overline{E})^c = (E^c)^\circ$.
- $(A\cup B)' = A'\cup B'$.
- Assume $S\notin\emptyset, S\neq \mathbb{R}^n$, then $\partial S \neq \emptyset$.
  - *Proof*: With the help of the connectedness of $\mathbb{R}^n$.

***Theorem (Bolzano-Weierstrass)***: Let $E\subset \mathbb{R}^n$ be a bounded, infinite set. Then $E$ has at least one cluster point.
- *Sketch of Proof*: $n=1$ is trivial. For $n=2$, divide the region into grids with width $\delta$. First, $\delta_1 = 1$, then there must be a grid with infinitely many points. Then divide this grid into $4$ sub-grids with width $\delta_2 = \frac12$. There must be a sub-grid with infinitely many points. Continue this process, the grid size will converge to $0$, and the intersection of all these grids will contain a cluster point. For $n>2$, use the same idea.