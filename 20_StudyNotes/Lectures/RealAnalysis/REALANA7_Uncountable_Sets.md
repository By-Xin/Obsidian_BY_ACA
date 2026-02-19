#RealAnalysis

***Definition (Uncountable Set)***: A set $A$ is **uncountable** if $A$ is infinite and not countable.

*Question 1: Does such uncountable set exist?*

*Question 2: How big is the cardinality of uncountable set? Is there a maximal cardinality?*


***Theorem (Cantor's Theorem)***: Let $M$ be an arbitrary set. Define $\mu$ as the set of all subsets of $M$. Then $\text{Card}(\mu) > \text{Card}(M)$. Such $\mu$ is often denoted as $2^M$, or the **power set** of $M$.
- *Example 1*: Let $M = \emptyset$, then $\mu = \{\emptyset\}$, and $\text{Card}(\mu) = 1 > 0 = \text{Card}(M)$.
- *Example 2*: Let $M = \{1\}$, then $\mu = \{\emptyset, \{1\}\}$, and $\text{Card}(\mu) = 2 > 1 = \text{Card}(M)$.
- *Example 3*: Let $M = \{1,2\}$, then $\mu = \{\emptyset, \{1\}, \{2\}, \{1,2\}\}$, and $\text{Card}(\mu) = 4 > 2 = \text{Card}(M)$.
- *Example 4*: Let $M = \{1,2,\cdots,n\}$, then $\text{Card}(\mu) = 2^n > n = \text{Card}(M)$. ($\mu$ is just to go over all elements in $M$ and decide whether to include each element or not.) 
- *Application 1:* Consider **Russell's Paradox** (The collection of all sets is not a set). It can be proved by Cantor's Set by contradiction. If the collection of all sets is a set, denoted as $M$, then $\mu$ is the set of all subsets of $M$, i.e. $\mu \subset M \Rightarrow \text{Card}(\mu) \leq \text{Card}(M)$. However, by Cantor's Set theorem, $\text{Card}(\mu) > \text{Card}(M)$, which is a contradiction.
- *Proof*: By contradiction, assume $\text{Card}(\mu) \leq \text{Card}(M)$. However, we can always construct an injective mapping from $M$ to $\mu$ by $m\mapsto\{m\}$. Thus, $\text {Card}(\mu) \ge \text{Card}(M)$. Then we get $\text{Card}(\mu) = \text{Card}(M)$, which is not true. 
  - If $\text{Card}(\mu) = \text{Card}(M)$, then there exists a bijective mapping $f: M \to \mu$, and denotes the inverse mapping as $g:=f^{-1}$. Denotes $A = \{x \in M | x \notin f(x)\}\subset M$, then $A\in\mu$. If $g(A)\in A$, then $g(A)\notin f(g(A)) = A$, which is a contradiction. If $g(A)\notin A$, then $g(A)\in f(g(A)) = A$, which is also a contradiction. Thus, $\text{Card}(\mu) > \text{Card}(M)$.

> **Answer to Question 2**: By Cantor's Theorem, the cardinality of the power set of $M$ is strictly larger than the cardinality of $M$. Thus, there is no maximal cardinality.

> **Answer to Question 1**: Yes, such uncountable set exists. For example, consider $2^{\mathbb{N_+}}$, by Cantor's Theorem, $\text{Card}(2^{\mathbb{N_+}}) > \text{Card}(\mathbb{N_+})$. Thus, $2^{\mathbb{N_+}}$ is uncountable.

***Fact:*** $\mathbb{R} \sim 2^{\mathbb{N}}$.

***Theorem***: $\mathbb{R}$ is uncountable.
- *Proof*: 
  - By contradiction, assume $\mathbb{R}$ is countable. Since $\mathbb{R} \sim (0,1)$, then $(0,1)$ is countable. 
  - Then we can list all numbers in $(0,1)$ as $a^{(1)}, a^{(2)}, a^{(3)}, \cdots$. Then we try to represent such sequence by decimal expansion:
    $$\begin{aligned}
    a^{(1)} &= 0.a_{11}a_{12}a_{13}\cdots \\
    a^{(2)} &= 0.a_{21}a_{22}a_{23}\cdots \\
    a^{(3)} &= 0.a_{31}a_{32}a_{33}\cdots \\
    &\cdots \\
    a^{(n)} &= 0.a_{n1}a_{n2}a_{n3}\cdots\\
    &\cdots
    \end{aligned}$$
  - Then we can construct a new number $b$ that is different from all $a^{(n)} \cdots$, which means $b$ is not in the list (cannot be listed, is not countable). In specific, define $b=0.b_1b_2b_3\cdots$, where $b_i = 1$ if $a^{(i)}_{i}\neq 1$, and $b_i = 2$ if $a^{(i)}_{i} = 1$. **Q.E.D.**


***Definition (Cardinal Number of $\mathbb{R}$ and $\mathbb{Z^+})$***: Denote the cardinal number of $\mathbb{R}$ as $\mathfrak{c}$, and the cardinal number of $\mathbb{Z^+}$ as $\aleph_0$. Then $\aleph_0 < \mathfrak{c}$. Such $\mathfrak{c}$ is called the **continuum**.

***Assumption (Continuum Hypothesis)***: There is no set whose cardinality is strictly between that of the integers and the real numbers. In other words, there is no set $A$ such that $\aleph_0 < \text{Card}(A) < \mathfrak{c}$.

---

**Now consider the operations of continumm**:

***Fact***: Given any interval $(a,b)$, it is equivalent to $\mathbb{R}$.Thus, $\text{Card}((a,b)) = \mathfrak{c}$. 

***Theorem 1***: Assume $\{A_n: n\in\mathbb{Z}\}$ disjoint sets, with $\text{Card}(A_n) = \mathfrak{c}$, then $\text{Card}\bigcup_{n = 1}^\infty A_n = \mathfrak{c}$.

***Theorem 2***: Assume $\{A_n: n\in\mathbb{Z}\}$ with $\text{Card}(A_n) = \mathfrak{c}$, then $\prod_{n = 1}^\infty A_n = \mathfrak{c}$.
- *Sketch of Proof*: Obviously, we can construct an injective mapping from $\mathbb{R} \to \prod_{n = 1}^\infty A_n$ by $x\mapsto (f_1(x), f_2(x), \cdots)$. Then $\text{Card}(\prod_{n = 1}^\infty A_n) \geq \mathfrak{c}$. Moreover, construct another injective mapping from $\prod_{n = 1}^\infty A_n \to \mathbb{R}$ by $(a_1, a_2, \cdots)\mapsto a_1 + a_2 + \cdots$. Then $\text{Card}(\prod_{n = 1}^\infty A_n) \leq \mathfrak{c}$. Thus, $\text{Card}(\prod_{n = 1}^\infty A_n) = \mathfrak{c}$.

***Corollary 2.1***: Countable Cartesian product of at most countable sets may not be countable.

***Corollary 2.2***: $\text{Card}(\mathbb{R}^n) = \mathfrak{c}$. (As $\mathbb{R}^n$ is $n$-fold Cartesian product of $\mathbb{R}$.)

***Corollary 2.3***: $\text{Card}(\mathbb{C}) = \mathfrak{c}$. (As $\mathbb{C}$ is equivalent to $\mathbb{R}^2$.)

***Corollary 2.4***: $\mathfrak{c}$ union of $\mathfrak{c}$ sets is still $\mathfrak{c}$, i.e. $\bigcup_{x\in\mathbb{R}}\{A_x\} = \mathfrak{c}$, where $\text{Card}(A_x) = \mathfrak{c}$. 
- *Intuition:* As such each $A_x$ can be mapped to a line in $\mathbb{R}^2$ (imagine a line $x=t$ in a $xoy$ plane), and then $\bigcup_{x\in\mathbb{R}}\{A_x\}$ is equivalent to $\mathbb{R}^2$ (imagine for from $x=t$, we move from $t=-\infty$ to $t=\infty$ and get the whole $xoy$ plane)

---
***Example 1***: $\{x: |f(x)\neq 0\} = \bigcap_{\epsilon \in \mathbb{R^+}}\{x: |f(x)| < \epsilon\} = \bigcap_{n = 1}^\infty\{x: |f(x)| < \frac{1}{n}\}$. 

***Example 2***: $\{x: \lim_{n\to\infty}f_n(x) = 0\} = \bigcap_{\epsilon \in \mathbb{R^+}}\bigcup_{N = 1}^\infty\bigcap_{n = N}^\infty\{x: |f_n(x)| < \epsilon\}$ = $\bigcap_{k = 1}^\infty\bigcup_{N = 1}^\infty\bigcap_{n = N}^\infty\{x: |f_n(x)| < \frac{1}{k}\}$.
