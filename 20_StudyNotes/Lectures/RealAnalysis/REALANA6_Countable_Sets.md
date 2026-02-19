#RealAnlaysis

***Definition (Countable Set)***: A set $A$ is **countable** if $A\sim\mathbb{N_+}$.
- *Example*: $\mathbb{N}, \mathbb{Z}, \{\text{Odd integers}\}, \cdots$ are all countable sets.

***Theorem 0***: All countable sets are infinite sets.
- *Proof*: By definition, assume $A$ is countable and finite, then $A\sim\{1,2,\ldots,n\}$ for some $n\in\mathbb{N}$. However, $\{1,2,\ldots,n\}\subset\mathbb{N_+}$, then $A\sim\{1,2,\ldots,n\}\sim\mathbb{N_+}$, which contradicts the assumption that $A$ is finite.

- *Example*: $\{1,2,3,\cdots,n\}$ is not equivalent to $\mathbb\{1,2,3,\cdots,m\}$ for $n\neq m$.

***Theorem 1***: Any infinite set at least contains a countable subset. *(i.e. countable sets are the 'smallest' infinite sets)*
- *Intuition:* For a infinite set $M$, we can always find an element $m_1\in M$. Then for the remaining set $M_1 = M-\{m_1\}$, it is still infinite. Continue this process, we can construct a sequence $m_1, m_2, m_3, \cdots$.
- *Proof*: By induction, first, as $M$ is infinite, we can always find an arbitrary element $m_1\in M$. Then assume we have found $m_1, m_2, \cdots, m_{n-1}$ which are all distinct. As $M$ is infinite, $M_n := M-\{m_1, m_2, \cdots, m_{n-1}\}\neq\emptyset$. Then we can find $m_n\in M_n$, and $m_n\notin\{m_1, m_2, \cdots, m_{n-1}\}$. Thus we can find a subset $M' = \{m_1, m_2, \cdots m_n, \cdots\}$ which is countable.

***Theorem 2***: Any infinite set of a countable set is still countable.
- *Intuition*: Let $S\subset M$ be the infinite subset of a countable set $M$. Then by Theorem 1, $\exists S_1 \subset S$ which is countable. Thus, $S_1 \subset S \subset M$, and $S_1$ is countable. By **Bernstein's Theorem**, $\overline{\overline{M}} = \overline{\overline{S}}$.
- *Corollary*: Any subset of a countable set is either finite or countable, which is called **at most countable**. **At most countable** set is equivalent to the subset of $\mathbb{Z_+}$.

***Theorem 3***: At most countable union of countable sets is still countable.
- *Example*: Obviously, finite union of finite sets is finite.
- *Proof*: **Here, we only give the proof of disjoint union.**
  - Denote 
  $$\begin{aligned}
    A_1 &= \{a_{11}, a_{12}, a_{13}, a_{14}, \cdots\} \\
    A_2 &= \{a_{21}, a_{22}, a_{23}, a_{24}, \cdots\} \\
    A_3 &= \{a_{31}, a_{32}, a_{33}, a_{34}, \cdots\} \\
    &\cdots
    \end{aligned}$$
  - We may count the union of $A_1, A_2, A_3, \cdots$ in the following way:
    - $a_{11}, a_{12}, a_{21}, a_{13}, a_{22}, a_{31}, a_{14}, a_{23}, a_{32}, a_{41}, \cdots$

***Corollary 3.1***: The set of all rational numbers $\mathbb{Q}$ is countable.
- *Proof*: Construct $A_i = \{\frac1i, \frac2i, \frac3i, \cdots\}$, which is countable. Then $\mathbb{Q^+} = \bigcup_{i=1}^\infty A_i$ is the countable union of countable sets, which is countable. Then $\mathbb{Q^-}\sim\mathbb{Q^+}$ is also countable. Then $\mathbb{Q} = \mathbb{Q^+}\cup\mathbb{Q^-}\cup\{0\}$ is countable. Thus, $\overline{\overline{\mathbb{Q}}} \leq \overline{\overline{\mathbb{Z_+}}}$. Moreover, $Z_+ \subset Q$, then $\overline{\overline{\mathbb{Z_+}}} \leq \overline{\overline{\mathbb{Q}}}$. By **Bernstein's Theorem**, $\overline{\overline{\mathbb{Q}}} = \overline{\overline{\mathbb{Z_+}}}$.

***Theorem 4***: **Finite** Cartesian product of countable sets is still countable.
- *Proof* by induction:
  - For $n=1$, $A_1$ is countable.
  - Assume for $n-1$ at most countable sets $A_1, A_2, \cdots, A_{n-1}$, their Cartesian product is countable.
  - Then for $n$: $A_1, A_2, \cdots, A_{n-1}, A_n$ are all at most countable. Denote $A_n = \{a_{n1}, a_{n2}, a_{n3}, \cdots\}$. By hypothesis, $A_1\times A_2\times\cdots\times A_{n-1}$ is countable. Then $\tilde A_k := A_1\times A_2\times\cdots\times A_{n-1}\times \{a_{nk}\} \sim A_1\times A_2\times\cdots\times A_{n-1}$, which is at most countable. 
    - Whilst, $A_1\times \cdots\times A_n = \bigcup_{k=1}^\infty \tilde A_k$, is a countable union of at most countable sets, which is countable.  

***Corollary 4.1***: The set of all algebraic numbers is countable.
  - **Algebraic numbers**: The roots of a non-zero polynomial with integer coefficients. If a number is not algebraic, it is called **transcendental**.
    - *Example*: $\sqrt{2}, \sqrt{-1}, \cdots$ are algebraic numbers.$\pi, e$ are transcendental numbers.  
  - *Proof*: Denote $A_n = \{\text{Integer Coefficient Polynomials of degree } n\} = \{f: f(x) = a_nx^n + a_{n-1}x^{n-1} + \cdots + a_0; a_i\in\mathbb{Z}, a_n\neq 0\}$. Then, $A_n \subset \mathbb{Z}\times\mathbb{Z}\times\cdots\times\mathbb{Z}$ (as $f\mapsto (a_n, a_{n-1}, \cdots, a_0)$). Then $A_n$ is the finite Cartesian product of countable sets, which is at most countable. Assume $B_n$ is the set of all roots of polynomials in $A_n$, i.e. $A_n \times \{1,2,\cdots,n\} \to B_n$ (in other words, $(f,k)\mapsto \text{k'th root of }f$), which is a *on-to* mapping. Thus, $\overline{\overline{B_n}} \leq \overline{\overline{A_n\times\{1,2,\cdots,n\}}}$. Then $B_n$ is at most countable. Then the set of all algebraic numbers is the countable union of countable sets, which is at most countable. Whilest, $\mathbb{Z_+}\subset\{\text {All Algebraic Numbers}\}$. Q.E.D.