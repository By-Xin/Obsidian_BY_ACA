#RealAnalysis

- What is a *Set*?
  - *Russell's Paradox*: The collection of all sets is NOT a set.
    - *[Proof]* By contradiction: Assume the collection of all sets is a set, denoted by $S$. 
      - Consider a subset of $S$: $S_1 = \{A \in S : A \notin A\}$.
      - Since $S$ is the set of all sets, $S_1 \in S$.
        - If $S_1 \in S_1$, then $S_1 \notin S_1$ (by definition of $S_1$). Contradiction.
        - If $S_1 \notin S_1$, then $S_1 \in S_1$ (by definition of $S_1$). Contradiction.


- Operations of two sets $A$ and $B$:
  - **Union**: $A \cup B = \{x : x \in A \text{ or } x \in B\}$.
  - **Intersection**: $A \cap B = \{x : x \in A \text{ and } x \in B\}$.
  - **Difference**: $A - B = \{x : x \in A \text{ and } x \notin B\}$.
    - **Complement**: $B^c = A - B$, if $B \subseteq A$.
  - **Cartesian product**: $A \times B = \{(a, b) : a \in A \text{ and } b \in B\}$.

- Index set $\Lambda$: A set that indexes a collection of sets.
  - e.g. $\{A_n\}_{n \in \mathbb{N}}$ is a collection of sets indexed by $\mathbb{N}$: $\{A_1, A_2, \ldots\}$.
  - e.g. $\{A_\lambda\}_{\lambda \in \mathbb{R}}$ is a collection of sets indexed by $\mathbb{R}$, we have 'as many sets as there are real numbers'.

- Operations for arbitrary many sets $\{A_\lambda\}_{\lambda \in \Lambda}$:
  - **Union**: $\bigcup_{\lambda \in \Lambda} A_\lambda = \{x :\exists \lambda \in \Lambda \text{ , } x \in A_\lambda\}$.
  - **Intersection**: $\bigcap_{\lambda \in \Lambda} A_\lambda = \{x : \forall \lambda \in \Lambda \text{ , } x \in A_\lambda\}$.
  - **Cartesian product** (only consider finite or countable index set $\Lambda$)
    -  Finite case: $A_1 \times A_2 \times \cdots \times A_n = \{(a_1, a_2, \ldots, a_n) : a_i \in A_i \text{ for } i = 1, 2, \ldots, n\}$.
    -  Countable case: $\prod_{i=1}^\infty A_i = \{(a_1, a_2, \ldots) : a_i \in A_i \text{ for } i = 1, 2, \ldots\}$.
  
- Laws of Operations:
  - $A \cap (\bigcup_{\lambda \in \Lambda} B_\lambda) = \bigcup_{\lambda \in \Lambda} (A \cap B_\lambda)$.
  - De Morgan's Laws:
    - $(\bigcup_{\lambda \in \Lambda} A_\lambda)^c = \bigcap_{\lambda \in \Lambda} A_\lambda^c$.
    - $(\bigcap_{\lambda \in \Lambda} A_\lambda)^c = \bigcup_{\lambda \in \Lambda} A_\lambda^c$.
      - *[Proof]* 
        - First, prove $\forall x \in (\bigcap_{\lambda \in \Lambda} A_\lambda)^c \Rightarrow x \in \bigcup_{\lambda \in \Lambda} A_\lambda^c$.
          - It means $x \notin \bigcap_{\lambda \in \Lambda} A_\lambda$, i.e. at least $\exists \lambda_0 \in \Lambda$ such that $x \notin A_{\lambda_0}$. Then, $x \in A_{\lambda_0}^c$. Therefore, $x \in \bigcup_{\lambda \in \Lambda} A_\lambda^c$. $LHS \subseteq RHS$.
        - Then, prove $\forall x \in \bigcup_{\lambda \in \Lambda} A_\lambda^c \Rightarrow x \in (\bigcap_{\lambda \in \Lambda} A_\lambda)^c$.
          - It means $\exists \lambda_0 \in \Lambda$ such that $x \in A_{\lambda_0}^c$. Then, $x \notin A_{\lambda_0}$. Therefore, $x \notin \bigcap_{\lambda \in \Lambda} A_\lambda$, i.e. $x \in (\bigcap_{\lambda \in \Lambda} A_\lambda)^c$. $RHS \subseteq LHS$.