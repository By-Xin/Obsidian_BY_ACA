#RealAnalysis

Recall that, for measurments, we hope that it should satisfy the following properties:
- Non-negativity: $m(E) \geq 0$ for all $E \in \mathcal{M}$.
- Countable additivity: If $E_1, E_2, \ldots \in \mathcal{M}$ are pairwise disjoint, then $m\left(\bigcup_{i=1}^{\infty} E_i\right) = \sum_{i=1}^{\infty} m(E_i)$.
- Regularity: For any $E \in \mathcal{M}$, we have $m(E) = \inf\{m(U) : E \subset U, U \text{ open}\}$.

Does it still hold for the outer measure? 

---

Recall its definition:
***Definition (Outer Measure):*** Define the outer measure of  a set $E \subset \mathbb{R}$ as: 
$$
m^*(E) = \inf\left\{\sum_{i=1}^{\infty} |I_i| : E \subset \bigcup_{i=1}^{\infty} I_i, I_i \text{ open interval}\right\}
$$
which tells us that:
1. For any open interval of $E$, say, $\{I_i\}_{i=1}^{\infty}$, we have $m^*(E) \leq \sum_{i=1}^{\infty} |I_i|$.
2. There always exists a sequence of open intervals $\{I_i\}_{i=1}^{\infty}$ such that $\forall \epsilon > 0$, $E \subset \bigcup_{i=1}^{\infty} I_i$ and $\sum_{i=1}^{\infty} |I_i| \leq m^*(E) + \epsilon$ (by the definition of infimum).

Then start from the definition of outer measure, here try to prove the following properties:
- **Non-negativity**: It's obvious that $\mu^*(E) \geq 0$ for all $E \subset \mathbb{R}$.
- **Sub-Countable subadditivity**: 
  - ***Definition (Subadditivity):*** For any $A_1, A_2, \ldots \subset \mathbb{R}$ and outer measure $m^*$, we have
  $$ m^*\left(\bigcup_{i=1}^{\infty} A_i\right) \leq \sum_{i=1}^{\infty} m^*(A_i) $$
    - *Intuition*:
      - Take an arbitrary open interval cover $\{I_{ij}\}_{j=1}^{\infty}$ for each $A_i$. Then $\bigcup_{i=1}^{\infty} A_i \subset \bigcup_{i=1}^{\infty} \bigcup_{j=1}^{\infty} I_{ij}$. And the RHS is countable union of open intervals, so $m^*(\bigcup_{i=1}^{\infty} A_i) \leq \sum_{i=1}^{\infty} \sum_{j=1}^{\infty} |I_{ij}|$ (by property 1 of outer measure).
      - As $\{I_{ij}\}_{j=1}^{\infty}$ is arbitrary, we can specify it to make $\sum_{i=1}^{\infty} \sum_{j=1}^{\infty} |I_{ij}| \leq \sum_{i=1}^{\infty} m^*(A_i) + \epsilon$ for any $\epsilon > 0$ (By letting each $m^*(A_i) \ge \sum_{j=1}^{\infty} |I_{ij}| - \frac{\epsilon}{2^i}$). 
      - Finally, we have $m^*(\bigcup_{i=1}^{\infty} A_i) \leq \sum_{i=1}^{\infty} m^*(A_i) + \epsilon$ for any $\epsilon > 0$ and conclude that $m^*(\bigcup_{i=1}^{\infty} A_i) \leq \sum_{i=1}^{\infty} m^*(A_i)$.
    - *Proof* $\forall \epsilon > 0$, there exists a sequence of open intervals $A_i \subset \{I_{ij}\}_{j=1}^{\infty}$ such that $\sum_{j=1}^{\infty} |I_{ij}| \leq m^*(A_i) + \frac{\epsilon}{2^i}$, and $\bigcup_{i=1}^{\infty} A_i \subset \bigcup_{i=1}^{\infty} \bigcup_{j=1}^{\infty} I_{ij}$.
      - Then   $m^*(\bigcup_{i=1}^{\infty} A_i) \leq \sum_{i=1}^{\infty} \sum_{j=1}^{\infty} |I_{ij}| \leq \sum_{i=1}^{\infty} m^*(A_i) + \epsilon$.
      - As $\epsilon$ is arbitrary, we have $m^*(\bigcup_{i=1}^{\infty} A_i) \leq \sum_{i=1}^{\infty} m^*(A_i)$.
- **Regrularity** ($m^*([0,1]) = 1$ / $m^*(I) = |I|$ 
  - *Proof*
    - First, assume $I$ is a closed interval. Then $\forall \epsilon > 0$, there always exists an open interval $I \subset I'$ such that $|I'| \leq |I| + \epsilon$. Then $m^*(I) \leq m^*(I') \leq |I| + \epsilon$. As $\epsilon$ is arbitrary, we have $m^*(I) = |I|$. On the other hand, $\forall \epsilon > 0$, there always exists an open interval $\{I_i\}_{i=1}^{\infty}$ of $I$ such that $m^*(I) \geq \sum_{i=1}^{\infty} |I_i| - \epsilon$. Then there exists a limit cover $\{I_i\}_{i=1}^{m}$ such that  $m^*(I) \geq \sum_{i=1}^{m} |I_i| - \epsilon \ge |I|-\epsilon$. As $\epsilon$ is arbitrary, we have $m^*(I) \geq |I|$. Thus, $m^*(I) = |I|$.
    - More generally, for any interval $I$, we can always find two closed intervals $I_1, I_2$ such that $I_1 \subset I \subset I_2$ and $|I_1| + \epsilon > |I| , |I_2| - \epsilon < |I|$. We have $m^*(I_1) = |I_1| $ and $m^*(I_2) = |I_2|$. Then $m^*(I_1) \leq m^*(I) \leq m^*(I_2)$, which implies $|I| - \epsilon < m^*(I) \leq m^*(I) \leq m^*(I_2) < |I| + \epsilon$. As $\epsilon$ is arbitrary, we have $m^*(I) = |I|$.

***Theorem (Properties of Monothonic):***  If $A \subset B$, then $m^*(A) \leq m^*(B)$.
- *Proof*: 
  - For an arbitrary open interval cover $\{I_i\}_{i=1}^{\infty}$ of $B$, it must also cover $A$. Then $m^*(A) \leq \sum_{i=1}^{\infty} |I_i| \leq m^*(B)$. The last inequality is due to $m^*(B)$ is the infimum of all open interval covers of $B$.