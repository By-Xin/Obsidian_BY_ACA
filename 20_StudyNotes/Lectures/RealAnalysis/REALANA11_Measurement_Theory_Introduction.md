#RealAnalysis

An intuitive question to ask is: how to define the length, area, and volume? In general, we have the following requirements for the natural measure of a set:
- Non-negativity: the measure of a set is non-negative.
- Finite additivity: the measure of the union of two disjoint sets is the sum of the measures of the two sets.
- Normalization: the measure of the unit interval is 1.

We want to extend the definition of the length of an interval to the length of a set, which gives the Lebesgue measure.

---

Another question is: how do we do the measurement? 
- By convention, we will get the lower bound and the upper bound of  a set, and approximate the measure by the "maximum of the lower bounds" and the "minimum of the upper bounds", or more formally, the infimum and supremum of the sum of the lengths of the covering intervals.
- Here, we define the **outer measure** of a set E as the infimum of the sum of the lengths of the covering intervals; and the **inner measure** of a set E as the supremum of the sum of the lengths of the covering intervals.
- If the outer measure equals the inner measure, we say the set is measurable, and the common value is the measure of the set.
- Yet we hope to further simplify the definition of the measure, and only consider the **outer measure** to define the measure of a set.


The last question is: is unmeasurable set exist? Is there any example of such a set?

---

***Definition (Outer Measure):*** The outer measure of a set $E \subset \mathbb{R}^n$ is defined as: $m^*(E) := \inf \{ \sum_{i=1}^{\infty} |I_i| : E \subset \bigcup_{i=1}^{\infty} I_i \}$, where $I_i$ are open intervals, and $|I_i|$ represents the volume of the interval $I_i$.
- We need **countable** covering intervals, rather than finite, as if we allow finite covering intervals, then the ends of the covering intervals must cover $[0,1]$, which makes the measure of rational number $ \ge 1$, meanwhile the outer measure of irrational number $\ge 1$, which is a contradiction.
- Yet, we allow some $I_i$ to be empty.

*Examples*:
- $m^*(\emptyset) = 0$.
- Given a  point $P\in \mathbb{R}$, $m^*({P}) = 0$.
  - *Proof*: For any $\epsilon > 0$, we can cover the point $P$ by the interval $I = (P-\epsilon/2, P+\epsilon/2)$, then $m^*(P) \le |I| = \epsilon$. As $\epsilon$ is arbitrary, we have $m^*(P) = 0$.
- Given a finite set $E = \{P_1, P_2, \cdots, P_K\}$, $m^*(E) = 0$.
  - *Proof*: For any $\epsilon > 0$, we can cover the set $E$ by the union of intervals $I_i = (P_i-\epsilon/2K, P_i+\epsilon/2K)$, then $\{P_1,\cdots,P_k\} \subset \bigcup_{i=1}^K I_i$, thus $m^*(E) \le \sum_{i=1}^{K} |I_i| = K \cdot \frac\epsilon K = \epsilon$. As $\epsilon$ is arbitrary, we have $m^*(E) = 0$.
- Given countable set $E = \{P_1, P_2, \cdots\}$, $m^*(E) = 0$.
  - *Proof Sketch*: For any $\epsilon > 0$, we can cover the set $E$ by the union of intervals $I_i = (P_i-\frac{\epsilon}{2\cdot 2^i}, P_i+\frac{\epsilon}{2\cdot 2^i})$, then $\{P_1,P_2,\cdots\} \subset \bigcup_{i=1}^{\infty} I_i$, thus $m^*(E) \le \sum_{i=1}^{\infty} |I_i| = \epsilon$. As $\epsilon$ is arbitrary, we have $m^*(E) = 0$.
- Outer measure of rational number in $[0,1]$ is 0.
- The statement above can be generalized from $\mathbb{R}$ to $\mathbb{R}^n$.