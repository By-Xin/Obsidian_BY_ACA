#RealAnalysis

# What is Real Analysis trying to do?

- Riemann Integral (Area under the curve)
    - Intuition: 
      - Divide the interval into small pieces as rectangles
      - Sum up the areas of the rectangles
      - Take the limit as the number of rectangles goes to infinity.
    - Limitation of Riemann Integral: 
      - Dirichlet function: 
      $$ D(x)  = \mathbb{I}_Q(x)  , x \in [0,1] $$ where $\mathcal{I}_Q(x)$ is the indicator function of the rational numbers.
        - Dirichlet function is not Riemann integrable. 
        - However, **there exists a sequence of increasing Riemann integrable functions that converges to the  Dirichlet function (Riemann unintegrable)**:
            - Arrange the rational numbers in a sequence: $q_1, q_2, q_3, \cdots, q_n, \cdots$
            - Define $D_n(x) = \mathbb{I}_{[0,1]\cap \{q_1, q_2, \cdots, q_n\}}(x)$. It is obvious that $D_n(x)$ is Riemann integrable.
            - It can be shown that $\lim_{n \to \infty} D_n(x) = D(x)$.
      - Exchanging limit and integral has complex conditions for Riemann integral.
      - Improper integral is also complex for Riemann integral. We also want to generalize to infinite intervals.
      - Riemann integral is only defined on Real Space. We want to generalize to other spaces.

- Lebasgue Integral
  - Intuition: 
    - Divide the *value* of the function into small pieces 
      - Need **Set Theory** for 'small pieces' 
      - Need **Point Set Topology** as these pieces are subset of $\mathbb{R}$
      - Need **Measure Theory** to measure the 'size' of the pieces, and which kind of $f$ is measurable.
    - Sum up the areas of tiny pieces and take the limit to converge to the integral.
      - Need **Lebesgue Integral** to sum up the 'size' of the pieces.

- Example: **Dirichlet function is Lebesgue integrable**. 
  - For Dirichlet function: $ I = 0\times \text{measure of the set of rational numbers} + 1 \times \text{measure of the set of irrational numbers} = 0$.