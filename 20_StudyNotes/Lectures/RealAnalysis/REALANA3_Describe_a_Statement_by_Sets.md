#RealAnalysis

First start with some intuitive examples of representing a statement by sets:
- $f$ is defined on $\mathbb{R}$, which has the upper bound $M$ on $[a,b]$ (i.e. $ \forall x \in [a,b], f(x) \leq M$)
    $$ [a,b] \subset \{x \in \mathbb{R} : f(x) \leq M\}$$
- $f$ is continuous on $\mathbb{R}$ (i.e. Given $x_0 \in \mathbb{R}$, $\forall \epsilon > 0, \exists \delta > 0$ such that  $ \forall x \in \mathbb{R}$, $|x - x_0| < \delta , $ we have $ |f(x) - f(x_0)| < \epsilon$)
    $$ (x_0 - \delta, x_0 + \delta) \subset \{x \in \mathbb{R} : |f(x) - f(x_0)| < \epsilon\}$$ or equivalently
    $$ f((x_0 - \delta, x_0 + \delta)) \subset (f(x_0) - \epsilon, f(x_0) + \epsilon)$$
- $f$ is defined on $\mathbb{R}$, which has an upper bound $M$ on $\mathbb{R}$.
    $$ \mathbb{R} = \{x : f(x) \leq M\} = \{x  : f(x) \leq M +1\}$$
- $f,g$ are defined on some set $E$. Then $\forall c \in \mathbb{R}$, we have:  
    $$\{x\in E : \max(f(x),g(x)) >c\} = \{x \in E : f(x) > c\} \cup \{x \in E : g(x) > c\}$$
    - *[Proof]*
      - $LHS \subset RHS$: For $\forall x \in \{x\in E : \max(f(x),g(x)) >c\}$, we have $\max(f(x),g(x)) > c$. Then, either $f(x) > c$ or $g(x) > c$. Therefore, $x \in \{x \in E : f(x) > c\}$ or $x \in \{x \in E : g(x) > c\}$. Therefore, $x \in \{x \in E : f(x) > c\} \cup \{x \in E : g(x) > c\}$. $LHS \subset RHS$.
      - $RHS \subset LHS$: For $\forall x \in \{x \in E : f(x) > c\} \cup \{x \in E : g(x) > c\}$, we have $f(x) > c$ or $g(x) > c$. Then, $\max(f(x),g(x)) > c$. Therefore, $x \in \{x\in E : \max(f(x),g(x)) >c\}$. $RHS \subset LHS$.
- $\{f_n(x)\}$ is a sequence of functions defined on $E$. Then $\forall c\in \mathbb{R}:$
  - a. $\{x \in E : \sup f_n(x) > c\} = \bigcup_{n=1}^\infty \{x \in E : f_n(x) > c\}$ 
    - *[Proof]*
        - [$L\subset R$] $\forall x\in \{x \in E : \sup f_n(x) > c\}$, then  $\exist n\in \mathbb{N}$, we have $f_n(x) > c ^\dagger$.  And this statement can be written in the form of sets as: $x \in \bigcup_{n\in\mathbb{N}} \{x \in E : f_n(x) > c\}$. 
          - $^\dagger$ If not, then $\forall n\in\mathbb{N}, f_n(x) \leq c$. Then, $\sup f_n(x) \leq c$, which contradicts the assumption that $\sup f_n(x) > c$. 
        - [$R\subset L$] $\forall x\in \bigcup_{n\in\mathbb{N}} \{x \in E : f_n(x) > c\}$, then $\exist n\in \mathbb{N}$, we have $f_n(x) > c$. Then, $\sup f_n(x) > c$ (by property of supremum). Therefore, $x \in \{x \in E : \sup f_n(x) > c\}$.
  - b. $\{x \in E : \inf f_n(x) \leq c\} = \bigcap_{n=1}^\infty \{x \in E : f_n(x) \leq c\}$

- (Nested Intervals Theorem) Given a sequence of nested intervals: $[a_n, b_n] \subset [a_{n-1}, b_{n-1}] $ which satisfies $\lim_{n \to \infty} (b_n - a_n) = 0$. Then, $\exists! \alpha \in \mathbb{R}$ such that $\alpha \in [a_n, b_n] \forall n \in \mathbb{N}$.
    - The existence of $\alpha$ **for all $n$** can be represented by the set: 
        $$ \alpha \in \bigcap_{n=1}^\infty [a_n, b_n]$$
    - The uniqueness of $\alpha$ can be represented by the set:
        $$\{\alpha\} = \bigcap_{n=1}^\infty [a_n, b_n]$$ which means the intersection of all intervals is a singleton set.

- $(a,b) = \bigcup_{n=1}^\infty [a+\frac1n,b-\frac1n]$
  - *[Proof]*
    - $[L\subset R]$: $\forall x\in (a,b)$, then $x-a >0, b-x>0$. Then for some sufficiently large $n$, we have $\frac1n < \min (x-a, b-x)$. Then $x\in [a+\frac1n,b-\frac1n]$.  
    - $[R\subset L]$: $\forall x\in \bigcup_{n=1}^\infty [a+\frac1n,b-\frac1n]$, then $\exist n\in \mathbb{N}$, we have $x\in [a+\frac1n,b-\frac1n] \subset (a,b)$. 

- $f$ is defined on $E$, then $\{x:f(x)>0\} = \bigcup_{n=1}^\infty \{x:f(x)>\frac1n\}$
  
**Then, we can conclude an important rule to represent a statement by sets:**
- **Union** of sets is used to represent a statement with **Existential Quantifier $\exist$**.
- **Intersection** of sets is used to represent a statement with **Universal Quantifier $\forall$**.

Consider the following examples:

- $\{f_n(x)\}$ is a sequence of functions defined on $E$. If there is some $x\in E, \text{ s.t. } \{f_n(x)\}$ is bounded, then it means: $\exist M>0, \text{ s.t. } \forall n\in \mathbb{N}, |f_n(x)| \leq M$. This can be represented by the set:
    $$ \{x \in E : \{f_n(x)\} \text{ is bounded}\} = \bigcup_{M\in\mathbb{R}_+}\bigcap_{n=1}^\infty\{|f_n(x)| \leq M\}$$
  And by contrast,
    $$\begin{aligned}
     \{x \in E : \{f_n(x)\} \text{ is unbounded}\} =& ( \bigcup_{M\in\mathbb{R}_+}\bigcap_{n=1}^\infty\{|f_n(x)| \leq M\})^c \\\stackrel{\text{De Morgan}}{=}& \bigcap_{M\in\mathbb{R}_+}\bigcup_{n=1}^\infty\{|f_n(x)| > M\}
     \end{aligned}$$


- ($\epsilon-\delta$ definition): $\{f_n(x)\}$ is a sequence of functions defined on $E$. Those $x$ that $\{f_n(x)\}$ converges to $0$ means: $\forall \epsilon>0, \exists N\in\mathbb{N}, \text{ s.t. } \forall n>N, |f_n(x)| < \epsilon$. This can be represented by the set:
    $$ \{x \in E : \{f_n(x)\} \text{ converges to } 0\} = \bigcap_{\epsilon\in\mathbb{R}_+}\bigcup_{N\in\mathbb{N}}\bigcap_{n>N}^\infty\{|f_n(x)| < \epsilon\}$$
  And by contrast,
    $$\begin{aligned}
     \{x \in E : \{f_n(x)\} \text{ does not converge to } 0\} =& ( \bigcap_{\epsilon>0}\bigcup_{N\in\mathbb{N}}\bigcap_{n=N+1}^\infty\{|f_n(x)| < \epsilon\})^c \\\stackrel{\text{De Morgan}}{=}& \bigcup_{\epsilon>0}\bigcap_{N\in\mathbb{N}}\bigcup_{n=N+1}^\infty\{|f_n(x)| \geq \epsilon\}
     \end{aligned}$$