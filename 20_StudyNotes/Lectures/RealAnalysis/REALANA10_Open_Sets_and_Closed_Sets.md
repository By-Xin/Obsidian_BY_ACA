#RealAnalysis

***Definition (Open Set):*** Given a set $E$, for all $x \in E$, $x$ is an interior point of $E$.
-  Intuitively, an open set is a set that does not contain its boundary points.
  
***Definition (Closed Set):*** Given a set $E$ and for any cluster point $x$, $x\in E$.

- *Example 1*: $\mathbb{R}^n$ is both open and closed.
- *Example 2*: $(-1,1)$ is open but not closed. $[-1,1]$ is closed but not open. $(0,1]$ is neither open nor closed. Generally, in $\mathbb{R}^n$, $(a_1,b_1)\times(a_2,b_2)\times\cdots\times(a_n,b_n)$ is open, $[a_1,b_1]\times[a_2,b_2]\times\cdots\times[a_n,b_n]$ is closed.
- *Example 3*: In $\mathbb{R}^2$, $\{(x,y):x^2+y^2<1\}$ is open. $\{(x,y):x^2+y^2\leq1\}$ is closed. However, $\{(x,y):x^2+y^2<1\}$ in $\mathbb{R}^3: \{(x,y,z):x^2+y^2<1, z=0\}$ is not an open set. 

---

***Theorem 1:*** $E$ is open if and only if $E = E^\circ$. 

***Theorem 2:*** $E$ is closed if and only if $E = \bar{E}$.

***Remark:*** $E^\circ$ is the 'largest' open set contained in $E$. $\bar{E}$ is the 'smallest' closed set containing $E$.

***Theorem 3:*** If $E$ is open, then $E^c$ is closed. If $E$ is closed, then $E^c$ is open.