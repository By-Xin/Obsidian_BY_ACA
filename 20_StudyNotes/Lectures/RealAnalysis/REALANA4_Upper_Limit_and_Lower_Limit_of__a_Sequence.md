#RealAnalysis

First discuss **upper limit**  of a numerical sequence in mathematical analysis.

- By intuition: The ultimate trend of the sequence to go up.
  - First, start from $a_1$, the highest value the sequence can reach: $\sup\{a_1, a_2, a_3, \ldots\}$
  - Then, start from $a_2$, the highest value the sequence can reach: $\sup\{a_2, a_3, a_4, \ldots\}$
  - Repeatedly, we get $\sup\{a_n, a_{n+1}, a_{n+2}, \ldots\}\triangleq\sup_{k\geq n}\{a_k\}$
  - The 'ultimate trend' is the limit of this sequence of supremums: $\lim_{n\to\infty}\sup_{k\geq n}\{a_k\}$, which is the **upper limit** of the sequence, denoted by $\overline{\lim}_{n\to\infty}a_n$.

- To review this intuition *Mathematically*:
  - $\sup_{k\geq n}\{a_k\}$ is a decreasing sequence, so it always converges to a limit (possibly $\infty$).  Then we just define the limit of this sequence as the upper limit of the sequence as $\lim_{n\to\infty}\sup_{k\geq n}\{a_k\}$. Furthermore, as this sequence is decreasing, the limit is the infimum of the sequence of supremums: $\lim_{n\to\infty}\sup_{k\geq n}\{a_k\}=\inf_{n\ge1}\sup_{k\geq n}\{a_k\}$.

Similarly, we can define the **lower limit** of the sequence as $\underline{\lim}_{n\to\infty}a_n\triangleq\lim_{n\to\infty}\inf_{k\geq n}\{a_k\} = \sup_{n\ge1}\inf_{k\geq n}\{a_k\}$.

---

Then we can generalize to the **upper limit** and **lower limit** of a sequence of sets. 

And first, we have to clearify the definition of **supremum** and **infimum** of a set.
- **Supremum** of a set sequence $\{A_n\}$ is: the 'smallest' one among all the sets that are 'larger' than $A_n$. 
  - Intuitively, $A$ is 'larger' than $B$ if $A\supset B$.
  - Then a set that is 'larger' than all $A_n$ should at least contain all elements in $A_n$, i.e. $\bigcup_{n=1}^\infty A_n$.
  - Formally, $\sup_{n\ge1}\{A_n\}\triangleq\bigcup_{n=1}^\infty A_n$.
- **Infimum** of a set sequence $\{A_n\}$ is: the 'largest' one among all the sets that are 'smaller' than $A_n$.
  - Intuitively, $A$ is 'smaller' than $B$ if $A\subset B$.
  - Then a set that is 'smaller' than all $A_n$ should at most be contained in all $A_n$, i.e. $\bigcap_{n=1}^\infty A_n$.
  - Formally, $\inf_{n\ge1}\{A_n\}\triangleq\bigcap_{n=1}^\infty A_n$.

Then we can define the **upper limit** and **lower limit** of a sequence of sets as:
- $\overline{\lim}_{n\to\infty}A_n\triangleq\bigcap_{n\ge1}\bigcup_{k\geq n}A_k$
- $\underline{\lim}_{n\to\infty}A_n\triangleq\bigcup_{n\ge1}\bigcap_{k\geq n}A_k$ 


Define the **limit** of the sequence of sets as: For $\{A_n\}$, if $\overline{\lim}_{n\to\infty}A_n=\underline{\lim}_{n\to\infty}A_n$, then $\{A_n\}$ has a limit, denoted by $\lim_{n\to\infty}A_n= \overline{\lim}_{n\to\infty}A_n=\underline{\lim}_{n\to\infty}A_n$.

---

*[Example]* (Monothonic sequence of sets)
- Increasing sequence of sets: $A_1\subset A_2\subset A_3\subset\ldots$.
  - The upper limit is $\overline{\lim}_{n\to\infty}A_n=\bigcap_{n\ge1}\bigcup_{k\geq n}A_k$.
- Decreasing sequence of sets: $A_1\supset A_2\supset A_3\supset\ldots$.
  - The lower limit is $\underline{\lim}_{n\to\infty}A_n=\bigcup_{n\ge1}\bigcap_{k\geq n}A_k$.

It can be proved that for a monothonic sequence of sets, the upper limit and lower limit are the same, and the limit exists.
- For increasing sequence of sets: $\lim_{n\to\infty}A_n=\bigcup_{n\ge1}A_n$
- For decreasing sequence of sets: $\lim_{n\to\infty}A_n=\bigcap_{n\ge1}A_n$

---

To restate the definition of **upper limit** and **lower limit** of a sequence of sets in quantifiers:
- Upper limit ( $\bigcap_{n\ge1}\bigcup_{k\geq n}A_k)$: For any $n\ge1$, there exists $k\geq n$ such that $x\in A_k$.
- Lower limit ( $\bigcup_{n\ge1}\bigcap_{k\geq n}A_k)$: There exists $n\ge1$ such that for any $k\geq n$, $x\in A_k$.

---

*[Example]*  Consider a set of points 
 $$ A_n = \begin{cases} 
      [0, 2 - \frac{1}{2m+1}] & n = 2m+1, m=0,1,2,\ldots \\
      [0, 1 + \frac{1}{2m}] & n = 2m, m=1,2,\ldots
    \end{cases}$$ Compute the upper limit and lower limit of the sequence of sets $\{A_n\}$.

**Solution**:
- First, $\bigcup_{k\geq n}A_k = [0,2)$. Then $\overline{\lim}_{n\to\infty}A_n = \bigcap_{n\ge1}\bigcup_{k\geq n}A_k = [0,2)$.
- Second, $\bigcap_{k\geq n}A_k = [0,1]$. Then $\underline{\lim}_{n\to\infty}A_n = \bigcup_{n\ge1}\bigcap_{k\geq n}A_k = [0,1]$.
$\square$

*[Example]*  $\lim_{n\to\infty}a_n = a$ if and only if $\forall \epsilon>0, \exists N\in\mathbb{N}$ such that $\forall n\geq N, |a_n-a|<\epsilon$.
This can be restated as: 
$$ a\in \bigcap_{\epsilon\in\mathbb{R}^+} \left[\bigcup_{N=1}^\infty\bigcap_{n\geq N}\{x: |a_n-x|<\epsilon\}\right]$$
The brackets is actually the lower limit: $\underline{\lim}_{n\to\infty}\{x: |a_n-x|<\epsilon\}$, which is the set of all points that are 'close' to $a$.
Thus, it can be restated as: 
$$\bigcap_{\epsilon\in\mathbb{R}^+} \underline{\lim}_{n\to\infty}\{x: |a_n-x|<\epsilon\} $$

