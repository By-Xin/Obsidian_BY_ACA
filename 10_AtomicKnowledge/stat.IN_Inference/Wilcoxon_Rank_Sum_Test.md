#NonparametricStatistics

Wilcoxon Rank Sum Test is actually equivalent to [[Mann-Whitney U Test]]. Both tests are used to compare the medians of two independent populations, which do not assume the distribution of the data, and the sample sizes do not need to be equal.

---

 Consider two independent samples $X = \{x_1, x_2, \ldots, x_m\}$ and $Y = \{y_1, y_2, \ldots, y_n\}$. Use the Wilcoxon Rank Sum Test to test if the medians of two populations are equal under given significance level $\alpha$.

 Given the null hypothesis $H_0: \text{median}(X) = \text{median}(Y)$, and alternative hypothesis $H_1: \text{median}(X) \neq \text{median}(Y)$.

Mix the two samples and rank the values, and share the rank for tied values. 

$$\begin{array}{|c|c|c|} \hline
\text{Pop.} & \text{Rank}  & \text{Rank Sum} \\ \hline
X & r(X_1)~  r(X_2)~  \cdots ~ r(X_m) & W_1 = \sum_{i=1}^{m} r(X_i) \\ \hline
Y & r(Y_1)~ r(Y_2) ~ \cdots ~ r(Y_n) & W_2 = \sum_{i=1}^{n} r(Y_i) \\ \hline
\end{array}$$

Conclude that:
$$
W_1+ W_2 = \frac{(m+n)(m+n+1)}{2}
$$

Define the test statistic $W$ to be the one with smaller sample size (here assume $m \leq n$ thus $W = W_1$).

Then the rejection region is:
$$
\mathcal{RR} = \left\{ W \leq w^L_{\alpha/2}(m,n) \right\} \cup \left\{ W \geq w^U_{\alpha/2}(m,n) \right\}
$$ where the lower critical value $w^L_{\alpha/2}(m,n)$ and upper critical value $w^U_{\alpha/2}(m,n)$ should be determined from the Wilcoxon Rank Sum Table.
- For two-tailed test, we choose $\mathcal{RR} = \left\{ W \leq w^L_{\alpha/2} \right\} \cup \left\{ W \geq w^U_{\alpha/2} \right\}$
- For one-tailed test
  - If $H_a: \text{median}(X) > \text{median}(Y)$, then $\mathcal{RR} = \left\{ W \geq w^U_{\alpha} \right\}$
  - If $H_a: \text{median}(X) < \text{median}(Y)$, then $\mathcal{RR} = \left\{ W \leq w^L_{\alpha} \right\}$

---

***Normal Approximation of Wilcoxon Rank Sum Test:***

For large sample sizes ($m,n \geq 10$), the Wilcoxon Rank Sum Test can be approximated by a normal distribution. 
$$\begin{aligned}
W_1 &\stackrel{\mathcal{L} | H_0}{\longrightarrow} \mathcal{N}\left(\frac{m(m+n+1)}{2}, \frac{mn(m+n+1)}{12}\right) \\
W_2 &\stackrel{\mathcal{L} | H_0}{\longrightarrow} \mathcal{N}\left(\frac{n(m+n+1)}{2}, \frac{mn(m+n+1)}{12}\right)
\end{aligned}$$

Either $W_1$ or $W_2$ can be used as the test statistic regardless of the sample sizes; yet $W_1$ may be more intuitive w.r.t. to the hypothesis.

$$
Z = \frac{W - \frac{m(m+n+1)}{2}}{\sqrt{\frac{mn(m+n+1)}{12}}}\stackrel{H_0}{\longrightarrow} \mathcal{N}(0,1)
$$

- For the two-tailed test, the rejection region is $\mathcal{RR} = \left\{ |Z| \geq z_{\alpha/2} \right\}$.
- For the one-tailed test
  - If $H_a: \text{median}(X) > \text{median}(Y)$, then $\mathcal{RR} = \left\{ Z \geq z_{\alpha} \right\}$.
  - If $H_a: \text{median}(X) < \text{median}(Y)$, then $\mathcal{RR} = \left\{ Z \leq -z_{\alpha} \right\}$.