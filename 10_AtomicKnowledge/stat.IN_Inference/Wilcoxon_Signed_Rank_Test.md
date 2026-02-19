#NonparametricStatistics

Wilcoxon Signed Rank still does not assume the distribution of the data. 

Applications of the Sign Test:
- Single population quantile (median) test
- Paired population median test

---

 Given $n$ data points $x_1, x_2, \ldots, x_n$. Given a null hypothesis $H_0: \text{median} = m_0$, and an alternative hypothesis $H_1: \text{median} \neq m_0$. Wilcoxon Sign Rank Test  is used to test the null hypothesis under given significance level $\alpha$. 

1. Go through the data points and remove the *ties* (data points equal to $m_0$). Redefine the sample size as $n'$.
2. Compute the difference between the data points and the hypothesized median $D_i = x_i - m_0$. Rank the absolute values of $D_i$: $r_i = \text{rank}(|D_i|)$ (if ranks are tied, average the ranks).
   $$\begin{array}{|c|c|c|c|c|} \hline
   i & 1 & 2 & \cdots & n \\ \hline
    x_i & x_1 & x_2 & \cdots & x_n \\ \hline
    D_i = x_i - m_0 & D_1 & D_2 & \cdots & D_n \\ \hline
    r_i = \text{rank}(|D_i|) & r_1 & r_2 & \cdots & r_n \\ \hline
    \end{array}$$


3. Compute $W_+$ as the sum of ranks of positive differences, and $W_-$ as the sum of ranks of negative differences. 
   - $W_+ = \sum_{i=1}^{n} r_i \mathbb{I}(D_i > 0)$ 
   - $W_- = \sum_{i=1}^{n} r_i \mathbb{I}(D_i < 0)$
   - $W_+ + W_- = \frac{n(n+1)}{2}$
4. Define the test statistic: 
   $$W = \min\left(W^+, W^-\right)$$
5. Compute the Reject Region (Note that $W$ always has a left-tail rejection region, thus we always use $\leq$):
    $$\mathcal{RR} = \{W \leq w_{\alpha/2} (n)\}$$  where $w_{\alpha/2}(n)$ should be determined from the Wilcoxon Signed Rank Table.
     - For two-tailed test, we choose $w_{\alpha/2}$ 
     - For one-tailed test, we choose $w_{\alpha}$. 
6. If $W_{\text{obs}} \in \mathcal{RR}$, reject the null hypothesis. Otherwise, fail to reject the null hypothesis.

---

***Normal Approximation of Wilcoxon Signed Rank Test***

For large sample sizes ($n\ge 15$), the Wilcoxon Signed Rank Test can be approximated by a normal distribution:
$$
W = \min\left(W^+, W^-\right) \stackrel{\mathcal{L}|H_0}{\longrightarrow} \mathcal{N}\left(\frac{n(n+1)}{4}, \frac{n(n+1)(2n+1)}{24}\right)
$$
Thus, we can compute the z-score and p-value for the test statistic $W$:
$$
Z = \frac{W - \frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24}}} \stackrel{H_0}{\longrightarrow} \mathcal{N}(0, 1)
$$
- For two-tailed test, the rejection region is $\mathcal{RR} = \{ Z \leq -z_{\alpha/2} \} $
- For one-tailed test, the rejection region is $\mathcal{RR} = \{ Z \leq -z_{\alpha} \}$


***[Example]*** A survey was conducted to compare the quality of two products A and B, with $n=5$ samples: $A_1,B_1,\cdots, A_5, B_5$. The null hypothesis is that the quality of two products are the same: $H_0: \text{median}(A) = \text{median}(B)$. The alternative hypothesis is that the quality of product A is not the same as B: $H_a: \text{median}(A) \neq \text{median}(B)$.
***[Solution]***
Given the data, we can compute the differences $D_i = A_i - B_i$ and the ranks of the absolute differences as:
$$\begin{array}{|c|c|c|c|c|c|} \hline
i & 1 & 2 & 3 & 4 & 5 \\ \hline
D_i = A_i-B_i &6&3&2&-1&5 \\ \hline
|r_i| & 5 & 3 & 2 & 1 & 4 \\ \hline
\end{array}$$
Then we can compute the sum of ranks of positive differences $W^+ = 5+3+2+4 = 14$, and the sum of ranks of negative differences $W^- = 1$. The test statistic is $W = \min(14, 1) = 1$. 
From the Wilcoxon Signed Rank Table, we can compute the rejection region for $n=5$ and $\alpha=0.1$: $\mathcal{RR} = \{W \leq w_{\alpha/2}(5) = 1\}$. Since $W_{\text{obs}} = 1 \in \mathcal{RR}$, we reject the null hypothesis.
