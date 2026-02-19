#NonparametricStatistics

Mann-Whitney U Test is actually equivalent to [[Wilcoxon Rank Sum Test]]. 

First, it is the same as the Wilcoxon Rank Sum Test that we should mix the samples of two independent populations (with sample size $n_1, n_2$), rank the values, and share the rank for tied values, giving the rank sum $W_1$ and $W_2$ for each population. 

Then, the test statistic $U_1, U_2$ is defined as:
$$\begin{aligned}
U_1 &= n_1 n_2 + \frac{n_1(n_1+1)}{2} - W_1 \\
U_2 &= n_1 n_2 + \frac{n_2(n_2+1)}{2} - W_2
\end{aligned}$$

Define the test statistic $U$ to be the one with smaller sample size :$U = \min(U_1, U_2)$. The rejection region is:
$$
\mathcal{RR} = \left\{ U \leq u_{\alpha/2}\right\}
$$
where $u_{\alpha/2}$ should be determined from the Mann-Whitney U Table. Note that as we have chosen the smaller sample size, **the rejection region is always on the left side.** The one-tailed test will be: $\mathcal{RR} = \left\{ U \leq u_{\alpha}\right\}$ (only change the significance level).


For large sample tests, the asymptotic normality of $U$ is the same as the Wilcoxon Rank Sum Test. 

---

***[Example]*** Consider the salary of two groups of employees, where the sample sizes are $n_1 = 12$ and $n_2 = 11$. Test if the median salary of two groups are equal under the significance level $\alpha = 0.05$.

***[Solution]***
1. Rank the salaries of two groups and calculate the rank sum $W_1, W_2$. Here the results are $W_1 = 149, W_2 = 127$.
2. Calculate the test statistic $U_1= 61, U_2 = 71$, and thus $U = 61$.
3. Find the rejection region from the Mann-Whitney U Table: $\mathcal{RR} = \{ U \leq u_{\alpha/2}^{L} \} = \{ U \leq 33 \} $.
4. Since $U_0 = \min(U_1, U_2) = U_1 = 61 \not\in \mathcal{RR}$, we fail to reject the null hypothesis. Thus, we conclude that there is no significant difference in the median salary of two groups.