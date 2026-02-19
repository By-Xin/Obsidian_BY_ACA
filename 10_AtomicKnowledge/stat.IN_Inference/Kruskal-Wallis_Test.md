#NonparametricStatistics

Kruskal-Wallis Test is a non-parametric test used to compare the medians of $k\ge 2$ independent population groups. It is a non-parametric alternative to the one-way ANOVA test. 

In the Kruskal-Wallis Test:
- No Normal distribution assumption to the data
- Still requires the homoscedasticity assumption (equal variance)

---

Assume there are $k$ independent population groups with sample sizes $n_1, n_2, n_3$. Given the null hypothesis $H_0: \text{median}_1 = \text{median}_2 = \text{median}_3$, and the alternative hypothesis $H_1: \text{medians are not all equal}$. 

Compute the test statistic $H$ as follows:
$$\begin{array}{|c|c|c|c|} \hline
\text{Pop.Grp. }& \text{Rank} & \text{Sample Size} & \text{Rank Sum} \\ \hline
1 & r_{1,1}, r_{1,2}, \ldots, r_{1,n_1} & n_1 & R_1 = \sum_{i=1}^{n_1} r_{1,i} \\ \hline
2 & r_{2,1}, r_{2,2}, \ldots, r_{2,n_2} & n_2 & R_2 = \sum_{i=1}^{n_2} r_{2,i} \\ \hline
\vdots & \vdots & \vdots & \vdots \\ \hline
k & r_{k,1}, r_{k,2}, \ldots, r_{k,n_k} & n_k & R_k = \sum_{i=1}^{n_k} r_{k,i} \\ \hline
\end{array}$$

Define the test statistic $H$ as:
$$
H = \frac{12}{n(n+1)}\left(\sum_{j=1}^{k} \frac{R_j^2}{n_j} - 3(n+1)\right) \stackrel{H_0}{\longrightarrow} \chi^2(k-1)
$$

As a chi-square test, the rejection region is always right-tailed:
$$
\mathcal{RR} = \{H \geq \chi^2_{\alpha}(k-1)\}
$$
