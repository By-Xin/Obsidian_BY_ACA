#NonparametricStatistics

Applications of the Sign Test:
- Single population quantile (median) test
- Single population proportion test
- Paired population median test

Sign Test is applicable for small sample sizes, and when the data is not normally distributed. 

---

 Given $n$ data points $x_1, x_2, \ldots, x_n$. Given a null hypothesis $H_0: \text{median} = m_0$, and an alternative hypothesis $H_1: \text{median} \neq m_0$. The Sign Test is used to test the null hypothesis under given significance level $\alpha$. 

1. Go through the data points and remove the *ties* (data points equal to $m_0$). Redefine the sample size as $n'$.
2. Compute the number of data points greater than $m_0$ denoted as $N_{+}$, and the number of data points less than $m_0$ denoted as $N_{-}$.
3. The test statistic is given by $S = N_{+}$ by convention. Under null hypothesis, $S\stackrel{H_0}{\sim} \text{Binomial}(n', 0.5)$.
4. Compute the p-value: 
   $$p = 2\cdot \mathbb{P}(S \geq s_{obs} | H_0) = 2\sum_{k=s_{obs}}^{n'} \binom{n'}{k} 0.5^{n'}$$ where $s_{obs}$ is the observed value of $S$ under the sample data.
   - If $H_a: \text{median} > m_0$, then $p = \mathbb{P}(S \geq s_{obs})$.
   - If $H_a: \text{median} < m_0$, then $p = \mathbb{P}(S \leq s_{obs})$.

---

Large sample approximation of the Sign Test:
- Empirically, for Sign Test on Median, if $n\ge 10$, then the test statistic $S$ can be approximated by a normal distribution.
- In this case, the Sign Test is equivalent to the $\hat{p}$ Z-test.
- Yet, for binomial distribution, we may also consider the adjusted continuity correction for the normal approximation.

---

***[Example]*** A survey was conducted on the heights of $n=10$ students. Conduct a Sign Test to test the null hypothesis that 3/4 quartile of the heights is 180 cm.
***[Solution]***
- Consider the null hypothesis $H_0: Q_3 = 180$ cm, and alternative hypothesis $H_1: Q_3 \neq 180$ cm.
- First check for ties. Then we can see that $N_+ = 8$ and $N_- = 2$. Here let $S = N_+ = 8$. 
- Under null hypothesis, $S\stackrel{H_0}{\sim} \text{Binomial}(n, 1/4)$.
- Compute the p-value: $p = 2 \mathbb{P}(S \geq 8|H_0) = 2 \sum_{k=8}^{10} \binom{10}{k} 0.25^{k} 0.75^{10-k} = 0.0008316 < 0.05$. Reject the null hypothesis.

***[Example]*** (Paired Sign Test) There are two types of tires, A and B. We want to test if A is significantly more durable than B. The data of $n=10$ cars are given. 
- If we can assume that the data is from Normal distribution, we can use the paired t-test to test if the mean differes significantly.
- Otherwise, we can use the Sign Test, without assuming the distribution of the data, and test if the median differes significantly.
  - Consider the null hypothesis $H_0: \text{median}(A) = \text{median}(B)$, and alternative hypothesis $H_1: \text{median}(A) > \text{median}(B)$.
  - First check for ties (if $A_i = B_j$) and remove them. 
  - Then compute the statistics $S = \sum_{i=1}^{n} \mathbb{I}(A_i > B_i) = 8$.
  - Compute the p-value: $p = \mathbb{P}(S \geq 8|H_0) = \sum_{k=8}^{10} \binom{10}{k} 0.5^{10} = 0.0547 > 0.05$. Fail to reject the null hypothesis.