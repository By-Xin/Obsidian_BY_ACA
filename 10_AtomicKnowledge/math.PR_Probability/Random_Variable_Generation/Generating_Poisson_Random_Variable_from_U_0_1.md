---
aliases: [泊松分布随机数生成, Generating Poisson Random Variable]
tags:
  - method
  - math/probability
  - stat/computational
related_concepts:
  - "[[Poisson_Distribution]]"
  - "[[Generating_Discrete_Random_Variables]]"
  - "[[Generating_Binomial_Random_Variable]]"
source: "计算统计学"
---

# 从 U(0,1) 生成泊松分布随机变量

#ComputationalStatistics 

## 模型简述：

- 该算法的思路与4.1一致：通过生成$U$模拟泊松分布的分布函数，再寻找其对应的自变量值$X$。不同之处在于泊松分布的函数正常计算较为复杂，故采取递推方式计算。
- 递推的思路：首先仍生成一均匀分布$U$代表泊松分布函数，然后开始循环讨论。从$F(i)=P\{X=i\},i=0$开始，看$F(i)$的值是否大于生成的$U$的值。若是则该$i$即为想要模拟的$x$，若不是则$i++$，继续讨论。
- 简而言之，对于递推形式，算法的核心在于依次比较$F(0),F(1),F(2),\cdots$与$U$的大小，第一个使得$F(i)>U$的i即为所求的$x$。
### 模拟目的：
生成服从参数为$\lambda$的泊松分布的随机变量$X$，即
$$
  p_i=P\{X=i\}=e^{-\lambda} \frac{\lambda^i}{i! } \quad i=0,1, \ldots
  $$
### 模拟方法：
首先，由于Poisson分布等分布函数涉及阶乘等，计算复杂度较高，通常采用递推的方式进行计算，即：
$$
  \begin{gathered}
  \frac{p_{i+1}}{p_i}=\frac{\frac{e^{-\lambda} \lambda^{i+1}}{(i+1) !}}{\frac{e^{-\lambda} \lambda^i}{i !}}=\frac{\lambda}{i+1} \\
  \boxed{p_{i+1}=\frac{\lambda}{i+1} p_i, \quad i \geqslant 0}
  \end{gathered}
  $$
**下对泊松分布进行模拟：**
> STEP 1: Generate a random number $U$.
> STEP 2: $i=0, p=e^{-\lambda}, F=p$.
> STEP 3: If $U<F$, set $X=i$ and stop.
> STEP 4: $p=\lambda p /(i+1), F=F+p, i=i+1$.
> STEP 5: Go to Step 3.
### 代码实现：
```matlab
% function X=cspoirnd(lam,n)
% This function will generate Poisson(lambda)

function x=cspoirnd(lam,n)
x=zeros(1,n);
j=1;
while j<n
flag =1;
  % initialize quantities
  u=rand(1);
  i=0;
  p=exp(-lam);
  F=p;
  while flag % generate the variate needed
	if u<=F % then accept
		x(j)=i;
		  flag=0;
		  j=j+1;
	  else % move to next probability
		p=lam*p/(i+1);
		  i=i+1;
		  F=F+p;
	  end
  end
end
```