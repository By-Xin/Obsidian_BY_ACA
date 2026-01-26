---
aliases: [二项分布随机数生成, Generating Binomial Random Variable]
tags:
  - method
  - math/probability
  - stat/computational
related_concepts:
  - "[[Binomial_Distribution]]"
  - "[[Generating_Discrete_Random_Variables]]"
  - "[[Generating_Poisson_Random_Variable]]"
  - "[[Central_Limit_Theorem]]"
source: "计算统计学"
---

# 从 U(0,1) 生成二项分布随机变量

#ComputationalStatistics 

## 模型概述：
- 与泊松分布的生成方法类似；
- 注意的是当$np$或泊松分布中的$\lambda$较大时，算法有较大的改进空间。但讲义中并未提及，故略去。
### 模拟目的：
生成二项分布$(n,p)$随机变量$X$，即：
$$
  P\{X=i\}=\frac{n !}{i !(n-i) !} p^i(1-p)^{n-i}, \quad i=0,1, \ldots, n
  $$
### 模拟方法：
同样采用递归形式：
$$
  P\{X=i+1\}=\frac{n-i}{i+1} \frac{p}{1-p} P\{X=i\}
  $$
> 注：在算法实现时，注意到$p/(1-p)$为常数（与$i$无关）故可以另记之方便计算.

**实现算法为：**
> STEP 1: Generate a random number $U$.
> STEP $2: c=p /(1-p), i=0, \mathrm{pr} =(1-p)^n, F= \mathrm{pr}$
> STEP 3: If $U<F$, set $X=i$ and stop.
> STEP 4: $\mathrm{pr}=[c(n-i) /(i+1)] \mathrm{pr}, F=F+\mathrm{pr}, i=i+1$.
> STEP 5: Go to Step 3.


### 说明：
- 这里$c$即为上述的常数项，$\mathrm{pr}$为递归形式的$P\{X=i\}$，$F$为累积的分布函数；
- 要注意到循环的次数总是比确定的$X$值大一。显然，即使$X=0$，也要经过一次循环比较才能确定；
- 根据二项分布的性质，当$p>1/2$时可以通过上述算法生成$Y\sim b(n,1-p)$，$X=n-Y$即为所求；
- 另一种实现方法为模拟$n$次实验的结果.
### 代码实现：
```matlab
% set up storage space for the variables
X=zeros(1,100);
% These are the x's in the domain
x=0:2;
% These are the prob. masses.
pr=[0.3 0.2 0.5];
% Generate 100 rv's from the desired distribution. 
for i=1:100
u=rand; %generate U
  if u<=pr(1)
	X(i)=x(1);
  elseif u<=sum(pr(1:2)) 
	% it has to be between 0.3 and 0.5
	X(i)=x(2);
  else
	X(i)=x(3);
	  % it has to be between 0.5 and 1.
  end
end
```