---
aliases: [离散随机变量生成, Generating Discrete Random Variables, 逆变换法]
tags:
  - method
  - math/probability
  - stat/computational
related_concepts:
  - "[[Generating_Continuous_Random_Variables]]"
  - "[[Generating_Poisson_Random_Variable]]"
  - "[[Generating_Binomial_Random_Variable]]"
  - "[[Generating_Geometric_Random_Variable]]"
source: "计算统计学"
---

# 离散随机变量的生成

#ComputationalStatistics

## 4.1 逆变换法 (Inverse Transform Method)
### 模型简述：
不论何种随机变量，定有一个相对应的分布函数，且由分布函数的性质可以确定其在$(0,1)$区间式单调递增的，故具有（广义）反函数。模拟的思路即为通过生成均匀分布随机数$U(0,1)$模拟其分布函数数值$F$，再通过寻找分布函数的反函数确定其随机变量$X$的数值。
### 模拟目的：
生成一系列离散型随机变量$X$，其概率密度函数服从：
$$P\{X=x_j\}=p_j,~~j=0,1,\cdots,~~\sum_j p_j=1$$
### 模拟方法：
1. 生成随机数$U$（服从$U(0,1)$的均匀分布）
2. 令$$ \begin{aligned}
  X= \begin{cases}x_0 & \text { If } U<p_0 \\ x_1 & \text { If } p_0 \leq U<p_0+p_1 \\ \vdots & \\ x_j & \text { If } \sum_{i=0}^{j-1} p_i \leq U<\sum_{i=0}^j p_i \\ \vdots & \end{cases}
  \end{aligned}$$
3. 对于$0<a<b<1, p\{a\le U <b\}=b-a$，有：$$\begin{aligned}
  p\left\{X=x_j\right\}=p\left\{\sum_{i=0}^{j-1} p_i \leq U<\sum_{i=0}^j p_i\right\}=p_j
  \end{aligned}$$此时的$X$即为所求.


### 注意：
1. 算法表达：
	Generate a random number $U$
	If $U<p_0$ set $X=x_0$ and stop
	If $U<p_0+p_1$ set $X=x_1$ and stop
	If $U<p_0+p_1+p_2$ set $X=x_2$ and stop
2. 若$x_i$是顺序排列的，即$x_0<x_1<\cdots$，且记$F(x_k)=\sum_{i=0}^kp_i$，则：$$X=x_j, ~~\mathrm{if}~~ F(x_{j-1}\leq U<F(x_j))$$换言之，该过程即为寻找$F^{-1}(U)$对应的$X$
### 例4a 生成给定分布离散随机变量
要求：生成随机变量$X$满足：
$$\begin{aligned}
  p_1=0.20, \quad p_2=0.15, \quad p_3=0.25, \quad p_4=0.40 \quad \text { where } p_j=P\{X=j\}
  \end{aligned}$$
**解法一：**
>Generate U
> If $U<0.20$ set $X=1$ and stop
> If $U<0.35$ set $X=2$ and stop
> If $U<0.60$ set $X=3$ and top
> Otherwise set $X=4$


**解法二：**
> Generate U
> If $U<0.40$ set $X=4$ and stop
> If $U<0.65$ set $X=3$ and stop
> If $U<0.85$ set $X=1$ and stop Otherwise set $X=2$

### 例4d [[Generating Geometric Random Variable from U(0,1)]]
要求：生成随机变量$X$满足参数为$p$的几何分布，即：
$$\begin{aligned}
  P\{X=i\}=p q^{i-1}, \quad i \geq 1, \quad \text { where } q=1-p
  \end{aligned}$$
**解：**
由几何分布的含义，$X$可认为是$n$次独立实验中首次成功的时间，且每次实验的成功概为$p$，故有：$$
\begin{aligned}
\sum_{i=1}^{j-1} P\{X=i\} &=1-P\{X>j-1\} \\
&=1-P\{\text { first } j-1 \text { trials are all failures }\} \\
&=1-q^{j-1}, \quad j \geq 1
\end{aligned}
$$
故可以生成随机数$U$并令：
$$
1-q^{j-1} \leq U<1-q^j ~~ \Rightarrow ~~q^j<1-U \leq q^{j-1}
$$
因此$X$为：
$$
X=\operatorname{Min}\left\{j: q^j<1-U\right\}~~\cdots(\star)
$$
下需解出$j$的具体数值。由对数函数的单调性，对$\star$式集合中不等式两侧求对数，有：
$$
  \begin{aligned}
  X &=\operatorname{Min}\{j: j \log (q)<\log (1-U)\} \\
  &=\operatorname{Min}\left\{j: j>\frac{\log (1-U)}{\log (q)}\right\}
  \end{aligned}
  $$
若用记号$\operatorname{Int}(x)$表示“不大于$x$的最大整数”，则有：
$$
X=\operatorname{Int}\left(\frac{\log (1-U)}{\log (q)}\right)+1
$$
其等价于：
$$
\boxed{X \equiv \operatorname{Int}\left(\frac{\log (U)}{\log (q)}\right)+1}
$$
## 4.2 生成泊松分布随机变量 [[Generating Poisson Random Variable from U(0,1)]]
### 模型简述：

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
## 4.3 生成二项分布随机变量 [[Generating Binomial Random Variable from U(0,1)]]

### 模型概述：
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