---
aliases: [指数分布随机数生成, Generating Exponential Random Variable]
tags:
  - method
  - math/probability
  - stat/computational
related_concepts:
  - "[[Exponential_Distribution]]"
  - "[[Generating_Continuous_Random_Variables]]"
  - "[[Probability_Distribution_Change_of_Variables]]"
source: "计算统计学"
---

# 从 U(0,1) 生成指数分布随机变量

#ComputationalStatistics 

## 目标

生成随机变量$X\sim\mathrm{exp}(\lambda)$，即：
$$
  F(x)=1-e^{-\lambda x}
  $$
**解：**
令
$$
u=F(x)=1-e^{-\lambda x}
$$
从上式中解出$x$有：
$$
x=-\frac1\lambda \log (1-u)
$$
等价于：
$$
\boxed {X=-\frac1\lambda \log (u)}
$$
**代码实现：**
```matlab
% set up the parameters. 
lam =2;
% generate the rv's
uni=rand(1,n);
X=-log(uni)/lam;