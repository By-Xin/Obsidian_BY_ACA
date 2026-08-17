---
aliases: [Box-Muller变换, Box-Muller Transformation, 极坐标法, Polar Method, 正态随机数生成]
tags:
  - method
  - math/probability
  - stat/computational
related_concepts:
  - "[[Normal_Distribution]]"
  - "[[Probability_Distribution_Change_of_Variables]]"
  - "[[Generating_Continuous_Random_Variables]]"
  - "[[Jacobian_Matrix]]"
source: "计算统计学"
---

# Box-Muller 变换：极坐标法生成正态随机变量

#ComputationalStatistics 

## 基本知识：
### 极坐标：
$R^2=X^2+Y^2$
$\tan\Theta = \frac Y X$
### 正态分布：
由于$X,Y$独立，其联合概率密度函数为
$$
\begin{aligned}
f(x, y) &=\frac{1}{\sqrt{2 \pi}} e^{-x^2 / 2} \frac{1}{\sqrt{2 \pi}} e^{-y^2 / 2} \\
&=\frac{1}{2 \pi} e^{-\left(x^2+y^2\right) / 2}
\end{aligned}
$$

### 函数变换：
$$P(X\in C, Y\in D)=\iint_{X\in C, Y\in D}f(x,y)dxdy=\iint_{u\in C^\prime, v\in D^\prime} f(g_1(u,v),g_2(u,v))|J|dudv$$

## Box-Muller变换
- 将$X,Y$的联合密度函数转化到极坐标系中. 令$d=x^2+y^2, \theta=\arctan(y/x)$,故：
$$
f(d, \theta)=\frac{1}{2} \frac{1}{2 \pi} e^{-d / 2}, \quad 0<d<\infty, 0<\theta<2 \pi
$$

- 注意到，上述密度函数可以认为是均值为2的指数分布($\frac12e^{-d/2}$)与$(0,2\pi)$的均匀分布的密度函数($\frac1{2\pi}$)乘积, 故有：
	- $R^2$与$\Theta$彼此独立
	- $R^2$服从均值为2的指数分布
	- $\Theta$服从$(0,2\pi)$的均匀分布

**故可以按如下步骤以极坐标法生成正态分布：**
> STEP1: 生成随机数$U_1, U_2$
> 
> STEP2: 令$R^2=-2\log U_1, ~~ \Theta=2\pi U_2$
> 
> STEP3: 令
> $$
 \begin{aligned}
 &X=R \cos \Theta=\sqrt{-2 \log U_1} \cos \left(2 \pi U_2\right) \\
 &Y=R \sin \Theta=\sqrt{-2 \log U_1} \sin \left(2 \pi U_2\right)
 \end{aligned} \quad(\star)
 $$

> 说明：Box-Muller变换在计算时的效率较低，这是因为其中在STEP3中涉及到了三角函数$\cos\sin$的计算（而这一计算耗时较长）。为了改进这一特点，下不再生成随机角度$\Theta$，而是直接通过模拟直角三角形的三边长度生成随机三角函数$\cos\Theta, \sin\Theta$，具体方法如下：
## 极坐标法：B-M变化的改进*
- 改进思路：不再计算模拟的随机角度的三角函数，而是通过模拟单位圆（面）直接计算三角函数的具体数值。
- 改进步骤：
	- 引入单位圆
	- 若$U\sim(0,1)$，则$2U-1\sim(-1,1)$，故令
$$\begin{aligned}
&V_1=2 U_1-1 \\
&V_2=2 U_2-1
\end{aligned}$$

- 不断生成随机数对$(V_1,V_2)$并保留满足$V_1^2+V_2^2\leq 1$的部分，则有$(V_1,V_2)$在如下图所示的单位圆上均匀分布：
		- 对于该随机数对$(V_1,V_2)$对应的极坐标方程，可知其对应的$R^2$服从$(0,1)$的均匀分布，而$\Theta$服从$(0,2\pi)$的均匀分布。
	- 模拟$\cos\sin$：
$$
\begin{aligned}
&\sin \Theta=\frac{V_2}{R}=\frac{V_2}{\left(V_1^2+V_2^2\right)^{1 / 2}} \\
&\cos \Theta=\frac{V_1}{R}=\frac{V_1}{\left(V_1^2+V_2^2\right)^{1 / 2}}
\end{aligned}
$$


- 对B-M的改进: 将上述模拟的$\sin\Theta,\cos\Theta$代入B-M中的$(\star)$，有
$$
\begin{aligned}
&X=(-2 \log U)^{1 / 2} \frac{V_1}{\left(V_1^2+V_2^2\right)^{1 / 2}} \\
&Y=(-2 \log U)^{1 / 2} \frac{V_2}{\left(V_1^2+V_2^2\right)^{1 / 2}}
\end{aligned}
$$

- 再令 $S=R^2$，则有：
$$
\begin{aligned}
&X=(-2 \log S)^{1 / 2} \frac{V_1}{S^{1 / 2}}=V_1\left(\frac{-2 \log S}{S}\right)^{1 / 2} \\
&Y=(-2 \log S)^{1 / 2} \frac{V_2}{S^{1 / 2}}=V_2\left(\frac{-2 \log S}{S}\right)^{1 / 2}
\end{aligned}
$$

综上，可知新的模拟步骤为：
> STEP1: 生成随机数$U_1,U_2$
> 
> STEP2: 令 $V_1=2U_1-1, V_2=2U_2-1, S=V_1^2+V_2^2$
> 
> STEP3: 若$S>1$，返回STEP1
> 
> STEP4: 否则可按如下原则生成一对标准正态分布：
> 
> $$
 X=\sqrt{\frac{-2 \log S}{S}} V_1, \quad Y=\sqrt{\frac{-2 \log S}{S}} V_2
 $$

