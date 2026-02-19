---
aliases: [连续随机变量生成, Generating Continuous Random Variables, 逆变换法, Inverse Transform, 接受拒绝法, Rejection Method]
tags:
  - method
  - math/probability
  - stat/computational
related_concepts:
  - "[[Probability_Distribution_Change_of_Variables]]"
  - "[[Generating_Discrete_Random_Variables]]"
  - "[[Box_Muller_Transformation]]"
  - "[[Normal_Distribution]]"
  - "[[Exponential_Distribution]]"
source: "计算统计学"
---

# 连续随机变量的生成

#ComputationalStatistics 

## 5.1 逆变换法 Inverse Transform Algorithm
### 原理：
若$U$为$(0,1)$区间随机数，则任意连续型随机变量可以通过下式确定：
$$X=F^{-1}(U)$$
### 例5b [[Generating Exponential Random Variable from U(0,1)]]
要求：生成随机变量$X\sim\mathrm{exp}(\lambda)$，即：
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
```
## 5.2 接受拒绝法 Rejection Method

### 模拟目的：
通过辅助分布$g(x)$生成服从较复杂的密度函数$f(x)$的随机变量.
### 模拟方法：
- 首先确定一个辅助的“建议分布”$Y$，已知其概率密度函数为$g_Y(y)$，用来产生候选样本；(理论上$Y$可服从任意分布，而在实际计算中通常采取与目标分布$f(x)$形状较为接近的分布)
- 另生成一个$U(0,1)$用于后续比较；
- 计算一个常数$c$，使得对于$\forall x$，都有$f(x)/g(x)\leq c$. (为了计算方便，常常选择满足条件的$c$中的最小值)
- 若不等式$U \leq \frac {f(Y)}{cg(Y)}$成立，则接受 $Y$（令$X=Y$），否则则重新生成进行比较。

**算法实现：**
> STEP 1: Generate $Y$ having density $g$.
> STEP 2: Generate a random number $U$.
> STEP 3: If $U \leqslant \frac{f(Y)}{c g(Y)}$, set $X=Y$. Otherwise, return to Step 1 .
### 原理：
令 $\mathrm{X}$ 为想要生成的指定分布的随机数, 令 $\mathrm{N}$ 为必要迭代次数：
$$
\begin{aligned}
P\{X \leq x\} &=P\left\{Y_N \leq x\right\} \\
&=P\{Y \leq x \mid U \leq f(Y) / \operatorname{cg}(Y)\} \\
&=\frac{P\{Y \leq x, U \leq f(Y) / \operatorname{cg}(Y)\}}{K} \\
&=\frac{\int P\{Y \leq x, U \leq f(Y) / \operatorname{cg}(Y) \mid Y=y\} g(y) d y}{K} \\
&=\frac{\int_{-\infty}^x(f(y) / \operatorname{cg}(y)) g(y) d y}{K} \\
&=\frac{\int_{-\infty}^x f(y) d y}{K c}
\end{aligned}
$$
其中 $\mathrm{K}=\mathrm{P}(\mathrm{U} \leq f(Y) / \operatorname{cg}(Y)\}$. 令 $x \rightarrow \infty$ 可知 $\mathrm{K}=1 / c$ 证毕.
> 注：在每次循环判断时，若$U>f/cg$，则在下一次循环时事实上可以不再重新生成随机数，而是可以令$\frac{U-f(Y) / \operatorname{cg}(Y)}{1-f(Y) / \operatorname{cg}(Y)}=\frac{c U g(Y)-f(Y)}{c g(Y)-f(Y)}$作为下一次的$U$以减少计算。

>[!NOTE] 
> - 该方法由Von Neumann创造，其中的$Y$为$(a,b)$区间的均匀分布；
> - 由于每次接受的概率为：$P(U\leq f(Y)/cg(Y))=1/c$，故平均循环次数的几何平均为$c$. 
> - 在循环中若拒绝，即$U>f(Y)/cg(Y)$，此时并不需要重新生成随机数，而是可以通过下面的公式直接利用先前拒绝的$U$计算出新的随机数，以减少运算量：
$$ \frac{U-f(Y)/cg(Y)}{1-f(Y)/cg(Y)}=\frac{cUg(Y)-f(Y)}{cg(Y)-f(Y)}$$
### 例5d
要求：
生成随机变量$X$服从：
$$f(x)=20x(1-x)^3$$
解：
令
$$g(x)=1,~~0<x<1$$
下求解最优$c$：
已知：
$$ \frac{f(x)}{g(x)}=20x(1-x)^3$$
通过求导可知上式的极大值点为$x=1/4$，极大值为$135/64$
故$c=135/64$
因此有：
$$\frac{f(x)}{cg(x)}=\frac{256}{27}x(1-x)^3$$
下开始模拟过程：
生成随机数$U_1,U_2$；
若$U_2 \leqslant \frac{256}{27} U_1\left(1-U_1\right)^3$则接受，令$X=U_1$，否则重复上述操作。
### 例5f
要求：生成标准正态随机数$Z\sim N(0,1)$
解：
先考虑$X=|Z|$的分布，即：
$$
f(x)=\frac{2}{\sqrt{2 \pi}} e^{-x^2 / 2} \quad 0<x<\infty
$$
令$g(x)$为$\mathrm{exp}(1)$的概率密度函数，故有：
$$
\frac{f(x)}{g(x)}=\sqrt{2 / \pi} e^{x-x^2 / 2}
$$
可求其最大值得到最优的$c$值：
$$c=\mathrm{max}\frac{f(x)}{g(x)}=\sqrt{\frac{2e}{\pi}}$$
故有：
$$
\begin{aligned}
\frac{f(x)}{c g(x)} &=\exp \left\{x-\frac{x^2}{2}-\frac{1}{2}\right\} \\
&=\exp \left\{-\frac{(x-1)^2}{2}\right\}
\end{aligned}
$$
因此可以按下过程生成随机数：
STEP 1: Generate $Y$, an exponential random variable with rate 1.
STEP 2: Generate a random number $U$.
STEP 3: If $U \leqslant \exp \left\{-(Y-1)^2 / 2\right\} \star$, set $X=Y$. Otherwise, return to Step 1 .
在生成了绝对值正态分布后，我们可以令$Z$以相等的概率等于$X$或$-X$，即有标准正态函数的分布。
改进：
对上述$\star$式左右取对数，有：
$$
-\log U \geqslant(Y-1)^2 / 2
$$
根据计算又知$-\log U$服从$\mathrm{exp}(1)$分布，故算法可改进为：
STEP 1: Generate $Y_1$, an exponential random variable with rate 1.
STEP 2: Generate $Y_2$, an exponential random variable with rate 1 .
STEP 3: If $Y_2-\left(Y_1-1\right)^2 / 2>0$, set $Y=Y_2-\left(Y_1-1\right)^2 / 2$ and go to Step 4 .
Otherwise, go to Step 1.
STEP 4: Generate a random number $U$ and set
$$
Z=\left\{\begin{array}{ccc}
Y_1 & \text { if } & U \leqslant \frac{1}{2} \\
-Y_1 & \text { if } & U>\frac{1}{2}
\end{array}\right.
$$
通过生成标准正态$Z$，其余正态函数可以通过$\mu+\sigma Z$生成.