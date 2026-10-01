#brainotes 
* 下一步的工作如下. 基本的 blog 分为如下几个层次:
	* 从微观层面, 我们希望针对 ReHLine 这个形式的优化结构构建一个可微分的求解器, 加速其微分过程. 
	* 从中观层面, 我们希望是能够解决这样一个 loss 问题 (rehline 原文的建模), 去解决例如回归等问题. (也就是如何将上面的纯优化问题, 尤其是这里的 $x$ 成功的翻译过度成 residual ) (即 inner loss)
	* 从宏观层面, 一旦我们实现了这样的求导路径, 我们便可以构建一个 bi-level 的求导过程, 可以通过 SGD 进一步对超参数进行优化,. 
- 提到几个类似但又不同的工作: 一部分是通过离散的类似遗传算法等进行超参数优化, 另一部分是更 focus on 数据加权等工作.  此处进一步参考 XJTU 孟DY 老师的相关工作. [待查找]
- 我们的一个 intuition 是和 rehline 一致的. 对于任何的 convex function (也就是 loss 的最基本形式), 我们都可以通过 rehline 进行近似. 所以我们希望对 rehline 提出更好的一种微分策略. 我们目前实现了纯 relu 的部分. 这相当于是一种一阶近似. 

---
+ 考虑复现一下戴老师的论文
+ 我们要求的并不是 $\min \frac12 z^\top Q z + q^\top Z$ 的导数, 而是这个 $\arg\min$ 的话本质上是一个 $z = f(Q)$, 我们希望考虑的是这个函数的导数, 也就是 $z = f(Q)$ 的导数. 尤其是对于rehline 有特殊的结构就有特殊的信息
+ 不等式约束? _这个看一下 KKT 应该是 kkt 中的结论_
+ dz, dlam 的 d 是微分嘛? 不是, 这个是变量, 类似于 cache 一下以备后用
+ rehline 的 kkt 的推导 (rehline thm 2)
+ 将
	$$ F(\omega^*, \theta) = \begin{bmatrix} F_1 \in \mathbb{R}^n:= Q_{(n\times n)}z_{(n)}^* + q_{(n)} + (A_{(m_\text{eq}\times n)})^\top \nu_{(m_\text{eq})}^* + (G_{(m_\text{ineq}\times n)})^\top \lambda_{\text{lgrg} (n_\text{ineq})}^* \\ F_2\in\mathbb{R}^{m_\text{eq}}:=A_{(m_\text{eq}\times n)}z_{(n)}^* - b_{(m_\text{eq})} \\ F_3\in\mathbb{R}^{m_\text{ineq}}:=D(\lambda_{\text{lgrg}}^*)_{(m_\text{ineq})}(G_{(m_\text{ineq}\times n)}z_{(n)}^* - h_{(m_\text{ineq})}) \end{bmatrix} = 0 $$
	写成UVST 的表达式, 求出 $\partial F/ \partial \theta$的导数
+ 把QA 表达式带进去, 善用分块矩阵求逆, 看看有没有什么能化简的
+ 把 rehline 7 变成 note (2) 中标准型. 从 toy 开始 , 先考虑 gamma (只有relu, 没有 rehu, 没有约束 Abeta+b>=0), 只有 UV 没有 ST. 写出 rehline 的 (7) (8) 的表达式

* 现在整体推导基本上顺下来了, 但是回过头来突然懵住有一个比较 naive 的问题、、、目前我们这个形式下标签 $y$ 是怎么建模的? 正常按照前面做 autoloss 时候的实验原理来说应该是
	$$ \ell^{(i)}(r^{(i)};\boldsymbol{\beta}) = \sum_{l=1}^L \text{ReLU}(U_l^{(i)} r^{(i)} + V_l^{(i)}) + \sum_{h=1}^H \text{ReHU}_\tau(S_h^{(i)} r^{(i)} + T_h^{(i)}) + \lambda \|\boldsymbol{\beta}\|_2^2 \tag{1} $$

	其中 $r^{(i)} = y^{(i)} - \mathbf{x}^{(i) \top} \boldsymbol{\beta}$.
	
	但是如果按照目前的 rehline 中的形式的话, 目前对于
	
	$$ \min_{\boldsymbol{\beta} \in \mathbb{R}^d} \left\{ \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}(u_{li} \mathbf{x}_i^\top \boldsymbol{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H \text{ReHU}_{\tau_{hi}}(s_{hi} \mathbf{x}_i^\top \boldsymbol{\beta} + t_{hi}) + \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 \right\} \quad \\ \text{s.t. } \mathbf{A}\boldsymbol{\beta} + \mathbf{b} \geq 0 \tag{2} $$
	
	这个优化问题整体的去进行求对偶求梯度这些我都能理解, 但是有点想不明白我们该如何处理标签信息. 我们后面进行推导的时候是按照 (1) 还是 (2) 的形式? 还是说这里面哪里的理解有问题.
