---
aliases: [信息论, 熵, KL散度, 交叉熵, Shannon Entropy, Information Theory, 香农熵]
tags:
  - concept
  - math/probability
  - ml/stats
related_concepts:
  - "[[Cross Entropy Loss]]"
  - "[[Maximum Likelihood Estimation]]"
---

# Information Theory

> Refs: 
>
> 1. 【"交叉熵"如何做损失函数？打包理解"信息量"、"比特"、"熵"、"KL散度"、"交叉熵"】 [https://www.bilibili.com/video/BV15V411W7VB/?share_source=copy_web&vd_source=9471c7cd3fca9ffedd9167aefed57c6d](https://www.bilibili.com/video/BV15V411W7VB/?share_source=copy_web&vd_source=9471c7cd3fca9ffedd9167aefed57c6d)  
> 2. CMU Advanced Probability II.  [https://www.stat.cmu.edu/~cshalizi/754/2006/notes/all.pdf](https://www.stat.cmu.edu/~cshalizi/754/2006/notes/all.pdf)
>

## 信息量与 Bit
信息量某种意义上是对事件发生的概率的度量. 太阳每天升起, 我们常认为这是一个必然事件 (以概率1发生), 因此该事件也就几乎不会带来任何信息量. 反而言之, 当一个小概率事件发生 (如彩票中了头奖), 那么这个事件就会带来很大的信息量.

因此我们将信息量定义为事件发生的概率的倒数之对数, 即:

$$ I(x) = \log_2 \frac{1}{\mathbb{P}(x)} = -\log_2 \mathbb{P}(x) $$

这里, 使用对数函数的一个理由是其具有可加性, 可以将一系列事件发生的概率乘法转化为信息量的加法. 例如, 如果事件 $x$ 和 $y$ 独立发生, 则它们的信息量之和为:

$$ I(x, y) = I(x) + I(y) = -\log_2 \mathbb{P}(x) - \log_2 \mathbb{P}(y) = -\log_2 (\mathbb{P}(x) \cdot \mathbb{P}(y)) $$

而对于以 2 为底的对数定义的信息量单位称为比特 (bit). 这也是在二进制的视角中对信息量的度量. 在其他进制下, 信息量的单位可以是纳特 (nat, 以自然对数为底) 或者哈特利 (hartley, 以 10 为底).

试想一个占用 4 bit 的二进制文件, 其可能的内容有 $2^4 = 16$ 种不同的组合. 因此其相当于一个概率为 $\frac{1}{16}$ 的均匀分布. 而当我们确定了文件的内容, 则也就可以反过来说我们获得了 4 bit 的信息量. 这就是信息量与比特的关系.

## Shannon Entropy (香农熵)
### 定义
信息论的起点是香农 (Shannon 1948) 的奠基工作, 它开创了将信息处理视为随机过程的方法. 信息论的核心是熵, 通常指一个随机变量(或其分布)所具有的不确定性程度. 为了与其他类型的熵(如热力学熵)区分, 这里特指 Shannon entropy (香农熵). 直觉上, 它描述了我们在不知道随机变量取值前, 对其结果的不确定性. 

Shannon Entropy 被定义为上述信息量的期望值. 以离散随机变量 $X$ 的概率分布 $\mathbb{P}(X)$ 为例, 香农熵定义为:

$$ H[X] \triangleq - \mathbb{E}[\log_2 \mathbb{P}(X)] = - \sum_{x \in X} \mathbb{P}(x) \log_2 \mathbb{P}(x) $$

直观看, 对于一个概率1事件, 我们没有任何的不确定性, 因此其熵为 0. 而对于一个均匀分布的随机变量, 其熵最大, 因为我们对每个可能的结果都没有偏好, 不确定性最高.

### 联合熵与条件熵
类似地, 还可以利用概率论的知识马上得到联合熵的定义:

$$ \begin{align*}
H[X, Y] &\triangleq - \mathbb{E}[\log_2 \mathbb{P}(X, Y)] \\
&= - \sum_{x \in X, y \in Y} \mathbb{P}(x, y) \log_2 \mathbb{P}(x, y)
\end{align*} $$

以及条件熵:

$$\begin{aligned}
H[X | Y] &\triangleq  \mathbb{E}_{X,Y}[-\log_2 \mathbb{P}(X | Y)] \quad{\small\text{(条件熵是重期望)}}\\
&= \mathbb{E}_Y \left[ \mathbb{E}_{X|Y}[-\log \mathbb{P}(X|Y)] \right]\\
&= \sum_y \mathbb{P}(Y = y) H[X|Y = y]\\
&= \sum_y \mathbb{P}(Y = y) \left( -\sum_x \mathbb{P}(X = x \mid Y = y) \log_2 \mathbb{P}(X = x \mid Y = y) \right)\\
&= - \sum_{x,y} \mathbb{P}(X = x, Y = y) \log_2 \mathbb{P}(X = x \mid Y = y)\\
\end{aligned} $$

+ 之所以条件熵是重期望, 是因为我们希望的条件熵是一个具体的期望取值, 而不是一个函数. 因为若只取一次期望, 我们得到的是 $H[X | Y=y]$, 相当于对 $Y$ 的某个具体取值 $y$ 下的 $X$ 的局部熵. 而我们真正希望得到的是这个不确定性的全局平均, 因此还要进一步对 $Y$ 取期望.

可以证明, 上述三种熵满足以下恒等式 (很类似全概率公式的加法版本):

$$ H[X, Y] \equiv H[Y] + H[X | Y] $$

> _证明:_
>
> 对于联合熵 $H[X, Y]$, 我们有:
>
> $$ H[X, Y] = - \sum_{x, y} \mathbb{P}(x, y) \log_2 \mathbb{P}(x, y) $$
>
> 将其中的 $\mathbb{P}(x, y)$ 展开为 $\mathbb{P}(x, y)=\mathbb{P}(y) \mathbb{P}(x | y)$, 则有:
>
> $$ \begin{align*}
H[X, Y] &= - \sum_{x, y} \mathbb{P}(y) \mathbb{P}(x | y) \log_2 (\mathbb{P}(y) \mathbb{P}(x | y)) \\
&= - \sum_{x, y} \mathbb{P}(y) \mathbb{P}(x | y) \left( \log_2 \mathbb{P}(y) + \log_2 \mathbb{P}(x | y) \right) \\
&= - \sum_{x,y} \mathbb{P}(y) \mathbb{P}(x | y) \log_2 \mathbb{P}(y) - \sum_{x, y} \mathbb{P}(y) \mathbb{P}(x | y) \log_2 \mathbb{P}(x | y) \\
&= - \sum_{y}\left(\sum_x\mathbb{P}(x,y)\right) \log_2 \mathbb{P}(y) - \sum_{x,y} \mathbb{P}(x,y) \log_2 \mathbb{P}(x | y) \\
&= -\sum_y \mathbb{P}(y) \log_2 \mathbb{P}(y) - \sum_{x,y} \mathbb{P}(x,y) \log_2 \mathbb{P}(x | y)  \\
&= H[Y] + H[X | Y]
\end{align*} $$
>

### Shannon 熵的性质
不加证明地列出一些 Shannon 熵的性质:

1. $H[X] \geq 0$ (熵非负性): 熵的值总是非负的, 因为信息量是对概率的对数, 而概率总是介于 0 和 1 之间.
2. $H[X] = 0 \Leftrightarrow \exists \, x_0 \in X: \mathbb{P}(X = x_0) = 1$ (熵为零的条件): 当随机变量 $X$ 只有一个确定的取值时, 熵为零.
3. $H[X] \leq \log_2 |X|$ (均匀分布的熵): 香农熵的最大值为当随机变量 $X$ 均匀分布时, 即每个取值的概率相等. 在这种情况下, 熵等于取值个数的对数.
4. $H[X] + H[Y] \geq H[X, Y]$ (subadditivity, 子加性): 联合熵不超过各自熵的和. 这反映了联合分布的不确定性总是小于或等于单个分布的不确定性之和. 当且仅 $X$ 和 $Y$ 独立时, 等式成立.
5. $H[X, Y] \ge H[X]$ (联合熵的下界): 联合熵总是大于或等于单个变量的熵. 这反映了联合分布的不确定性总是大于或等于单个分布的不确定性.
6. $H[X | Y] \leq H[X]$ (条件熵的下界): 条件熵总是不超过单个变量的熵. 这反映了当给出额外的信息 $Y$ 时, 往往可以减少对 $X$ 的不确定性, 至少不会增加. 当且仅当 $X$ 和 $Y$ 独立时, 等式成立, 此时 $Y$ 对 $X$ 没有任何信息增益.

## KL 散度 (Kullback-Leibler Divergence)  与交叉熵
### KL 散度的定义
Shannon 熵在连续空间中存在一些技术上的问题. KL 散度作为一个更稳定和相对的信息度量常被优先使用. KL 散度可以看作是两个概率分布之间的距离度量, 但它并不是一个真正的距离, 因为它不满足对称性和三角不等式. 

这里给出在离散分布下的 KL 散度定义. 考虑在同一可数样本空间 $\mathcal{X}$ 上的两个概率分布 $\mathrm{P}(X)$ 和 $\mathrm{Q}(X)$, KL 散度定义为:

$$ D_{KL}(\mathrm{P} || \mathrm{Q}) = \sum_{x \in \mathcal{X}} \mathrm{P}(x) \log_2 \frac{\mathrm{P}(x)}{\mathrm{Q}(x)} $$

该式仅在 $\mathrm{P}(x) > 0$ 必然推出 $\mathrm{Q}(x) > 0$ 时 (即 $\mathrm{P} \ll \mathrm{Q}$) 有定义. 

+ 直觉上, KL 散度衡量了用 $\mathrm{Q}$ 来近似 $\mathrm{P}$ 时的平均信息损失. 换言之, 若我们用 $\mathrm{Q}$ 来编码或预测真实分布 $\mathrm{P}$, 则每次将损失约 为 $D_{KL}(\mathrm{P} || \mathrm{Q})$ 比特的信息量.
+ 注意由于 $\log{\mathrm{P}/\mathrm{Q}} \neq \log{\mathrm{Q}/\mathrm{P}}$, 因此 KL 散度是非对称的, 即 $D_{KL}(\mathrm{P} || \mathrm{Q}) \neq D_{KL}(\mathrm{Q} || \mathrm{P})$.

### KL 散度与交叉熵
整理 KL 散度的表达式, 我们有:

$$ \begin{align*}
D_{KL}(\mathrm{P} || \mathrm{Q}) &=\sum_{x \in \mathcal{X}} \mathrm{P}(x) \log_2 \frac{\mathrm{P}(x)}{\mathrm{Q}(x)} \\
&= \sum_{x \in \mathcal{X}} \mathrm{P}(x) \log_2 \mathrm{P}(x) - \sum_{x \in \mathcal{X}} \mathrm{P}(x) \log_2 \mathrm{Q}(x)\\
&= H[\mathrm{P}] - \mathbb{E}_{\mathrm{P}}[\log_2 \mathrm{Q}(X)]\\
&\triangleq -H[\mathrm{P}] + H(\mathrm{P}, \mathrm{Q})
\end{align*} $$

+ 其中 $H(\mathrm{P}, \mathrm{Q})$ 表示用 $\mathrm{Q}$ 来编码 $\mathrm{P}$ 的熵, 即交叉熵: 

$$ H(\mathrm{P}, \mathrm{Q}) = -\sum_{x \in \mathcal{X}} \mathrm{P}(x) \log_2 \mathrm{Q}(x) $$

+ 注意区分交叉熵 $H(\mathrm{P}, \mathrm{Q})$ 和联合熵 $H[\mathrm{P}, \mathrm{Q}]$, 前者是用 $\mathrm{Q}$ 来编码 $\mathrm{P}$ 的熵, 表达式如上; 后者是联合分布的熵, 表达式为 $H[\mathrm{P}, \mathrm{Q}] = -\sum_{x,y} \mathrm{P}(x,y) \log_2 \mathrm{P}(x,y)$.

综上, 我们有:

$$ H(\mathrm{P}, \mathrm{Q}) = H[\mathrm{P}] + D_{KL}(\mathrm{P} || \mathrm{Q}) $$

+ $H[\mathrm{P}]$ 是真实分布 $\mathrm{P}$ 的熵, 描述了 $\mathrm{P}$ 本身的不确定性/平均信息量.
+ $H(\mathrm{P}, \mathrm{Q})$ 是用 $\mathrm{Q}$ 来编码 $\mathrm{P}$ 的熵, 描述了在 $\mathrm{Q}$ 的假设下对 $\mathrm{P}$ 的编码效率 (即平均编码长度/信息量).
+ $D_{KL}(\mathrm{P} || \mathrm{Q})$ 则是模型 $\mathrm{Q}$ 的错误导致的额外编码长度 (即信息损失), 衡量了 $\mathrm{Q}$ 对 $\mathrm{P}$ 的近似程度.

> _Collary:_
>
> KL 散度在模型空间的 Hessian 矩阵就等于 Fisher Information! (KL 散度在局部像一个二次型距离, 而 Fisher Information 描述了这个模型空间的曲率.)
>

### KL 散度与均匀分布
考虑样本空间 $\mathcal{X} = \{1,2,\cdots, n\}$ 上的均匀分布, 其概率分布为 $\mathrm{U}(x) = \frac{1}n$, 以及真实分布 $\mathrm{P}(x)$. 我们可以定义 KL 散度为:

$$ \begin{align*}
D_{KL}(\mathrm{P} || \mathrm{U}) 
&= \sum_{x \in \mathcal{X}} \mathrm{P}(x) \log_2 \frac{\mathrm{P}(x)}{\mathrm{U}(x)} \\
&= \sum_{x \in \mathcal{X}} \mathrm{P}(x) \log_2 \mathrm{P}(x) + \sum_{x \in \mathcal{X}} \mathrm{P}(x) \log_2 n\\
&= -H[\mathrm{P}] + \log_2 n
\end{align*} $$

因此整理得到:

$$ H[\mathrm{P}] = \log_2 n - D_{KL}(\mathrm{P} || \mathrm{U}) $$

其可以理解为, 真实分布 $\mathrm{P}$ 的熵等于最大可能熵 (即均匀分布的熵) 减去该分布偏离均匀分布的 KL 散度. 这也说明了 KL 散度可以看作是对熵的一个修正, 衡量了真实分布与均匀分布之间的差异.

## 分类问题中的交叉熵, KL 散度与极大似然
考虑一个具体的多分类建模问题. 考虑输入特征 $x\in \mathcal{X}$ 和输出标签 $y\in \mathcal{Y} = \{1, 2, \ldots, K\}$. 其真是的分布为 $\mathrm{P}(y|x)$, 但我们并不知道这个分布. 我们的目标是通过训练数据 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 来得到一个分布的估计 $\mathrm{Q}_\theta(y|x)$, 其中 $\theta$ 是模型的参数.

**从统计学的视角**, 很自然的我们希望通过最大化训练数据的似然函数来估计参数 $\theta$. 似然函数定义为:

$$ \mathcal{L}(\theta) = \prod_{i=1}^N \mathrm{Q}_\theta(y_i | x_i) $$

因此其对数似然函数为:

$$ \log \mathcal{L}(\theta) = \sum_{i=1}^N \log \mathrm{Q}_\theta(y_i | x_i) $$

另一方面, **在机器学习中**我们定义交叉熵损失函数为:

$$ \mathcal{L}_{\mathrm{CE}}(\theta) = -\sum_{i=1}^N \sum_{k=1}^K \mathbb{I}(y_i = k) \log \mathrm{Q}_\theta(k | x_i) $$

+ 其中 $\mathbb{I}(y_i = k)$ 是指示函数, 当 $y_i = k$ 时为 1, 否则为 0.这和上面的对数似然函数是等价的, 因为 $\sum_{k=1}^K \mathbb{I}(y_i = k) \log \mathrm{Q}_\theta(k | x_i)$ 只在 $y_i$ 的真实类别 $k$ 上取值, 其他类别的项为 0.

再**从信息论的视角**, 我们可以将数据认为服从一个经验分布 $\mathrm{P}_{\text{emp}}$, 其在每一个样本 $(x_i, y_i)$ 上的质量集中在真实标签 $y_i$ 上:

$$ \mathrm{P}_{\text{emp}}(y|x_i) = \mathbb{I}(y = y_i) =
\begin{cases}
1 & \text{if } y = y_i \\
0 & \text{otherwise}
\end{cases} $$

而 $\mathrm{Q}_\theta(y|x)$ 则是我们模型的预测分布, 是一种对 $\mathrm{P}_{\text{emp}}$ 的近似.  
因此, 我们可以将交叉熵损失函数理解为:

$$ \begin{align*}
H(\mathrm{P}_{\text{emp}}, \mathrm{Q}_\theta) &= \mathbb{E}_{\mathrm{P}_{\text{emp}}}[-\log \mathrm{Q}_\theta(y|x)] \\
&= -\sum_{i=1}^N \mathrm{P}_{\text{emp}}(y_i|x_i) \log \mathrm{Q}_\theta(y_i|x_i) \\
&= -\sum_{i=1}^N \mathbb{I}(y_i = y) \log \mathrm{Q}_\theta(y|x_i) \\
\end{align*} $$

这也和上面的交叉熵损失函数定义一致.

此外, 由于 KL 散度的定义为:

$$ D_{KL}(\mathrm{P}_{\text{emp}} || \mathrm{Q}_\theta) = - H(\mathrm{P}_{\text{emp}}) + H(\mathrm{P}_{\text{emp}}, \mathrm{Q}_\theta) $$

故其中 $H(\mathrm{P}_{\text{emp}})$ 是一个与 $\theta$ 无关的常数项, 因此最小化 KL 散度等价于最小化交叉熵损失函数:

$$ \nabla_\theta D_{KL}(\mathrm{P}_{\text{emp}} || \mathrm{Q}_\theta) = \nabla_\theta H(\mathrm{P}_{\text{emp}}, \mathrm{Q}_\theta) $$

