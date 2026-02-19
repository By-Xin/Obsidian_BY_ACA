#StatisticalLearning 

在上文GDA的假设中，我们所处理的数据$y$是离散的0-1形式，而$x$是连续的特征指标；这里Naive Bayes我们将处理$x$同样为离散的情景.
- 本节将以Spam Mail Filter为例. 其中$y\in\{0,1\}$ 表示否/是Spam Mail；$x$为一个0-1向量，每一个元素$x_i\in\{0,1\}$表示是否包含某个单词.

### 4.2.1 Naive Bayes 模型构造 (Multi-variate Bernouli Event Model)

#### Naive Bayes Assumption

- 作为Generative Model，我们要通过Bayes公式计算$P(y|x)$，而为了计算它我们就需要得到$P(x|y)$和$P(y)$. 

- 其中$P(x|y)$由于是离散的 (假设$x\in R^k$) 会有 $2^{k}-1$ 种可能的结果，这会造成维度爆炸. 

- 为简化模型，Naive Bayes做了一个强假设，称为*Naive Bayes Assumption*，而由此得到的模型就称为*Naive Bayes Classifier*.


**[Assumption]** 
*在给定$y$的条件下，各$x_i$彼此独立，即*
$$ p(x_1, \dots, x_k | y) = \prod_{i=1}^k p(x_i | y) $$

#### Naive Bayes 推导

**给出记号：**
- $\phi_{i|y=1} = p(x_i =1 | y=1 )$, $\phi_{i|y=0} = p(x_i =1 | y=0 )$
- $\phi_{y=1} = p(y=1)$
- Training Set: $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$

**得到联合似然函数：**
$$ \mathcal{L(\phi_y, \phi_{j|y=0},\phi_{j|y=1})} = \prod_{i=1}^m p(x^{(i)}, y^{(i)};\phi_y, \phi_{j|y=0},\phi_{j|y=1}) $$

**得到MLE的参数估计：**

$$ \phi_{j|y=1} = \frac{\sum_{i=1}^m 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 1\}}{\sum_{i=1}^m 1\{y^{(i)} = 1\}} \\ \phi_{j|y=0} = \frac{\sum_{i=1}^m 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 0\}}{\sum_{i=1}^m 1\{y^{(i)} = 0\}} \\ \phi_y = \frac{\sum_{i=1}^m 1\{y^{(i)} = 1\}}{m} $$

> 这也是自然的：$\phi_{j|y=1}$ 就是在所有$y=1$的样本中（如是spam的样本中）, 第$j$个特征取值为1的样本（出现单词j的）占比. $\phi_{j|y=0}$ 同理. $\phi_y$ 就是$y=1$的样本占总样本的比例.

**做出判断**：

判断是否是spam，也就是判断概率$p(y=1|x)$的大小，具体地，将得到的MLE参数带入判别式，有
$$ \begin{align*} p(y=1|x) &= \frac{p(x|y=1)p(y=1)}{p(x|y=1)p(y=1)+p(x|y=0)p(y=0)} \\ &= \frac{\prod_{i=1}^n p(x_i|y=1)p(y=1)}{\prod_{i=1}^n p(x_i|y=1)p(y=1)+\prod_{i=1}^n p(x_i|y=0)p(y=0)} \end{align*} $$

**注意！**

在上式中，如果在test set中出现了一个dictionary不存在的新单词，会出现问题。因为在这种情况下，$p(x|y=1) = p(x|y=0) = 0$，所以分子分母都是0.

下文的 *Laplace Smoothing* 可以解决这一问题。 

简而言之，得到的新的Naive Bayes Classifier 参数估计为：

$$ \phi_{i|y=1} = \frac{\sum_{j=1}^{m} 1\{x_i^{(j)} = 1 \wedge y^{(j)} = 1\} + 1}{\sum_{j=1}^{m} 1\{y^{(j)} = 1\} + 2} \\
\phi_{i|y=0} = \frac{\sum_{j=1}^{m} 1\{x_i^{(j)} = 1 \wedge y^{(j)} = 0\} + 1}{\sum_{j=1}^{m} 1\{y^{(j)} = 0\} + 2} $$

### 4.2.2 Laplace smoothing

**原始问题**：上文提到的$\frac00$问题可以概括为，在统计学意义上，我们直接将数据集中没有出现过的结果都认为其出现概率为0. 

具体而言，假设我们有一个有限样本容量$m$的数据集$\{z^{(1)},...,z^{(m)}\}$，每个样本$z^{(i)}$以概率$\phi_i = p(z=i)$取值为$\{1,...,k\}$的一个. Initially，对于参数$\phi_j$（$z$取第$j$种取值）的估计为：
$$ \phi_j = \sum_{i=1}^m \mathit{1}_{\{z^{(i)}=j\}}/m $$

#### Laplace Smoothing Formula
$$\phi_j = \frac{1+\sum_{i=1}^m \mathit{1}_{\{z^{(i)}=j\}}}{m+k} $$

回顾：这里的$k$是$z^{(i)}$有可能取到的取值个数

通过*Laplace Smoothing*变换的数据依然符合***归一化***，即$\sum_{j=1}^k \phi_j = 1$，且解决了$0/0$的问题。

### 4.2.3 Multinomial Event Model

对于同样的Spam Detection问题，我们采用另一种表示方法：

- 记$|V|$为词典的规模（含有的单词数量）, $n$为识别的email中的token总数
- 向量$x \in \mathbb{R}^{n}$表示一整个email，每个分量$x_{i}\in \{1,2,...,|V|\} $中下标$i$ 表示email按照行文顺序的第$i$个字符，其取值表示该位置的这个词在词典中的编号。

同样施加Naive Bayes Assumption：
$$ p(x,y) = p(x|y)p(y) = \prod_{i=1}^{n}p(x_i|y)p(y) \textit{ (by N.B. assum.)}$$

#### Spam Detection Example

 - $\phi_y = p(y=1) $
 - $\phi_{k|y=0} = p(x_j=k|y=0)$ ：在类别为0的情况下，email中第j个单词取到字典中第k个单词的概率【注意到这里的$\phi$并不依赖于$j$，这是在假设单词的出现内容与其所在的位置无关】


得到似然函数：

$$
\begin{aligned}
\mathcal{L}\left(\phi, \phi_{k \mid y=0}, \phi_{k \mid y=1}\right) & =\prod_{i=1}^m p\left(x^{(i)}, y^{(i)}\right) \\
& =\prod_{i=1}^m\left(\prod_{j=1}^{n_i} p\left(x_j^{(i)} \mid y ; \phi_{k \mid y=0}, \phi_{k \mid y=1}\right)\right) p\left(y^{(i)} ; \phi_y\right) .
\end{aligned}
$$

> *其中$m$为email数（training set样本数），$n_i$为第$i$个email中的token数。*

得到MLE：
$$
\begin{aligned}
\phi_{k \mid y=1} & =\frac{\sum_{i=1}^m \sum_{j=1}^{n_i} 1\left\{x_j^{(i)}=k \wedge y^{(i)}=1\right\}}{\sum_{i=1}^m 1\left\{y^{(i)}=1\right\} n_i} \\
\phi_{k \mid y=0} & =\frac{\sum_{i=1}^m \sum_{j=1}^{n_i} 1\left\{x_j^{(i)}=k \wedge y^{(i)}=0\right\}}{\sum_{i=1}^m 1\left\{y^{(i)}=0\right\} n_i} \\
\phi_y & =\frac{\sum_{i=1}^m 1\left\{y^{(i)}=1\right\}}{m} .
\end{aligned}
$$

该理解也是自然的：
- $\phi_{k|y=0}$ 表示我们遍历每一个email样本($i=1 \text{ to } m $)，再遍历每个email样本中的全部token，得到标记为spam的email且出现单词k的次数；再除以全部email中标记为spam的总的词汇数目。
- 其余参数也以此类推。
