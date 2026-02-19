#StatisticalLearning 
### 多元正态分布

- $\mathcal{N(\mu,\Sigma)}$ pdf：
    $$
    p(x ; \mu, \Sigma)=\frac{1}{(2 \pi)^{n / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\right)
    $$

### Gauss 判别原理

- 模型假设:
  - $y \sim  \text{Bernoulli}(\phi) \cdots *$
  - $x|y=0 \sim \mathcal{N}(\mu_0, \Sigma) \cdots \triangle$
  - $x|y=1 \sim \mathcal{N}(\mu_1, \Sigma) \cdots \triangle$

<br>

- 具体而言:
  - $p(y) = \phi^y(1-\phi)^{1-y}$
  - $p(x|y=0) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)\right)$
  - $p(x|y=1) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\right)$

<br>

- 对于全部$m$个样本，可以写出数据的 **（联合）对数似然:**

  $$ \begin{aligned}
  l(\phi, \mu_0, \mu_1, \Sigma) &= \log \prod_{i=1}^{m} p(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma) \\ 
  &= \log \prod_{i=1}^{m} \underbrace {p(x^{(i)}|y^{(i)}; \mu_0, \mu_1, \Sigma) }_{\triangle} \underbrace {p(y^{(i)}; \phi)}_{\ast} \textit{  (by cond. prob.)} 
  \\\end{aligned}$$

  其中：$\mu_0, \mu_1 \in \R^{n}, \Sigma \in \R^{n\times n}, \phi \in \R$

- 通过求偏导等求解对数极大似然，得到下列**参数估计：**

  $$
  \begin{aligned}
  \phi & =\frac{1}{m} \sum_{i=1}^m 1\left\{y^{(i)}=1\right\} \\
  \mu_0 & =\frac{\sum_{i=1}^m 1\left\{y^{(i)}=0\right\} x^{(i)}}{\sum_{i=1}^m 1\left\{y^{(i)}=0\right\}} \\
  \mu_1 & =\frac{\sum_{i=1}^m 1\left\{y^{(i)}=1\right\} x^{(i)}}{\sum_{i=1}^m 1\left\{y^{(i)}=1\right\}} \\
  \Sigma & =\frac{1}{m} \sum_{i=1}^m\left(x^{(i)}-\mu_{y^{(i)}}\right)\left(x^{(i)}-\mu_{y^{(i)}}\right)^T .
  \end{aligned}
  $$

  这些估计的含义也是自然的：
  - $\phi$：作为Bernoulli参数，表示$y=1$占总数的比例
  - $\mu_0/\mu_1$：表示$y=0/1$组中$x$分布的均值，其公式相当于求期望
  - $\Sigma$：表示$x$的协方差矩阵，因为两组label数据的方差是相同的，所以可以在一起求协方差

<BR>

- Empirically，我们可以直接通过得到的MLE参数公式直接得到估计结果（这也说明GDA是一个很有效率的算法）

- 说明：

  - 总结上述过程， GDA相当于分别对两个类别的数据进行了两套分布假设，通过标签$y$是Bernoulli分布的假定作为桥梁写出了似然函数，通过MLE的方法求出了对于两种类型数据的参数设置（注意这里边假定***分布的协方差矩阵是相等的***），只不过二者的均值中心不同。

  - 这也可以从下图中看到，两种类别的椭圆的大小形状是相似的，这说明了协方差矩阵是相等的；而彼此的中心点不同，说明了均值不同。不同的大小的椭圆等高线表示了不同的概率密度，或者可以粗略的理解成是置信度（因为不同的椭圆的半径相当于表示了不同的$\sigma$）。每当有一个新的数据进入参与判别，就可以通过两个已经确定好参数的数据分布，通过Bayes公式得到属于两个类别的概率。

    ![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202308102028255.png)

  - 这里也体现了Generative算法和Discriminative算法的区别：
    - Generative 算法的极大似然为 $L(\theta) = \prod p(x,y;\theta)$
    - Discriminative算法的极大似然为 $L(\theta) = \prod p(y|x;\theta)$

### GDA 与 Logistics

**GDA推导logistics regression**

可以证明，固定参数$\phi, \mu_0, \mu_1, \Sigma$ ，公式：

$$ \begin{aligned} p(y=1|x;\phi,\mu_0,\mu_1,\Sigma) &=  \frac{p(x|y=1;\phi,\mu_1,\Sigma)p(y=1;\phi)}{p(x;\phi,\mu_0,\mu_1,\Sigma)} \\ &= \frac{1}{1+e^{-\theta^Tx}} \end{aligned} $$

即为Sigmoid Function.


**GDA与Logistics比较**

- GDA 假设
  - $x|y=0 \sim N(\mu_0, \Sigma)$
  - $x|y=1 \sim N(\mu_1, \Sigma)$
  - $y\sim Bernoulli(\phi)$


- Logistics Reg. 假设
  - $p(y=1|x;\theta) = h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}$

在上可知，GDA的假设更强，$GDA \Rightarrow LR$，但反之不然。对于一组符合假设分布的数据GDA可能会有更好的效果。

事实上，假设两组数据都服从同一类型的指数分布族，且只有natural parameters不同时都可类似的推出如上的*Sigmoid Function*的形式。