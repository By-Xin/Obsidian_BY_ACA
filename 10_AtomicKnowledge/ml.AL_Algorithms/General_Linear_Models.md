#StatisticalLearning 
**引入**

回忆上文，对于ols和logistics，我们分别假设：
- ols：$y|X;\theta \sim N(\mu, \sigma^2)$ (其中$\mu = X\theta$)
- logistics：$y|X;\theta \sim Bernoulli(\phi)$ (其中$\phi = \frac{1}{1+e^{-X\theta}}$)
  
事实上，这两种模型假设都可以得以统一，得到更加通用的GLM模型。

## 3.1 [[Exponential Family]] 指数分布族

给出exponential family distributions 的定义：
$$ p(y;\eta) = b(y) \exp(\eta^T T(y) - a(\eta)) $$
其中：
- $\eta$ 是 natural parameter 或者 canonical parameter，是决定分布的一个参数
- $T(y)$ 是标签$y$的一个充分统计量(sufficient statistic)，有时会取$T(y) = y$
- $a(\eta)$ 是 log partition function，是一个归一化因子，使得指数分布族中的分布pdf积分为1

当固定$T$的选择后，不同的$a,b$就会确定不同的分布族，这些分布族都是指数分布族，其分布的参数由$\eta$决定。

事实上，诸如*Gaussian, Bernoulli, Binomial, Poisson, Exponential, Gamma, Beta, Dirichlet*等分布都是指数分布族的一种。

### Bernoulli 分布与指数分布族

已知Bournoulli Distribution：
$$\begin{align*} p(y;\phi) &= \phi^y(1-\phi)^{1-y} \\&= \exp(y\log\phi + (1-y)\log(1-\phi)) \\ &= \exp[(\log\frac{\phi}{1-\phi})y + \log(1-\phi) ] \end{align*}$$



参照GLM的定义，可以发现Bernoulli的分布是令GLM中：
- $T(y) = y$
- $\eta = \log(\frac{\phi}{1-\phi})$  *(有趣的是，其等价于$\phi = \frac{1}{1+e^{-\eta}}$，即为logistic function)*
- $a(\eta) = -\log (1-\phi) = \log(1 + e^{\eta})$
- $b(y) = 1$

### 正态分布与指数分布族

不失一般性，令正态分布的$\sigma^2=1$（因正态分布假定的标准差不会影响参数的取值），则有：
$$\begin{align} p(y;\mu)  &= \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}(y-\mu)^2\right) \\ &= \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}y^2\right)\exp\left(\mu y-\frac{1}{2}\mu^2\right) \end{align}$$

对比GLM的定义，可以发现：
- $\eta = \mu$
- $T(y) = y$
- $a(\eta) = \mu^2/2 = \eta^2/2$
- $b(y) = (1/\sqrt{2\pi}\exp(-y^2/2)$))

## 3.2 GLM的建模

1. 假定在给定$x$和其线性组合的组合系数$\theta$后，$y$服从以$\eta$为参数的指数分布族分布，即：$y|x;\theta \sim \text{Exponential Family}(\eta)$
2. 建模的目的是：当给定数据$x$，找到一个合理的假设$h(x)$来较好的预测$T(y)$（常令$T(y)=y$），即：$h(x)=\mathbb{E}[T(y)|x]$
3. $\eta$和$x$之间的关系为：$\eta = \theta^T x$

## 3.3 Softmax Regression

**考虑如下多分类问题：**

- 假设$y$可能有$k$种可能的分类结果，即$y\in\{1,2,...,k\}$。

- 为了避免虚拟变量陷阱问题，我们可以用$k-1$个参数 $\phi_1, \phi_2, ..., \phi_{k-1}$ 来描述$y$于第$i$类的概率，即$\phi_i = p(y=i;\phi)$ (我们可以进一步推知$\phi_k=1-\sum_{i=1}^{k-1}\phi_i ~ (\ast)$)

**这里想要同样将这个多分类问题整理为一个GLM问题，因此参考GLM定义进行如下的转换：**

- 将$y$映射成一个$k-1$维的向量 $T(y) \in \R^{k-1}$，具体而言：
    $$T(1) = \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, T(2) = \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, ..., T(k-1) = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix} ,T(k) = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}$$

- 为说话方便，记$T(y)_i$为$y$的第$i$个分量；此外引入示性函数$\mathit{1}(\cdot)$.


- 则我们的优化目标即为尽可能的预测$T(y)$的每一个分量（预测结果尽可能接近每个分量的期望），即：
    $$ E[T(y)_i ] = P(y=i) = \phi_i$$

**下说明 *Multinomial Distribution 也是 exponential fanmilily的一种 :***

$$ \begin{align*}
p(y;\phi) &= \phi_1^{\mathit{1}(y=1)} \phi_2^{\mathit{1}(y=2)} \cdots \phi_k^{\mathit{1}(y=k)} \\ &= \phi_1^{\mathit{1}(y=1)} \phi_2^{\mathit{1}(y=2)} \cdots \phi_k^{1-\sum_{i=1}^{k-1}\mathit{1}(y=i)} \textit{  (by *)} \\ &= \phi_1^{T(y)_1} \phi_2^{T(y)_2} \cdots \phi_k^{T(y)_k} \textit{  (by def of T())}  \\& \equiv \exp [T(y)_1\log(\phi_1) + T(y)_2\log(\phi_2) + \cdots + (1-\sum_{i=1}^{k-1}T(y)_i)\log(\phi_k)] \\&\equiv \exp[ T(y)_1\log(\phi_1/\phi_k) + T(y)_2\log(\phi_2/\phi_k) + \cdots + T(y)_{k-1}\log(\phi_{k-1}/\phi_k) + \log(\phi_k) ] \\& := b(y) \exp[\eta^T T(y) - a(\eta)] \\ \textit{Q.E.D.}\end{align*} $$

其中：
- $\eta = [\log(\phi_1/\phi_k), \log(\phi_2/\phi_k), \cdots, \log(\phi_{k-1}/\phi_k)]^T$
- $a(\eta) = -\log(\phi_k)$
- $b(y) = 1$

**由GLM推导*Softmax Function*：**

在上面的证明中，我们得到的多分类问题对应的GLM参数为：
$$\eta_i = \log\frac{\phi_i}{\phi_k}$$
为方便起见，人为定义$\eta_k = \log(\phi_k/\phi_k) = 0$作为初始条件；且结合$\phi$的定义得到等式 $\sum_{i=1}^k\phi_i = 1$，最终将上述递归式子整理成：
$$\phi_i = \frac{\exp(\eta_i)}{\sum_{j=1}^k\exp(\eta_j)}$$

*这就是大名鼎鼎的 Softmax Function！*

**通过GLM的建模策略完成*Softmax Regression***


延续上面的推导，我们有：

$$
\begin{aligned}
h_\theta(x) & =\mathrm{E}[T(y) \mid x ; \theta] \\
& =\mathrm{E}\left[\begin{array}{c}
1\{y=1\} \\
1\{y=2\} \\
\vdots \\
1\{y=k-1\}
\end{array}\right] \textit{(Given x;}\theta\textit{)} \\
& =\left[\begin{array}{c}
\phi_1 \\
\phi_2 \\
\vdots \\
\phi_{k-1}
\end{array}\right]\\
&=  {\left[\begin{array}{c}
\frac{\exp \left(\theta_1^T x\right)}{\sum_{j=1}^k \exp \left(\theta_j^T x\right)} \\
\frac{\exp \left(\theta_2^T x\right)}{\sum_{j=1}^k \exp \left(\theta_j^T x\right)} \\
\vdots \\
\frac{\exp \left(\theta_{k-1}^T x\right)}{\sum_{j=1}^k \exp \left(\theta_j^T x\right)}
\end{array}\right] . }
\end{aligned}
$$


事实上，这里的预测值就是给出了（对于一次实验，或对于一个样本观测可能的结果$y$，其属于）$k-1$个类别的可能发生的概率。因此，我们可以将这个模型看作是一个概率模型，就可以同样使用*MLE*来求解参数$\theta$。

假设共有$n$个训练数据$ \left\{ \left(x^{(i)}, y^{(i)}\right) \right\}_{i=1}^n $，通过计算，最终得到的对数似然函数为：
$$
\begin{aligned}
\ell(\theta) & =\sum_{i=1}^n \log p\left(y^{(i)} \mid x^{(i)} ; \theta\right) \\
& =\sum_{i=1}^n \log \prod_{l=1}^k\left(\frac{e^{\theta_l^T x^{(i)}}}{\sum_{j=1}^k e^{\theta_j^T x^{(i)}}}\right)^{1\left\{y^{(i)}=l\right\}}
\end{aligned}
$$

通过相似的优化算法即可完成最后的求解。
