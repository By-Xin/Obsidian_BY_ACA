#StatisticalLearning 
# OLS

$$ h(x) = \sum_{i=0}^n \theta_i x_i = \theta^T x $$

其中称$\theta$为参数; 特别地, $\theta_0$称为截距项(intercept term) (其对应的$x$为1).
## 1.1 OLS的Cost Function
$$J(\theta) = \frac{1}{2}\sum_{i=1}^{n}(h_{\theta}(x^{(i)})-y^{(i)})^2$$

## 1.2 优化算法：梯度下降法 Gradient Descent

### Gradient Descent 的一般形式


$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

其中$j=1,2,...,p$，也就是同步更新所有的特征

#### 对于OLS case

对损失函数关于某个特征求偏导有：

$$ \begin{aligned} \frac{\partial}{\partial \theta_{j}} J(\theta) &=\frac{\partial}{\partial \theta_{j}} \frac{1}{2}\left(h_{\theta}(x)-y\right)^{2} \\ &=2 \cdot \frac{1}{2}\left(h_{\theta}(x)-y\right) \cdot \frac{\partial}{\partial \theta_{j}}\left(h_{\theta}(x)-y\right) \\ &=\left(h_{\theta}(x)-y\right) \cdot \frac{\partial}{\partial \theta_{j}}\left(\sum_{i=0}^{n} \theta_{i} x_{i}-y\right) \\ &=\left(h_{\theta}(x)-y\right) \cdot x_{j} \end{aligned} $$

因此对于一个样本$i (i=1,2,...,n)$的每个特征$(j =1,2,...,q)$，梯度下降的更新公式为：

$$ \theta_{j}:=\theta_{j}+\alpha\left(y^{(i)}-h_{\theta}(x^{(i)})\right) \cdot x_{j}^{(i)} $$

这一更新公式是符合 *最小均方误(MSE)* 的，因此也成为LMS(least mean square)更新规则，或Widrow-Hoff规则。

### Batch Gradient Descent 批量梯度下降法

算法描述：

>Repeat until convergence {
>$$ \theta := \theta + \alpha \sum_{i=1}^n (y^{(i)} -  h_\theta(x^{(i)}) )x^{(i)} $$
>}





批量梯度下降法每次迭代都要用到所有的训练样本，所以当训练集很大时，会很慢。

### Stochastic Gradient Descent 随机梯度下降法

算法描述:
> Loop{
> 
>  for i = 1 to n {
> 
> $$ \theta := \theta + \alpha(y^{(i)} - h_\theta(x^{(i)}))x^{(i)} $$
>       
>    }
> 
> }

随机梯度下降法每次只用一个样本来更新参数，而不是用所有的样本.

## 1.3 关于OLS的数学补充

### 矩阵求导

定义:

$$ \nabla_A f(A) = \begin{bmatrix} \frac{\partial f}{\partial A_{11}} & \frac{\partial f}{\partial A_{12}} & \cdots & \frac{\partial f}{\partial A_{1n}} \\ \frac{\partial f}{\partial A_{21}} & \frac{\partial f}{\partial A_{22}} & \cdots & \frac{\partial f}{\partial A_{2n}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f}{\partial A_{m1}} & \frac{\partial f}{\partial A_{m2}} & \cdots & \frac{\partial f}{\partial A_{mn}} \end{bmatrix} $$

例如:

若$f(A) = \frac23 A_{11} + 5A_{12}^2 + A_{21}A_{22}$, 则

$$ \nabla_A f(A) = \begin{bmatrix} \frac23 & 10A_{12} \\A_{22} &A_{21} \end{bmatrix} $$

*统计学讨论部分略*



