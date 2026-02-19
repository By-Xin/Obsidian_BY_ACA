#StatisticalLearning 
# 局部线性回归

希望在设计回归方程时, 若考虑当前某点的作用, 希望距离该点临近的点提供更大的信息权重, 距离该点较远的点提供较小的信息权重.

### 欠拟合与过拟合

- underfitting
- overfitting

### LWR的损失函数

$$ J(\theta) = \sum_i \omega^{(i)}(y^{(i)} - \theta^T x^{(i)})^2 $$

$\omega^{(i)}$ 是人为设计的一个参数, 其中一种比较成熟的设计为(高斯核):

$$ \omega^{(i)} = \exp \left( - \frac{ (x^{(i)} - x) ^2}{2 \tau^2} \right) $$

其中称$\tau$为带宽, 用来控制$\omega^{(i)}$的衰减速度. $\tau$越大, $\omega^{(i)}$衰减越慢, $\tau$越小, $\omega^{(i)}$衰减越快.

若用矩阵形式表达,则为:

$$ J(\theta) = (y - X\theta)^T W (y-X\theta)$$

$$\theta = (X^{T}WX)^{-1}X^{T}Wy$$


### LWR的python实现

记: 数据样本量为m


```python
#coding = utf-8
import numpy as np
import scipy.stats as stats
from math import *
import matplotlib.pyplot as plt

def getw(x0,x,k):
    '''
    :param x0: 1*m 是当前样本点
    :param x: n*m 是整个样本集
    :param k: 高斯核的带宽,也就是\tau
    :return: w: n*n
    '''
    w = np.zeros([m,m]) # 初始化, m是样本数
    for i in range(m):
        # w[i,i] = exp((x1[i,1] - x0)**2/(-2*k*k))
        w[i, i] = exp((np.linalg.norm(x0 - x[i])) / (-2 * k ** 2))
    return w

def getyvalue(x1,x,y,k):
    '''
    :param x1: n*2 是
    :param x: m*2 是
    :param y: m*1
    :param k: 高斯核的带宽,也就是\tau

    '''
    y_value = np.zeros(m)
    #w = np.matrix(np.zeros((m, m)))
    w = np.zeros([m,m])

    for i in range(m):
        w = getw(x[i],x, k)
        theta = np.linalg.inv(x1.T.dot(w).dot(x1)).dot(x1.T).dot(w).dot(y)
        y_value[i] = theta[0] + theta[1] * x[i]
    return y_value

if __name__ == "__main__":
    x = np.arange(1, 101)
    x = np.array([float(i) for i in x])
    y = x + [10 * sin(0.3 * i) for i in x] + stats.norm.rvs(size=100, loc=0, scale=1.5)
    #plt.figure(figsize=(12, 6))
    #plt.scatter(x, y)
    #plt.show()

    x = x.reshape(-1, 1)
    x1 = np.c_[np.ones((100, 1)), x]
    y = y.reshape(-1, 1)
    m = len(x)

    y_lwlr = np.zeros(m)
    y_lwlr = getyvalue(x1,x,y,k=1.2)
    plt.figure(figsize=(12,6))
    plt.scatter(x, y)
    plt.plot(x, y_lwlr, 'r')
    plt.show()
```


说明:

- 注意看, 在上面的代码实现中,并没有使用循环迭代收敛求解,而是通过类似于统计学的方法推导出LWLR的解析解, 直接通过矩阵计算完成的(代码中的循环都是为了生成矩阵元素等)
- 如果说的话, 全部的计算流程和典型的$(X^TX)^{-1}X^TY$是别无二致的,只不过相比之下多了几个关于$X$的运算
- 这个运算相当于每次提取一个样本, 和全部样本集合进行一个矩阵运算,再通过Gauss核得到一个权重, 记为权重矩阵的一个元素, 以此类推得到整个权重.