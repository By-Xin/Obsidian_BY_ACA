# Optimization Algorithms

> Refs: 邱怡轩 2023 深度学习讲义 (邱门!); Dive into Deep Learning (D2L) 2021; 神经网络与深度学习 (邱锡鹏); Linda 2025 Deep Learning.

## Optimization and Deep Learning

在深度学习中，神经网络的训练几乎总可以归结为一个优化问题: 我们的模型相当于一个依赖参数 $\mathbf{\theta}$ 的复杂函数 $f_\theta(\cdot)$，我们的目标就是找到一组参数 $\mathbf{\theta}^*$ 使得输入数据 $\mathbf{X}$ 对应的输出 $\mathbf{Y}$ 和真实标签 $\mathbf{Y}^*$ 之间的某种距离(或者说损失) $\mathcal{L}(\mathbf{Y}, f_\theta(\mathbf{X}))$最小. 这个问题可以形式化为如下的优化问题:

$$
\mathbf{\theta}^* = \arg\min_{\mathbf{\theta}} \mathcal{L}(\mathbf{Y}, f_\theta(\mathbf{X}))
$$

这个 loss function 也称为 objective function. 在深度学习中, 通常 objective function 是没有解析解的, 因此我们需要使用一些数值的优化算法来求解这个问题. 

### GD and SGD

在传统的统计模型中, 经常用牛顿法 (Newton's method) 进行优化. 但是其需要同时利用一阶和二阶导数信息, 对于参数量为 $p$ 的神经网络, 对应计算复杂度为 $\mathcal{O}(p^3)$, 这对于大规模的神经网络来说是并不现实的. 

因此, 在深度学习中几乎只能依赖于梯度, 通常称为一阶优化方法.  最简单的优化算法就是梯度下降 (Gradient Descent, GD) 算法.  其更新规则如下:

$$
\mathbf{\theta}^{(k+1)} = \mathbf{\theta}^{(k)} - \eta \nabla_{\mathbf{\theta}} \mathcal{L}(\theta^{(k)})
$$
其中
- $\mathbf{\theta}^{(k)}$ 是第 $k$ 次迭代的参数值
- $\eta$ 是学习率 (learning rate)
- $\nabla_{\mathbf{\theta}} \mathcal{L}(\theta^{(k)})$ 是 loss function 关于参数 $\mathbf{\theta}^{(k)}$ 的**精确梯度**, 也就是说是在整个训练集上计算的梯度:
$$
\nabla_{\mathbf{\theta}} \mathcal{L}(\theta^{(k)}) = \frac{1}{n} \sum_{i=1}^n \nabla_{\mathbf{\theta}} \mathcal{L}(\mathbf{Y}_i, f_{\theta^{(k)}}(\mathbf{X}_i))
$$

然而, 在深度学习中, 通常训练集的规模 $n$ 非常大, 因此计算上述的精确梯度是非常耗时的. 为了加速计算, 我们可以使用随机梯度下降 (Stochastic Gradient Descent, SGD) 算法.  其更新规则如下:

$$
\mathbf{\theta}^{(k+1)} = \mathbf{\theta}^{(k)} - \eta \nabla_{\mathbf{\theta}} \ell(\theta^{(k)})
$$
其中
- $\nabla_{\mathbf{\theta}} \ell(\theta^{(k)})$ 是 loss function 关于参数 $\mathbf{\theta}^{(k)}$ 的**随机梯度**, 也就是说是在训练集中随机抽取部分样本 (mini-batch) 计算的梯度.
- 具体而言, 一个 mini-batch 的大小通常为 $2^m, m \in \mathbb{N}$, 通常取 $32 \sim 256$ 之间.
    ![Mini-Batch](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250206161220.png)

### Problems in Optimization

而在这个优化过程中, 常见的挑战包括:
- 局部最优解 local minima
- 鞍点 saddle points
- 梯度消失/爆炸 vanishing/exploding gradients

#### Local Minima

![Local Min and Global Min](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250206152037.png)

局部和全局最小值的定义如下:
- **Local minima** 是指对于函数 $f(x)$, 在某个点 $x_0$ 处, 如果在 $x_0$ 附近 (存在某个邻域 $N(x_0)$) 的所有点 $x$ 都有 $f(x) \geq f(x_0)$, 那么 $x_0$ 就是一个 local minimum. 
- **Global minima** 是指对于函数 $f(x)$, 在整个定义域内, 存在一个点 $x^*$ 使得对于所有的 $x$, 都有 $f(x^*) \leq f(x)$, 那么 $x^*$ 就是一个 global minimum.

通常的神经网络的 loss function 会存在很多的局部最小值, 因此一些优化算法就可能会陷入局部最小值而无法收敛到全局最小值.  这时, 反而如果在算法中注入一些随机性 (noise) 可能会有助于跳出局部最小值. 因此, SGD 算法反而可能会比 GD 算法更容易收敛到全局最小值.

#### Saddle Points (鞍点)

一个函数的鞍点是指在该点处的梯度为零, 但是该点不是局部最小值或者最大值. 这尤其在高维空间中是非常常见的. 例如, 对于一个二维的函数  $f(x, y) = x^2 - y^2$, 在点 $(0, 0)$ 处就是一个鞍点. 而这点是关于 $x$ 的局部最小值, 但是关于 $y$ 的局部最大值.

一个判断鞍点的方法是通过函数的二阶导数信息. 一个点是鞍点的充要条件是其 Hessian 矩阵的特征值中有正有负. 具体而言, 对于一个$p$ 维的函数 $f(\mathbf{x}): \mathbb{R}^p \to \mathbb{R}$, 其 Hessian 矩阵为:
$$
\mathbf{H} = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_p} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_p} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_p \partial x_1} & \frac{\partial^2 f}{\partial x_p \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_p^2}
\end{bmatrix}
$$
当我们在一个点 $\mathbf{x}_0$ 处计算其 Hessian 矩阵 $\mathbf{H}_0$ 对这个数值的 Hessian 矩阵进行特征值分解, 得到特征值 $\lambda_1, \lambda_2, \ldots, \lambda_p$. 我们判断:
- 如果所有的特征值都是正的, 那么该点是局部最小值
- 如果所有的特征值都是负的, 那么该点是局部最大值
- 如果特征值中有正有负, 那么该点是鞍点

在高维的情况下, Hessian 矩阵的特征值分解为有正有负的情况是非常常见的. 因此, 在深度学习中, 鞍点是一个非常常见的问题.

#### Vanishing/Exploding Gradients

Vanishing gradients 是指在神经网络的训练过程中, 由于梯度消失, 导致梯度下降算法无法收敛. 一个经典的例子是在使用 tanh 激活函数时 ($f(x) = \tanh(x), \quad f'(x) = 1 - \tanh^2(x)$), 当输入的绝对值较大时, 其导数会接近于零 (即使是 $f'(4)\approx 0.0013$), 导致梯度消失. 这也是为什么 ReLU 激活函数 ($f(x) = \max(0, x), \quad f'(x) = \mathbb{I}(x > 0)$) 在深度学习中更加常用的原因之一. 相反, Exploding gradients 是指在神经网络的训练过程中, 由于梯度爆炸, 导致梯度下降算法无法收敛. 

---

总而言之, 在深度学习中, 通常没有一个万能的优化算法, 选择合适的优化算法取决于具体的问题. 我们也并不追求找到全局最小值, 而是希朼找到一个足够好的局部最小值. 

## Convexity in Optimization

Convexity (凸性) 是优化问题中的一个重要概念. 

### Convex sets and functions

***定义*** **(Convex Set)**: 若一个集合 $\mathcal{X}$ 满足对于任意两个点 $a, b \in \mathcal{X}$, 以及任意 $0 \leq \lambda \leq 1$, 都有 
$$
\lambda a + (1-\lambda) b \in \mathcal{X} \quad \forall a, b \in \mathcal{X}, \lambda \in [0, 1]
$$
则称 $\mathcal{X}$ 是一个凸集 (convex set).

- 直观上从几何的角度来看, 一个任取集合内的两点, 连接这两点的线段上的所有点都在这个集合内, 那么这个集合就是凸集, 否则就是非凸集.

    ![左1为 non-convex, 其余为 convex](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250307135329.png)

***定义*** **(Convex Function)**: 若一个一元函数 $f(x)$ 满足对于任意两个点 $x, x' \in \mathbb{R}$, 以及任意 $0 \leq \lambda \leq 1$, 都有
$$
f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x')
$$
则称 $f(x)$ 是一个凸函数 (convex function).

- 直观上从几何的角度来看, 一个函数的图像上任意两点的连线都在函数的图像上方, 那么这个函数就是凸函数, 否则就是非凸函数.
    ![中间为 non-convex, 其余为 convex](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250307135648.png)

- 若函数是二阶可微的, 则其是凸函数的充要条件是其 Hessian 矩阵是半正定的, 即对于任意的 $\mathbf{x} \in \mathbb{R}^p$, 都有 $\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$.
- 对于一个凸函数, 若存在 local minimum, 那么这个 local minimum 就是 global minimum (不过可能存在多个 global minimum, 或者不存在 global minimum). 


## Gradient Descent

### GD 的数学原理

以一个连续可微的一元函数 $f: \mathbb{R} \to \mathbb{R}$ 为例. 根据 Taylor 展开, 我们有:
$$f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!} (x - a)^2 + \frac{f'''(a)}{3!} (x - a)^3 + \dots + \mathcal{O}((x-a)^n)$$
因此在 $x$ 展开的一阶近似为:
$$f(x) = f(a) + f'(a)(x - a) + \mathcal{O}((x-a)^2)$$
考虑一个微小的更新 $x:=x+\epsilon$, 我们有:
$$f(x+\epsilon) = f(x) + f'(x) \epsilon + \mathcal{O}(\epsilon^2)$$

考虑在 GD 中, 我们令 $\epsilon = -\eta f'(x), \eta>0$, 则有:
$$f(x - \eta f'(x)) = f(x) - \eta f'(x)^2 + \mathcal{O}(\eta^2 f'(x)^2)$$

因此, 只要 $f'(x) \neq 0$, 那么我们都有 $f(x - \eta f'(x))< f(x)$, 即 GD 算法总是朝着梯度的反方向更新参数使得函数值减小. 

**学习率的设置**
- 较小的 $\eta$ 的取值也保证了误差项不会主导更新过程, 从而保证了收敛性; 但是同时也会影响收敛速度.
- 较大的 $\eta$ 的取值会加速收敛, 但是可能会导致算法不稳定, 甚至发散. 此外, 陷入局部最小值并不一定是过小/过大学习率的专属问题, 对于非凸函数, 过大学习率也可能导致算法陷入局部最小值, 即使函数在一个合理的 initial point 上. 
    ![f(x) = x cos(cx)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250307161724.png)

---

多元的情况是完全类似的. 对于一个多元函数 $f: \mathbb{R}^d \to \mathbb{R}$ 以及 input $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$, 我们有:
$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & \cdots & \frac{\partial f}{\partial x_d} \end{bmatrix}^\top$$

同样进行多元的 Taylor 展开, 我们有:
$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2)$$

我们取下降最快的方向为 $-\eta \nabla f(\mathbf{x})$, 则有:
$$ \mathbf{x}  \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$$


### Newton's Method

理论上, 我们可以通过牛顿法 (Newton's method) 来加速收敛. 其思想为通过引入二阶导数信息来更新参数. 这样算法不仅考虑了当前的梯度信息, 还考虑了梯度的变化率 (即 curvature $\nabla^2 f(\mathbf{x})$).

![Gradient descent versus Newton’s method for minimizing some arbitrary loss function. Starting from the point (5, 5), gradient descent converges to the minimum in 229 steps, whereas Newton’s method does so in only six. (Image by Adrian Lam)](https://towardsdatascience.com/wp-content/uploads/2020/11/1Rn_xMky49Xa19liTkS3NFg.png)

其更新规则为:
$$
\mathbf{x} \leftarrow \mathbf{x} - \mathbf{H}^{-1} \nabla f(\mathbf{x}) 
$$

其原理为, 将 $f(\mathbf{x})$ 在 $\mathbf{x}$ 处进行二阶泰勒展开:
$$
f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \mathbf{H} \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3) 
$$

我们希望找到一个 $\boldsymbol{\epsilon}$ 使得 $f(\mathbf{x} + \boldsymbol{\epsilon})$ 最小, 因此对上式关于 $\boldsymbol{\epsilon}$ 求导并令其为零, 我们有:
$$\begin{aligned}
\nabla_{\boldsymbol{\epsilon}} f(\mathbf{x} + \boldsymbol{\epsilon}) &= \nabla_{\boldsymbol{\epsilon}} \left( f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \mathbf{H} \boldsymbol{\epsilon} \right) \\
&= \boldsymbol{0} + \nabla_\mathbf{x} f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = \boldsymbol{0} \\
\end{aligned}$$

故有
$$\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x})$$

此时, 只要 $\mathbf{H}$ 是正定的, 那么更新 $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{H}^{-1} \nabla f(\mathbf{x})$ 就是使得 $f(\mathbf{x})$ 减小的方向.

---


**需要指出, 计算 Hessian 矩阵 ($\mathcal{O}(d^2)$)并求逆的时间复杂度很高, 因此牛顿法在大规模的神经网络中并不适用.** 不过, 牛顿法在一些小规模的优化问题中仍然是一个非常有效的算法, 并且可以作为一些更复杂的优化算法的基础和参考.

一个常见的改进算法是拟牛顿法 (Quasi-Newton method), 其通过近似 Hessian 矩阵来减少计算量.  通用的形式是:
$$
\mathbf{x}_{t+1} \leftarrow \mathbf{x}_t - \eta_t\mathbf{B}_t^{-1} \nabla f(\mathbf{x}_t)
$$
其中 $\eta$ 是学习率, 在实践中从可以是最简单的恒为$1$, 也可以是通过线搜索等方法来确定. $\mathbf{B}_t$ 是一个对 Hessian 矩阵的近似, 通常要求其是正定的. 并且同样需要在每次迭代中更新 $\mathbf{B}_t$ (或直接更新其逆矩阵). 其具体的设计有很多种,一个比较简单的思想是只选择 Hessian 矩阵的对角线元素, 即 $\mathbf{B}_t = \text{diag}(\mathbf{H}_t)$. 

另一个更为常见的拟牛顿法是 BFGS 算法 (Broyden-Fletcher-Goldfarb-Shanno algorithm), 该算法在 1970 年代由这四人分别提出, 故以此得名. 其具体算法介绍可以参考 https://towardsdatascience.com/bfgs-in-a-nutshell-an-introduction-to-quasi-newton-methods-21b0e13ee504/. 后面也会专门进行介绍.

![From left to right: Broyden, Fletcher, Goldfarb, and Shanno.](https://towardsdatascience.com/wp-content/uploads/2020/11/1UyPDBKccaqfRTc00l1rKEg.jpeg)

## Stochoastic Gradient Descent

在 GD 中, 我们考虑的是在全部数据集上的目标函数, 因此对应的也是全部数据集上的梯度:
$$
f(\mathbf{x};\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n f_i(\mathbf{x};\mathbf{w})
\quad
\nabla f(\mathbf{x};\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \nabla f_i(\mathbf{x};\mathbf{w})
$$ 
对于一个规模为 $n$ 的数据集, 计算梯度的时间复杂度为 $\mathcal{O}(n)$, 因此在大规模数据集上计算梯度是非常耗时的.

SGD 相当于对全体数据集进行随机采样. 最化简的情况是在每次的迭代中只随机采样一个样本, 即 mini-batch 的大小为 $1$. 其更新规则为:
$$ 
\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla f_i(\mathbf{x};\mathbf{w})
$$
即用一个样本的梯度来更新这个模型的参数.

可以证明, 由于这个样本是随机采样的, 因此 SGD 也是建立在 unbiased 的梯度估计上的, 因此也会收敛到全局最小值. 
$$
\mathbb{E}_i [\nabla f_i(\mathbf{x};\mathbf{w})] = \frac 1 n \sum_{i=1}^n \nabla f_i(\mathbf{x};\mathbf{w}) = \nabla f(\mathbf{x};\mathbf{w})
$$
不过这个估计的方差是比较大的, 含有噪声, 因此 SGD 的收敛速度理论上会比 GD 慢.

### Dynamic Learning Rate

在实践中当采用 SGD 时, 通常会采用动态学习率 (dynamic learning rate) 的方法根据训练的不同阶段来调整学习率. 

通常由如下几种方法:
- Piecewise constant learning rate: 在训练的不同 iteration 阶段直接强制指定不同的学习率. 前期的学习率较大, 后期的学习率较小.
- Exponential decay: $\eta_t = \eta_0 \exp(-\lambda t)$, 其中 $\eta_0$ 是初始学习率, $\lambda$ 是衰减率, $t$ 是迭代次数. 这个方法并不会特别常用, 因为其衰减速度是指数级的, 有时候会导致学习率过早衰减.
- Polynomial decay: $\eta_t = \eta_0 (1 + \lambda t)^{-\alpha}$, 其中 $\alpha$ 是一个超参数. 这个方法的衰减速度是多项式级的, 通常会比指数级的衰减速度慢一些. 一个常见的选择是 $\alpha=0.5$.

## Minibatch SGD

在实践中, 通常会采用 mini-batch SGD (并且当我们说 SGD 时, 通常指的是 mini-batch SGD). 由于 CPU/GPU 的并行计算能力, 我们可以同时计算多个样本的梯度, 从而加速计算. 具体而言, 在每次迭代中, 我们会随机采样一个 mini-batch 的样本, 计算这个 mini-batch 的梯度, 并更新参数:
$$
\mathbf{g}_t = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \nabla f_i(\mathbf{x};\mathbf{w})
\quad
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{g}_t
$$
其中 $\mathcal{B}_t$ 是第 $t$ 次迭代的 mini-batch, $|\mathcal{B}_t|$ 是 mini-batch 的大小.

Mini-batch SGD 同样是建立在 unbiased 的梯度估计上的:
$$
\mathbb{E} (\mathbf{g}_t) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbb{E} \left[ \nabla f_i(\mathbf{x};\mathbf{w}) \right] =\frac{|B_t|}{|B_t|} \frac{1}{n} \sum_{i=1}^n \nabla f_i(\mathbf{x};\mathbf{w}) = \nabla f(\mathbf{x};\mathbf{w})
$$
并且 mini-batch 对应的方差同样会比全样本的方差小 (缩小了 $|\mathcal{B}_t|$ 倍):
$$
\text{Var}(\mathbf{g}_t) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \text{Var} \left[ \nabla f_i(\mathbf{x};\mathbf{w}) \right]  = \frac{1}{|\mathcal{B}_t|} \text{Var} \left[ \nabla f_i(\mathbf{x};\mathbf{w}) \right]
$$

在 PyTorch 中, 我们可以通过 `optimizer = optim.SGD(model.parameters(), lr=LR)` 来构建一个 SGD 优化器, 并且在训练时通过 `optimizer.step()` 来更新参数.

> ***Exercise***
>
> - 如果全样本数据集大小为 $n$, mini-batch 的大小为 $m$, 则每个 epoch 中参数会被更新 $n/m$ 次.
> - 一个 epoch 的含义就是所有的数据集被用于训练一次. 在实践 mini-batch SGD 时, 哦我们并不是真正通过采样的方式来生成 mini-batch, 而是先将数据集随机打乱, 然后按照顺序划分为多个 mini-batch. 这样可以保证每个 epoch 中每个样本都会被用到, 并且每个样本都有相同的机会被采样到. 
>   - 在 PyTorch 中, 这个步骤是在构建数据集的时候通过 `dataloader = torch.utils.data.DataLoader(dataset, batch_size=m, shuffle=True)` 来实现的.
>   - 在具体的训练中, 我们通过 `for batch_x, batch_y in dataloader:` 就可以依此取出每个 mini-batch 的数据进行训练.


## Momentum (动量法)

Momentum 方法是一种在 SGD 的基础上加入动量的方法. 其思想是在更新参数时, 不仅考虑当前的梯度信息, 还考虑历史的梯度信息. 具体而言, 对于目标函数的梯度 $\mathbf{g}_t$, 我们引入一个动量参数 $\mathbf{v}_t$ 其维度和梯度相同, 并且在每次迭代中更新为:
$$
\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \mathbf{g}_t =\sum_{\tau=0}^t \beta^{\tau} \mathbf{g}_{t-\tau}
, \quad \mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}
$$
其中 $\beta\in(0,1)$ 是动量参数, 相当于进行一个指数加权平均. 一般会初始化 $\mathbf{v}_0 = \mathbf{0}$.

这时, 参数并不会向着当前的梯度方向更新, 而是会综合考虑历史的梯度信息, 以一个更稳健平滑的方向更新参数.

Moment 方法在一些非常 ill-conditioned 的问题上会有更好的表现. 例如, 考虑下面这个二维的函数:
$$
f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2
$$
已知其最小值在 $(0,0)$ 处. 假设初始化的起点为 $(-5,-2)$. 若采用 SGD 的方法, 会发现算法会在 $x_1$ 上收敛的非常慢, 而若加快学习率, 则会导致在 $x_2$ 上发散. 对于这样的不平衡的 ill-conditioned 问题, Moment 方法会有更好的表现. 这是因为, 由于考虑了历史的梯度信息, 因此对于 $x_1$ 上的梯度信息会有更好的保留, 而对于 $x_2$ 上的梯度信息会由于这样反复震荡的模式而相互抵消, 有所缓解.

## Adagrad

Adagrad 是一种自适应学习率的方法. 其思想是对于每个参数, 根据其历史的梯度信息来调整学习率以 'adapt' 每个参数的收敛情况. 具体而言, 我们需要维护一个历史梯度信息 $\mathbf{s}_t$, 并且以此来调节学习率$\eta$:
$$\begin{aligned}
\mathbf{s}_{t+1} &= \mathbf{s}_t + \mathbf{g}_t \odot \mathbf{g}_t \\
\mathbf{w}_{t+1} &= \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{s}_{t+1} + \epsilon}} \odot \mathbf{g}_t
\end{aligned}$$
其中 $\odot$ 表示逐元素相乘, $\epsilon > 0$ 是一个很小的常数 (如 $10^{-6}$), 用于防止分母为零. 初始化 $\mathbf{s}_0 = \mathbf{0}$.
- 若历史的梯度信息较大, 则 $\mathbf{s}_{t+1}$ 会较大, 从而导致学习率较小, 从而保证了收敛的稳定性.

在 PyTorch 中, 我们可以通过 `optimizer = optim.Adagrad(model.parameters(), lr=LR)` 来构建一个 Adagrad 优化器.

但是, Adagrad 也有一些缺点, 例如其 $\mathbf{s}_t$ 是累加的因此只会随着时间增长而增大 (并且是以平方速率), 从而导致学习率不断减小, 有时候会导致算法收敛过慢.

## RMSprop

RMSprop 是对 Adagrad 的一种改进. 其思想是改掉 Adagrad 中对梯度的平方和的累加, 而是通过 leaky average (即指数加权平均) 来更新历史的梯度信息. 具体而言, 我们同样需要维护一个历史梯度信息 $\mathbf{s}_t$, 但是其更新规则为:
$$\begin{aligned}
\mathbf{s}_{t+1} &= \gamma \mathbf{s}_t + (1-\gamma) \mathbf{g}_t \odot \mathbf{g}_t 
\end{aligned}$$
对于参数的更新还是相同的:
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{s}_{t+1} + \epsilon}} \odot \mathbf{g}_t
$$

## Adam

Adam 是一种结合了 Momentum 和 RMSprop 的方法. 其思想是同时考虑历史的梯度信息和历史的梯度平方信息. 具体而言, 我们需要维护两个历史信息 $\mathbf{v}_t$ 和 $\mathbf{s}_t$, 并且以此来调节学习率:
$$\begin{aligned}
\mathbf{v}_{t+1} &= \beta_1 \mathbf{v}_t + (1-\beta_1) \mathbf{g}_t \\
\mathbf{s}_{t+1} &= \beta_2 \mathbf{s}_t + (1-\beta_2) \mathbf{g}_t \odot \mathbf{g}_t 
\end{aligned}$$
其中 $\beta_1, \beta_2 \in (0,1)$ 是动量参数, $\odot$ 表示逐元素相乘.在该模型提出时, 作者建议超参取值 $\beta_1=0.9, \beta_2=0.999$.

但此时还不能直接使用 $\mathbf{v}_t$ 和 $\mathbf{s}_t$ 来更新参数, 因为这两个信息都是 biased 的. 因此我们需要对其进行修正:
$$\begin{aligned}
\hat{\mathbf{v}}_{t+1} &= \frac{\mathbf{v}_{t+1}}{1-\beta_1^{t+1}} \\
\hat{\mathbf{s}}_{t+1} &= \frac{\mathbf{s}_{t+1}}{1-\beta_2^{t+1}}
\end{aligned}$$
(注意这里的 $\beta^t$ 是指 $\beta$ 的 $t$ 次方而非 iteration index). 
- 这里的 bias 是指:
    $$\begin{aligned}
    \mathbb{E}[\mathbf{v}_t] &= \mathbb{E}[\beta_1 \mathbf{v}_{t-1} + (1-\beta_1) \mathbf{g}_t] \\
    &= (1-\beta_1) \sum_{j=0}^{t-1} \beta_1^j \mathbb{E}[\mathbf{g}_{t-j}]  \\
    &= (1-\beta_1)^t \mathbb{E}[\mathbf{g}_t] 
    \end{aligned}$$
    - 第二个等号是将递归展开, 第三个等号是假设 $\mathbf{g}_t$ 是 stationary 的, 即其期望不随时间变化, 每期期望相同.

最终的参数更新规则为:
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\hat{\mathbf{v}}_{t+1}}{\sqrt{\hat{\mathbf{s}}_{t+1} }+ \epsilon}
$$