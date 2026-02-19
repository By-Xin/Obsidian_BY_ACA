---
aliases: [核方法, Kernel Methods, Kernel Trick, 核技巧]
tags:
  - concept
  - ml/stats
  - math/linear-algebra
related_concepts:
  - "[[RKHS]]"
  - "[[SVM]]"
---

# Kernel Methods

## Brief Summary 

> Refs: 李政軒 (https://www.youtube.com/watch?v=p4t6O9uRX-U&list=PLt0SBi1p7xrRKE2us8doqryRou6eDY)

参考教材:
- Kernel Methods for Pattern Analysis, John Shawe-Taylor & Nello Cristianini
- Learning with Kernels, Bernhard Schölkopf & Alexander J. Smola


## Basic Idea

Kernel Method 的核心思想是通过**非线性映射 (Non-linear Mapping)** 将原始数据映射到一个更高维的特征空间 (Feature Space), 使得在该空间中, 原本线性不可分的问题变得线性可分. 然后, 我们可以在这个高维空间中应用线性模型 (例如线性分类器或线性回归) 来处理数据.

具体地, 对于这样的线性不可分数据 $x$, 我们定义一个映射函数 $\phi: \mathbb{R}^d \to \mathcal{H}$, 将数据点 $x$ 映射到高维 Hilbert 空间 $\mathcal{H}$ 中. 此外会定义一个核函数 (Kernel Function) $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$, 用于衡量输入数据点之间的相似性, 即映射后空间中对应点的内积:
$$\kappa(x, z) = \langle \phi(x), \phi(z) \rangle.$$

根据理论可以保证, 任意非线形可分的数据集, 只要选择合适的映射函数 $\phi$ 和核函数 $\kappa$, 就可以将其映射到一个高维空间中, 使得数据在该空间中线性可分. 这使得我们能够利用线性模型来处理复杂的非线性问题.

### Feature Mapping 

![如图是一个经典的线性不可分问题, 但如果我们将数据映射到更高维的空间, 就有可能变成线性可分的问题.](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/Figure_1.png)

- 对于原始 $\mathbb{R}^2$ 空间 (后称 Original Space) 中的数据点 $(x_1, x_2)$, 我们需要使用非线性边界 (例如椭圆 $\frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} = 1$) 来进行分类. 
- 但是如果我们将数据映射到 $\mathbb{R}^3$ 空间 (后称 Feature Space $\mathcal{H}$), 使用映射函数: $\phi: \mathbb{R}^2 \to \mathbb{R}^3$, $(x_1, x_2) \mapsto (z_1, z_2, z_3) = (x_1^2 , x_2^2 , \sqrt{2} x_1 x_2)$, 则此时的椭圆形边界在新空间中变成了一个平面边界: $z_1/a^2 + z_2/b^2 = 1$. 
- 因此, 通过适当的映射函数 $\phi$, 我们可以将原本线性不可分的问题转化为线性可分的问题.

### Kernel Function

同样以上述映射为例, 进一步关注映射前后数据点之间的内积关系:
- 在原始空间中, 两个数据点 $(x_1, x_2)$ 和 $(x_1', x_2')$ 之间的内积为: 
    $$
    \langle x, x' \rangle = x_1 x_1' + x_2 x_2'
    $$
- 在映射后的空间中, 对应的两个数据点 $(z_1, z_2, z_3)$ 和 $(z_1', z_2', z_3')$ 之间的内积为: 
  $$
  \langle \phi(x), \phi(x') \rangle = z_1 z_1' + z_2 z_2' + z_3 z_3' = (x_1^2)(x_1'^2) + (x_2^2)(x_2'^2) + (\sqrt{2} x_1 x_2)(\sqrt{2} x_1' x_2') = (x_1 x_1' + x_2 x_2')^2
  $$

不难发现, 映射后空间中的内积可以通过原始空间中的内积来表示, 简记为 $\kappa (x, x') = \langle \phi(x), \phi(x') \rangle$. 这个函数 $\kappa(x, x')$ 就被称为**核函数 (Kernel Function)**, 在本例中, 具体形式为: $\kappa(x, x') = (x_1 x_1' + x_2 x_2')^2 = ( \langle x, x' \rangle )^2$.

***Definition** (Kernel Function)*. 称函数 $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ 为 $\mathcal{X}$ 上的**核函数 (Kernel Function)**, 若存在 feature mapping $\phi: \mathbb{R}^d \to \mathcal{H}$, 使得对于任意 $x, x'\in \mathcal{X}$, 存在一个 feature mapping $\phi: \mathcal{X} \to \mathcal{H}$, 使得:
$$\kappa(x, x') = \langle \phi(x), \phi(x') \rangle.$$

- 因此, 这种 Kernel Function 变成了高维映射 $\phi$ 的一种替代表示. 只要我们知道核函数 $\kappa(x, x')$, 就等价于知道了映射 $\phi$ 后的内积关系. 因此若我们只关注高维空间中的内积运算, 则无需显式地知道映射 $\phi$ 本身. 这就是**核方法 (Kernel Method)** 的核心思想.

### Geometric Properties from Kernel Function

这里希望说明, 即使我们只知道了 Feature Space 中的内积关系 (即核函数), 也能推导出一些最重要的几何性质, 例如距离和角度. 因此可以进一步说明我们完全可以在不显式知道映射 $\phi$ 的情况下, 进行很多机器学习算法的计算.

**Distance in Feature Space**

定义 Feature Space 中两个点 $\phi(x)$ 和 $\phi(x')$ 之间的距离为:
$$
\begin{aligned}
d^2(\phi(x), \phi(x')) &= \| \phi(x) - \phi(x') \|^2 \\
&= (\phi(x)-\phi(x'))^\top (\phi(x)-\phi(x')) \\
&= \langle \phi(x), \phi(x) \rangle - 2 \langle \phi(x), \phi(x') \rangle + \langle \phi(x'), \phi(x') \rangle \\
&= \kappa(x, x) - 2 \kappa(x, x') + \kappa(x', x')
\end{aligned}
$$

**Angle in Feature Space**

定义 Feature Space 中两个点 $\phi(x)$ 和 $\phi(x')$ 之间的夹角 $\theta$  为:
$$
\begin{aligned}
    \theta &= \arccos  \frac{\langle \phi(x), \phi(x') \rangle}{\|\phi(x)\| \|\phi(x')\|} \\
    &= \arccos  \frac{\kappa(x, x')}{\sqrt{\kappa(x, x) \kappa(x', x')}}
\end{aligned}$$

### Inner Product Matrix / Gram Matrix / Kernel Matrix

给定一组数据点 $\{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\}$, 我们可以构造一个 $m \times m$ 的矩阵 $K$:
$$K =
\begin{bmatrix}
    \kappa(x^{(1)}, x^{(1)}) & \kappa(x^{(1)}, x^{(2)}) & \cdots & \kappa(x^{(1)}, x^{(m)}) \\
    \kappa(x^{(2)}, x^{(1)}) & \kappa(x^{(2)}, x^{(2)}) & \cdots & \kappa(x^{(2)}, x^{(m)}) \\
    \vdots & \vdots & \ddots & \vdots \\
    \kappa(x^{(m)}, x^{(1)}) & \kappa(x^{(m)}, x^{(2)}) & \cdots & \kappa(x^{(m)}, x^{(m)})
\end{bmatrix}
$$

这个矩阵 $K$ 被称为**内积矩阵 (Inner Product Matrix)**, **Gram 矩阵 (Gram Matrix)** 或 **核矩阵 (Kernel Matrix)**. 它包含了所有数据点在 Feature Space 中的内积信息, 因此可以用于计算距离和角度等几何性质.

### Characterization of Kernels

核定理指出, 只要一个函数 $\kappa(x, x')$ 满足有限半正定 (finite positive semi-definite) 的条件, 就存在一个映射 $\phi$, 使得 $\kappa(x, x') = \langle \phi(x), \phi(x') \rangle$. 这意味着我们可以通过设计合适的核函数, 来隐式地定义一个高维映射 $\phi$, 而无需显式地构造该映射.


***Theorem** (Moore-Aronszajn Theorem)*. 设 $\mathcal{X}$ 为任意非空集合, 若函数 $\kappa: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 是对称函数. 则下列命题等价:
1. $\kappa$ 是有限半正定核 (PSD Kernel), 即对于任意有限点集 $\{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\} \subset \mathcal{X}$, 及任意实数系数 $\{c_1, c_2, \ldots, c_m\} \subset \mathbb{R}$, 有:
$$\sum_{i=1}^m \sum_{j=1}^m c_i c_j \kappa(x^{(i)}, x^{(j)}) \geq 0;$$
2. 由 $\kappa$ 定义的 Gram 矩阵 $K$ 为半正定矩阵, $K\succeq 0$;
3. 存在一个 Hilbert 空间 $\mathcal{H}$ 和一个特征映射 $\phi: \mathcal{X} \to \mathcal{H}$, 使得对于任意 $x, x' \in \mathcal{X}$, 有:
$$\kappa(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}.$$

故此定理保证了, 只要我们设计的核函数 $\kappa$ 满足有限半正定的条件, 就一定存在一个对应的高维映射 $\phi$ 和 Hilbert 空间 $\mathcal{H}$, 使得 $\kappa(x, x')$ 可以表示为 $\phi(x)$ 和 $\phi(x')$ 在该空间中的内积. 这为核方法提供了理论基础, 使得我们可以通过核函数来隐式地处理高维映射问题.

### Commonly Used Kernels

以下是一些常用的核函数:

**Polynomial Kernel**: 对于任意 $x, x' \in \mathbb{R}^N$ 及常数 $c \geq 0$ 和整数 $d \geq 1$, 定义多项式核为:
    $$
    \kappa(x, x') = (x^\top x' + c)^d,
    $$
  
- 例如 $N=2, d=2$, 则
    $$\begin{aligned}
    \kappa(x, x') &= (x_1 x_1' + x_2 x_2' + c)^2 \\
    &= x_1^2 x_1'^2 + x_2^2 x_2'^2 + 2 x_1 x_1' x_2 x_2' + 2 c x_1 x_1' + 2 c x_2 x_2' + c^2 \\
    &= \begin{bmatrix} x_1^2 \\ x_2^2 \\ \sqrt{2} x_1 x_2 \\ \sqrt{2 c} x_1 \\ \sqrt{2 c} x_2 \\ c \end{bmatrix}^\top \begin{bmatrix} x_1'^2 \\ x_2'^2 \\ \sqrt{2} x_1' x_2' \\ \sqrt{2 c} x_1' \\ \sqrt{2 c} x_2' \\ c \end{bmatrix}
    \end{aligned}$$
   对应的 feature mapping 为:
   $$\phi\left(\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}\right) = \begin{bmatrix} x_1^2 \\ x_2^2 \\ \sqrt{2} x_1 x_2 \\ \sqrt{2 c} x_1 \\ \sqrt{2 c} x_2 \\ c \end{bmatrix}.$$
- 事实上可以计算出, 对于任意 $N$ 维输入和 $d$ 次多项式核, 对应的 feature mapping $\phi$ 会将输入映射到一个维度为 $\binom{N+d}{d}$ 的空间中.

**Gaussian Kernel / Radiial Basis Function (RBF) Kernel**: 对于任意 $x, x' \in \mathbb{R}^N$ 及常数 $\sigma > 0$, 定义高斯核为:
    $$
    \kappa(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2 \sigma^2}\right).
    $$

**Laplacian Kernel**: 对于任意 $x, x' \in \mathbb{R}^N$ 及常数 $\sigma > 0$, 定义拉普拉斯核为:
    $$
    \kappa(x, x') = \exp\left(-\frac{\|x - x'\|_1}{\sigma}\right).
    $$

**Sigmoid Kernel**: 对于任意 $x, x' \in \mathbb{R}^N$ 及常数 $\alpha > 0$ 和 $c \in \mathbb{R}$, 定义 Sigmoid 核为:
    $$
    \kappa(x, x') = \tanh(\alpha x^\top x' + c).
    $$

## Dual Representation 

考虑在 feature space $\mathcal{H}$ 中的线性函数 $f(x) = w^\top \phi(x) + b$, 其中 $w \in \mathcal{H}$, $b \in \mathbb{R}$. 然而我们能够否仅通过核函数 $\kappa(x, x')$ 来表示这个函数 $f(x)$ 呢?

***Theorem** (Representer Theorem)*.  给定训练数据集 $\{(x^{(i)}, y^{(i)})\}_{i=1}^N$, 假设我们希望最小化以下正则化经验风险:
$$J(w) = \sum_{i=1}^N L(y^{(i)}, f(x^{(i)})) + \lambda \|w\|^2,$$
其中 $L$ 是损失函数, $\lambda > 0$ 是正则化参数. 则最优解 $w^*$ 可以表示为训练数据点在 feature space 中的线性组合:
$$w^* = \sum_{i=1}^N \alpha_i \phi(x^{(i)}),$$
其中 $\alpha_i \in \mathbb{R}$ 是系数.

- 换言之, 我们求解的最优权重向量 $w^*$ 可以完全由训练数据点的映射 $\phi(x^{(i)})$ 线性组合而成, 即 $w^* \in \text{span}\{x^{(1)}, x^{(2)}, \ldots, x^{(N)}\}$ 落在训练数据点的张成空间中.
- 这意味着我们可以将原始的优化问题 (求解 $w^*\in \mathcal{H}$) 转化为关于系数 $\alpha_i$ 的优化问题 (求解 $\boldsymbol{\alpha} \in \mathbb{R}^N$), 这在 $N \ll \dim(\mathcal{H})$ 时尤其有用, 甚至当 $\mathcal{H}$ 是无限维空间时也适用.

故将 $w^*$ 代入线性函数 $f(x)$ 中, 可得 Dual Representation:
$$\begin{aligned}
f(\phi(x)) &= w^{*\top} \phi(x) + b \\
&= \left(\sum_{i=1}^N \alpha_i \phi(x^{(i)})\right)^\top \phi(x) + b \\
&= \sum_{i=1}^N \alpha_i \langle \phi(x^{(i)}), \phi(x) \rangle + b \\
&= \sum_{i=1}^N \alpha_i \kappa(x^{(i)}, x) + b
\end{aligned}$$

- 同样地, 我们成功地将线性函数 $f(x)$ 表示为核函数 $\kappa(x^{(i)}, x)$ 的线性组合, 其中系数为 $\alpha_i$, 这种表示方式被称为**对偶表示 (Dual Representation)**. 这使得我们可以在不显式计算映射 $\phi$ 的情况下, 利用核函数来进行预测和学习. 其极大地简化了计算过程, 特别是在高维或无限维的 feature space 中.

## Kernel Trick

综上, 我们可以通过核函数 $\kappa(x, x')$ 来隐式地处理高维映射 $\phi$, 这就是所谓的**核技巧 (Kernel Trick)**. 具体来说:
- 通过 feature mapping $\phi$, 我们可以将原始空间中的数据点映射到高维空间 $\mathcal{H}$
- 在高维空间中, 我们可以使用线性模型 (例如线性分类器或线性回归) 来处理数据
- 通过核函数 $\kappa(x, x')$, 我们可以直接计算高维空间中的内积, 而无需显式地计算映射 $\phi$
- 这使得我们能够在高维空间中进行有效的计算, 通过 dual representation, 我们可以将模型表示为核函数的线性组合, 得到更简洁的形式. 


## Application: Kernel-Based Linear Regression Model

### Problem Setup

给定训练数据集 $\{(\boldsymbol{x_i}, y_i)\}_{i=1}^N$, 其中 $\boldsymbol{x_i}\in \mathbb{R}^d$ 为输入特征, $y_i \in \mathbb{R}$ 为对应的目标值. 此外, 给定一个核函数 $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$, 用于衡量输入特征之间的相似性, 对应的映射函数为 $\phi: \mathbb{R}^d \to \mathcal{H}$, 其中 $\mathcal{H}$ 为高维特征空间.

得到的输入矩阵为:
$$
{\Phi} := \begin{bmatrix} 1 & \phi(\boldsymbol{x_1})^\top \\ 1 & \phi(\boldsymbol{x_2})^\top \\ \vdots & \vdots \\ 1 & \phi(\boldsymbol{x_N})^\top \end{bmatrix} \in \mathbb{R}^{N \times (D+1)}
$$
输出标签仍为 $\boldsymbol{y} = [y_1, y_2, \ldots, y_N]^\top \in \mathbb{R}^N$ 不变. 

假设的线性回归模型为:
$$
\boldsymbol{y} = {\Phi} \boldsymbol{w} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I)
$$

对应的 OLS 问题为: $\min_{\boldsymbol{w}}\;\|\boldsymbol{y}-\Phi\boldsymbol{w}\|_2^2$, 其 normal equation 为:
$$
\Phi^\top\Phi\,\hat{\boldsymbol{w}}=\Phi^\top\boldsymbol{y}.
$$
若 ${\Phi}^\top {\Phi}$ 可逆, 则有闭式解:
$$
\hat{\boldsymbol{w}} = \left({\Phi}^\top {\Phi}\right)^{-1} {\Phi}^\top \boldsymbol{y}.
$$

### Dual Representation of Kernel Linear Regression    

根据 Representer Theorem, 最优解 $\boldsymbol{w^*}$ 可以表示为训练数据点在特征空间中的线性组合, 即存在系数向量 $\boldsymbol{\alpha} \in \mathbb{R}^N$, 使得 (注意此处没有唯一性要求):
$$\boldsymbol{w^*} = {\Phi}^\top \boldsymbol{\alpha}.$$

为了求解 $\boldsymbol{\alpha}$, 我们将 $\boldsymbol{w^*}$ 代入 normal equation:
$$\|\boldsymbol{y}-\Phi\boldsymbol{w^*}\|_2^2 = \|\boldsymbol{y}-\Phi{\Phi}^\top \boldsymbol{\alpha}\|_2^2.$$
定义核矩阵 (Kernel Matrix) $K = \Phi \Phi^\top \in \mathbb{R}^{N \times N}$, 其中 $K_{ij} = 1 + \kappa(\boldsymbol{x_i}, \boldsymbol{x_j})$. 则上式可写为 dual 表示:
$$
\min_{\boldsymbol{\alpha}} \|\boldsymbol{y} - K \boldsymbol{\alpha}\|_2^2.
$$
对应的 normal equation 为:
$$
K^\top K \,\hat{\boldsymbol{\alpha}} = K^\top \boldsymbol{y}.
$$
若 $K$ 可逆, 则有闭式解:
$$
\hat{\boldsymbol{\alpha}} = K^{-1}  \boldsymbol{y}.
$$

对于任意新的输入样本 $\boldsymbol{x_*}$, 其预测值为:
$$\begin{aligned}
\hat{y_*} &= \phi(\boldsymbol{x_*})^\top \boldsymbol{w^*} \\
&= \phi(\boldsymbol{x_*})^\top {\Phi}^\top \hat{\boldsymbol{\alpha}} \\
&= \sum_{i=1}^N \hat{\alpha_i} \left(1 + \kappa(\boldsymbol{x_i}, \boldsymbol{x_*})\right).
\end{aligned}$$

此即为 kernel-based linear regression model 的预测公式.

注意: 在实践中, 往往通过 KRR (Kernel Ridge Regression) 来避免核矩阵 $K$ 不可逆的问题, 即在目标函数中加入正则化项 $\lambda \|\boldsymbol{\alpha}\|_2^2$:
$$
\boldsymbol{\alpha} = (K + \lambda I)^{-1} \boldsymbol{y}.
$$


## Reproducting Kernel Hilbert Space (RKHS)

关于 RKHS, 我们最终想得到如下目标:
1. 从一个核函数 $\kappa$ 出发, 构造出一个以函数为元素的 Hilbert 空间 $\mathcal{H}$;
2. 该 Hilbert 空间 $\mathcal{H}$ 可以定义一个内积 $\langle \cdot, \cdot \rangle_{\mathcal{H}}$, 使得 $\kappa(x,z) = \langle \kappa(\cdot, x), \kappa(\cdot, z) \rangle_{\mathcal{H}}$;
3. 在该内积下, 任意函数 $f \in \mathcal{H}$ 满足**再生性质 (Reproducing Property)**, 即其点值可以通过内积表示: $f(x) = \langle f, \kappa(\cdot, x) \rangle_{\mathcal{H}}$.

为了达到这一目标, 其整体思路为:
1. 首先定义 PSD 核函数 $\kappa$ 和对应的核截断 (Kernel Section) $\kappa(\cdot, x)$;
2. 利用核截断构造一个函数线性空间 $\mathcal{F}_\kappa$;
3. 在该函数空间上定义由 kernel 诱导的内积;
4. 用这个内积直接推出 reproducing property;
5. 最后对该空间进行完备化, 得到 RKHS.

### Preliminaries

- 实对称矩阵 $A \in \mathbb{R}^{n \times n}$ 存在正交矩阵 $Q$ 和对角矩阵 $\Lambda$ 使得 $A = Q \Lambda Q^\top$, 其中 $\Lambda$ 的对角线元素为 $A$ 的特征值, $Q$ 的列向量为对应的单位特征向量.
- 实对称矩阵 $A$ 半正定的充分必要条件是其所有特征值均非负.
- 对于一个函数 $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$, 若对于任意有限点集 $\{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\} \subset \mathbb{R}^d$, 构造的核矩阵 $K$ 半正定, 则称 $\kappa$ 为有限半正定函数.
- 给定 kernel 函数 $\kappa$, 符合 Cauchy-Schwarz Inequality: 对于 $x,z\in\mathbb{R}^d$, 则 $|\kappa(x,z)|^2 \leq \kappa(x,x) \kappa(z,z)$.
  - *Proof*: 对于 $\kappa$, 其对应的 Kernel Matrix $K = \begin{bmatrix} \kappa(x,x) & \kappa(x,z) \\ \kappa(z,x) & \kappa(z,z) \end{bmatrix}$ 半正定, 则 $\det(K) = \kappa(x,x)\kappa(z,z) - |\kappa(x,z)|^2 \geq 0$, 故结论成立.
  - 该不等式也可以看作是内积空间中的 Cauchy-Schwarz Inequality 的推广: $|\langle\phi(x), \phi(z)\rangle|^2 \leq \|\phi(x)\|^2 \|\phi(z)\|^2$.
- 记 $\mathbb{R}^\mathcal{X}$ 为所有从集合 $\mathcal{X}$ 到实数域 $\mathbb{R}$ 的函数所构成的空间, 即:
$$\mathbb{R}^\mathcal{X} = \{ f \mid f: \mathcal{X} \to \mathbb{R} \}.$$

- ***Definition** (Vector Space over $\mathbb{R}$)*. 设 $V$ 为一个 $\mathbb{R}$ 上 的非空集合, 若在 $V$ 上定义了加法运算 $+: V \times V \to V$ 和数乘运算 $\cdot: \mathbb{R} \times V \to V$, 且满足以下公理, 则称 $V$ 为**实向量空间 (Vector Space over $\mathbb{R}$)**:
  1. 加法交换律: 对于任意 $u,v\in V$, 有 $u+v = v+u$.
  2. 加法结合律: 对于任意 $u,v,w\in V$, 有 $(u+v)+w = u+(v+w)$.
  3. 存在加法单位元: 存在元素 $0\in V$, 使得对于任意 $v\in V$, 有 $v+0 = v$.
  4. 存在加法逆元: 对于任意 $v\in V$, 存在元素 $-v\in V$, 使得 $v + (-v) = 0$.
  5. 数乘结合律: 对于任意 $a,b\in\mathbb{R}$ 和 $v\in V$, 有 $a(bv) = (ab)v$.
  6. 数乘分配律: 对于任意 $a,b\in\mathbb{R}$ 和 $u,v\in V$, 有 $(a+b)v = av + bv$ 和 $a(u+v) = au + av$.
  7. 数乘分配律2: 对于任意 $a\in\mathbb{R}$ 和 $u,v\in V$, 有 $a(u+v) = au + av$.
  8. 数乘单位元: 对于任意 $v\in V$, 有 $1 \cdot v = v$.

### Kernel Section and Function Space Induced by Kernel

***Definition** (Kernel Section)*.  给定核函数 $\kappa: \mathcal{X}\times\mathcal{X} \to \mathbb{R}$, 对于任意给定的 $x \in \mathcal{X}$, 定义**核截断 (Kernel Section)** 为函数 $\kappa(\cdot, x): \mathcal{X} \to \mathbb{R}$, 其作用为:
$$[\kappa(\cdot, x)](z) = \kappa(z, x), \quad \forall z \in \mathcal{X}.$$

- 这个函数 $\kappa(\cdot, x)$ 可以看作是核函数 $\kappa$ 固定一个参数 $x$ 后得到的函数, 它将任意输入 $z$ 映射到实数 $\kappa(z, x)$. 可以类比分布函数, 如给定指数分布的参数 $\lambda$, 则对应了一个具体的分布函数 $f(x; \lambda) = \lambda e^{-\lambda x}$.
- 定义对应的映射 $\phi_x: \mathcal{X} \to \mathbb{R}^\mathcal{X}$, 使得 $\phi_x(z) = \kappa(z, x)$, 则 $\kappa(\cdot, x)$ 可以看作是映射 $\phi_x$ 的输出.
- 总而言之, 此处的输入为一个点 $x\in \mathcal{X}$, 输出为一个函数 $\kappa(\cdot, x) \in \mathbb{R}^\mathcal{X}$. 这种映射是为了让 $\kappa(\cdot, x) $ 可以被当作空间中的向量来处理.


***Definition** (Span of Kernel Sections)*.  给定核函数 $\kappa: \mathcal{X}\times\mathcal{X} \to \mathbb{R}$, 对每个 $x \in \mathcal{X}$ 定义核截断 $\kappa(\cdot, x)$. 则**核截断的张成空间 (Span of Kernel Sections)** 定义为:
$$\mathcal{F}_{\kappa} = \left\{ f: \mathcal{X} \to \mathbb{R} \middle | 
f(\cdot) = \sum_{i=1}^m \alpha_i \kappa(\cdot, x^{(i)}), \; m \in \mathbb{N}, \;\alpha_i \in \mathbb{R}, \; x^{(i)} \in \mathcal{X} \right\}.$$ 
- $\mathcal{F}_{\kappa}$: 用核提供的一族函数作为生成元, 构造最小的线性空间.
- 等价地, $f\in \mathcal{F}_{\kappa}$ 当且仅当存在 $n, \alpha_i, x^{(i)}$ 使得对于任意 $z\in \mathcal{X}$, 有: $f(z) = \sum_{i=1}^m \alpha_i \kappa(z, x^{(i)}).$
- 可以证明, $\mathcal{F}_{\kappa}$ 在函数加法和数乘下构成一个实向量空间.

### Inner Product Induced by Kernel

我们已经定义了向量空间 $\mathcal{F}_\kappa=\left\{\sum_{i=1}^n \alpha_i \kappa(\cdot,x_i)\right\}$. 对于每个 $x\in\mathcal{X}$, 我们有一个函数 (函数) $k_x(\cdot)=\kappa(\cdot,x) \in \mathbb{R}^{\mathcal{X}}$. 并且 $\mathcal{F}_\kappa$ 就是这些生成元 $k_x$ 的有限线性组合所构成的空间. 我们希望接着在该空间上定义一个内积. 

首先规定 Kernel Sections 之间的内积. 对于任意 $x,z\in\mathcal{X}$, 定义
$$\langle k_x,k_z\rangle_{\mathcal{F}_k} := \langle \kappa(\cdot,x),\kappa(\cdot,z)\rangle_{\mathcal{F}_\kappa}:=\kappa(x,z).$$

那么取两个一般元素(即函数):
$$
f = \sum_{i=1}^m \alpha_i k_x^{(i)} = \sum_{i=1}^m \alpha_i \kappa(\cdot,x^{(i)}), \quad
g = \sum_{j=1}^n \beta_j k_z^{(j )} = \sum_{j=1}^n \beta_j \kappa(\cdot,z^{(j)}).
$$
为满足线性空间的内积定义的线形性要求, 可得到:
$$
\begin{aligned}
\langle f,g\rangle_{\mathcal{F}_k} &= \sum_{i=1}^m \sum_{j=1}^n \alpha_i \beta_j \kappa(x^{(i)},z^{(j)}) = \sum_{i=1}^m \sum_{j=1}^n \alpha_i \beta_j \langle k_{x^{(i)}}, k_{z^{(j)}} \rangle_{\mathcal{F}_k} = \boldsymbol{\alpha}^\top K_{XZ} \boldsymbol{\beta},\end{aligned}
$$
- 其中 $K_{XZ} \in \mathbb{R}^{m \times n}$ 为核矩阵, 其元素为 $K_{XZ}^{ij} = \kappa(x^{(i)}, z^{(j)})$; $\boldsymbol{\alpha} = [\alpha_1, \alpha_2, \ldots, \alpha_m]^\top$, $\boldsymbol{\beta} = [\beta_1, \beta_2, \ldots, \beta_n]^\top$.

- 可以验证, 该内积满足对称性, 线性性和正定性等内积的基本性质. 特别地, 正定性来源于核函数 $\kappa$ 的有限半正定性质.

### Reproducing Property

给定一个内积函数空间 $(\mathcal{H}, \langle \cdot, \cdot \rangle_{\mathcal{H}})$, reproducing property 指的是对于任意点 $x \in \mathcal{X}$, 存在空间中的一个特殊函数 $k_x(\cdot)$, 使得对于任意函数 $f \in \mathcal{H}$, 有:
$$f(x) = \langle f, k_x \rangle_{\mathcal{H}}.$$
- 该性质表明, 函数 $f$ 在点 $x$ 处的取值可以通过函数 $f$ 与特殊函数 $k_x$ 之间的内积来表示. 这使得我们可以将函数的点值信息转化为内积信息, 从而方便地进行分析和计算.

而在 RKHS 中, 这个特殊函数 $k_x$ 就是核截断 $k_x(\cdot) = \kappa(\cdot, x)$. 故
$$f(x) = \langle f, k_x \rangle_{\mathcal{H}_\kappa} = \langle f, \kappa(\cdot, x) \rangle_{\mathcal{H}_\kappa}.$$

具体推导证明如下. 

回顾, 对于线性空间 $\mathcal{F}_\kappa=\left\{f(\cdot)=\sum_{i=1}^m \alpha_i \kappa(\cdot,x_i)\right\}$, 定义记号 $k_x(\cdot)=\kappa(\cdot,x)$, 且定义内积 $\left\langle \sum_{i=1}^m \alpha_i k_{x_i},\sum_{j=1}^n \beta_j k_{z_j}\right\rangle_{\mathcal{F}_\kappa}=\sum_{i=1}^m\sum_{j=1}^n \alpha_i\beta_j \kappa(x_i,z_j).$ 取任意元素(函数) $f\in \mathcal{F}_\kappa$, 将其写作
$$f(\cdot)=\sum_{i=1}^m \alpha_i k_{x_i}(\cdot)=\sum_{i=1}^m \alpha_i \kappa(\cdot,x_i).$$
下欲说明, 对于任意 $x\in \mathcal{X}$, 有
$$
f(x) = \langle f, k_x \rangle_{\mathcal{F}_\kappa}.
$$

- 该式 LHS:
    $$
    \begin{aligned}
    f(x) &= \sum_{i=1}^m \alpha_i \kappa(x, x_i).
    \end{aligned}$$

- 该式 RHS,  由于线性性, 有
    $$\begin{aligned}
    \langle f, k_x \rangle_{\mathcal{F}_\kappa} &= 
    \left\langle \sum_{i=1}^m \alpha_i k_{x_i}, 1\cdot k_x \right\rangle_{\mathcal{F}_\kappa} \\
    &= \sum_{i=1}^m \alpha_i  \cdot \kappa(x_i, x) \\
    \end{aligned}$$

故只要 Kernel Function $\kappa$ 满足对称性 (即 $\kappa(x,z) = \kappa(z,x)$), 则有 $f(x) = \langle f, k_x \rangle_{\mathcal{F}_\kappa}$. 这就证明了再生性质.


### Representer Theorem

对于典型的正则经验风险最小化问题, 可以形式化给出 RKHS 的模型结构:
$$
\min_{f\in\mathcal{H}_\kappa}\ \sum_{i=1}^n L(y_i,f(x_i))+\lambda \|f\|_{\mathcal{H}_\kappa}^2.
$$

Representer Theorem 保证了最优解 $f^*$ 可以表示为训练数据点的核截断的线性组合:
$$f^*(\cdot) = \sum_{i=1}^n \alpha_i \kappa(\cdot, x_i).$$

即我们不需要在整个无限维空间 $\mathcal{H}_\kappa$ 中搜索最优函数, 只需在由训练数据点生成的有限维子空间中搜索即可. 这极大地简化了优化问题, 并使得核方法在实际应用中变得可行.

而新的输入点 $x_*$ 的预测值为:
$$\begin{aligned}
f^*(x_*) &= \sum_{i=1}^n \alpha_i \kappa(x_*, x_i).
\end{aligned}$$

RKHS 范数可以表示为:
$$\begin{aligned}
\|f^*\|_{\mathcal{H}_\kappa}^2 &= \left\langle \sum_{i=1}^n \alpha_i \kappa(\cdot, x_i), \sum_{j=1}^n \alpha_j \kappa(\cdot, x_j) \right\rangle_{\mathcal{H}_\kappa} \\
&= \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j \kappa(x_i, x_j) \\
&= \boldsymbol{\alpha}^\top K \boldsymbol{\alpha},
\end{aligned}$$
其中 $K \in \mathbb{R}^{n \times n}$ 为核矩阵, 其元素为 $K_{ij} = \kappa(x_i, x_j)$.

而原问题可转化为关于系数 $\boldsymbol{\alpha} \in \mathbb{R}^n$ 的优化问题:
$$\min_{\boldsymbol{\alpha} \in \mathbb{R}^n}\ \sum_{i=1}^n L\left(y_i, \sum_{j=1}^n \alpha_j \kappa(x_i, x_j)\right) + \lambda \boldsymbol{\alpha}^\top K \boldsymbol{\alpha}.$$

例如, 考虑 Kernel Ridge Regression (KRR). 取平方损失
$$L(y, f(x)) = (y - f(x))^2,$$
则优化问题为:
$$\min_{\boldsymbol{\alpha} \in \mathbb{R}^n}\ \sum_{i=1}^n \left(y_i - \sum_{j=1}^n \alpha_j \kappa(x_i, x_j)\right)^2 + \lambda \boldsymbol{\alpha}^\top K \boldsymbol{\alpha}= \min_{\boldsymbol{\alpha} \in \mathbb{R}^n}\ \|\boldsymbol{y} - K \boldsymbol{\alpha}\|_2^2 + \lambda \boldsymbol{\alpha}^\top K \boldsymbol{\alpha}.$$

对 $\boldsymbol{\alpha}$ 求导并令其为零, 可得闭式解:
$$\hat{\boldsymbol{\alpha}} = (K + \lambda I)^{-1} \boldsymbol{y}.$$

对新数据点 $x_*$ 的预测为:
$$\hat{y_*} = \sum_{i=1}^n \hat{\alpha_i} \kappa(x_*, x_i) = \begin{bmatrix} \kappa(x_*, x_1) & \kappa(x_*, x_2) & \ldots & \kappa(x_*, x_n) \end{bmatrix} \hat{\boldsymbol{\alpha}}.$$

因此在机器学习语境中, RKHS 提供了一个强大的框架, 使得我们能够利用核函数来构建复杂的非线性模型, 同时保持计算的可行性和效率. 总的而言, 就是说将预测函数 $f$ 限制在由某个核函数 $\kappa$ 所生成的 RKHS 中, 并通过最小化经验风险损失加上 RKHS 范数的正则化项来学习模型.