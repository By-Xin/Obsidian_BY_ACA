#formulation 
## 详细推导

在 [[Autoloss 优化部分推导 (diffopt 文档解读)]] 中已经对整体问题进行了叙述. 然而 Autoloss 中的内容是 generalized 的. 我们需要具体到我们目前的 ReHLine 损失函数的任务中. 目前的终极目标就在于隐函数求导来求解这个双层的优化问题. 我们这里从一个 toy model 出发, 只考虑 ReHLine 中的 ReLU, 且不加任何线性约束, 不过仍然考虑要保留正则项. 在当前任务下如何进行高效的数值计算. 若该数值计算有解决思路, 我们或许就可以进一步进行迁移.

### 1. ReHLine 模型与 Primal 问题定义


暂时先考虑纯粹的 ReLU, 并且不考虑在 regression 的具体场景. 目前只当作一个优化问题来解决. 

考虑如下简化 ReHLine 模型:
$$\min_{\boldsymbol{\beta}\in\mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}(u_{l,i} \mathbf{x}_i^\top \boldsymbol{\beta} + v_{l,i})+\frac12 \|\boldsymbol{\beta}\|^2$$

其中 $U = (u_{l,i}) , V = (v_{l,i}) \in \mathbb{R}^{L \times n}$ 是模型参数, $L$ 是 ReLU 的个数, $n$ 是样本的个数, $\text{ReLU}(x) = \max(0,x)$ 是 ReLU 函数. $\mathbf{x}_i \in \mathbb{R}^d$ 是输入特征. 


---

根据 ReLU 的等价变换, 进一步定义最终要求解的 Primal 问题为:
$$\begin{aligned} \min_{\boldsymbol{\beta},\Pi} &\quad \sum_{i=1}^n \sum_{l=1}^L \pi_{l,i} + \frac12 \|\boldsymbol{\beta}\|^2 \\ \text{s.t. } &\quad  u_{l,i} \mathbf{x}_i^\top \boldsymbol{\beta} + v_{l,i} - \pi_{l,i} \geq 0,\quad \forall i,l \\ &\quad  \pi_{l,i} \geq 0,\quad \forall i,l\end{aligned}$$

其中 $\Pi = (\pi_{l,i}) \in \mathbb{R}^{L \times n}$ 是 ReLU 的松弛变量.


---
### 2. Lagrangian 及其 KKT 条件

写出上述问题的 Lagrangian:
$$\begin{aligned} \mathcal{L}_{\mathcal P}(\boldsymbol{\beta}, \Pi;  \Lambda, \Delta) = & \sum_{i=1}^n \sum_{l=1}^L \pi_{l,i}  + \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 \\ &  - \sum_{i=1}^n \sum_{l=1}^L \lambda_{l,i} (\pi_{l,i} - u_{l,i} \mathbf{x}_i^\top \boldsymbol{\beta} - v_{l,i}) \\ & - \sum_{i=1}^n \sum_{l=1}^L \delta_{l,i} \pi_{l,i} \end{aligned}$$

其中 $\Lambda = (\lambda_{l,i}) \in \mathbb{R}^{L \times n}$ 和 $\Delta = (\delta_{l,i}) \in \mathbb{R}^{L \times n}$ 是 Lagrange 乘子.

---

写出上述问题的 KKT 条件. 

#### ***Stationary 条件***

*1. 对于 $\Pi$ 求导*

$$\frac{\partial \mathcal{L}_{\mathcal P}}{\partial \pi_{l,i}} = \frac{\partial}{\partial \pi_{l,i}} ( \pi_{l,i} - \lambda_{l,i} \pi_{l,i} - \delta_{l,i} \pi_{l,i} + \text{others}) = 1 - \lambda_{l,i} - \delta_{l,i} = 0 .$$

故 $\lambda_{l,i} = 1 - \delta_{l,i}\in [0, 1]$, 其中 $\delta_{l,i} \ge 0$ 是松弛变量的拉格朗日乘子. 因此有约束: 
$\boldsymbol{1}_{L \times n} \ge \boldsymbol{\Lambda} \ge \boldsymbol{0}_{L \times n}.$

*2. 对于 $\beta$ 求导*

考虑所有含 $\boldsymbol{\beta}$ 的项:
$$\begin{aligned} \mathcal{L}_{\mathcal P}^{\boldsymbol{\beta}} &= \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 + \sum_{i=1}^n \sum_{l=1}^L \lambda_{l,i} u_{l,i} \mathbf{x}_i^\top \boldsymbol{\beta} \\ &= \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 + \sum_{l=1}^L \left( \sum_{i=1}^n \lambda_{l,i} u_{l,i} \mathbf{x}_i^\top \right)^\top \boldsymbol{\beta} \\&= \frac{1}{2} \|\boldsymbol{\beta}\|_2^2 +  \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda})   \boldsymbol{\beta}.\end{aligned}$$

最后一步是通过引入 tensor 记号, 记 $\mathsf{\bar{U}} = (u_{lij}):=(u_{li}x_{ij})\in \mathbb{R}^{L\times n\times d}$,  则其 mode-3 unfolding 分别为 $\mathsf{\bar{U}}_{(3)} \in\mathbb{R}^{d\times nL}$. 为便于区分, 记 $x^{(i)}_j$ 为 $\mathbf{x}_i$ 的第 $j$ 个分量 (第 $j$ 维特征), 即 $\mathbf{x}_i = (x^{(i)}_1, x^{(i)}_2, \ldots, x^{(i)}_d)^\top$. 具体地:
$$\begin{aligned} \mathsf{\bar{U}}_{(3)}&= 
\begin{bmatrix}
| & | & & | & | & & | \\
u_{11} \mathbf{x}_1 & u_{12} \mathbf{x}_2 & \cdots & u_{1n} \mathbf{x}_n & u_{21} \mathbf{x}_1 & \cdots & u_{Ln} \mathbf{x}_n \\
| & | & & | & | & & |
\end{bmatrix} \\
&=\begin{bmatrix}
u_{11} x^{(1)}_1 & u_{12} x^{(2)}_1 & \cdots & u_{1n} x^{(n)}_1 & u_{21} x^{(1)}_1 & \cdots & u_{Ln} x^{(n)}_1 \\
u_{11} x^{(1)}_2 & u_{12} x^{(2)}_2 & \cdots & u_{1n} x^{(n)}_2 & u_{21} x^{(1)}_2 & \cdots & u_{Ln} x^{(n)}_2 \\
\vdots & \vdots & \cdots & \vdots & \vdots & \cdots & \vdots \\
u_{11} x^{(1)}_d & u_{12} x^{(2)}_d & \cdots & u_{1n} x^{(n)}_d & u_{21} x^{(1)}_d & \cdots & u_{Ln} x^{(n)}_d
\end{bmatrix}\in \mathbb{R}^{d \times nL}
\end{aligned}$$


故有如下等价变换:
$$\sum_{l,i}\lambda_{li}u_{li} \mathbf{x}_i = \mathsf{\bar{U}}_{(3)}\text{vec}(\boldsymbol{\Lambda}) $$

> 这里再次对上述记号进行 verify: 已知 $\text{vec}(\Lambda^*) = \begin{bmatrix} \lambda_{1,1}^* \\ \vdots \\ \lambda_{L,n}^* \end{bmatrix} \in \mathbb{R}^{Ln}$, 故
$$\begin{aligned}\mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}) &= \begin{bmatrix}
u_{11} x^{(1)}_1  & \cdots & u_{1n} x^{(n)}_1 & u_{21} x^{(1)}_1 & \cdots & u_{Ln} x^{(n)}_1 \\
u_{11} x^{(1)}_2  & \cdots & u_{1n} x^{(n)}_2 & u_{21} x^{(1)}_2 & \cdots & u_{Ln} x^{(n)}_2 \\
\vdots  & \cdots & \vdots & \vdots & \cdots & \vdots \\
u_{11} x^{(1)}_d  & \cdots & u_{1n} x^{(n)}_d & u_{21} x^{(1)}_d & \cdots & u_{Ln} x^{(n)}_d
\end{bmatrix} \begin{bmatrix} \lambda_{1,1}^* \\ \vdots \\ \lambda_{L,n}^* \end{bmatrix} \\
&= \begin{bmatrix}
u_{11} x^{(1)}_1 \lambda_{1,1}^* + \cdots  + u_{Ln} x^{(n)}_1 \lambda_{L,n}^* \\
u_{11} x^{(1)}_2 \lambda_{1,1}^* + \cdots  + u_{Ln} x^{(n)}_2 \lambda_{L,n}^* \\
\vdots  \\
u_{11} x^{(1)}_d \lambda_{1,1}^* + \cdots + u_{Ln} x^{(n)}_d \lambda_{L,n}^*
\end{bmatrix} = \begin{bmatrix}
\sum_{i=1}^n \sum_{l=1}^L \lambda_{l,i} u_{l,i} x^{(i)}_1 \\
\sum_{i=1}^n \sum_{l=1}^L \lambda_{l,i} u_{l,i} x^{(i)}_2 \\
\vdots \\
\sum_{i=1}^n \sum_{l=1}^L \lambda_{l,i} u_{l,i} x^{(i)}_d
\end{bmatrix} \\& = \sum_{l=1}^L \sum_{i=1}^n \lambda_{l,i} u_{l,i} \begin{bmatrix} x^{(i)}_1 \\ x^{(i)}_2 \\ \vdots \\ x^{(i)}_d \end{bmatrix} = \sum_{l=1}^L \sum_{i=1}^n \lambda_{l,i} u_{l,i} \mathbf{x}_i
\end{aligned}$$

因此对 $\boldsymbol{\beta}$ 求导并令其为 0, 有:
$$\boldsymbol{\beta}^* = -  \sum_{i=1}^n\sum_{l}\lambda_{li}u_{li} \mathbf{x}_i= -\mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}) $$

再将其代回 Lagrangian 中, 得到:
$$\begin{aligned}\mathcal{L}_{\mathcal P}^{\boldsymbol{\beta}^*} &= \frac{1}{2} \|\boldsymbol{\beta}^*\|_2^2 - \boldsymbol{\beta}^{*\top} g^* =-\frac{1}{2} \|\boldsymbol{\beta}^*\|_2^2 \\
\\&= -\frac{1}{2} \text{vec}(\boldsymbol{\Lambda})^\top \mathsf{\bar{U}}_{(3)}^\top \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}) \end{aligned}$$

将 $\boldsymbol{\beta}^*$ 代入 Lagrangian 中, 得到:
$$\begin{aligned}
g(\Lambda) = \inf_{\boldsymbol{\beta}} \mathcal{L}_{\mathcal{P}} = -\frac{1}{2} \text{vec}(\boldsymbol{\Lambda})^\top \mathsf{\bar{U}}_{(3)}^\top \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}) + \mathrm{Tr}(\boldsymbol{\Lambda} \mathbf{V}^\top)
\end{aligned}$$

其中 $\mathbf{V} = (v_{l,i}) \in \mathbb{R}^{L \times n}$, 故 $\mathrm{Tr}(\boldsymbol{\Lambda} \mathbf{V}^\top) = \sum_{l=1}^L \sum_{i=1}^n \lambda_{l,i} v_{l,i}$.

---

#### ***Complementary Slackness 条件***

对于每个 $l,i$, 有:
$$\begin{aligned}\lambda_{l,i} (\pi_{l,i} - u_{l,i} \mathbf{x}_i^\top \boldsymbol{\boldsymbol{\beta}} - v_{l,i}) = 0\\
\end{aligned}$$


---
### 3. KKT 条件的向量化

因此对于一组最优解 $(\boldsymbol{\beta}^*, \boldsymbol{\Pi}^*, \boldsymbol{\Lambda}^*)$, 有如下关系成立:

$$\begin{aligned} 
\boldsymbol{\beta}^* +  \sum_{i=1}^n\sum_{l}\lambda^*_{li}u_{li} \mathbf{x}_i =  \boldsymbol{\beta}^*+ \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}) &= \mathbf{0}_{d \times 1} \\
\lambda^*_{l,i}(\pi_{l,i}^* - u_{l,i} \mathbf{x}_i^\top \boldsymbol{\beta}^* - v_{l,i}) &= 0, \quad \forall l,i \quad (2)\\
\delta^*_{l,i} \pi_{l,i}^* =(1-\lambda^*_{l,i})\pi_{l,i}^* &= 0, \quad \forall l,i\quad(3)\\
\end{aligned}$$

上述一共包含 $d+Ln + Ln$ 个方程. 这也与 $(\Pi^*, \boldsymbol{\beta}^*, \Lambda^*)$ 的变量个数能够对应. 下尝试通过向量化进行整理.

记 (部分记号前面已经出现, 这里进行整理):
$$\begin{aligned}
\text{vec}(\Pi^*) &= \begin{bmatrix} \pi_{1,1}^* \\ \vdots \\ \pi_{L,n}^* \end{bmatrix} \in \mathbb{R}^{Ln}, \quad \text{vec}(\Lambda^*) = \begin{bmatrix} \lambda_{1,1}^* \\ \vdots \\ \lambda_{L,n}^* \end{bmatrix} \in \mathbb{R}^{Ln}, \\
\text{vec}(U) &= \begin{bmatrix} u_{1,1} \\ \vdots \\ u_{L,n} \end{bmatrix} \in \mathbb{R}^{Ln}, \quad \text{vec}(V) = \begin{bmatrix} v_{1,1} \\ \vdots \\ v_{L,n} \end{bmatrix} \in \mathbb{R}^{Ln},\\
\text{vec}(\theta) &= \begin{bmatrix} \text{vec}(U) \\ \text{vec}(V) \end{bmatrix} \in \mathbb{R}^{2Ln}.\\
\end{aligned}$$

**对于 $(2)$, 我们希望构建一个 $Ln$ 维向量.**

注意到:
$$\begin{aligned}
(\lambda^*_{l,i}\pi_{l,i}^*) = \text{diag}(\text{vec}(\Lambda^*)) \text{vec}(\Pi^*)  \in \mathbb{R}^{Ln},\\
(\lambda^*_{l,i} v_{l,i}) = \text{diag}(\text{vec}(\Lambda^*)) \text{vec}(V)  \in \mathbb{R}^{Ln}
\end{aligned}$$
且有:
$$\begin{aligned}
\mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta} &= 
\begin{bmatrix}
u_{11} x^{(1)}_1 & u_{11}x_2^{(1)} & \cdots & u_{11} x^{(1)}_d \\
\vdots & \vdots & \cdots & \vdots \\
u_{1n} x^{(n)}_1 & u_{1n}x_2^{(n)} & \cdots & u_{1n} x^{(n)}_d \\
u_{21} x^{(1)}_1 & u_{21}x_2^{(1)} & \cdots & u_{21} x^{(1)}_d \\
\vdots & \vdots & \cdots & \vdots \\
u_{Ln} x^{(n)}_1 & u_{Ln}x_2^{(n)} & \cdots & u_{Ln} x^{(n)}_d
\end{bmatrix} 
\begin{bmatrix}
\beta_1 \\ \beta_2 \\ \vdots \\ \beta_d
\end{bmatrix} = \begin{bmatrix}
u_{11} x^{(1)}_1 \beta_1 +  \cdots + u_{11} x^{(1)}_d \beta_d \\
u_{12} x^{(2)}_1 \beta_1 + \cdots + u_{12} x^{(2)}_d \beta_d \\
\vdots \\
u_{Ln} x^{(n)}_1 \beta_1 + \cdots + u_{Ln} x^{(n)}_d \beta_d
\end{bmatrix} \\&= \begin{bmatrix}
u_{11} \sum_{j=1}^d x^{(1)}_j \beta_j \\
u_{12} \sum_{j=1}^d x^{(2)}_j \beta_j \\
\vdots \\
u_{Ln} \sum_{j=1}^d x^{(n)}_j \beta_j
\end{bmatrix} =\begin{bmatrix}\vdots \\ u_{li} \mathbf{x}_i^\top \boldsymbol{\beta} \\ \vdots \end{bmatrix} \in \mathbb{R}^{Ln}
\end{aligned}$$


因此有:
$$\begin{aligned}
(2) \Leftrightarrow \text{diag}(\text{vec}(\Lambda^*)) \left(
    \text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)\right) = \mathbf{0}_{Ln} \\
\end{aligned}$$

**对于 $(3)$, 我们希望构建一个 $Ln$ 维向量.**

同理易得:
$$\begin{aligned}
(3) \Leftrightarrow \left(I_{Ln} - \text{diag}(\text{vec}(\Lambda^*))\right) \text{vec}(\Pi^*) = \mathbf{0}_{Ln} \\
\end{aligned}$$

综上所述, 我们得到:
$$\begin{aligned}
F := \begin{bmatrix} F_1 \\ F_2 \\ F_3 \end{bmatrix} :=
\begin{bmatrix}
\boldsymbol{\beta}^* + \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}^*) \\
\text{diag}(\text{vec}(\Lambda^*)) \left(
    \text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)\right) \\
\left(I_{Ln} - \text{diag}(\text{vec}(\Lambda^*))\right) \text{vec}(\Pi^*)
\end{bmatrix} = \begin{bmatrix} \boldsymbol{0}_{d \times 1} \\ \mathbf{0}_{Ln} \\ \mathbf{0}_{Ln} \end{bmatrix} \in \mathbb{R}^{(d+Ln+Ln) \times 1} 
\end{aligned}$$

进一步, 
$$
\omega^* := \begin{bmatrix} \text{vec}(\Pi^*) \\ \boldsymbol{\beta}^* \\ \text{vec}(\Lambda^*) \end{bmatrix} \in \mathbb{R}^{(Ln+d+Ln) \times 1}, \quad \theta = \begin{bmatrix} \text{vec}(U) \\ \text{vec}(V) \end{bmatrix} \in \mathbb{R}^{2Ln \times 1}
$$
因此 $F$ 可以看作是 $\omega^*, \theta$ 之函数, 记为 $F(\omega^*,\theta)$.

---
### 4. 隐函数 $F$ 关于 $\omega$ 之梯度

下求解 $F$ 关于 $\omega^*$ 之导数. 

$$\begin{aligned} \nabla_\omega F  =\begin{bmatrix} \frac{\partial F_1}{\partial \text{vec}(\Pi^*)} & \frac{\partial F_1}{\partial \boldsymbol{\beta}^*} & \frac{\partial F_1}{\partial \text{vec}(\Lambda^*)} \\ \frac{\partial F_2}{\partial \text{vec}(\Pi^*)} & \frac{\partial F_2}{\partial \boldsymbol{\beta}^*} & \frac{\partial F_2}{\partial \text{vec}(\Lambda^*)} \\ \frac{\partial F_3}{\partial \text{vec}(\Pi^*)} & \frac{\partial F_3}{\partial \boldsymbol{\beta}^*} & \frac{\partial F_3}{\partial \text{vec}(\Lambda^*)}\end{bmatrix}\end{aligned}$$

关于各项依次求解. 

**对于 $\partial F_1 / \partial \text{vec}(\Pi^*)$**

$$\frac{\partial F_1}{\partial \text{vec}(\Pi^*)} = \boldsymbol{0}_{d \times Ln}$$

**对于 $\partial F_1 / \partial \boldsymbol{\beta}^*$**
$$\begin{aligned}
\frac{\partial F_1}{\partial \boldsymbol{\beta}^*} &= \frac{\partial}{\partial \boldsymbol{\beta}^*} \left(\boldsymbol{\beta}^* + \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}^*)\right) = \boldsymbol{I}_{d \times d}
\end{aligned}$$

**对于 $\partial F_1 / \partial \text{vec}(\Lambda^*)$**
$$\begin{aligned}
\frac{\partial F_1}{\partial \text{vec}(\Lambda^*)} &= \frac{\partial}{\partial \text{vec}(\Lambda^*)} \left(\boldsymbol{\beta}^* + \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}^*)\right) = \mathsf{\bar{U}}_{(3)} \\ 
&= \begin{bmatrix}
u_{11} x^{(1)}_1 & u_{12} x^{(2)}_1 & \cdots & u_{1n} x^{(n)}_1 & u_{21} x^{(1)}_1 & \cdots & u_{Ln} x^{(n)}_1 \\
u_{11} x^{(1)}_2 & u_{12} x^{(2)}_2 & \cdots & u_{1n} x^{(n)}_2 & u_{21} x^{(1)}_2 & \cdots & u_{Ln} x^{(n)}_2 \\
\vdots & \vdots & \cdots & \vdots & \vdots & \cdots & \vdots \\
u_{11} x^{(1)}_d & u_{12} x^{(2)}_d & \cdots & u_{1n} x^{(n)}_d & u_{21} x^{(1)}_d & \cdots & u_{Ln} x^{(n)}_d
\end{bmatrix}\in \mathbb{R}^{d \times Ln} 
\end{aligned}$$

**对于 $\partial F_2 / \partial \text{vec}(\Pi^*)$**
$$\begin{aligned}
\frac{\partial F_2}{\partial \text{vec}(\Pi^*)} &= \frac{\partial}{\partial \text{vec}(\Pi^*)} \left(\text{diag}(\text{vec}(\Lambda^*)) \left(
    \text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)\right)\right) \\
&= \text{diag}(\text{vec}(\Lambda^*))= \begin{bmatrix}
 \lambda^*_{11} & 0 & \cdots & 0 \\
 0 & \lambda^*_{12} & \cdots & 0 \\
 \vdots & \vdots & \ddots & \vdots \\
 0 & 0 & \cdots & \lambda^*_{Ln}
\end{bmatrix}\in \mathbb{R}^{Ln \times Ln}
\end{aligned}$$

**对于 $\partial F_2 / \partial \boldsymbol{\beta}^*$**
$$\begin{aligned}
\frac{\partial F_2}{\partial \boldsymbol{\beta}^*} &= \frac{\partial}{\partial \boldsymbol{\beta}^*} \left(\text{diag}(\text{vec}(\Lambda^*)) \left(
    \text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)\right)\right) \\
&= -\text{diag}(\text{vec}(\Lambda^*)) \mathsf{\bar{U}}_{(3)}^\top \\&=-\begin{bmatrix}
 \lambda^*_{11} & 0 & \cdots & 0 \\
 0 & \lambda^*_{12} & \cdots & 0 \\
 \vdots & \vdots & \ddots & \vdots \\
 0 & 0 & \cdots & \lambda^*_{Ln}
\end{bmatrix} \begin{bmatrix}
u_{11} x^{(1)}_1 & u_{11}x_2^{(1)} & \cdots & u_{11} x^{(1)}_d \\
\vdots & \vdots & \cdots & \vdots \\
u_{1n} x^{(n)}_1 & u_{1n}x_2^{(n)} & \cdots & u_{1n} x^{(n)}_d \\
u_{21} x^{(1)}_1 & u_{21}x_2^{(1)} & \cdots & u_{21} x^{(1)}_d \\
\vdots & \vdots & \cdots & \vdots \\
u_{Ln} x^{(n)}_1 & u_{Ln}x_2^{(n)} & \cdots & u_{Ln} x^{(n)}_d
\end{bmatrix} \\&=
\begin{bmatrix}
-\lambda^*_{11} u_{11} x^{(1)}_1 & -\lambda^*_{11} u_{11}x_2^{(1)} & \cdots & -\lambda^*_{11} u_{11} x^{(1)}_d \\
\vdots & \vdots & \cdots & \vdots \\
-\lambda^*_{1n} u_{1n} x^{(n)}_1 & -\lambda^*_{1n} u_{1n}x_2^{(n)} & \cdots & -\lambda^*_{1n} u_{1n} x^{(n)}_d \\
\vdots & \vdots & \cdots & \vdots \\
-\lambda^*_{Ln} u_{Ln} x^{(n)}_1 & -\lambda^*_{Ln} u_{Ln}x_2^{(n)} & \cdots & -\lambda^*_{Ln} u_{Ln} x^{(n)}_d
\end{bmatrix} \in \mathbb{R}^{Ln \times d}
\end{aligned}$$

**对于 $\partial F_2 / \partial \text{vec}(\Lambda^*)$**
$$\begin{aligned}
\frac{\partial F_2}{\partial \text{vec}(\Lambda^*)} &= \frac{\partial}{\partial \text{vec}(\Lambda^*)} \left(\text{diag}(\text{vec}(\Lambda^*)) \left(
    \text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)\right)\right) = \text{diag}\left(\text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)\right) \\&=\text{diag}\left(\begin{bmatrix} \pi_{1,1}^* - u_{1,1} \mathbf{x}_1^\top \boldsymbol{\beta}^* - v_{1,1} \\ \vdots \\ \pi_{L,n}^* - u_{L,n} \mathbf{x}_n^\top \boldsymbol{\beta}^* - v_{L,n} \end{bmatrix}\right) \\ &= \begin{bmatrix}
\pi_{1,1}^* - u_{1,1} \mathbf{x}_1^\top \boldsymbol{\beta}^* - v_{1,1} & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & \pi_{L,n}^* - u_{L,n} \mathbf{x}_n^\top \boldsymbol{\beta}^* - v_{L,n}
\end{bmatrix}
\in \mathbb{R}^{Ln \times Ln}
\end{aligned}$$



**对于 $\partial F_3 / \partial \text{vec}(\Pi^*)$**
$$\begin{aligned}
\frac{\partial F_3}{\partial \text{vec}(\Pi^*)} &= \frac{\partial}{\partial \text{vec}(\Pi^*)} \left(\left(I_{Ln} - \text{diag}(\text{vec}(\Lambda^*))\right) \text{vec}(\Pi^*)\right) \\
&= I_{Ln} - \text{diag}(\text{vec}(\Lambda^*)) =
\begin{bmatrix}
1 - \lambda^*_{11} & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & 1 - \lambda^*_{Ln}
\end{bmatrix}
\in \mathbb{R}^{Ln \times Ln}
\end{aligned}$$

**对于 $\partial F_3 / \partial \boldsymbol{\beta}^*$**
$$\begin{aligned}
\frac{\partial F_3}{\partial \boldsymbol{\beta}^*} &= \frac{\partial}{\partial \boldsymbol{\beta}^*} \left(\left(I_{Ln} - \text{diag}(\text{vec}(\Lambda^*))\right) \text{vec}(\Pi^*)\right) = \boldsymbol{0}_{Ln \times d}
\end{aligned}$$

**对于 $\partial F_3 / \partial \text{vec}(\Lambda^*)$**
$$\begin{aligned}
\frac{\partial F_3}{\partial \text{vec}(\Lambda^*)} &= \frac{\partial}{\partial \text{vec}(\Lambda^*)} \left(\left(I_{Ln} - \text{diag}(\text{vec}(\Lambda^*))\right) \text{vec}(\Pi^*)\right) \\
&= -\text{diag}(\text{vec}(\Pi^*)) = \begin{bmatrix}
-\pi_{1,1}^* & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & -\pi_{L,n}^*
\end{bmatrix} \in \mathbb{R}^{Ln \times Ln}
\end{aligned}$$

因此, 有:
$$\begin{aligned}
\nabla_\omega F &= \begin{bmatrix} \boldsymbol{0}_{d \times Ln} & \boldsymbol{I}_{d \times d} & \mathsf{\bar{U}}_{(3)} \\ \text{diag}(\text{vec}(\Lambda^*)) & -\text{diag}(\text{vec}(\Lambda^*)) \mathsf{\bar{U}}_{(3)}^\top & \text{diag}(\text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)) \\  I_{Ln} - \text{diag}(\text{vec}(\Lambda^*)) & \boldsymbol{0}_{Ln \times d} & -\text{diag}(\text{vec}(\Pi^*))\end{bmatrix} 
\end{aligned}$$

其维度示意图为:
$$\begin{aligned}
\begin{bmatrix} \mathbb{R}^{d \times Ln} & \mathbb{R}^{d \times d} & \mathbb{R}^{d \times Ln} \\ \mathbb{R}^{Ln \times Ln} & \mathbb{R}^{Ln \times d} & \mathbb{R}^{Ln \times Ln} \\  \mathbb{R}^{Ln \times Ln} & \mathbb{R}^{Ln \times d} & \mathbb{R}^{Ln \times Ln}\end{bmatrix}\end{aligned}$$

进一步, 简记上矩阵为:
$$\nabla_\omega F = \begin{bmatrix} \boldsymbol{0} & \boldsymbol{I} & \mathsf{\bar{U}}_{(3)} \\ D_\Lambda & -D_\Lambda \mathsf{\bar{U}}_{(3)}^\top & D_{2} \\ I - D_\Lambda & \boldsymbol{0} & -D_\Pi \end{bmatrix}$$

----

### 5. 隐函数 $F$ 关于 $\theta$ 之梯度

此外, 我们还需要求解 $\nabla_\theta F$, 其中 $\theta = \begin{bmatrix} \text{vec}(U) \\ \text{vec}(V) \end{bmatrix} \in \mathbb{R}^{2Ln \times 1}$.

$$\begin{aligned}
F := \begin{bmatrix} F_1 \\ F_2 \\ F_3 \end{bmatrix} :=
\begin{bmatrix}
\boldsymbol{\beta}^* + \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}^*) \\
\text{diag}(\text{vec}(\Lambda^*)) \left(
    \text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)\right) \\
\left(I_{Ln} - \text{diag}(\text{vec}(\Lambda^*))\right) \text{vec}(\Pi^*)
\end{bmatrix} = \begin{bmatrix} \boldsymbol{0}_{d \times 1} \\ \mathbf{0}_{Ln} \\ \mathbf{0}_{Ln} \end{bmatrix} \in \mathbb{R}^{(d+Ln+Ln) \times 1} 
\end{aligned}$$

整体上, 其形式结构应当为:
$$\begin{aligned}
\nabla_\theta F = \begin{bmatrix} \frac{\partial F_1}{\partial \text{vec}(U)} & \frac{\partial F_1}{\partial \text{vec}(V)} \\ \frac{\partial F_2}{\partial \text{vec}(U)} & \frac{\partial F_2}{\partial \text{vec}(V)} \\ \frac{\partial F_3}{\partial \text{vec}(U)} & \frac{\partial F_3}{\partial \text{vec}(V)}\end{bmatrix} \in \mathbb{R}^{(d+Ln+Ln) \times 2Ln}
\end{aligned}$$

下面依次求解各项.

**对于 $\partial F_1 / \partial \text{vec}(U)$**
$$\begin{aligned}
\frac{\partial F_1}{\partial \text{vec}(U)} &= \frac{\partial}{\partial \text{vec}(U)} \left(\boldsymbol{\beta}^* + \mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}^*)\right) = \frac{\partial}{\partial \text{vec}(U)} \left(\mathsf{\bar{U}}_{(3)} \text{vec}(\boldsymbol{\Lambda}^*)\right) \\
&= \frac{\partial}{\partial \text{vec}(U)} \left(\begin{bmatrix}
u_{11} x^{(1)}_1 \lambda_{1,1}^* + \cdots  + u_{Ln} x^{(n)}_1 \lambda_{L,n}^* \\
u_{11} x^{(1)}_2 \lambda_{1,1}^* + \cdots  + u_{Ln} x^{(n)}_2 \lambda_{L,n}^* \\
\vdots  \\
u_{11} x^{(1)}_d \lambda_{1,1}^* + \cdots + u_{Ln} x^{(n)}_d \lambda_{L,n}^*
\end{bmatrix}\right) \\
&= \begin{bmatrix}
x^{(1)}_1 \lambda_{1,1}^* & x^{(2)}_1 \lambda_{1,2}^* & \cdots & x^{(n)}_1 \lambda_{1,n}^* & x^{(1)}_1 \lambda_{2,1}^* & \cdots & x^{(n)}_1 \lambda_{L,n}^* \\
x^{(1)}_2 \lambda_{1,1}^* & x^{(2)}_2 \lambda_{1,2}^* & \cdots & x^{(n)}_2 \lambda_{1,n}^* & x^{(1)}_2 \lambda_{2,1}^* & \cdots & x^{(n)}_2 \lambda_{L,n}^* \\
\vdots & \vdots & \cdots & \vdots & \vdots & \cdots & \vdots \\
x^{(1)}_d \lambda_{1,1}^* & x^{(2)}_d \lambda_{1,2}^* & \cdots & x^{(n)}_d \lambda_{1,n}^* & x^{(1)}_d \lambda_{2,1}^* & \cdots & x^{(n)}_d \lambda_{L,n}^*
\end{bmatrix} \\
&:= \mathcal{X} D_\Lambda \in \mathbb{R}^{d \times nL}
\end{aligned}$$
其中， $\mathcal{X} =  \underbrace{[X, \cdots, X]}_{L \text{ times}} \in \mathbb{R}^{d \times nL}$, $D_\Lambda = \text{diag}(\text{vec}(\Lambda^*)) \in \mathbb{R}^{Ln \times Ln}$.


**对于 $\partial F_2 / \partial \text{vec}(U)$**
$$\begin{aligned}
\frac{\partial F_2}{\partial \text{vec}(U)} &= \frac{\partial}{\partial \text{vec}(U)} \left(\text{diag}(\text{vec}(\Lambda^*)) \left(
    \text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)\right)\right) \\
&=  -\text{diag}(\text{vec}(\Lambda^*)) \frac{\partial}{\partial \text{vec}(U)} \left(\mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^*\right) \\
&= -\text{diag}(\text{vec}(\Lambda^*))  \frac{\partial}{\partial \text{vec}(U)}\begin{bmatrix}
u_{11} \sum_{j=1}^d x^{(1)}_j \beta_j \\
u_{12} \sum_{j=1}^d x^{(2)}_j \beta_j \\
\vdots \\
u_{Ln} \sum_{j=1}^d x^{(n)}_j \beta_j
\end{bmatrix} \\&= -\text{diag}(\text{vec}(\Lambda^*))  
\begin{bmatrix} 
\mathbf{x}_1^\top \boldsymbol{\beta}^* & 0 & \cdots & 0 \\
0 & \mathbf{x}_2^\top \boldsymbol{\beta}^* & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \mathbf{x}_n^\top \boldsymbol{\beta}^*
\end{bmatrix} \\
&=-D_\Lambda \text{diag}(\mathcal X^\top \boldsymbol{\beta}^*) = -D_\Lambda D_{\mathcal X^\top \beta} \in \mathbb{R}^{Ln \times Ln}
\end{aligned}$$


**对于 $\partial F_2 / \partial \text{vec}(V)$**
$$\begin{aligned}
\frac{\partial F_2}{\partial \text{vec}(V)} &= \frac{\partial}{\partial \text{vec}(V)} \left(\text{diag}(\text{vec}(\Lambda^*)) \left(
    \text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)\right)\right) \\
&= -\text{diag}(\text{vec}(\Lambda^*)) \frac{\partial}{\partial \text{vec}(V)} \left( - \text{vec}(V)\right) \\
&= - \text{diag}(\text{vec}(\Lambda^*)) = - D_\Lambda \in \mathbb{R}^{Ln \times Ln}
\end{aligned}$$

其他分块均为 $\boldsymbol{0}$ , 因此:
$$\begin{aligned}
\nabla_\theta F &= 
\begin{bmatrix} 
(\mathcal{X} {D_\Lambda})_{d\times Ln} & \boldsymbol{0}_{d \times Ln} \\ 
(-D_\Lambda D_{\mathcal{X}^\top \boldsymbol{\beta}})_{Ln\times Ln} & (-D_\Lambda)_{Ln\times Ln} \\ 
\boldsymbol{0}_{Ln \times Ln} & \boldsymbol{0}_{Ln \times Ln} \end{bmatrix} \in \mathbb{R}^{(d+Ln+Ln) \times 2Ln}
\end{aligned}$$

综上，我们的隐函数为：

$$\begin{aligned}
\begin{bmatrix} \boldsymbol{0}_{d \times Ln} & \boldsymbol{I}_{d \times d} & \mathsf{\bar{U}}_{(3)} \\ \text{diag}(\text{vec}(\Lambda^*)) & -\text{diag}(\text{vec}(\Lambda^*)) \mathsf{\bar{U}}_{(3)}^\top & \text{diag}(\text{vec}(\Pi^*) - \mathsf{\bar{U}}_{(3)}^\top \boldsymbol{\beta}^* - \text{vec}(V)) \\  I_{Ln} - \text{diag}(\text{vec}(\Lambda^*)) & \boldsymbol{0}_{Ln \times d} & -\text{diag}(\text{vec}(\Pi^*))\end{bmatrix}^{-1} 
\begin{bmatrix} 
(\mathcal{X} {D_\Lambda})_{d\times Ln} & \boldsymbol{0}_{d \times Ln} \\ 
(-D_\Lambda \mathcal{X}^\top \boldsymbol{\beta}^*)_{Ln\times Ln} & (-D_\Lambda)_{Ln\times Ln} \\ 
\boldsymbol{0}_{Ln \times Ln} & \boldsymbol{0}_{Ln \times Ln} \end{bmatrix} 
\end{aligned}$$
 
去掉冗余部分后，简记为：
$$\begin{aligned}
\begin{bmatrix} \boldsymbol{0} & \boldsymbol{I} & \mathsf{\bar{U}}_{(3)} \\ D_\Lambda & -D_\Lambda \mathsf{\bar{U}}_{(3)}^\top & D_{2} \\ I - D_\Lambda & \boldsymbol{0} & -D_\Pi \end{bmatrix}^{-1}
\begin{bmatrix} 
\mathcal{X} D_\Lambda & \boldsymbol{0} \\ 
-D_\Lambda D_{\mathcal{X}^\top \boldsymbol{\beta}^*} & -D_\Lambda \\
\boldsymbol{0} & \boldsymbol{0} \end{bmatrix}
\end{aligned}$$

## 总结概括

### 1 · 符号与维度约定

| 记号                        | 含义                                               | 维度                        |
| ------------------------- | ------------------------------------------------ | ------------------------- |
| $d$                       | 特征维度                                             | —                         |
| $n$                       | 样本数                                              | —                         |
| $L$                       | ReLU 数                                           | —                         |
| $\mathbf x_i$             | 第 $i$ 个样本特征向量                                    | $\mathbb R^{d}$           |
| $U=(u_{l,i})$             | 斜率参数矩阵                                           | $\mathbb R^{L\times n}$   |
| $V=(v_{l,i})$             | 偏置参数矩阵                                           | $\mathbb R^{L\times n}$   |
| $\Pi=(\pi_{l,i})$         | ReLU 松弛变量                                        | $\mathbb R^{L\times n}$   |
| $\Lambda=(\lambda_{l,i})$ | 主动约束乘子                                           | $\mathbb R^{L\times n}$   |
| $\Delta=(\delta_{l,i})$   | 非负约束乘子                                           | $\mathbb R^{L\times n}$   |
| $\bar{\mathsf U}$         | 三阶张量，元素 $u_{l,i}x_{ij}$                          | $L\times n\times d$       |
| $\bar{\mathsf U}_{(3)}$   | $\bar{\mathsf U}$ 的 mode-3 展开                    | $\mathbb R^{d\times nL}$  |
| $D_\Lambda$               | $\operatorname{diag}(\operatorname{vec}\Lambda)$ | $\mathbb R^{Ln\times Ln}$ |
| $D_\Pi$                   | $\operatorname{diag}(\operatorname{vec}\Pi)$     | 同上                        |
| $\mathcal X$              | $[X,\dots,X]$（$L$ 次水平拼接）                         | $\mathbb R^{d\times nL}$  |

---

### 2 · 原始问题

$$
\min_{\boldsymbol\beta\in\mathbb R^{d}}\;
     \sum_{i=1}^{n}\sum_{l=1}^{L}\operatorname{ReLU}\!\bigl(u_{l,i}\mathbf x_i^{\top}\boldsymbol\beta+v_{l,i}\bigr)
     +\tfrac12\|\boldsymbol\beta\|^{2}.
$$

经 **ReLU 线性化** 得等价带松弛的 Primal：

$$
\begin{aligned}
\min_{\boldsymbol\beta,\Pi}\;&
   \sum_{l,i}\pi_{l,i}+\tfrac12\|\boldsymbol\beta\|^{2}\\
\text{s.t. }&\;
u_{l,i}\mathbf x_i^{\top}\boldsymbol\beta+v_{l,i}-\pi_{l,i}\ge0,\;
\pi_{l,i}\ge0.
\end{aligned}
$$

---

### 3 · Lagrangian 与 Stationary 条件

$$
\mathcal L_{\mathcal P}=
\sum_{l,i}\pi_{l,i}+\tfrac12\|\boldsymbol\beta\|^{2}
-\sum_{l,i}\lambda_{l,i}(\pi_{l,i}-u_{l,i}\mathbf x_i^{\top}\boldsymbol\beta-v_{l,i})
-\sum_{l,i}\delta_{l,i}\pi_{l,i}.
$$

* **对 $\Pi$**：$1-\lambda_{l,i}-\delta_{l,i}=0 \;\Rightarrow\; 0\le\lambda_{l,i}\le1$.
* **对 $\boldsymbol\beta$**：

  $$
  \boldsymbol\beta
  +\bar{\mathsf U}_{(3)}\operatorname{vec}(\Lambda)=\mathbf0
  \;\;\Longrightarrow\;\;
  \boldsymbol\beta^{\star}=-\bar{\mathsf U}_{(3)}\operatorname{vec}(\Lambda).
  $$

---

### 4 · 对偶函数

$$
g(\Lambda)=
-\tfrac12\operatorname{vec}(\Lambda)^{\top}
      \bar{\mathsf U}_{(3)}^{\top}\bar{\mathsf U}_{(3)}
      \operatorname{vec}(\Lambda)
+\operatorname{Tr}\!\bigl(\Lambda V^{\top}\bigr),
\quad
0\le\Lambda\le1.
$$

（对偶问题取 $\max g(\Lambda)$。）

---

### 5 · KKT 条件（向量化形式）

设

$$
\omega=\begin{bmatrix}\operatorname{vec}\Pi\\ \boldsymbol\beta\\ \operatorname{vec}\Lambda\end{bmatrix},
\qquad
\theta=\begin{bmatrix}\operatorname{vec}U\\ \operatorname{vec}V\end{bmatrix},
$$

则定义

$$
F(\omega,\theta)=
\begin{bmatrix}
\boldsymbol\beta+\bar{\mathsf U}_{(3)}\operatorname{vec}\Lambda\\[2pt]
D_\Lambda\!\bigl(\operatorname{vec}\Pi-\bar{\mathsf U}_{(3)}^{\top}\boldsymbol\beta-\operatorname{vec}V\bigr)\\[2pt]
(I-D_\Lambda)\operatorname{vec}\Pi
\end{bmatrix}
=\mathbf0\in\mathbb R^{d+2Ln}.
$$

---

### 6 · 雅可比矩阵

$$
\nabla_\omega F=
\begin{bmatrix}
\mathbf0 & I_d & \bar{\mathsf U}_{(3)}\\
D_\Lambda & -D_\Lambda\bar{\mathsf U}_{(3)}^{\top} &
            \operatorname{diag}\!\bigl(\operatorname{vec}\Pi-\bar{\mathsf U}_{(3)}^{\top}\boldsymbol\beta-\operatorname{vec}V\bigr)\\
I-D_\Lambda & \mathbf0 & -D_\Pi
\end{bmatrix}.
$$

$$
\nabla_\theta F=
\begin{bmatrix}
\mathcal X D_\Lambda & \mathbf0\\
-\,D_\Lambda\,\operatorname{diag}(\mathcal X^{\top}\boldsymbol\beta) & -D_\Lambda\\
\mathbf0 & \mathbf0
\end{bmatrix}.
$$

（所有块尺寸分别为 $d$ 或 $Ln$ 见表 1。）

---

### 7 · 隐函数求导框架

若 $\nabla_\omega F$ 在最优点可逆，则

$$
\frac{\partial\omega}{\partial\theta}
= -\bigl(\nabla_\omega F\bigr)^{-1}\nabla_\theta F.
$$
