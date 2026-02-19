#TimeSeries 
# Autoregressive Model (II)

## Estimation of $\text{AR}(p)$ Model

对于一组数据 $x_1, x_2, \ldots, x_n$ 假设我们已经判断出该模型应该服从一个 $\text{AR}(p)$ 模型:
$$
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + W_t
$$
其中 $W_t \sim \text{WN}(0, \sigma^2)$, $p<n$ (即模型的滞后阶数 $p$ 小于样本的大小 $n$), 现在我们需要估计模型的参数 $\phi_1, \phi_2, \ldots, \phi_p$ 和 $\sigma^2$. 常见的方法有 Method of Moments Estimation (MME,矩估计), Conditional Least Squares Estimation (CLS, 条件最小二乘估计), Maximum Likelihood Estimation (MLE, 最大似然估计) 等. 

### Method of Moments Estimation (MME)

MME 的核心思想是: 通过通过**样本矩**(基于样本数据求得)来估计**总体矩**(基于总体未知参数求得), 并通过样本矩与总体矩的关系来求解未知参数. 

对于上述 $\text{AR}(p)$ 模型, 我们不妨首先进行中心化. 记 $\mu = \mathbb{E}(X_t)$, 有:
$$
X_t - \mu = \phi_1 (X_{t-1} - \mu) + \phi_2 (X_{t-2} - \mu) + \cdots + \phi_p (X_{t-p} - \mu) + W_t
$$

***$\mu$ 的矩估计***

根据
$$\hat \mu = \bar X = \frac{1}{n} \sum_{t=1}^n X_t$$
即样本均值估计总体期望. 

***$\phi_1, \phi_2, \ldots, \phi_p$ 的矩估计*** 

  - 根据 $\text{AR}(p)$ 模型的定义, 我们曾求过其 ACF 的表达式:
    $$
    \rho_k = \phi_1 \rho_{k-1} + \phi_2 \rho_{k-2} + \cdots + \phi_p \rho_{k-p}, \quad k=1, 2, \ldots
    $$
    将 $k=1, 2, \ldots, p$ 代入上式, 记为矩阵形式:
    $$
    \begin{pmatrix}
    1 & \rho_1 & \rho_2 & \cdots & \rho_{p-1} \\
    \rho_1 & 1 & \rho_1 & \cdots & \rho_{p-2} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \rho_{p-1} & \rho_{p-2} & \cdots & \cdots & 1
    \end{pmatrix}
    \begin{pmatrix}
    \phi_1 \\
    \phi_2 \\
    \vdots \\
    \phi_p
    \end{pmatrix}
    =
    \begin{pmatrix}
    \rho_1 \\
    \rho_2 \\
    \vdots \\
    \rho_p
    \end{pmatrix}
    $$
    即 Yule-Walker 方程. 
  - 反过来, 当我们已知一组数据 $X_1, X_2, \ldots, X_n$ 时, 我们就可以通过样本自相关函数 (sample ACF) $r_k$ 来估计总体ACF $\rho_k$ 进而求解 $\phi_1, \phi_2, \ldots, \phi_p$. 样本自相关函数的定义为:
    $$
    r_k = \frac{\sum_{t=k+1}^n (X_t - \bar X)(X_{t-k} - \bar X)}{\sum_{t=1}^n (X_t - \bar X)^2}, \quad k=1,2, \ldots, p
    $$
    注意到在求解样本ACF时, 我们没有实现假设其是平稳的. 另一方面其分子的求和也只能从 $k+1$ 开始.
  - 因此我们用 $r_1,\cdots,r_p$ 来代替 $\rho_1, \cdots, \rho_p$, 代入 Yule-Walker 方程, 得到: 
    $$
    \begin{pmatrix}
    1 & r_1 & r_2 & \cdots & r_{p-1} \\
    r_1 & 1 & r_1 & \cdots & r_{p-2} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    r_{p-1} & r_{p-2} & \cdots & \cdots & 1
    \end{pmatrix}
    \begin{pmatrix}
    \phi_1 \\
    \phi_2 \\
    \vdots \\
    \phi_p
    \end{pmatrix}
    =
    \begin{pmatrix}
    r_1 \\
    r_2 \\
    \vdots \\
    r_p
    \end{pmatrix}
    $$
    该方程组的解即为 $\phi_1, \phi_2, \ldots, \phi_p$ 的 MME:
    $$
    \begin{pmatrix}
    \hat \phi_1 \\
    \hat \phi_2 \\
    \vdots \\
    \hat \phi_p
    \end{pmatrix}
    =
    \begin{pmatrix}
    1 & r_1 & r_2 & \cdots & r_{p-1} \\
    r_1 & 1 & r_1 & \cdots & r_{p-2} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    r_{p-1} & r_{p-2} & \cdots & \cdots & 1
    \end{pmatrix}^{-1}
    \begin{pmatrix}
    r_1 \\
    r_2 \\
    \vdots \\
    r_p
    \end{pmatrix}
    $$

***$\sigma^2$ 的矩估计***

根据 $\text{AR}(p)$ 模型的定义, 我们亦曾推导过如下结论:
$$\begin{aligned}
\gamma_0 &= \phi_1 \gamma_1 + \phi_2 \gamma_2 + \cdots + \phi_p \gamma_p + \mathbb{E}(W_t X_t) \\
&= \phi_1 \gamma_1 + \phi_2 \gamma_2 + \cdots + \phi_p \gamma_p + \sigma^2
\end{aligned}$$
即
$$\sigma^2 = \gamma_0 - \phi_1 \gamma_1 - \phi_2 \gamma_2 - \cdots - \phi_p \gamma_p$$
因此我们可以用样本方差函数
$$\begin{aligned}
\hat \gamma_0 &= \frac{1}{n-1} \sum_{t=1}^n (X_t - \bar X)^2 \\
\hat \gamma_k &= r_k \hat \gamma_0, \quad k=1, 2, \ldots, p
\end{aligned}$$
来估计总体的 $\gamma_0, \gamma_1, \ldots, \gamma_p$, 以此来估计 $\sigma^2$:
$$\begin{aligned}
\hat \sigma^2 &= \hat \gamma_0 - \hat \phi_1 \hat \gamma_1 - \hat \phi_2 \hat \gamma_2 - \cdots - \hat \phi_p \hat \gamma_p \\
&= \hat \gamma_0 - \hat \phi_1 r_1 \hat \gamma_0 - \hat \phi_2 r_2 \hat \gamma_0 - \cdots - \hat \phi_p r_p \hat \gamma_0 \\
&= \hat \gamma_0 (1 - \hat \phi_1 r_1 - \hat \phi_2 r_2 - \cdots - \hat \phi_p r_p)
\end{aligned}$$
其中 $r_k$ 是样本自相关函数的估计值, $\hat \phi_k$ 是前面通过 Yule-Walker 方程组求得的 $\phi_k$ 的估计值.

### Conditional Least Squares Estimation (CLS)