---
aliases: [再生核希尔伯特空间, RKHS, Reproducing Kernel Hilbert Space, Representer Theorem]
tags:
  - concept
  - ml/stats
  - math/functional-analysis
related_concepts:
  - "[[Kernel Methods]]"
  - "[[Hilbert Space]]"
---

# RKHS and Representer Theorem

## Reproducting Kernel Hilbert Space (RKHS)

### Function Space and Hilbert Space

***Definition** (Function Space)*. 记 $\mathbb{R}^{\mathbb{R}^d}$ 为所有从 $\mathbb{R}^d$ 到 $\mathbb{R}$ 的函数的集合, 即 $\mathbb{R}^{\mathbb{R}^d} = \{f | f: \mathbb{R}^d \to \mathbb{R}\}$. 该集合称为**函数空间 (Function Space)**.

***Definition** (Reproducing Kernel Map)*. 给定核函数 $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$, 定义映射 $\Phi: \mathbb{R}^d \to \mathbb{R}^{\mathbb{R}^d}$, 使得对于任意 $x\in\mathbb{R}^d$, 有:
$$\Phi(z) = \kappa(\cdot, z),$$
即
$$[\Phi(z)](x) = \kappa(x, z), \quad \forall x \in \mathbb{R}^d.$$
  - Reproducing Kernel Map 将每个 $d$ 维输入点 $z$ 映射为一个 kernel 函数 $\kappa(\cdot, z)$, 该函数以任意 $d$ 维输入点 $x$ 为自变量, 输出为 $\kappa(x, z)\in\mathbb{R}$.
    - 一个理解当前映射的方式是类比分布函数 $\mathcal{N}(\mu,1)$ 中的均值参数 $\mu$, 它将每个实数 $\mu$ 映射为一个函数 $f_\mu(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2}\right)$, 该函数以任意实数 $x$ 为自变量, 输出为概率密度值 $f_\mu(x)\in\mathbb{R}$ (当然额外还需要满足 kernel 的半正定性质).

***Definition** (Vector Space over $\mathbb{R}$)*. 设 $V$ 为一个 $\mathbb{R}$ 上 的非空集合, 若在 $V$ 上定义了加法运算 $+: V \times V \to V$ 和数乘运算 $\cdot: \mathbb{R} \times V \to V$, 且满足以下公理, 则称 $V$ 为**实向量空间 (Vector Space over $\mathbb{R}$)**:
  1. 加法交换律: 对于任意 $u,v\in V$, 有 $u+v = v+u$.
  2. 加法结合律: 对于任意 $u,v,w\in V$, 有 $(u+v)+w = u+(v+w)$.
  3. 存在加法单位元: 存在元素 $0\in V$, 使得对于任意 $v\in V$, 有 $v+0 = v$.
  4. 存在加法逆元: 对于任意 $v\in V$, 存在元素 $-v\in V$, 使得 $v + (-v) = 0$.
  5. 数乘结合律: 对于任意 $a,b\in\mathbb{R}$ 和 $v\in V$, 有 $a(bv) = (ab)v$.
  6. 数乘分配律: 对于任意 $a,b\in\mathbb{R}$ 和 $u,v\in V$, 有 $(a+b)v = av + bv$ 和 $a(u+v) = au + av$.
  7. 数乘分配律2: 对于任意 $a\in\mathbb{R}$ 和 $u,v\in V$, 有 $a(u+v) = au + av$.
  8. 数乘单位元: 对于任意 $v\in V$, 有 $1 \cdot v = v$.


***Definition** (Linear Span of Kernel Sections)* 对于任意给定的核函数 $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$, 定义由任意有限个 kernel sections 线性组合所构成的函数集合:
$$\mathcal{F}_\kappa = \left\{f: \mathbb{R}^d \to \mathbb{R} \Bigg| f(\cdot) = \sum_{i=1}^n \alpha_i \kappa(\cdot, x_i), n\in\mathbb{N}, \alpha_i\in\mathbb{R}, x_i\in\mathbb{R}^d\right\}.$$
  - 该集合称为由核函数 $\kappa$ 所生成的**线性空间 (Linear Span)**. 且该空间是一个实向量空间.
  - 若记 $\Phi(x) = \kappa(\cdot, x)$ 为核映射, 则 $\mathcal{F}_\kappa$ 可表示为:
$$\mathcal{F}_\kappa = \left\{f: \mathbb{R}^d \to \mathbb{R} \Bigg| f(\cdot) = \sum_{i=1}^n \alpha_i \Phi(x_i), n\in\mathbb{N}, \alpha_i\in\mathbb{R}, x_i\in\mathbb{R}^d\right\}.$$


***Definition** (Inner Product Space)*. 给定一个实向量空间 $V$, 若在 $V$ 上定义了内积运算 $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$, 且满足以下公理, 则称 $V$ 为**内积空间 (Inner Product Space)**:
  1. 对称性: 对于任意 $u,v\in V$, 有 $\langle u, v \rangle = \langle v, u \rangle$.
  2. 线性性: 对于任意 $u,v,w\in V$ 和 $a,b\in\mathbb{R}$, 有 $\langle au + bv, w \rangle = a\langle u, w \rangle + b\langle v, w \rangle$.
  3. 正定性: 对于任意 $v\in V$, 有 $\langle v, v \rangle \geq 0$, 且当且仅当 $v=0$ 时取等号.

***Definition** (Inner Product Induced by Kernel)*. 给定核函数 $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$, 定义在 $\mathcal{F}_\kappa$ 上的内积运算 $\langle \cdot, \cdot \rangle_{\mathcal{F}_\kappa}: \mathcal{F}_\kappa \times \mathcal{F}_\kappa \to \mathbb{R}$ 如下: 对于任意 $f,g\in\mathcal{F}_\kappa$, 即 $f(\cdot) = \sum_{i=1}^m \alpha_i \Phi(x_i)=\sum_{i=1}^m \alpha_i \kappa(\cdot, x_i)$,  和 $g(\cdot) = \sum_{j=1}^n \beta_j \Phi(z_j)=\sum_{j=1}^n \beta_j \kappa(\cdot, z_j)$:, 定义
$$\langle f, g \rangle_{\mathcal{F}_\kappa} = \sum_{i=1}^m \sum_{j=1}^n \alpha_i \beta_j \kappa(x_i, z_j).$$
  - 上述内积也可以用矩阵形式表示为:
    $$\langle f, g \rangle_{\mathcal{F}_\kappa} = \boldsymbol{\alpha}^\top K_{XZ} \boldsymbol{\beta},$$
    其中 $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_m)^\top$, $\boldsymbol{\beta} = (\beta_1, \ldots, \beta_n)^\top$, $K_{XZ}$ 为 $m\times n$ 的核矩阵, 其 $(i,j)$ 元素为 $\kappa(x_i, z_j)$.
  -  同时还有如下性质:
        $$\langle f, g \rangle_{\mathcal{F}_\kappa} = \sum_{i=1}^m \alpha_i g(x_i) = \sum_{j=1}^n \beta_j f(z_j).$$

***Theorem** (Reproducing Property)*. 给定核函数 $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$, 则对于任意 $f\in\mathcal{F}_\kappa$ 和任意 $x\in\mathbb{R}^d$, 有:
$$f(x) = \langle f, \kappa(\cdot, x) \rangle_{\mathcal{F}_\kappa},$$
  - 该性质称为**再生性质 (Reproducing Property)**, 即函数 $f$ 在点 $x$ 处的函数值可以通过 $f$ 与 kernel section $\kappa(\cdot, x)$ 的内积来表示.
  - 且立刻有 $\langle \kappa(\cdot, x), \kappa(\cdot, z) \rangle_{\mathcal{F}_\kappa} = \kappa(x, z)$.

***Definition** (Metric Space)* . 设 $M$ 为一个非空集合, 若在 $M$ 上定义了距离函数 $d: M \times M \to \mathbb{R}$, 且满足以下公理, 则称有序对 $(M, d)$ 为**度量空间 (Metric Space)**:
  1. 非负性: 对于任意 $x,y\in M$, 有 $d(x,y) \geq 0$, 且当且仅当 $x=y$ 时取等号.
  2. 对称性: 对于任意 $x,y\in M$, 有 $d(x,y) = d(y,x)$.
  3. 三角不等式: 对于任意 $x,y,z\in M$, 有 $d(x,z) \leq d(x,y) + d(y,z)$.
  4. 辨识性: 对于任意 $x,y\in M$, $d(x,y) = 0$ 当且仅当 $x=y$.

***Definition** (Cauchy Sequence in Metric Space)*. 给定度量空间 $(M,d)$, 序列 $\{x_n\}$ 称为该空间中的**柯西序列 (Cauchy Sequence)**, 若对于任意 $\epsilon > 0$, 存在正整数 $N$, 使得当 $m,n > N$ 时, 有 $d(x_n, x_m) \leq \epsilon$.
  - Cauchy 序列的定义表明, 随着序列下标的增大, 序列元素之间的距离可以任意小.

***Definition** (Completeness of Metric Space)*. 给定度量空间 $(M,d)$, 若该空间中的任意 Cauchy 序列均收敛于 $M$ 中的某个元素, 则称该空间为**完备的 (Complete)**.

***Definition** (Hilbert Space)*. 给定实内积空间 $\mathcal{H}$, 若该空间在由内积所诱导的距离函数下是完备的, 则称 $\mathcal{H}$ 为**希尔伯特空间 (Hilbert Space)**.
  - 其中由内积 $\langle \cdot, \cdot \rangle_{\mathcal{H}}$ 所诱导的距离函数定义为: 对于任意 $u,v\in\mathcal{H}$, 有
$$d(u,v) = \sqrt{\langle u-v, u-v \rangle_{\mathcal{H}}}.$$

***Definition** (Reproducing Kernel Hilbert Space (RKHS))* . 对于任意给定的核函数 $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$, 若函数空间 $\mathcal{F}_\kappa$ 在由 $\kappa$ 所诱导的内积 $\langle \cdot, \cdot \rangle_{\mathcal{F}_\kappa}$ 下是一个希尔伯特空间, 则称该空间为**再生核希尔伯特空间 (Reproducing Kernel Hilbert Space, RKHS)**.
  - RKHS 是一个完备的内积空间, 其中的函数可以通过核函数的线性组合来表示, 且满足再生性质.