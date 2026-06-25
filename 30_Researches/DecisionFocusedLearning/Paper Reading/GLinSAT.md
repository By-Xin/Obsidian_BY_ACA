# GLinSAT [施工中]

> https://arxiv.org/abs/2409.17500

## Introduction

GLinSAT 是一个可以将任意神经网络输出 $\mathbf{c} \in \mathbb{R}^{n'}$ 映射到满足线性约束的可行解 $\mathbf{x} \in \mathbb{R}^{n'}$ 的可微映射. 其核心是将原始的线性规划问题 (LP) 转化为一个对偶问题, 并使用 Nesterov 加速梯度下降法求解对偶问题, 从而得到原始问题的最优解.


![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260623200941.png)

## Methodology

### Reformulation of the neural network output projection problem

首先考虑一般的约束问题.

- 对于神经网络输出 $\mathbf{c'} \in \mathbb{R}^{n'}$, 我们希望通过可微的方式将其进行规范化, 使得规范后的结果 $\mathbf{x'} = \text{GLinSAT}(\mathbf{c'}) \in \mathbb{R}^{n'}$ 满足如下可能的约束条件:
    $$
    \begin{aligned}
    \text{(1a) } \qquad & \mathbf{A}_1' \mathbf{x'} &&\leq \mathbf{b}_1' \\
    \text{(1b) } \qquad & \mathbf{A}_2' \mathbf{x'} &&\geq \mathbf{b}_2' \\
    \text{(1c) } \qquad & \mathbf{A}_3' \mathbf{x'} &&= \mathbf{b}_3' \\
    \text{(1d) } \qquad  & \mathbf{x'} &&\in [\mathbf{l'}, \mathbf{u'}]
    \end{aligned}
    $$

- 其中 $\mathbf{A}_1' \in \mathbb{R}^{m_1' \times n'}$, $\mathbf{A}_2' \in \mathbb{R}^{m_2' \times n'}$, $\mathbf{A}_3' \in \mathbb{R}^{m_3' \times n'}$ 分别是线性约束的系数矩阵, $\mathbf{b}_1' \in \mathbb{R}^{m_1'}$, $\mathbf{b}_2' \in \mathbb{R}^{m_2'}$, $\mathbf{b}_3' \in \mathbb{R}^{m_3'}$ 分别是线性约束的右端向量, $\mathbf{l'}$ 和 $\mathbf{u'}$ 分别是变量的下界和上界. 并且假设这个问题是可行的.

通过标准化的处理手段引入 slack variables $\mathbf{s}_1 \in \mathbb{R}^{m_1'}$, $\mathbf{s}_2 \in \mathbb{R}^{m_2'}$, 可以整合为如下标准形式的线性约束 (故后文默认采用标准形式):
$$
\begin{aligned}
\text{(2a) } \qquad & \mathbf{A} \mathbf{x} &&= \mathbf{b} \\
\text{(2b) } \qquad & \mathbf{x} &&\in [\mathbf{0}, \mathbf{u}]
\end{aligned}
$$

- 这里, 作者使用内积 $\langle \mathbf{c}, \mathbf{x}\rangle$ 衡量 GLinSAT  的输出 $\mathbf{x}$ 与输入 $\mathbf{c}$ 的相似性.
- 注意, 文中同时将这个内积叫作 **projection**, 但是其有别于欧式投影 ($\|\mathbf{c} - \mathbf{x}\|^2 = \|\mathbf{c}\|^2 + \|\mathbf{x}\|^2 - 2\langle \mathbf{c}, \mathbf{x}\rangle$). 因此内积的核心是让输出 $\mathbf{x}$ 尽量遵循 $\mathbf{c}$ 给出的各个分量的分数排序偏好, 但不要求输出 $\mathbf{x}$ 与输入 $\mathbf{c}$ 在数值上尽量接近. 

因此理论上, 我们可以直接考虑如下的 LP 问题:
$$
\begin{aligned}
\text{maximize } \qquad & \langle \mathbf{c}, \mathbf{x}\rangle \\
\text{subject to } \qquad & \mathbf{A} \mathbf{x} = \mathbf{b} \\
& \mathbf{x} \in [\mathbf{0}, \mathbf{u}]
\end{aligned} \tag{LP}
$$
- 然而一个比较显式的问题是, 对于 LP 问题, 其最优解 $\mathbf{x}^\star(\mathbf{c})$ 并不一定是唯一的, 并且通常取在线性约束的多面体构成的顶点 $\mathcal{V} = \{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ 上. 因此, LP 事实上是在执行一个 argmax 操作
    $$
    \mathbf{x}^\star(\mathbf{c}) \in \arg\max_{\mathbf{v}_k \in \mathcal{V}} \langle \mathbf{c}, \mathbf{v}_k\rangle
    $$
    即一个分段线性函数. 故这样的映射 $\mathbf{c} \mapsto \mathbf{x}^\star(\mathbf{c})$ 是一个几乎处处梯度为 0, 在分段点处不可微的函数, 这对于神经网络的反向传播是一个很大的障碍. 

因此, 作者在这里提出了 **entropy regulation** 进行光滑化处理, 得到的正则后的目标函数为:
$$
\begin{aligned}
\text{minimize }_{0 \leq \mathbf{x} \leq \mathbf{u}} \qquad & -\langle \mathbf{c}, \mathbf{x}\rangle + \frac{1}{\theta} \sum_{j =1}^n \left[\frac{x_j}{u_j} \log \frac{x_j}{u_j} + \left(1 - \frac{x_j}{u_j}\right) \log \left(1 - \frac{x_j}{u_j}\right)\right] \\
\text{subject to } \qquad & \mathbf{A} \mathbf{x}= \mathbf{b}
\end{aligned} \tag{LP-Ent}
$$

- 对于这个目标函数的理解如下. 
  - 由于 $0 \leq x_j \leq u_j$, 因此得到归一化变量 $p_j = x_j / u_j \in [0, 1]$.
  - 对应的标量函数 (延拓定义 $\phi(0) = 0\log 0 :=0$) (注意到其恰为 Bernoulli 的负 entropy)
    $$
    \phi(p) = p \log p + (1 - p) \log (1 - p), \quad p \in [0, 1]
    $$
    其二阶导 $\phi''(p) = \frac{1}{p(1 - p)} \geq 4$, 因此 $\phi(p)$ 是一个严格凸函数, 且在 $p = 1/2$ 处取得最小值 $\phi(1/2) = -\log 2$.
    其图象如下所示.
    ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/phi_p.png)
    则正则项可以表达为:
    $$
    \frac{1}{\theta} \sum_{j =1}^n \left[\frac{x_j}{u_j} \log \frac{x_j}{u_j} + \left(1 - \frac{x_j}{u_j}\right) \log \left(1 - \frac{x_j}{u_j}\right)\right] = \frac{1}{\theta} \sum_{j=1}^n \phi(p_j)
    $$
    也由于 $\phi$ 的特性, 这个正则项会鼓励每个 $p_j$ 尽量接近 $1/2$, 也就是鼓励每个 $x_j$ 尽量接近 $u_j / 2$, 即将 $x$ 推向内部的平滑解. 

  - 若只考虑一维的情况, 则目标函数为
      $$
      \min_{0\le x\le u} -cx + \frac1\theta \left[ \frac xu\log\frac xu + \left(1-\frac xu\right) \log\left(1-\frac xu\right) \right].
      $$
      可以给出对应的最优 closed form solution:
      $$
      x_\theta^\star(c) = \frac{u}{1 + \exp(-\theta u c)} := u \sigma(\theta u c)
    $$
    其中 $\sigma(\cdot)$ 是 sigmoid 函数. 另外, 观察到 $\theta$ 的作用类似于 temperature (严格说是温度的倒数), 控制了得到的输出的平滑以及均匀程度,具体而言:
    - 当 $\theta \to \infty$, 则 $x_\theta^\star(c) \to u \mathbf{1}_{\{c > 0\}}$, 即恢复到原始的 argmax 的非光滑解. 故 $\theta$ 越大, 则越接近原始的 LP 解, 但是梯度也越不稳定. (注意, 当 LP 非退化时, 该逼近方才成立)
    - 当 $\theta \to 0$, 则 $x_\theta^\star(c) \to u/2$, 由 entropy 直接得到区间中间的正则解. 故 $\theta$ 越小, 则越接近平滑解 (越平均), 但是偏离原始的 LP 解也越远.
    
接着, 我们先暂时只考虑 $\text{LP-Ent}$ 的等式约束, 将不等式 box-constrain 暂时作为定义域的限制. 逐步考虑其对偶问题.
- 引入 Lagrangian Multiplier $\mathbf{y} \in \mathbb{R}^m$, 则对应的 Lagrangian 为:
    $$
    \begin{aligned}
    \mathcal{L}(\mathbf{x}, \mathbf{y}) &= -\langle \mathbf{c}, \mathbf{x}\rangle + \frac{1}{\theta} \sum_{j =1}^n \phi\left(\frac{x_j}{u_j}\right) - \langle \mathbf{y}, \mathbf{A} \mathbf{x} - \mathbf{b} \rangle\\
    &= -\langle \mathbf{c}, \mathbf{x}\rangle + \frac{1}{\theta} \boldsymbol{1}^\top \left[ \mathbf{x} \oslash \mathbf{u} \odot \log (\mathbf{x} \oslash \mathbf{u}) + ( \mathbf{1} - \mathbf{x} \oslash \mathbf{u}) \odot \log ( \mathbf{1} - \mathbf{x} \oslash \mathbf{u}) \right] - \langle \mathbf{y}, \mathbf{A} \mathbf{x} - \mathbf{b} \rangle
    \end{aligned}
    $$
    - 其中 $\oslash$ 表示逐元素除法, $\odot$ 表示逐元素乘法, $\boldsymbol{1}$ 表示全 1 向量.

- 对应的对偶函数为:
    $$
    \begin{aligned}
    g(\mathbf{y}) &= \inf_{0 \leq \mathbf{x} \leq \mathbf{u}} \mathcal{L}(\mathbf{x}, \mathbf{y}) = \inf_{0 \leq \mathbf{x} \leq \mathbf{u}} \left( -\langle \mathbf{c}, \mathbf{x}\rangle + \frac{1}{\theta} \sum_{j =1}^n \phi\left(\frac{x_j}{u_j}\right) - \langle \mathbf{y}, \mathbf{A} \mathbf{x}\rangle \right) + \langle \mathbf{y}, \mathbf{b} \rangle \\
    &= \inf_{0 \leq \mathbf{x} \leq \mathbf{u}} \left( -\langle \mathbf{A}^\top \mathbf{y} + \mathbf{c}, \mathbf{x}\rangle + \frac{1}{\theta} \sum_{j =1}^n \phi\left(\frac{x_j}{u_j}\right) \right) + \langle \mathbf{y}, \mathbf{b} \rangle
    \end{aligned}
    $$
    - 其内层最小化问题
        $$
        \inf_{0 \leq \mathbf{x} \leq \mathbf{u}} \left( - \sum_{j =1}^n (\mathbf{A}^\top \mathbf{y}+ \mathbf{c})_j x_j + \frac{1}{\theta} \sum_{j =1}^n \phi\left(\frac{x_j}{u_j}\right) \right) := \inf_{0 \leq \mathbf{x} \leq \mathbf{u}} \psi(\mathbf{x})
        $$
       -  注意到其全部都是逐分量求和的, 且分量间彼此解耦, 故可以独立对每个分量求最小值. 故考虑标量函数
            $$
            \psi_j(x_j ; y) = - (\mathbf{A}^\top \mathbf{y} + \mathbf{c})_j x_j + \frac{1}{\theta} \phi\left(\frac{x_j}{u_j}\right)
            $$
            其一阶导数为
            $$
            \frac{\partial \psi_j(x_j ; y)}{\partial x_j} = - (\mathbf{A}^\top \mathbf{y} + \mathbf{c})_j + \frac{1}{\theta u_j} \log\frac{x_j}{u_j - x_j}.
            $$
            且有, $\lim_{x_j \to 0^+} \psi'_j(x_j ; y) \to -\infty$, $\lim_{x_j \to u_j^-} \psi'_j(x_j ; y) \to \infty$. 又根据其二阶导数
            $$
            \frac{\partial^2 \psi_j(x_j ; y)}{\partial x_j^2} = \frac{1}{\theta} \frac{1}{x_j (u_j - x_j)} > 0, \quad \forall 0 < x_j < u_j.
            $$
            由此可知, $\psi_j(x_j ; y)$ 是严格凸函数, 且在区间 $(0, u_j)$ 内部有唯一的最小值, 且对全部 $j$ 都成立.


      - 对于每个分量的最小值, 由一阶导数为零的条件可得
          $$
          - (\mathbf{A}^\top \mathbf{y} + \mathbf{c})_j + \frac{1}{\theta u_j} \log\frac{x_j}{u_j - x_j} \implies x_j(\mathbf{y}) = \frac{u_j}{1 + \exp(-\theta u_j (\mathbf{A}^\top \mathbf{y} + \mathbf{c})_j)} = u_j \cdot \sigma (\theta u_j (\mathbf{A}^\top \mathbf{y} + \mathbf{c})_j).
          $$
          故用向量整合, 则为:
          $$
          \mathbf{x}(\mathbf{y}) = \mathbf{u} \odot \sigma(\theta \mathbf{u} \odot (\mathbf{A}^\top \mathbf{y} + \mathbf{c})).
          $$
      
    - 在成功求出内层最小化问题 $\mathbf{x}(\mathbf{y})$ 后, 则可以将其代入对偶函数 $g(\mathbf{y})$ 中, 得到
        $$
        g(\mathbf{y}) = \frac{1}{\theta} \boldsymbol{1}^\top\log\sigma(\theta \mathbf{u} \odot ( - \mathbf{A}^\top \mathbf{y} - \mathbf{c})) + \langle \mathbf{y}, \mathbf{b} \rangle
        $$
        因此, 对偶问题可以写为:
        $$
        \min_{\mathbf{y} \in \mathbb{R}^m} -g(\mathbf{y}) = \min_{\mathbf{y} \in \mathbb{R}^m} -\frac{1}{\theta} \boldsymbol{1}^\top\log\sigma(\theta \mathbf{u} \odot ( - \mathbf{A}^\top \mathbf{y} - \mathbf{c})) - \langle \mathbf{y}, \mathbf{b} \rangle
        $$
        又根据原函数和对偶函数的关系, 可以证明 $-g(\mathbf{y})$ 是 Lipschitz 光滑的, 因此所有梯度下降法都可以用来求解这个对偶问题.
        - 这里强调, 对于内层问题求出的解 $\mathbf{x}(\mathbf{y})$, 其是
          $$
          \mathbf{x}(\mathbf{y}) \in \arg\min_{0 \leq \mathbf{x} \leq \mathbf{u}} \mathcal{L}(\mathbf{x}, \mathbf{y})
          $$
          的最小值. 其求解过程保证了 box-constrain $0 \leq \mathbf{x} \leq \mathbf{u}$ 是一定可以被满足的. 然而对于等式约束 $\mathbf{A} \mathbf{x} = \mathbf{b}$, 其仍然是在 Lagrangian 中被当做惩罚项处理的, 因此 $\mathbf{x}(\mathbf{y})$ 并不一定满足等式约束 $\mathbf{A} \mathbf{x} = \mathbf{b}$. 
          - 只有当 $\mathbf{y}$ 取到最优解 $\mathbf{y}^\star$ 时, 才能保证 $\mathbf{x}(\mathbf{y}^\star)$ 满足等式约束 $\mathbf{A} \mathbf{x} = \mathbf{b}$. 理由如下:
            - 原问题的 Lagrangian 为: $\mathcal{L}(\mathbf{x}, \mathbf{y}) = -\langle \mathbf{c}, \mathbf{x}\rangle + \frac{1}{\theta} \sum_{j =1}^n \phi\left(\frac{x_j}{u_j}\right) - \langle \mathbf{y}, \mathbf{A} \mathbf{x} - \mathbf{b} \rangle$. 对偶问题为 $g(\mathbf{y}) = \inf_{0 \leq \mathbf{x} \leq \mathbf{u}} \mathcal{L}(\mathbf{x}, \mathbf{y}) = \mathcal{L} (\mathbf{x}(\mathbf{y}), \mathbf{y})$. 由对偶问题的最优性条件可知, 
              $$
              \nabla g(\mathbf{y}) = \nabla_\mathbf{y} \mathcal{L}(\mathbf{x}(\mathbf{y}), \mathbf{y}) + [\nabla_\mathbf{x} \mathcal{L}(\mathbf{x}(\mathbf{y}), \mathbf{y})]^\top \frac{\partial \mathbf{x}(\mathbf{y})}{\partial \mathbf{y}} 
              $$
              而又根据 $\nabla_\mathbf{x} \mathcal{L}(\mathbf{x}(\mathbf{y}), \mathbf{y}) = 0$ 的最优性条件, 则有 
              $$
              \nabla g(\mathbf{y}) = \nabla_\mathbf{y} \mathcal{L}(\mathbf{x}(\mathbf{y}), \mathbf{y}) = -\mathbf{A} \mathbf{x}(\mathbf{y}) + \mathbf{b}.
              $$
              因此对于最优解 $\mathbf{y}^\star$, 有 $\nabla g(\mathbf{y}^\star) = 0 \implies \mathbf{A} \mathbf{x}(\mathbf{y}^\star) = \mathbf{b}$, 即 $\mathbf{x}(\mathbf{y}^\star)$ 满足等式约束 $\mathbf{A} \mathbf{x} = \mathbf{b}$.

### Forward pass in GLinSAT

由上面的推导, 我们得到对于神经网络的输出 $\mathbf{c} \in \mathbb{R}^n$, GLinSAT 需要向前传播得到满足约束的 $\mathbf{x} \in \mathbb{R}^n$. 而该约束已经转化成了一个对偶问题, 其形式为:
$$
\min_{\mathbf{y} \in \mathbb{R}^m} F(\mathbf{y}) 
$$
其中 $F(\mathbf{y}) = -g(\mathbf{y})$ 是一个 Lipschitz 光滑函数, 且其梯度为
$$
\nabla F(\mathbf{y}) = \mathbf{A} \mathbf{x}(\mathbf{y}) - \mathbf{b}, \qquad \mathbf{x}(\mathbf{y}) = \mathbf{u} \odot \sigma(\theta \mathbf{u} \odot (\mathbf{A}^\top \mathbf{y} + \mathbf{c}))
$$

故笼统地讲, GLinSAT 的前向传播过程即为:
$$
\mathbf{c} \mapsto \mathbf{y}^\star \mapsto \mathbf{x}(\mathbf{y}^\star)
$$

故下面的核心是如何高效地求解对偶问题 $\min_{\mathbf{y} \in \mathbb{R}^m} F(\mathbf{y})$. 
- 已知 $F(\mathbf{y})$ 是一个 Lipschitz 光滑函数, 且其梯度为 $\nabla F(\mathbf{y}) = \mathbf{A} \mathbf{x}(\mathbf{y}) - \mathbf{b}$, 其中 $\mathbf{x}(\mathbf{y}) = \mathbf{u} \odot \sigma(\theta \mathbf{u} \odot (\mathbf{A}^\top \mathbf{y} + \mathbf{c}))$. 故对任意 $\mathbf{y}_1, \mathbf{y}_2 \in \mathbb{R}^m$, 有
    $$
    \begin{aligned}
    \|\nabla F(\mathbf{y}_1) - \nabla F(\mathbf{y}_2)\| &= \|\mathbf{A} \mathbf{x}(\mathbf{y}_1) - \mathbf{A} \mathbf{x}(\mathbf{y}_2)\| \\
    &\leq \|\mathbf{A}\| \cdot \|\mathbf{x}(\mathbf{y}_1) - \mathbf{x}(\mathbf{y}_2)\| \\ 
    &= \|\mathbf{A}\| \cdot \|\mathbf{u} \odot \sigma(\theta \mathbf{u} \odot (\mathbf{A}^\top \mathbf{y}_1 + \mathbf{c})) - \mathbf{u} \odot \sigma(\theta \mathbf{u} \odot (\mathbf{A}^\top \mathbf{y}_2 + \mathbf{c}))\| \\
    &\leq \frac{\theta}{4} \max_j u_j^2 \cdot \|\mathbf{A}\|^2 \cdot \|\mathbf{y}_1 - \mathbf{y}_2\|
    \end{aligned}
    $$
    - 其中, 最后一个不等式是根据 sigmoid 的性质: $|\sigma(a) - \sigma(b)| \leq \frac{1}{4} |a - b|$ 得到的. 故 $F(\mathbf{y})$ 是一个 Lipschitz 光滑函数, 且其 Lipschitz 常数为 $L = \frac{\theta}{4} \max_j u_j^2 \cdot \|\mathbf{A}\|^2$.

- 给定 $F$ 的 convex & Lipschitz 光滑性质, 因此可以使用 Nesterov 型的 primal-dual 梯度加速方法 (**A**daptive**P**rimal**D**ual**A**ccelerated**G**radient**D**escent) 对这个目标函数进行优化, 以得到 $\mathcal{O}(1/k^2)$ 的收敛率. 
  - 之所以不使用 vanilla NAG, 是因为我们的本质期待还是高效的求解 primal 问题 $\mathbf{x}^\star$. 因此我们在对 $F(\mathbf{y})$ 进行优化的过程中, 同时要维护一个 $\mathbf{x}^k$ 的优化序列, 使得对于对偶问题的加速求解能够有效转化为 primal 问题的加速求解. 此外 Adaptive 这里是说我们在每次迭代中, 通过某种 linesearch 方法, 得到一个更为适配的符合当前迭代函数的 Lipschitz 常数 (记为 $M^k$), 以便更快地收敛 (而不是使用全局保守的 Lipschitz 下界).
  - 下具体讨论一下 GLinSAT 的 Nesterov 加速算法. 其主体同样采用三阶段的 Nesterov 加速方法:
    $$
    \begin{aligned}
    \boldsymbol{\lambda}^{k+1} &= (1-\tau^{k+1}) \boldsymbol{\eta}^k + \tau^{k+1} \boldsymbol{\zeta}^k \\
    \boldsymbol{\zeta}^{k+1} &= \boldsymbol{\zeta}^k - \alpha^{k+1} \nabla F(\boldsymbol{\lambda}^{k+1}) \\
    \boldsymbol{\eta}^{k+1} &= (1-\tau^{k+1}) \boldsymbol{\eta}^k + \tau^{k+1} \boldsymbol{\zeta}^{k+1}
    \end{aligned}
    $$
    - 其中, $\boldsymbol{\eta}^k \in \mathbb{R}^m$ 是对偶问题的加速序列, $\boldsymbol{\zeta}^k \in \mathbb{R}^m$ 是对偶问题的梯度下降序列, $\boldsymbol{\lambda}^{k+1} \in \mathbb{R}^m$ 是进行梯度计算的查询点. 其都是在对偶空间中完成的. 真正的 primal 变量是通过 $\mathbf{x}(\boldsymbol{\lambda}^{k+1})$ 来计算的. 
    - $\tau^{k+1}$ 是 Nesterov 的加速参数, $\alpha^{k+1}$ 是步长参数, 其可以通过某种 linesearch 方法自适应地选择. 
    - 初始化时, 令 $k=0$, $\boldsymbol{\eta}^0 = \boldsymbol{\zeta}^0 = \mathbf{y}^0$. 且 $\mathbf{x}^0 = \mathbf{x}(\mathbf{y}^0)$. 以及 Lipschitz 常数 $M^0 = L^0$. 额外两个加速过程的迭代参数, $\beta^0 = \alpha^0  = 0$. 
    - 算法最终的停止条件为
      $$
      \|\mathbf{A} \mathbf{x}^k - \mathbf{b}\| \leq \epsilon
      $$

    - 首先, 计算 $\alpha^{k+1}$, 其为如下二次方程  $M^k (\alpha^{k+1})^2 - \alpha^{k+1} - \beta^k = 0$ 的正根, 即
      $$
      \alpha^{k+1} := \frac{1 + \sqrt{1 + 4 M^k \beta^k}}{2 M^k}.
      $$
      并接着定义
      $$
      \beta^{k+1} := \beta^k + \alpha^{k+1}.
      $$
      通过上述的定义方法, 恒有如下等式成立, 其是 Nesterov 加速方法推导的核心:
      $$
      M^k (\alpha^{k+1})^2 \equiv \beta^{k+1} 
      $$
      该表达式在迭代中始终被满足. 

    - 接着, 计算加权系数 $\tau^{k+1}$, 其为
      $$
      \tau^{k+1} := \frac{\alpha^{k+1}}{\beta^{k+1}}.
      $$

    - 然后, 计算查询点 $\boldsymbol{\lambda}^{k+1}$, 其为
      $$
      \boldsymbol{\lambda}^{k+1} := (1 - \tau^{k+1}) \boldsymbol{\eta}^k + \tau^{k+1} \boldsymbol{\zeta}^k.
      $$


    - 然后计算查询点对应的 primal 变量 $\mathbf{x}(\boldsymbol{\lambda}^{k+1})$, 其为
      $$
      \mathbf{x}(\boldsymbol{\lambda}^{k+1}) := \mathbf{u} \odot \sigma( - \theta \mathbf{u} \odot ( - \mathbf{A}^\top \boldsymbol{\lambda}^{k+1} -  \mathbf{c})).
      $$
      以及梯度信息
      $$
        \nabla F(\boldsymbol{\lambda}^{k+1}) = \mathbf{A} \mathbf{x}(\boldsymbol{\lambda}^{k+1}) - \mathbf{b}.
        $$

    - 然后更新对偶变量 $\boldsymbol{\zeta}^{k+1}$, 其为
      $$
      \boldsymbol{\zeta}^{k+1} := \boldsymbol{\zeta}^k - \alpha^{k+1} \nabla F(\boldsymbol{\lambda}^{k+1}).
      $$

    - 然后更新对偶变量 $\boldsymbol{\eta}^{k+1}$, 其为
      $$
        \boldsymbol{\eta}^{k+1} := (1 - \tau^{k+1}) \boldsymbol{\eta}^k + \tau^{k+1} \boldsymbol{\zeta}^{k+1}.
        $$
        若将上述更新过程进行合并整理, 可以得到
        $$
        \boldsymbol{\eta}^{k+1} = \boldsymbol{\lambda}^{k+1} - \frac{1}{M^k} \nabla F(\boldsymbol{\lambda}^{k+1}).
        $$

    - 最后, 进行自适应的尝试更新 Lipschitz 常数 $M^{k+1}$. 观察上面的等式, $1/M^k$ 是更新的步长. 如果 $M^k$ 太大, 则步长太小, 更新过于保守. 因此我们需要随着迭代的不断进行, 不断尝试得到一个更小更为准确的曲率系数 $M^k$, 以得到更为合理的步长. 事实上, 我们进行尝试的选择依据就是 $F(\mathbf y) \le F(\mathbf x)+\left\langle\nabla F(\mathbf x),\mathbf y-\mathbf x\right\rangle+\frac{M}{2}\|\mathbf y-\mathbf x\|_2^2$. 将具体的迭代点代入, 最终的 linesearch 的条件可以化简整理为:
        $$
        F(\boldsymbol{\eta}^{k+1}) \leq F(\boldsymbol{\lambda}^{k+1}) - \frac{1}{2M^k} \|\nabla F(\boldsymbol{\lambda}^{k+1})\|^2
        $$
        也就是我们本身是希望 $M^k$ 选择的越小越好, 但是如果当前步骤不满足上述条件, 则说明 $M^k$ 选择的过小, 则将该 $M^k$ double, 直到满足上述条件为止. 然后再下一轮的时候再尝试选择一个更小的 $M^{k+1}$ (取 half), 以便更快地收敛. 
        - 当然在实践中还会有一些工程参数, 例如还会额外引入一个 tolerance 阈值 $\delta>0$, 只有当 $F(\boldsymbol{\eta}^{k+1}) > F(\boldsymbol{\lambda}^{k+1}) - \frac{1}{2M^k} \|\nabla F(\boldsymbol{\lambda}^{k+1})\|^2 + \delta$ 时, 才视为成功, 并且只有连续成功两次时, 才会正式更新 $M^{k+1} = M^k / 2$. 这样可以避免过于频繁的更新 $M^k$, 导致不稳定.

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260624213825.png)


上述的迭代复杂度为
$$
\mathcal O\left(
\|\mathbf A\|_2\max(\mathbf u)
\sqrt{\frac{\theta R}{\varepsilon}}
\right),
$$
- $\theta$ 越大, 意味着 entropy regularization 越弱, 输出更接近原始 LP 的极点解; 但同时对偶目标的光滑常数变大, forward pass 更难求解.



### Backward pass in GLinSAT

整体而言 GLinSAT 在模型 pipeline 中的作用如下. 

$$
\text{DATA} \xrightarrow{\text{Neural Network}} \mathbf{c} \xrightarrow{\text{GLinSAT}} \mathbf{x} \xrightarrow{\text{Loss Function}} \ell (\mathbf{x})
$$

若将 GLinSAT 内部再进一步拆开, 则可以得到如下的流程图:
$$
\text{DATA} \xrightarrow{\text{Neural Network}} \mathbf{c} \xrightarrow{\text{Dual Problem}} \mathbf{y}^\star(\mathbf{c}) \xrightarrow{\text{Primal Solution}} \mathbf{x}(\mathbf{y}^\star(\mathbf{c})) \xrightarrow{\text{Loss Function}} \ell (\mathbf{x})
$$

因此, 在 backward pass 中, 我们需要计算 $\frac{\partial \ell}{\partial \mathbf{c}}$. 

***AutoGrad***

一种简单的方法就是自动调用 PyTorch 等深度学习框架进行自动的反向传播, 由于注意到在我们求解 GLinSAT 内部时, 要进行 $K$ 次的 APDAGD 迭代, 尽管这个迭代过程本身全部是可微的, 但若 unrolling 展开的话会极大的增加计算图的复杂度, 使得反向传播的计算量和显存消耗都非常大. 

***CG-based Backward***

更为经济的做法是探究最终最优点处的模型结构, 利用模型的相关数学方程结构来帮助推导. 
- 回顾, 优化的对偶目标为
    $$
    \min_{\mathbf{y} \in \mathbb{R}^m} F(\mathbf{y})
    $$
    且
    $$
    \nabla F(\mathbf{y}) = \mathbf{A} \mathbf{x}(\mathbf{y}) - \mathbf{b}
    $$
    在最优点处, 根据 $\nabla F(\mathbf{y}^\star) = 0$, 则有
    $$
    \mathbf{h}(\mathbf{y},  \mathbf{c}) :=\mathbf{A} \mathbf{x}(\mathbf{y}^\star) - \mathbf{b} = \mathbf{A} \left(\mathbf{u} \odot \sigma(-\theta \mathbf{u} \odot (-\mathbf{A}^\top \mathbf{y}^\star - \mathbf{c}))\right) - \mathbf{b} = 0
    $$

- 因此, 在最优点处, 我们有一个隐函数关系 $\mathbf{h}(\mathbf{y}^\star; \mathbf{c}) = 0$. 根据隐函数定理, 我们可以对 $\mathbf{y}^\star$ 关于 $\mathbf{c}$ 求导, 得到
    $$
    \frac{\partial\mathbf y}{\partial\mathbf c}
    =
    -
    \left(
    \frac{\partial\mathbf h}{\partial\mathbf y}
    \right)^{-1}
    \frac{\partial\mathbf h}{\partial\mathbf c}.
    $$

    其中可以分别代数求得
    $$
    \frac{\partial\mathbf h}{\partial\mathbf y}
    =
    \mathbf A
    \operatorname{Diag}
    \left(
    \theta\mathbf x\odot(\mathbf u-\mathbf x)
    \right)
    \mathbf A^\top.
    $$
    以及
    $$
    \frac{\partial\mathbf h}{\partial\mathbf c}
    =
    \mathbf A
    \operatorname{Diag}
    \left(
    \theta\mathbf x\odot(\mathbf u-\mathbf x)
    \right).
    $$
    因此最终得到的
    $$
    \frac{\partial\mathbf y}{\partial\mathbf c}
    =
    -
    \left[
    \mathbf A
    \operatorname{Diag}
    \left(
    \theta\mathbf x\odot(\mathbf u-\mathbf x)
    \right)
    \mathbf A^\top
    \right]^{-1}
    \mathbf A
    \operatorname{Diag}
    \left(
    \theta\mathbf x\odot(\mathbf u-\mathbf x)
    \right).
    $$

- 因此再代入更完整的计算图中, 从 $\mathbf{c}$ 出发, 最终得到的损失函数的梯度关系为:
    $$
    \frac{d l}{d\mathbf c}
    =
    \frac{\partial l}{\partial\mathbf x}
    \left(
    \frac{\partial\mathbf x}{\partial\mathbf c}
    +
    \frac{\partial\mathbf x}{\partial\mathbf y}
    \frac{\partial\mathbf y}{\partial\mathbf c}
    \right).
    $$
    故代入隐式微分结果, 有
    $$
    \frac{\partial l}{\partial\mathbf c}
    =
    \frac{\partial l}{\partial\mathbf x}
    \frac{\partial\mathbf x}{\partial\mathbf c}
    -
    \frac{\partial l}{\partial\mathbf x}
    \frac{\partial\mathbf x}{\partial\mathbf y}
    \left(
    \frac{\partial\mathbf h}{\partial\mathbf y}
    \right)^{-1}
    \frac{\partial\mathbf h}{\partial\mathbf c}.
    $$
    不过在论文中还进行了一个更 generalized 的处理. 其若允许对偶变量 $\mathbf{y}$ 也和 $\mathbf{x}$ 一样, 直接参与到损失函数中, 则最终的梯度公式为:
    $$
    \begin{aligned}
    \frac{\partial l}{\partial\mathbf c}
    ={}&
    \frac{\partial l}{\partial\mathbf x}
    \frac{\partial\mathbf x}{\partial\mathbf c}
    &-
    \left(
    \frac{\partial l}{\partial\mathbf x}
    \frac{\partial\mathbf x}{\partial\mathbf y}
    +
    \frac{\partial l}{\partial\mathbf y}
    \right)
    \left(
    \frac{\partial\mathbf h}{\partial\mathbf y}
    \right)^{-1}
    \frac{\partial\mathbf h}{\partial\mathbf c}.
    \end{aligned}
    $$
    若想还原回原始的 $\mathbf{y}$ 不参与损失函数的情况, 则只需要令 $\frac{\partial l}{\partial\mathbf y} = 0$ 即可.


- 在得到理论的梯度传播公式后, 我们还需要考虑如何高效地计算这个 VJP. 故我们可以将其转化为一个线性方程组求解问题, 其形式为
    $$
    \left(
    \frac{\partial l}{\partial\mathbf x}
    \frac{\partial\mathbf x}{\partial\mathbf y}
    +
    \frac{\partial l}{\partial\mathbf y}
    \right)
    \left(
    \frac{\partial\mathbf h}{\partial\mathbf y}
    \right)^{-1} \implies 
    \left(
    \frac{\partial\mathbf h}{\partial\mathbf y}
    \right)\mathbf v
    =
    \left(
    \frac{\partial l}{\partial\mathbf x}
    \frac{\partial\mathbf x}{\partial\mathbf y}
    +
    \frac{\partial l}{\partial\mathbf y}
    \right)^\top \iff \mathbf{H}\mathbf{v} = \mathbf{r}^\top
    $$
    从上述线性系统中求解出 $\mathbf{v}$. 其中为方便起见, 记 $\mathbf{H} = \frac{\partial\mathbf h}{\partial\mathbf y}$, $\mathbf{r} = \frac{\partial l}{\partial\mathbf x} \frac{\partial\mathbf x}{\partial\mathbf y} + \frac{\partial l}{\partial\mathbf y}$.
    - 由于 $h(\mathbf{y}) = \mathbf{A} \mathbf{x}(\mathbf{y}) - \mathbf{b}$, 故将具体的表达式代入整理, 则上述线性系统 LHS 的系数为
    $$
    \mathbf{H} = 
    \frac{\partial\mathbf h}{\partial\mathbf y}
    =
    \mathbf A
    \operatorname{Diag}
    \left(
    \theta\mathbf x\odot(\mathbf u-\mathbf x)
    \right)
    \mathbf A^\top := \mathbf{A} \mathbf{D} \mathbf{A}^\top.
    $$
    - 且 $\frac{\partial\mathbf h}{\partial\mathbf y}$ 是对称半正定的. 若进一步 $\mathbf{A}$ 是行满秩, 则是对称正定的. 因此可以使用 CG 方法高效求解该线性系统, 以得到 $\mathbf{v}$. 这是因为, 上面我们展示了, $\mathbf{H} = \mathbf{A} \mathbf{D} \mathbf{A}^\top$ 是对称正定的, 因此求解 $\mathbf{H}\mathbf{v} = \mathbf{r}^\top$ 就等价于求解 QP
        $$
        \min_{\mathbf{v}} \frac{1}{2} \mathbf{v}^\top \mathbf{H} \mathbf{v} - \mathbf{r}^\top \mathbf{v}
        $$
        而这样的方法就很适合使用 CG 方法.

    - Conjugate Gradient (CG) 方法是求解对称正定线性系统的经典方法. 其基本思想就是沿着共轭方向去进行搜索迭代, 而共轭方向可以简单理解为由 $\mathbf{H}$ 诱导的正交方向. 其具体迭代过程如下:
      - 初始化 $\mathbf{v}^0 = 0$, 初始化残差 $\mathbf{s}^0 = \mathbf{r} - \mathbf{H} \mathbf{v}^0 = \mathbf{r}$, 初始化搜索方向 $\mathbf{p}^0 = \mathbf{s}^0$. 即第一步沿着负梯度方向进行搜索.
      - 在第 $k$ 步迭代中, 当前迭代点为 $\mathbf{v}^k$, 搜索方向为 $\mathbf{p}^k$, 步长为 $\gamma^k$, 我们希望沿着直线 $\mathbf{v}^{k+1} = \mathbf{v}^k + \gamma^k \mathbf{p}^k$ 进行搜索, 使得 $\mathbf{v}^{k+1}$ 最小化二次函数 $\min_\gamma Q(\mathbf{v}) = \frac{1}{2} \mathbf{v}^\top \mathbf{H} \mathbf{v} - \mathbf{r}^\top \mathbf{v}$. 故可以求得第 $k$ 步的步长为
        $$
        \gamma^k = \frac{(\mathbf{s}^k)^\top \mathbf{s}^k}{(\mathbf{p}^k)^\top \mathbf{H} \mathbf{p}^k}.
        $$
        进而以此更新迭代点
        $$
        \mathbf{v}^{k+1} = \mathbf{v}^k + \gamma^k \mathbf{p}^k.
        $$
        更新残差
        $$
        \mathbf{s}^{k+1} = \mathbf{s}^k - \gamma^k \mathbf{H} \mathbf{p}^k.
        $$
      - 检查迭代停止条件
          $$
          \frac{\|\mathbf{s}^{k+1}\|}{\|\mathbf{r}\|} \leq \epsilon
          $$
          若满足, 则停止迭代, 否则继续迭代, 更新搜索方向:
          $$
          \mathbf{p}^{k+1} = \mathbf{s}^{k+1} + \frac{(\mathbf{s}^{k+1})^\top \mathbf{s}^{k+1}}{(\mathbf{s}^k)^\top \mathbf{s}^k} \mathbf{p}^k.
          $$
          - 这里 $\frac{(\mathbf{s}^{k+1})^\top \mathbf{s}^{k+1}}{(\mathbf{s}^k)^\top \mathbf{s}^k}$ 是 CG 方法中用于保证共轭方向的系数. 其可以通过 Gram-Schmidt 正交化的方式得到, 以保证 $\mathbf{p}^{k+1}$ 与 $\mathbf{p}^k$ 在 $\mathbf{H}$ 诱导的内积下正交.
    - 可以看到, 通过引入 CG 方法, 我们完全摒弃了矩阵求逆, 而是化成了一系列的矩阵乘法.而且中间的 $\mathbf{D}$ 还是对角矩阵, 其实质上进一步被简化为标量乘法. 计算量被大幅简化.           

