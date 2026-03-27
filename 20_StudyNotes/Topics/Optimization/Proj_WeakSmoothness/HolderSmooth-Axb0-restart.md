# From Canonical to Affine: The Feasible Affine Model and Restart Smoothing

本节考虑一个稍微 general 一些的优化问题, 即在 Canonical 模型的基础上, 将残差 $\mathbf A\mathbf x$ 扩展为 affine 模型 $\mathbf A\mathbf x-\mathbf b$. 

具体地, 令 $\mathbf{A} \in \mathbb{R}^{m\times n}$, $\mathbf{b} \in \mathbb{R}^m$, $1<p \leq 2$, 对应共轭指数 $q = \frac{p}{p-1} \in [2,+\infty)$.

我们研究如下优化问题:
$$
\min_{\mathbf x\in\mathbb R^n} F_{\mathbf b}(\mathbf x)
$$
其中
$$
F_{\mathbf b}(\mathbf x)
:= h(\mathbf A\mathbf x-\mathbf b) = 
\frac1p\|\mathbf A\mathbf x-\mathbf b\|_p^p = \max_{\mathbf y\in\mathbb R^m}\left\{\langle \mathbf A\mathbf x-\mathbf b,\mathbf y\rangle - \frac1q\|\mathbf y\|_q^q\right\}.
$$

对应的 smooth surrogate function 可以定义为
$$
F_{\mathbf b, \mu}(\mathbf x) := \max_{\mathbf y\in\mathbb R^m}\left\{\langle \mathbf A\mathbf x-\mathbf b,\mathbf y\rangle - \frac1q\|\mathbf y\|_q^q - \frac\mu2\|\mathbf y\|_2^2\right\}.
$$

余下的部分我们会分为如下两种情况进行讨论:
1. 对于 feasible point, 即 $\mathcal{X}^\star = \{\mathbf x\in\mathbb R^n: \mathbf A\mathbf x=\mathbf b\} \neq \varnothing$. 此时, 对于任意 $\tilde{\mathbf x}\in\mathcal{X}^\star$, 任意 $\mathbf x\in\mathbb R^n$, 都有
    $$
    \mathbf{A}\mathbf{x} - \mathbf{b}  = \mathbf{A}\mathbf{x} - \mathbf{A}\tilde{\mathbf{x}} := \mathbf{A}\mathbf{z},
    $$
    即相当于在 feasible point 上的 Canonical 模型. 因此, 该问题的分析与 Canonical 模型几乎完全一致, 可以直接进行推广. 
    - 对于这类情况的分析, 我们的研究重点在于指出其可以通过 Restarting 技术来达到线性收敛的效果. 
    - 纯粹的由推广可以较易得到的结论的证明将不再赘述.

2. 反之, 若 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 无解, 则此时最优值将不再为 $0$, 此时我们将重新给出更细致的相应分析.


## Feasible Case： $\mathcal{X}^\star = \{\mathbf x\in\mathbb R^n: \mathbf A\mathbf x=\mathbf b\} \neq \varnothing$

### 从 Canonical 模型到 Affine 模型的推广

***Proposition* (原函数的最大值解)**: 对于 $F_{\mathbf b}(\mathbf x) = \frac1p\|\mathbf A\mathbf x-\mathbf b\|_p^p$, 其最大值解 $\mathbf y_F^\star$ 满足
$$
\mathbf y_F^\star = \arg\max_{\mathbf y\in\mathbb R^m}\left\{\langle \mathbf A\mathbf x-\mathbf b,\mathbf y\rangle - \frac1q\|\mathbf y\|_q^q\right\} = 
\text{sign}(\mathbf A\mathbf x-\mathbf b)\odot|\mathbf A\mathbf x-\mathbf b|^{p-1}.
$$

其中 $\odot$ 与 $\text{sign}(\cdot)$ 等运算符号的定义与 Canonical 模型中的相同, 都是逐元素的.


***Proposition* (平滑 surrogate 函数的最优性条件, 梯度表达与 Lipschitz Smoothness)**: 对于 $F_{\mathbf b, \mu}(\mathbf x) = \max_{\mathbf y\in\mathbb R^m}\left\{\langle \mathbf A\mathbf x-\mathbf b,\mathbf y\rangle - \frac1q\|\mathbf y\|_q^q - \frac\mu2\|\mathbf y\|_2^2\right\}$, 该最大化问题在给定 $\mathbf x$ 时具有唯一的最优解 $\mathbf y_\mu^\star(\mathbf x)$, 满足如下一阶最优性条件:
$$
\mathbf A\mathbf x-\mathbf b - \nabla h(\mathbf y_\mu^\star(\mathbf{x})) - \mu \mathbf y_\mu^\star(\mathbf x) = \mathbf 0.
$$

此外,  其梯度与最优解 $\mathbf y_\mu^\star(\mathbf x)$ 之间的关系为
$$
\nabla F_{\mathbf b, \mu}(\mathbf x) = \mathbf A^\top \mathbf y_\mu^\star(\mathbf x).
$$

并且 $\nabla F_{\mathbf{b},\mu}$  是 Lipschitz 连续的, 其 Lipschitz 常数为 $L_\mu = \frac{\|\mathbf A\|^2}{\mu}$.


$\diamond$

- *Proof*
    - 上述证明与 Canonical 模型中的证明完全一致.

$\square$

***Proposition* (可行情况下, 原问题与平滑问题共享最优解)**: 若 $\mathcal{X}^\star  = \{\mathbf x\in\mathbb R^n: \mathbf A\mathbf x=\mathbf b\} \neq \varnothing$, 则对于原问题:
$$
\min_{\mathbf x\in\mathbb R^n} F_{\mathbf b}(\mathbf x) = \min_{\mathbf x\in\mathbb R^n} \frac1p\|\mathbf A\mathbf x-\mathbf b\|_p^p,
$$
与任意 $\mu > 0$ 下的平滑 surrogate 问题
$$
\min_{\mathbf x\in\mathbb R^n} F_{\mathbf b, \mu}(\mathbf x) = \min_{\mathbf x\in\mathbb R^n} \max_{\mathbf y\in\mathbb R^m}\left\{\langle \mathbf A\mathbf x-\mathbf b,\mathbf y\rangle - \frac1q\|\mathbf y\|_q^q - \frac\mu2\|\mathbf y\|_2^2\right\},
$$
具有完全相同的最优解集合 $\mathcal{X}^\star$, 且最优值也都同为 $0$.

- *Proof*
  - 对于原问题和平滑 surrogate 问题的证明都可以通过将 $\mathbf A\mathbf x-\mathbf b$ 代入到 Canonical 模型中的相应证明中来得到.
  - 事实上, 对于 canonical 模型与当前 Feasible Case 的 affine 模型, 其最优解集合相当于通过 $\mathbf A\mathbf x-\mathbf b$ 进行了一次仿射变换, 因此两者的最优解集合是完全相同的.
  

$\square$


***Proposition* (pointwise smoothing bias)**: 对于任意 $\mathbf x\in\mathbb R^n$, 任意 $\mu > 0$, 都有
$$
0 \leq F_{\mathbf b}(\mathbf x) - F_{\mathbf b, \mu}(\mathbf x) \leq \frac{\mu}{2} m^{\frac{2-p}{p}} \|\mathbf A\mathbf x-\mathbf b\|_p^{2(p-1)} = D_{p,m} \mu F_{\mathbf b}(\mathbf x)^{2-2/p}.
$$
其中 $D_{p,m} = \frac{1}{2} m^{\frac{2-p}{p}} p^{2-\frac{2}{p}}$ 是一个仅依赖于 $p$ 和 $m$ 的常数.

- *Proof*
  - 关于下界的证明可以直接由定义得到. 
  - 关于上界, 我们可以通过如下步骤来得到 (事实上也是完全相同的):
    - 将原问题的最优解 $\mathbf y_F^\star$ 代入到平滑 surrogate 问题的定义中, 可以得到
        $$
        F_{\mathbf b, \mu}(\mathbf x) \geq \langle \mathbf A\mathbf x-\mathbf b,\mathbf y_F^\star\rangle - \frac1q\|\mathbf y_F^\star\|_q^q - \frac\mu2\|\mathbf y_F^\star\|_2^2 = F_{\mathbf b}(\mathbf x) - \frac\mu2\|\mathbf y_F^\star\|_2^2.
        $$
    - 同时, 注意到 $\|\mathbf y_F^\star\|_q^q = \|\mathbf A\mathbf x-\mathbf b\|_p^p$, 因此进一步由范数的关系可知
        $$
        \|\mathbf y_F^\star\|_2^2 \leq m^{\frac{2-p}{p}} \|\mathbf A\mathbf x-\mathbf b\|_p^{2(p-1)}.
        $$
    - 综上, 可以得到
        $$
        F_{\mathbf b}(\mathbf x) - F_{\mathbf b, \mu}(\mathbf x) \leq \frac\mu2\|\mathbf y_F^\star\|_2^2 \leq \frac{\mu}{2} m^{\frac{2-p}{p}} \|\mathbf A\mathbf x-\mathbf b\|_p^{2(p-1)} = D_{p,m} \mu F_{\mathbf b}(\mathbf x)^{2-\frac{2}{p}}.
        $$


***Proposition* (精度转换)**: 对于任意 $\varepsilon > 0$, $\mu \leq  \frac{1}{2 D_{p,m}} \varepsilon^{\frac{2}{p}-1}$, 则对于任意 $\mathbf x\in\mathbb R^n$, 都有
$$
F_{\mathbf b, \mu}(\mathbf x)  \leq \frac{\varepsilon}{2}  \implies F_{\mathbf b}(\mathbf x) \leq \varepsilon.
$$


***Proposition* (单阶段优化算法的收敛率)**: 定义 $\mu_{\varepsilon} := \frac{1}{2 D_{p,m}} \varepsilon^{\frac{2}{p}-1}$, 且设 $R_0 = \text{dist}(\mathbf x_0, \mathcal{X}^\star)$. 若对光滑问题 $\min_{\mathbf x\in\mathbb R^n} F_{\mathbf b, \mu_\varepsilon}(\mathbf x)$ 应用 accelerated gradient method, 直到某迭代点 $\mathbf x_k$ 满足 $F_{\mathbf b, \mu_\varepsilon}(\mathbf x_k) \leq \frac{\varepsilon}{2}$, 则必有 $F_{\mathbf b}(\mathbf x_k) \leq \varepsilon$. 且迭代次数 $k$ 只需要满足
$$
k+1 \geq 2 \sqrt{2D_{p,m}} \|\mathbf A\|_2 R_0 \varepsilon^{-\frac{1}{p}},
$$
即可保证上述条件的满足. 从而, 
$$
k = \mathcal{O}\left(\|\mathbf A\|_2 R_0 \varepsilon^{-\frac{1}{p}}\right).
$$


### Restarting 技术下的线性收敛

下面这个部分, 我们将指出, 对于 Canonical 及延伸的 Feasible Case 的 affine 模型, 我们都可以利用其 Sharpness 的性质, 通过 Restarting 技术来达到线性收敛的效果.

首先我们给出如下引理. 该引理将给出 Feasible Case 下残差 $\mathbf A\mathbf x-\mathbf b$ 和距离 $\text{dist}(\mathbf x, \mathcal{X}^\star)$ 之间的关系, 即一个 error bound, 这本身也体现了 Feasible Case 下的 Sharpness. 

***Lemma*  (Feasible Case 下的 Sharpness)**:  设 $\mathbf{A} \in \mathbb{R}^{m\times n}$, $\mathbf{b} \in \mathbb{R}^m$, 且假设线性系统 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 是可行的, 即 $\mathcal{X}^\star = \{\mathbf x\in\mathbb R^n: \mathbf A\mathbf x=\mathbf b\} \neq \varnothing$. 记 $\sigma_{\min}^+(\mathbf A)$ 是 $\mathbf A$ 的最小非零奇异值,  则对于任意 $\mathbf x\in\mathbb R^n$, 都有
$$
\operatorname{dist}(\mathbf x, \mathcal{X}^\star)
\leq
\frac{\|\mathbf A\mathbf x-\mathbf b\|_2}{\sigma_{\min}^+(\mathbf A)} \leq
\frac{\|\mathbf A\mathbf x-\mathbf b\|_p}{\sigma_{\min}^+(\mathbf A)} \leq 
\frac{p^{\frac{1}{p}}}{\sigma_{\min}^+(\mathbf A)} F_{\mathbf b}(\mathbf x)^{\frac{1}{p}}.
$$
其中 
$$
F_{\mathbf b}(\mathbf x) := \frac1p\|\mathbf A\mathbf x-\mathbf b\|_p^p, \quad 1 < p \leq 2.
$$

$\diamond$

- *Proof*
  - 任取一可行点 $\tilde{\mathbf x}\in\mathcal{X}^\star$ (即满足 $\mathbf{A}\tilde{\mathbf{x}} = \mathbf{b}$), 以及任意给定 $\mathbf{x} \in  \mathbb{R}^n$, 有:
    - 对于 $\mathbf{A}$ 的核空间 $\text{Null}(\mathbf A)$, 由于其是 $\mathbb{R}^n$ 的线性子空间, 因此存在唯一的 $\mathbf{u} \in \text{Null}(\mathbf A)$, 与对应的正交补空间上的 $\mathbf{v} \in \text{Null}(\mathbf A)^\perp$, 使得 $\mathbf{x} - \tilde{\mathbf{x}} = \mathbf{u} + \mathbf{v} \in \mathbb{R}^n$. 即 $\mathbf{x} = \tilde{\mathbf{x}} + \mathbf{u} + \mathbf{v}$.
    - 定义 $\mathbf{x}^\Pi := \tilde{\mathbf{x}} + \mathbf{u}$, 则立刻有如下性质:
      - $\mathbf{x}^\Pi \in \mathcal{X}^\star$. 理由如下.
        - $\tilde{\mathbf{x}} \in \mathcal{X}^\star$ 且 $\mathbf{u} \in \text{Null}(\mathbf A)$, 因此 $\mathbf{A}\mathbf{x}^\Pi = \mathbf{A}\tilde{\mathbf{x}} + \mathbf{A}\mathbf{u} = \mathbf{b}$. 换言之, $\mathcal{X}^\star = \{\tilde{\mathbf x} + \mathbf z: \mathbf A\mathbf z = \mathbf 0\} := \tilde{\mathbf x} + \text{Null}(\mathbf A)$.
      - $\mathbf{x}^\Pi$ 是 $\mathbf{x}$ 到 $\mathcal{X}^\star$ 的 Euclidean projection. 理由如下. 
        - 对于任意 $\mathbf{z} \in \mathcal{X}^\star$,  都存在 $\mathbf{w} \in \text{Null}(\mathbf A)$, 使得 $\mathbf{z} = \tilde{\mathbf x} + \mathbf{w}$. 
        - 因此 $\mathbf{x} - \mathbf{z} = (\tilde{\mathbf{x}} + \mathbf{u} + \mathbf{v}) - (\tilde{\mathbf x} + \mathbf{w}) = (\mathbf{u} - \mathbf{w}) + \mathbf{v}$.
        - 注意到, $\mathbf{u} - \mathbf{w} \in \text{Null}(\mathbf A)$, $\mathbf{v} \in \text{Null}(\mathbf A)^\perp$, 因此 $\langle \mathbf{u} - \mathbf{w}, \mathbf{v}\rangle = 0$. 从而
          $$
          \|\mathbf{x} - \mathbf{z}\|_2^2 = \|\mathbf{u} - \mathbf{w}\|_2^2 + \|\mathbf{v}\|_2^2 \geq \|\mathbf{v}\|_2^2 ,
          $$
        - 上式当且仅当 $\mathbf{w} = \mathbf{u}$ 时取到等号. 而此时恰有: 
          $$
          \mathbf{z} = \tilde{\mathbf x} + \mathbf{w} = \tilde{\mathbf x} + \mathbf{u} = \mathbf{x}^\Pi.
          $$
          且知
          $$
          \text{dist}(\mathbf x, \mathcal{X}^\star) = \|\mathbf{x} - \mathbf{x}^\Pi\|_2 = \|\mathbf{v}\|_2.
          $$
          此时我们建立起 $\mathbf{x}$ 到 $\mathcal{X}^\star$ 的距离与 $\mathbf{v}$ 之间的关系.
    - 另一方面, 我们将讨论 $\mathbf{A}\mathbf{x} - \mathbf{b}$ 与 $\mathbf{v}$ 之间的关系, 并根据奇异值将上述距离给出有效的上界. 
      - 由于 $\mathbf{A}\mathbf{x} - \mathbf{b} = \mathbf{A}\mathbf{x} - \mathbf{A}\tilde{\mathbf{x}} = \mathbf{A}\mathbf{v} + \mathbf{A}\mathbf{u} = \mathbf{A}\mathbf{v}$, 因此 $\mathbf{A}\mathbf{x} - \mathbf{b}$ 与 $\mathbf{v}$ 之间的关系完全由 $\mathbf{A}$ 的奇异值来决定.
      - 根据 SVD 的标准结论, 由于 $\mathbf{v} \in \text{Null}(\mathbf A)^\perp$, 而 $\sigma_{\min}^+(\mathbf A)$ 作为 $\mathbf{A}$ 的最小非零奇异值恰表示 $\mathbf{A}$ 在 $\text{Null}(\mathbf A)^\perp$ 上的最小伸缩因子, 因此
        $$
        \|\mathbf A\mathbf x-\mathbf b\|_2 = \|\mathbf{A}\mathbf{v}\|_2 \geq \sigma_{\min}^+(\mathbf A) \|\mathbf{v}\|_2 = \sigma_{\min}^+(\mathbf A) \text{dist}(\mathbf x, \mathcal{X}^\star).
        $$
    - 综上, 可以得到
      $$
        \text{dist}(\mathbf x, \mathcal{X}^\star) \leq \frac{\|\mathbf A\mathbf x-\mathbf b\|_2}{\sigma_{\min}^+(\mathbf A)} .
      $$

  - 最后根据在 $p \in (1,2]$ 区间内的范数关系, 以及 $F_{\mathbf b}(\mathbf x) = \frac1p\|\mathbf A\mathbf x-\mathbf b\|_p^p$, 可以得到
    $$
    \|\mathbf A\mathbf x-\mathbf b\|_2 \leq \|\mathbf A\mathbf x-\mathbf b\|_p = p^{\frac{1}{p}} F_{\mathbf b}(\mathbf x)^{\frac{1}{p}}.
    $$
    这样就得到了最后的结论
    $$
    \text{dist}(\mathbf x, \mathcal{X}^\star) \leq \frac{\|\mathbf A\mathbf x-\mathbf b\|_2}{\sigma_{\min}^+(\mathbf A)} \leq \frac{\|\mathbf A\mathbf x-\mathbf b\|_p}{\sigma_{\min}^+(\mathbf A)} \leq \frac{p^{\frac{1}{p}}}{\sigma_{\min}^+(\mathbf A)} F_{\mathbf b}(\mathbf x)^{\frac{1}{p}}.
    $$

$\square$


在得到该引理之后, 我们就可以通过 Restarting 技术来达到线性收敛的效果. 具体算法如下. 

***Algorithm* (Restarting Accelerated Gradient Method for Feasible Case)**: 

- INPUT: 初始点 $\mathbf x_0 \in \mathbb{R}^n$, 精度要求 $\varepsilon > 0$, 收缩因子 $\beta \in (0,1)$, Smooth surrogate 的优化算法 $\mathcal{M}$ (此处为 AGD).

- OUTPUT: 满足 $F_{\mathbf b}(\mathbf x_N) \leq \varepsilon$ 的 $\mathbf x_N$.

- 算法流程:
    1. 初始化: 给定 $\mathbf x_0 \in \mathbb{R}^n$, 设当前阶段数 $n \leftarrow 0$.
    2. 计算当前阶段的目标函数值之 gap:
        $$
        \Delta_n := F_{\mathbf b}(\mathbf x_n).
        $$
    3. 若 $\Delta_n \leq \varepsilon$, 则输出 $\mathbf x_n$ 并停止算法.
    4. 设定本阶段目标精度: 令 
        $$
        \varepsilon_n := \beta \Delta_n.
        $$
    5. 设定本阶段的平滑问题: 
        $$
         F_{\mathbf b, \mu_n}(\mathbf x) = \max_{\mathbf y\in\mathbb R^m}\left\{\langle \mathbf A\mathbf x-\mathbf b,\mathbf y\rangle - \frac1q\|\mathbf y\|_q^q - \frac{\mu_n}{2}\|\mathbf y\|_2^2\right\},
        $$
        其中 $\mu_n := \frac{1}{2 D_{p,m}} \varepsilon_n^{\frac{2}{p}-1}$.
    6. 在本阶段, 从 $\mathbf x_n$ 出发, 对 $\min_{\mathbf x\in\mathbb R^n} F_{\mathbf b, \mu_n}(\mathbf x)$ 应用优化算法 $\mathcal{M}$ (此处为 AGD), 直到某迭代点 $\mathbf x_{n+1}$ 满足 $F_{\mathbf b, \mu_n}(\mathbf x_{n+1}) \leq \frac{\varepsilon_n}{2}$.
    7. 更新阶段数: $n \leftarrow n + 1$, 返回步骤 2.


<!-- Algorithm 1 Restarted smoothing for the feasible affine residual model

Input:
    initial point x_0 ∈ R^n
    target accuracy ε > 0
    contraction factor β ∈ (0,1)
    inner solver M = AGD

Initialize:
    n ← 0

Repeat:
    Δ_n ← F_b(x_n) = (1/p) ||Ax_n - b||_p^p

    if Δ_n ≤ ε then
        return x_n
    end if
    
    ε_n ← β Δ_n
    μ_n ← (2D_{p,m})^{-1} ε_n^{2/p - 1}
    
    define Φ_n(x) := F_{b,μ_n}(x)
    
    starting from x_n, run AGD on Φ_n
    until an iterate x_{n+1} is obtained such that
        Φ_n(x_{n+1}) ≤ ε_n / 2
    
    n ← n + 1

Output:
    x_n -->

$\diamond$

***Theorem* (Restarting 策略每阶段的收缩率)**: 在上述 Restarting Accelerated Gradient Method for Feasible Case 中, 应用上述算法. 令 $R_n := \text{dist}(\mathbf x_n, \mathcal{X}^\star)$, 则对于任意阶段 $n$, 其对应循环步数 $k_n$ 满足
$$
k_n + 1 \geq 2 \sqrt{2D_{p,m}} \|\mathbf A\|_2 R_n \varepsilon_n^{-\frac{1}{p}},
$$
则可以保证在该阶段的输出点 $\mathbf x_{n+1}$ 满足
$$
F_{\mathbf b}(\mathbf x_{n+1})  \leq \beta F_{\mathbf b}(\mathbf x_n) \iff \Delta_{n+1} \leq \beta \Delta_n.
$$

$\diamond$

- *Proof*
  - 首先, 对于当前 Feasible Case 的 affine 模型, 其最优值 $F_{\mathbf b}^\star$ 是 $0$. 因此, 对于任意阶段 $n$, 都有 $\Delta_n = F_{\mathbf b}(\mathbf x_n) - F_{\mathbf b}^\star = F_{\mathbf b}(\mathbf x_n)$.
  - 对于第 $n$ 个阶段, 由前面的定理, 只要 $k_n + 1 \geq 2 \sqrt{2D_{p,m}} \|\mathbf A\|_2 R_n \varepsilon_n^{-\frac{1}{p}}$, 则可以保证 $F_{\mathbf b, \mu_n}(\mathbf x_{n+1}) \leq \frac{\varepsilon_n}{2}$.
  - 由精度转换的定理, 可以得到 $F_{\mathbf b}(\mathbf x_{n+1}) \leq \varepsilon_n = \beta \Delta_n$. 从而, 可以得到 $\Delta_{n+1} = F_{\mathbf b}(\mathbf x_{n+1}) \leq \beta \Delta_n$.

$\square$


***Corollary* (由 Sharpness 保证的每 stage 下的常数更新)**: 在上述 *仿射可行系统的 sharpness / error bound*  的 Lemma 条件下, 上述算法的每个阶段 $n$ 的内部循环步数 $K$ 只要满足:
$$
K := \left\lceil 2 \sqrt{2D_{p,m}}   p^{\frac{1}{p}} \beta^{-\frac{1}{p}} \frac{\|\mathbf A\|_2}{\sigma_{\min}^+(\mathbf A)} \right\rceil,
$$
即可保证
$$
\Delta_{n+1} \leq \beta \Delta_n.
$$

- *Proof*
  - 由前面的引理, 可以得到, 对于任意 $\mathbf x \in \mathbb{R}^n$, 都有
    $$
    \text{dist}(\mathbf x, \mathcal{X}^\star) \leq \frac{p^{\frac{1}{p}}}{\sigma_{\min}^+(\mathbf A)} F_{\mathbf b}(\mathbf x)^{\frac{1}{p}} = \frac{p^{\frac{1}{p}}}{\sigma_{\min}^+(\mathbf A)} \Delta_n^{\frac{1}{p}}.
    $$

  - 而又知 $\varepsilon_n = \beta \Delta_n$, 因此
    $$
    R_n \varepsilon_n^{-\frac{1}{p}} \leq \frac{p^{\frac{1}{p}}}{\sigma_{\min}^+(\mathbf A)} \Delta_n^{\frac{1}{p}} (\beta \Delta_n)^{-\frac{1}{p}} = \frac{p^{\frac{1}{p}}}{\sigma_{\min}^+(\mathbf A)} \beta^{-\frac{1}{p}}.
    $$

  - 再代回每个stage固定比例收缩的定理的充分条件中, 即有:
    $$
    k_n + 1 \geq 2 \sqrt{2D_{p,m}}   p^{\frac{1}{p}} \beta^{-\frac{1}{p}} \frac{\|\mathbf A\|_2}{\sigma_{\min}^+(\mathbf A)} \geq 2 \sqrt{2D_{p,m}} \|\mathbf A\|_2 R_n \varepsilon_n^{-\frac{1}{p}} .
    $$

  - 从而, 只要取 $K := \left\lceil 2 \sqrt{2D_{p,m}}   p^{\frac{1}{p}} \beta^{-\frac{1}{p}} \frac{\|\mathbf A\|_2}{\sigma_{\min}^+(\mathbf A)} \right\rceil$, 就可以保证 $\Delta_{n+1} \leq \beta \Delta_n$.


$\square$



***Corollary* (Restarting 技术下的线性收敛)**: 在上述 Restarting Accelerated Gradient Method for Feasible Case 中, 设定每个阶段的内部循环上述 Corollary 中的 $K$ 次, 则 residual $\Delta_n$ 有:
$$
\Delta_n \leq \beta^n \Delta_0.
$$

因此, 为了达到 $F_{\mathbf b}(\mathbf x_n) = \Delta_n \leq \varepsilon$, 只需要满足
$$
N \geq \frac{\log(\Delta_0 / \varepsilon)}{\log(1/\beta)}.
$$

对应总的 AGD 迭代次数满足:
$$
NK  = 
\mathcal{O}\left(\sqrt{D_{p,m}}   p^{\frac{1}{p}} \beta^{-\frac{1}{p}} \frac{\|\mathbf A\|_2}{\sigma_{\min}^+(\mathbf A)} \log\left(\frac{\Delta_0}{\varepsilon}\right)\right).
$$

若进一步将 $p, m, \beta$ 视为常数, 则总的 AGD 迭代次数满足
$$
NK  = 
\mathcal{O}\left(\frac{\|\mathbf A\|_2}{\sigma_{\min}^+(\mathbf A)} \log\left(\frac{\Delta_0}{\varepsilon}\right)\right).
$$

- *Proof*
  - 由前面的 Corollary, 每个阶段 $n$ 的 residual $\Delta_n$ 满足 $\Delta_{n+1} \leq \beta \Delta_n$. 因此, 可以得到 $\Delta_n \leq \beta^n \Delta_0$.
  - 为了达到 $\Delta_n \leq \varepsilon$, 只需要满足 $\beta^n \Delta_0 \leq \varepsilon$, 从而 $n \geq \frac{\log(\Delta_0 / \varepsilon)}{\log(1/\beta)}$.
  - 每个阶段的内部循环步数为 $K$, 因此总的 AGD 迭代次数为 $NK = K\frac{\log(\Delta_0 / \varepsilon)}{\log(1/\beta)}$. 将 $K$ 的表达式代入, 即可得到总的 AGD 迭代次数的表达式.

$\square$


总结本章, 一旦我们从 $\frac1p\|\mathbf A\mathbf x\|_p^p$ 的 Canonical 模型推广到 $\frac1p\|\mathbf A\mathbf x-\mathbf b\|_p^p$ 的 affine 模型, 则只需保证 Feasible 假设
$$
\mathcal{X}^\star = \{\mathbf x\in\mathbb R^n: \mathbf A\mathbf x=\mathbf b\} \neq \varnothing,
$$
则几乎所有关于 Canonical 模型的结论都可以通过 $\mathbf A\mathbf x-\mathbf b$ 的仿射变换来进行推广.  其中, 最为重要的, 我们依然保留了
$$
F^\star = 0, \quad \mathcal{X}^\star = \{\mathbf x\in\mathbb R^n: \mathbf A\mathbf x=\mathbf b\},
$$
的良好性质, 因此依然可以将 $\Delta_n$ 定义为 $F_{\mathbf b}(\mathbf x_n) - F_{\mathbf b}^\star = F_{\mathbf b}(\mathbf x_n)$ 作为一个可以直接观测的 gap 来进行分析. 

在此基础上, 我们可以通过 Restarting 技术, 借助 Feasible Case 下的 Sharpness 
$$
\text{dist}(\mathbf x, \mathcal{X}^\star) \lesssim F_{\mathbf b}(\mathbf x)^{\frac{1}{p}},
$$
来达到线性收敛的效果.  

此时, 单阶段的迭代复杂度仍然是 $\mathcal{O}\left(\|\mathbf A\|_2 R_n \varepsilon_n^{-\frac{1}{p}}\right)$, 但由于每个阶段都能保证 $\Delta_{n+1} \leq \beta \Delta_n$, 因此总的迭代复杂度将是 $\mathcal{O}\left(\frac{\|\mathbf A\|_2}{\sigma_{\min}^+(\mathbf A)} \log\left(\frac{\Delta_0}{\varepsilon}\right)\right)$, 从而得到了额外的收益.



## Non-feasible Case: $\mathbf{A}\mathbf{x} = \mathbf{b}$ 无解

在 Non-feasible Case 中, 由于 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 无解, 因此最优值 $F_{\mathbf b}^\star$ 将不再为 $0$, 而是一个大于 $0$ 的数. 此时, 我们将重新给出更细致的分析. 

首先, 我们可以直接继承如下定义、结论或性质, 因为其证明完全不依赖于 Feasible Case 的假设.

同样, 给定 $\mathbf b\in\mathbb R^m$, 
$$
F_{\mathbf b}(\mathbf x) := \frac1p\|\mathbf A\mathbf x-\mathbf b\|_p^p = \max_{\mathbf y\in\mathbb R^m}\left\{\langle \mathbf A\mathbf x-\mathbf b,\mathbf y\rangle - \frac1q\|\mathbf y\|_q^q\right\}, 
$$
其在给定 $\mathbf x$ 时的最大值解 $\mathbf y_F^\star(\mathbf x)$ 满足
$$
\mathbf y_F^\star(\mathbf x) = \text{sign}(\mathbf A\mathbf x-\mathbf b)\odot|\mathbf A\mathbf x-\mathbf b|^{p-1}.
$$

对应的平滑 surrogate 函数
$$
F_{\mathbf b, \mu}(\mathbf x) := F_\mu = \max_{\mathbf y\in\mathbb R^m}\left\{\langle \mathbf A\mathbf x-\mathbf b,\mathbf y\rangle - \frac1q\|\mathbf y\|_q^q - \frac\mu2\|\mathbf y\|_2^2\right\},
$$

并且其一阶最优性条件, 梯度表达与 Lipschitz Smoothness 的性质与 Feasible Case 完全相同:
$$
\mathbf A\mathbf x-\mathbf b - \nabla h(\mathbf y_\mu^\star(\mathbf{x})) - \mu \mathbf y_\mu^\star(\mathbf x) = \mathbf 0, \quad \nabla F_{\mathbf b, \mu}(\mathbf x) = \mathbf A^\top \mathbf y_\mu^\star(\mathbf x), \quad L_\mu = \frac{\|\mathbf A\|^2}{\mu}.
$$

且同样有 pointwise smoothing bias 的性质, 即对于任意 $\mathbf x\in\mathbb R^n$, 任意 $\mu > 0$, 都有
$$ 
0 \leq F_{\mathbf b}(\mathbf x) - F_{\mathbf b, \mu}(\mathbf x) \leq \frac{\mu}{2} m^{\frac{2-p}{p}} \|\mathbf A\mathbf x-\mathbf b\|_p^{2(p-1)} = D_{p,m} \mu F_{\mathbf b}(\mathbf x)^{2-2/p}.
$$
其中 $D_{p,m} = \frac{1}{2} m^{\frac{2-p}{p}} p^{2-\frac{2}{p}}$ 是一个仅依赖于 $p$ 和 $m$ 的常数.

此外, 继承 feasibility 的 residual space 的有关分析, 记 $\mathcal{R} := \{\mathbf A\mathbf x-\mathbf b: \mathbf x\in\mathbb R^n\} \subset \mathbb{R}^m$,  则原问题等价为
$$
\min_{\mathbf r\in\mathcal{R}} \frac1p\|\mathbf r\|_p^p.
$$

注意到, 此时由于 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 无解, 因此 $\mathbf{0} \notin \mathcal{R}$.  且由于函数 $\mathbf{r} \mapsto \frac1p\|\mathbf r\|_p^p$ 是一个严格凸函数, 因此存在唯一的 $\mathbf{r}^\star \in \mathcal{R}$ 使得 $F_{\mathbf b}^\star = \frac1p\|\mathbf r^\star\|_p^p > 0$. 并且, 对任意满足 $\mathbf A\mathbf {\bar x}-\mathbf b = \mathbf r^\star$ 的 $\mathbf {\bar x}$, 都是原问题的最优解, 故最终的最优解集合 $\mathcal{X}^\star$ 可以表示为:
$$
\mathcal{X}^\star = \{\mathbf x\in\mathbb R^n: \mathbf A\mathbf x-\mathbf b = \mathbf r^\star\} = \mathbf {\bar x} + \text{Null}(\mathbf A).
$$

同理, 对于平滑 surrogate 问题, 
$$
F_{\mathbf b, \mu}^\star = \min_{\mathbf x\in\mathbb R^n} F_{\mathbf b, \mu}(\mathbf x) = \min_{\mathbf r\in\mathcal{R}} \max_{\mathbf y\in\mathbb R^m}\left\{\langle \mathbf r,\mathbf y\rangle - \frac1q\|\mathbf y\|_q^q - \frac\mu2\|\mathbf y\|_2^2\right\}.
$$
其具有严格凸且可微的目标函数, 因此存在唯一的 $\mathbf r_\mu^\star \in \mathcal{R}$ 使得 $F_{\mathbf b, \mu}^\star = \min_{\mathbf x} F_{\mathbf b, \mu}(\mathbf x)$. 对应的最优解集合 $\mathcal{X}_\mu^\star$ 可以表示为 $\{\mathbf x\in\mathbb R^n: \mathbf A\mathbf x-\mathbf b = \mathbf r_\mu^\star\} = \mathbf {\bar x}_\mu + \text{Null}(\mathbf A)$, 其中 $\mathbf {\bar x}_\mu$ 是满足 $\mathbf A\mathbf {\bar x}_\mu-\mathbf b = \mathbf r_\mu^\star$ 的任意点.

一般而言, $\mathbf r_\mu^\star \neq \mathbf r^\star$, $\mathcal{X}_\mu^\star \neq \mathcal{X}^\star$, 原问题与平滑 surrogate 问题的最优值也不相同. 故我们需要对于 Non-feasible Case 的核心误差进行更细致的分解. 对于任意 $\mathbf x\in\mathbb R^n$, 都有
$$
F_{\mathbf b}(\mathbf x) - F_{\mathbf b}^\star = \underbrace{F_{\mathbf b}(\mathbf x) - F_{\mathbf b, \mu}(\mathbf x)}_{\text{pointwise smoothing bias}} + \underbrace{F_{\mathbf b, \mu}(\mathbf x) - F_{\mathbf b, \mu}^\star}_{\text{optimization error}} + \underbrace{F_{\mathbf b, \mu}^\star - F_{\mathbf b}^\star}_{\text{surrogate optimal value error} \leq0}.
$$
其中, $F^\star_{\mathbf b, \mu} - F_{\mathbf b}^\star < 0$ 是由于 $F_{\mathbf b, \mu}(\mathbf x) \leq F_{\mathbf b}(\mathbf x)$ 对任意 $\mathbf x$ 都成立. 因此, 有如下不等式:
$$
F_{\mathbf b}(\mathbf x) - F_{\mathbf b}^\star \leq \bigl(F_{\mathbf b}(\mathbf x) - F_{\mathbf b, \mu}(\mathbf x)\bigr) + \bigl(F_{\mathbf b, \mu}(\mathbf x) - F_{\mathbf b, \mu}^\star\bigr).
$$

该分解说明, 对于 Non-feasible Case 的 affine 模型, 其核心误差主要由如下两部分控制: 优化误差 $F_{\mathbf b, \mu}(\mathbf x) - F_{\mathbf b, \mu}^\star$ 和 surrogate 平滑误差 $F_{\mathbf b}(\mathbf x) - F_{\mathbf b, \mu}(\mathbf x) \leq D_{p,m} \mu F_{\mathbf b}(\mathbf x)^{2-2/p}$.  在 Feasible Case 中, 由于 $F_{\mathbf b}^\star = F_{\mathbf b, \mu}^\star = 0$, 因此只需要保证平滑误差即可. 但在 Non-feasible Case 中, 由于 $F_{\mathbf b}^\star > 0$, 因此我们需要对这两个误差来源进行同时控制, 实现 trade-off 的平衡.

***Proposition* (Non-feasible Case 下的精度转换)**: 对于任意给定精度 $\varepsilon > 0$, 令 
$$
M_\varepsilon := F_{\mathbf b}^\star + \varepsilon.
$$
当 $\mu > 0$ 满足
$$
\mu \leq \frac{\varepsilon}{4 D_{p,m} M_\varepsilon^{2-2/p}} ,
$$
且对于某 $\mathbf x\in\mathbb R^n$, 满足
$$
F_{\mathbf b, \mu}(\mathbf x) - F_{\mathbf b, \mu}^\star \leq \frac{\varepsilon}{2},
$$
则必有
$$
F_{\mathbf b}(\mathbf x) - F_{\mathbf b}^\star \leq \varepsilon.
$$

- *Proof*
  - 用反证法. 若不然, 假设在 $F_{\mathbf b, \mu}(\mathbf x) - F_{\mathbf b, \mu}^\star \leq \frac{\varepsilon}{2}$ 和 $\mu \leq \frac{\varepsilon}{4 D_{p,m} M_\varepsilon^{2-2/p}}$ 的条件下, 仍有 $F_{\mathbf b}(\mathbf x) - F_{\mathbf b}^\star \geq \varepsilon$. 或等价地, 记作 $F_{\mathbf b}(\mathbf x) \geq F_{\mathbf b}^\star + \varepsilon := M_\varepsilon$.
  - 然而, 根据 pointwise smoothing bias 的性质, 若 $F_{\mathbf b}(\mathbf x) \geq F_{\mathbf b}^\star +\varepsilon = M_\varepsilon$, 则
    $$
    F_{\mathbf b, \mu}(\mathbf x) \geq F_{\mathbf b}^\star + \frac{3\varepsilon}{4} .
    $$
      - 这是由于, 根据 pse, $0\leq F_{\mathbf b}(\mathbf x) - F_{\mathbf b, \mu}(\mathbf x) \leq D_{p,m} \mu F_{\mathbf b}(\mathbf x)^{2-2/p}$. 即 $F_{\mathbf b, \mu}(\mathbf x) \geq F_{\mathbf b}(\mathbf x) - D_{p,m} \mu F_{\mathbf b}(\mathbf x)^{2-2/p}$. 为方便起见, $\varphi(t) := t - D_{p,m} \mu t^{2-2/p}$, 则上述的 pse 不等式等价于 $F_{\mathbf b, \mu}(\mathbf x) \geq \varphi(F_{\mathbf b}(\mathbf x))$. 这部分都是已知的事实. 
      - 接下来试图说明, 若 $F_{\mathbf b}(\mathbf x) \geq M_\varepsilon$ (反证法假设), 则可以推出 $\varphi(F_{\mathbf b}(\mathbf x)) \geq \varphi(M_\varepsilon)$. 因为若该不等式成立, 则立刻再根据 pse, 就有 
        $$
        F_{\mathbf b, \mu}(\mathbf x) \stackrel{pse}{\geq} \varphi(F_{\mathbf b}(\mathbf x)) \stackrel{cont}{\geq} \varphi(M_\varepsilon).
        $$
        - 这是因为 $\varphi(t) = t - D_{p,m} \mu t^{2-2/p}$ 是一个关于 $t$ 的函数, 其导数为 $\varphi'(t) = 1 - D_{p,m} \mu (2-2/p) t^{1-2/p}$. 经过计算, 其在所有 $t \geq M_\varepsilon$ 的区间上, 有 $\varphi'(t) > 0$.  因此, 只要 $F_{\mathbf b}(\mathbf x) \geq M_\varepsilon$, 就必然有 $\varphi(F_{\mathbf b}(\mathbf x)) \geq \varphi(M_\varepsilon)$.
        - $\varphi'(t) > 0$ 在 $t \geq M_\varepsilon$ 上成立的具体证明如下. 为方便起见, 进一步记 $\varphi'(t) = 1 - D\mu \alpha t^{\alpha-1}$, 其中 $D = D_{p,m}$, $\alpha = 2 - \frac{2}{p}\in (0,1]$. 因此, 函数 $t \mapsto t^{\alpha-1}$ 是一个单调递减函数. 因此, 对于任意 $t \geq M_\varepsilon$, 都有 $t^{\alpha-1} \leq M_\varepsilon^{\alpha-1}$. 从而, $\varphi'(t) = 1 - D\mu \alpha t^{\alpha-1} \geq 1 - D\mu \alpha M_\varepsilon^{\alpha-1} \geq 1 - \frac{\varepsilon}{4 M_\varepsilon} \alpha > 0$, 对任意 $t \geq M_\varepsilon$ 都成立. 
        
    - 而此时, 经过计算, $\varphi(M_\varepsilon) = M_\varepsilon - D_{p,m} \mu M_\varepsilon^{2-2/p} \geq M_\varepsilon - \frac{\varepsilon}{4} = F_{\mathbf b}^\star + \frac{3\varepsilon}{4}$. 其中不等式是由 $\mu \leq \frac{\varepsilon}{4 D_{p,m} M_\varepsilon^{2-2/p}}$ 得到的. 
      
  - 又因为 $F_{\mathbf b, \mu}^\star \leq F_{\mathbf b}^\star$, 因此
    $$
    F_{\mathbf b, \mu}(\mathbf x) \geq F_{\mathbf b}^\star + \frac{3\varepsilon}{4} \geq F_{\mathbf b, \mu}^\star + \frac{3\varepsilon}{4}.
    $$
  - 这与初始条件 $F_{\mathbf b, \mu}(\mathbf x) - F_{\mathbf b, \mu}^\star \leq \frac{\varepsilon}{2}$ 矛盾. 因此, 必有 $F_{\mathbf b}(\mathbf x) - F_{\mathbf b}^\star \leq \varepsilon$.

$\square$

上述定理给出了 Non-feasible Case 下的精度转换. 其核心思想仍是, 只要我们将 smoothing surrogate 的优化误差 $F_{\mathbf b, \mu}(\mathbf x) - F_{\mathbf b, \mu}^\star$ 控制到一个足够小的水平 (即 $\frac{\varepsilon}{2}$), 同时将 smoothing parameter $\mu$ 设定到一个足够小的水平 (即 $\mu \lesssim \frac{\varepsilon}{(F_{\mathbf b}^\star + \varepsilon)^{\alpha}}$, $\alpha = 2-2/p \in (0,1]$) , 以控制 smoothing bias  , 就可以保证 Non-feasible Case 下的 affine 模型的核心误差 $F_{\mathbf b}(\mathbf x) - F_{\mathbf b}^\star$ 不超过 $\varepsilon$. 这也很好的体现出了整体的 tradeoff 关系. 当 $\mu$ 越大, 平滑的效果就越强 (平滑 Lipschitz constant $L_\mu = \frac{\|\mathbf A\|^2}{\mu}$ 越小) , 此时 surrogate $F_{\mathbf b, \mu}(\mathbf x)$ 本身的优化就越容易控制, 但此时 surrogate 的平滑误差, 即和原问题的最优值之间的 gap 就越大.  反过来 , 当 $\mu$ 越小, surrogate 的平滑误差就越小, 但此时 surrogate 的优化误差就越难以控制. 因此, 只有当我们同时将 surrogate 的优化误差和 surrogate 的平滑误差都控制到一个足够小的水平, 才能保证 Non-feasible Case 下的 affine 模型的核心误差 $F_{\mathbf b}(\mathbf x) - F_{\mathbf b}^\star$ 不超过 $\varepsilon$. 


在给出了 Non-feasible Case 下的精度转换之后, 我们就可以给出 Non-feasible Case 下, 给定 $\mu$ 时的单阶段收敛分析.

***Theorem* (Non-feasible Case 下的单阶段收敛)**: 在 Non-feasible Case 下, 设定 smoothing parameter 
$$
\mu_\varepsilon := \frac{\varepsilon}{4 D_{p,m} (F_{\mathbf b}^\star + \varepsilon)^\alpha}, \quad \alpha = 2 - \frac{2}{p} \in (0,1],
$$
且记
$$
\mathcal{X}_{\mu_\varepsilon}^\star = \arg\min_{\mathbf x\in\mathbb R^n} F_{\mathbf b, \mu_\varepsilon}(\mathbf x) = \{\mathbf x\in\mathbb R^n: \mathbf A\mathbf x-\mathbf b = \mathbf r_{\mu_\varepsilon}^\star\},
$$
以及初始点 $\mathbf x_0 \in \mathbb{R}^n$ 到 surrogate 最优解集合 $\mathcal{X}_{\mu_\varepsilon}^\star$ 的距离
$$
R_{0,\mu_\varepsilon} := \text{dist}(\mathbf x_0, \mathcal{X}_{\mu_\varepsilon}^\star).
$$

若对光滑 surrogate 问题 $\min_{\mathbf x\in\mathbb R^n} F_{\mathbf b, \mu_\varepsilon}(\mathbf x)$ 应用 AGD, 直到某迭代点 $\mathbf x_k$ 满足
$$
F_{\mathbf b, \mu_\varepsilon}(\mathbf x_k) - F_{\mathbf b, \mu_\varepsilon}^\star \leq \frac{\varepsilon}{2},
$$
则必有
$$
F_{\mathbf b}(\mathbf x_k) - F_{\mathbf b}^\star \leq \varepsilon.
$$

并且, 迭代次数 $k$ 的充分条件为
$$
k+1 \geq \frac{2\|\mathbf A\| R_{0,\mu_\varepsilon}}{\sqrt{\mu_\varepsilon \varepsilon}} .
$$

若进一步代入 $\mu_\varepsilon$ 的表达式, 则迭代次数 $k$ 满足
$$
k = \mathcal{O}\left( \|\mathbf A\| R_{0,\mu_\varepsilon}  \frac{(F_{\mathbf b}^\star + \varepsilon)^{1-1/p}}{\varepsilon} \right).
$$

$\diamond$

- *Proof*
  - 对于 $F_{\mathbf b, \mu_\varepsilon}(\mathbf x)$, 其 Lipschitz constant 为 $L_{\mu_\varepsilon} = \frac{\|\mathbf A\|^2}{\mu_\varepsilon}$, 因此由 AGD 可以给出:
    $$
    F_{\mathbf b, \mu_\varepsilon}(\mathbf x_k) - F_{\mathbf b, \mu_\varepsilon}^\star \leq
    \frac{2L_{\mu_\varepsilon} R^2_{0,\mu_\varepsilon}}{(k+1)^2} 
    $$

  - 因此, 只要 $\frac{2L_{\mu_\varepsilon} R^2_{0,\mu_\varepsilon}}{(k+1)^2} \leq \frac{\varepsilon}{2}$, 即
    $$
    k+1  \geq 2 R_{0,\mu_\varepsilon} \sqrt{\frac{L_{\mu_\varepsilon}}{\varepsilon}} = 2 R_{0,\mu_\varepsilon} \|\mathbf A\| \frac{1}{\sqrt{\mu_\varepsilon \varepsilon}},
    $$
    就可以保证 $F_{\mathbf b, \mu_\varepsilon}(\mathbf x_k) - F_{\mathbf b, \mu_\varepsilon}^\star \leq \frac{\varepsilon}{2}$.

  - 由前面的精度转换定理, 可以得到 $F_{\mathbf b}(\mathbf x_k) - F_{\mathbf b}^\star \leq \varepsilon$.

$\square$

对于上述收敛性质, 我们有如下解读. 
- 从上述表达可以看出 Feasible Case 下的收敛率确实为 Non-feasible Case 下的一个特例. 这可以通过令 $F_{\mathbf b}^\star = 0$ 来得到原先的 $\mathcal{O}(\varepsilon^{-1/p})$ 的迭代复杂度. 
- 在 non-feasible, 且 $\varepsilon \to 0$ 的极限情况下, 固定 $F_{\mathbf b}^\star > 0$, 则迭代复杂度
  $$
  (F_{\mathbf b}^\star + \varepsilon)^{1-1/p} \varepsilon^{-1} \asymp (F_{\mathbf b}^\star)^{1-1/p} \varepsilon^{-1},
  $$
  因此此时的迭代复杂度为 $\mathcal{O}(\varepsilon^{-1})$, 这与 Feasible Case 下的 $\mathcal{O}(\varepsilon^{-1/p})$ 的迭代复杂度相比, 是一个更慢的收敛率. 这也很好地体现了, feasible / overpara 的情况下, 给我们带来的不仅只有证明上的方便, 还实际地通过避免了 surrogate optimal value error 的存在, 从而实现了更快的收敛率.

- 在 feasible 的情况下, 我们考虑的一直是 $R_0 = \text{dist}(\mathbf x_0, \mathcal{X}^\star)$ (因为 smoothing surrogate 和原问题的最优解集合是完全重合的), 而在 non-feasible 的情况下, 我们考虑的则是 $R_{0,\mu_\varepsilon} = \text{dist}(\mathbf x_0, \mathcal{X}_{\mu_\varepsilon}^\star)$, 而本身 $\mathcal{X}_{\mu_\varepsilon}^\star$ 也是一个依赖 $\mu$ 的. 因此, 我们目前的分析仍然局限在给定且固定 $\mu$ 的情况下的单阶段收敛分析. 然而, 我们并未对 $\mu$ 的设定进行指导性的建议, 这也是我们下一步的重点分析内容.

- 此外, 上述的分析相当于一个 Oracle 的问题尺度的严格推导. 在实际分析中, 由于 $F_{\mathbf b}^\star$ 是未知的, 因此我们无法直接设定 $\mu_\varepsilon$ 的数值. 一个最基础的 implementation 是用一个已知的上界, 例如 $B_0 := F_{\mathbf b}(\mathbf x_0)$ 来作为 $F_{\mathbf b}^\star$ 的一个估计, 从而进行后续的分析. 但这仍然是一个比较粗糙保守的设定. 因此, 我们下一步的重点分析内容, 将是如何在 Non-feasible Case 下, 通过其他方法来更好进一步得到改进. 



---

下一步行动路线：
1. 在 Oracle 情况下, 给定 $\varepsilon$ 或 budget $K$ 的情况下, 如何设定 $\mu$ 来达到最优的 tradeoff?
   - Oracle  theorem: 给定 $\varepsilon$ 或 budget $K$, 以及假设 $F_{\mathbf b}^\star$ 已知的情况下, 给出 $\mu$ 的设定建议, 从而达到最优的 tradeoff. 明确 $\mu$ 对复杂度的影响作用方式, 并且得到一个最优 的 $\mu$ 的理论标度. 该部分也是后续进阶分析的一个重要 baseline.
   - Implementable corollary: 通过一个保守的上界 $B_0$ 来替代 $F_{\mathbf b}^\star$, 从而得到一个可实施的 $\mu$ 的设定建议. 该部分虽然比较粗糙, 但至少是一个可行的 baseline.

2. Variable Smoothing:
    - 首先是 schedule-based, 该部分是一个预设的直接依赖于 iteration index 的 $\mu$ 的设定, 例如 $\mu_k = \frac{c}{k^2}$ 等. 该模式相对易于分析, 是较方便从 fixed $\mu$ 的单阶段分析过渡到 variable $\mu$ 的多阶段分析的一个重要 stepping stone. 通过该模块的分析, 我么可以给出一个总的关于动态 $\mu$ 的一个理论分析框架, 了解其对于我们整体复杂度讨论的影响作用方式. 该部分的分析也可以为后续的 adaptive $\mu$ 的设定提供一些启发性的指导.
    - 其次是 general adaptive. 此时, 例如总的迭代次数 $K$ 是已知的, 那么我们期望先反推出一个合理的 smoothing scale 以决定误差尺度, 再由此来反推出一个合理的 $\mu$ 的设定建议. 
    - 最后是根据当前的优化轨迹动态变化, 让 $\mu_k$ 以来一个可观测量来进行动态调整. 这就逐渐在向 parameter-free 的 adaptive 设定建议过渡了.

3. Parameter-free 的 adaptation
    - 通过一些 proxy 来代替未知量. 例如其中最核心的一个问题是 $R_{0,\mu}$ 的估计, 因此我们可以通过一些 proxy, 例如 $\|x_0 - x_k\|$ 来进行估计, 从而得到一个 adaptive 的 $\mu$ 的设定建议. 整体而言, 可以先做一个 heuristic algorithm 的设计, 给出一个合理的 $\mu$ 的设定建议, 接着可以再做出一些 theoretical guarantee 来说明该 heuristic algorithm 的有效性. 