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


> Throughout this section, we fix $\mathbf b\in\mathbb R^m$, and for notational simplicity write
> $$
> F(\mathbf x) := F_{\mathbf b}(\mathbf x), \quad F_\mu(\mathbf x) := F_{\mathbf b, \mu}(\mathbf x).
> $$
> When necessary, we revert to the full notation $F_{\mathbf b}(\mathbf x)$ and $F_{\mathbf b, \mu}(\mathbf x)$ to distinguish from the canonical model.

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


***Proposition* (Pointwise Smoothing Error)**: 对于任意 $\mathbf x\in\mathbb R^n$, 任意 $\mu > 0$, 都有
$$
0 \leq F_{\mathbf b}(\mathbf x) - F_{\mathbf b, \mu}(\mathbf x) \leq \frac{\mu}{2} m^{\frac{2-p}{p}} \|\mathbf A\mathbf x-\mathbf b\|_p^{2(p-1)} = D_{p,m} \mu F_{\mathbf b}(\mathbf x)^{\frac{2(p-1)}{p}}.
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