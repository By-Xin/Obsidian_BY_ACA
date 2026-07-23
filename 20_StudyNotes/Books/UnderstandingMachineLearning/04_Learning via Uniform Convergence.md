#  Section 4: Learning via Uniform Convergence

>- Book Reference: Understanding Machine Learning: From Theory to Algorithms, Shai Shalev-Shwartz and Shai Ben-David.


这一章讲考虑一个一般的 loss function, 只在限制其取值有界的条件下, 试图用 agnostic PAC learning 的概念来刻画学习算法的泛化能力. 这里要借助的主要工具为 **uniform convergence**, 其本质上是统计学的 concentration inequality.


## 4.1 Uniform Convergence Is Sufficient for Agnostic PAC Learning

给定一个 hypothesis class $\mathcal{H}$, 以及 example space $\mathcal{Z}$, 定义 loss function $\ell: \mathcal{H} \times \mathcal{Z} \to \mathbb{R}_+$. 

回忆 ERM. 给定 training set $S = \{z_1, z_2, \ldots, z_m\} \in \mathcal{Z}^m$, 对于每个 $h \in \mathcal{H}$, 定义其 empirical risk 为:
$$
L_S(h) := \frac{1}{m} \sum_{i=1}^m \ell(h, z_i)
$$
ERM 选择 empirical risk 最小的 hypothesis:
$$
h_S \in \arg\min_{h \in \mathcal{H}} L_S(h)
$$

我们期待 ERM 能够输出一个泛化误差较小的 hypothesis, 即希望其 true risk $L_{\mathcal{D}}(h_S)$ 也较小. 


***Definition* ($\epsilon$-Representative Sample)** 给定 training set $S$, 称其是关于 $\mathcal{H}, \mathcal{Z}, \ell, \mathcal{D}$ 的 $\epsilon$-representative sample, 若对于任意的 $h \in \mathcal{H}$, 都有:
$$
|L_S(h) - L_{\mathcal{D}}(h)| \leq \epsilon
$$
$\diamond$

- 由于在 $S$ 得到之前, $L_S(h)$ 是一个随机变量, 因此只有保证 $\mathcal{H}$ 中的任意 hypothesis 的 empirical risk 都接近其 true risk, 才能保证 ERM 输出的 hypothesis 的 true risk 是较小的. 

***Lemma* (ERM on $\epsilon/2$-Representative Sample)** 若训练集 $S$ 是 $\epsilon/2$-representative sample, 则任意的 ERM hypothesis $h_S \in \arg\min_{h \in \mathcal{H}} L_S(h)$ 都满足:
$$
L_{\mathcal{D}}(h_S) \leq \min_{h \in \mathcal{H}} L_{\mathcal{D}}(h) + \epsilon
$$

*Proof*. 

- 记 $h^* := \arg\min_{h \in \mathcal{H}} L_{\mathcal{D}}(h)$ 为在 $\mathcal{H}$ 中 true risk 最小的 hypothesis. 
- 由于 $S$ 是 $\epsilon/2$-representative sample, 对于任意的 $h \in \mathcal{H}$, 都有:
    $$
    |L_S(h) - L_{\mathcal{D}}(h)| \leq \frac{\epsilon}{2} 
    $$
- 由 $h$ 的任意性, 对 ERM 的 $h_S$ 也成立 $^{(1)}$, 且 $h_S$ 是 $L_S(h)$ 的最小值 $^{(2)}$, 故有:
    $$
    L_{\mathcal{D}}(h_S) \stackrel{(1)}{\leq} L_S(h_S) + \frac{\epsilon}{2}\stackrel{(2)}{\leq} L_S(h^*) + \frac{\epsilon}{2}
    $$
- 另一方面其对 $h^*$ 也成立, 故有:
    $$
    L_S(h^*) \leq L_{\mathcal{D}}(h^*) + \frac{\epsilon}{2}
    $$
    代入上式, 得到:
    $$
    L_{\mathcal{D}}(h_S) \leq L_{\mathcal{D}}(h^*) + \epsilon = \min_{h \in \mathcal{H}} L_{\mathcal{D}}(h) + \epsilon.
    $$

$\square$

上述 lemma 保证, 只要训练集是 $\epsilon/2$-representative sample, 则 ERM 输出的 excees risk 就不会超过 $\epsilon$. 下面的 uniform convergence 的概念则刻画了以何种概率, 训练集是 $\epsilon$-representative sample 的.

***Definition* (Uniform Convergence)** 称 hypothesis class $\mathcal{H}$ 是关于 $\mathcal{Z}, \ell$ 的 uniform convergence, 若
- 存在一个函数 $m^{\text{UC}}_{\mathcal{H}}: (0, 1)^2 \to \mathbb{N}$, 
- 对任意的 $\epsilon, \delta \in (0, 1)$, 及任意 $\mathcal{Z}$ 上的 distribution $\mathcal{D}$, 
- 只要从 $\mathcal{D}$ 中 i.i.d. 采样得到的训练集 $S = \{z_1, z_2, \ldots, z_m\}$ 满足样本量 $m \geq m^{\text{UC}}_{\mathcal{H}}(\epsilon, \delta)$,
- 就有 $S$ 至少以概率 $1 - \delta$ 是 $\epsilon$-representative sample, 即
    $$
    \mathbb{P}_{S \sim \mathcal{D}^m}\Bigl[\forall h \in \mathcal{H}, ~~|L_S(h) - L_{\mathcal{D}}(h)| \leq \epsilon\Bigr] \geq 1 - \delta
    $$

$\diamond$

说明:
- uniform 是指: 对于同一个训练集 $S$, 要同时保证所有的 hypothesis $h \in \mathcal{H}$ 都满足 $|L_S(h) - L_{\mathcal{D}}(h)| \leq \epsilon$, 或等价地
    $$
    \sup_{h \in \mathcal{H}} |L_S(h) - L_{\mathcal{D}}(h)| \leq \epsilon
    $$

- 立刻可以得到一个结论: 如果 hypothesis class $\mathcal{H}$ 是关于 $\mathcal{Z}, \ell$ 的 uniform convergence, 则 $\mathcal{H}$ 是 agnostic PAC learnable, 且 ERM 是一个 agnostic PAC learning algorithm. 其样本复杂度为:
    $$
    m_{\mathcal{H}}(\epsilon, \delta) \leq m^{\text{UC}}_{\mathcal{H}}(\epsilon/2, \delta)
    $$
    - 这说明, $m_\mathcal{H}^{\text{UC}}(\epsilon, \delta)$ 是一个 sufficient 的上界, 而实际中往往需要样本量更少. 

总的而言, 这一小节建立的推导关系为:
$$
\text{Sufficiently large sample } (m^\text{UC}_\mathcal{H}) \\ 
\implies S \text{ is } \frac{\epsilon}{2}\text{-representative w.h.p.}\\
 \implies \text{ERM excess risk} \leq \epsilon
$$

## 4.2 Finite Classes Are Agnostic PAC Learnable

本节的核心结论: **任意有限的 hypothesis class $\mathcal{H}$ 都是 agnostic PAC learnable**, 而根据上面的推论, 只需证明 $\mathcal{H}$ 是关于 $\mathcal{Z}, \ell$ 的 uniform convergence 即可. 


在上述命题的证明过程中, 需要用到 Hoeffding's Inequality, 其又依赖于 Hoeffding's Lemma. 下面首先给出两命题的叙述和证明.

***Lemma* (Hoeffding's Lemma)** 设 $X\in [a, b]$ 是一个随机变量, 且 $\mathbb{E}[X] = 0$, 则对任意 $\lambda > 0$, 有
$$
\mathbb{E}[\exp(\lambda X)] \leq \exp\left(\frac{\lambda^2(b - a)^2}{8}\right)
$$
*Proof*. 

- 根据指数函数 $x \mapsto \exp(\lambda x)$ 的凸性, 对任意 $x \in [a, b]$, 有
    $$
    \exp(\lambda x) \leq \alpha \exp(\lambda a) + (1 - \alpha) \exp(\lambda b)
    $$
    令 $\alpha := \frac{b - x}{b - a} \in [0, 1]$, 则
    $$
    \exp(\lambda x) \leq \frac{b - x}{b - a} \exp(\lambda a) + \frac{x - a}{b - a} \exp(\lambda b)
    $$
- 左右两边取期望, 得到
    $$
    \begin{aligned}
    \mathbb{E}[\exp(\lambda X)] &\leq \frac{b - \mathbb{E}[X]}{b - a} \exp(\lambda a) + \frac{\mathbb{E}[X] - a}{b - a} \exp(\lambda b) \\
    &= \frac{b}{b - a} \exp(\lambda a) + \frac{-a}{b - a} \exp(\lambda b)\\
    &= \exp(-hp) \left(1 - p + p\exp(h)\right) \\& := \exp(L(h))    
    \end{aligned}
    $$
    - 其中, $h := \lambda(b - a)$, $p := \frac{-a}{b - a} \in [0, 1]$, 且
        $$
        L(h) := -hp + \log(1 - p + p\exp(h))
        $$

- 故原命题等价于证明 $L(h) \leq h^2/8$. 由于 $L(0) = 0$, 且 $L'(0) = 0$, 且经过计算可以得到 $L''(h) \leq 1/4$, 故由 Taylor expansion 可得
    $$
    L(h) = L(0) + L'(0)h + \frac{1}{2}L''(\xi)h^2 \leq \frac{1}{8}h^2
    $$

$\square$

***Lemma* (Hoeffding's Inequality)** 设 $Z_1, Z_2, \ldots, Z_m$ 是 i.i.d. 的随机变量, 且 almost surely $Z_i \in [a, b]$, $\mu := \mathbb{E}[Z   _i], \forall i$. 则对任意 $\epsilon > 0$, 有
$$
\mathbb{P}\left[\left|\frac{1}{m} \sum_{i=1}^m Z_i - \mu\right| > \epsilon\right] \leq 2\exp\left(-\frac{2m\epsilon^2}{(b - a)^2}\right)
$$


*Proof*

- 首先不妨中心化 $X_i := Z_i - \mu$, 则 $X_i \in [a - \mu, b - \mu]$, 且 $\mathbb{E}[X_i] = 0$. 此时只需考虑 $\mathbb{P}\left[\bar{X} > \epsilon\right]$, 其中 $\bar{X} := \frac{1}{m} \sum_{i=1}^m X_i$.

- 考虑右尾 $\mathbb{P}[\bar{X} \geq \epsilon]$. 其概率上由 Markov inequality 可得
    $$
    \mathbb{P}[\bar{X} \geq \epsilon] = \mathbb{P}[\exp(\lambda \bar{X}) \geq \exp(\lambda \epsilon)] \leq {\mathbb{E}[\exp(\lambda \bar{X})]}{\exp(-\lambda \epsilon)}
    $$

    对于 $\mathbb{E}[\exp(\lambda \bar{X})]$, 由于 $X_i$ 是 i.i.d. $^{(1)}$, 对于任意 $\lambda > 0$, 有
    $$
    \mathbb{E}[\exp(\lambda \bar{X})] = \mathbb{E}\left[\exp\left(\frac{\lambda}{m} \sum_{i=1}^m X_i\right)\right] \stackrel{(1)}{=} \prod_{i=1}^m \mathbb{E}\left[\exp\left(\frac{\lambda}{m} X_i\right)\right]
    $$
    由 Hoeffding's Lemma, 对任意 $i$, 有
    $$
    \mathbb{E}\left[\exp\left(\frac{\lambda}{m} X_i\right)\right] \leq \exp\left(\frac{\lambda^2(b - a)^2}{8m^2}\right)
    $$
    因此
    $$
    \mathbb{E}[\exp(\lambda \bar{X})] \leq \exp\left(\frac{\lambda^2(b - a)^2}{8m}\right)
    $$
    故代回 Markov inequality, 得到
    $$
    \mathbb{P}[\bar{X} \geq \epsilon] \leq \exp\left(\frac{\lambda^2(b - a)^2}{8m} - \lambda \epsilon\right)
    $$
    由于上式对任意 $\lambda > 0$ 都成立, 因此可以对 $\lambda$ 进行优化, 得到指数部分关于 $\lambda$ 的最小值为 $\lambda^* = \frac{4m\epsilon}{(b - a)^2}$, 此时
    $$
    \mathbb{P}[\bar{X} \geq \epsilon] \leq \exp\left(-\frac{2m\epsilon^2}{(b - a)^2}\right)
    $$
- 对于左尾 $\mathbb{P}[\bar{X} \leq -\epsilon]$,  由对称性, 也有
    $$
    \mathbb{P}[\bar{X} \leq -\epsilon] \leq \exp\left(-\frac{2m\epsilon^2}{(b - a)^2}\right)
    $$

- 综上, 有
    $$
    \mathbb{P}\left[\left|\frac{1}{m} \sum_{i=1}^m Z_i - \mu\right| > \epsilon\right] = \mathbb{P}[\bar{X} > \epsilon] + \mathbb{P}[\bar{X} < -\epsilon] \leq 2\exp\left(-\frac{2m\epsilon^2}{(b - a)^2}\right)
    $$

$\square$


下正式给出并证明有限 hypothesis class 的 uniform convergence 性质. 首先说明有限 hypothesis class 是 uniform convergence 的. 

***Proposition 1* (Finite Classes are Uniform Convergence)** 设 $\mathcal{H}$ 是一个有限的 hypothesis class, 且 loss function $\ell: \mathcal{H} \times \mathcal{Z} \to [0, 1]$. 对于任意定义在 $\mathcal{Z}$ 上的 distribution $\mathcal{D}$, 以及任意 $\epsilon, \delta \in (0, 1)$, 若训练集 $S = \{z_1, z_2, \ldots, z_m\}$ 是从 $\mathcal{D}$ 中 i.i.d. 采样得到的, 且 sample complexity 满足
$$
m^\text{UC}_\mathcal{H}(\epsilon, \delta) \leq \left\lceil \frac{\log(2|\mathcal{H}|/\delta)}{2\epsilon^2} \right\rceil
$$
即只要样本量 $m$ 满足 $m \geq \left\lceil \frac{\log(2|\mathcal{H}|/\delta)}{2\epsilon^2} \right\rceil$, 则有
$$
\mathbb{P}_{S \sim \mathcal{D}^m}\Bigl[\forall h \in \mathcal{H}, ~~|L_S(h) - L_{\mathcal{D}}(h)| \leq \epsilon\Bigr] \geq 1 - \delta  \tag{1}
$$

$\diamond$

*Proof*. 证明 (1) 只需证明其补集的概率不超过 $\delta$, 即存在超过 $\epsilon$ 的 hypothesis 的概率不超过 $\delta$.

- 根据集合性质: 存在性可以转换为事件的并集, 因此有:
    $$
    \begin{aligned}
    \mathbb{P}_{S \sim \mathcal{D}^m}\Bigl[\exists h \in \mathcal{H}, ~~|L_S(h) - L_{\mathcal{D}}(h)| > \epsilon\Bigr] & = \mathbb{P}_{S \sim \mathcal{D}^m}\Bigl[\bigcup_{h \in \mathcal{H}} \{|L_S(h) - L_{\mathcal{D}}(h)| > \epsilon\}\Bigr] \\
    & \leq \sum_{h \in \mathcal{H}} \mathbb{P}_{S \sim \mathcal{D}^m}\Bigl[|L_S(h) - L_{\mathcal{D}}(h)| > \epsilon\Bigr] \qquad \text{\small{(by union bound)}} \\
    \end{aligned}
    $$

- 故对于每个给定 $h \in \mathcal{H}$, 需控制其概率 $\mathbb{P}_{S \sim \mathcal{D}^m}\Bigl[|L_S(h) - L_{\mathcal{D}}(h)| > \epsilon\Bigr]$, 对其使用 Hoeffding's Inequality 如下. 
    - 对于每个给定 $h \in \mathcal{H}$, 记 $\theta_i := \ell(h, z_i)$ 是 i.i.d. 的随机变量, $i \in [m]$, 且 $\theta_i \in [0, 1]$, 其期望为:
        $$
        \mathbb{E}[\theta_i] = \mathbb{E}_{Z_i \sim \mathcal{D}}[\ell(h, Z_i)] = L_{\mathcal{D}}(h)
        $$
        而样本均值为:
        $$
        L_S(h) = \frac{1}{m} \sum_{i=1}^m \theta_i
        $$
    - 由 Hoeffding's Inequality, 对任意 $\epsilon > 0$, 有
        $$
        \mathbb{P}_{S \sim \mathcal{D}^m}\Bigl[|L_S(h) - L_{\mathcal{D}}(h)| > \epsilon\Bigr] = \mathbb{P}\left[\left|\frac{1}{m} \sum_{i=1}^m \theta_i - L_{\mathcal{D}}(h)\right| > \epsilon\right] \leq 2\exp(-2m\epsilon^2)
        $$

- 将 Hoeffding's Inequality 的结果代入 union bound, 便给出了一个具体的犯错概率上界:
    $$
    \mathbb{P}_{S \sim \mathcal{D}^m}\Bigl[\exists h \in \mathcal{H}, ~~|L_S(h) - L_{\mathcal{D}}(h)| > \epsilon\Bigr] \leq 2|\mathcal{H}|\exp(-2m\epsilon^2)
    $$

- 因此, 当样本量 $m$ 足够大时, 上述概率可以被控制在 $\delta$ 以内: 令上述犯错概率上界不超过 $\delta$, 即
    $$
    2|\mathcal{H}|\exp(-2m\epsilon^2) \leq \delta \implies m \geq \frac{\log(2|\mathcal{H}|/\delta)}{2\epsilon^2}
    $$
    故只要样本量满足
    $$
    m^\text{UC}_\mathcal{H}(\epsilon, \delta) \leq \left\lceil \frac{\log(2|\mathcal{H}|/\delta)}{2\epsilon^2} \right\rceil
    $$
    则有
    $$
    \mathbb{P}_{S \sim \mathcal{D}^m}\Bigl[\forall h \in \mathcal{H}, ~~|L_S(h) - L_{\mathcal{D}}(h)| \leq \epsilon\Bigr] \geq 1 - \delta
    $$

$\square$

**Notes**:
- 上述的证明依赖于假设: 损失函数 $\ell$ 的取值范围为 $[0, 1]$. 其可以推广到任意有界的损失函数, 然而若对于 MSE 等无约束的损失函数则不可以直接应用.


***Proposition 2* (Finite Classes are Agnostic PAC Learnable)** 对于上述的有限 hypothesis class $\mathcal{H}$, 只要样本量 $m$ 满足
$$
m_\mathcal{H}(\epsilon, \delta) \leq \left\lceil \frac{2\log(2|\mathcal{H}|/\delta)}{\epsilon^2} \right\rceil
$$
则有 $\mathcal{H}$ 是 agnostic PAC learnable, 即任意 ERM 输出 $h_S \in \arg\min_{h \in \mathcal{H}} L_S(h)$ 都以至少概率 $1 - \delta$ 满足
$$
L_{\mathcal{D}}(h_S) \leq \min_{h \in \mathcal{H}} L_{\mathcal{D}}(h) + \epsilon.
$$

$\diamond$
