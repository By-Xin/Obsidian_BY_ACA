# Section 3: A Formal Learning Model

>- Book Reference: Understanding Machine Learning: From Theory to Algorithms, Shai Shalev-Shwartz and Shai Ben-David.

## 3.1 PAC Learning

Probably Approximately Correct (PAC) learning 是泛化理论中一个重要的概念, 用来刻画学习算法在有限样本下, 以高概率输出一个误差较小的 predictor 的能力. 这里的说明以 0-1 二分类问题为例, 不过 PAC learning 的概念可以推广到更一般的学习问题中.

***Definition* (PAC Learning)** 给定一个 hypothesis class $\mathcal{H}$, 称之为 PAC learnable, 若:
- 存在一个固定的学习算法 $\mathcal{A}$ (从样本到 hypothesis 的映射), 以及一个样本复杂度函数 $m_{\mathcal{H}}(\epsilon, \delta): (0, 1)^2 \to \mathbb{N}$ (且要求 $m_{\mathcal{H}}(\epsilon, \delta)$ 也是独立于后文的 $\mathcal{D}$ 和 $f$)
- 使得对于任意的 (定义在输入空间 $\mathcal{X}$ 上的) 分布 $\mathcal{D}$, 任意 error tolerance level $\epsilon \in (0, 1)$ 及 confidence level $\delta \in (0, 1)$, 对任意的 ground truth labeling function $f: \mathcal{X} \to \mathcal{Y} = \{0, 1\}$
- 只要 realizable assumption 成立, 且从 distribution $\mathcal{D}$ 中 i.i.d. 采样得到的训练集 $S = \{(\mathbf{x}_i, f(\mathbf{x}_i))\}_{i=1}^m$ 满足样本量 $m \geq m_{\mathcal{H}}(\epsilon, \delta)$,
  - *Realizability Assumption*: 存在一个 hypothesis $h^* \in \mathcal{H}$, 使得真实误差 $L_{\mathcal{D}, f}(h^*) = 0$. 换言之, 
    $$
    \mathbb{P}_{\mathbf{X} \sim \mathcal{D}}[h^*(\mathbf{X}) \neq f(\mathbf{X})] = 0
    $$
  - 该假设说明 label 中没有不可解释的噪声, 也没有 misspecification.
- 就能够以概率至少 $1 - \delta$ 输出一个 hypothesis $h \in \mathcal{H}$, 使得其真实误差 $L_{\mathcal{D}, f}(h) \leq \epsilon$, 即
    $$
    \mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}, f}(h) \leq \epsilon] \geq 1 - \delta
    $$

关于 PAC learning, 其本质上有两层嵌套的概率, 若全部显式表出, 则为:
$$
\mathbb{P}_{S \sim \mathcal{D}^m_f}\Bigl[\mathbb{P}_{\mathbf{X} \sim \mathcal{D}}[h(\mathbf{X}) \neq f(\mathbf{X})] \leq \epsilon\Bigr] \geq 1 - \delta
$$

上一个 Section中曾经推导过: **任意 finite hypothesis class $\mathcal{H}$ 都是 PAC learnable 的**, 且其样本复杂度为:
$$
m_{\mathcal{H}}(\epsilon, \delta) \leq  \left\lceil\frac{\log(|\mathcal{H}|/\delta)}{\epsilon}\right\rceil
$$

## 3.2 A More General Model

### 3.2.1 Agnostic PAC Learning

上述的 PAC learning 的定义是以 binary classification 为例的, 且假设 realizable assumption 成立. 然而这个假设往往是不成立的, 其可能包含如下误差:
- Irreducible error: 现实中或许并不存在一个 deterministic 的 ground truth labeling function $f$, 也就是说即使给定完全的相同 feature $\mathbf{x}$, 其对应的 label $y$ 也可能是随机的. 
- Misspecification error: 即使真的存在 ground truth labeling function $f$, 但是它可能不在 hypothesis class $\mathcal{H}$ 中.

当不再假设 realizable assumption, 需要对原先的概念进行扩展. 
- 分布由原先的定义在 $\mathcal{X}$ 上的分布 $\mathcal{D}$ 和 ground truth labeling function $f$ 变为在 $\mathcal{X} \times \mathcal{Y}$ 上的 joint distribution $\mathcal{D}$, 在样本上也是 i.i.d. 采样得到的训练集 $S = \{(\mathbf{x}_i, y_i)\}_{i=1}^m$. 
- 此时的 true risk 变为:
    $$
    L_{\mathcal{D}}(h) := \mathbb{P}_{(\mathbf{X}, Y) \sim \mathcal{D}}[h(\mathbf{X}) \neq Y] := \mathcal{D}((\mathbf{x}, y): h(\mathbf{x}) \neq y)
    $$
    
- 此时, 无法再直接要求 $L_{\mathcal{D}}(h) \leq \epsilon$, 因为 $\inf_{h \in \mathcal{H}} L_{\mathcal{D}}(h)$ 可能大于 0, 即当前的 hypothesis class $\mathcal{H}$ 中最好的情况也可能无法达到 $\epsilon$ 的误差要求. 故需要引入 agnostic PAC learning 的概念.

在正式讨论 agnostic PAC learning 之前, 先引入一个概念: **Bayes optimal predictor**. 

***Definition* (Bayes Optimal Predictor)** (以 0-1 二分类问题为例) 给定一个 $\mathcal{X} \times \{0,1\}$ 上的 joint distribution $\mathcal{D}$, 若一个可测函数 $f_\mathcal{D}: \mathcal{X} \to \{0, 1\}$ 满足:
$$
f_\mathcal{D} \in \arg\min_{h: \mathcal{X} \to \{0, 1\}} L_{\mathcal{D}}(h)
$$
其中在 0-1 binary classification 中, $L_{\mathcal{D}}(h) := \mathbb{P}_{(\mathbf{X}, Y) \sim \mathcal{D}}[h(\mathbf{X}) \neq Y]$ 是 predictor $h$ 在 distribution $\mathcal{D}$ 下的 true risk, 则称 $f_\mathcal{D}$ 为 Bayes optimal predictor.

- 这里强调, Bayes optimal predictor 是定义在 population 真实分布 $\mathcal{D}$ 上的, 而不是定义在训练集 $S$ 上的. 此外, 这一 optimality 是针对某一个特定(但可以任意指定)的损失函数而言的, 例如这里是 0-1 loss, 也可以是其他的 loss function, 如 regression 中的 LSE, LAE 等. 
- 一个反例, LASSO regression, 由于考虑了 regularizer, 即使是考虑 population level, 也通常不是 Bayesian optimal 的, 因为其相当于牺牲了一部分 bias 来换取 variance 的降低.

具体在 0-1 二分类问题中, Bayes optimal predictor 可以显式地表示为:
$$
f^{\text{Bayes}}_\mathcal{D}(\mathbf{x}) := \begin{cases}
1, & \text{if } \mathbb{P}[Y = 1 | \mathbf{X} = \mathbf{x}] \geq \frac{1}{2} \\
0, & \text{otherwise}
\end{cases}
$$
- 显然, 当已知 $\mathbb{P}[Y = 1 | \mathbf{X} = \mathbf{x}]$ 时, 若预测 $\hat{y} = 1$, 则预测错误的概率为 $\mathbb{P}[Y = 0 | \mathbf{X} = \mathbf{x}] = 1 - \mathbb{P}[Y = 1 | \mathbf{X} = \mathbf{x}]$, 若预测 $\hat{y} = 0$, 则预测错误的概率为 $\mathbb{P}[Y = 1 | \mathbf{X} = \mathbf{x}]$. 因此, 为了最小化预测错误的概率, 应该选择 $\hat{y} = 1$ 当且仅当 $\mathbb{P}[Y = 1 | \mathbf{X} = \mathbf{x}] \geq 1/2$.
- 本质上, 0-1 loss 的 Bayes optimal predictor 是取条件众数. 



事实上, 对于任意的 predictor $h: \mathcal{X} \to \{0, 1\}$,  Bayes optimal predictor 都是最优的, 即:
$$
L_{\mathcal{D}}(f^{\text{Bayes}}_\mathcal{D}) \leq L_{\mathcal{D}}(h)
$$

*Proof*. 简记 $p(\mathbf{x}) := \mathbb{P}[Y = 1 | \mathbf{X} = \mathbf{x}]$, 则对于任意 predictor $h: \mathcal{X} \to \{0, 1\}$, 其 true risk 为:
$$
L_{\mathcal{D}}(h) = \mathbb{P}_{(\mathbf{X}, Y) \sim \mathcal{D}}[h(\mathbf{X}) \neq Y] = \mathbb{E}_{\mathbf{X} \sim \mathcal{D}_\mathcal{X}}[\mathbb{P}[h(\mathbf{X}) \neq Y | \mathbf{X}]]
$$
现固定任意 $\mathbf{x} \in \mathcal{X}$. 

对于任意 predictor $h$, 其在 $\mathbf{x}$ 上的预测错误概率为:
- 若 $h(\mathbf{x}) = 1$, 则 $\mathbb{P}[h(\mathbf{x}) \neq Y | \mathbf{X} = \mathbf{x}] = \mathbb{P}[Y = 0 | \mathbf{X} = \mathbf{x}] = 1 - p(\mathbf{x})$.
- 若 $h(\mathbf{x}) = 0$, 则 $\mathbb{P}[h(\mathbf{x}) \neq Y | \mathbf{X} = \mathbf{x}] = \mathbb{P}[Y = 1 | \mathbf{X} = \mathbf{x}] = p(\mathbf{x})$.

因此对于给定的 $\mathbf{x}$, arbitrary predictor $h$ 的预测错误概率为 $p(\mathbf{x})$ 或 $1 - p(\mathbf{x})$.

对于 Bayes optimal predictor, 
- 当 $p(\mathbf{x}) \geq 1/2$ 时, $f^{\text{Bayes}}_\mathcal{D}(\mathbf{x}) = 1$, 此时预测错误概率为 $1 - p(\mathbf{x})$
- 当 $p(\mathbf{x}) < 1/2$ 时, $f^{\text{Bayes}}_\mathcal{D}(\mathbf{x}) = 0$, 此时预测错误概率为 $p(\mathbf{x})$

相当于, Bayes optimal predictor 总是选择在 $\mathbf{x}$ 上预测错误概率较小的 label, 即 
$$
\mathbb{P}[f^{\text{Bayes}}_\mathcal{D}(\mathbf{x}) \neq Y | \mathbf{X} = \mathbf{x}] = \min(p(\mathbf{x}), 1 - p(\mathbf{x}))
$$

因此 pointwisely, 对于任意 $\mathbf{x} \in \mathcal{X}$, 都有
$$
\mathbb{P}[f^{\text{Bayes}}_\mathcal{D}(\mathbf{x}) \neq Y | \mathbf{X} = \mathbf{x}] \leq \mathbb{P}[h(\mathbf{x}) \neq Y | \mathbf{X} = \mathbf{x}]
$$

故对 $\mathbf{X} \sim \mathcal{D}_\mathcal{X}$, 取期望后, 有
$$
L_{\mathcal{D}}(f^{\text{Bayes}}_\mathcal{D}) = \mathbb{E}_{\mathbf{X} \sim \mathcal{D}_\mathcal{X}}[\mathbb{P}[f^{\text{Bayes}}_\mathcal{D}(\mathbf{X}) \neq Y | \mathbf{X}]] \leq \mathbb{E}_{\mathbf{X} \sim \mathcal{D}_\mathcal{X}}[\mathbb{P}[h(\mathbf{X}) \neq Y | \mathbf{X}]] = L_{\mathcal{D}}(h)
$$

$\square$

Bayes optimal predictor 说明, 即使完全知道了整个数据的分布 $\mathcal{D}$, 在一般情况下也无法得到 0 的 true risk, 我们最小的 acheivable true risk 是 $L_{\mathcal{D}}(f^{\text{Bayes}}_\mathcal{D})$, 这是一个 oracle benchmark, 往往也称之为 *Bayes error*.

不过在实作当中, Bayes error 也是一个过强的下界, 因此往往只要求算法能够达到给定 hypothesis class $\mathcal{H}$ 中最小的 true risk, 即 $\inf_{h \in \mathcal{H}} L_{\mathcal{D}}(h)$, 故有层级关系:
$$
L_{\mathcal{D}}(f^{\text{Bayes}}_\mathcal{D}) \leq \inf_{h \in \mathcal{H}} L_{\mathcal{D}}(h)
$$
- 如果 $\mathcal{H}$ 是一个非常丰富的 hypothesis class, 则 $\inf_{h \in \mathcal{H}} L_{\mathcal{D}}(h)$ 可能接近 $L_{\mathcal{D}}(f^{\text{Bayes}}_\mathcal{D})$, 反之则不然. 

***Definition* (Agnostic PAC Learning)** 给定一个 hypothesis class $\mathcal{H}$, 称之为 agnostic PAC learnable, 若:
- 存在一个固定的学习算法 $\mathcal{A}$, 一个样本复杂度函数 $m_{\mathcal{H}}(\epsilon, \delta): (0, 1)^2 \to \mathbb{N}$ 
- 对任意定义在 $\mathcal{X} \times \mathcal{Y}$ 上的 joint distribution $\mathcal{D}$, 及任意 $\epsilon, \delta \in (0, 1)$, 
- 只要从 distribution $\mathcal{D}$ 中 i.i.d. 采样得到的训练集 $S = \{(\mathbf{x}_i, y_i)\}_{i=1}^m$ 满足样本量 $m \geq m_{\mathcal{H}}(\epsilon, \delta)$,
- 就能够以概率至少 $1 - \delta$ 输出一个 hypothesis $h$, 使得
    $$
    \mathbb{P}_{S \sim \mathcal{D}^m}[L_{\mathcal{D}}(h) \leq \min_{h' \in \mathcal{H}} L_{\mathcal{D}}(h') + \epsilon] \geq 1 - \delta
    $$

$\diamond$

说明:

- 由上可见, agnostic PAC learning 是对 PAC learning 的一个推广, 其控制的是输出结果与 $\mathcal{H}$ 内最优 predictor 的差距. 因此从学习的角度, 我们不再要求学习的效果是 arbitrarily small 的, 因为有可能本身 $\mathcal{H}$ 的表现就不够好. 
- 此外, 上述定义中事实上并没有限制算法输出的 $h$ 必须在 $\mathcal{H}$ 中, 也就是说算法可以输出一个不在 $\mathcal{H}$ 中的 hypothesis (称为 improper learning). 只不过一切的比较都是以 $\mathcal{H}$ 中最优的 predictor 为 benchmark. 

### 3.2.2 The Scope of Learning Problems Modeled

事实上, PAC learning 和 agnostic PAC learning 的定义是非常 general 的, 其可以推广到更一般的学习问题中. $L_{\mathcal{D}}(h)$ 不一定是错误概率, 其可以是任意适当的 loss function, 甚至是不必要是 supervised learning 的问题. 

抽象意义上, 给定 Hypothesis class $\mathcal{H}$, 以及某个 observation 集合 $\mathcal{Z}$, 就可以定义一个 loss function $\ell: \mathcal{H} \times \mathcal{Z} \to \mathbb{R}_+$.


- 在一般的 supervised learning 中, $\mathcal{Z} = \mathcal{X} \times \mathcal{Y}$, 故 loss 可以定义为分类的
    $$
    \ell(h, (\mathbf{x}, y)) := \mathbf{1}[h(\mathbf{x}) \neq y]
    $$ 
    或 OLS 的 
    $$
    \ell(h, (\mathbf{x}, y)) := (h(\mathbf{x}) - y)^2
    $$ 
- 在 unsupervised learning 中也可以定义 loss. 例如, 在 k-means clustering 中, $\mathcal{Z} = \mathcal{X}$, 样本只有 feature $S = \{\mathbf{x}_i\}_{i=1}^m \subset \mathcal{X}$, 而 loss function 可以定义为:
    $$
    \ell(h, \mathbf{x}) := \min_{j=1, \ldots, k} \|\mathbf{x} - {h}_j\|^2
    $$
    其中 $\mathbf{h} = (h_1, \ldots, h_k)$ 是 k 个 cluster centers. 并且注意, 即使是对于 unsupervised learning, 我们也是可以讨论其泛化性能的: 在训练集上 cluster 很好的, 并不意味着在 population 上也能 cluster 得很好. 因此, 只要定义了 loss function, 就可以讨论泛化性能.

故在给定 $\ell$ 后, 可以统一定义 true risk:
$$
L_{\mathcal{D}}(h) := \mathbb{E}_{Z \sim \mathcal{D}}[\ell(h, Z)]
$$ 

对应 empirical risk:
$$
L_S(h) := \frac{1}{m} \sum_{i=1}^m \ell(h, z_i)
$$

并且上述的 PAC learning 和 agnostic PAC learning 的定义也可以直接推广到更一般的 loss function 上.

