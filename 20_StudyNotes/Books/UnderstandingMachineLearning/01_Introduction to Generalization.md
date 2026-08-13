# Section 2: A Gentle Start


>- Book Reference: Understanding Machine Learning: From Theory to Algorithms, Shai Shalev-Shwartz and Shai Ben-David.
>
> - Video Reference : https://www.bilibili.com/video/BV1k64y1r7Dv/?spm_id_from=333.1387.homepage.video_card.click&vd_source=8a00dab0be94d29388f2286892ba8d50


## A Formal Model - The Statistical Learning Framework

***Intuition***

泛化理论在讨论什么？
- 我们期望最小化 true risk.
- 实际上我们只能获得有限的样本，并最小化 training loss (ERM).
  
显然二者存在差距, 并且这个差距并不是永远可以忽略的. 

考虑一个典型的反例:
- Ground truth 是 $f(\mathbf{x})$, 我们有一个训练集 $S = \{(\mathbf{x}_i, f(\mathbf{x}_i))\}_{i=1}^m$.
- 一个失败的模型 $h$ 只能"背住"所有见过的样本, 对于未见过的样本, 它的输出是 0, 即
    $$
    h(\mathbf{x}) = \begin{cases}
        f(\mathbf{x}), & \mathbf{x} \in S \\
        0, & \mathbf{x} \notin S
    \end{cases}
    $$
- 显然, 训练集上的 loss 是 0, 但是 true risk 很大. 它无法泛化到任何未见过的样本.

从直觉上, 认为如下一些因素可能会影响泛化性能:
- 噪声. 若对噪声进行了过多的拟合, 则可能会损害泛化性能 (overfitting).
- 优化. 不同的优化算法可能会导致不同的泛化性能.
- 函数类的限制. 限制函数类可能有助于泛化.
- 数据量. 更多的数据可能有助于泛化.

这些因素都可以通过泛化理论来进行定量分析.

***Statistical Learning Framework***

整体而言, 一个 Learning 的框架如下. 
- 给定 domain set $\mathcal{X} \subset \mathbb{R}^d$ 和 label set $\mathcal{Y} \subset \mathbb{R}$. 我们认为 feature $\mathbf{X}$ 来自某种未知的分布 $\mathcal{D}$, 即 $\mathbf{X} \sim \mathcal{D}$. 此外, 额外假设存在一个 ground truth function $f: \mathcal{X} \to \mathcal{Y}$ 作为某种"oracle", 其能够给定任意的 feature $\mathbf{x} \in \mathcal{X}$, 输出对应的 label $y = f(\mathbf{x})$. 在实际中, 我们能够观测到从这个分布中 i.i.d. 采样得到的训练集 $S = \{(\mathbf{x}_i, y_i)\}_{i=1}^m$. 
- 我们期望 learner 能够学习一种预测规则 (称 predictor / hypothesis / classifier) $h: \mathcal{X} \to \mathcal{Y}$ 能够给定 feature $\mathbf{x}$, 尽可能准确的预测 label $y$. 此外, 引入 **algorithm** $\mathcal{A}$, 其负责从训练集 $S$ 中学习 predictor $h$. 也就是说, $\mathcal{A}$ 是一个 mapping: $\mathcal{A}: S \to h$.

- 以 0-1 二分类问题为例, 我们定义 prediction rule 的 loss function 为:
    $$
    L_{\mathcal{D}, f}(h) := \mathbb{P}_{\mathbf{X} \sim \mathcal{D}}[h(\mathbf{X}) \neq f(\mathbf{X})] := \mathcal{D}(\mathbf{x}: h(\mathbf{x}) \neq f(\mathbf{x}))
    $$
    - 直观地讲, $L_{\mathcal{D}, f}(h)$ 表示 predictor $h$ 在 distribution $\mathcal{D}$ 下的 true risk, 即 predictor $h$ 在所有可能的 feature 上预测错误的概率. 第二个等式相当于以测度的视角进行定义, 其中 $\{\mathbf{x}: h(\mathbf{x}) \neq f(\mathbf{x})\}$ 表示所有预测错误的 feature 的集合 (可以理解为一个事件), 而 $\mathcal{D}(\mathbf{x}: h(\mathbf{x}) \neq f(\mathbf{x}))$ 表示这个事件发生的概率, 即错误集合的测度.
    - 注意, 对于 $L$, 其本质相当于一个从 $\mathcal{H} \to [0, 1]$ 的 mapping, 其中 $\mathcal{H}$ 表示所有可能的 predictor 的集合 (hypothesis space). 这也引出 Learning 中的优化目标:
        $$
        \min_{h \in \mathcal{H}} L_{\mathcal{D}, f}(h)
        $$
        然而, 由于我们无法直接访问 distribution $\mathcal{D}$, 因此无法直接优化 true risk. 于是, 我们引入 empirical risk 等概念. 


## Empirical Risk Minimization (ERM)

给定训练集 $S = \{(\mathbf{x}_i, y_i)\}_{i=1}^m$, 算法 $\mathcal{A}$ 的目标是学习一个 predictor $h_S$ 使得其在训练集上的 loss 尽可能小. 也就是说, 我们希望最小化 empirical risk / empirical error:
$$
L_S(h) := \frac{1}{m} \sum_{i=1}^m \mathbf{1}[h(\mathbf{x}_i) \neq y_i]
$$

然而最小化 ERM 的思路, 尽管直接, 却并不总是能够保证泛化性能. 
- 例如上述背住训练集的反例, 其在训练集上的 empirical risk 是 0, 但是其 almost surely 在 true risk 上得到正确预测. 
-  一个最直观的解释是, hypothesis space $\mathcal{H}$ 太大了, 因为如果我们允许所有可能的 $\mathcal{X} \to \mathcal{Y}$ 的 mapping (也可以简写为 $\mathcal{Y}^{\mathcal{X}}$), 那么我们总是可以找到一个 predictor $h$ 能够在训练集上得到 0 loss. 
-  因此在后续的泛化理论中, 我们会引入一些限制条件来限制 hypothesis space 的大小, 以便能够保证泛化性能.

若从经验测度的角度来看, 
- population risk 为:
    $$
    L_{\mathcal{D}, f}(h) = \int \mathbf{1}[h(\mathbf{x}) \neq f(\mathbf{x})] \mathrm{d}\,\mathcal{D}(\mathbf{x})
    $$
- 而 empirical risk 为:
    $$
    L_S(h) = \int \mathbf{1}[h(\mathbf{x}) \neq f(\mathbf{x})] \mathrm{d}\,\widehat{\mathcal{D}}_S(\mathbf{x})
    $$

    - 其中 $\widehat{\mathcal{D}}_S$ 表示训练集 $S$ 的经验分布, 即 $\widehat{\mathcal{D}}_S = \frac{1}{m} \sum_{i=1}^m \delta_{\mathbf{x}_i}$, 其中 $\delta_{\mathbf{x}_i} = \mathbf{1}[\mathbf{x} = \mathbf{x}_i]$ 表示 Dirac measure. 也就是说, $\widehat{\mathcal{D}}_S$ 是一个离散分布, 其在训练集中的每个样本点上都有相同的概率 $1/m$.


## Empirical Risk  Minimization  with Inductive Bias

因此既然上述的问题是 hypothesis space 太大, 那么我们可以通过限制 hypothesis space 的大小来解决这个问题, 即引入 **inductive bias**. 具体而言, 我们需要先验地 (在看到训练集之前) 给出一个受限的 hypothesis space $\mathcal{H} \subset \mathcal{Y}^{\mathcal{X}}$, 其包含了我们认为可能的 predictor, 然后在这个受限的 hypothesis space 上进行 ERM, 即:
$$
\text{ERM}_{\mathcal{H}}(S) := \arg\min_{h \in \mathcal{H}} L_S(h)
$$

- 这里的 inductive bias 可以理解为我们对问题的先验知识, 包含了我们的某种结构性偏好. 
- 这也从直观上揭示了最基本的一个 tradeoff: 对于 $\mathcal{H}$ 的假设既防止了过拟合, 但又可能阻止模型学习到真实规律. 

### Finite Hypothesis Space

考虑一个最简单的情况, 即假设 hypothesis space 的 cardinality 是有限的, 即 $|\mathcal{H}| < \infty$. 下面将要证明, 对于任意给定的精度 $\epsilon > 0$ 和置信度 $\delta > 0$, 当训练集的大小 $m$ 足够大时, 我们可以保证 ERM 的泛化性能以至少 $1 - \delta$ 的概率达到 $\epsilon$ 精度. 也就是说, 我们可以保证:
$$
\mathbb{P}\left( L_{\mathcal{D}, f}(h_S) - L_S(h_S) \leq \epsilon \right) \geq 1 - \delta
$$
简而言之: 只要样本足够大, 有限类的 ERM 就不会过拟合.  某种意义上, 任何现实的问题, 由于机器精度的限制, 都可以被认为是有限类问题, 不过确实其 cardinality 可能非常大就是. 

首先考虑一个更理想的情况. 

***Realizability Assumption***

假设在 hypothesis space $\mathcal{H}$ 中存在一个 predictor $h^*$ 能够完美拟合 ground truth function $f$, 即:
$$
\exists \, h^\star \in \mathcal{H}, \quad \text{s.t.  } L_{\mathcal{D}, f}(h^\star) = 0
$$
- 换言之, almost surely, 对于任意的 $\mathbf{X} \sim \mathcal{D}$, 我们有 $h^\star(\mathbf{X}) = f(\mathbf{X})$.  可以立刻推得, 对于任意的训练集 $S$, 我们有 $L_S(h^\star) = 0$ 以概率 1 成立.  又因为 $h_S$ 是 ERM 的解, 故有
    $$
    L_S(h_S) \leq L_S(h^\star) = 0
    $$
    即 ERM 的解在训练集上的 loss 也是 0, w.p.1. 

不过这并不足够, 上面的训练误差为 0, not necessarily 代表 true risk 也为 0. 因此我们需要进一步分析 ERM 的解在 true risk 上的表现, 即 $L_{\mathcal{D}, f}(h_S)$. 这里考虑样本是 i.i.d. 的, 即 $S \sim \mathcal{D}^m$. 
- 由于样本是抽样产生的, 因此 $S \xrightarrow{\mathcal{A}} h_S$ 是一个随机过程, 因此 $h_S$ 也是一个随机变量, 因此 $h_S \mapsto L_{\mathcal{D}, f}(h_S)$ 也是一个随机变量. 我们无法保证当前的样本一定能产生一个泛化性能良好的 predictor, 我们只能保证在大概率的意义下, 也就是 w.p.$1 - \delta$, 从 $\mathcal{D}^m$ 中抽样得到的训练集 $S$ 能够产生一个泛化性能良好的 predictor $h_S$. 也就是说, 我们希望保证:
    $$
    \mathbb{P}_{S \sim \mathcal{D}^m}\left( L_{\mathcal{D}, f}(h_S) \leq \epsilon \right) \geq 1 - \delta
    $$

- 或等价地, 确保"有问题的"情况足够少. 若用测度记号可以表示为
    $$
    \mathcal{D}^m\left( \{S|_{\mathbf{x}} : L_{\mathcal{D}, f}(h_S) > \epsilon\} \right)  < \delta
    $$
    - 其中 $S|_{\mathbf{x}}$ 表示训练集 $S$ 中的 feature 部分, 即 $S|_{\mathbf{x}} = \{\mathbf{x}_i\}_{i=1}^m$. 

- 这里额外定义"问题" hypothesis 为 $\mathcal{H}$ 中所有 true risk 大于 $\epsilon$ 的 predictor, 即
    $$
    \mathcal{H}_B := \{h \in \mathcal{H} : L_{\mathcal{D}, f}(h) > \epsilon\}
    $$
    进一步定义, 一个训练集 $S$ 是 misleading 的, 如果确实存在一个问题 hypothesis $h \in \mathcal{H}_B$ 能够在训练集上得到 0 loss, 即
    $$
    M = \{S|_{\mathbf{x}} : \exists h \in \mathcal{H}_B, L_S(h) = 0\} = \bigcup_{h \in \mathcal{H}_B} \{S|_{\mathbf{x}} : L_S(h) = 0\}
    $$
    
- 注意到, 在 realizability 假设下, 所有 generalization failure 的训练集都是 misleading 的, 即
    $$
    \{S|_{\mathbf{x}} : L_{\mathcal{D}, f}(h_S) > \epsilon\} \subseteq M 
    $$
    $$
    \implies \mathcal{D}^m\left( \{S|_{\mathbf{x}} : L_{\mathcal{D}, f}(h_S) > \epsilon\} \right) \leq \mathcal{D}^m(M) \leq \sum_{h \in \mathcal{H}_B} \mathcal{D}^m(\{S|_{\mathbf{x}} : L_S(h) = 0\})
    $$
    - 第一个集合关系是因为在 realizability 假设下, ERM 的解 $h_S$ 会选择那个完美的在训练集上 loss 为 0 的 predictor $h^\star$, 然而这个训练集本身又是不满足泛化误差要求的, 因此这个训练集一定是 misleading 的. 
    - 测度关系是由事实 $\mathcal{D}(A\cup B) \leq \mathcal{D}(A) + \mathcal{D}(B)$ 并代入 $M$ 的定义得到的. 

- 下面继续对 RHS 的每个给定的 $h \in \mathcal{H}_B$ 的 $\mathcal{D}^m(\{S|_{\mathbf{x}} : L_S(h) = 0\})$ 分析其上界. 
  - 对于泛化失败的 hypothesis $h \in \mathcal{H}_B$, 其在训练集上的误差 $L_S(h) = 0$ 等价于 $\forall i \in [m], h(\mathbf{x}_i) = f(\mathbf{x}_i)$, 因此由于数据是 i.i.d. 的, 有
    $$
    \mathcal{D}^m(\{S|_{\mathbf{x}} : L_S(h) = 0\}) = \prod_{i=1}^m \mathcal{D}\left(\left\{\mathbf{x}_i : h(\mathbf{x}_i) = f(\mathbf{x}_i)\right\}\right) 
    $$
  - 对固定 $h \in \mathcal{H}_B$, 其 true risk 大于 $\epsilon$, 即任意预测错误的概率大于 $\epsilon$, 因此反过来预测正确的概率小于 $1 - \epsilon$, 即
    $$
    \mathcal{D}\left(\left\{\mathbf{x}_i : h(\mathbf{x}_i) = f(\mathbf{x}_i)\right\}\right) = 1 - L_{\mathcal{D}, f}(h) \leq 1 - \epsilon
    $$
  - 因此代入上述乘积, 得到
    $$
    \mathcal{D}^m(\{S|_{\mathbf{x}} : L_S(h) = 0\}) \leq (1 - \epsilon)^m \leq \exp(-\epsilon m)
    $$
    - 即一个泛化失败 (真实错误率不低于 $\epsilon$) 的 hypothesis 在训练集上得到 0 loss 的概率不超过 $(1 - \epsilon)^m \leq \exp(-\epsilon m)$. 也就是说, 随着训练集的增大, 这个概率会指数级下降.

- RHS 的每一项都会被 $\exp(-\epsilon m)$ bound 住, 因此综合有
    $$
    \mathcal{D}^m\left( \{S|_{\mathbf{x}} : L_{\mathcal{D}, f}(h_S) > \epsilon\} \right) \leq \sum_{h \in \mathcal{H}_B} \mathcal{D}^m(\{S|_{\mathbf{x}} : L_S(h) = 0\}) \leq |\mathcal{H}_B| \exp(-\epsilon m) \leq |\mathcal{H}| \exp(-\epsilon m)
    $$
    该不等式表明: 
    $$
    \mathrm{Pr} \{\text{ERM 输出真实错误率大于 $\epsilon$ (泛化失败)}\} \leq |\mathcal{H}| \exp(-\epsilon m)
    $$
    - 即泛化失败的概率上界是 hypothesis space 的大小乘以一个关于样本大小与精度的指数衰减项. 当 hypothesis space 越大, 泛化失败的概率越大; 当样本量越大, 泛化失败的概率越小. 这也从数值上揭示了泛化的 tradeoff.

- 根据上面的概率结论
    $$
    \mathcal{D}^m\left( \{S|_{\mathbf{x}} : L_{\mathcal{D}, f}(h_S) > \epsilon\} \right) \leq |\mathcal{H}| \exp(-\epsilon m) < \delta
    $$
    也可以立刻反推对应的样本量下界, 即
    $$
    m \geq \frac{\log(|\mathcal{H}|/\delta)}{\epsilon}
    $$


## Generalization Gap

 当 realization assumption 不成立时, 数据集本身也可能包含噪声, 即给定完全相同的 feature $\mathbf{x}$, 其对应的 label $y$ 仍然可能不同. 则此时, 即使全局最优的 predictor 也无法得到 0 的 true risk, 即 $\inf_{h} L_{\mathcal{D}}(h) > 0$. 

 因此在 ERM 学习中, 我们只考虑相较于最优 predictor 的相对 risk, 即
$$
L_{\mathcal{D}}(h_S) - \inf_{h} L_{\mathcal{D}}(h) = \underbrace{L_{\mathcal{D}}(h_S) - \inf_{h \in \mathcal{H}} L_{\mathcal{D}}(h)}_{\text{excess risk}} + \underbrace{\inf_{h \in \mathcal{H}} L_{\mathcal{D}}(h) - \inf_{h} L_{\mathcal{D}}(h)}_{\text{approximation error}}
$$
- excess risk 表示当前训练集求出的 predictor 相较于 hypothesis space 中最优 predictor 的 true risk 差距 
- approximation error 表示 hypothesis space 本身的设立相较于全局最优 predictor 的 true risk 差距

而 excess risk 可以进一步分解为 generalization gap 与 empirical risk, 即
$$
L_{\mathcal{D}}(h_S) - \inf_{h \in \mathcal{H}} L_{\mathcal{D}}(h) = \underbrace{L_{\mathcal{D}}(h_S) - L_S(h_S)}_{\text{generalization gap}} + \underbrace{L_S(h_S) - L_S(h^\star)}_{\text{empirical risk} \leq 0} + \underbrace{L_S(h^\star) - L_{\mathcal{D}}(h^\star)}_{\text{concentration}}
$$
其中 $h^\star = \arg\min_{h \in \mathcal{H}} L_{\mathcal{D}}(h)$ 是 population 里 $\mathcal{H}$ 中最优的 predictor, 即 $h^\star = \arg\min_{h \in \mathcal{H}} L_{\mathcal{D}}(h)$. 