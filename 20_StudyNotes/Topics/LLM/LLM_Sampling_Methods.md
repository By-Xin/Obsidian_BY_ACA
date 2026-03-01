---
aliases: [LLM Sampling, 采样方法, Decoding Strategies, Top-k, Top-p, Nucleus Sampling]
tags:
  - concept
  - ml/deep-learning
  - cs/nlp
related_concepts:
  - "[[Temperature Sampling]]"
  - "[[Beam Search]]"
  - "[[LanguageModels]]"
  - "[[Uncertainty_in_LLMs]]"
  - "[[Deep_Reasoning_for_LLMs]]"
  - "[[Softmax]]"
  - "[[Entropy]]"
---

# LLM Sampling 方法综述

## 1. 引言

本综述旨在对当前主流的采样方法进行整理和概述，主要关注大型语言模型（LLMs）中的文本生成解码策略。当前解码方法主要分为确定性采样和随机性采样两大类。

*   **确定性采样 (Deterministic Decoding)**:
    *   Greedy Search
    *   Beam Search
*   **随机性采样 (Stochastic Decoding)**:
    *   Top-k Sampling
    *   Top-p (Nucleus) Sampling
    *   Temperature Sampling

以下将详细梳理多篇关于采样方法的文献。

## 2. 核心采样方法梳理

### 2.1 Gambel-Top-k*

*   **论文标题**: [Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement](https://www.arxiv.org/abs/1903.06059)
*   **作者**: Wouter Kool, Herke van Hoof, Max Welling
*   **年份**: 2019 (1903)

#### 1. Motivation

*   **Beam Search**: 确定性解码，能够生成高联合概率的序列但多样性差。
*   **Gumbel-Max Trick:**
    *   把 Gumbel 噪声视为为每个类别“掷骰子”的不确定性扰动。Gumbel 的尾部偏胖（支持极大值理论），保证 argmax 的选择概率正比于原始 softmax 概率。
    *   定义 Gumbel 分布为: 若 $U\sim \mathcal U(0,1)$, 则 $G = -\log(-\log U) \sim \text{Gumbel}(0)$. 进一步定义带位置的 Gumbel 为 $G+\phi\sim \text{Gumbel} (\phi)$.
    *   给定离散类别集合 $i \in \{1, \dots, n\}$, 以及通过 softmax 的 logtis $p_i = \frac{\exp(\phi_i)}{\sum_j \exp(\phi_j)}$. 对于每个 $i$, 采样独立噪声 $G_i\sim \text{Gumbel}(0)$, 定义 $G_{\phi_i} = \phi_i + G_i$. 对于该分数进行采样:
        $$
        I^*= \arg\max_i G_{\phi_i}
        $$
    *   进一步扩展到 top-k 的 Gumbel 采样即为:
        $$
        I_1^\ast, \dots, I_k^\ast = \arg\text{top-k}(G_{\phi_i})
        $$
*   本文希望提出一个以这个 Gumbel 为地层分布的能够无放回的采样 $k$ 个序列的方法。然而问题的难点在于序列模型的搜索空间为指数大的, 遍历全部空间不可能。我们需要在不枚举所有序列的前提下进行分布建模。

#### 2. Methodology

*   将 sequence model 表示为一棵概率树, 对应将一个完整的序列 $\mathbf{y}_{1:T}$ 表示为一条从根到叶的路径:
    *   设 $\mathcal{Y}$ 为所有合法完整序列, $|\mathcal{Y}| = n.$ 其中每个 $\boldsymbol y\in\mathcal{Y}$ 是一个长度为 $T$ 的向量, $\boldsymbol y\in\mathcal V^T$, 对应树中的一个叶节点. 对于任意节点, 其都对应了一个部分序列 $\boldsymbol{y}_{1:t} = (y_1, \dots, y_t)$.
    *   每条路径的 log 概率为
        $$
        \phi(\boldsymbol{y}_{1:T}) = \log p_\theta(\boldsymbol{y}_{1:T}) = \sum_{t=1}^T p_\theta(y_t | \boldsymbol{y}_{1:t-1}) \in\mathbb{R}
        $$
    *   即使是对于同样的 $1:T$ 的序列也有不同的路径, 因此对于某个路径这里简记为 $\boldsymbol{y}^{(i)}$. 若在不引起歧义的情况下, 有时也忽略序列长度记上述对数概率分布为 $\phi_i$.
*   Perturbed Log- Probability on Partial Sequences
    *   延续上述记号, 用 $y^{(i)} \in S\subset \mathcal Y$ 表示以某个中间节点延伸出的所有叶节点的集合 (所有以该 prefix 开头接龙生发出的完整序列). 对 $S$ 中的每个向量再计算其 Gumbel 得分:
        $$
        G_{\phi_i} = \phi_i +G_i, ~G_i\stackrel{i.i.d}{\sim} \text{Gumbel}(0)
        $$
    *   计算这一组 Gumbel 分数的最大值作为 $S$ 的 Gumbel 分数:
        $$
        G_{\phi_S} = \max_{i \in S} G_{\phi_i}
        $$
    *   根据 Gumbel-max 性质（Maddison et al. 2014）.
        $$
        G_{\phi_S} \sim \text{Gumbel}\left(\phi_S\right),\quad  \phi_S := \log \sum_{i \in S}\exp (\phi_i)
        $$
        我们称 $G_{\phi_S}$ 为部分序列 $S$ 的 perturbed log-probability.

    > **Intuition:**
    >
    > *   每个 partial sequence $y_{1:t}$ 可以代表一个子树，其对应完整序列集合为 $S$
    > *   我们可以为这个子树打一个 perturbed score $G_{\phi_S}$，而不用展开所有 $y \in S$
    > *   **高的 $G_{\phi_S}$** 表示这个 prefix 有可能生成落入 top-k 的完整序列，因此应该保留并扩展它

    ***Algorithm Implementation***

    ![20250604151640](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250604151640.png)

#### 3. Experiment

*   实验 1: 验证 Gumbel-Top-k 是否可以用于从 softmax 分布中 无放回地采样 top-k 项
*   实验2: 检查 Stochastic Beam Search（SBS）生成的 top-k 序列是否符合 Gumbel-Top-k 理论分布
*   实验3: SBS 用于生成任务 (BLEU 与 Entropy 的估计)
    *   模型：Transformer-based seq2seq 模型（用于机器翻译，En → De）
    *   对比模型: Beam Search, Top-k Sampling, Ancestral Sampling, SBS
    *   评价标准: BLEU, Entropy

### 2.2 Nucleus Sampling / Top-p Sampling

*   **论文标题**: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
*   **作者**: Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y.
*   **年份**: 2019 (1904)

#### 1. Motivation

常见的解码策略有如下几种:

*   **Maximization-Based Decoding (e.g. Greedy, Beam Search)**
    *   目标: 寻找联合概率分布最高的一条路径.
    *   导致文本 “退化”: 内容重复 (repitition loop), 缺乏多样性 (blandness), 上下文不连贯 (incoherent), 与人类文本的概率分布差异极大

*   **Sampling Strategies**
    *   **Pure Sampling**
        *   每步直接从 softmax 中进行采样
        *   虽然避免重复, 但是 unreliable tail 词汇会扰乱语义, 输出经常不连贯或无意义
    *   **Top-k Sampling**
        *   只保留 top-k 高概率词，再在其中采样
        *   固定 k 无法适应不同上下文分布的“头部平坦/陡峭”问题。
        *   选太小 → 生成无趣；太大 → 引入无意义 token。
    *   **Temperature Sampling**
        *   $\mathbb P(X_i) = \frac{\exp{(p_i / T)}}{\sum_j \exp{(p_j / T)}}$
        *   降低温度能改善质量，但严重损失多样性

#### 2. Methodology

动态截断尾部噪声词，仅在累积概率质量 $≥ p$ 的最小子集上采样

$$
V(p) = \min \{ S \subseteq \mathcal{V} \mid \sum_{x \in S} P(x \mid x_{1:i-1}) \geq p \}
$$

*   保证采样集中在模型可信区域
*   避免重复、保持连贯、提高多样性
*   实验表明，在 $p ∈ [0.9, 0.98]$ 范围内性能最佳

#### 3. Experiment

基座模型: GPT-2 Large (762M)

*   Likelihood Evaluation
    *   以模型自身计算生成文本的 Perplexity：
        *   Beam Search 生成 **perplexity** 过低：意味着过于平庸重复
        *   Pure Sampling perplexity 过高：过度随机，语义漂移
        *   Nucleus Sampling 接近 gold text
*   Distributional Evaluation
    *   **Zipf 分布**（词频统计规律）
        *   自然语言呈 Zipf 分布（幂律型）
        *   Pure Sampling 和 Nucleus Sampling 最接近人类分布
    *   **Self-BLEU**（多样性衡量）
        *   越低越好. Nucleus 与人类最接近，Top-k 和低温度采样明显重复
    *   **Repetition Rate**
        *   Beam Search 最严重（可达 70% 重复片段）
        *   Top-k 和 Nucleus 明显更低，尤其在参数合理设置时
    *   **Human Evaluation：HUSE**
        *   HUSE = Human + Statistical Evaluation（来自 Hashimoto et al., 2019）
        *   结果显示：
            *   Nucleus Sampling 在所有方法中得分最高（0.97）
            *   top-k 次之，其他如 temperature 或 Beam Search 明显较差

#### 4. Key Takeaways

*   人类语言不等于概率最大化的语言：
    *   高质量语言往往故意规避可预测性（参见 Grice’s Maxims）。
    *   最大似然训练与生成质量不一致。
*   尾部分布问题本质：
    *   模型在尾部生成罕见词时变得不可控。
    *   强调采样空间控制的重要性。
*   规模越大，退化越严重：
    *   实验发现 GPT-2 Large/XL 的 Beam Search 会早早停止，表明过拟合高概率路径。

### 2.3 Mirostat

*   **论文标题**: [Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity](https://www.arxiv.org/abs/2007.14966)
*   **作者**: Sourya Basu, Govardana S. Ramachandran, Nitish S. Keskar, Lav R. Varshney
*   **年份**: 2020 (2007)

#### 1. Motivation

*   **Top-k** 和 **Top-p (nucleus)** 采样方法在生成文本时存在困境：
    *   小 k/p：容易陷入重复陷阱（boredom trap）
    *   大 k/p：容易生成不连贯内容（confusion trap）
*   论文实验表明，**人类更喜欢介于这两者之间的文本质量（即中等 perplexity）**
*   Mirostat 的核心目标是找一个机制动态调整 token 候选集的大小，使生成文本的平均 surprise $S(x_t) = -\log P(x_t \mid x_{<t})$ 逼近目标值 $\tau$. 换言之, 当我们给定目标 $\mathbb E S(x_t) \approx \tau$, 我们就可以从中反解出对于 $P(x_t)$ 的取值.

#### 2. Methodology

*   理论推导: Mirostat Sampling
*   给定超参数: $\tau \in \mathbb{R}_+$ 为目标 surprise; 学习率 $\eta\in(0,1]$. 另记词表 为 $\mathcal V, |\mathcal V| = N$ 为词表大小, LLM 在当前时刻的 logits 输出向量 $l_t$, 由 Softmax 得到的概率分布 $P_t$, 对应 token 的 surprise $S(x) = -\log P(x)$.
*   本文假设 token 序列服从 Zipf Distribution, 即
    $$
    P(x_{(j)})=(z \cdot j^s)^{-1}, ~ j=1,2,\cdots,N
    $$
    *   $x_{(j)}$ 表示 token 从高概率到低概率排序后对应项
    *   $s$ 是 Zipf 指数. 自然语言中的词频分布近似服从 $s\approx 1$
    *   $Z = \sum_{j=1}^N 1/j^s$ 为归一化常数.
*   当我们限制只从前 k 个 token 中进行采样时, 我们要求期望 surprise 为 $\mathbb E S(x_t) \approx \tau$. 则由 Zipf 分布:
    $$
    \mathbb{E}[S(x)] = -\sum_{i=1}^k P(i) \log P(i)
    $$
    因此我们可以通过选择合适的 $k$ 使得该期望近似于给定目标 $\tau$.

    ***Algorithm Implimentation***

    1.  估计 Zipf 分布参数:
        $$
        \hat{s} = \frac{\sum_{i=1}^{m-1} t_i b_i}{\sum_{i=1}^{m-1} t_i^2}
        $$
        其中 $t_i = \log\left( \frac{i+1}{i} \right),\quad b_i = \log\left( \frac{p_i}{p_{i+1}} \right)$, $m$ 是参与估计的 top-m (常设为 10~20).
    2.  确定实现当前目标 $\tau$ 所需要的 top-k 的 $k$ 的个数
        $$
        k = \left( \frac{\hat{\varepsilon}^2 \mu}{1 - N^{-\hat{\varepsilon}}} \right)^{1/\hat{s}},\quad \hat{\varepsilon} = \hat s-1
        $$
        *   $\mu$ 为内部状态变量, 用来跟踪 surprise. 初始化为 $2\tau$. 会参与迭代.
    3.  从 top-k 中采样 token
        *   按照 logits 对 token 进行排序, 取前 $k$ 个, 做 softmax 归一化, 然后采样出当前步骤下的一个 token $x_t$
        *   计算其实际 surprise: $S(x_t) = -\log P(x_t)$
    4.  反馈控制更新 $\mu$
        *   计算 surprise 之误差 $e = S(x_t) - \tau$
        *   更新 $\mu := \mu - \eta e$
            *   若 surprise 太大, $e>0$, 说明当前 token 太难, 则减少 $\mu$, 从而减少 $k$ . 反之亦然.

#### 3. Experiment

*   实验配置: GPT-2 (117M)
*   数据集: 每组实验使用 **相同的上下文输入**，每次生成 200 tokens. 输入语料包括短新闻、开放域文本等，无需 fine-tune.
*   评价指标:
    *   Cross-entropy rate: 平均 surprise（信息量），衡量 perplexity 控制能力
    *   Repetition rate (1-gram / 6-gram): 测量重复情况，衡量生成质量
    *   人类评估: Fluency (1–7）, Coherence (1–7）, 被人类判为“非 AI”的概率

### 2.4 Arithmetic Coding*

*   **论文标题**: [Arithmetic Sampling: Parallel Diverse Decoding for Large Language Models](https://www.arxiv.org/abs/2210.15458v2)
*   **作者**: Luke Vilnis, Yury Zemlyanskiy, Patrick Murray, Alexandre Passos, Sumit Sanghai
*   **年份**: 2022 (2210)

#### 1. Motivation

*   常见的文本生成解码策略存在明显的多样性–并行性权衡问题. 传统的语言模型采样规则为一步一步根据 $\mathbb{P}(w_t |w_{<t})$ 进行采样:
    *   搜索类方法（如 Beam Search、Gumbel top-k）
        *   能生成多样化的样本（如语义/词汇不同的候选翻译）
        *   但需要跨 beam 同步、难以并行（对硬件要求高）
    *   采样类方法（如温度采样、top-k、nucleus sampling）：
        *   完全可并行（不同样本间独立）
        *   但样本往往高度重复，缺乏结构性多样性（e.g. 生成重复的高概率候选）

#### 2. Methodology

*   Arithmetic Sampling 理论
    *   本质上, 语言模型定义了一个在 $\mathcal{X}$ 上的离散概率分布.
        $$
        {P}(\mathbf{x}) = \mathbb P(x_1, ..., x_T) = \prod_{t=1}^T \mathbb P(x_t \mid x_{<t})
        $$
        因此对于每个序列 $\mathbf{x}$, 相当于语言模型将其映射到了 $[0,1)$ 区间中一个长度为 $P(\mathbf{x})$ 的子区间.
    *   对于每个序列 $\mathbf{x} \in \mathcal{X}$ 可以定义区间
        $$
        I(\mathbf{x}) = [L(\mathbf{x}), L(\mathbf{x}) + P(\mathbf{x}))
        $$
        *   其中 $L(\mathbf{x})$ 是所有字典序在 $\mathbf{x}$ 之前的序列的累积概率:
            $$
            L(\mathbf x) = \sum_{\mathbf x' \prec \mathbf x} P(\mathbf x')
            $$
        *   这样我们就得到了一个完整的序列划分 $\{I(\mathbf x)\}$, 其中每个区间互不重叠, 且 $\bigcup_{\mathbf x\in\mathcal X}I(\mathbf x) = [0,1)$. 我们称这个 partition 为 *codebook (编码字典).*
    *   在定义了上述 codebook 之后, 就可以反之当给定 $c\in[0,1)$ 时, 找到 $x = \arg c\in I(x)$.
*   递归构造 Code-book
    *   理论上每一个序列 $\mathbf{x}$ 都对应了一个编码区间 $I(\mathbf{x}) = [L(\mathbf{x}), L(\mathbf{x}) + P(\mathbf{x}))$. 但是我们不可能在实践中遍历所有的序列 $L(\mathbf{x})$ 然后查询.

#### 3. Experiment

*   实验配置: Google T5-base
*   对比算法:
    *   Greedy: 每步取最大概率 token
    *   Standard Sampling: Softmax 采样，无截断
    *   Beam Search: 搜索前 k 条最优路径
    *   Gumbel Top-k: 多样性强，但非并行
    *   Arithmetic + Top-k: 加入概率空间截断，增强控制力

### 2.5 η-Sampling

*   **论文标题**: [Truncation Sampling as Language Model Desmoothing](https://www.arxiv.org/abs/2210.15191)
*   **作者**: John Hewitt, Christopher D. Manning, Percy Liang
*   **年份**: 2022 (2210)

#### 1. Motivation

*   **过度截断高概率词**：Top-p 方法可能在保留累计概率达到 p 的词集合时，意外地截断了仍具有较高概率的词，导致信息丢失。
*   **缺乏对概率分布形状的适应性**：现有方法未能根据当前概率分布的形状动态调整截断策略，可能在模型不确定时引入噪声，或在模型确定时限制多样性。

#### 2. Methodology

***Theory***

*   语言模型的 Smoothing 框架

    > LM 的输出视为 True Distribution 与平滑分布 Smoothing Distribution 的混合. 因此在采样生成时，我们需要一个合理机制来“反平滑”，即剔除仅由于平滑造成的噪声候选词，恢复接近真实分布的支持集
    >
    $$
    P_{\text{LM}} = (1 - \lambda) P_{\text{true}} + \lambda P_{\text{smooth}}
    $$
    *   $P_{\text{LM}}$：语言模型的输出分布, 即 LM 中的最终 Softmax 输出
    *   $P_{\text{true}}$：真实分布，表示模型对当前上下文的真实预测
    *   $P_{\text{smooth}}$：平滑分布，通常为均匀分布，用于避免无限困惑度
    *   $\lambda$：平滑系数，控制平滑分布的影响程度。

*   Desmoothing Framework 去平滑建模

    截断采样中, 根据上述的 Smoothing 框架, 我们希望估计如下集合:
    $$
    S^*_{\mathbf{x}_{<t}} = \{ x \in \mathcal{V} \mid P_{\text{true}}(x \mid \mathbf{x}_{<t}) > 0 \}
    $$
    *   $P_{\text{true}}(x \mid \mathbf{x}_{<t})$ 表示给定上文 $\mathbf{x}_{<t} = (x_1,\cdots,x_{t-1})^\top$时预测 $x$ 的真实条件概率. 是我们希望 Denoise 的目标.
    *   $S^*_{\mathbf{x}_{<t}}$ 因此表示在真实分布下条件概率非零的 token 所组成的集合. 在统计学语言上相当于 $P_{\text{true}}$ 的 Support. 而在实践中即为通过某一特定阈值保留的词汇库.

*   在进行估计时有两大核心 Truncation 原则:

    1.  Absolute probability principle：若某个词的概率 $P_{\text{true}}(x \mid \mathbf{x}_{<t})$ 明显高于平滑项上限，则应保留.
    2.  Relative probability principle：若当前分布熵很高（模型不确定），则应允许更多词被采样，反之则保守.

***Algorithm***

*   **η-Sampling:** 根据当前概率分布的熵动态调整截断阈值
    1.  计算当前时刻模型输出的词分布 $\{p_i\}_{i=1}^V$ (词表大小为 $V$) 之[信息熵](https://www.notion.so/1fd60a551112802cb5a5e0c7c941d765?pvs=21)
        $$
        H = -\sum_{i} p_i \log p_i
        $$
    2.  设定截断值
        $$
        \eta(H) = \min(\varepsilon, \alpha \cdot \exp(-H))
        $$
        *   $\epsilon$ 为绝对概率下限, 防止截断所有词. 常取 $\epsilon = 0.001$.
        *   $\alpha >0$ 为超参数, 控制采样的保守程度, 越大则对应的阈值越大.推荐 $\alpha = \sqrt{\epsilon}$.
        *   $\exp(-H)$ 为信息熵的指数函数，熵越大说明概率分布越平均 (模型越不确定)，阈值越小（采样越开放）
    3.  截断低频词
        $$
        \mathcal{S} = \{x_i\in\mathcal{V}:p_i \ge \tau\}
        $$
        *   只保留 Softmax 结果中概率高于 $p_i$ 的 token. 从保留集合中归一化后采样，等价于对模型原始输出进行动态调整的 top-p 操作, 但阈值为函数 $\tau(H)$而非静态参数.

#### 3. Experiment

*   实验配置: GPT-2 (124M ~ 1.5B)
*   数据集: GPT-2 的原始训练语料 WebText 的验证集和测试集中的 held-out 数据
*   对比模型:
    *   Top-p（核采样）：保留累计概率达到 p 的最小 token 集合进行采样 ($p\in [0.8, 0.95]$)
    *   Typical Decoding：根据 token 的典型性进行采样，优先选择熵接近平均值的 token ( $p\in [0.8, 0.95]$)
    *   ϵ-sampling：基于熵的采样方法，动态调整采样池 ($p\in [0.01, 0.1]$)
*   评价指标:
    *   MAUVE 分数: MAUVE 是一个自动化指标，用于评估生成文本的质量和多样性，结合了精确度和召回率
    *   人类评估: 比较不同采样方法生成的长文本后缀的连贯性
    *   重复率: 分析了不同采样方法生成文本的重复率

#### 4. Key Takeaways

*   η-sampling 的优势：在多个评估指标上，η-sampling 均表现优越，特别是在生成长文本时，其连贯性和多样性均优于传统的 Top-p 和 Typical Decoding 方法。
*   动态截断策略的有效性：通过根据当前分布的熵动态调整截断阈值，η-sampling 能更好地适应不同的上下文，避免了固定阈值方法可能导致的信息丢失或引入噪声的问题。
*   实用性：η-sampling 方法简单易实现，计算开销较低，适合在实际应用中部署。

### 2.6 Min-p

*   **论文标题**: [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](https://www.arxiv.org/abs/2407.01082)
*   **作者**: Minh Nguyen, Andrew Baker, Clement Neo, Allen Roush, Andreas Kirsch, Ravid Shwartz-Ziv
*   **年份**: 2024 (2407)

#### 1. Motivation

静态阈值无法根据模型信心动态收敛或放宽候选集

*   Top-p：当温度升高时，为保证多样性会接纳概率极低的 token，导致语义漂移或胡言乱语
*   Top-k：阈值刚性，可能排除情境上合适但低概率的 token

#### 2. Methodology

*   计算截断阈值: $p_{\text{scaled}} \;=\; p_{\text{base}} \times p_{\max},$ $p_{\text{base}}$ 为超参数 (0.05~0.1), $p_{\text{max}}$ 为当前步最大概率.
*   低信心时 $p_{\text{scaled}}$ 降低，候选集自动扩张；高信心时则收紧，从而自适应地权衡创造性与连贯性

#### 3. Experiment

*   实验配置:
    *   **Mistral 7B** 模型, 并在部分实验中使用了 **Mistral Large（123B 参数）** 模型，以评估 min-p 方法在不同规模模型上的表现
    *   采用了 **vLLM** 框架进行推理
*   数据集:
    涵盖从小学数学到博士级别的推理任务，以及创意写作任务
    *   **GPQA**：博士级别的常识问答数据集，评估模型的高级推理能力
    *   **GSM8K**：小学数学问题解答数据集，采用 Chain-of-Thought（CoT）提示，评估模型的数学推理能力
    *   **AlpacaEval Creative Writing**：创意写作任务，评估模型在生成富有创造性和连贯性的文本方面的能力
*   对比模型:
    *   **Top-p（核采样）**：保留累计概率达到 p 的最小 token 集合进行采样。
    *   **Top-k**：保留概率最高的 k 个 token 进行采样。
    *   **ϵ-sampling**：基于熵的采样方法，动态调整采样池。
    *   **η-sampling**：另一种基于熵的采样方法，动态调整采样池。
    *   **Mirostat sampling**：控制困惑度的采样方法，旨在保持输出的多样性和连贯性。
    *   **Greedy Decoding**：每一步选择概率最高的 token，属于确定性解码方法。
*   评估指标:
    *   **准确率（Accuracy）**：在 GPQA 和 GSM8K 数据集上, 评估模型生成的答案与参考答案的匹配程度.
    *   **人类评估（Human Evaluation）**：在 AlpacaEval Creative Writing 任务中, 采用双盲偏好测试，评估生成文本的质量和创造性.
*   关键发现 :
    1.  在温度 $T\ge1.2$ 区间，min-p 在“连贯度—创造性”二维上显著优于 top-p、ϵ-sampling、Mirostat.
    2.  人类评审对故事叙事性与画面感给出更高偏好比例.
    3.  负载开销与 top-p 基本持平，GPU 推理速度差异<3%.

#### 4. Key Takeaways

*   **稳定性**：min-p 在不同温度设置下表现稳定，特别是在高温度下仍能保持较高的准确率，适用于需要高多样性输出的任务。
*   **参数选择**：作者建议将参数 $p_{\text{base}}$ 设置为 0.05 或 0.1，实验证明在不同任务中该值表现良好。
*   **实现简便**：该方法易于实现，只需在 logits 上进行简单的统计操作，无需复杂的概率计算

### 2.7 Top-nσ

*   **论文标题**: [Top-nσ: Not All Logits Are You Need](https://www.arxiv.org/abs/2411.07641)
*   **作者**: Chenxia Tang, Jianchun Liu, Hongli Xu, Liusheng Huang
*   **年份**: 2024 (2411)

#### 1. Motivation

*   传统采样方法（如 top-k、top-p、min-p）在处理推理任务时往往面临“多样性 vs. 准确性”的权衡，尤其在高温度（temperature）下表现不佳. 尤其是 temperature scaling + top-p 在温度升高时会无意识地引入更多垃圾 token
*   作者观察到 logit 分布实际上具有非常清晰的 结构性双峰模式：
    *   一个 Gaussian-like 噪声区: 包含低概率的 token，可能引入不连贯或不相关的内容.
    *   一个 离群高值“信息区”: 包含高概率的 token，通常对应于模型较为确定的输出.

#### 2. Methodology

> 与传统的 top-k / top-p 等在 softmax 概率后进行 token 筛选不同，Top-nσ 是直接在 logit 空间中基于统计距离筛选 token
>
*   设当前步骤的 logits 分布为: $l = \{l_1, \dots, l_V\}$. 记其中最大值为 $M = \max(l)$, 标准差为 $\sigma$. 设超参数阈值系数为 $n$.
*   计算 candidate token set 为:
    $$
    \{x_i \in \mathcal V | \text{logit}(x_i) \ge M-n\sigma \}
    $$
    *   该操作的直觉为: 只保留在最高 logit 附近的 token，排除 Gaussian“噪声区”
*   在得到当前 candidate token 后再进行标准 softmax 采样

***Theory***

*   Logit 分布建模：
    *   “噪声区”：近似 Gaussian（大量无关 token），
    *   “信息区”：logits 显著偏高，主导 softmax 输出。
    *   实测中 top token 经常距离均值超出 $10σ$。
*   与 top-p / min-p 等方法的对比：
    *   top-p 会随着温度升高引入更多噪声 token；
    *   min-p 实质等价于 logits 空间的固定阈值截断（即 uniform 分布上的截断）；
    *   Top-nσ 则具有“温度不变性”（temperature invariance）：
        *   由于筛选条件基于相对距离（标准差的倍数），因此不会受温度缩放影响；
        *   其他方法会因 temperature 改变 logits 分布形状，导致选择集变化。
*   公式性质分析：
    *   推导了 Gaussian 与 Uniform 情形下阈值与累积概率质量 $p$ 的对应关系 (参考 [Nucleus Mass](https://www.notion.so/Nucleus-Mass-1fd60a55111280d8848de69b75a2ca6e?pvs=21) : Top-nσ 在不同分布类型下都能保留高质量的概率核)
    *   合理的 n 的取值范围为: $n \in (0, 2\sqrt{3}) \approx (0, 3.46)$

#### 3. Experiments

*   实验配置:
    *   LLaMA-3-8B-Instruct 模型进行评估
    *   采用了 vLLM 框架进行推理
*   数据集:
    以推理为主的问答数据集上进行了评估，涵盖从小学数学到博士级别的问题. 所有数据集都被转换为开放式生成任务，模型需要生成答案，然后与参考答案进行比较.
    1.  **AQuA**：代数问题解答数据集。
    2.  **MATH**：涵盖高中到研究生水平的数学问题。
    3.  **GSM8K**：小学数学问题解答数据集。
    4.  **GPQA**：博士级别的常识问答数据集。
*   评估指标:
    *   Exact Match (EM)：单次采样生成的答案与参考答案完全匹配的比例
    *   Maj@N：模型生成 N 个不同的响应，通过多数投票确定最终答案，然后计算与参考答案的匹配度
*   对比模型:
    *   **Top-k**：保留概率最高的 k 个 token 进行采样 ($k=20$)
    *   **Top-p (Nucleus Sampling)**：保留累计概率达到 p 的最小 token 集合进行采样 ($p=0.9$)
    *   **Min-p**：保留概率高于某个最小阈值的 token 进行采样 ($p_{\min} = 0.1$)
    *   **Temperature Sampling**：通过调整温度参数控制采样的随机性
    *   **Greedy Decoding**：每一步选择概率最高的 token，属于确定性解码方法。

#### 4. Key Takeaways

*   **无需温度调节**：该方法在不依赖温度参数的情况下，稳定地控制采样空间，简化了超参数的调节过程。
*   **适用于推理任务**：在需要高准确性和连贯性的任务中，Top-nσ 提供了更可靠的采样策略。

## 3. 综述性文献与相关工作

*   **论文标题**: [A Thorough Examination of Decoding Methods in the Era of LLMs](https://www.arxiv.org/abs/2402.06925)
*   **作者**: Chufan Shi, Haoran Yang, Deng Cai, Zhisong Zhang, Yifan Wang, Yujiu Yang, Wai Lam
*   **年份**: 2024 (2402)
*   **内容**: 当前 Decoding 方法有如下主流策略:
    *   确定性采样 Determined:
        *   Greedy Search
        *   Beam Search
    *   随机性采样 Stochastic:
        *   Top-k
        *   Top-p (Nucleus)
        *   Temperature Sampling

---

**文献列表**

*   **Locally Typical Sampling**. [Meister, C., Cotterell, R., Vieira, T.](https://arxiv.org/abs/2202.00666). Published: 2022–2023.
*   **Truncation Sampling as Language Model Desmoothing**. [Hewitt, J., Li, X.L., Yu, K.](https://openreview.net/forum?id=W1G1JZEIy5_). Published: 2022.
*   **Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs**. [Nguyen, T.T., Kalomaze, R., Menhguin, E., et al.](https://openreview.net/forum?id=FBkpCyujtS). Published: ICLR 2025.
*   **Contrastive Decoding: Open-ended Text Generation as Optimization**. [Li, X.L., Li, X., Jurafsky, D.](https://arxiv.org/abs/2210.15097). Published: 2022.
*   **Hierarchical Neural Story Generation**. [Fan, A., Lewis, M., Dauphin, Y.](https://arxiv.org/abs/1805.04833). Published: 2018.
*   **PaLM: Scaling Language Modeling with Pathways**. [Chowdhery, A., Narang, S., Devlin, J., et al.](https://arxiv.org/abs/2204.02311). Published: 2022.
*   **Contrastive Decoding Improves Reasoning in Large Language Models**. [Li, X.L., Wei, J., Du, Y., et al.](https://openreview.net/forum?id=SzV37yefM4). Published: 2023.
*   **Accelerating Large Language Model Decoding with Speculative Sampling**. [Lee, K., Zoph, B., Shazeer, N., et al.](https://arxiv.org/pdf/2302.01318). Published: 2023.
*   **Tail-Free Sampling**. Bricken, T.. Published: 2019 (blog post).
*   **Literature Review on Sampling Techniques for Language Models**. [Kumar, N.J.](https://www.njkumar.com/literature-review-sampling-techniques/). Published: 2023 (blog article).
*   top-p、ϵ-sampling、Mirostat