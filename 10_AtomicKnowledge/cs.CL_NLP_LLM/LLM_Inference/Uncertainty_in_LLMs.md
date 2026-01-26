---
aliases: [LLM不确定性, Uncertainty in LLMs, Confidence Estimation, LLM置信度, Semantic Entropy]
tags:
  - concept
  - ml/deep-learning
  - cs/nlp
related_concepts:
  - "[[Perplexity]]"
  - "[[Entropy]]"
  - "[[Deep_Reasoning_for_LLMs]]"
  - "[[Information_Theory]]"
  - "[[KL_Divergence]]"
  - "[[Conformal_Prediction]]"
  - "[[Context_Engineering]]"
  - "[[BERT]]"
source: "Semantic Uncertainty (ICLR 2023); Detecting Hallucinations (Nature 2024)"
---

# Uncertainty in LLMs 综述

## Introduction: Measuring Confidence of LLMs

首先给出一点背景介绍, 以及一些典型的对于任务的衡量方法. 本小节的介绍逻辑及记号主要参考: Scalable Best-of-N Selection for Large Language Models via Self-Certainty (https://arxiv.org/pdf/2502.18581)

## LLM Background

通常考虑一个自回归形式的语言模型 (Language Model, LM) $p(\cdot \mid x)$, 其中 $x = (x_1, x_2, ..., x_n)$ 是输入序列, 每一个 $x_i\in \mathcal{V}$ 是词汇表 $\mathcal{V}$ 中的一个 token. 语言模型的本质是建模这个 sequence 的联合分布，即：
$$
L_x = (\ell_1, \ell_2, ..., \ell_n) , \quad \ell_i \in \mathbb{R}^{|\mathcal{V}|}
$$
其中 $\ell_i$ 是 $x_i$ 的 logits 向量, 其长度等于词汇表大小 $|\mathcal{V}|$, 相当于每个 token 在词汇表中的概率分布. 在得到 $y = (y_1, \cdots, y_m)$ 的输出后, 预测下一个 token 的概率分布为:
$$
p(y_{m+1} \mid x, y_{1}, \cdots, y_m) 
$$
这反映了模型基于输入 $x$ 和已生成的输出 $y$ 对下一个 token 的预测, 或 "belief".

## Confidence Estimation

这里的 confidence 稍微有别于统计学中频率学派的置信度 (confidence interval), 其更接近于衡量模型的主观 certainty. 但是某种程度上其也和模型的分布, 以及分布的集中与离散程度有关. 下面列出一些较为常见的衡量.

### Sentence-Level Problistic Confidence

第一类方法是直接使用模型实际生成的序列中已采样的 token 来直接衡量模型在生成该序列时的 confidence. 

#### Average Log-Probability

$$
\text{AvgLogP}:= \frac{1}{n} \sum_{i=1}^{n} \log p(y_i \mid x, y_{\lt i})
$$

- 该方法直接计算生成每个 token 时的条件概率 (即每一步的 softmax probability) 的对数平均值.
- 就相当于当前序列的对数似然 (log-likelihood) 的平均值.
- 该方法简单直接, AvgLogP 越大, 说明模型对生成的序列越有信心.

#### Perplexity
$$
\text{Perplexity}:= \exp\left(-\frac{1}{n} \sum_{i=1}^{n} \log p(y_i \mid x, y_{\lt i})\right)
$$
- 该方法是对 Average Log-Probability 的指数化处理, 二者某种程度上是等价的.
- Perplexity 越小, 说明模型对生成的序列越有信心.

### Distributional Confidence

上述两种方法局限于生成的序列本身. 但更科学的做法是充分考虑生成时的每一个时间步的完整概率分布, 以刻画模型是否更"集中地相信某些 token".

总的而言, 该 distributional confidence 可以总结为如下范式:
- 设 $P_{y\mid x} = (p(\cdot \mid x), p(\cdot \mid x, y_1), \cdots, p(\cdot \mid x, y_{n-1}))$ 是模型在每个时间步的概率分布.
- $f(\cdot)$ 是一个局部函数, 作用于每个时间步的概率分布 $C:=f(p(\cdot \mid x, y_{\lt i}))$.
- $F(\cdot)$ 是一个汇总函数 (例如取平均值), 将上一步的局部函数结果汇总为全局的 confidence 值
根据上述思路, 可以定义如下的 distributional confidence:
$$
\text{Distributional Confidence} := F\left(f(P_{y\mid x})\right)
$$
以均值汇总为例, 则有:
$$
F(C) = \frac{1}{n} \sum_{i=1}^{n} C_i = \frac{1}{n} \sum_{i=1}^{n} f(p(y_i \mid x, y_{\lt i}))
$$

具体地, 根据 $f(\cdot)$ 的不同, 可以得到不同的 distributional confidence 方法.  不过整理一下几种方法可见, 其本质上都是 $\sum^{V}_{j=1} p(j \mid x, y_{\lt i})$ 的某种函数.

#### Kullback-Leibler Divergence (KL-Divergence, KL 散度)

$$C_i^{\text{KL}} = \text{KL}(U \| p(\cdot \mid x, y_{\lt i})) = \sum_{j=1}^{V} \frac{1}{V} \log\left( \frac{1/V}{p(j \mid x, y_{\lt i})} \right) = -\frac{1}{V} \sum_{j=1}^{V} \log\left( V \cdot p(j \mid x, y_{\lt i}) \right)$$

- 衡量当前分布与 uniform 分布的差异
- 差异越大 → 分布越集中 → 置信度越高；
- Self-Certainty 就是用 KL 散度构建的度量.



#### Gini Impurity

$$C_i^{\text{Gini}} = \sum_{j=1}^{V} p(j \mid x, y_{\lt i})^2$$

- 表示"两次随机采样取到不同 token"的概率；
- 越集中，Gini 值越高 → 置信度越高


#### Entropy

$$C_i^{\text{Entropy}} = \sum_{j=1}^{V} p(j \mid x, y_{\lt i}) \log p(j \mid x, y_{\lt i})$$

- 熵越高 → 分布越发散 → 越不确定；

- 所以作者用 负熵 来表示"置信度"；

#### Distributional Preplexity

$$C_i^{\text{DP}} = -\exp\left( -\sum_{j=1}^{V} p(j \mid x, y_{\lt i}) \log p(j \mid x, y_{\lt i}) \right)$$

- 实际上是将熵指数化后的反向量；

- 越小表示模型更 confident；

- 本质上与负熵类似，只是数值缩放不同.



# Related Work

下面是总结的一些与模型的 confidence 相关的工作. 

## Meta-Cognitive: P(True) & P(IK)

Anthropic 提出的两个关于模型在生成回答时自我反思来衡量其 confidence 的方法. 其核心思想是让模型自己判断其回答的正确性, 以此来衡量模型的 certainty. 某种意义上这反映了模型自我的元认知能力 (meta-cognition).
- [Language Models (Mostly) Know What They Know (2207)](https://arxiv.org/abs/2207.05221)

### P(True)


#### Methodology

P(True) 表示模型对其生成的某个答案是否"在事实层面上正确"所给出的概率估计. 它是关于"外部世界事实"的判断，这与传统任务如 QA 的区别在于，模型不只是给出一个答案，而是要对自己生成的这个答案是否正确做出评估。这是一种 self-evaluation，反映的是模型的"元认知能力"

其形式上被定义为:
$$P(\text{True}) = \mathbb{P}[\text{answer } a \text{ is correct} \mid q, a]$$

评估流程为:
1. 给定问题 $q$ 
2. 采样得到模型的答案 $a\sim p(\cdot \mid q)$ (常用 $T=1$ 的采样)
3. 用以下 prompt 让模型判断答案的正确性:
    ```
    Question: {q}
    Answer: {a}
    Is the proposed answer:
    (A) True
    (B) False
    ```
4. 从中提取模型对答案正确性的概率 $P(\text{True})$.

不过 P(True) 也有一些局限性:
- 模型难以区分"生成来自自己" vs "来自他人"的文本，会在自我样本上"过于自信"
- 任务分布外泛化有限，仅在见过的任务上表现良好
- 无监督训练的模型可能只是学会了"答案听起来像真的"，而非真的理解语义正确性
- 不适用于无 ground-truth 或多解空间任务（如哲学性问题或开放式问题）

#### Experiments

***评估任务***

| 数据集             | 类型     | 简要说明                      |
| --------------- | ------ | ------------------------- |
| TriviaQA        | 事实问答   | 闭卷知识检索                    |
| Lambada         | 语言建模填空 | 长上下文后续预测                  |
| Codex HumanEval | 代码生成   | 写出实现某个函数的 Python 代码       |
| GSM8k           | 数学问题   | 基于 Chain-of-Thought 推理的题目 |
| Arithmetic      | 基础运算   | 简单整数四则运算等                 |

所有任务都使用 T=1 采样多个样本，对其中每一个样本进行自我评估。

***Few-shot vs Zero-shot***

实验证明：

- 在 zero-shot 下，模型的 $P(\text{True})$ 校准性较差，预测值集中在中间（如 0.4～0.6）；
- few-shot 提示显著提升校准性和区分度
- 提供多个候选答案（如5个 T=1 样本）再要求判断某个答案的真实性，可进一步提升效果



***性能评估***

- Accuracy conditioned on $P(\text{True})>0.5$: 验证那些"模型自以为正确"的答案的实际正确率.
- Brier Score（综合判别力与校准性）: $\text{Brier} = \frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2$ 其中 $y_i$ 是样本 $i$ 的实际标签 (0或1), $P_i$ 是模型预测的概率. Brier Score 越低，表示模型的预测越准确且校准性越好.
- AUROC: 衡量模型能否把正确和错误样本有效区分开，但不考虑概率是否校准。
- Calibration Curve 和 Expected Calibration Error (ECE): 衡量模型输出概率的统计一致性。

### P(IK)

P(IK) 是 $P(\text{I-Know})$ 的缩写，表示：模型认为它是否"知道"某个问题答案的概率。与 P(True) 不同，P(IK) 是一个类似前验的考虑, 是在模型生成答案之前就对问题的知识状态进行评估. 

#### Methodology

首先, 我们定义一个模型对于 I-Know 的 Ground-truth 的定义:
$$\text{Ground-truth } P(\text{IK}) = \mathbb{1}\left[\frac{1}{N} \sum_{i=1}^N \mathbb{1}[a_i \text{ correct}] > 0.5\right]$$ 
即对于某个问题, 令模型进行多次采样输出. 如果输出的答案经过评价判断发现有超过一半的答案是正确的, 则认为模型对该问题是 I-Know 的. 

在得到 Ground-truth 后, 我们可以通过有监督的方法来训练一个语言模型使其能够预测 P(I-Know). 具体而言, 其在原有语言模型的结构上添加一个分类问题的 value head, 该线性层的输出就是一个标量, 就用来预测模型对于当前输出的 I-Know 概率. 

此外这个结构的设计是相对独立于文本的输出的, 其不改变任何的输出策略, 只单纯的对模型的输出进行一个 I-Know 的概率预测. 

## Semantic Entropy

Semantic Entropy 是一种基于语义的熵度量方法, 旨在衡量语言模型 (LLM) 在生成自然语言文本时的语义不确定性. 其核心思想是将 token 级别的概率分布转化为语义级别的概率分布, 通过对语义类进行聚类和熵计算来评估模型的信心.


- [Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation (ICLR 2023)](https://arxiv.org/abs/2302.09664)

该文章重点侧重于 Semantic Entropy 在 QA 任务中的表现. 


### Methodology: Semantic Entropy

核心思想： 将 token 级别的单位判断转化为 Semantic Class. 例如对于问题 `What is the capital of France?` 可以定义一个语义等价类 `c = {'Paris', 'The capital of France is Paris', 'France's capital is Paris', ...}` . 尽管其 token 层面的表达方式各不相同, 但是只要 LM 回答的结果落入这个语义类中, 我们就认为是正确答案.

#### 数学模型

记模型的输入(如 QA 问题) 为 $x$ (例如 `What is the capital of France?` ),  LLM 在 $M$ 次采样后得到输出样本 $s_1, s_2, ..., s_M$ (例如 `Paris`). 此外假设我们可以判断每次的输出属于某个语义类 $s_i \in c \in \mathcal{C}$,  其中 $\mathcal{C}$ 是所有语义类的集合.

则模型生成某个语义类 $c$ 的总概率可用生成的样本中落入该语义类的频率来估计:
$$
p(c \mid x) = \frac{1}{M} \sum_{i=1}^{M} \mathbb{I}(s_i \in c)
$$

因此定义语义熵 (Semantic Entropy) 为：
$$
\text{SE}(x) = -\sum_{c \in \mathcal{C}} p(c|x) \log p(c|x)
$$
- 如果模型所有的回答都落入同一个语义类,  那么语义熵为 0,  表示模型非常确定；
- 如果模型的回答分布在多个语义类中,  那么语义熵会较高,  表示模型不确定.


#### 实现步骤

**1. 抽样生成 (Sampling Generation)**

- 目标: 使用语言模型 $p(y \mid x)$ 生成 $M$ 个样本 $s_1, s_2, ..., s_M$ 近似输出分布.
- 方法:
  - 使用 Temperature Samping 从模型生成多个样本.
  - 不使用 top-k 或 top-p 截断以确保最大熵估计精度.

**2. 语义聚类 (Semantic Clustering)**

- 目标: 将不同的输出样本聚类到语义等价类中.
- 核心思想: *两个句子被划分到同一个语义类当中, 当且仅当它们双向蕴含 bidirectional entailment:
    $$
    \text{Entail}(s_i, s_j) \land \text{Entail}(s_j, s_i) = \text{True}
    $$
- 判定方法: 使用经过 MNLI fine-tuning 的 DeBERTa-Large 模型，作为文本蕴含判断器.
- 聚类方法: Agglomerative clustering:
  - 初始时每个样本都是一个独立的簇.
  - 对所有的句子对都判定是否双向蕴含
  - 如果两个句子对满足双向蕴含, 则将它们合并到同一个簇中.
  - 重复, 直到收敛. 
-  最终模型得到若干 disjoint 语义类 $\mathcal{C} = \{c_1, c_2, ..., c_K\}$, 
  每个语义类 $c_k$ 包含所有满足双向蕴含的样本, 即认为它们表达了相同的语义.
    - 文中指出, 在 QA 任务中语义类个数一般为 2~5 个.
- 时间复杂度为 $\mathcal{O}(M^2)$, 其中 $M$ 是样本数量.

> ***补充***
> 1. **Natural Language Inference (NLI) 任务**
> - NLI 是 NLP 中的一个重要任务, 其输入两个句子 *A*, *B*, 要求输出从三种关系中判断它们的关系:
>   - **Entailment**: *A* 蕴含 *B*, 即若 *A* 为真, 则 *B* 必为真.
>   - **Contradiction**: *A* 和 *B* 互相矛盾, 即若 *A* 为真, 则 *B* 必为假.
>   - **Neutral**: *A* 和 *B* 之间没有蕴含或矛盾关系, 即 *A* 为真时 *B* 可能为真也可能为假.
>
> 2. **MNLLI (Multi-Genre Natural Language Inference)**
> - MNLI 是 GLUE benchmark 中的一个大规模 NLI 数据集, 每个样本是一个 `(premise, hypothesis)` 对,  并标注了它们之间的关系 (Entailment, Contradiction, Neutral).
> - 训练集规模约 40 万条.
>
> 3. **DeBERTa-Large 模型**
> - DeBERTa-Large 是微软提出的一种增强 BERT 架构的 Transformer 模型, 通常有 24 层 Transformer, 约 304 M 参数.
> - 经过 MNLI 数据集的微调, 使其能够很好地判断句子之间的蕴含关系.
  
**3. 熵估计 (Entropy Estimation)**
- 目标: 计算每个语义类的概率分布 $p(c|x)$ 并计算语义熵:
    $$\text{SE}(x) = -\sum_{c \in \mathcal{C}} p(c|x) \log p(c|x)$$

- 方法:
  - 对每个语义类 $c$ 计算其概率:
      $$p(c|x) = \frac{1}{M} \sum_{i=1}^{M} \mathbb{I}(s_i \in c)$$
  - 代入到语义熵公式中计算 $\text{SE}(x)$.


### Experiments

#### 实验设计

- 目标问题: Semantic Entropy 是否能更有效地衡量语言模型的"语义不确定性"？当模型不确定时，其回答在语义上是否更加多样？

- 设计思路:
  - 使用 QA 任务作为测试平台（有明确"正确答案"）；
  - 比较语义熵与传统 token-level 熵、打分器、输出相似度等方法的效能；
  - 衡量指标为：是否能识别出错误的生成回答（即不确定性高是否意味着回答错误）；

#### 数据集,模型和评价指标

- 数据集: 
  - TriviaQA
    - 封闭式问答任务，有标准答案
    - 使用 exact match 和字符串重叠的规则匹配来判断回答是否正确. 如果模型生成的回答与 GT answers 足够接近（例如包含正确实体），视为正确. 
  - CoQA
    - 开放式问答任务, 每个问题提供多个参考答案；
    - 作者使用 MNLI-DeBERTa 模型判断生成回答是否与任一参考答案"语义等价"（双向蕴含）. 若满足这一点，则标记为"正确"

  
- 模型: GPT-J (6B), GPT-NeoX (20B, 30B). 温度为 0.5, 不使用 top-k 或 top-p 截断. 采样 10 次.

- 评价指标: AUROC (Area Under the Receiver Operating Characteristic Curve), 衡量模型在识别正确回答和错误回答时的性能.
  $$\text{AUROC} = \mathbb{P}\left\{u(x_{\text{wrong}}) > u(x_{\text{correct}})\right\}$$
  - 其中 $u(x)$ 是模型对输入 $x$ 的不确定性估计（如熵值）.

> **Critical Note**:
>
> 这里有一个疑问:
> - 如我们之前讨论：CoQA 的 correctness label 是用 DeBERTa-MNLI 判定的；Semantic Entropy 也是依赖这个 NLI 模型聚类构造的；所以 AUROC 分数可能过高、不能泛化.
> - 这意味着：SE 能预测"什么是正确的"，部分是因为 correctness 本身就是由它的下游模型决定的.这会让 AUROC 结果偏高而不具有 external validity.
> - 其他 baseline（如 token entropy、p(True) classifier、Rouge-L）并不依赖 DeBERTa，它们的结果是更纯粹的数据驱动的.  
> - 怀疑其中存在信息泄漏/模型共犯结构（collusion）

#### Baselines

核心指标: AUROC (Area Under the Receiver Operating Characteristic Curve), 衡量模型在识别正确回答和错误回答时的性能.

- Normalised Entropy: 计算每个 token 的概率分布熵, 作为不确定性度量. 并且通过归一化处理, 使其在 0 到 1 之间. $\hat H(\mathbf{p}) = - \sum_{i=1}^{K} p_i \log p_i / \log K$, 其中 $K$ 是分布的类别数， 也即熵的最大值. (ref: arxiv.org/abs/2002.07650)
- p(True) Classifier: 使用专门训练的 LLM 判断回答是否为"正确回答".
- Lexical Similarity (Average Rouge-L): 计算所有候选回答中与其它回答最相似的得分, 反映回答之间的 lexical overlap.


#### 实验流程

1. 一批输入问题 $x$ (TriviaQA 或 CoQA)；
2. 对每个问题，使用 LLM 生成 $M$ 个回答样本 $s_1, s_2, ..., s_M$.
3. 对于每个输入输出对 $(x, s_i)$, 计算其语义熵 $\text{SE}(x)$ 或其他 baseline 的对应分数.
4. 同时对于每个输入输出计算其是否为正确回答（TriviaQA 的 exact match 或 CoQA 的双向蕴含判断）, 并通过前述算法 label 其为正确或错误.
5. 最后计算 AUROC 值，比较不同方法的性能.

#### 实验结果

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250604220523.png)


上述文章还有一个后续 follow-up 工作:

- [Detecting hallucinations in large language models using semantic entropy (Nature, Vol 630, June 2024)](https://www.nature.com/articles/s41586-024-07421-0)




### Motivation

LLMs 在生成文本时会出现 hallucination（幻觉）现象，即生成的内容与事实不符或虚构信息. 其中一种常见的幻觉为 **confabulatory hallucination**，文中定义为: 对同样的 prompt 输入, 模型在不同采样种子下输出语义不一致 (且往往错误) 的回答 (*LLMs fluently make claims that are both wrong and arbitrary—by which we mean that the answer is sensitive to irrelevant details such as random seed*) . 

例如, 在反复提问 `Where is the Eiffel Tower?` 时, 模型可能会生成不同的回答, 如 `Paris`, `It's Paris`, `It's Rome`, `Berlin`, 等等. 这其中可能存在正确的答案, 也可能没有. Confabulation 强调的主要是模型的回答的不一致性和随机性.

在生成时, 针对同一个输入可能会生成多个不同但甚至互斥的回答, 而模型内部并没有明确的信心指定哪一个回答是正确的. 这说明模型是 **semantic uncertain** 的, 即对同一输入的不同输出在语义上存在不一致. 因此作者希望通过 semantic entropy 来检测这种幻觉现象.

![Fig.1 Confabulatory Hallucination 现象与 Naive/Semantic Entropy 的示意图](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250604213204.png)

### Methodology: Semantic Entropy

文章的方法与前文的 Semantic Entropy 类似:

1. **抽样生成 (Sampling Generation)**: 对于每个输入 $x$, 使用 LLM 生成 $M$ 个样本 $\{s_1, s_2, ..., s_M\}$.


2. **语义聚类 (Bi-directional Entailment Clustering)**: 使用 NLI 的分类器, 如果两个样本 $s_i, s_j$ 满足 $\text{Entail}(s_i, s_j) \land \text{Entail}(s_j, s_i) = \text{True}$, 则将它们聚类到同一个语义类 $c_k$. 这样可以得到若干 disjoint 语义类 $\mathcal{C} = \{c_1, c_2, ..., c_K\}$.
   - 分类器既可以是经过微调的 NLI 模型 (如 DeBERTa-Large-MNLI), 也可以是更 advanced 的通用 LLM 模型 (如 GPT-3.5).

3. **Entropy Estimation**: 计算每个语义类的概率分布 $\mathbb{P}(c_k|x) \approx \frac{1}{M} \sum_{i=1}^{M} \mathbb{I}(s_i \in c_k)$, 然后计算语义熵:
   $$
   \text{SE}(x) = -\sum_{i=1}^{K} \mathbb{P}(c_k|x) \log \mathbb{P}(c_k|x)
    $$

若计算出较高的 semantic entropy, 则说明概率的分布较为分散, 即模型对该输入的回答存在较大的不确定性, 可能存在 confabulatory hallucination.

### Experiments

#### 数据集

作者设计了实验来覆盖两类典型任务:
1. QA 任务: 关注短文本, 事实性回答
   - SQuAD v1.1: 封闭式问答任务, 来源于维基百科
   - TriviaQA: 开放域问答 (基于多文档的推理)
   - NQ-Open: Google 搜索日志的开放问答
   - BioASQ: 生物医学问答, 包括信息检索与抽取
   - SVAMP: 小学数学文字题 (包含算数与推理)
2. 长文本生成任务: 关注真实背景下连续语言输出的 factual consistency
   - FactualBio: 输入真实人物名, 用 LLM 生成对应的生平事迹. 每句话被人工标注为 True/False/Unverifiable, 用以验证句子级别的 semantic entropy 是否与 hallucination 概率一致.

![BioASQ 任务示意图](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250604215943.png)

#### 基座模型与采样策略

| 模型               | 类型 | 参数量          | 获取方式         |
| ---------------- | -- | ------------ | ------------ |
| LLaMA-2 Chat     | 开源 | 7B, 13B, 70B | Meta         |
| Falcon Instruct  | 开源 | 7B, 40B      | TII          |
| Mistral Instruct | 开源 | 7B           | Mistral      |
| GPT-4            | 封闭 | 不公开          | OpenAI（调用接口） |

采用多种采样策略:
- Top-p: $p=0.9, T=1$
- Top-k: $k=50, T=1$
- Temperature: $T=0.1$ 作为 'best generation' of the model to the context

#### Baselines

- Token Entropy: 计算每个 token 的概率分布熵, 作为不确定性度量.
- P(True): 使用分类器预测回答为真的概率, 作为不确定性度量.
- Embedding Regression: 使用语义嵌入训练回归模型预测回答的正确概率.
- MC Dropout: 使用 dropout 近似不确定性, 通过多次前向传播计算熵.


#### 评估指标与实验结果

- AUROC: 在 binary classification 任务中，衡量模型在正确回答和错误回答上的区分能力. 完全可分时 $\text{AUROC} = 1$, 随机猜测时 $\text{AUROC} = 0.5$.
- AURAC: 用于 selective answering. 构造一个 rejection threshold $\tau$, 若 entropy 过高则拒绝回答. AURAC 衡量在不同阈值下拒绝回答后的剩余准确率变化.

![Semantic entropy outperforms leading baselines and naive entropy (average over five datasets)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250604220010.png)


## Distributional Self-Certainty

- [Scalable Best-of-N Selection for Large Language Models via Self-Certainty (arXiv 2024)](https://arxiv.org/abs/2502.18581)

### Motivation

- 在推理任务中，Best-of-N sampling（从 N 个生成中选最优）是提升大语言模型（LLMs）输出准确性的常用方法.

- 目前主流做法使用 reward models（如 ORMs 和 PRMs）对候选输出进行打分选择，然而：
  - 训练成本高；
  - 对训练分布变化敏感；
  - 容易被"reward hacking"利用；
  - 无法很好泛化到开放式生成任务.

> 注意区分这里的 Sampling 策略和诸如 Beam Search, Top-p 等搜索方法:
> - 前者属于 output-level reranking/selection 方法, 是对 decoding 之后的样本进行挑选; 是一种后处理技术
> - 后者是 token-level 在序列生成中的一种选择策略.

- 因此作者这里提出了一种基于分布的 Self-certainty 计算框架. 

### Methodology

#### Self-Certainty

在最开始的介绍部分, 我们提到模型关于模型输出内容的输出逻辑, 其中模型每一步的 token 输出都是参照着一个分布 $p_i = p(\cdot | x, y_{\lt i})$ 进行的. 作者通过这种输出分布构建了一个模型输出的 certainty 的衡量框架:
$$\text{Self-Certainty} = F\left(f_1(p_1), f_2(p_2), ..., f_n(p_n)\right)$$
- $f_i(p_i)$ 是第 $i$ 步的 distributional confidence score, 可以是例如 KL, entropy 等
- $F$ 是一个整体的整合函数, 例如求均值等.

具体地, 可以是:

$$\text{Self-certainty} = -\frac{1}{nV} \sum_{i=1}^{n} \sum_{j=1}^{V} \log \left( V \cdot p(j \mid x, y_{\lt i}) \right)$$

或

$$\text{Self-certainty (CE)} = -\frac{1}{nV} \sum_{i=1}^{n} \sum_{j=1}^{V} \log p(j \mid x, y_{\lt i})$$
- 在每个时间步 $i$ 上, 遍历所有可能的 token $j$ (即词汇表 $\mathcal{V}$ 中的每个 token), 计算其概率分布 $p(j \mid x, y_{\lt i})$ 的对数并求和求平均.

---

![Negative Perplexity (Sentence-level 的似然值) 与 Self- certainty 在推理任务上的评价分数区别](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250611144724.png)

上图所示是一个 QA 数学推理问题的例子. 其中 LLM 模型一共生成了两个样本, 第一次其第一步就出现错误, 没有能够得到正确答案; 第二次则是正确的进行了推理求解.

- 如果采用 negative perplexity, 即 
  $\text{PPL} = \exp\left(-\frac{1}{n} \sum_{i=1}^n \log p(y_i | x, y_{\lt i})\right)$ (其本质即为这个句子采样出的似然值). 它只是对模型实际采样出来的 token 的概率做平均. 如果前面出错，但后面每一步都采样到了高概率 token，总体分数仍可能很高. 从结果来看, negative perplexity 也没能很好的区分这两次的 sampling 结果.
- 如果采用 self-certainty, 即 $\text{Self-Certainty} = -\frac{1}{nV} \sum_{i=1}^n \sum_{j=1}^V \log(V \cdot  p(j | x, y_{\lt i}))$, 则可以很好地区分出两次的结果. 这是因为如果某一步模型分布很发散（不 confident），即使采样到了高概率 token，它也会被惩罚；并且由于语言模型自回归的特性, 一旦前面生成出错 token，后续的输出往往也变得相对更为不 confident. 这也可以被 self-certainty 更好的捕捉到.


#### Self-Certainty with Borda Voting Method

尽管 self-certainty（基于 KL divergence 与 uniform 分布的分布式置信度指标）比常规平均 log 概率或 perplexity 更能区分对错，但它本身仍存在被局部高置信样本"欺骗"的风险（即某些错误但具有虚假高置信度的样本可能被选中）.而传统的 majority voting 又不考虑生成样本间置信度差异，因此容易受到频率失衡或"等票"问题的影响. 因此这里提出 **Self-Certainty** + **Borda Voting** 策略.

具体流程为:

1. **Self-Certainty 计算与排序**: 对同一个输入 $x$，使用 LLM 生成 $N$ 个平行的候选输出 $y_1, y_2, \ldots, y_N$，并计算每个候选输出的 self-certainty 值 $C(y_i)$, 并根据该 self-certainty 值对候选输出进行排序, 对应于 $y_i$ 记为 $r_i$. 其中 $r_i \in \{1, 2, \ldots, N\}$, $r_i = 1$ 表示 self-certainty 最大的候选输出.
2. **Vote 权重计算**: 对于每个候选输出 $y_i$ 与对应的排序 $r_i$, 计算其投票权重:
  $$ v(r_i) = (N - r_i + 1) ^p $$
  - 其中 $p$ 是一个超参数, 控制投票权重的衰减程度. 当 $p=0$ 时, 相当于所有候选输出的投票权重相同, 为 majority voting; 当 $p\to\infty$ 时, 只有 self-certainty 最大的候选输出 $y_i$ 的投票权重为 $N$, 其他候选输出的投票权重为 0, 相当于 $\text{argmax}$ 策略.
3. **Borda Voting**: 将所有输出根据 final answer 进行分组, 同一个答案的所有候选输出的投票权重相加, 得到每个答案的总投票权重:
  $$ V(y) = \sum_{i: y_i \in \mathcal{Y}} v(r_i) $$
4. **选择最佳答案**: 最终选择总投票权重最大的答案作为最终输出:
  $$ y^* = \arg\max_{y \in \mathcal{Y}} V(y) $$

