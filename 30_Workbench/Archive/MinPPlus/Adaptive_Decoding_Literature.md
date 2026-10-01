---
tags:
  - llm/sampling
  - paper/literature_review
---

# LLM Token Sampling Strategies

传统 Sampling 策略 (如 Greedy、Beam Search, Top-k、Top-p) 等的局限性:
- 单一策略难适应多样化的任务: 实际应用中，LLM经常需要既能回答封闭事实问题，又能完成开放式创意写作。如果始终使用同一温度 T 或 Top-k，就难以兼顾不同需求.
- 保rank预测: 无论是 Top-k 还是 Top-p，其本质都没有改变预测分布的 token 排序, 模型得到的高 logits token 仍然是最可能的答案. 然而实际上, 模型的预测分布本身可能也存在着系统偏差 (如过度偏好常见词或重复短语等).


## 自适应解码策略


[Adaptive Decoding via Latent Preference Optimization (2411)](https://arxiv.org/abs/2411.09661v1)

- 该文章设计了一个神经网络模块在 LLM 最后的隐藏层上 (Adaptive Decoder) 来预测每个 token 的采样温度以动态调整采样策略. 

### Methodology

#### Adaptive Decoder (AD)

考虑大语言模型 $\mathcal{M}$ 的 $t$ 时刻的隐藏状态:
$$
h_t = \mathcal{M}(x_1, x_2, \ldots, x_{t})
$$
通过考虑温度的 Softmax 进行采样:
$$
x_{t+1} \sim \text{Softmax}\left(W h_t / \tau\right)$$
其中 $W$ 是一个线性变换矩阵，$\tau$ 是温度参数.

文章引入了一个 AD 模块 (本质上为一层 MLP), 当我们预先给定一组离散的待选温度 $\tau_1, \tau_2, \ldots, \tau_k$ 时, AD 模块会输出一个概率分布:
$$
\pi_k = \mathbb{P}(\tau = \tau_k | h_t) = \text{AD}(h_t)
$$
然后根据这个分布选择一个温度 $\tau$, 并且利用这个温度进行采样:
- Greedy 采样: $\tau^* = \arg\max_k \pi_k$,    $x_{t+1} \sim \text{Softmax}(W h_t / \tau^*)$
- Stochastic 采样: $\tau^* \sim \pi_k$, $x_{t+1} \sim \text{Softmax}(W h_t / \tau^*)$
- 期望采样 (在训练阶段): $x_{t+1} \sim \sum_k \pi_k \cdot \text{Softmax}(W h_t / \tau_k)$

此外, 根据 AD 输出的颗粒度大小, 还可以分为 Token 级别的和 Sequence 级别的 Adaptive Decoding:

- Token 级别:
    $$
    \tau_t \sim \text{AD}(h_t), \quad  x_{t+1} \sim \text{Softmax}(W h_t / \tau_t)
    $$
- Sequence 级别 (只在 prompt 最后一个 token $x_T$ 的隐藏状态 $h_T$ 上预测一次温度)
    $$
    \tau \sim \text{AD}(h_T), \quad  x_{t+1} \sim \text{Softmax}(W h_t / \tau)
    $$


#### Latent Preference Optimization (LPO)

由于温度等内容没有直接的监督信号, 没有办法直接插入到损失函数中进行反向传播. 因此我们需要设计类似 RLHF 等类似 RL 方法来对 AD 的模块进行训练. 而 DPO 又是在 RLHF 的基础上衍生出的一种偏好学习方法, 但是其略过了 RL 部分, 直接对偏好进行类似有监督的优化训练. 

***Direct Preference Optimization (DPO)*** 

在 DPO 中, 对于一组偏好数据:
- Prompt $x$
- 正样本 $y^c$ (chosen), 负样本 $y^r$ (rejected)

我们希望训练语言模型 $\pi(y \mid x)$ 使得正样本的概率更高. 因此可以设计一个损失函数:
$$
\mathcal{L}_{\text{DPO}} = - \log \sigma\left( \beta \log \frac{\pi(y^c|x)}{\pi_{\text{ref}}(y^c|x)} - \beta \log \frac{\pi(y^r|x)}{\pi_{\text{ref}}(y^r|x)} \right)
$$
- $\pi(y|x)$ 是当前模型的预测分布
- $\pi_{\text{ref}}(y|x)$ 是参考模型的预测分布 (用于 regularization). 
  - Reference Model 通常是一个 SFT 模型或一个不加 AD 的原始 LLM, 相当于一个训练的 baseline 或 anchor, 它代表了我们当前行为的“背景常识”，用于定义什么是“相对偏好更好”的样本.该模型也不会参与梯度更新, 是一个固定的参考点.
- $\beta$ 是一个超参数, 控制正负样本的偏好程度
- $\sigma$ 是 Sigmoid 函数

该损失函数会鼓励
$$
\log \frac{\pi(y^c|x)}{\pi_{\text{ref}}(y^c|x)} \gg \log \frac{\pi(y^r|x)}{\pi_{\text{ref}}(y^r|x)}$$


***Latent Preference Optimization (LPO)**

LPO 是 DPO 的一个变体.  在 DPO 中, 优化的对象依然是如何更好的输出 token 的概率分布. 但是在当前任务重, 我们要优化的不是 token 的概率分布, 而是温度的概率分布, 是一个 latent variable. 因此我们需要对 DPO 的损失函数进行修改, 使得优化的对象变成温度的概率分布.

其基本流程如下. 对于每个 input $x$:
1. **样本生成**: 通过 AD 模块生成 $N$ 个温度采样, 并分别对应生成 $N$ 个输出 $\{ (y^{(1)}, \tau^{(1)}), (y^{(2)}, \tau^{(2)}), \ldots, (y^{(N)}, \tau^{(N)}) \}$
2. **响应评分**: 对每个响应 $y^{(i)}$ 进行打分, 对应 $N$ 个分数 $\{ R^{(1)}, R^{(2)}, \ldots, R^{(N)} \}$
   - 若是 reasoning 任务 (如 GSM8K), 则可以用 ground-truth label 计算准确率
   - 若是开放性任务 (如 UltraFeedback), 则可以用 reward model (如 ArmoRM) 进行打分
3. **偏好数据构建**: 有几种不同的构建策略可以得到 $(y^c,\tau^c)$ 和 $(y^r,\tau^r)$ 的偏好对:
    - Top-1 v.s. Bottom-1 (默认策略): 取最高分对应的数据为正样本, 取最低分为负样本
    - Top-k v.s. Bottom-k (应用于 constrained stories 等需要兼顾两套得分标准的任务): 取第一个得分标准中的前 $k$ 大的响应中, constrained 得分最高的作为正样本; 取第一个得分标准中的后 $k$ 小的响应中, constrained 得分最低的作为负样本
    - Pairwise (理论方向, 但本文没有实现): 对每一对响应两两进行比较, 生成正负样本对
4. **偏好优化**: 对于偏好对 $(y^c,\tau^c)$ 和 $(y^r,\tau^r)$, 同样有如下三种不同的方式构建偏好对. 
    - Temperatures as Tokens
      - 将温度 $\tau$ 视为一个特殊的 token, 认为 $\mathbb{P}(y,\tau) = \mathbb{P}(y|\tau) \cdot \mathbb{P}(\tau)$
      - 直接使用 DPO 的损失函数:
        $$\begin{aligned}
        \mathcal{L}_{\text{LPO}} &= - \log \sigma\left( \beta \log \frac{\mathbb{P}(y^c,\tau^c)}{\mathbb{P}_{\text{ref}}(y^c,\tau^c)} - \beta \log \frac{\mathbb{P}(y^r,\tau^r)}{\mathbb{P}_{\text{ref}}(y^r,\tau^r)} \right) \\
        &= - \log \sigma\left( \beta \log \frac{\mathbb{P}(y^c)}{\mathbb{P}_{\text{ref}}(y^c)} - \beta \log \frac{\mathbb{P}(y^r)}{\mathbb{P}_{\text{ref}}(y^r)} + \beta \log {\mathbb{P}(\tau^c)}- \beta \log {\mathbb{P}(\tau^r)} \right)
        \end{aligned}$$
   - Temperatures as Tokens (Separate)
      - 将温度 $\tau$ 作为一个伪 token, 只关注温度的概率分布, 而不关注 token 的生成
        $$\mathcal{L}_{\text{LPO}} = -\log \sigma \left[ \beta \log P(\tau^c) - \beta \log P(\tau^r) \right]$$
    - Temperatures as Latents (理论最优但计算复杂)
       - 将温度视为一个 latent variable最终的响应概率为:
        $$\mathbb{P'}(y) = \sum_{\tau} \mathbb{P}(y|\tau) \cdot \mathbb{P}(\tau)$$
        - 对应的损失函数为:
            $$\mathcal{L}_{\text{LPO}} = -\log \sigma \left[
            \beta \log \frac{P'(y^c)}{P'_{\text{ref}}(y^c)} - \beta \log \frac{P'(y^r)}{P'_{\text{ref}}(y^r)}
            \right]$$


### Experiments

- 基座模型: Llama 3-8B-Instruct（参数冻结）
- 实验任务
  - 主实验: GSM8K, UltraFeedback,  Stories 数据集, 对比 Fixed Temperature 的采样结果. 对比准确率及reward. 说明: AdaptiveDecoder 自动学会不同任务对温度的偏好. 
  - 重复率控制实验: Wikitext-2 数据集, 测试 AdaptiveDecoder 是否能防止 degenerate 输出（如重复 token）. 测试发现 AD 可以自动避免 greedy 采样的重复问题. 
  - 限制创意写作: Constrained Stories 数据集. 要求每个句子必须以 ‘Ab’ 开头, 后续内容尽可能丰富. 实验发现 AD 可以成功学习局部控温策略. 
    ![Constrained Creative Writing 实验结果](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250612121653.png)
  - 多样性控制实验
  - 消融实验

## MLE 本身可能的问题

[Neural Text Degeneration with Unlikelihood Training (ICLR 2020)](https://arxiv.org/abs/1908.04319)

- 文章指出重复 token 等 degeneration 问题的根源并非是解码策略, 而是在于 MLE 的估计目标 $\log \mathbb{P}_\theta(x_t | x_{<t})$ 本身错误地提高了重复 token 和高频 token 的概率分布 

- 本文提出 Unlikelihood Training 对“不该生成”的 token 显式惩罚，通过引入 Unlikelihood Loss 来优化模型:
    $$L_t^{UL} = - \sum_{c \in C_t} \log (1 - p_\theta(c|x_{<t}))$$
    - 其中 $C_t$ 是当前时刻 $t$ 的所有不该生成的 token 集合 (如重复 token, 低频 token 等)

- 将似然的损失函数和 Unlikelihood Loss 结合起来, 得到最终的训练目标:
    $$L_t^{\text{UL-token}} = -\log p_\theta(x_t|x_{<t}) - \alpha \sum_{c \in C_t} \log(1 - p_\theta(c|x_{<t}))$$



## Contrastive Decoding 对比解码

多模型对比与融合策略提供了一种解码阶段灵活控制输出的方法：通过引入辅助模型（小型LM或分类器等）实时修正大模型的下一步分布，实现了模型输出在特定属性上的可控性，同时往往还能提升文本的整体质量。

---

[Contrastive Decoding: Open-ended Text Generation as Optimization (ACL 2023)](https://arxiv.org/abs/2210.15097)


文章提出了一个 Contrastive Decoding Score. Contrastive Decoding（CD）是一种搜索式的文本生成方法，通过比较两个语言模型（一个大模型 expert，一个小模型 amateur）的输出偏好，来实现高质量文本生成。

其基本思想是: 大模型虽强，但也有生成重复/偏题/冗余等低质量输出的倾向。小模型更容易暴露这些失败模式，因此我们可以利用 大小模型之间的对比差异来“剔除”这些问题。

### Methodology

#### Contrastive Decoding Score (CD-score)

具体目标函数如下:
$$
\text{CD-score}(x_i; x_{<i}) = \log p_{\text{EXP}}(x_i|x_{<i}) - \log p_{\text{AMA}}(x_i|x_{<i})$$

- 若某个 token $x_i$ 在大模型中概率高而在小模型中概率低, 则说明该 token 是 epxert 特有的一个高质量输出, 应当被保留.
- 若某个 token $x_i$ 在 amateur 中概率很高, 则说明其可能是一个低质量, 重复, 模版化的输出, 应当被剔除.

#### Plausibility Constraint

上述的目标有两个极端的情况:
1. False Positives：一些语义荒谬但两模型分歧极大的 token（如“NetMessage”）会被错误选中；

2. False Negatives：一些简单、语法正确但两模型都高度认同的 token（如子词“#orn”）会被误判（对比得分很低）。


因此引入约束机制: 
$$
V_{\text{head}}(x_{<i}) = \left\{ x_i \in V \mid p_{\text{EXP}}(x_i | x_{<i}) \ge \alpha \cdot \max_{w \in V} p_{\text{EXP}}(w | x_{<i}) \right\}
$$

- 其本质相当于对大模型的输出进行一个阈值过滤, 只保留概率超过某个阈值 $\alpha$ 的 token 作为候选参与比较的词库 $V_{\text{head}}$.

在实践中, 其可行性约束过程为: 
1. 可行性筛选:
   - 用 expert LM 给出当前 token 的概率分布
   - 根据阈值 $\alpha$ 筛选出 $V_{\text{head}}$ 中的候选 token
2. 对每个候选 token $x_i \in V_{\text{head}}$ 计算 CD-score
3. 通过 CD-score 作为排序依据, 使用 beam search 等进一步采样策略进行采样.


# 常用数据集、任务与评价指标

以下是将你提供的内容整理成详细、结构清晰的 Markdown 笔记：

---

# Evaluation of Decoding Strategies in Language Generation Tasks


## Task-Specific Evaluation Settings


### Open-ended Text Generation

* **典型数据集**：

  * Reddit WritingPrompts
  * OpenAI 提供的故事提示数据
* **评估方法**：

  * **人工评价**：流畅性、创意、连贯性评分
  * **多样性指标**：Distinct-N（不同 $n$-gram 的占比）
  * **混合指标**：

    * **HUSE**（Holtzman et al.）：结合人工评分与统计指标衡量质量与多样性
* **代表工作**：

  * 核采样工作引入 HUSE 与人类文本对比进行评估
  * Meister 的典型采样（Typical Sampling）：在减少重复三元组比例的同时，保持连贯性评分优良

### Summarization

* **典型数据集**：

  * CNN/DailyMail
  * XSum
* **自动指标**：

  * **ROUGE**：与参考摘要的覆盖率（但可能存在鼓励重复的问题）
* **质量关注点**：

  * 流畅性、信息涵盖度、避免重复和晦涩
* **代表工作**：

  * 典型采样在保持 ROUGE 分数的同时减少重复
  * 直接指标优化方法同时优化 ROUGE 和重复率，并通过人工评价验证信息涵盖、简洁性、流畅性上的优势

### Dialogue & Instruction Following

* **数据集**：

  * OpenAI Real Prompts（如 Self-Instruct）
  * UltraFeedback
* **评估指标**：

  * **人工偏好（A/B 测试）**
  * **奖励模型得分**（如 GPT-4）
  * **胜率（win rate）**
  * **毒性评分**（Perspective API）
* **代表工作**：

  * Adaptive Decoding：统计“自适应温度” vs “固定温度”之间的胜率，UltraFeedback 数据集上胜率皆高于 50%
  * DExperts：在 RealToxicityPrompts 上显著降低输出的毒性，同时保持自然度

### Mathematical and Logical Reasoning

* **数据集**：

  * GSM8K（数学文字题）
  * MATH、CommonsenseQA、AIME 等
* **评估方式**：

  * **准确率**（最终答案正确率）
  * 对 **推理链** 的一致性与简洁性评估
* **代表方法**：

  * Self-Consistency（多路径采样 + 多数表决）：提升 GSM8K 上 GPT-3 的 one-shot 推理准确率 10%+
  * ProRL：在 AIME 等高难度测试中刷新纪录，引入高熵策略
  * “Beyond the 80/20 Rule”：评估强化学习后 Chain-of-Thought 的长度变化，考察是否引入冗余推理

### Controlled Text Generation

* **数据集**：

  * Yelp 评论数据（正/负情感标签）
  * Persona 对话数据集
* **评估指标**：

  * **控制成功率**：如情感分类准确率
  * **内容保真度**：与原始输入一致程度
  * **流畅度**：如困惑度（PPL）
  * **多样性**：避免模板化输出
* **代表工作**：

  * DExperts：情感生成分类准确率高于 baseline，同时困惑度接近原模型
  * PPLM：使用类似指标评估

