
本文将介绍一种新的采样方法 Min-$p$ sampling, 以及近期针对该文章的一个反驳文章. 首先是其原文. 

## Turning Up the Heat: Min-$p$ Sampling for Creative and Coherent LLM Outputs 原文详细阅读

> 论文链接: [https://arxiv.org/abs/2407.01082, ICLR 2025.](https://arxiv.org/abs/2407.01082)

### 1. Introduction

当前的 LLM 在生成文本时面临的一个天然的 trade-off 在于 creativity 和 coherence 之间的平衡. 
- 现有两种主流技术:
  - top-$p$ sampling (nucleus sampling): 截取累计概率 $\geq p$ 的 token 子集;
  - temperature scaling: 通过调整 softmax 的温度参数来控制采样的多样性.
- 上述方法的局限性: creativity 和 coherence 的权衡是全局静态实现的(例如温度), 无法进行动态调整.

本文的主要贡献:
1. 提出了一种新的采样方法 Min-$p$ sampling, 该方法通过获取模型下一个 token 的概率分布中的最大概率来作为 tokne 的置信度, 并基于此置信度动态调整采样的阈值以控制采样的多样性. 使得即使在高温度下也能保持稳定输出.
2.  强调结果的泛化性和广度:
    - 任务: GPQA (https://arxiv.org/abs/2311.12022, PhD 层级的理化生问题), GSM8K (数学推理), Creative Writing (故事生成)
    - 模型: Mistral 7B/123B, LLaMA 3 70B.
3.  引入人工评估来补充自动化指标, 以更全面地评估生成文本的质量.
4.  提供工程部署的经验指导.
5.  社区集成: 该方法已被集成到 HuggingFace 的 Transformers, vLLM 等主流库中.


### 2. Related Work

主流的采样方法:
- Greedy Decoding & Beam Search
- Stochastic Sampling
  - Temperature Scaling
  - Top-$k$ Sampling
- Top-$p$ Sampling
- Entropy-Based Sampling 
  - $\eta$-sampling: 通过输出分布的信息熵的某种函数来作为抽样截断的阈值依据.
- Dynamic Threshold Methods


### 3. Methodology: Min-$p$ Sampling

#### 3.1 Overview

动态抽样的核心思想在于通过模型的置信度来调整采样的多样性.
- 置信度低: 模型不确定, 保留更多的选项, 增加创造力.
- 置信度高: 模型确定, 筛选掉低概率选项, 保持连贯性.

记当前时间步 $t$ 下, vocabulary 为 $\mathcal{V}$, 模型对于下一个 token $x_t$ 的预测概率分布为 $P(x_t|x_{1:t-1})$. 整体计算流程如下. 
1. 计算模型的最大概率作为置信度:
   $$ p_{\max} = \max_{x \in \mathcal{V}} P(x|x_{1:t-1}) $$
2. 计算动态阈值: 给定基础概率 $p_{\text{base}} \in (0,1]$, 得到:
   $$ p_{\text{scaled}} = p_{\text{base}} \cdot p_{\max} $$
3. 构建候选 token 集合:
   $$ \mathcal{V}_{\min} = \{ v \in \mathcal{V} : P(v|x_{1:t-1}) \geq p_{\text{scaled}} \} $$
4. 从 $\mathcal{V}_{\min}$ 中进行归一化采样:
   $$ P'(v) = \frac{P(v|x_{1:t-1})}{\sum_{u \in \mathcal{V}_{\min}} P(u|x_{1:t-1})}, \quad v \in \mathcal{V}_{\min} $$


#### 3.2 Intuition

Min-$p$ sampling 主要在于改进 top-$p$ sampling 的静态阈值问题.
![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807225452.png)

- 上表左例是一个不确定性较高的场景. 此时 $p_{\max}$ 较低, 使得 $p_{\text{scaled}}$ 也较低, 从而保留了更多的 token 选项, 增加创造力, min-$p$ 采样的效果与 top-$p$ 采样类似. 
- 右例是一个确定性较高的场景. 此时 $p_{\max}$ 较高, 使得 $p_{\text{scaled}}$ 也较高, 从而筛选掉了低概率选项, 保持连贯性, 而 top-$p$ 采样则可能保留了过多的低概率选项, 导致其他不合理的输出也有一定概率被采样到.

#### 3.3 Advantages Over Existing Methods

该方法主要有如下三点优势:

1. Creativity-Coherence 平衡
2. 在高温度下依然稳定
3. 工程友好简洁


#### 3.4 Implementation Details

**算法部署**

Min-$p$ 已经被 HuggingFace 的 Transformers, vLLM 等主流库集成. 在实现时的主要流程为:
1. 模型输出 logits.
2. temperature scaling 后进行 softmax 得到概率分布.
3. 计算 $p_{\text{scaled}} = p_{\text{base}} \cdot p_{\max}$ 得到 $\mathcal{V}_{\min}$.
4. 从 $\mathcal{V}_{\min}$ 中归一化采样.

这一过程只需要在推理时进行修改, 不需要额外的训练, 且支持向量化计算, 对计算效率影响较小.


**参数选择**

$p_{\text{base}}$ 是这里唯一重要的超参数. 对于其选择:
- 经验上 $p_{\text{base}}$ 取值在 $[0.05, 0.1]$ 之间效果较好.
- 更高的 $p_{\text{base}}$ (例如 $0.5\sim0.7$, 甚至趋近于 $1$) 会使得采样更接近 greedy decoding.

温度选择:
- Min-$p$ 对于温度的选择较不敏感.
- 即使是在高温 (如 $\tau = 2, 3$) 下该方法仍能控制生成不坍塌, 避免 incoherent 输出.

与其他方法的结合:
- 可以与 repetition penalty, sampling penalty 等方法结合;
- 不推荐与其他截断机制, 如 top-$k$, top-$p$ 等结合. 双重阶段可能会导致归一化问题, 并且各自超参分别调整较为复杂.

**社区接纳**

- Min-$p$ 已获得超过 667000 stars, 290000 多的相关仓库依赖.

- DeepSeek R1 ( https://arxiv.org/abs/2411.09661) 在实现中也引入了 Min-$p$ 采样.

### 4. Case Studies: Illustrative Examples

与上表中所示的例子类似, 依然在强调在不确定场景 (Case 1) 和确定场景 (Case 2) 下, Min-$p$ 采样的优势. 其能够在 exploration 和 exploitation 之间取得更好的平衡.

具体例子略. 

### 5. Experiments


#### 5.1 Experimental Setup

模型选择
- 主要模型为 Mistral 7B
- 在 LLaMA-3 70B, Mistral Large (123B) 上也进行了可扩展性验证. 

数据集任务
- 逻辑推理: GPQA, 基于准确率打分
- 数学推理: GSM8K CoT, 基于准确率打分
- 创意写作: Creative Writing, 采取 LC-Win Rate 由 Evaluator 自动打分

对比模型: 
- $\star$ min-$p$ sampling ($p_{\text{base}} = 0.05$ and $0.1$)
- Top-$p$ sampling ($p = 0.9$)
- Temperature scaling
- $\eta$-sampling
- $\epsilon$-sampling
- Mirostat sampling
  
温度设置主要在 $0.7\sim 3.0$ 之间, 还尝试了 $0\sim 5$ 的更极端范围. 其中, 在 $0\sim 0.5$ 范围, min-$p$ 与 top-$p$ 采样的效果相似.

#### 5.2 Results

**GPQA 主任务**

- GPQA 是一个多选 benchmark, 难度较高.
- 使用 5-shot prompting, 每个测试样本前加入 5 个已知 QA 样本作为上下文增强. 
- 模型采取 Mistral 7B


Table 2 左侧是在不同温度下, 不同采样方法的表现.
![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807224756.png)

- min-$p$ 在所有温度下皆为最优
- 优势在 $\tau \geq 1.5$ 时更为明显, 其他方法崩溃而 min-$p$ 仍能保持较高的准确率.

该任务还在 Mistral Large 上进行了验证, 结果与 Mistral 7B 类似, 见下表 Table 3(a).

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807224855.png)

下表是特别关注在较低温度下 ($0\sim 0.5$) 的表现. 可以看到, 在这个温度范围, min-$p$ 采样的效果与 top-$p$ 采样相似.
![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807224612.png)


**GSM8K CoT 主任务**

- GSM8K CoT 是一个数学推理 benchmark.
- 采取 8-shot prompting 来采样中间的推理步骤.
- 模型采取 Mistral 7B

Table 2 右侧是在不同温度下, 不同采样方法的表现.
![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807224756.png)

- 从表中可见, 在大多数温度下, min-$p$ 采样的效果均优于其他方法.

此外在 Appendix Table 8 中还给出了在一系列不同大小的 base model 下两个数据集的表现. 

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807224945.png)
![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807225009.png)

在小温度下的测试如下表 Table 11:

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807224555.png)

关于计算效率:
- min-$p$, top-$p$ 和 top-$k$ 的计算复杂度相似. 即使在高温下, min-$p$ 采样的计算效率也未发生退化. 
- $\eta$-sampling 和 $\epsilon$-sampling 需要额外的 entropy 计算, 在 $\tau \geq 1.5$ 时计算超时失效.

**Creative Writing 主任务**

- 采用 AlpacaEval Creative Writing 数据集
- 评测机制: LLM-as-a-judge, 通过 GPT-4 Turbo 作为评测模型
- 底层模型: openchat-3.5-0106
- 评分指标:
  - Win Rate: 与 baseline (top-$p$) 进行对比, 该模型被认为更好的比例;
  - LC-Win Rate: 在 Win Rate 的基础上, 进一步考虑生成文本的长度, 以避免 bias.


实验结果如下表 Table 3(b) 所示:
![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807224855.png)


此外, 这里还引入了 **EQ-Bench Creative Writing** 作为补充. 
- 该数据集包含一系列写作的 prompt, 旨在要求模型创作复杂情节, 角色或写作风格. 
- 此外还配套包含一个评测模型, 该模型基于一些高质量的 LLM, 针对一系列详细的评分标准进行评估, 其包含了: 故事的连贯性, 角色的深度等. 


min-$p$ 的整体平均表现优于 top-$p$ 采样, 见下表 Table 16:
![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807225230.png)


在多种温度范围内, 以及在不同的基座模型上, min-$p$ 的表现如下 Table 17. 在高温下, 两个模型均没有太强的表现. 
![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807225253.png)

下表 Table 18 进一步给出了一系列设置下的对比结果. 
![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250807225321.png)

#### 5.3 Ablation Study


### Appendix

#### A.2 Hyperparameters Settings (Selection Methodology)