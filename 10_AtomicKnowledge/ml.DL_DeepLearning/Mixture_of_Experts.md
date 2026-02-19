---
aliases: [MoE, 混合专家模型, Mixture of Experts]
tags:
  - concept
  - ml/deep-learning
  - ml/architecture
related_papers: 
  - "[[Switch Transformer]]"
  - "[[DeepSeek MoE]]"
---

# Mixture of Experts

> Refs:
> 1. CS336 (2025) Lecture 4: Mixture of Experts (MoE) https://stanford-cs336.github.io/spring2025/
> 2. 【【文献梳理】混合专家模型 MoE：从基础到前沿】 https://www.bilibili.com/video/BV1ZQXHYwEqJ

  
## What is MoE?  

  

Mixture of Experts (MoE) 是一种模型架构, 旨在通过使用多个专家模型来提高模型的表达能力和性能. 其在当下相较于稠密的模型架构得到了更广泛的应用.

  

![MoE 的一个典型结构示意图 (arxiv.org/pdf/2101.03961)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250629130644.png)

  

- 典型的 MoE 在结构上引入了若干子模型 (称为 experts), 这些 experts 本质上就是将原先的一个大的前馈神经网络 (FNN) 替换为多个不同的子 FFN, 加上一个选择机制 (router) 来决定在每次前向传播时使用哪几个 experts.

  

- 通常而言 MoE 中的 experts 都是稀疏激活的, 也就是每次只有少数几个 experts 被 router 选中使用. 这使得我们可以引入更多的参数 (experts) 而不会显著增加实际的计算量 (因为每次只激活使用少数几个 experts). 不过确实需要承认我们引入了更多的参数, 这会导致模型的存储开销增加.

    ![实验指出, 在相同的运算量下, MoE 往往能达到更好的测试结果 (arxiv.org/pdf/2101.03961)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250629131910.png)

  

- MoE 还可以被分发到多个设备上进行并行计算 (expert parallelism), 这使得它在大规模分布式训练中非常有用.

- 不过 MoE 是一个非常复杂的模型架构, 其优化等方面也有很多挑战.

  

## Key Components of MoE

  

在 MoE 的设计中, 有如下几个关键组件:

- Routing Function: 我们如何设计 router 来选择哪些 experts 被激活.

- Expert Models: 我们该如何设计每个 expert 模型, 其大小和结构如何.

- Training Optimization: 我们不能将所有的 experts 都激活, 这会导致计算量过大. 因此我们需要设计一些优化方法来训练 MoE 模型.

  

### Routing Function

  

Routing 本质上相当于在对 token 进行选择性处理. 对于一批输入 token, 不是所有的 experts 都要参与到对 token 的处理中. 只有部分的 router 会对部分的 token 进行处理计算. 具体而言有三种可能的选择方法 (假设有 token (严格说是 token 的 hidden states) $T_1, T_2, T_3$ 和 experts $E_1, E_2, \cdots, E_5$):

- Token 选择 Expert: 每个 token 都会选择 top-k 个 experts 来处理. *(该方法通过一些 ablation 实验被证明最为有效.)*

-  Expert 选择 Token: 每个 expert 都会选择 top-k 个 token 来处理.

-  混合选择: 通过全局优化进行 token 和 expert 的分配.

  

![整体而言有三种选择方法. (arxiv.org/pdf/2101.03961)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250629133640.png)

  

$\diamond$

  

具体而言, 一个 Token 选择 Expert 的具体结构如下.

- Top-k Routing 在本质上有点类似于 attention 机制, 其通过与输入 token 的 hidden states 进行内积计算 (引入一个可学习的线性层 $W$) 并再进行 softmax 得到一个概率分布, 然后选择 top-k 个 experts 且根据 softmax 的概率作为加权进行求和.

- 一个更有趣的发现是, 即使是通过 Hashing 进行选择 (对语义没有任何信息的处理) 分配, 也能得到不错的效果.

- 还有其他的一些对于 Routing 的设计, 例如通过 RL 或一些其他优化算法进行分配.

    ![Top-k Router 与 Baseline Hashing 的对比. (arxiv.org/pdf/2101.03961)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250629135619.png)

  
  

$\diamond$

  

Top-K Routing 也是现在最常用的方法. 其具体的计算过程如下 (这里参考的是 DeepSeek (V1-2), Grok, Qwen 等模型的设计. Ref: arxiv.org/pdf/2401.06066) :

  

- 首先对于第 $t$ 步, 第 $l$ 层的输入 $\mathbf{u}_t^l$, 引入一组可学习的 router 参数 $\mathbf{e}_i^l \in \mathbb{R}^d$. 通过内积并经过 softmax 表示输入与第 $i$ 个 expert 的关联性:

    $$ s_{i,t} = \text{softmax}_i \left({\mathbf{u}_t^l}^\top  \mathbf{e}_i^l\right)$$

- 对于得到的各个 expert 的得分, 保留前 $k$ 大的 expert, 记为一个门控 $g_{i,t} = \boldsymbol{1}_{\{s_{i,t} \text{ is top-k largest}\}}\cdot s_{i,t}$.

- 将 $\mathbf{u}_t^l$ 作为输入传入 experts (假设一共有$N$个 experts, 每个 experts 都是一个 FFN), 并且按照门控得分 $g_{i,t}$ 进行赋权求和, 最后再进行一次残差连接, 得到通过 experts 后的隐藏表示 $\mathbf{h}^l_t$:

    $$\mathbf{h}^l_t = \sum_{i=1}^N \left( g_{i,t} \cdot \text{FFN}_i(\mathbf{u}^l_t) \right) + \mathbf{u}^l_t$$

> 稍有不同的是, 在 DeepSeek V3 等版本中, 其会在进行过 Top-k 的选择后再进行 softmax.

  

### Expert Models

  

最近, DeepSpeed MoE 等还进一步提出了 fine-grained experts 以及搭配 shared experts 的思路. 该模式也在 Qwen 和 DeepSeek 等开源模型中得到广泛运用.

![Fine-grained 与 Shared MoE 结构示意图. (arxiv.org/pdf/2401.06066)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250629145419.png)

- 原本的 vanilla MoE 中, 各个 experts 的 FFN 一般而言是和原始非 MoE 架构的 LLM 大小相同的.

- 而改进的 fine-grained MoE 试图减小每个 experts 的规模, 而增大总的 experts 数量.

- 更进一步地, 我们还可以保留部分公共的 experts, 其对于任何任务都是被激活的. 这样可以使得一些 common sense 的处理参数得到更好的应用.

上述架构的优越性也在 DeepSeekMoE 中通过一系列 ablations 得到体现:

  

![GShard 是一个标准的 baseline. 通过 fine-grind 和 shared expert 的模型的表现会得到明显提升.  (arxiv.org/pdf/2401.06066)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250629145514.png)

  
  
  

目前的主流 LLM 中的 MoE 配置情况如下表所示:

  

| Model          | Routed | Active | Shared | Fine-grained ratio |
|----------------|--------|--------|--------|---------------------|
| GShard         | 2048   | 2      | 0      |                     |
| Switch Transformer | 64     | 1      | 0      |                     |
| ST-MOE         | 64     | 2      | 0      |                     |
| Mixtral        | 8      | 2      | 0      |                     |
| DBRX           | 16     | 4      | 0      |                     |
| Grok           | 8      | 2      | 0      |                     |
| DeepSeek v1    | 64     | 6      | 2      | 1/4                 |
| Qwen 1.5       | 60     | 4      | 4      | 1/8                 |
| DeepSeek v3    | 256    | 8      | 1      | 1/14                |
| OlMoE          | 64     | 8      | 0      | 1/8                 |
| MiniMax        | 32     | 2      | 0      | ~1/4                |
| Llama 4 (maverick) | 128  | 1      | 1      | 1/2                 |

  

### Training of MoE

  

由于 MoE 本身的参数量非常大, 我们并不能将所有的 experts 都激活. 而这种稀疏激活的方式会在微分计算时带来麻烦. 可能的解决方案包括:

- **通过 RL 类方法进行直接优化**: RL 是最直接的一个解决思路. 但是一些实验发现其并没有展现出过强的优势.

- **随机探索策略**: 应用类似退火等随机探索策略进行优化.

- **通过 Heuristic 等启发方法设计损失函数进行优化.**

  

$\diamond$

  

现阶段更常用的是第三种方法. 在 Switch Transformer (arxiv.org/pdf/2101.03961) 中, 其损失函数设计如下.

  

Switch Transformer 中采用的是 token 选择 expert 的方式, 且只保留 Top-1 的 expert. 其具体的计算过程如下.

  

假设当前的 batch $\mathcal{B}$ 中有 $|\mathcal{B}| = T$ 个 token, 及给定 $N$ 个 experts. 通过 router, 对于当前 token $x\in \mathcal{B}$, router 会计算出每个 expert 的 softmax 概率 ${p} (x)  =  [p_1(x), p_2(x), \cdots, p_N(x)]^\top \in \mathbb{R}^N$. 由于是 top-1 选择, 所以我们只需要选择概率最大的 expert, 记为 $i^* = \arg\max_{i=1,\cdots,N} p_i(x)$.

  

故可求在当前 batch 中, 每个 expert 被选择的频率 $f_i, i=1,\cdots,N$:

$$ f_i = \frac{1}{T} \sum_{x\in \mathcal{B}} \boldsymbol{1}_{\{\arg\max p(x) = i\}}$$

  

还可以求得每个 expert 的平均 softmax 得分 $P_i$:

$$ P_i = \frac{1}{T} \sum_{x\in \mathcal{B}} p_i(x)$$

  

给定一个超参数强度因子 $\alpha>0$, 可以定义如下的损失函数:

$$\text{loss} = \alpha N \sum_{i=1}^N f_i P_i

$$

  

对于该损失的优化就相当于希望每个 expert 能够被均匀地激活, 且每个 expert 的平均 softmax 得分也能保持在一个合理的范围内.

  
  

## Advance Research  on MoE

  

下面是一些 MoE 相关的前沿研究.

  

### 专家数及层数是不是越多越好?

  

*ViMoE: An Empirical Study of Designing Vision Mixture-of-Experts (arxiv.org/pdf/2410.15732)*

  

文章将 12层的 MLP 全部替换为了 experts. 而实验发现 experts 数量最多 $N=8$ 的组反而表现最差. 这可能暗示着一些专家的冗余性. 因此文章进一步也提出 shared experts 的思路.

  

![专家数量的影响(arxiv.org/pdf/2410.15732)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250630151501.png)

  
  

此外在如下图所示的 semantic segmentation 任务上, 实验发现在浅层的 MoE 中并没有展现出明显的 experts 的区别; 只有在深层的 MoE 中才能展现出 experts 的差异性. 该现象在例如 CIFAR-10 等图像分类任务上也有类似的发现.

  

![专家层数的影响(arxiv.org/pdf/2410.15732)](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250630151040.png)

  
  

文章进一步提出了一个 **Routing Degree** 的概念, 其定义如下:

$$\text{Routing Degree} = (C_N^k)^L$$

其中 $C_N^k$ 是组合数, 表示在 $N$ 个 experts 中选择 top-$k$ 个的组合数, $L$ 是 MoE 的层数. 该指标可以用来衡量 MoE 的复杂度. 其指出, 若 Routing Degree 过小则会影响模型性能, 过大则会导致冗余. 其通过实验发现在 $\text{Routing Degree} \in [32,64]$ 左右时模型性能较好.

  
  

### Soft MoE

  

*From Sparse to Soft Mixture of Experts (arxiv.org/pdf/2308.00951)*

  

其主体思想是将 MoE 中的门控机制带来的稀疏激活改为软激活, 即利用 softmax 得分进行加权求和. 不过这么做也会牺牲其稀疏性带来计算量的增加.

  

![Soft MoE 示意图](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250630152932.png)