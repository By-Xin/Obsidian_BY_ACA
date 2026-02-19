## Introduction



我们的任务分为两部分: pre-training 与 fine-tuning, 各自的目标如下. 

**Pre-training**

- 类似于 BERT 的 Masked Language Model (MLM) 任务, 我们将随机 mask 掉$\boldsymbol{X}$ 中的一些行 (即部分 region), 记为 $X^{\text{masked}} = \{ \boldsymbol{x}_i = \textbf{m}\mid i\in \mathcal{I}_M \}$, 其中 $\mathcal{I}_M \subset \{1,\cdots,N\}$ 表示被随机选中的行的索引集合, $\textbf{m}\in \mathbb{R}^{d}$ 为 mask 后的特征表示, 也是一个 learnable 的向量. 
- Pre-training 的最终目标就是尽可能地还原被 mask 的部分, 即最小化以下损失函数:
$$\mathcal{L}_{\text{pre}} = \sum_{i \in \mathcal{I}_M} \|f_\Theta(\boldsymbol{X}^{\text{masked}}) - \boldsymbol{x}_i\|^2$$



## Model Architecture 

给出问题定义. 
- 对于一个基因 $g$ 在细胞类型 $c$ 中, 其调控区域窗口 (长 2 Mbp) 可检测到 $N$ (本文 $N = 200$) 个 region / peak, 记为 $\{r_i\}_{i=1}^N$. 
- 对于每个 region $r_i$, 我们会对 motif 进行分析, 得到 motif 打分 $\boldsymbol{m}_i \in \mathbb{R}^{d_m}$, 其中  $d_m = 282$ 为 motif 的维度. 
- 对于该 region, 还对应了其染色质可及度 (chromatin accessibility) 的信息, 记为 $a_i \in \mathbb{R}$, 其通过 scATAC-seq 实验测得, 用 logCPM 计数. 
- 将上述 $\boldsymbol{m}_i$ 和 $a_i$ 进行拼接, 得到该 region 的特征表示 $\boldsymbol{x}_i = [\boldsymbol{m}_i, a_i] \in \mathbb{R}^{d}$, 简记 $d := d_m + 1$.
- 在上述 2 Mbp 内全部 $N$ 个 region 都这样编码, 最终得到输入特征矩阵 
  $${X} = \begin{bmatrix} \boldsymbol{x}_1^\top \\ \boldsymbol{x}_2^\top \\ \vdots \\ \boldsymbol{x}_N^\top \end{bmatrix} \in \mathbb{R}^{N \times d}.$$

### RegionEmb

首先将上述的特征矩阵 ${X}$ 输入到一个 RegionEmb 模型中通过线性变换 (无activation) 进行升维:
$$X' := \text{RegionEmb}(X) = X W_{\text{Emb}} \in \mathbb{R}^{N \times D}$$
其中 $W_{\text{Emb}} \in \mathbb{R}^{d \times D}$ 为线性变换的权重矩阵, 文中 $D = 768$.
- 文中指出, 由于原始向量中 motif 的相关性较高, 因此不希望过早引入非线性变化产生压缩/干扰.



### Token-wise Self-Attention

为了建模 peak 间的调控关系 (包括 cis-interaction 和 trans-interaction), 作者采用 12 层 Transformer 的 Encoder, 对所有 $\boldsymbol{h}_i \in \mathbb{R}^{768}$ 进行 token-wise self-attention.

其具体思路与标准 Transformer 相同. 
- 通过引入可学习线性变换 $W_q, W_k \in \mathbb{R}^{D \times d_k}, W_v \in \mathbb{R}^{D \times d_v}$, 计算 $Q = X' W_q, K = X' W_k, V = X' W_v$.
- 得到当前 head $h$ 的输出为 $O_h = \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)V \in \mathbb{R}^{N \times d_v}$.
- 并行地计算所有的 head 并进行拼接与全链接, 最终得到 Transformer 的输出. 
- 此外模型在设计时还包括常见的例如残差连接 (residual connection) 和层归一化 (layer normalization), 以及全连接层 (feed-forward layer) 等. 

## Training Procedure

### Pre-training

在输入 $X$ 中, 随机 mask 掉部分 region (这里设计的为随机抽样一半的 region), 记被 mask 的部分的指标集为 $\mathcal{M} \subset \{1,\cdots,N\}, |\mathcal{M}| = \frac{N}{2}$. 对于被 mask 的部分, 将令 $\boldsymbol{x}_i = \boldsymbol{m}$, 其中 $\boldsymbol{m} \in \mathbb{R}^{d}$ 为一个 learnable 的向量. 将这样一个经过 mask 处理的输入样本记为 $X^{\text{masked}}$.

接着将 $X^{\text{masked}}$ 输入到上述模型中进行训练:
1. 经过 RegionEmb 模型, 得到 $H^{\text{masked}} = \text{RegionEmb}(X^{\text{masked}}) = X^{\text{masked}} W_{\text{Emb}} \in \mathbb{R}^{N \times D}$.
2. 经过 Transformer 模型, 得到 $Z^{\text{masked}} = \text{Transformer}(H^{\text{masked}}) \in \mathbb{R}^{N \times D}$.
3. 对于每个被 mask 的区域 $r_i, i \in \mathcal{M}$, 用一个线性层恢复其表示: $\boldsymbol{\hat{x}}_i = \boldsymbol{z}^{\text{masked}}_i W_{\text{dec}} \in \mathbb{R}^{d}$, 其中 $W_{\text{dec}} \in \mathbb{R}^{D \times d}$ 为解码的权重矩阵.
4. 计算损失函数: $\mathcal{L} = \mathbb{E}\left[ \sum_{i \in M} -\log \mathbb{P}(x_i \mid X^{\text{masked}}) \right]$ . 在实现上通常为 $\mathcal{L} = \sum_{i \in M} \| \hat{x}_i - x_i \|_2^2 \quad \text{or} \quad \sum_{i \in M} \mathrm{BCE}(\hat{x}_i, x_i)$. 

抽象地讲, 我们首先引入 Transformer 的 Encoder (记为 $p$), 将输入 $X$ 映射为 latent representation $H$. 然后再通过 prediction head. $g_\theta$ 还原回 $\hat X$:
$$X \xrightarrow{p} H \xrightarrow{g_\theta} \hat{x}$$

优化目标为:
$$\min_{p, g} \ \mathbb{E}_{X, \mathcal{M}} \left[ \sum_{i\in\mathcal{M}} \left\| g_\theta(p(X))_i - x_i \right\|^2 \right]$$



文中该 pre-training 阶段的训练参数为:
- 优化器: AdamW, with weight decay 0.05. 
- Batch Size: 256
- Epochs: 800, with 40 linear warmups.
- Max Learning Rate: 1.5\times 10^{-4}
- Training time: 7 days on 16 V100 GPUs.

## Fine-tuning

当我们通过大量的 masked pre-training, 我们会得到一个表现效果良好的 latent representation 函数 $p$. 因此在 fine-tuning 阶段, 我们希望借用 $p$ 的知识来提升下游任务的性能. 

具体地, 这里的任务是: 给定一个基因 $g$ 在细胞类型 $c$ 中的调控区域 $X$, 预测其 RNA 表达水平 $\hat{y}$. 我们的输入仍然是 $\boldsymbol{x}_i = [\boldsymbol{m}_i, a_i] \in \mathbb{R}^{d}, \quad d = d_m + 1 = 283$, 拼接得到 $X = [x_1^\top; \dots; x_N^\top] \in \mathbb{R}^{N \times d}$.

复用我们预训练好的模型 $p$, 先将 $X$ 映射到潜在表示空间 $H = p(X) = [h_1^\top; \dots; h_N^\top] \in \mathbb{R}^{N \times D}$

由于我们有 $N$ 个 region, 我们需要将其聚合成一个全局表示. 具体地, 我们通过如下方法进行加权求和 (称为 attention pooling):
$$z = \sum_{i=1}^N \alpha_i h_i \in \mathbb{R}^D, \quad \text{where } \alpha_i = \frac{\exp(w^\top h_i)}{\sum_{j=1}^N \exp(w^\top h_j)}$$
- 其中 $w \in \mathbb{R}^D$ 是一个 learnable 的向量, 用于计算每个 region 的权重 $\alpha_i$.

最终将 pooling 后的全局表示 $z \in \mathbb{R}^D$ 通过 MLP 进行预测:
$$\hat{y} = f_\phi (z)$$
其中 $f_\phi$ 是一个多层感知机 (MLP) 模型, 其参数为 $\phi$, 为微调阶段的最主要模块. 

最终我们的优化目标为:
$$\min_{\phi, \theta} \sum_{(g,c)} \mathcal{L}_{\text{expr}}\left(f_\phi(p_\theta(X_{g,c})), y_{g,c}\right)$$
- 其中 $(g,c)$ 强调了基因 $g$ 和细胞类型 $c$ 的组合, 使得模型能够针对特定的基因和细胞类型进行优化.
- $\mathcal{L}_{\text{expr}}$ 是表达水平预测的损失函数, 文中定义为 Poisson Negative Log Likelihood (NLL) 损失, 适用于计数数据的回归任务. 部分任务也被设置为 MSE. 