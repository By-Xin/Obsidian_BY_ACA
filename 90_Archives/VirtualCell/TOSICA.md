## Briefing

TOSICA 的总体目标为细胞类型分类 (cell type annotation). 
- 其输入为单细胞转录组表达谱 $\mathbf{e} \in \mathbb{R}^n$ , $n$ 为基因数 (10000+). 
- 输出为预测该细胞属于哪一类型 $\hat{y} \in \{1, ..., C\}$. 其中 $C$ 是细胞类型数 (10~50). 

TOSICA基于Transformer架构，引入**生物学先验知识的mask机制**，其设计包含三层：

- Cell Embedding: 对于每个细胞的表达将其映射到一个低维度的表示空间, 其每一个维度对应一个生物的 pathway / 调控子集. 
- Multi-Head Self-Attention: 通过 attention 机制对上层的 cell embedding 与 分类 CLS 进行注意力计算, 以获得细胞类型的上下文信息.
- Cell-Type Classifier: 通过全连接层将注意力信息映射到细胞类型空间, 以实现最终的分类预测.

## Methodology

### Cell Embedding

首先给出记号. 对于每个细胞 $\mathbf{e}\in \mathbb{R}^n$ , 其中 $n$ 为基因数, 我们希望通过全链接网络将其映射成一个低维表示 
$$\mathbf{t} = \mathbf{W'}\mathbf{e}\in\mathbb{R}^k $$ 其中 $\mathbf{W'}\in \mathbb{R}^{k\times n}$ 为可学习的权重矩阵. 在该 $k$ 维空间中, 每个维度相当于一个 token, 每个 token 表示一个生物学的 pathway / 调控子集.

由于我们希望 $t$ 能够结构化的每个维度表示一个特定的生物学意义的 pathway, 因此权重矩阵 $\mathbf{W'}$ 需要引入先验知识经过特殊设计. 
- 引入一个一般的权重矩阵 $\mathbf{W}\in \mathbb{R}^{k\times n}$ 
- 根据先验知识还会构建一个 mask matrix $\mathbf{M}\in \{0, 1\}^{k\times n}$
  - 其中 $M_{ij} = 1$ 当且仅当第 $i$ 个 pathway 包含第 $j$ 个基因. 而这一知识是通过外部数据库来获取的 (此处为 Gene Set Enrichment Analysis 数据库中的 gene set 数据集)
  - 因此只有同一个 pathway 中的基因才会被映射到同一个 token 上. 这避免混合组分的不可解释性, 不过也限制了模型的表达能力.
- 对这两个矩阵进行 Hadamard 乘积 (element-wise product), 得到最终的权重矩阵 
	$$\begin{aligned} \mathbf{W'} &= \mathbf{M} \odot \mathbf{W} \\
	\mathbf{t} &= \mathbf{W'}\mathbf{e}\in\mathbb{R}^k \\
	\end{aligned}$$ 
- 为增强表现能力, 上述操作会并行地进行 $m$ 次 ($m=48$), 拼接得到  
  	$$\mathbf{T}:=\begin{bmatrix} \mathbf{t}_1 & \mathbf{t}_2 & \cdots & \mathbf{t}_m \end{bmatrix} \in \mathbb{R}^{k\times m}.$$

### Multi-Head Self-Attention

下面希望引入注意力机制开始进行分类. 

首先引入一个可学习的 dummy token $\mathbf{cls} \in \mathbb{R}^m$, 并将其拼接在 $\mathbf{T}$ 的第一行, 得到新的输入:
$$ \mathbf{I} := \begin{bmatrix} \mathbf{cls}^\top \\ \mathbf{T} \end{bmatrix} = \begin{bmatrix} c_1 & c_2 & \cdots & c_m \\ \mathbf{t}_1 & \mathbf{t}_2 & \cdots & \mathbf{t}_m \end{bmatrix} \in \mathbb{R}^{(1+k)\times m}.$$
- 这是在 Transformer 等 NLP 中的常用做法. CLS 是一个额外的, 可学习的虚拟的 token. 由于 attention 操作会输出一个和 input 相同形状的表示, 因此可以用这个 CLS 的输出作为对当前输入的全局信息的提取结果, 类似于一个当前输入的统计量, 作为后续分类的 token. 

将这个输入 $\mathbf{I}$ 送入注意力机制中. 先讨论一个 head 的情况. 整体而言, 我们有 $\mathbf{O} = \text{Attention}(\mathbf{I}) \in \mathbb{R}^{(1+k)\times m}.$ 其具体计算过程细节如下:
- 分别通过三个线性变换 $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v \in \mathbb{R}^{(1+k)\times (1+k)}$ 将输入映射到 Query, Key, Value 三个空间:
  $$\begin{aligned}
  \mathbf{Q} &= \mathbf{W}_q \mathbf{I} \in \mathbb{R}^{(1+k)\times m}, \\
  \mathbf{K} &= \mathbf{W}_k \mathbf{I} \in \mathbb{R}^{(1+k)\times m}, \\
  \mathbf{V} &= \mathbf{W}_v \mathbf{I} \in \mathbb{R}^{(1+k)\times m}.
  \end{aligned}$$
-  通过计算注意力得分来获得上下文信息:
  $$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top }{\sqrt{d_k}}\right) \in \mathbb{R}^{(1+k)\times (1+k)}.$$
  其中 $d_k = m$ 是 Key 向量的维度. 这个注意力得分矩阵 $\mathbf{A}$ 表示了不同 token 之间的相似度.
-  通过加权求和来获得每个 token 的上下文表示:
  $$\mathbf{O} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{(1+k)\times m}.$$


多个 head 的情况与上述类似, 相当于并行地进行 $H$ 次计算.
- 对于第 $h$ 个 head, 我们独立地引入可学习权重矩阵 $\mathbf{W}_q^h, \mathbf{W}_k^h, \mathbf{W}_v^h \in \mathbb{R}^{(1+k)\times (1+k)}$. 最终得到输出 $\mathbf{O}^h = \text{Attention}^h(\mathbf{I}) \in \mathbb{R}^{(1+k)\times m}.$
- 将 $H$ 个独立的输出拼接在一起, 再额外通过一个线性映射整理回原始的输入形状 $\mathbf{\widetilde{O}} = \mathbf{W}_o \begin{bmatrix} \mathbf{O}^1 & \mathbf{O}^2 & \cdots & \mathbf{O}^H \end{bmatrix} \in \mathbb{R}^{(1+k)\times m}.$

### Cell-Type Classifier

最终, 由于 attention 机制的 input 和 output 的形状相同, 因此我们提取 $\mathbf{O}$ 的第一行 (不妨记为 $\widetilde{\mathbf{cls}}$), 即输入中 $<\mathbf{cls}>$ 经过注意力机制后的输出, 作为对当前输入的全局信息的提取结果, 将这个结果通过全链接神经网络完成分类:
$$
\mathbf{p} = \text{softmax}(\mathbf{W}_c \widetilde{\mathbf{cls}})\in \mathbb{R}^{\text{nc}}.
$$
其中 $\mathbf{W}_c \in \mathbb{R}^{\text{nc} \times (1+k)}$ 是可学习的分类权重矩阵, $\text{nc}$ 是细胞类型数.

### Model Architecture

其整体的流程及其他诸如 residual connection 的细节如下图所示.

主体结构为:

![主体结构](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250815142412.png)


几个子模块的具体实现结构为:
![子模块实现结构](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250815142118.png)

### Training Details

- 数据划分
  - 用不同的 study 与 biological state 进行样本划分 (跨数据集/批次的, 而非同一来源下的简单分割).
  - 训练集中的 $30\%$ 再被划分为 validation set.
- 损失函数与优化器:
  - 采用 Cross Entropy 损失函数.
  - 使用 SGD 优化器
  - 学习率引入 cosine learning rate decay.
- 训练周期: 20 epochs 内收敛.


