#ResearchNotes #Biostats

GEARS 是一个融合了图神经网络（GNN）与深度表征学习的模型，它试图预测某组变量（基因）被人为激活/抑制之后，高维输出向量（基因表达）的变化趋势。它特别强调在缺乏样本（zero-shot）的情形下，如何利用先验结构信息提升预测能力。

## 研究任务

**任务核心**: 对于某一个细胞，在某种特定的扰动下（如某些基因被抑制或激活），预测它的全基因表达水平向量.

对于一个细胞样本 $i$, 我们有如下输入信息:
$$(\mathbf{x}^{(i)}, \mathcal{P}^{(i)}, \mathbf{y}^{(i)})$$
- $\mathbf{x}^{(i)}\in \mathbb{R}^K$ 是细胞 $i$ 的基因表达向量，$K$ 是基因的数量. 
- $\mathcal{P}^{(i)} = \{(g_j^{(i)},a_j^{(i)})\}_{j=1}^M$ 是细胞 $i$ 的扰动信息, $M\ll K$ 是扰动的数量 (常为 1~2 个). $g_j$ 是被扰动的基因, $a_j\in \{-1, 1\}$ 表示该基因被抑制或激活.
  - 注意, 本质上这个 perturb 应该是一个三值函数: 激活, 抑制或未处理. 这里的 $a_j$ 本身就是带有方向性的, 不是一个 0-1 的二值概念. 而未处理由于其稀疏性, 因此不会以0显式存储. 
- $\mathbf{y}^{(i)}\in \mathbb{R}^K$ 是细胞 $i$ 在受到扰动 $\mathcal{P}^{(i)}$ 后的真实基因表达向量 (label).

我们的整体学习任务是函数:
$$f: (\mathbf{x}^{(i)}, \mathcal{P}^{(i)}) \mapsto \hat{\mathbf{y}}^{(i)}$$

## 模型架构

GEARS 整体结构可以分解为如下模块:

### 1. Embedding 

整体而言, GEARS 设计了两种 embedding:
- Gene Embedding: 对输入向量的每个基因 (feature) 进行编码
- Perturbation Embedding: 对输入向量的每个扰动 (perturbation) 进行编码

#### Gene Embedding

首先, 对于每个基因 $u\in \{1, \dots, K\}$, GEARS 通过一个图神经网络 (GNN) 来生成它的 embedding. 该 GNN 的输入是一个基因 $u$ 的初始输入 $x_u$ 和一个基因间的共表达图 $\mathcal{G}_{\text{coexp}}$, 该图连接了与基因 $u$ 共表达的其他基因.

整体而言, 该部分的数学表达为:
$$h^{\text{gene}}_u = \text{GNN}_{\Theta_{\text{gene}}}(x_u; \mathcal{G}_{\text{coexp}})\in \mathbb{R}^d$$
- $x_u$ 是基因 $u$ 对应的表达向量 (feature vector)
- $\mathcal{G}_{\text{coexp}}$ 是基因间的 co-expression graph.  
  - 图中每个节点是一个基因 (因此共有 $K$ 个节点)
  - 边的权重是基因间的 Pearson 相关系数 $w_{u,v} = \text{corr}(x_u, x_v)$, 其中 $x_u, x_v$ 是基因 $u, v$ 的 (across all cells) 表达向量.
    - 构建的过程相当于: 对于所有的(未扰动细胞)样本 $\mathbf{x}^{(i)}, i=1, \dots, N$, 计算每对基因 $u, v$ 的 Pearson 相关系数, 得到一个 $K\times K$ 的相关系数矩阵. 
    - *生物学上, 这张图捕捉了基因之间的“功能相似性”或“调控同步性”*

具体 GNN 的细节暂时不表.

我们一共有 $K$ 个基因, 因此最终得到的 gene embedding 是一个 $K\times d$ 的矩阵 $H^{\text{gene}} \in \mathbb{R}^{K\times d}$. 其第 $u$ 行是基因 $u$ 的 embedding $h^{\text{gene}}_u$.

> 类比 NLP, GEARS 的每一行表示一个细胞, 类比为一个句子; 每一列表示一个基因, 类比为一个词. 因此这里的 gene embedding 类比为词向量 (word embedding). 事实上后面提到的 co-expression graph 类似于 Word2Vec 中的 pointwise mutual information (PMI) 矩阵.

#### Perturbation Embedding

类似地, 我们将每一个被扰动的基因 $u\in \{1, \dots, K\}$ 也通过一个图神经网络 (GNN) 来生成它的 embedding. 之所以要有别于 gene embedding, 是因为:
- 其表示的不是基因本身, 而是“基因被扰动”的信息
- GEARS 要对未见过的扰动进行预测, 因此需要一个更 generalizable 的 embedding 方法

总的而言, 该步骤的 embedding 相当于在当前细胞环境内, 通过先验的生物学知识, 与具体处理独立地给出一个编码好的 perturbation 编码字典. 

此外, 注意这个图在构建的时候我们要用到 GO (Gene Ontology) 的信息:
- GO 是一个人类先验的对基因的知识图谱
- GO 的每个节点是一个基因, 边表示基因间的功能相关性(生物学意义上)

形式上类似地, 我们有:
$$h^{\text{pert}}_u = \text{GNN}_{\Theta_{\text{pert}}}(x_u; \mathcal{G}_{\text{GO}}) \Rightarrow H^{\text{pert}}\in \mathbb{R}^{K\times d}$$
- 注意: 尽管我们最多给 $M$ 个 perturbations, 但由于每个基因都可能被扰动, 因此在构建这个图时, 我们仍然需要考虑所有 $K$ 个基因.
- $x_u'$ 是基因 $u$ 的 perturbation embedding. 注意区别上一个 GNN 中的 $x_u$, 二者是不一样的.  在 GNN 中, 由于图的设计不一样, 所以二者其实唯一的共同点是都表示的基因 $u$ 的表达向量, 但具体的取值和含义是不同的.


### 2. Compositional Module

这一步要考虑的是目前已有的扰动的具体组合. 输入细胞 i 的扰动集合: $\mathcal{P}^{(i)} = \{(g_j^{(i)}, a_j^{(i)})\}_{j=1}^M$, 其中 $M\ll K$ 是扰动的数量, $g_j^{(i)}$ 是被扰动的基因, $a_j^{(i)}\in \{-1, 1\}$ 表示该基因被抑制或激活. 期待的输出是编码 $h_{\mathcal{P}^{(i)}} \in \mathbb{R}^d$.

其具体算法如下:

**Step 1:** 单基因扰动 embedding. 通过上面得到的 $h_{g_j}^{\text{pert}}$, 乘上扰动方向, 得到
$$ \tilde{h}_{g_j}^{\text{pert}} = a_j h_{g_j}^{\text{pert}} \in \mathbb{R}^d$$

**Step 2:** Permutation-invariant aggregation. 对所有的 perturbation embedding 进行求和, 得到一个组合后的 perturbation embedding:
$$h_{\text{sum}} = \sum_{j=1}^M \tilde{h}_{g_j}^{\text{pert}} \in \mathbb{R}^d$$
- 之所以进行求和, 是为了保持 permutation-invariance, 即无论 perturbation 的顺序如何, 最终的组合结果都是一样的 (先 perturb $g_1$, 再 perturb $g_2$ 和先 perturb $g_2$, 再 perturb $g_1$ 的结果是一样的). 并且保证对于任意数量的 perturbation 都能处理.

**Step 3:** 非线性变换. 通过一个 MLP 对求和结果进行非线性变换, 得到最终的扰动组合 embedding:
$$h_{\mathcal{P}^{(i)}} = \text{MLP}_{\Theta_{\text{comp}}}(h_{\text{sum}}) \in \mathbb{R}^d$$


综上可以总结为:
$$h_{\mathcal{P}^{(i)}} = \text{MLP}_{\Theta_{\text{comp}}}\left(\sum_{j=1}^M a_j h_{g_j}^{\text{pert}}\right) \in \mathbb{R}^d$$


### 3. Perturbed Gene Embedding Construction

将基因自己的 gene embedding 与 perturbation embedding 进行组合, 得到 perturbed gene embedding. 具体而言, 对于每个基因 $u\in \{1, \dots, K\}$, 我们有:
$$h_u^{\text{post}} = \text{MLP}_{\Theta_{\text{pp}}}(h_u^{\text{gene}} + h_{\mathcal{P}}) \in \mathbb{R}^d, \quad u \in \{1, \dots, K\}$$


### 4. Decoder

Decoder 的任务是将受到扰动后的基因的 embedding 进一步表达为扰动后的表达值 $\hat{\mathbf{y}}^{(i)} \in \mathbb{R}^K$.  并且需要指出, 我们并不是要直接得到表达值, 而是要预测扰动导致的变化量 $\Delta := {\mathbf{y}}^{(i)} - \mathbf{x}^{(i)}$, 也就是预测扰动后的表达值相对于未扰动的表达值的变化.

具体而言, Decoder 分为如下几步:

**Step 1: Gene-wise Linear Projection** 

对每个基因 embedding $h_u^{\text{post}}\in \mathbb{R}^d$ 进行线性投影到标量:
$$z_u = w_u^\top h^{\text{post}}_u + b_u\in \mathbb{R}$$

进一步将一个细胞内的所有基因进行拼接, 得到 $\mathrm{z}^{(i)} := [z_1, \dots, z_K]^\top \in \mathbb{R}^K$. 

**Step 2: Cell-level Global Context**

通过引入新的 MLP，将一个细胞内的所有基因的线性投影进行整合，得到细胞级别的全局上下文信息:
$$h^{\text{cg}} = \text{MLP}_{\Theta_{\text{cg}}}(\mathrm{z}^{(i)}) \in \mathbb{R}^{d_{\text{cg}}}$$

这是一个全局上下文向量，包含扰动对整个细胞表达模式的系统性影响. 

**Step 3: Local-Global Fusion**
对于每个基因 $u$, 将其局部信息 $z_u$ 与全局上下文 $h^{\text{cg}}$ 进行融合, 对扰动增量进行预测:
$$\hat{\Delta}_u = \text{MLP}_u([z_u, h^{\text{cg}}]) \in \mathbb{R}$$

对于一个细胞, 整合所有基因, 得到
$$\hat{\Delta}^{(i)} = [\hat{\Delta}_1, \dots, \hat{\Delta}_K]^\top \in \mathbb{R}^K$$

**Step 4: Final Prediction**
将预测的扰动增量 $\hat{\Delta}^{(i)}$ 加到未扰动的表达向量 $\mathbf{x}^{(i)}$ 上, 得到最终的预测表达向量:
$$\hat{\mathbf{y}}^{(i)} = \mathbf{x}^{(i)} + \hat{\Delta}^{(i)} \in \mathbb{R}^K$$

### 5. Loss Function

我们终极的目标是拟合扰动后基因表达量 $\hat{\mathbf{y}}^{(i)}$ 与真实值 $\mathbf{y}^{(i)}$ 之间的差异. GEARS 使用了一个自适应的损失函数, 结合了两部分:
- **Autofocus Loss**: 主要关注预测值与真实值之间的差异, 通过加权 MSE 实现
- **Direction-aware Loss**: 关注预测值与真实值的方向一致性, 确保预测的增量与真实增量在符号上保持一致.
$$\mathcal{L}_{\text{GEARS}} = \mathcal{L}_{\text{autofocus}} + \lambda \mathcal{L}_{\text{direction}}$$

#### Autofocus Loss

对于一个 minibatch 中的 $B$ 个细胞, 每个细胞对应$M^{(i)}$ 个扰动, 每个细胞有 $K$ 个基因, 预测值为 $\hat{\mathbf{y}}^{(i)}$, 真实值为 $\mathbf{y}^{(i)}$, 则 autofocus loss 定义为:
$$\mathcal{L}_{\text{autofocus}} = \frac{1}{B} \sum_{i=1}^B \frac{1}{M^{(i)}} \sum_{j=1}^{M^{(i)}} \frac{1}{K} \sum_{u=1}^K (y_u^{(i)} - \hat{y}_u^{(i)})^{2 + 2\gamma}$$
- 其中 $\gamma \geq 0$ 是一个超参数, 相当于是一种自适应的加权机制, 使得误差较大的基因在损失中占更大比重.

#### Direction-aware Loss
方向感知损失定义为:
$$\mathcal{L}_{\text{direction}} = \frac{1}{B} \sum_{i=1}^B \frac{1}{M^{(i)}} \sum_{j=1}^{M^{(i)}} \frac{1}{K} \sum_{u=1}^K [\text{sign}(y_u^{(i)} - x_u^{(i)}) - \text{sign}(\hat{y}_u^{(i)} - x_u^{(i)})]^2$$
- 该损失确保预测的增量与真实增量在符号上保持一致. 如果方向一致, 则该项为0. 如果方向相反, 则该项为4. 若一个几乎不变, 另一个具有方向, 则该项为1.
- 因此可以根据方向的不同偏离程度, 给予不同的惩罚.

