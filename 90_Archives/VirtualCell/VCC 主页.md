[[VCC Intro Letter]]
## Timeline
- **2025 年 6 月 26 日：** 挑战赛宣布并提供数据。
    
- **2025 年 10 月 27 日：** 发布最终测试集。
    
- **2025 年 11 月 3 日：** 最终提交截止日期。
    
- **2025 年 12 月上旬：** 获奖者公布。

## 任务

这个挑战的目标是 **预测人类细胞在特定“基因干扰”之后的基因表达变化**。也就是说：
- 每次“干扰”一个基因（比如把它关闭或下调表达），就会触发整个细胞内部其他基因的响应。
- 这些响应可以用一种叫 **单细胞RNA测序（scRNA-seq）** 的方法测量。它会生成一个矩阵，每一行是一个细胞，每一列是一个基因，每个元素是这个细胞里这个基因的表达强度。
你的任务是：**根据已有的干扰-响应样本，预测新的干扰会导致哪些表达变化。**

在这个任务中, 数据的存储方式是表达谱:
- 每一个细胞可以看作一个样本（observation）；
- 每一个基因是一个特征（feature）；
- 一个细胞在某个基因上的“表达量”（expression level）(取值)是一个非负数，可以近似理解为“该基因被激活的程度”，类似计数数据（UMI count），可以认为是Poisson-like。
    
于是你可以想象数据结构如下：

|Cell ID|Gene1|Gene2|Gene3|...|Gene18080|Target Gene Perturbed|
|---|---|---|---|---|---|---|
|cell_001|0|10|3|...|4|CHMP3|
|cell_002|5|0|1|...|9|non-targeting|
|...|...|...|...|...|...|...|
其中：
- 行是每个细胞的“表达谱”；
- 最后一列是实验中这个细胞所受的基因干扰（也可能是“未干扰”，即控制组）；
- 总共有大约 22 万个细胞和 1.8 万个基因。

## 数据详情

### 训练集 `adata_Training.h5ad`

根据你提供的信息，最关键的数据文件是 **训练集** (`adata_Training.h5ad`)。它的维度可以概括为：
- **细胞数（n_obs）**：`221,273`
- **基因数（n_vars）**：`18,080`

这意味着你的核心数据矩阵是一个 `221,273 × 18,080` 的矩阵，其中：
- **行** 代表 **221,273 个** 单细胞。
- **列** 代表 **18,080 个** 基因。

这个矩阵中的每一个值都代表一个特定细胞中某个特定基因的表达量。


除了核心数据矩阵，还有一些重要的元数据（metadata）与这些维度相关联：

- **观测值（`obs`）**：这是关于**每个细胞**的描述信息。它的维度是 `221,273 × 4`。这些描述包括：
    - `cell barcode-batch index`：每个细胞的唯一标识。
    - `target_gene`：这个细胞是受到哪个基因的扰动，或者是否为对照组（`non-targeting`）。
    - `guide_id`：用于进行扰动的具体工具ID。
    - `batch`：数据收集的批次信息，这在数据预处理中很重要，可以帮助你处理批次效应。
        
- **变量（`var`）**：这是关于**每个基因**的描述信息。它的维度是 `18,080 × 2`。这些描述包括：
    - `gene name index`：基因的通用名称（例如 `SAMD11`）。
    - `gene_id`：Ensembl 数据库中的唯一基因ID（例如 `ENSG00000187634`）。

**什么是扰动？**

- 在你的训练集 `adata_Training.h5ad` 的 `obs`（细胞元数据）中，`target_gene` 列就是一个**类别变量**。

- **如何体现：** 每个细胞都带有一个标签，说明它受到了哪种扰动。例如，`target_gene` 的值可能是：
    - `CHMP3`
    - `SAMD11`
    - `non-targeting` (表示无扰动的对照组)

- **建模应用：** 在构建模型时，你可以将这个类别变量作为 **输入特征**（input feature）。对于大多数机器学习模型，你需要将这些类别标签进行编码，例如用 **One-Hot Encoding**。
    - **例子：** 如果你的模型是神经网络，输入层的一个节点组可能代表所有的扰动，当输入 `CHMP3` 扰动时，代表 `CHMP3` 的那个节点会被激活。

### 挑战任务与维度

你的任务是基于训练集的 `221,273 × 18,080` 矩阵，构建一个模型，这个模型可以为**验证集**和**最终测试集**中的**新扰动**，预测出**新的 `n_cells × 18,080` 矩阵**（每个扰动一个矩阵）。

- **验证集**和**最终测试集**只提供了**扰动列表**（例如 `50` 个和 `100` 个），你需要预测每个扰动下，基因表达的分布情况。
    
- 每个扰动需要预测的细胞数（`n_cells`）可能不同，你需要根据验证集文件中的信息来确定。

所以，这个挑战本质上是一个高维的预测问题：你的模型输入是一个扰动基因，输出是一个`18,080`维的基因表达向量（或者更准确地说，是一个概率分布或一个代表一组细胞的矩阵）。




## 重要文献
- Transformer based algorithm:
	- [[A foundation model of transcription across human cell types]]
	* [[Transformer for one stop interpretable cell type annotation]]
*  Mask auto-encoder
	* [https://arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377 "https://arxiv.org/abs/2111.06377"). We may want to add some sparsity to existing methods. I know Yixuan has some sparse mask paper. 


## 生物术语速查

- **基因调控 (regulation):** 
	- 生物体中的每个细胞都有相同的 DNA（约30亿碱基），但不同细胞只“打开”了一部分基因。这种“打开/关闭”基因表达的控制过程，称为 基因调控（gene regulation）。
		- DNA 相当于一个固定语料库；
		- 不同细胞通过选择性地启用某些句子（基因）来执行特定任务；
		- “调控”就是控制哪些句子能被读取（即哪些基因能被转录）。

- **转录因子 (transcription factor, TF):**
	- TF 是一种蛋白质, 能够识别特定的碱基序列 motif, 如果能够成功匹配就可以启动或抑制对应基因的表达. 

- **碱基序列 (Motif):**
	- motif 是一段短的 DNA 子序列（如 6–12bp），表示某个转录因子（TF）的结合偏好模式。
	- motif 通常表示为 PWM（position weight matrix），每个位置 4 个碱基的概率。

- **碱基得分 (Motif Score):** ^f9d626
	- TF 与 Motif 本身的匹配并不是 exact match, 因为其本质是一个物理化学过程, 其反应的发生能够允许一定的错配. Motif 得分本质上反映的是亲和力 affinity. 

- **位置权重矩阵 (Position Weight Matrix, PWM)**
	- 由于上面说的位置匹配的模糊性, 因此我们需要这样一个 PWM 矩阵进行衡量. 
	- 给定一个 Motif , 我们理论上对应了一个其 PWM. 这个 PWM 就反映了当前的 motif 的每个位置都对于哪种碱基感兴趣. 
	- 假设一个 Motif 的碱基序列长度为 $L$, 则一共会有 $4^L$ 中序列可能. 这里引入一个$P\in\mathbb {R}^{L\times 4}$ 的 PWM 矩阵, 表示每一个位置出现对应碱基 (ATCG之一) 的概率或其他某种得分. 其某位置 $p_{i,b}$ 就表示这个碱基序列中第 $i$ 位出现 $b$ 这个碱基的偏好程度/得分. 其形如: $$\begin{array}{c} P = \begin{bmatrix}
P_{1,A} & P_{1,C} & P_{1,G} & P_{1,T} \\
P_{2,A} & P_{2,C} & P_{2,G} & P_{2,T} \\
\vdots & \vdots & \vdots & \vdots \\
P_{L,A} & P_{L,C} & P_{L,G} & P_{L,T}
\end{bmatrix} \end{array}$$
	- 具体而言, 这个 PWM 是通过 TF 进行匹配实验从而进行统计的一个频率矩阵. 
	- 从 PWM 到 Motif Score:
		- 二者的关系: **motif score 就是用 PWM 对一段 DNA 序列打分的结果.** 这是一个标量, 表示当前这段 DNA 的, 对于当前 motif 的匹配程度. 
		- 数学建模
			- 给定: 转录因子 TF, 可以对应得到其 PWM $P \in \mathbb{R}^{L \times 4}$. 同时给定一个 DNA 序列 $s = s_1 s_2 \dots s_L \in \{A, C, G, T\}^L$. 
			- Motif Score 可以计算为: $$\text{MotifScore}(s; P) = \sum_{i=1}^{L} \log \left( \frac{P_{i, s_i}}{B(s_i)} \right)$$
				- $P_{i,s_{i}}$ 是 PWM 在第 $i$ 位置上对碱基 $s_i$ 的得分
				- $B(s_i)$ 是 baseline 概率, 表示碱基在“随机 DNA”中出现的基线频率, 常设置为 $1/4$. 

- **染色质窗口 (chromatin window):** 
	- 染色质（chromatin）是 DNA 包裹在蛋白质（组蛋白）上的实际物理结构。
	- 一个“染色质窗口”指的是 在基因组中围绕某个基因的一段固定长度（如 ±1 Mbp）的线性序列片段，用来作为模型的输入上下文. 
		- 就像 Transformer 模型中我们取一个固定长度的 context window（如512 tokens）来做语言建模，在这里我们取：
		- 某个基因的上游和下游各 1 Mbp，共 2 Mbp；
		- 在这段“窗口”中寻找调控元件和 motif。

- **染色质的可及性 (chromatin accessibility):**
	- DNA 在细胞中并不是裸露的，而是缠绕在蛋白质（如组蛋白）上，形成 **染色质（chromatin）**。有些区域是 “开放的”，即 DNA 暴露出来了，**TF（转录因子）可以结合上去**，从而调控基因表达；有些区域是 “关闭的”，即被 tightly packed，没有 TF 可以进来。
	- 单细胞 ATAC-seq（scATAC-seq）提供了一个对于 accessibility 的量化测量指标. 对于每个 region $r_i$, 在细胞类型 $c_i$ 下, 统计该区域的测序碎片数 (总数为 $t$):
		$$a_i = \log_{10} \left( \frac{c_i}{t} + 1 \right)$$
	- accessibility 是**动态特征**，体现细胞状态 $\text{调控活性} = f(\text{motif}_i, \text{accessibility}_i)$

- **转录起始位点 (Transcription Start Site/TSS)** 
	- TSS 是每个基因在基因组上的“起始位置”，即 RNA 合成从这里开始. 
	- 一个基因 g 通常有一个明确的 TSS（也可能有多个可选位点，但主TSS是确定的）
	- 大多数调控作用（尤其是 promoter）都发生在 TSS 附近

- **通路 Pathway**:
	- **Pathway** 是指一组在某个生物过程中**协同表达、共同作用**的基因. 类似于**功能模块**，对应某一类细胞功能或代谢行为，如「细胞周期」、「DNA修复」、「细胞凋亡」、「炎症反应」、「MAPK 信号通路」等
	- 设 $G = \{g_1, g_2, ..., g_n\}$ 是所有基因全集, 则 $\text{Pathway}_i \subseteq G$. 

- **调控子集 (regulon)**:
	- **Regulon** 是指由同一个转录因子（TF, transcription factor）**共同调控**的一组基因。
	- $\text{Regulon}_t = \{g_j \in G\ |\ g_j \text{ 有 TF } t \text{ 的 binding site 或调控证据} \}$ 

- **扰动 (perturbation)**
	1. **基因扰动（Genetic Perturbation）**: 通过特定的技术（比如 CRISPRi，即 CRISPR interference），有选择地**沉默**或**降低**某个特定基因的表达水平。想象一下，一个细胞就像一个庞大的工厂，里面有成千上万个工人（基因），每个工人都有自己的工作。基因扰动就是让某个工人暂时“停工”，然后观察这对整个工厂的生产（所有其他基因的表达）产生了什么影响。
		- 在你的数据中，`target_gene` 列中的基因名称（比如 `CHMP3`）就代表了这种基因扰动。
	2. **化学扰动（Chemical Perturbation）**: 通过施加某种化学物质或药物来影响细胞。虽然你这次挑战主要涉及基因扰动，但在其他一些公共数据集中（比如 `Tahoe-100M`），扰动也可以指药物处理。这就像是给工厂的管理层下达了一条新的命令，观察工人们如何调整他们的工作。
    

---

