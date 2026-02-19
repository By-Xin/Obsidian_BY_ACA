# TOSICA: Transformer for One-Stop Interpretable Cell-type Annotation

## Briefing

The overall objective of TOSICA is cell type annotation.
- Its input is a single-cell transcriptome expression profile $\mathbf{e} \in \mathbb{R}^n$, where $n$ is the number of genes (10000+).
- The output is the predicted cell type $\hat{y} \in \{1, ..., C\}$, where $C$ is the number of cell types (10~50).

TOSICA is based on the Transformer architecture and introduces a **mask mechanism with biological prior knowledge**. Its design consists of three layers:

- Cell Embedding: Maps each cell's expression to a low-dimensional representation space, where each dimension corresponds to a biological pathway/regulatory module.
- Multi-Head Self-Attention: Uses attention mechanism to compute attention between the cell embedding and classification CLS token to obtain contextual information for cell types.
- Cell-Type Classifier: Maps attention information to cell type space through fully connected layers to achieve final classification prediction.

## Methodology

### Cell Embedding

We first establish notation. For each cell $\mathbf{e}\in \mathbb{R}^n$, where $n$ is the number of genes, we aim to map it to a low-dimensional representation through a fully connected network:
$$\mathbf{t} = \mathbf{W'}\mathbf{e}\in\mathbb{R}^k$$ 
where $\mathbf{W'}\in \mathbb{R}^{k\times n}$ is a learnable weight matrix. In this $k$-dimensional space, each dimension serves as a token, and each token represents a biological pathway/regulatory module.

Since we want $t$ to have each dimension represent a specific biologically meaningful pathway in a structured manner, the weight matrix $\mathbf{W'}$ needs to be specially designed with prior knowledge.
- We introduce a general weight matrix $\mathbf{W}\in \mathbb{R}^{k\times n}$
- Based on prior knowledge, we construct a mask matrix $\mathbf{M}\in \{0, 1\}^{k\times n}$
  - Where $M_{ij} = 1$ if and only if the $i$-th pathway contains the $j$-th gene. This knowledge is obtained from external databases (specifically, gene set datasets from the Gene Set Enrichment Analysis database)
  - Therefore, only genes within the same pathway are mapped to the same token. This avoids the non-interpretability of mixed components, though it also limits the model's expressive power.
- We perform Hadamard product (element-wise product) on these two matrices to obtain the final weight matrix:
	$$\begin{aligned} \mathbf{W'} &= \mathbf{M} \odot \mathbf{W} \\
	\mathbf{t} &= \mathbf{W'}\mathbf{e}\in\mathbb{R}^k \\
	\end{aligned}$$ 
- To enhance performance, the above operation is performed in parallel $m$ times ($m=48$), concatenated to obtain:
  	$$\mathbf{T}:=\begin{bmatrix} \mathbf{t}_1 & \mathbf{t}_2 & \cdots & \mathbf{t}_m \end{bmatrix} \in \mathbb{R}^{k\times m}.$$

### Multi-Head Self-Attention

Next, we introduce the attention mechanism to begin classification.

First, we introduce a learnable dummy token $\mathbf{cls} \in \mathbb{R}^m$ and concatenate it as the first row of $\mathbf{T}$ to obtain the new input:
$$ \mathbf{I} := \begin{bmatrix} \mathbf{cls}^\top \\ \mathbf{T} \end{bmatrix} = \begin{bmatrix} c_1 & c_2 & \cdots & c_m \\ \mathbf{t}_1 & \mathbf{t}_2 & \cdots & \mathbf{t}_m \end{bmatrix} \in \mathbb{R}^{(1+k)\times m}.$$
- This is a common practice in Transformer and other NLP models. CLS is an additional, learnable virtual token. Since attention operations output a representation with the same shape as the input, we can use the output of this CLS as the global information extraction result for the current input, similar to a statistic of the current input, serving as the token for subsequent classification.

We feed this input $\mathbf{I}$ into the attention mechanism. We first discuss the case of a single head. Overall, we have $\mathbf{O} = \text{Attention}(\mathbf{I}) \in \mathbb{R}^{(1+k)\times m}.$ The specific computational details are as follows:
- Through three linear transformations $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v \in \mathbb{R}^{(1+k)\times (1+k)}$, we map the input to Query, Key, and Value spaces:
  $$\begin{aligned}
  \mathbf{Q} &= \mathbf{W}_q \mathbf{I} \in \mathbb{R}^{(1+k)\times m}, \\
  \mathbf{K} &= \mathbf{W}_k \mathbf{I} \in \mathbb{R}^{(1+k)\times m}, \\
  \mathbf{V} &= \mathbf{W}_v \mathbf{I} \in \mathbb{R}^{(1+k)\times m}.
  \end{aligned}$$
- We compute attention scores to obtain contextual information:
  $$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top }{\sqrt{d_k}}\right) \in \mathbb{R}^{(1+k)\times (1+k)}.$$
  where $d_k = m$ is the dimension of the Key vectors. This attention score matrix $\mathbf{A}$ represents the similarity between different tokens.
- We obtain contextual representations for each token through weighted summation:
  $$\mathbf{O} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{(1+k)\times m}.$$

The multi-head case is similar to the above, equivalent to performing $H$ computations in parallel.
- For the $h$-th head, we independently introduce learnable weight matrices $\mathbf{W}_q^h, \mathbf{W}_k^h, \mathbf{W}_v^h \in \mathbb{R}^{(1+k)\times (1+k)}$. We finally obtain the output $\mathbf{O}^h = \text{Attention}^h(\mathbf{I}) \in \mathbb{R}^{(1+k)\times m}.$
- We concatenate the $H$ independent outputs and apply an additional linear mapping to reshape back to the original input shape: $\mathbf{\widetilde{O}} = \mathbf{W}_o \begin{bmatrix} \mathbf{O}^1 & \mathbf{O}^2 & \cdots & \mathbf{O}^H \end{bmatrix} \in \mathbb{R}^{(1+k)\times m}.$

### Cell-Type Classifier

Finally, since the attention mechanism's input and output have the same shape, we extract the first row of $\mathbf{O}$ (denoted as $\widetilde{\mathbf{cls}}$), which is the output of the input $<\mathbf{cls}>$ after the attention mechanism, as the global information extraction result for the current input. We pass this result through a fully connected neural network to complete classification:
$$
\mathbf{p} = \text{softmax}(\mathbf{W}_c \widetilde{\mathbf{cls}})\in \mathbb{R}^{\text{nc}}.
$$
where $\mathbf{W}_c \in \mathbb{R}^{\text{nc} \times (1+k)}$ is the learnable classification weight matrix, and $\text{nc}$ is the number of cell types.

### Model Architecture

The overall process and other details such as residual connections are shown in the following figures.

Main structure:

![Main Structure](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250815142412.png)

Specific implementation structures of submodules:
![Submodule Implementation Structure](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250815142118.png)

### Training Details

- Data splitting
  - Sample splitting is performed using different studies and biological states (cross-dataset/batch splitting, rather than simple splitting from the same source).
  - 30% of the training set is further divided into a validation set.
- Loss function and optimizer:
  - Cross Entropy loss function is used.
  - SGD optimizer is used.
  - Cosine learning rate decay is introduced for the learning rate.
- Training period: Convergence within 20 epochs.