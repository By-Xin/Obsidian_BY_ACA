# A Foundation Model of Transcription Across Human Cell Types

## Introduction

Our task is divided into two parts: pre-training and fine-tuning, each with the following objectives.

**Pre-training**

- Similar to BERT's Masked Language Model (MLM) task, we randomly mask some rows in $\boldsymbol{X}$ (i.e., partial regions), denoted as $X^{\text{masked}} = \{ \boldsymbol{x}_i = \textbf{m}\mid i\in \mathcal{I}_M \}$, where $\mathcal{I}_M \subset \{1,\cdots,N\}$ represents the index set of randomly selected rows, and $\textbf{m}\in \mathbb{R}^{d}$ is the feature representation after masking, which is also a learnable vector.
- The ultimate goal of pre-training is to reconstruct the masked parts as accurately as possible, i.e., minimize the following loss function:
$$\mathcal{L}_{\text{pre}} = \sum_{i \in \mathcal{I}_M} \|f_\Theta(\boldsymbol{X}^{\text{masked}}) - \boldsymbol{x}_i\|^2$$

## Model Architecture 

We first provide the problem definition.
- For a gene $g$ in cell type $c$, its regulatory region window (2 Mbp long) can detect $N$ (in this paper $N = 200$) regions/peaks, denoted as $\{r_i\}_{i=1}^N$.
- For each region $r_i$, we perform motif analysis to obtain motif scores $\boldsymbol{m}_i \in \mathbb{R}^{d_m}$, where $d_m = 282$ is the motif dimension.
- For this region, we also have chromatin accessibility information, denoted as $a_i \in \mathbb{R}$, which is measured through scATAC-seq experiments using logCPM counts.
- We concatenate the above $\boldsymbol{m}_i$ and $a_i$ to obtain the feature representation of this region: $\boldsymbol{x}_i = [\boldsymbol{m}_i, a_i] \in \mathbb{R}^{d}$, where we denote $d := d_m + 1$.
- All $N$ regions within the above 2 Mbp are encoded in this way, finally obtaining the input feature matrix:
  $${X} = \begin{bmatrix} \boldsymbol{x}_1^\top \\ \boldsymbol{x}_2^\top \\ \vdots \\ \boldsymbol{x}_N^\top \end{bmatrix} \in \mathbb{R}^{N \times d}.$$

### RegionEmb

First, we input the above feature matrix ${X}$ into a RegionEmb model for dimensionality expansion through linear transformation (without activation):
$$X' := \text{RegionEmb}(X) = X W_{\text{Emb}} \in \mathbb{R}^{N \times D}$$
where $W_{\text{Emb}} \in \mathbb{R}^{d \times D}$ is the weight matrix for linear transformation, with $D = 768$ in the paper.
- The paper notes that since the motifs in the original vectors have high correlation, we do not want to introduce nonlinear transformations too early to avoid compression/interference.

### Token-wise Self-Attention

To model regulatory relationships between peaks (including cis-interactions and trans-interactions), the authors employ a 12-layer Transformer Encoder to perform token-wise self-attention on all $\boldsymbol{h}_i \in \mathbb{R}^{768}$.

The specific approach is the same as the standard Transformer.
- By introducing learnable linear transformations $W_q, W_k \in \mathbb{R}^{D \times d_k}, W_v \in \mathbb{R}^{D \times d_v}$, we compute $Q = X' W_q, K = X' W_k, V = X' W_v$.
- The output for the current head $h$ is $O_h = \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)V \in \mathbb{R}^{N \times d_v}$.
- We compute all heads in parallel and perform concatenation and fully connected operations to finally obtain the Transformer output.
- Additionally, the model design includes common components such as residual connections and layer normalization, as well as feed-forward layers.

## Training Procedure

### Pre-training

In the input $X$, we randomly mask some regions (designed to randomly sample half of the regions), with the masked part's index set denoted as $\mathcal{M} \subset \{1,\cdots,N\}, |\mathcal{M}| = \frac{N}{2}$. For the masked parts, we set $\boldsymbol{x}_i = \boldsymbol{m}$, where $\boldsymbol{m} \in \mathbb{R}^{d}$ is a learnable vector. We denote such a masked input sample as $X^{\text{masked}}$.

We then input $X^{\text{masked}}$ into the above model for training:
1. Through the RegionEmb model, we obtain $H^{\text{masked}} = \text{RegionEmb}(X^{\text{masked}}) = X^{\text{masked}} W_{\text{Emb}} \in \mathbb{R}^{N \times D}$.
2. Through the Transformer model, we obtain $Z^{\text{masked}} = \text{Transformer}(H^{\text{masked}}) \in \mathbb{R}^{N \times D}$.
3. For each masked region $r_i, i \in \mathcal{M}$, we use a linear layer to recover its representation: $\boldsymbol{\hat{x}}_i = \boldsymbol{z}^{\text{masked}}_i W_{\text{dec}} \in \mathbb{R}^{d}$, where $W_{\text{dec}} \in \mathbb{R}^{D \times d}$ is the decoding weight matrix.
4. Calculate the loss function: $\mathcal{L} = \mathbb{E}\left[ \sum_{i \in M} -\log \mathbb{P}(x_i \mid X^{\text{masked}}) \right]$. In implementation, this is typically $\mathcal{L} = \sum_{i \in M} \| \hat{x}_i - x_i \|_2^2 \quad \text{or} \quad \sum_{i \in M} \mathrm{BCE}(\hat{x}_i, x_i)$.

Abstractly speaking, we first introduce a Transformer Encoder (denoted as $p$) to map the input $X$ to a latent representation $H$. Then we use a prediction head $g_\theta$ to reconstruct $\hat X$:
$$X \xrightarrow{p} H \xrightarrow{g_\theta} \hat{x}$$

The optimization objective is:
$$\min_{p, g} \ \mathbb{E}_{X, \mathcal{M}} \left[ \sum_{i\in\mathcal{M}} \left\| g_\theta(p(X))_i - x_i \right\|^2 \right]$$

The training parameters for this pre-training phase in the paper are:
- Optimizer: AdamW, with weight decay 0.05
- Batch Size: 256
- Epochs: 800, with 40 linear warmups
- Max Learning Rate: $1.5\times 10^{-4}$
- Training time: 7 days on 16 V100 GPUs

## Fine-tuning

After extensive masked pre-training, we obtain a well-performing latent representation function $p$. Therefore, in the fine-tuning stage, we aim to leverage the knowledge from $p$ to improve performance on downstream tasks.

Specifically, the task here is: given a regulatory region $X$ of gene $g$ in cell type $c$, predict its RNA expression level $\hat{y}$. Our input is still $\boldsymbol{x}_i = [\boldsymbol{m}_i, a_i] \in \mathbb{R}^{d}, \quad d = d_m + 1 = 283$, concatenated to form $X = [x_1^\top; \dots; x_N^\top] \in \mathbb{R}^{N \times d}$.

We reuse our pre-trained model $p$ to first map $X$ to the latent representation space $H = p(X) = [h_1^\top; \dots; h_N^\top] \in \mathbb{R}^{N \times D}$

Since we have $N$ regions, we need to aggregate them into a global representation. Specifically, we perform weighted summation through the following method (called attention pooling):
$$z = \sum_{i=1}^N \alpha_i h_i \in \mathbb{R}^D, \quad \text{where } \alpha_i = \frac{\exp(w^\top h_i)}{\sum_{j=1}^N \exp(w^\top h_j)}$$
- where $w \in \mathbb{R}^D$ is a learnable vector used to compute the weight $\alpha_i$ for each region.

Finally, we pass the pooled global representation $z \in \mathbb{R}^D$ through an MLP for prediction:
$$\hat{y} = f_\phi (z)$$
where $f_\phi$ is a multi-layer perceptron (MLP) model with parameters $\phi$, which is the main module in the fine-tuning stage.

Our final optimization objective is:
$$\min_{\phi, \theta} \sum_{(g,c)} \mathcal{L}_{\text{expr}}\left(f_\phi(p_\theta(X_{g,c})), y_{g,c}\right)$$
- where $(g,c)$ emphasizes the combination of gene $g$ and cell type $c$, enabling the model to optimize for specific gene and cell type combinations.
- $\mathcal{L}_{\text{expr}}$ is the loss function for expression level prediction, defined in the paper as Poisson Negative Log Likelihood (NLL) loss, suitable for regression tasks on count data. Some tasks are also set to use MSE.

