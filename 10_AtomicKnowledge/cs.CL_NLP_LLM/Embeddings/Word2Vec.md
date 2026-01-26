---
aliases: [Word2Vec, Skip-gram, CBOW, 词向量模型]
tags:
  - concept
  - cs/nlp
  - ml/deep-learning
related_concepts:
  - "[[Word_Embeddings]]"
  - "[[Negative_Sampling]]"
  - "[[GloVe]]"
  - "[[BERT]]"
  - "[[Softmax]]"
  - "[[Cross_Entropy]]"
source: "Mikolov et al. 2013; CS224N Lecture 1"
---

# Word2Vec

> *Ref: [Efficient Estimation of Word Representations in Vector Space, Mikolov et al. 2013](https://arxiv.org/abs/1301.3781)*

## 基本思想

**Word2Vec** 是一种通过预测上下文来学习词向量的模型。

核心思路：
1. 拥有一个大型文本 **corpus**（语料库）
2. 模型扫描文本，每次选取一个**中心词** $\mathrm{c}$ 和其**上下文词** $\mathrm{o}$
3. 通过最大化"在 $\mathrm{c}$ 的上下文中 $\mathrm{o}$ 出现的概率"来优化词向量

![Word2Vec 滑动窗口示意](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250116130511.png)

Word2Vec 有两种实现方式：
- **Skip-gram**: 给定中心词，预测上下文词 ⭐ 更常用
- **CBOW** (Continuous Bag of Words): 给定上下文词，预测中心词

---

## Skip-gram 模型

### 直观理解

给定句子 *'the quick brown **fox** jumps over the lazy dog'*：
- 中心词: *'fox'*
- 上下文词（窗口=2）: *'quick'*, *'brown'*, *'jumps'*, *'over'*

**目标**: 最大化上下文词出现的概率，最小化非上下文词的概率。

![Skip-gram 预测示意](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250116170501.png)

### 模型定义

**符号定义**：
- 文本位置 $t = 1, 2, \ldots, T$
- 词向量 $w_t \in \mathbb{R}^d$，维度 $d$ 常取 100~300
- 上下文窗口大小 $m$（通常 2~4）

**似然函数**：
$$\mathcal{L}(\theta) = \prod_{t=1}^T \prod_{-m \leq j \leq m, j \neq 0} \mathbb{P}(w_{t+j}|w_t ; \theta)$$

**损失函数**（负对数似然）：
$$\mathcal{J}(\theta) = - \frac{1}{T} \sum_{t=1}^T \sum_{-m \leq j \leq m, j \neq 0} \log \mathbb{P}(w_{t+j}|w_t ; \theta)$$

### 概率计算

**双向量表示**：为计算方便，对同一词汇使用两种向量：
- $v_\mathrm{w}$: 当 $\mathrm{w}$ 作为**中心词**时的向量
- $u_\mathrm{w}$: 当 $\mathrm{w}$ 作为**上下文词**时的向量

**条件概率**（Softmax）：
$$\mathbb{P}(\mathrm{o}|\mathrm{c}) = \frac{\exp(u_\mathrm{o}^\top v_\mathrm{c})}{\sum_{\mathrm{w} \in \mathcal{V}} \exp(u_\mathrm{w}^\top v_\mathrm{c})}$$

其中 $\mathcal{V}$ 为词汇表。内积 $u_\mathrm{o}^\top v_\mathrm{c}$ 衡量两个词向量的相似性。

**损失函数**（交叉熵形式）：
$$\mathcal{J}(\theta) = \sum_{d \in \mathcal{D}} \sum_{t=1}^{|T_d|} \sum_{-m \leq j \leq m, j \neq 0} - \log \mathbb{P}(w^{(d)}_{t+j}|w^{(d)}_t ; \theta)$$

等价于最小化交叉熵：
$$\min_{U,V} \mathbb{E}_{\mathrm{c},\mathrm{o}}[-\log \mathbb{P}_{U,V}(\mathrm{o}|\mathrm{c})]$$

---

## 梯度推导

### 梯度下降

初始化：$U, V \sim \mathcal{N}(0, 0.001)^{|\mathcal{V}| \times d}$

更新规则：
$$U^{(i+1)} := U^{(i)} - \alpha \nabla_U \mathcal{J}(U^{(i)}, V^{(i)})$$

### 梯度计算

对于中心词 $\mathrm{c}$ 和上下文词 $\mathrm{o}$，求 $\nabla_{v_\mathrm{c}} \log \mathbb{P}(\mathrm{o}|\mathrm{c})$：

$$\nabla_{v_\mathrm{c}} \log \mathbb{P}(\mathrm{o}|\mathrm{c}) = \nabla_{v_\mathrm{c}} \log \exp(u_\mathrm{o}^\top v_\mathrm{c}) - \nabla_{v_\mathrm{c}} \log \sum_{\mathrm{w} \in \mathcal{V}} \exp(u_\mathrm{w}^\top v_\mathrm{c})$$

**第一项**：
$$\nabla_{v_\mathrm{c}} \log \exp(u_\mathrm{o}^\top v_\mathrm{c}) = u_\mathrm{o}$$

**第二项**：
$$\nabla_{v_\mathrm{c}} \log \sum_{\mathrm{w}} \exp(u_\mathrm{w}^\top v_\mathrm{c}) = \frac{\sum_{\mathrm{x}} \exp(u_\mathrm{x}^\top v_\mathrm{c}) \cdot u_\mathrm{x}}{\sum_{\mathrm{w}} \exp(u_\mathrm{w}^\top v_\mathrm{c})} = \sum_{\mathrm{x}} \mathbb{P}(\mathrm{x}|\mathrm{c}) u_\mathrm{x}$$

### 最终结果

$$\boxed{\nabla_{v_\mathrm{c}} \log \mathbb{P}(\mathrm{o}|\mathrm{c}) = u_\mathrm{o} - \mathbb{E}_{\mathrm{w} \sim \mathbb{P}(\mathrm{w}|\mathrm{c})}[u_\mathrm{w}]}$$

**直觉解释**：
- $u_\mathrm{o}$: 观测到的上下文词向量（**observed**）
- $\mathbb{E}[u_\mathrm{w}]$: 根据当前模型预测的期望上下文词向量（**expected**）
- 梯度 = Observed - Expected
- 通过梯度下降，使词向量**接近**观测到的上下文，**远离**预期的上下文

---

## Softmax 的问题

**计算瓶颈**：分母需要对整个词汇表 $\mathcal{V}$ 求和
$$\sum_{\mathrm{w} \in \mathcal{V}} \exp(u_\mathrm{w}^\top v_\mathrm{c})$$

- 词汇表 $|\mathcal{V}|$ 通常很大（10万+）
- 每次更新都需要 $\mathcal{O}(|\mathcal{V}|)$ 复杂度

**解决方案**：[[Negative_Sampling]] - 用少量负样本近似全词汇表

---

## 神经网络视角

Skip-gram 模型本质上是一个**浅层神经网络**。

![Skip-gram 网络结构](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250116200255.png)

### 网络结构

| 层 | 形状 | 说明 |
|----|------|------|
| **输入层** | $\mathbf{x} \in \mathbb{R}^V$ | 中心词的 one-hot 向量 |
| **隐藏层** | $\mathbf{h} \in \mathbb{R}^d$ | 词嵌入向量 |
| **输出层** | $\mathbf{z} \in \mathbb{R}^V$ | 预测分数（logits） |

### 前向传播

1. **输入 → 隐藏层**（Embedding 查表）
   $$\mathbf{h} = \mathbf{x}^\top W_{\text{in}} = W_{\text{in}}[c, :]$$
   - $W_{\text{in}} \in \mathbb{R}^{V \times d}$: 中心词嵌入矩阵
   - one-hot 乘法等价于取第 $c$ 行

2. **隐藏层 → 输出层**（线性变换）
   $$\mathbf{z} = W_{\text{out}} \mathbf{h}$$
   - $W_{\text{out}} \in \mathbb{R}^{V \times d}$: 上下文词嵌入矩阵
   - $z_j = \mathbf{u}_j^\top \mathbf{h}$: 第 $j$ 个词的得分

3. **Softmax / Sigmoid**
   - Full Softmax: $\hat{y}_j = \frac{\exp(z_j)}{\sum_w \exp(z_w)}$
   - Negative Sampling: $\sigma(z_j)$ 或 $\sigma(-z_j)$

### 训练过程

```
1. Forward:  one-hot → embedding → logits → softmax/sigmoid
2. Loss:     cross-entropy / negative sampling loss
3. Backward: 计算梯度，更新 W_in, W_out
4. Repeat:   采样 (center, context) 对，迭代训练
```

**关键洞察**：
- 无隐藏层激活函数（恒等映射）
- 本质是两层线性变换 + softmax/sigmoid
- 模型参数就是两个嵌入矩阵

---

## 矩阵分解视角

### Full Softmax 的矩阵形式

1. **取中心词向量**
   $$\mathbf{v}_c = W_{\text{in}}[c, :] \in \mathbb{R}^{1 \times d}$$

2. **计算所有词的得分**
   $$\mathbf{z} = \mathbf{v}_c \times W_{\text{out}}^\top \in \mathbb{R}^{1 \times V}$$

3. **Softmax 归一化**
   $$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z}) \in \mathbb{R}^{1 \times V}$$

4. **交叉熵损失**
   $$\mathcal{L} = -\log \hat{y}_o$$

### Negative Sampling 的矩阵形式

1. **正样本得分**
   $$z_{\text{pos}} = \mathbf{v}_c \cdot \mathbf{u}_o$$

2. **负样本得分**（批量计算）
   $$\mathbf{z}_{\text{neg}} = \mathbf{v}_c \times \mathbf{U}_{\text{neg}}^\top \in \mathbb{R}^{1 \times K}$$
   - $\mathbf{U}_{\text{neg}} \in \mathbb{R}^{K \times d}$: $K$ 个负样本的嵌入

3. **损失函数**
   $$\mathcal{L} = -\log \sigma(z_{\text{pos}}) - \sum_{k=1}^K \log \sigma(-z_{\text{neg},k})$$

---

## PyTorch 实现

### 1. 数据准备

```python
import torch
import random
import numpy as np
from collections import Counter

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 示例语料库
corpus = [
    "i like to eat apples and bananas",
    "i like to watch movies and cartoons",
    "the cat likes to eat fish",
    "john loves to read books about python"
]

# 分词
tokenized_sentences = [sent.lower().split() for sent in corpus]

# 构建词表
all_tokens = [t for sent in tokenized_sentences for t in sent]
word_counter = Counter(all_tokens)
vocab = sorted(word_counter.keys())
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

print(f"Vocabulary size = {vocab_size}")
```

### 2. 生成 Skip-gram 训练样本

```python
def make_skipgram_data(tokenized_sentences, word2idx, window_size=2):
    """生成 (center_idx, outside_idx) 对"""
    pairs = []
    for tokens in tokenized_sentences:
        token_ids = [word2idx[w] for w in tokens]
        length = len(token_ids)
        for i, center_id in enumerate(token_ids):
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, length)
            for j in range(start, end):
                if j != i:
                    pairs.append((center_id, token_ids[j]))
    return pairs

skipgram_pairs = make_skipgram_data(tokenized_sentences, word2idx, window_size=2)
print(f"Total skip-gram pairs: {len(skipgram_pairs)}")
```

### 3. 模型定义（含负采样）

```python
import torch.nn as nn

class SkipGramNegSample(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_negatives=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_negatives = num_negatives
        
        # 两个嵌入矩阵
        self.in_embed = nn.Embedding(vocab_size, embed_dim)   # 中心词
        self.out_embed = nn.Embedding(vocab_size, embed_dim)  # 上下文词
        
        # 负采样分布 (词频^0.75)
        word_freq = np.array([word_counter[idx2word[i]] for i in range(vocab_size)], dtype=np.float32)
        word_freq = word_freq ** 0.75
        self.neg_sampling_dist = word_freq / word_freq.sum()
        
        # 初始化
        nn.init.uniform_(self.in_embed.weight, -0.5, 0.5)
        nn.init.uniform_(self.out_embed.weight, -0.5, 0.5)
    
    def forward(self, center_ids, outside_ids):
        batch_size = center_ids.size(0)
        
        # 1. 查表得到嵌入
        center_embed = self.in_embed(center_ids)      # (batch, embed_dim)
        outside_embed = self.out_embed(outside_ids)   # (batch, embed_dim)
        
        # 2. 正样本得分与损失
        pos_scores = torch.sum(center_embed * outside_embed, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8)
        
        # 3. 采样负样本
        neg_samples = np.random.choice(
            range(self.vocab_size),
            size=(batch_size, self.num_negatives),
            p=self.neg_sampling_dist
        )
        neg_samples = torch.LongTensor(neg_samples)
        
        # 4. 负样本得分与损失
        neg_embed = self.out_embed(neg_samples)  # (batch, num_neg, embed_dim)
        center_expand = center_embed.unsqueeze(1)  # (batch, 1, embed_dim)
        neg_scores = torch.bmm(neg_embed, center_expand.transpose(1, 2)).squeeze()
        neg_loss = -torch.log(torch.sigmoid(-neg_scores) + 1e-8)
        
        # 5. 总损失
        total_loss = (pos_loss + neg_loss.sum(1)).mean()
        return total_loss
```

### 4. 训练循环

```python
embed_dim = 8
num_negatives = 4
model = SkipGramNegSample(vocab_size, embed_dim, num_negatives)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 3
pairs_list = skipgram_pairs[:]

for epoch in range(num_epochs):
    random.shuffle(pairs_list)
    total_loss = 0.0
    
    for (center_id, outside_id) in pairs_list:
        center_tensor = torch.LongTensor([center_id])
        outside_tensor = torch.LongTensor([outside_id])
        
        loss = model(center_tensor, outside_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(pairs_list)
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss={avg_loss:.4f}")
```

### 5. 验证结果

```python
def get_embedding(model, word):
    idx = word2idx[word]
    return model.in_embed.weight[idx].detach().numpy()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def most_similar_words(model, query_word, top_k=3):
    query_emb = get_embedding(model, query_word)
    sims = []
    for w in vocab:
        if w == query_word:
            continue
        sim_score = cosine_sim(query_emb, get_embedding(model, w))
        sims.append((w, sim_score))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

# 测试
for w in ["eat", "movies", "python"]:
    if w in word2idx:
        print(f"\n[Most similar to '{w}']")
        for candidate, score in most_similar_words(model, w):
            print(f"   {candidate:<10} cos_sim = {score:.4f}")
```

---

## 总结

| 视角 | 核心思想 |
|------|----------|
| **神经网络** | 两层线性变换，one-hot → embedding → logits |
| **矩阵分解** | 隐式分解 PMI 矩阵 |
| **概率模型** | 最大化上下文词的条件概率 |

**关键点**：
1. 以可微方式从输入映射到输出（前向）
2. 通过反向传播更新参数
3. 本质是学习两个嵌入矩阵 $W_{\text{in}}, W_{\text{out}}$

---

## 延伸阅读

- [[Negative_Sampling]] - 负采样优化
- [[GloVe]] - 另一种词向量方法
- [[Word_Embeddings]] - 词向量概述
- [[BERT]] - 上下文相关的动态词向量

