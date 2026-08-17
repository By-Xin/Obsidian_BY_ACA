---
aliases: [Word Embeddings, 词向量, 词嵌入, Word_Embeddings]
tags:
  - concept
  - cs/nlp
  - ml/deep-learning
related_concepts:
  - "[[Word2Vec]]"
  - "[[GloVe]]"
  - "[[BERT]]"
  - "[[Negative_Sampling]]"
  - "[[Softmax]]"
  - "[[SVD]]"
  - "[[Information_Theory]]"
---

# Word Embeddings 词向量

> 词嵌入：将词映射到连续向量空间的表示学习方法

---

## 核心问题

NLP 的首要问题是**如何表示词**。

### 传统方法: One-Hot Encoding
- 每个词是一个维度为 $V$ 的向量（$V$ = 词汇表大小）
- 表示是**离散的、稀疏的**
- 缺点: 词与词之间都是正交的，无法表示相似性

### 现代方法: Word Embeddings
- 将每个词映射到一个**连续的低维向量空间**
- 语义相近的词在空间中距离也相近
- 将**高维稀疏**表示转化为**低维稠密**表示

---

## 词向量的数学表示

给定词嵌入矩阵 $U \in \mathbb{R}^{V \times d}$：
- $V$ = 词汇表大小
- $d$ = 词向量维度（超参数，常取 100~300）
- 第 $i$ 行 $u_i$ 表示第 $i$ 个词的向量

对于 one-hot 表示的句子 $X \in \mathbb{R}^{V \times T}$（$T$ = 句子长度），词向量表示为：
$$X^\top U \in \mathbb{R}^{T \times d}$$

---

## 词向量方法分类

| 方法类型 | 代表模型 | 核心思想 | 优缺点 |
|----------|----------|----------|--------|
| **Prediction-based** | [[Word2Vec]] | 通过预测上下文学习词向量 | ✅ 捕捉语义关系 ❌ 无全局信息 |
| **Count-based** | LSA, [[GloVe]] | 基于共现矩阵学习词向量 | ✅ 利用全局信息 ❌ 计算复杂 |
| **Contextual** | [[BERT]], ELMo | 动态词向量，依赖上下文 | ✅ 消歧能力强 ❌ 计算成本高 |

---

## 📝 笔记索引

| 文件 | 主题 | 关键概念 |
|------|------|----------|
| [[Word2Vec]] | Word2Vec 模型 | Skip-gram, CBOW, 神经网络视角, PyTorch 实现 |
| [[Negative_Sampling]] | 负采样技巧 | 采样分布, 二分类近似 |
| [[GloVe]] | GloVe 模型 | Co-occurrence Matrix, PMI, 全局向量 |

---

## 🗺️ 知识图谱

```
Word Embeddings (本文档)
  │
  ├─→ Word2Vec (Prediction-based)
  │     └─→ Negative_Sampling (优化技巧)
  │
  ├─→ GloVe (Count-based)
  │
  └─→ BERT (Contextual, 见 ../Pretrained_Models/)
```

---

## 延伸阅读

- [[BERT]] - 上下文相关的动态词向量
- `cs.IT_InfoTheory/` - 信息论基础（熵、KL散度）
- `math.LA_LinearAlgebra/` - SVD 降维

