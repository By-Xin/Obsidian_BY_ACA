---
aliases: [Negative Sampling, 负采样, NEG, Skip-gram with Negative Sampling]
tags:
  - concept
  - cs/nlp
  - ml/deep-learning
related_concepts:
  - "[[Word2Vec]]"
  - "[[Word_Embeddings]]"
  - "[[Softmax]]"
  - "[[Cross_Entropy]]"
  - "[[GloVe]]"
  - "[[Probability_Distribution]]"
source: "Mikolov et al. 2013; CS224N Lecture 1"
---

# Negative Sampling 负采样

> *Ref: [Distributed Representations of Words and Phrases and their Compositionality, Mikolov et al. 2013](https://arxiv.org/abs/1310.4546)*

## 动机

### Softmax 的问题

在标准 [[Word2Vec]] 中，条件概率使用 Softmax：
$$\mathbb{P}(\mathrm{o}|\mathrm{c}) = \frac{\exp(u_\mathrm{o}^\top v_\mathrm{c})}{\sum_{\mathrm{w} \in \mathcal{V}} \exp(u_\mathrm{w}^\top v_\mathrm{c})}$$

**分母**（Partition Function）的问题：
1. **计算复杂度**：需要对整个词汇表 $\mathcal{V}$ 求和，$\mathcal{O}(|\mathcal{V}|)$
2. **词汇表很大**：通常 $|\mathcal{V}| > 100,000$
3. **优化效率低**：每次更新都要计算全部词的得分

### Partition Function 的作用

从两个角度理解分母：
- **概率角度**：归一化项，保证概率和为 1
- **优化角度**：
  - 提高分子 $\exp(u_\mathrm{o}^\top v_\mathrm{c})$（正样本相似度）
  - 降低分母 $\sum_{\mathrm{w}} \exp(u_\mathrm{w}^\top v_\mathrm{c})$（负样本相似度）

---

## Negative Sampling 方法

### 核心思想

> **我们没必要遍历整个词汇表，只需要对少量有代表性的词汇进行采样即可。**

将多分类问题转化为**多个二分类问题**：
- 正样本：真实的 (center, context) 对 → 标签 = 1
- 负样本：随机采样的 (center, random) 对 → 标签 = 0

### 新的损失函数

对于中心词 $\mathrm{c}$ 和上下文词 $\mathrm{o}$：

$$\boxed{\mathcal{J} = -\log \sigma(u_\mathrm{o}^\top v_\mathrm{c}) - \sum_{\mathrm{k} \in \mathcal{K}} \log \sigma(-u_\mathrm{k}^\top v_\mathrm{c})}$$

其中：
- $\sigma(x) = \frac{1}{1+\exp(-x)}$ 是 Sigmoid 函数
- $\mathcal{K}$ 是 $K$ 个负样本的集合（通常 $K = 5 \sim 20$）

### 损失函数解读

| 项 | 公式 | 目标 |
|-----|------|------|
| **正样本项** | $-\log \sigma(u_\mathrm{o}^\top v_\mathrm{c})$ | 让 $\sigma(\cdot) \to 1$，即正样本得分高 |
| **负样本项** | $-\log \sigma(-u_\mathrm{k}^\top v_\mathrm{c})$ | 让 $\sigma(-\cdot) \to 1$，即负样本得分低 |

等价地，负样本项可以写成：$-\log(1 - \sigma(u_\mathrm{k}^\top v_\mathrm{c}))$

---

## 负采样分布

### 采样策略

负样本 $\mathrm{k}$ 从分布 $p_{\text{neg}}$ 中采样：

$$p_{\text{neg}}(\mathrm{w}) = \frac{\text{freq}(\mathrm{w})^{\alpha}}{\sum_{\mathrm{w'} \in \mathcal{V}} \text{freq}(\mathrm{w'})^{\alpha}}$$

其中：
- $\text{freq}(\mathrm{w})$ 是词 $\mathrm{w}$ 在语料库中的词频
- $\alpha \in (0, 1)$，经验建议 $\alpha = 0.75$

### 为什么用 $\alpha = 0.75$？

- $\alpha = 1$：按原始词频采样，高频词（如 "the", "a"）被过度采样
- $\alpha = 0$：均匀采样，忽略词频信息
- $\alpha = 0.75$：**平滑**词频分布，既考虑词频又避免过度集中

**示例**：假设 "the" 出现 100 次，"cat" 出现 1 次
- 原始比例：100:1
- 平滑后（$\alpha=0.75$）：$100^{0.75} : 1^{0.75} = 31.6 : 1$

---

## Softmax vs Negative Sampling 对比

|  | **Full Softmax** | **Negative Sampling** |
|--|------------------|----------------------|
| **计算的 logits** | 全部 $V$ 个词 | 1 个正样本 + $K$ 个负样本 |
| **复杂度** | $\mathcal{O}(V)$ | $\mathcal{O}(K)$，$K \ll V$ |
| **损失函数** | $-\log \frac{\exp(u_\mathrm{o}^\top v_\mathrm{c})}{\sum_{\mathrm{w}} \exp(u_\mathrm{w}^\top v_\mathrm{c})}$ | $-\log \sigma(u_\mathrm{o}^\top v_\mathrm{c}) - \sum_{\mathrm{k}} \log \sigma(-u_\mathrm{k}^\top v_\mathrm{c})$ |
| **归一化** | ✅ 全局归一化（概率和=1） | ❌ 无全局归一化 |
| **分类类型** | 多分类 | 多个二分类 |

### 矩阵形式

**Full Softmax**：
$$\mathbf{z} = \mathbf{v}_c \times W_{\text{out}}^\top \in \mathbb{R}^{1 \times V}$$
$$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})$$
$$\mathcal{L} = -\log \hat{y}_o$$

**Negative Sampling**：
$$\mathcal{L} = -\log \sigma(\mathbf{v}_c \cdot \mathbf{u}_o) - \sum_{k=1}^{K} \log(1 - \sigma(\mathbf{v}_c \cdot \mathbf{u}_k))$$

---

## 实践建议

### 超参数选择

| 参数 | 建议值 | 说明 |
|------|--------|------|
| $K$（负样本数） | 5-20 | 小数据集用大 $K$，大数据集用小 $K$ |
| $\alpha$（平滑指数） | 0.75 | 经验最优值 |

### 预处理步骤

```python
# 统计词频
word_freq = Counter(all_tokens)

# 计算负采样概率
alpha = 0.75
freq_array = np.array([word_freq[w] for w in vocab])
neg_sampling_dist = (freq_array ** alpha) / (freq_array ** alpha).sum()
```

---

## 延伸阅读

- [[Word2Vec]] - Skip-gram 模型原理与实现
- [[GloVe]] - 另一种词向量方法

