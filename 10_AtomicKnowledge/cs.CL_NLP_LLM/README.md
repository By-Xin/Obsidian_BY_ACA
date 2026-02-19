# cs.CL - Computation and Language (NLP & LLM)

> 自然语言处理与大语言模型 | arXiv: [cs.CL](https://arxiv.org/list/cs.CL/recent)

---

## 📁 目录结构

```
cs.CL_NLP_LLM/
├── Embeddings/                    # 词嵌入
│   ├── README.md                  # 词向量概述
│   ├── Word2Vec.md                # Word2Vec (理论+实现)
│   ├── Negative_Sampling.md       # 负采样优化
│   └── GloVe.md                   # 全局向量
│
├── Pretrained_Models/             # 预训练模型
│   └── BERT.md                    # BERT 模型
│
├── LLM_Training/                  # LLM 训练
│   ├── Pretrain_and_Alignment_for_LLMs.md
│   └── Post-Training_and_Forgetting.md
│
└── LLM_Inference/                 # LLM 推理与分析
    ├── Deep_Reasoning_for_LLMs.md
    ├── Uncertainty_in_LLMs.md
    └── Context_Engineering.md
```

---

## 📝 笔记索引

### Embeddings/ - 词嵌入

| 文件 | 主题 | 关键概念 |
|------|------|----------|
| `README.md` | 词向量概述 | 方法分类, One-hot vs Dense |
| `Word2Vec.md` | Word2Vec 模型 | Skip-gram, CBOW, 神经网络视角, PyTorch 实现 |
| `Negative_Sampling.md` | 负采样技巧 | 采样分布, 二分类近似 |
| `GloVe.md` | GloVe 模型 | Co-occurrence Matrix, PMI, 全局向量 |

### Pretrained_Models/ - 预训练模型

| 文件 | 主题 | 关键概念 |
|------|------|----------|
| `NGram_LanguageModels.md` | 语言模型基础 | N-gram, Markov 假设 |
| `BERT.md` | BERT 模型 | Self-Supervised Learning, Pre-training, Fine-tuning |

### LLM_Training/ - LLM 训练

| 文件 | 主题 | 关键概念 |
|------|------|----------|
| `Pretrain_and_Alignment_for_LLMs.md` | 预训练与对齐 | SFT, RLHF, Knowledge Distillation |
| `Post-Training_and_Forgetting.md` | 后训练与遗忘 | Catastrophic Forgetting, Continual Learning |

### LLM_Inference/ - LLM 推理与分析

| 文件 | 主题 | 关键概念 |
|------|------|----------|
| `Deep_Reasoning_for_LLMs.md` ⭐ | LLM 深度推理 | Chain of Thought, Test-Time Compute, RL |
| `Uncertainty_in_LLMs.md` ⭐ | 不确定性量化 | Semantic Entropy, Confidence Estimation |
| `Context_Engineering.md` | 上下文工程 | Prompt Engineering, In-Context Learning |

---

## 📚 相关专题 (20_StudyNotes/Topics/)

| 文件 | 主题 |
|------|------|
| `LLM_Sampling_Methods.md` | 采样方法综述 (Top-k, Top-p, Temperature, Mirostat) |
| `Latent_Space_LLM_Survey_2024.md` | 潜空间 LLM 前沿技术综述 |

## 📄 相关论文 (20_StudyNotes/Papers/)

| 文件 | 主题 |
|------|------|
| `Mitigating_LLM_Hallucinations_via_Conformal_Abstention.md` | 幻觉缓解 (Conformal Prediction) |

---

## 🗺️ 知识图谱

```
词嵌入 (Static Embeddings)
  ├─ Embeddings/README (概述)
  │     ├─→ Word2Vec (Skip-gram/CBOW + 实现)
  │     │     └─→ Negative_Sampling (优化技巧)
  │     └─→ GloVe (全局向量, PMI)
  │
预训练模型
  ├─ BERT (Contextual Embeddings)
  │
LLM 训练
  ├─ Pretrain_and_Alignment (SFT, RLHF)
  │     └─→ Post-Training_and_Forgetting (遗忘问题)
  │
LLM 推理与分析
  ├─ Deep_Reasoning_for_LLMs (推理能力)
  │     └─→ [Topics] Latent_Space_LLM_Survey
  │
  ├─ Uncertainty_in_LLMs (置信度)
  │     └─→ [Papers] Conformal_Abstention
  │
  └─ Context_Engineering (提示工程)
```

---

## 🔗 跨目录关联

- **理论基础**: `cs.IT_InfoTheory/` (熵、KL散度、互信息)
- **深度学习**: `ml.DL_DeepLearning/` (Transformer, Attention)
- **机器学习**: `ml.AL_Algorithms/` (Transfer Learning, LifeLong Learning)
- **统计推断**: `stat.IN_Inference/` (Conformal Prediction)
