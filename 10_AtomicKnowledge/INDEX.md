# Atomic Knowledge 索引

> 基于 arXiv 分类体系的知识结构 | 最后更新: 2026-01-17 (v3.0 扁平化重构)

## 📊 当前文件统计

- **总笔记数**: **~110 篇**
- **数学基础**: 48 篇
  - 线性代数 10 | 优化 2 | 概率论 12 | 实分析 14
- **统计学**: 26 篇
  - 统计推断 17 | 时间序列 9
- **机器学习**: 39 篇
  - 算法 27 | 理论 2 | 深度学习 10
- **NLP/LLM**: 11 篇
- **CS基础**: 2 篇
- **计量经济学**: 1 篇
- **杂项**: 6 篇

---

## 🗂️ 完整目录结构

```
10_AtomicKnowledge/
├── INDEX.md                          ← 总索引
├── SOP.md                           ← 整理规范
│
│ ══════════════ 数学基础 ══════════════
├── math.LA_LinearAlgebra/            # 线性代数 (10篇)
├── math.OC_Optimization/             # 优化理论 (2篇)
├── math.PR_Probability/              # 概率论 (12篇)
│   └── Random_Variable_Generation/   # 随机变量生成
├── math.RA_RealAnalysis/             # 实分析 (14篇)
│
│ ══════════════ 统计学 ══════════════
├── stat.IN_Inference/                # 统计推断 (17篇) ⭐新
├── stat.TS_TimeSeries/               # 时间序列 (9篇) ⭐新
│
│ ══════════════ 机器学习 ══════════════
├── ml.AL_Algorithms/                 # ML算法 (27篇) ⭐新
│   # 包含: 线性模型、分类、树模型、集成学习、核方法、GLM等
├── ml.TH_Theory/                     # 学习理论 (2篇) ⭐新
├── ml.DL_DeepLearning/               # 深度学习 (10篇) ⭐新
│
│ ══════════════ CS与应用 ══════════════
├── cs.CL_NLP_LLM/                    # NLP与LLM (11篇)
├── cs.IT_InfoTheory/                 # 信息论 (1篇)
├── cs.DS_Algorithms/                 # 算法 (1篇)
├── econ.EM_Econometrics/             # 计量经济学 (1篇)
│
│ ══════════════ 其他 ══════════════
└── _Misc/                            # 杂项 (6篇)
```

---

## 📁 各目录详情

### 数学基础

| 目录 | 笔记数 | 核心内容 |
|------|--------|----------|
| `math.LA_LinearAlgebra/` | 10 | 矩阵、特征值、分解 |
| `math.OC_Optimization/` | 2 | 凸优化、对偶理论 |
| `math.PR_Probability/` | 12 | 极限定理、分布、随机变量生成 |
| `math.RA_RealAnalysis/` | 14 | 测度论、拓扑、集合论 |

### 统计学

| 目录 | 笔记数 | 核心内容 |
|------|--------|----------|
| `stat.IN_Inference/` | 17 | Fisher信息、UMVUE、非参数检验 |
| `stat.TS_TimeSeries/` | 9 | AR模型、平稳过程、预测 |

### 机器学习

| 目录 | 笔记数 | 核心内容 |
|------|--------|----------|
| `ml.AL_Algorithms/` | 27 | 回归、分类、集成、核方法、GLM |
| `ml.TH_Theory/` | 2 | GMM、MDP (预留 PAC/VC) |
| `ml.DL_DeepLearning/` | 10 | CNN、RNN、Transformer、MoE |

### CS与应用

| 目录 | 笔记数 | 核心内容 |
|------|--------|----------|
| `cs.CL_NLP_LLM/` | 11 | BERT、LLM、Embeddings |
| `cs.IT_InfoTheory/` | 1 | 熵、KL散度 |
| `cs.DS_Algorithms/` | 1 | 演化计算 |
| `econ.EM_Econometrics/` | 1 | 面板数据 |

---

## 🔄 v3.0 重构变更记录 (2026-01-17)

### 结构扁平化
- ❌ 删除 `math.ST_Statistics/` (二级目录过深)
- ❌ 删除 `ml.TH_MachineLearning/` (二级目录过深)
- ❌ 删除 `stat.ME_Methodology/` (合并到 ml.AL)
- ✅ 新建 `stat.IN_Inference/` (17篇)
- ✅ 新建 `stat.TS_TimeSeries/` (9篇)
- ✅ 新建 `ml.AL_Algorithms/` (27篇，合并经典ML+方法+统计方法论)
- ✅ 新建 `ml.TH_Theory/` (2篇，预留学习理论)
- ✅ 新建 `ml.DL_DeepLearning/` (10篇)

### 删除笔记
- 删除学派笔记 (School_of_Baysian.md, School_of_Frequency.md)

---

## 📌 待补充的重要主题

### 高优先级
- [ ] Conformal Prediction → `ml.AL_Algorithms/`
- [ ] Bootstrap → `ml.AL_Algorithms/`
- [ ] 贝叶斯推断 → `stat.IN_Inference/`

### 中优先级
- [ ] PAC Learning → `ml.TH_Theory/`
- [ ] VC Dimension → `ml.TH_Theory/`
- [ ] MCMC → `math.PR_Probability/`

---

## 💡 使用建议

1. **快速定位**:
   - 统计推断（估计、检验）→ `stat.IN_Inference/`
   - 时间序列 → `stat.TS_TimeSeries/`
   - ML 算法（回归、分类、集成）→ `ml.AL_Algorithms/`
   - 深度学习 → `ml.DL_DeepLearning/`
   - NLP/LLM → `cs.CL_NLP_LLM/`

2. **学习路径**:
   - **统计学**: 概率论 → 统计推断 → 时间序列
   - **机器学习**: ML算法 → 学习理论 → 深度学习
   - **NLP**: Embeddings → Transformer → BERT → LLM

---

*整理规范: [`SOP.md`](SOP.md)*
