# ByAca 知识库整理规范 SOP

> Standard Operating Procedure for Organizing the ByAca Knowledge Base  
> Version 5.0 | 最后更新: 2026-01-17

---

## 📋 目录

1. [知识库全局结构](#知识库全局结构)
2. [核心原则](#核心原则)
3. [10_AtomicKnowledge 结构](#10_atomicknowledge-结构)
4. [20_StudyNotes 结构](#20_studynotes-结构)
5. [命名规范](#命名规范)
6. [YAML Frontmatter 规范](#yaml-frontmatter-规范)
7. [双链规范](#双链规范)
8. [分类决策树](#分类决策树)
9. [原子性检查与拆分](#原子性检查与拆分)
10. [README 维护](#readme-维护)
11. [操作流程](#操作流程)
12. [质量检查清单](#质量检查清单)
13. [常见问题](#常见问题)

---

## 🏛️ 知识库全局结构

```
ByAca/
├── 00_Dailies/              # 每日笔记 (时间线)
├── 10_AtomicKnowledge/      # 原子知识 (单一概念, 可复用)
├── 20_StudyNotes/           # 学习笔记 (课程、书籍、专题、论文)
├── 30_Workbench/            # 工作台 (进行中的项目)
├── 40_Toolbox/              # 工具箱 (方法论、最佳实践)
├── 90_Archives/             # 归档 (已完成/待处理)
└── 99_Assets/               # 资源 (模板、图片、脚本)
```

### 内容分流原则

| 内容类型 | 存放位置 | 特点 |
|---------|---------|------|
| **单一概念** | `10_AtomicKnowledge/` | 原子化、可链接、可复用、50-300行 |
| **课程笔记** | `20_StudyNotes/Lectures/` | 按课程组织、线性结构 |
| **书籍笔记** | `20_StudyNotes/Books/` | 按书籍章节组织 |
| **专题综述** | `20_StudyNotes/Topics/` | 多概念整合、全景视角、500+行 |
| **论文阅读** | `20_StudyNotes/Papers/` | 单篇论文的阅读笔记 |
| **项目文档** | `30_Workbench/` | 进行中的研究/代码 |
| **方法论** | `40_Toolbox/` | 可复用的工作流程 |

### Atomic vs Topics 的区分

| | 10_AtomicKnowledge | 20_StudyNotes/Topics |
|---|---|---|
| **粒度** | 单一核心概念 | 多概念综合 |
| **目的** | 知识单元、可链接 | 领域概览、导航地图 |
| **长度** | 通常 50-300 行 | 可较长 (500+ 行) |
| **示例** | `Word2Vec.md`, `BERT.md` | `Latent_Space_LLM_Survey_2024.md` |
| **链接关系** | 被 Topics 引用 | 引用多个 Atomic Notes |

---

## 🎯 核心原则

### 1. 原子性 (Atomic)
- 每个笔记只讲解**一个核心概念**
- 避免在单个文件中混杂多个主题

### 2. 扁平化 (Flat Structure)
- 一级目录即为最终分类，**避免过深的二级目录**
- 每个目录 5~30 篇笔记为宜

### 3. 英文优先 (English-first)
- 文件名使用英文 Pascal_Snake_Case
- 中文别名写入 YAML `aliases` 字段

### 4. 可追溯性 (Traceable)
- 记录信息来源、相关项目、概念间链接

---

## 📁 10_AtomicKnowledge 结构

```
10_AtomicKnowledge/
├── INDEX.md                          ← 总索引
├── SOP.md                           ← 本文档
│
│ ══════════════ 数学基础 ══════════════
├── math.LA_LinearAlgebra/            # 线性代数
├── math.OC_Optimization/             # 优化理论
├── math.PR_Probability/              # 概率论
│   └── Random_Variable_Generation/   # (唯一允许的二级目录)
├── math.RA_RealAnalysis/             # 实分析
│
│ ══════════════ 统计学 ══════════════
├── stat.IN_Inference/                # 统计推断
├── stat.TS_TimeSeries/               # 时间序列
│
│ ══════════════ 机器学习 ══════════════
├── ml.AL_Algorithms/                 # ML算法（经典+现代+统计方法）
├── ml.TH_Theory/                     # 学习理论（预留）
├── ml.DL_DeepLearning/               # 深度学习
│
│ ══════════════ CS与应用 ══════════════
├── cs.CL_NLP_LLM/                    # NLP与LLM
├── cs.IT_InfoTheory/                 # 信息论
├── cs.DS_Algorithms/                 # 算法
├── econ.EM_Econometrics/             # 计量经济学
│
│ ══════════════ 其他 ══════════════
└── _Misc/                            # 杂项（临时存放）
```

### 分类体系对照表

| 文件夹名称 | 领域名称 | 覆盖范围 |
|-----------|---------|---------|
| `math.LA_LinearAlgebra` | 线性代数 | 矩阵、特征值、分解 |
| `math.OC_Optimization` | 优化理论 | 凸优化、对偶、KKT |
| `math.PR_Probability` | 概率论 | 极限定理、分布、随机过程基础 |
| `math.RA_RealAnalysis` | 实分析 | 测度论、拓扑 |
| `stat.IN_Inference` | 统计推断 | 估计、检验、非参数统计 |
| `stat.TS_TimeSeries` | 时间序列 | AR模型、平稳过程、预测 |
| `ml.AL_Algorithms` | ML算法 | 回归、分类、集成、核方法、GLM |
| `ml.TH_Theory` | 学习理论 | PAC、VC维、泛化理论（预留） |
| `ml.DL_DeepLearning` | 深度学习 | CNN、RNN、Transformer、MoE |
| `cs.CL_NLP_LLM` | NLP/LLM | BERT、GPT、Embeddings |
| `cs.IT_InfoTheory` | 信息论 | 熵、KL散度 |
| `cs.DS_Algorithms` | 算法 | 数据结构、复杂度 |
| `econ.EM_Econometrics` | 计量经济学 | 面板数据、因果推断 |
| `_Misc` | 杂项 | 临时存放 |

---

## 📁 20_StudyNotes 结构

```
20_StudyNotes/
├── Books/                    # 书籍笔记
│   └── {BookName}/           # 按书籍分目录
│       └── Chapter_X.md
│
├── Lectures/                 # 课程笔记
│   └── {CourseName}/         # 按课程分目录
│       └── Lecture_X.md
│
├── Topics/                   # 专题综述 (Survey/Review)
│   └── {Topic_Name}.md       # 多概念整合的综合性文档
│
├── Papers/                   # 论文阅读笔记
│   └── {Paper_Title}.md      # 单篇论文的阅读记录
│
└── Paper_Reading_List.md     # 论文阅读清单
```

### Topics vs Papers 的区分

| | Topics (专题) | Papers (论文) |
|---|---|---|
| **内容** | 领域综述、多论文整合 | 单篇论文阅读笔记 |
| **来源** | 自己整理或 Survey 论文 | 具体一篇论文 |
| **示例** | `Latent_Space_LLM_Survey_2024.md` | `Mitigating_LLM_Hallucinations_via_Conformal_Abstention.md` |

---

## 📝 命名规范

### 文件夹命名
- **格式**: `{领域代码}.{缩写}_{英文描述}`
- **示例**: `math.LA_LinearAlgebra`, `ml.AL_Algorithms`

### 文件命名
- **格式**: `{Concept_Name}.md` (Pascal_Snake_Case)

| ✅ 正确 | ❌ 错误 | 原因 |
|---------|---------|------|
| `Linear_Regression.md` | `线性回归.md` | 不使用中文 |
| `Support_Vector_Machine.md` | `SVM.md` | 避免纯缩写 |
| `Transformer_Architecture.md` | `transformer.md` | 使用 Pascal Case |
| `BERT.md` | `bert-model.md` | 缩写可用，但不用连字符 |
| `Autoregressive_Model_Part2.md` | `Model(1).md` | 不使用括号 |

---

## 🏷️ YAML Frontmatter 规范

### 模板

```yaml
---
aliases: [中文名称, English Alias]      # 至少一个中文别名
tags:
  - concept                              # 类型标签
  - domain/subdomain                     # 领域标签
related_concepts:
  - "[[Related_Concept]]"                # 至少一个双链
source: "来源 (可选)"
---
```

### 类型标签

| 标签 | 适用场景 |
|------|----------|
| `concept` | 概念、定义 |
| `theorem` | 定理、引理 |
| `method` | 方法论、流程 |
| `algorithm` | 算法 |
| `paper` | 论文笔记 |
| `survey` | 综述 |

### 领域标签

| 领域 | 标签格式 |
|------|----------|
| 线性代数 | `math/linear-algebra` |
| 优化 | `math/optimization` |
| 概率论 | `math/probability` |
| 实分析 | `math/real-analysis` |
| 统计推断 | `stat/inference` |
| 时间序列 | `stat/time-series` |
| ML 算法 | `ml/algorithms` |
| 学习理论 | `ml/theory` |
| 深度学习 | `ml/deep-learning` |
| NLP/LLM | `cs/nlp`, `cs/llm` |
| 信息论 | `cs/information-theory` |
| 计量经济 | `econ/econometrics` |

---

## 🔗 双链规范

### 基本规则

1. **格式**: `"[[Concept_Name]]"` （YAML 中需要引号）
2. **命名**: 使用 `Pascal_Snake_Case`
3. **范围**: 可以链接到同目录或其他目录的笔记

### 常见双链关系

| 关系类型 | 示例 |
|----------|------|
| 前置知识 | `[[Linear_Algebra]]` → `[[Matrix_Decomposition]]` |
| 扩展/进阶 | `[[Word2Vec]]` → `[[BERT]]` |
| 技术细节 | `[[Word2Vec]]` → `[[Negative_Sampling]]` |
| 相关方法 | `[[Word2Vec]]` ↔ `[[GloVe]]` |
| 应用场景 | `[[Transformer]]` → `[[BERT]]`, `[[GPT]]` |

### 跨目录双链示例

```yaml
related_concepts:
  - "[[Word2Vec]]"                    # 同目录
  - "[[Softmax]]"                     # ml.DL_DeepLearning/
  - "[[Cross_Entropy]]"               # cs.IT_InfoTheory/
  - "[[SVD]]"                         # math.LA_LinearAlgebra/
```

---

## 🌲 分类决策树

```
开始
  │
  ├─ 数学基础？
  │   ├─ 线性代数         → math.LA_LinearAlgebra/
  │   ├─ 优化理论         → math.OC_Optimization/
  │   ├─ 概率论           → math.PR_Probability/
  │   └─ 实分析/测度论    → math.RA_RealAnalysis/
  │
  ├─ 统计学？
  │   ├─ 估计/检验        → stat.IN_Inference/
  │   └─ 时间序列         → stat.TS_TimeSeries/
  │
  ├─ 机器学习？
  │   ├─ 算法（回归/分类/集成/核/GLM）→ ml.AL_Algorithms/
  │   ├─ 理论（PAC/VC/泛化）→ ml.TH_Theory/
  │   ├─ 深度学习架构     → ml.DL_DeepLearning/
  │   └─ NLP/LLM 特有     → cs.CL_NLP_LLM/
  │
  ├─ CS 基础？
  │   ├─ 信息论           → cs.IT_InfoTheory/
  │   └─ 算法/数据结构    → cs.DS_Algorithms/
  │
  ├─ 计量经济学？        → econ.EM_Econometrics/
  │
  ├─ 综述/多概念整合？   → 20_StudyNotes/Topics/
  │
  └─ 难以归类？          → _Misc/
```

### 常见边界案例

| 概念 | 正确归属 | 理由 |
|------|----------|------|
| Linear Regression | `ml.AL_Algorithms/` | ML 算法 |
| GLM | `ml.AL_Algorithms/` | 统计建模方法 |
| Kernel Methods | `ml.AL_Algorithms/` | ML 方法 |
| UMVUE / Fisher Info | `stat.IN_Inference/` | 统计推断理论 |
| ARIMA | `stat.TS_TimeSeries/` | 时间序列模型 |
| Transformer | `ml.DL_DeepLearning/` | 通用 DL 架构 |
| BERT | `cs.CL_NLP_LLM/` | NLP 特化模型 |
| MoE | `ml.DL_DeepLearning/` | DL 架构 |
| PAC Learning | `ml.TH_Theory/` | 学习理论 |

---

## ✂️ 原子性检查与拆分

### 拆分信号

当文件满足以下条件时，考虑拆分：

- 行数 > 300 行
- 包含多个独立的主题/方法
- 可以明确分为"理论"和"实现"两部分

### 拆分模式

**模式 1: 概念 + 实现**
```
Concept.md               → 理论、定义、推导
Concept_Implementation.md → 代码、实践
```

**模式 2: 概述 + 细节**
```
Topic_Overview.md        → 概述、索引
Subtopic_1.md            → 细节 1
Subtopic_2.md            → 细节 2
```

**模式 3: 方法对比**
```
Method_A.md              → 方法 A
Method_B.md              → 方法 B
```

### 拆分后的双链

拆分后必须在每个文件的 `related_concepts` 中互相链接：

```yaml
# Word2Vec.md
related_concepts:
  - "[[Negative_Sampling]]"
  - "[[Word2Vec_Implementation]]"

# Negative_Sampling.md
related_concepts:
  - "[[Word2Vec]]"
```

---

## 📑 README 维护

每个目录应有一个 `README.md` 索引文件。

### 模板

```markdown
# {目录名称}

> 简要描述

## 📝 笔记索引

| 文件 | 主题 | 关键概念 |
|------|------|----------|
| `File_1.md` | 主题1 | 关键词1, 关键词2 |
| `File_2.md` | 主题2 | 关键词3, 关键词4 |

## 🗺️ 知识图谱

概念A
  ├─→ 概念B
  │     └─→ 概念C
  └─→ 概念D

## 🔗 跨目录关联

- **理论基础**: `other_dir/` (说明)
- **应用**: `another_dir/` (说明)
```

---

## ⚙️ 操作流程

### 新建笔记
1. 根据决策树确定目标文件夹
2. 创建 `Concept_Name.md`
3. 填写 YAML Frontmatter
4. 编写内容
5. 更新对应 README.md

### 整理已有笔记
1. 确定核心概念
2. 使用决策树分类
3. 检查命名规范
4. 添加 YAML Frontmatter
5. 移动到目标文件夹
6. 检查是否有重复

### 检查单个文件
1. 读取文件内容
2. 检查 YAML Frontmatter（缺少→添加，不完整→补充）
3. 分析内容，识别相关概念
4. 添加 related_concepts 双链
5. 检查文件长度（>300行→建议拆分）
6. 确认命名规范

### 拆分大文件
1. 分析文件结构，确定拆分点
2. 创建新文件
3. 为每个新文件添加 YAML
4. 建立互相的双链
5. 删除原文件
6. 更新 README.md

---

## ✅ 质量检查清单

### 文件级
- [ ] 英文 Pascal_Snake_Case 命名
- [ ] 无中文、括号、多余符号
- [ ] YAML 包含 aliases, tags, related_concepts
- [ ] 内容聚焦单一概念
- [ ] 双链指向存在的文件

### 目录级
- [ ] README.md 存在且更新
- [ ] 无重复文件
- [ ] 双链双向（A→B 则 B→A）
- [ ] 目录规模 5~30 篇

### 仓库级
- [ ] INDEX.md 已更新
- [ ] `_Misc/` 定期清理
- [ ] 无孤立文件

---

## 📊 维护周期

| 周期 | 任务 |
|------|------|
| 每次添加 | 更新 README, 添加链接 |
| 每周 | 检查重复, 补充 YAML |
| 每月 | 更新 INDEX, 清理 _Misc |

---

## ❓ 常见问题

### Q: 文件已有 `#tag` 形式的标签怎么办？
保留行内标签，但确保 YAML 中也有规范的 tags。

### Q: 概念难以归类到单一目录？
放在**主要应用场景**对应的目录，用双链关联其他目录。

### Q: 双链目标不存在怎么办？
可以先添加双链，Obsidian 会显示为未创建的链接，后续可创建。

### Q: 是否需要删除原文件中的 `#tag`？
不需要，保持兼容性。

### Q: 综述类内容放哪里？
放在 `20_StudyNotes/Topics/`，不要放在 `10_AtomicKnowledge/`。

---

**版本历史**
- v5.0 (2026-01-17): 合并 SOP 和 SOP_for_AI，统一为单一文档
- v4.0 (2026-01-17): 扩展为全局知识库 SOP，添加 20_StudyNotes 结构说明
- v3.1 (2026-01-17): 添加 AI 协作 SOP 和笔记模板
- v3.0 (2026-01-17): 扁平化重构，合并 ML 算法/方法/统计方法论
