# 待处理文件夹说明

本文件夹包含从 `10_Projects/` 提取出来的、需要进一步处理的内容。

---

## 📁 文件夹结构

```
_ToProcess/
├── README.md                    ← 本说明文件
├── Notion_Inbox/                ← 应移至 Notion 的内容
│   ├── MinPPlus/
│   │   ├── 实验记录/           ← 实验日志、任务规划
│   │   │   ├── MinPPluss实验日志_0718_01.md
│   │   │   ├── MinPPlus 实验相关情况汇总.md
│   │   │   ├── 0719 行动指南.md
│   │   │   └── figs/           ← 实验截图
│   │   └── GPT/                ← LLM 对话记录
│   │       └── ChatGPT-代码审核与修复建议.md
│   │
│   └── ConformalPrediction/
│       └── 工作记录.md          ← 项目工作流水账
│
└── Archive_4PT/                 ← 4PT 项目归档
    └── 4PT/                     ← 完整项目文件夹
        ├── 4pt Code Book (Old Ver.).md
        ├── General Notes.md
        ├── Brief Introduction of LLM Techniques on 4PT Analysis.md
        ├── Codebook four problem types_Master File.pdf
        ├── Beginning/          ← 早期 LLM 研究对话
        │   ├── Claude.md
        │   ├── StartUp Research ChatGPT.md
        │   └── StartUp Research Gemini.md
        └── archive/
            └── 加一段 **Q0 出圈校验**（与 4PT 宇宙一致）.md
```

---

## 🎯 处理建议

### 1️⃣ **Notion_Inbox/** 
**目标**: 移至 Notion 项目管理系统

#### MinPPlus/
- ✅ **实验记录/** 
  - 内容：实验日志、实验汇总、任务行动指南
  - 处理：在 Notion 中创建 "MinPPlus 项目" 页面，将这些内容作为 Database 或子页面导入
  - 类型：任务管理 + 实验追踪
  
- ✅ **GPT/** 
  - 内容：与 ChatGPT 的代码审核对话
  - 处理：可作为参考资料归档到 Notion 的 "AI 对话记录" 部分
  - 类型：参考资料

#### ConformalPrediction/
- ✅ **工作记录.md**
  - 内容：项目工作流水账
  - 处理：导入到 Notion 的 "ConformalPrediction 项目" 页面
  - 类型：项目日志

---

### 2️⃣ **Archive_4PT/**
**目标**: 归档到 `40_Archives/` 或保留在 `10_Projects/`（如果项目仍活跃）

#### 4PT 项目
- **性质**: 早期独立研究项目
- **内容**: 
  - 项目笔记 (General Notes, Code Book)
  - LLM 技术应用介绍
  - 早期探索性对话 (Beginning/ 文件夹)
  
- **处理选项**:
  
  **选项A (推荐)**: 如果项目已完成/暂停
  ```bash
  mv _ToProcess/Archive_4PT/4PT/ 40_Archives/
  ```
  
  **选项B**: 如果项目仍在进行
  ```bash
  # 保留在原位置，但整理笔记结构
  # 将 Beginning/ 中的 LLM 对话移至 Notion
  ```

---

## ✅ 完成后的清理

处理完上述内容后，执行以下清理：

```bash
# 1. 删除原 10_Projects 中的对应文件
rm -rf 10_Projects/MinPPlus/实验记录
rm -rf 10_Projects/MinPPlus/GPT
rm -rf 10_Projects/ConformalPrediction/工作记录.md
rm -rf 10_Projects/4PT

# 2. 清理空文件夹
rm -rf 10_Projects/Autoloss        # 已空
rm -rf 10_Projects/MinPPlus        # 已空（移除实验记录后）
rm -rf 10_Projects/ConformalPrediction  # 已空（移除工作记录后）

# 3. 删除此临时文件夹
rm -rf _ToProcess
```

---

## 📊 文件统计

- **Notion_Inbox**: 5个文件 + 1个图片文件夹
- **Archive_4PT**: 8个文件 + 1个PDF

**总计**: 约 13 个文件需要处理

---

## 🗓️ 创建时间
2026-01-17

**操作记录**: 从 `10_Projects/` 提取任务管理和归档内容

