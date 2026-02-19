好的，我们就按“**不做本地评分，只做本地打包，然后用官网评测**”的路线推进。我给你一套**最短路径的提交流水线**，以及在不改大框架的前提下、可逐步加入的小改动（优先 MAE 辅助、再到 DES 辅助、再到 PDS 对比），每一步都有**明确的交付物**与**合规检查点**。这样你可以快速形成“训练 → 生成 → 打包 → 官网评测”的闭环，后续只需替换 checkpoint 或打开某个改动，再次打包并提交即可。

---

## A. 你现在已经具备的基础与目标

- 你已用 `starter.toml` 成功训练了 STATE（`model=state_sm`，加载了 `ESM2_pert_features.pt`，并按示例将数据拆成 train/val/test 场景），从输出的目录结构看**已跑通**，后续可以直接用生成的 checkpoint 做推理与打包。
    
- 官方提供了验证阶段每个扰动建议生成的**细胞数**和**中位 UMI**（`pert_counts_Validation.csv`，50 个目标基因），这是你在**生成/打包**时需要用到的清单。
    
- 提交物必须满足 `.vcc`/H5AD 的**硬约束**：`.var` 的基因集合必须是 18,080 个且顺序一致；`.obs['target_gene']` 存在；包含 `non-targeting` 控制细胞；`.X` 为 `float32`，数值要么是整数 counts，要么是 **log1p-normalized**；总细胞数不超过 **100,000**。这些要求以**官网评分页/打包说明**为准。
    

---

## B. “不本地评分”的**最短提交流水线**

下面只做**推理与打包**（不做本地打分）。你要关心的是：怎么从 checkpoint 产出一个合规的 H5AD，再用官方工具**打包**为 `.vcc`，最后上传官网评测。

### B1. 确定“要预测的扰动”与“每个扰动的细胞数”

- 用官方 `pert_counts_Validation.csv` 做“目标扰动清单”。每个目标基因 kk 都有建议的 nkn_k（细胞数）与 `median_umi_per_cell`。如果总和可能超过 100,000，需要按比例缩放：
    
    nkfinal  =  ⌊α nkrec⌋,α  =  min⁡ ⁣(1,  100,000−nctrl∑knkrec).n_k^{\text{final}} \;=\; \left\lfloor \alpha \, n_k^{\text{rec}} \right\rfloor,\quad \alpha \;=\; \min\!\Bigg(1,\; \frac{100{,}000 - n_{\text{ctrl}}}{\sum_k n_k^{\text{rec}}}\Bigg).
    
    其中 nctrln_{\text{ctrl}} 为你计划放入的 `non-targeting` 控制细胞数（建议保留 5%–10% 预算，见下条）。
    

### B2. 准备 `non-targeting` 控制细胞

- 训练数据中有 **38,176** 个 `non-targeting` 控制细胞。官方允许你在提交时“预测控制细胞”或“直接从训练集拷贝控制细胞表达”进入提交文件（二选一），后者实现最简单、且**合规则**。
    
- 由于总上限 100,000，你可以**抽样**一部分控制细胞（例如 5,000 个，或按预算决定），并把 `.obs['target_gene']` 填成 `non-targeting`。这对差异表达统计是有意义的（作为对照）。
    

### B3. 用 STATE 生成“受扰细胞”的表达

- **输入**：从未扰动对照细胞中采样一个“控制池”（按 `starter.toml` 的训练方案，取与你要预测的扰动相匹配的背景/细胞类型/批次的对照更稳健）。**扰动条件**来自你已加载的 `ESM2_pert_features.pt`。
    
- **输出**：对每个目标基因 kk，生成 nkfinaln_k^{\text{final}} 个**受扰细胞**的表达向量。
    
- **数值域的选择**（二选一，均合规）：
    
    - **方案 A（推荐，简单稳妥）**：输出 **log1p-normalized** 的表达矩阵，`dtype=float32`。这样可以**不必**去拟合整数 UMI 分布，也避免不必要的反归一化误差（官方明确允许 log1p）。
        
    - **方案 B（进阶，可贴近 `median_umi`）**：若你后续加入 ZINB/NB 解码，可按 `median_umi_per_cell` 生成近似的 UMI counts，再 `float32` 保存为**整数计数**矩阵（或直接保存 counts，而不是 log1p）。这会涉及计数采样/标定，对 MAE/PDS 可能有益，但实现复杂（放到后续迭代）。
        

> 关键点：无论 A 或 B，**保证 `.var` 的基因顺序与训练集完全一致**（18,080 个），这是评分与提交的硬要求。最保险的做法是**直接继承训练集 AnnData 的 `.var` 与索引顺序**来构造你的预测 AnnData。

### B4. 组装预测 AnnData（单个 H5AD 文件）

- `.X`：拼接所有受扰细胞与控制细胞，`dtype=float32`；值为 log1p 或 counts 的二者之一，不要混用。
    
- `.obs`：至少包含列 `target_gene`，其值为目标基因符号或 `non-targeting`（对照）。如方便，也可附带 `batch` 等列，但不是硬要求。
    
- `.var`：严格使用 18,080 基因集合与顺序（与训练 `.var` 一致）。
    
- **总细胞数上限**：受扰 + 对照之和 ≤ 100,000。
    

### B5. 用官方工具**打包**为 `.vcc`（只用打包，不用本地评分）

- 官方提供的 `cell-eval` 工具链既支持评分，也支持**准备/打包**（合规性检查 + 封装）。你现在可以只用**准备/打包**这一部分，确保结构与字段合规。
    
- 打包完成得到 `.vcc` 文件后，上传官网进行评测即可。
    

> **强制自检清单**（提交前最后 5 项）：
> 
> 1. `.var` 中基因数正好 18,080，顺序与训练一致。
>     
> 2. `.obs['target_gene']` 存在，且包含 `non-targeting`。
>     
> 3. `.X` 为 `float32`，且全体要么是 log1p、要么是 counts。
>     
> 4. 总细胞数 ≤ 100,000。
>     
> 5. 你的目标扰动列表与 `pert_counts_Validation.csv` 一致（或按预算缩放后的一致版本）。
>     

[[STATE 0911_B部分修改建议]]


---

## C. 沿用官网评测的**快速迭代梯子**（不改框架，只替换 checkpoint）

你可以保持上述打包流程不变，只替换“生成阶段”使用的 checkpoint（或训练配置），来进行有控制的 A/B 提交对比。建议按下面顺序逐步加入改动：

### C1. 直接用你当前的 STATE checkpoint（Baseline 提交 

- 不改训练代码，直接推理→打包→提交，得一套“基线分数”。
    
- 这一步的意义：确保你的打包与提交链路完全合规，得到一个可对照的**官方分数基线**。
    

### C2. 在训练中加入“伪 bulk 的 MAE 辅助损失”（提交 
- 动机：MAE 指标要求 ℓ1\ell_1 级别的**整体数值精度**；STATE 原生以 MMD 为主，可能对数值精度关注不够。
    
- 做法：在训练 loop 中，令每个扰动的预测单细胞取均值为 y^bulk\hat{y}_{\text{bulk}}，与真实伪 bulk ybulky_{\text{bulk}} 做 ℓMAE=∥y^bulk−ybulk∥1\ell_{\text{MAE}} = \lVert \hat{y}_{\text{bulk}} - y_{\text{bulk}} \rVert_1；总损失形如
    
    L  =  LMMD  +  λMAE LMAE.\mathcal{L} \;=\; \mathcal{L}_{\text{MMD}} \;+\; \lambda_{\text{MAE}} \, \mathcal{L}_{\text{MAE}}.
- 预期：**MAE 显著回落**，PDS 稳定，DES 有时也会受益（伪 bulk 更准）。
    
- 流程不变：重新训练→用新 ckpt 推理→打包→提交。
    

### C3. 再加入“差异基因识别”辅助头（提交 

- 动机：DES 评的是“显著差异基因集合”的重合度；增加一个**每基因二分类**的辅助头（是否差异）能让模型对显著基因更敏感。
    
- 做法：预先用训练集内同样统计流程（Wilcoxon + 多重校正）得每个扰动的真值差异基因集合，然后训练一个输出每基因显著概率的头，配合 BCE/Focal Loss；总损失
    
    L  =  LMMD+λMAELMAE+λDELDE.\mathcal{L} \;=\; \mathcal{L}_{\text{MMD}} + \lambda_{\text{MAE}}\mathcal{L}_{\text{MAE}} + \lambda_{\text{DE}}\mathcal{L}_{\text{DE}}.
- 预期：**DES 提升**，MAE 不受损，整体分数更稳。
    
- 流程不变：训练→推理→打包→提交。
    
- 备注：评测侧的 DES 定义（差异集交并比、阈值与集合裁剪规则）与官网一致即可，不需要你本地算分，但训练前得到标签时要对齐官方统计流程。
    

### C4. 最后加入“扰动区分”的排序/对比损失（提交 

- 动机：PDS 看“预测伪 bulk 和真正对应扰动的伪 bulk 是否排第一”；引入一个**三元组/排序损失**直接优化“排第一”。
    
- 做法：令
    
    Lrank=max⁡ ⁣(0, d(y^p,yp)−d(y^p,yq)+δ),L_{\text{rank}} = \max\!\bigl(0,\, d(\hat{y}_p,y_p) - d(\hat{y}_p,y_q) + \delta \bigr),
    
    其中 pp 为当前扰动，qq 为 batch 内随机负扰动，dd 可用 L1L_1 或 1−1-cosine，相容 PDS。
    
- 预期：**PDS 提升**，整体分数再上一档。
    
- 整体损失
    
    L  =  LMMD+λMAELMAE+λDELDE+λrankLrank.\mathcal{L} \;=\; \mathcal{L}_{\text{MMD}} + \lambda_{\text{MAE}}\mathcal{L}_{\text{MAE}} + \lambda_{\text{DE}}\mathcal{L}_{\text{DE}} + \lambda_{\text{rank}}\mathcal{L}_{\text{rank}}.

> 建议初始权重取 λMAE=1, λDE=0.5, λrank=0.5\lambda_{\text{MAE}}=1,\ \lambda_{\text{DE}}=0.5,\ \lambda_{\text{rank}}=0.5；后续根据**官网分数的相对提升**来微调。

---

## D. 两个实现上的抉择（都与提交合规对齐）

### D1. 预测输出到底用 log1p 还是 counts？

- **log1p**：实现最简，直接把 STATE 输出当作 log1p，`float32` 存 `.X` 即可；合规且稳定。
    
- **counts**：若你引入 ZINB/NB 解码并按 `median_umi_per_cell` 做采样标定，可能更贴近真实测序深度（对 MAE/PDS 有潜在益处），但实现复杂、调参多。
    

> 建议：先用 **log1p** 打通“训练→打包→官网评测”的闭环；等你完成 C2/C3/C4 的损失对齐后，再考虑 counts 版本。

### D2. `non-targeting` 控制细胞用预测还是拷贝？

- 官方明确允许**拷贝训练集的控制细胞**进入提交。这样实现最简单且规避无谓风险。
    
- 若你有充裕时间，也可用 STATE 生成“非扰动条件”的预测对照，但实操风险更高，且通常收益不明显。
    

---

## E. 少样本 H1 适配与批次效应（等你要上 H1 时再做）

一旦你需要在 H1 背景生成预测，建议先做**最小化 few-shot 适配**来校正分布差异（只更新“背景标记/输出偏置”等极少量参数），再做推理与打包；这是官方设定强调的“跨上下文泛化”，少样本适配更稳妥。  
另外，注意控制**批次效应**：例如统一 log1p 归一化尺度、在 `.obs` 中保留 `batch` 便于后续检查；如果用 counts，需要按 `median_umi_per_cell` 做合理标定。

---

## F. 你今天就可以执行的 5 步

1. **确定清单**：读取 `pert_counts_Validation.csv`，得到目标基因列表与建议 `n_cells` 与 `median_umi_per_cell`；按上限 100,000 与你预留的 `non-targeting` 数量计算缩放系数 α\alpha。
    
2. **生成预测矩阵**（log1p 方案）：用你现有 checkpoint，对每个目标基因生成 nkfinaln_k^{\text{final}} 个受扰细胞；从训练集拷贝一批 `non-targeting` 细胞。
    
3. **组装 AnnData**：`.X=float32`；`.obs['target_gene']` 正确填充；`.var` 直接继承训练集顺序（18,080 基因）。
    
4. **打包 `.vcc`**：用官方 `cell-eval` 的准备/打包命令（只做合规检查，不做本地评分）。重点检查上限与字段。
    
5. **上传评测**：得到一版官方分数，作为后续 C2/C3/C4 的对照基线。
    

---

### 最后提醒（只关注合规、避免踩坑）

- `.var` 顺序与集合不对，**三项指标都会失真**；务必从训练集直接拷贝 `.var`。
    
- `.X` `dtype` 必须 `float32`，且**二选一**（全部 log1p 或全部 counts），不要混用。
    
- 总细胞数 ≤ 100,000；别忘了把 `non-targeting` 也计入总数。
    
- 打包前最后一眼：核对 `target_gene` 命名拼写、确保 `non-targeting` 存在（字面一致）。
    

如果你愿意，下一步我可以把“**生成阶段**的操作细节”拆解为一份伪代码级别的任务清单（不写具体代码），例如：如何从对照细胞池采样、如何组织 batch 推理、如何构造 AnnData 对象与字段、如何把训练集的 `.var` 注入到你的预测对象。这会让你的工程同学照单执行，尽快形成第一版可提交文件。