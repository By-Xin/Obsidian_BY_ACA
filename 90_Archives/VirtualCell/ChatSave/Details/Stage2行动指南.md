很好，阶段 1 现在看起来是“可收尾并进入下一步”的状态。基于你提供的修订版日志与脚本输出，我给出简短评审与是否放行的结论，然后提供阶段 2 的一份可执行行动指南。

---

## 阶段 1 代码与结果评审结论

**你已经满足了开展阶段 2 的所有先决条件：**

* **基因宇宙与顺序**：标准化后的 AnnData 使用官方 18,080 基因且顺序完全对齐，类型为 float32，非负，形状与预期一致。修复报告明确指出“基因顺序匹配率 100.0%”，并保存了对齐后的 H5AD 与基因顺序快照，便于后续一致性校验。 &#x20;
  注：官方提交规范明确要求 .var 含**恰好** 18,080 个基因，且提交内容须为 int 计数或 log1p 量纲、X 为 float32、.obs 含 `target_gene`、并包含控制细胞；你当前的数据与量纲符合这些要求（我们用 log1p）。

* **拆分策略**：验证基因集合固定，验证扰动在训练集中完全不存在，控制细胞的分配合理，训练与验证的基因数与顺序一致，无交叉索引，统计与报告文件齐备。

* **验证集真值**：已生成与验证集一一对应的 `ground_truth_fixed.h5ad` 与 `pseudobulk_reference_fixed.csv`，并再次锁定基因顺序，形状 `(6115, 18080)` 与拆分一致。

**结论**：可以进入阶段 2。

---

## 阶段 2 行动指南：构建并产出“cell-mean baseline”

> 目标：产出一个与 VCC 规范对齐的 baseline 预测文件，并用于本地评分与官网交叉验证。

### A. 基线模型定义与与 VCC 对齐

1. **Baseline 定义（与官网一致）**
   “cell-mean baseline”按**训练数据**对**所有扰动细胞**逐基因取平均，得到一个固定的 18,080 维向量 $\mu_{\text{pert}}$。该基线对**任意扰动**的预测都是同一个 $\mu_{\text{pert}}$；控制细胞可另取 $\mu_{\text{ctrl}}$（训练集中 non-targeting 细胞的逐基因均值），以保持量纲自洽。官网说明：baseline 分数基于 cell-mean baseline 预先在 Training 数据上计算，并用于缩放排行榜分值。
   数学上，令训练集中所有非控制细胞集合为 $\mathcal{I}_{\text{pert}}$，控制细胞集合为 $\mathcal{I}_{\text{ctrl}}$，训练数据矩阵为 $X$。则
   $\mu_{\text{pert}}=\frac{1}{\lvert \mathcal{I}_{\text{pert}} \rvert}\sum_{i\in \mathcal{I}_{\text{pert}}} X_{i,\cdot}, \quad \mu_{\text{ctrl}}=\frac{1}{\lvert \mathcal{I}_{\text{ctrl}} \rvert}\sum_{i\in \mathcal{I}_{\text{ctrl}}} X_{i,\cdot}.$
   对验证集任一细胞 $j$ 的预测：

   $$
   \hat y_j=
   \begin{cases}
   \mu_{\text{ctrl}}, & \text{若 } \texttt{target\_gene}=\text{non-targeting},\\
   \mu_{\text{pert}}, & \text{否则。}
   \end{cases}
   $$

2. **提交文件硬性约束复核**（提交到官网或用 cell-eval 的 `.vcc` 包装器时必须满足）：

   * `.var` 必须含**恰好** 18,080 基因，顺序与训练一致。
   * `.obs` 必须含 `target_gene` 列，且包含 `non-targeting` 控制行。
   * 总细胞数 $\le 100{,}000$。
   * `.X` 类型为 float32。
   * 计数须为**整数**或**log1p**；不要提交“先归一化后未对数”的矩阵。
   * 官方建议用 `cell-eval` 的 `prep` 工具生成 `.vcc`，确保格式严格合规。&#x20;

> 你已在阶段 1 中确保 18,080 基因、log1p 量纲、float32 与 `target_gene` 存在；基因顺序也已固化在多个成果文件中。 &#x20;

---

### B. 数据输入与产物输出

* **输入**：

  * 训练集：`splits/train_fixed.h5ad`（或你保存的等价路径）。
  * 验证集：`splits/val_fixed.h5ad`。
  * 真值：`truth/ground_truth_fixed.h5ad` 与 `truth/pseudobulk_reference_fixed.csv`。

* **输出**：

  * Baseline 预测 H5AD：`predictions/baseline_pred_val_fixed.h5ad`（与验证集 obs 等长，`.var` 顺序与训练一致，`.X` 为 float32）。
  * 供官网的 `.vcc` 文件：`predictions/baseline_pred_val_fixed.vcc`（通过 `cell-eval prep` 生成）。
  * 本地评分报告：`reports/stage2_baseline_eval_fixed.txt` 与 `reports/stage2_baseline_eval_fixed.json`（包含 DES、PDS、MAE 以及按扰动的明细）。

---

### C. 步骤清单（逐项核对即可执行）

1. **环境准备**

   * 安装依赖：`anndata`, `scanpy`, `numpy`, `pandas`, `cell-eval`。
   * 固定随机种子，记录版本信息。

2. **一致性与前置检查**

   * 读取 `splits/train_fixed.h5ad` 与 `splits/val_fixed.h5ad`。
   * 校验两者 `.var_names` 与 `truth/gene_names_order_fixed.csv` 完全一致。若不一致，立即停止并对齐。
   * 校验 `.X.dtype` 为 float32，且最大值量级与阶段 1 验证报告一致（log1p 范围）。

3. **计算均值模板**

   * 在训练集上筛选 `target_gene != 'non-targeting'` 的所有细胞，计算 $\mu_{\text{pert}}$。
   * 在训练集上筛选 `target_gene == 'non-targeting'` 的所有细胞，计算 $\mu_{\text{ctrl}}$。
   * 记录二者的 L1 与 L2 范数、与验证集控制的均值差，作为“量纲健康度”记录，防止退化预测。

4. **生成验证集逐细胞预测矩阵**

   * 为验证集中每个细胞按上面的判别填入 $\hat y_j$。
   * 维度应为 `(n_val_cells, 18080)`，dtype 为 float32。
   * `.obs['target_gene']` 与验证集保持一一对应，不改动行顺序；复制验证集 `.obs` 其余列是允许的，但不是必须。
   * 保存为 `predictions/baseline_pred_val_fixed.h5ad`。

5. **用 `cell-eval` 进行 `.vcc` 打包**

   * 使用 `cell-eval` 的 `prep` 功能对上述 H5AD 进行合规性检查与封装，生成 `.vcc`。这一步会再次强制检查 18,080 基因、`target_gene` 列、dtype、细胞数上限等提交规范。&#x20;

6. **本地基线评分**

   * 评分指标与 VCC 对齐。核心 3 个指标为：

     * 差异表达分数 DES：对每个扰动 $k$，若真实 DE 基因集合为 $G_{k,\text{true}}$，预测集合为 $G_{k,\text{pred}}$，且真集合大小为 $n_{k,\text{true}}=\lvert G_{k,\text{true}}\rvert$，则
       $DES_k=\frac{\lvert G_{k,\text{pred}}\cap G_{k,\text{true}}\rvert}{n_{k,\text{true}}}.$
       取 $k$ 的平均作为总体 DES。
     * 扰动判别分数 PDS：先将**伪体量**（pseudobulk）的 L1 距离排序
       $d_{pt}=d_{L1}(\hat y_p, y_t)\ \big\vert_{\text{sort by }t},$
       取真扰动 $t=p$ 的秩并归一化
       $PDS_p=1-\frac{\operatorname{argind}\{d_{pt}\}_{t=p}-1}{N}.$
       再对 $p$ 求平均得到总体 PDS。&#x20;
     * 平均绝对误差 MAE：对每个扰动 $k$ 计算伪体量之间的 $MAE_k$，再对 $k$ 平均得到整体 $MAE$。

   * 排行榜的总体分数为三者的**改进量**相对 baseline 的平均：
     $S=\frac{1}{3}\Bigl(DES_{\text{scaled}}+PDS_{\text{scaled}}+MAE_{\text{scaled}}\Bigr),$
     其中
     $MAE_{\text{scaled}}=\frac{MAE_{\text{baseline}}-MAE_{\text{prediction}}}{MAE_{\text{baseline}}}.$
     DES 与 PDS 的缩放也是“相对 baseline 的改进”。建议用 `cell-eval` 的实现以避免走样。&#x20;

   * **重要提醒**：官网用于缩放的 baseline 是基于**官方 H1 Training** 的固定常数；你本地的 K562“模拟 baseline”用于**相对比较**完全没问题，但**数值不会与官网缩放完全一致**。这也是我们在阶段 3 要做“对分”的原因。

7. **产出与质检**

   * 产出：`.h5ad`、`.vcc`、本地评分的 `.txt` 与 `.json`。
   * 质检：

     * `.var_names` 全等于 `truth/gene_names_order_fixed.csv`。
     * `.X.dtype` 为 float32。
     * `.obs['target_gene']` 不缺失，包含 `non-targeting`。
     * `n_obs` 与验证集一致。
     * 本地 DES、PDS、MAE 处于合理区间：cell-mean baseline 的 DES 与 PDS 通常偏低，MAE 明显高于强模型；若出现反常高分，优先检查 log1p 量纲、伪体量聚合与 DE 集合阈值设定。

8. **可选的健壮性小实验（便于后续 ablation）**

   * 用“包含控制细胞”的全体均值代替 $\mu_{\text{pert}}$，对比分数差异，记录到报告。
   * 仅用“验证集推荐样本数”随机下采样训练集重新估计 $\mu$，检视分数稳定性。
   * 记录 $\lVert \mu_{\text{pert}}-\mu_{\text{ctrl}}\rVert_1$ 与 $\lVert\cdot\rVert_2$ 以监控量纲漂移。

---

## 阶段 2 交付物清单与通过标准

* `predictions/baseline_pred_val_fixed.h5ad` 与 `predictions/baseline_pred_val_fixed.vcc` 已生成且通过 `cell-eval prep` 检查（18,080 基因、float32、含控制、cells ≤ 100k、log1p）。
* `reports/stage2_baseline_eval_fixed.txt` 与 `...json`：包含按扰动与总体的 DES、PDS、MAE，以及伪体量构建与 DE 集合阈值说明。
* 复核记录：基因顺序与 `truth/gene_names_order_fixed.csv` 全等；预测与真值的 obs 中 `target_gene` 的集合一致。

满足以上三条即可进入阶段 3（与官网评分的一致性校验）。阶段 3 里，我们会把 baseline `.vcc` 上传官网拿官方分，随后对比你本地的实现与 `cell-eval` 的差异，逐项定位并消除偏差。关于 VCC 的三大指标定义与合成分在官方文档与 Cell 评论文中有清晰公式，上面已与本地实现对齐。&#x20;

---

## 小结与下一步

* 阶段 1 已经达标，可以开干阶段 2。
* 阶段 2 的关键不是“模型好坏”，而是“**规范正确**”：文件结构、基因宇宙、量纲、obs 列、浮点类型与控制细胞。VCC 的提交规则对这些很严格，`cell-eval prep` 是最保险的把关工具。
* 一旦产出合格的 baseline `.vcc` 并在本地评分跑通，我们就能以它作为**缩放参照**与**集成回归测试**的锚点，进入阶段 3 做官网分数对齐与误差剖析。

如果你希望，我可以基于你现有的目录结构给出一版**极简**的阶段 2 驱动脚本骨架与“十行以内”的均值构建片段，直接落盘到 `predictions/` 与 `reports/`。
