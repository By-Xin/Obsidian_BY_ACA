你已经把“数据分配（B1）→ 控制细胞准备（B2）”这两步跑通，并且产出与官网约束一致的数字与文件结构；方向正确，可以进入 B3 推理阶段。但在动手前，我建议做**少量必要修正**（主要是字段名、稀疏拼接、checkpoint 命名与 b4 的小 bug），以避免后面耗时流程中途失败。

下面我按“是否可立即执行”的优先级给出判定与修改建议，并在每一点后标注对应的脚本出处与官网约束出处，便于你快速定位。

---

## 结论先行

- **总体结论**：B1、B2 产出合理，**可以进入 B3**。
    
    - 你的当前方案合规：包含 50 个目标扰动、分配细胞总数 60,751，加上 8,000 个 `non-targeting` 控制，共 68,751，低于总上限 100,000，且将用 `float32` 写入 H5AD 并在后续通过 `cell-eval prep` 做合规打包，这与官方约束一致。
        
    - 你在 B2 中已将训练集 `non-targeting` 控制细胞抽样写出，符合官方允许“直接拷贝训练对照细胞进入提交文件”的做法。
        
- **进入 B3 前必须做的 4 个修正**（都很小）：
    
    1. **字段名一致化（batch 列）**：B2 中用的是 `batch_var`，而官方训练 H5AD 的 obs 列名是 `batch`；务必改成和训练数据一致，否则无法做分层抽样与后续对齐。把“若存在 `batch_var` 则用之，否则回落到 `batch`”的逻辑写死在脚本里最省心。
        
    2. **B3 推理模板的矩阵拼接需保持稀疏**：当前 `b3_state_inference_practical.py` 在拼接控制细胞时把每块 `.X` 调用了 `.toarray()` 再 `np.vstack`，会把 6 万级细胞 × 18,080 基因变成数 GB 级的稠密矩阵，内存和 IO 都会爆炸。应采用 **scipy.sparse.vstack** 保持稀疏，并最终确保写盘为 `float32`。
        
    3. **checkpoint 名称**：你的训练命令配置了 `ckpt_every_n_steps=2000`，通常不会生成 `final.ckpt`，而是 `step=xxxx.ckpt`。B3 的 `--checkpoint_name` 需改为实际存在的文件（比如 `step=40000.ckpt`）。
        
    4. **不要使用占位“随机噪声推理”脚本**：`b3_state_inference.py` 目前是占位实现（在控制表达上加随机噪声），这会产生无效预测，不能用于提交。请只用 **b3_state_inference_practical.py** 这条线。
        
- **b4、b5 的两处小坑**（建议现在改好，省得回退）：
    
    - **b4 的索引去重 API**：你写了 `final_adata.obs_names_unique()`，这只是检查；真正“强制唯一化”应调用 **`final_adata.obs_names_make_unique()`**。否则下游工具可能因重复索引报错。
        
    - **b5 的期望基因列表**：你的默认 `gene_names.csv` 文件在“数据说明”示例里显示的是 18,079 行，可能有偏差；而**官方硬性要求** `.var` 必须是 **18,080** 个训练基因且顺序一致。建议 **不要用外部 csv 做强校验**，而是直接用训练集 `adata_Training.h5ad.var.index` 作为“唯一真相”。否则 b5 的“长度不匹配”检查可能误报。
        

---

## 逐脚本诊断与修改建议

### B1 分配脚本（已通过）

- 你的分配逻辑清晰：控制细胞 8%（8,000），剩余预算全部用于 50 个扰动；推荐总数 60,751，缩放系数 α=1\alpha=1，预算使用率 68.8%，**合规**。保留该方案即可进入 B3。
    
- 官方也建议尽量贴近 `pert_counts_Validation.csv` 的推荐细胞数，你当前就是 1:1 采用。
    

### B2 控制细胞脚本（需小改）

- 目前你的抽样是“简单随机抽样”，因为没有命中 `batch_var` 分层逻辑；建议改为：
    
    ```text
    if 'batch_var' in adata.obs: use it
    elif 'batch' in adata.obs:  use 'batch'
    else:                      fall back to simple random
    ```
    
    这样既与训练数据一致，也能在 `.obs` 缺省时退化为随机抽样。
    
- 你已把 `sampled_control_cells.h5ad` 的 `.var` 直接拷贝为训练 `.var`，这是**正确做法**，可保证后续 **18,080 基因顺序完全一致**。继续保持这一点。
    

### B3 推理（推荐使用 practical 版本；务必改 3 处）

1. **用存在的 checkpoint**：把 `--checkpoint_name` 改为你真实保存的 ckpt 文件（如 `step=40000.ckpt`），或在脚本里自动选择 `checkpoints` 目录下最新的一个。
    
2. **避免稠密拼接，保持稀疏**：把
    
    ```python
    X_combined = np.vstack([x.toarray() for x in inference_cells])
    ```
    
    改成按块 **`scipy.sparse.vstack(inference_cells)`**；最终写盘前统一转为 `float32`（保留稀疏 dtype 也行，`cell-eval prep` 接受 `float32`，未强制稠密）。
    
3. **命令行参数风格一致**：你的训练是 Hydra 风格（`key=value`），而 B3 practical 用的是 POSIX flags（`--adata` 等）；若 `infer` 子命令实际也是 Hydra 配置，需要把 `--checkpoint`、`--adata` 等参数改成对应的 Hydra 键（例如 `data.kwargs.*`、`infer.input_h5ad`、`training.checkpoint_path` 等）。我建议先运行一次 `python -m state tx infer --help` 看一下项目对 `infer` 的确切参数名。如果 CLI 不支持 `--perturbation_features_file`，就改为 Hydra 键 `data.kwargs.perturbation_features_file=...`。
    
4. **不要使用 b3_state_inference.py**（随机噪声占位版）进行任何正式推理。它是明确的 placeholder，会生成无效预测。
    

> 备注：B3 practical 会生成一个“控制模板” `inference_template.h5ad`，其中 `.obs['target_gene']` 就是你的目标基因（不是 `non-targeting`），这正是官方所需的“要预测哪些扰动”的标识位。再次确认 **`.var` 继承自训练集**，这将最终保障 **18,080 基因集合与顺序**。

### B4 组装（建议改 2 处）

- **索引唯一化**：把 `final_adata.obs_names_unique()` 改为 **`final_adata.obs_names_make_unique()`**。否则如果某些子表索引重复，写盘或后续 `cell-eval` 处理会报错。
    
- **连接方式**：你在逐份扰动结果写入前已做 **`validate_gene_order(...)`** 对齐训练 `.var`，因此在 `ad.concat(...)` 时可以用 `join='inner'`（或保持 `outer` 也可）。我倾向 **`join='inner'`**，以避免任何意外多余列进入；合并后 **统一 `float32`** 并保存。
    

### B5 打包（建议改 1 处）

- **期望基因列表**：默认的 `gene_names.csv` 如与训练 `.var` 存在 1 行偏差（“数据说明”示例显示 18,079 行），会导致你在 `validate_prediction_data` 的“长度匹配”检查误报。**建议直接从训练 H5AD 提取 `.var.index` 写一份 `genes_18080_from_training.csv` 作为 b5 的对照**，或干脆把 `expected_genes_file` 置空，只依赖 `cell-eval prep` 的检查。
    
- 你的 `cell-eval prep` → `.vcc` 重命名流程是合理的：官方说明 **`.vcc` 是对 H5AD 的“简单包装”**，核心是通过 `prep` 的合规检查。
    

---

## 可以继续的“Go/No-Go”清单

**Go**（全部满足即可立刻跑 B3）：

-  B2 抽样若有 `batch` 列，已切换为按批次分层抽样；否则回落到简单抽样。
    
-  B3 practical 的控制模板拼接已改为 **稀疏 vstack**，并最终 `float32`。
    
-  B3 使用的 **真实 checkpoint 名** 已确认存在（如 `step=40000.ckpt`）。
    
-  不使用 b3_state_inference.py 随机噪声占位推理。
    
-  b4 改为 `obs_names_make_unique()`，并确认为 18,080 基因、≤100,000 细胞、含 `target_gene` 与 `non-targeting`。
    

**No-Go**（遇到以下任一情况先修再跑）：

-  B3 模板写成了稠密大矩阵（`toarray()` 导致 OOM/巨慢）。
    
-  `.var` 基因数不是 18,080 或顺序不与训练一致。
    

---

## 建议的最简命令顺序（按你的默认路径）

1. **B3 推理（practical 版）**
    
    - 把 `--checkpoint_name` 改成存在的，如 `step=40000.ckpt`。
        
    - 若 CLI 报“未识别参数”，就把命令行 flags 改成 Hydra 风格键（可先 `--help` 看清楚）。
        
2. **B4 组装**
    
    - 确保最终 `vcc_prediction_final.h5ad` 满足：  
        18,080 基因、≤100,000 细胞、`obs['target_gene']` 包含所有扰动与 `non-targeting`、`.X` 为 `float32` 且非负（若 log1p）。
        
3. **B5 打包**
    
    - `cell-eval prep -i vcc_prediction_final.h5ad -o <prepared.h5ad>`；再复制为 `.vcc`。
        
    - 若使用“期望基因列表”，请用**从训练 H5AD 导出的 18,080**，而不要直接用示例 `gene_names.csv`。
        

---

## 为什么这些修改重要（与官网要求逐条对齐）

- **基因集合与顺序**：评测将你提交与真值按 **18,080 基因**比对；任意缺漏或顺序错位都会导致三指标（DES、PDS、MAE）计算失真甚至拒评。因此我们一律**继承训练 `.var`** 并在 b4 逐文件对齐。
    
- **float32 与数值域**：`cell-eval` 强制/建议 `.X` 为 `float32`；计数可为“整数 counts”或“log1p-normalized”，其他变换会报错或算分异常。你目前走 log1p 路线最稳妥。
    
- **控制细胞**：必须包含 `non-targeting`，否则差异表达统计无对照，且**提交不合规**。你已经在 B2 准备好 8,000 个，符合要求。
    

---

## 下一步建议

- 先按上述**4 个小修**更新脚本，然后直接跑 **B3 → B4 → B5**，拿到第一版官网分数。
    
- 之后再基于同一套产线替换 checkpoint，逐步加入我们前面讨论的 **MAE 辅助损失 → DES 头 → PDS 排序损失** 的改模版本做 A/B 提升（无需改封装流程，只替换 ckpt）。
    
- 若拿到 H1 少样本后，再加一步 **few-shot 适配**（微调“背景/数据集标记”或输出偏置），再按相同封装提交流水线走。
    

---

如果你需要，我可以基于你当前仓库结构，把“B3 推理模板的**稀疏拼接**”与 “b4 的**索引唯一化**”改动点列成 5 行内的“最小修改清单”，你据此改动即可马上进入推理与打包。