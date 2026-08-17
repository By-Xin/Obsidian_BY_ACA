(torch) sczd425@ln01:~/run/xinby/vcc_local$ cd /data/home/sczd425/run/xinby/vcc_local && python step3_generate_ground_truth_fixed.py | cat
🚀 Step 3 修复版: 生成验证真值(Ground Truth)
随机种子: 2025
🔧 关键修复: 增强基因顺序完整性验证
🧬 加载官方基因顺序...
✅ 官方基因数量: 18080
📂 加载验证集: splits/val_fixed.h5ad
🔍 全面验证验证集格式...
🔍 完整验证基因顺序一致性...
✅ 基因顺序与官方完全一致
验证集格式检查:
==================================================
✓ 基因顺序完整性: 与官方18,080个基因完全一致
✓ 基因数量: 18080
✓ 数据类型: float32
✓ target_gene列: 存在
✓ 控制细胞数: 3208
✓ 扰动类型数: 15
✓ 总细胞数: 6115
✓ 数据非负: 符合log1p要求

验证集格式: ✅ 合格

📊 验证集统计:
形状: (6115, 18080)
扰动分布:
target_gene
non-targeting    3208
ARPC2             200
DHX36             200
C1QBP             200
JAZF1             200
HIRA              200
MED1              200
EIF3H             200
TMSB10            200
MED12             200
OXA1L             200
NDUFB6            190
USF2              185
SMARCB1           182
MAX               180
DNMT1             170
Name: count, dtype: int64

🎯 创建Ground Truth数据...
✅ 基因顺序已经正确
✅ Ground Truth形状: (6115, 18080)
   数据类型: float32
   扰动分布:
     non-targeting: 3208
     ARPC2: 200
     DHX36: 200
     C1QBP: 200
     JAZF1: 200
     HIRA: 200
     MED1: 200
     EIF3H: 200
     TMSB10: 200
     MED12: 200
     OXA1L: 200
     NDUFB6: 190
     USF2: 185
     SMARCB1: 182
     MAX: 180
     DNMT1: 170

✅ 全面验证Ground Truth格式...
🔍 完整验证基因顺序一致性...
✅ 基因顺序与官方完全一致
Ground Truth格式检查:
==================================================
✓ 基因顺序完美匹配: 18,080个基因与官方完全一致
✓ 基因数: 18080
✓ X数据类型: float32
✓ target_gene: present
✓ 控制细胞存在: 3208 cells
✓ 扰动细胞: 2907 cells
✓ 扰动类型: 15 types
✓ 基因名称格式: valid
✓ 细胞名称格式: valid
✓ 数据非负: non-negative
✓ log1p量纲: p95=1.452

Ground Truth格式: ✅ 合格

💾 保存Ground Truth: truth/ground_truth_fixed.h5ad

📊 创建Pseudobulk参考数据（含基因顺序验证）...
计算每个扰动的pseudobulk表达:
  MED12: 200 cells -> pseudobulk
  EIF3H: 200 cells -> pseudobulk
  OXA1L: 200 cells -> pseudobulk
  HIRA: 200 cells -> pseudobulk
  DHX36: 200 cells -> pseudobulk
  MED1: 200 cells -> pseudobulk
  TMSB10: 200 cells -> pseudobulk
  ARPC2: 200 cells -> pseudobulk
  C1QBP: 200 cells -> pseudobulk
  JAZF1: 200 cells -> pseudobulk
  NDUFB6: 190 cells -> pseudobulk
  USF2: 185 cells -> pseudobulk
  SMARCB1: 182 cells -> pseudobulk
  MAX: 180 cells -> pseudobulk
  DNMT1: 170 cells -> pseudobulk
  non-targeting: 3208 cells -> pseudobulk
✅ Pseudobulk基因顺序与官方一致
✅ Pseudobulk参考: (18080, 16)
   保存到: truth/pseudobulk_reference_fixed.csv
   基因顺序: 与官方18080个基因完全一致

💾 保存验证信息（含基因顺序验证）...
✅ 扰动统计: truth/target_gene_stats_fixed.csv
✅ 基因顺序: truth/gene_names_order_fixed.csv
✅ 详细报告: reports/step3_ground_truth_report_fixed.txt
🧬 基因顺序匹配率: 100.0%

🎉 Step 3 修复完成！
✅ Ground Truth: truth/ground_truth_fixed.h5ad
✅ Pseudobulk参考: truth/pseudobulk_reference_fixed.csv
🧬 基因顺序: 与官方18,080个基因完全一致
📋 可进入阶段2：构建baseline model

---

Step 3 Ground Truth Generation Report (FIXED)
============================================================
Generation date: 2025-09-12 22:05:27.069845
Random seed: 2025
CRITICAL FIXES:
- 增强基因顺序完整性验证
- 确保Ground Truth基因顺序与官方完全一致
- 基因顺序匹配率: 100.0%

Ground Truth Dataset:
  Shape: (6115, 18080)
  Data type: float32
  Total cells: 6115
  Total genes: 18080
  Gene order validation: PERFECT

Target Gene Distribution:
  non-targeting (Control): 3208 cells
  ARPC2 (Perturbation): 200 cells
  DHX36 (Perturbation): 200 cells
  C1QBP (Perturbation): 200 cells
  JAZF1 (Perturbation): 200 cells
  HIRA (Perturbation): 200 cells
  MED1 (Perturbation): 200 cells
  EIF3H (Perturbation): 200 cells
  TMSB10 (Perturbation): 200 cells
  MED12 (Perturbation): 200 cells
  OXA1L (Perturbation): 200 cells
  NDUFB6 (Perturbation): 190 cells
  USF2 (Perturbation): 185 cells
  SMARCB1 (Perturbation): 182 cells
  MAX (Perturbation): 180 cells
  DNMT1 (Perturbation): 170 cells

Summary:
  Control cells: 3208
  Perturbation types: 15
  Perturbation cells: 2907

Gene Order Validation:
  Official genes loaded: 18080
  Ground Truth genes: 18080
  Perfect match: YES
