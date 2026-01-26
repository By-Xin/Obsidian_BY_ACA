(torch) sczd425@ln01:~/run/xinby/local_score_0912$ python step3_generate_ground_truth.py | cat
🚀 Step 3: 生成验证真值(Ground Truth)
随机种子: 2025
📂 加载验证集: splits/val.h5ad
🔍 验证验证集格式...
==================================================
✓ 基因数量: 18080
✓ 数据类型: float32
✓ target_gene列: 存在
✓ 控制细胞数: 3208
✓ 扰动类型数: 15
✓ 总细胞数: 4746

验证集格式: ✅ 合格

📊 验证集统计:
形状: (4746, 18080)
扰动分布:
target_gene
non-targeting    3208
MED12             200
EIF3H             147
OXA1L             137
HIRA              136
DHX36             107
MED1              101
TMSB10             91
ARPC2              90
C1QBP              84
JAZF1              83
NDUFB6             76
USF2               74
MAX                72
SMARCB1            72
DNMT1              68
Name: count, dtype: int64

🎯 创建Ground Truth数据...
✅ Ground Truth形状: (4746, 18080)
   数据类型: float32
   扰动分布:
     non-targeting: 3208
     MED12: 200
     EIF3H: 147
     OXA1L: 137
     HIRA: 136
     DHX36: 107
     MED1: 101
     TMSB10: 91
     ARPC2: 90
     C1QBP: 84
     JAZF1: 83
     NDUFB6: 76
     USF2: 74
     MAX: 72
     SMARCB1: 72
     DNMT1: 68

✅ 验证Ground Truth格式...
Ground Truth格式检查:
==================================================
✓ 基因数: 18080
✓ X数据类型: float32
✓ target_gene: present
✓ 控制细胞存在: 3208 cells
✓ 扰动细胞: 1538 cells
✓ 扰动类型: 15 types
✓ 基因名称格式: valid
✓ 细胞名称格式: valid
✓ 数据非负: non-negative

Ground Truth格式: ✅ 合格

💾 保存Ground Truth: truth/ground_truth.h5ad

📊 创建Pseudobulk参考数据...
计算每个扰动的pseudobulk表达:
  non-targeting: 3208 cells -> pseudobulk
  MED12: 200 cells -> pseudobulk
  EIF3H: 147 cells -> pseudobulk
  OXA1L: 137 cells -> pseudobulk
  HIRA: 136 cells -> pseudobulk
  DHX36: 107 cells -> pseudobulk
  MED1: 101 cells -> pseudobulk
  TMSB10: 91 cells -> pseudobulk
  ARPC2: 90 cells -> pseudobulk
  C1QBP: 84 cells -> pseudobulk
  JAZF1: 83 cells -> pseudobulk
  NDUFB6: 76 cells -> pseudobulk
  USF2: 74 cells -> pseudobulk
  SMARCB1: 72 cells -> pseudobulk
  MAX: 72 cells -> pseudobulk
  DNMT1: 68 cells -> pseudobulk
✅ Pseudobulk参考: (18080, 16)
   保存到: truth/pseudobulk_reference.csv

💾 保存验证信息...
✅ 扰动统计: truth/target_gene_stats.csv
✅ 基因顺序: truth/gene_names_order.csv
✅ 详细报告: reports/step3_ground_truth_report.txt

🎉 Step 3 完成！
✅ Ground Truth: truth/ground_truth.h5ad
✅ Pseudobulk参考: truth/pseudobulk_reference.csv
📋 可进入阶段2：构建baseline model

Step 3 Ground Truth Generation Report
==================================================
Generation date: 2025-09-12 15:42:58.284277
Random seed: 2025

Ground Truth Dataset:
  Shape: (4746, 18080)
  Data type: float32
  Total cells: 4746
  Total genes: 18080

Target Gene Distribution:
  non-targeting (Control): 3208 cells
  MED12 (Perturbation): 200 cells
  EIF3H (Perturbation): 147 cells
  OXA1L (Perturbation): 137 cells
  HIRA (Perturbation): 136 cells
  DHX36 (Perturbation): 107 cells
  MED1 (Perturbation): 101 cells
  TMSB10 (Perturbation): 91 cells
  ARPC2 (Perturbation): 90 cells
  C1QBP (Perturbation): 84 cells
  JAZF1 (Perturbation): 83 cells
  NDUFB6 (Perturbation): 76 cells
  USF2 (Perturbation): 74 cells
  MAX (Perturbation): 72 cells
  SMARCB1 (Perturbation): 72 cells
  DNMT1 (Perturbation): 68 cells

Summary:
  Control cells: 3208
  Perturbation types: 15
  Perturbation cells: 1538
