(torch) sczd425@ln01:~/run/xinby/local_score_0912$ python step2_train_val_split.py 
🚀 Step 2: 构建训练/验证数据拆分
随机种子: 2025
📂 加载数据: data_raw/k562_standardized.h5ad
数据形状: (18465, 18080)
🔍 分析扰动分布...
总扰动类型数: 54
总细胞数: 18465
控制细胞数: 10691
扰动类型数: 53

可用于验证的扰动 (>=150细胞):
数量: 21
前10个扰动:
target_gene
MED12     573
EIF3H     369
OXA1L     344
HIRA      341
DHX36     269
MED1      254
TMSB10    228
ARPC2     227
C1QBP     210
JAZF1     209

✅ 选定验证扰动: 15个
  1. MED12: 573细胞
  2. EIF3H: 369细胞
  3. OXA1L: 344细胞
  4. HIRA: 341细胞
  5. DHX36: 269细胞
  6. MED1: 254细胞
  7. TMSB10: 228细胞
  8. ARPC2: 227细胞
  9. C1QBP: 210细胞
  10. JAZF1: 209细胞
  11. NDUFB6: 190细胞
  12. USF2: 185细胞
  13. SMARCB1: 182细胞
  14. MAX: 180细胞
  15. DNMT1: 170细胞

🔄 进行分层抽样...
控制细胞: 7483训练 + 3208验证

处理验证扰动:
  ✅ MED12: 373训练 + 200验证
  ✅ EIF3H: 222训练 + 147验证
  ✅ OXA1L: 207训练 + 137验证
  ✅ HIRA: 205训练 + 136验证
  ✅ DHX36: 162训练 + 107验证
  ✅ MED1: 153训练 + 101验证
  ✅ TMSB10: 137训练 + 91验证
  ✅ ARPC2: 137训练 + 90验证
  ✅ C1QBP: 126训练 + 84验证
  ✅ JAZF1: 126训练 + 83验证
  ✅ NDUFB6: 114训练 + 76验证
  ✅ USF2: 111训练 + 74验证
  ✅ SMARCB1: 110训练 + 72验证
  ✅ MAX: 108训练 + 72验证
  ✅ DNMT1: 102训练 + 68验证

处理剩余扰动 (全部训练): 38个
  ATP6V0B: 147训练
  ATP6V0C: 53训练
  UQCRB: 96训练
  NDUFB4: 166训练
  MAU2: 66训练
  LRPPRC: 110训练
  SUPT4H1: 74训练
  SUPV3L1: 80训练
  MBTPS1: 68训练
  HDAC3: 60训练
  MAT2A: 52训练
  RNF20: 124训练
  SMARCE1: 31训练
  SSBP1: 119训练
  METTL3: 138训练
  TFAM: 82训练
  SEC62: 5训练
  DPH2: 164训练
  UQCRQ: 166训练
  SDC1: 170训练
  COX6C: 117训练
  SMAGP: 117训练
  CHMP3: 130训练
  FDPS: 10训练
  EIF4B: 41训练
  HTATSF1: 96训练
  NISCH: 144训练
  METTL17: 122训练
  STRAP: 106训练
  RRM1: 103训练
  HMGCR: 166训练
  KIF20A: 96训练
  EWSR1: 48训练
  METTL14: 121训练
  DNAJA3: 127训练
  PPP2R3C: 52训练
  SLC25A3: 114训练
  WAC: 162训练

📊 拆分统计:
训练细胞数: 13719
验证细胞数: 4746
验证细胞数限制: 100000
✅ 验证细胞数符合要求

📁 创建拆分数据集...
训练集: (13719, 18080)
训练集扰动分布:
target_gene
non-targeting    7483
MED12             373
EIF3H             222
OXA1L             207
HIRA              205
SDC1              170
HMGCR             166
UQCRQ             166
NDUFB4            166
DPH2              164
Name: count, dtype: int64

验证集: (4746, 18080)
验证集扰动分布:
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

✅ 验证拆分结果...
==================================================
✓ 基因数量一致: 训练:18080, 验证:18080
✓ 基因顺序一致: 一致
✓ 数据类型: float32, float32
✓ 验证扰动匹配: 期望15, 实际15
✓ 训练集控制细胞: 7483个
✓ 验证集控制细胞: 3208个
✓ 细胞无重叠: 无重叠

总体结果: ✅ 通过

💾 保存拆分数据...
✅ 训练集: splits/train.h5ad
✅ 验证集: splits/val.h5ad

💾 保存拆分信息...
✅ 验证基因列表: splits/validation_genes.csv
✅ 拆分统计: splits/split_statistics.csv
✅ 详细报告: reports/step2_split_report.txt

Step 2 Train-Val Split Report
==================================================
Random seed: 2025
Split date: 2025-09-12 15:39:45.090694

Training Set:
  Cells: 13719
  Genes: 18080
  Perturbations: 54
  Control cells: 7483

Validation Set:
  Cells: 4746
  Genes: 18080
  Perturbations: 16
  Control cells: 3208

Validation Perturbations:
  MED12: 200 cells
  EIF3H: 147 cells
  OXA1L: 137 cells
  HIRA: 136 cells
  DHX36: 107 cells
  MED1: 101 cells
  TMSB10: 91 cells
  ARPC2: 90 cells
  C1QBP: 84 cells
  JAZF1: 83 cells
  NDUFB6: 76 cells
  USF2: 74 cells
  MAX: 72 cells
  SMARCB1: 72 cells
  DNMT1: 68 cells

random_seed,total_genes,train_cells,val_cells,train_perturbations,val_perturbations,train_control_cells,val_control_cells

2025,18080,13719,4746,54,16,7483,3208

target_gene,train_cells,val_cells

MED12,373,200

EIF3H,222,147

OXA1L,207,137

HIRA,205,136

DHX36,162,107

MED1,153,101

TMSB10,137,91

ARPC2,137,90

C1QBP,126,84

JAZF1,126,83

NDUFB6,114,76

USF2,111,74

SMARCB1,110,72

MAX,108,72

DNMT1,102,68

