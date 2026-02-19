(torch) sczd425@ln01:~$ cd /data/home/sczd425/run/xinby/local_score_0912 && python step1_validate_data.py | cat
/data/home/sczd425/.conda/envs/torch/lib/python3.11/functools.py:909: ImplicitModificationWarning: Transforming to str index.
  return dispatch(args[0].__class__)(*args, **kw)
使用数据路径:
  K562 H5: /data/home/sczd425/run/xinby/competition_support_set/k562.h5
  基因名称: /data/home/sczd425/run/xinby/cell-eval/vcc_data/gene_names.csv
  输出路径: data_raw/k562_standardized.h5ad
🔍 Step 1: 验证K562数据格式...
检查基因名称文件...
基因文件shape: (18079, 1)
首列名称: SAMD11
前5个基因: ['NOC2L', 'KLHL17', 'PLEKHN1', 'PERM1', 'HES4']
检查H5文件: /data/home/sczd425/run/xinby/competition_support_set/k562.h5
H5文件结构:
X: (18465, 18080), float32
layers: (group)
obs: (group)
  UMI_count: (18465,), float32
  batch_var: (group)
    categories: (48,), object
    codes: (18465,), int8
  cell_barcode: (18465,), object
  cell_type: (group)
    categories: (1,), object
    codes: (18465,), int8
  core_adjusted_UMI_count: (18465,), float32
  core_scale_factor: (18465,), float32
  gem_group: (18465,), int64
  gene: (group)
    categories: (54,), object
    codes: (18465,), int8
  gene_id: (group)
    categories: (54,), object
    codes: (18465,), int8
  gene_transcript: (group)
    categories: (153,), object
    codes: (18465,), int16
  mitopercent: (18465,), float32
  sgID_AB: (group)
    categories: (153,), object
    codes: (18465,), int16
  target_gene: (group)
    categories: (54,), object
    codes: (18465,), int8
  transcript: (group)
    categories: (4,), object
    codes: (18465,), int8
  z_gemgroup_UMI: (18465,), float32
obsm: (group)
obsp: (group)
uns: (group)
var: (group)
  _index: (18080,), object
varm: (group)
varp: (group)

📊 表达矩阵信息:
Shape: (18465, 18080)
Dtype: float32
是否全为整数: False
95分位数: 1.4442
非零值比例: 0.1794
✅ 数据似乎是log1p转换后的数据

🧬 基因验证:
基因数量: 18080
前10个基因: ['SAMD11', 'NOC2L', 'KLHL17', 'PLEKHN1', 'PERM1', 'HES4', 'ISG15', 'AGRN', 'RNF223', 'C1orf159']
与预期基因匹配情况: [True, True, True, True, True]
✅ 基因数量正确: 18080

🎯 扰动信息:
扰动类型数量: 54
总细胞数: 18465
控制细胞数 (non-targeting): 10691

前10个扰动的细胞数:
non-targeting    10691
MED12              573
EIF3H              369
OXA1L              344
HIRA               341
DHX36              269
MED1               254
TMSB10             228
ARPC2              227
C1QBP              210

可用于验证的扰动(≥150细胞): 21
✅ 足够的扰动用于验证集构建

✅ 数据验证通过！

🔄 转换H5文件为AnnData格式...
✅ 创建AnnData对象: (18465, 18080)
   Dtype: float32
   基因数: 18080
   细胞数: 18465
✅ 保存到: data_raw/k562_standardized.h5ad
📋 验证报告保存到: reports/step1_validation_report.txt

Step 1 Validation Report
==================================================
Random seed: 2025
Data shape: (18465, 18080)
Data dtype: float32
Genes count: 18080
Cells count: 18465
Target genes in obs: True
Perturbation types: 54
Control cells: 10691
Non-control perturbations: 53


