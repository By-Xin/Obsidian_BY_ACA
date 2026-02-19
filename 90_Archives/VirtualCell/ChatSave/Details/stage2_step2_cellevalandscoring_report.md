(torch) sczd425@ln01:~/run/xinby/vcc_local$ python stage2_celleval_and_scoring.py 
🔧 阶段2扩展: cell-eval包装和本地baseline评分
============================================================

🔧 步骤5: 使用cell-eval进行.vcc打包...
✅ cell-eval已安装
运行命令: cell-eval prep
	  输入: predictions/baseline_pred_val_fixed.h5ad
  基因列表: /data/home/sczd425/run/xinby/cell-eval/vcc_data/gene_names.csv
  输出: predictions/baseline_pred_val_fixed.vcc
❌ cell-eval prep 失败
标准输出: 
错误输出: INFO:cell_eval._cli._prep:Reading input anndata
INFO:cell_eval._cli._prep:Reading gene list
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/data/run01/sczd425/xinby/0831/cell-eval/src/cell_eval/_cli/_prep.py", line 271, in run_prep
    pl.read_csv(args.genes, has_header=False).to_series(0).cast(str).to_list()
                ^^^^^^^^^^
AttributeError: 'Namespace' object has no attribute 'genes'

⚠️ .vcc文件生成失败，但继续进行本地评分

📈 步骤6: 本地baseline评分...
预测数据: (6115, 18080)
真实数据: (6115, 18080)
评测目标: 16个条件

📊 计算DES (Differential Expression Score)...
  ARPC2: 预测DE=7841, 真实DE=1, 交集=1, DES=1.0000
  SMARCB1: 预测DE=7841, 真实DE=6, 交集=6, DES=1.0000
  EIF3H: 预测DE=7841, 真实DE=564, 交集=564, DES=1.0000
  NDUFB6: 预测DE=7841, 真实DE=6, 交集=6, DES=1.0000
  DHX36: 预测DE=7841, 真实DE=278, 交集=278, DES=1.0000
  C1QBP: 预测DE=7841, 真实DE=129, 交集=129, DES=1.0000
  USF2: 预测DE=7841, 真实DE=1, 交集=1, DES=1.0000
  OXA1L: 预测DE=7841, 真实DE=523, 交集=523, DES=1.0000
  TMSB10: 预测DE=7841, 真实DE=12, 交集=12, DES=1.0000
  HIRA: 预测DE=7841, 真实DE=7, 交集=7, DES=1.0000
  DNMT1: 预测DE=7841, 真实DE=27, 交集=27, DES=1.0000
  MAX: 预测DE=7841, 真实DE=1261, 交集=1261, DES=1.0000
  MED1: 预测DE=7841, 真实DE=1889, 交集=1889, DES=1.0000
  JAZF1: 预测DE=7841, 真实DE=19, 交集=19, DES=1.0000
  MED12: 预测DE=7841, 真实DE=2479, 交集=2479, DES=1.0000
平均DES: 1.0000

🎯 计算PDS (Perturbation Discrimination Score)...
  ARPC2: 排名=2/16, PDS=0.9375
  SMARCB1: 排名=8/16, PDS=0.5625
  EIF3H: 排名=13/16, PDS=0.2500
  NDUFB6: 排名=3/16, PDS=0.8750
  DHX36: 排名=11/16, PDS=0.3750
  C1QBP: 排名=9/16, PDS=0.5000
  USF2: 排名=5/16, PDS=0.7500
  OXA1L: 排名=12/16, PDS=0.3125
  TMSB10: 排名=7/16, PDS=0.6250
  HIRA: 排名=4/16, PDS=0.8125
  DNMT1: 排名=10/16, PDS=0.4375
  MAX: 排名=14/16, PDS=0.1875
  MED1: 排名=15/16, PDS=0.1250
  JAZF1: 排名=6/16, PDS=0.6875
  MED12: 排名=16/16, PDS=0.0625
平均PDS: 0.5000

📏 计算MAE (Mean Absolute Error)...
  ARPC2: MAE=0.0137
  SMARCB1: MAE=0.0157
  EIF3H: MAE=0.0222
  NDUFB6: MAE=0.0139
  DHX36: MAE=0.0189
  C1QBP: MAE=0.0169
  USF2: MAE=0.0142
  OXA1L: MAE=0.0206
  TMSB10: MAE=0.0151
  HIRA: MAE=0.0141
  DNMT1: MAE=0.0183
  MAX: MAE=0.0345
  MED1: MAE=0.0404
  JAZF1: MAE=0.0142
  MED12: MAE=0.0487
平均MAE: 0.0214

🏆 Baseline评分总结:
  DES: 1.0000
  PDS: 0.5000
  MAE: 0.0214

💾 保存评价结果...
✅ JSON结果: reports/stage2_baseline_eval_fixed.json
✅ 文本报告: reports/stage2_baseline_eval_fixed.txt

🎉 阶段2评分完成!
✅ 预测文件: predictions/baseline_pred_val_fixed.h5ad
✅ 评分结果: reports/stage2_baseline_eval_fixed.json
✅ 评分报告: reports/stage2_baseline_eval_fixed.txt

📊 Baseline性能总结:
  DES: 1.0000 (差异表达识别)
  PDS: 0.5000 (扰动区分能力)
  MAE: 0.0214 (平均绝对误差)

🎯 下一步: 阶段3官网评分验证
准备上传 手动生成的.vcc文件 到VCC官网