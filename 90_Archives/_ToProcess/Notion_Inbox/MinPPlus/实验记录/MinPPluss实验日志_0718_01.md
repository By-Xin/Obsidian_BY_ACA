## 相关配置


| Category            | Parameter             | Value / Description                         |
| ------------------- | --------------------- | ------------------------------------------- |
| **Model Paths**     | `model_path`          | `/root/autodl-tmp/llama3-2_3b_instruct`     |
|                     | `data_path`           | `/root/autodl-tmp/dataset/alpaca.jsonl`     |
| **Min-P+ Params**   | `K`                   | `5` (number of truncation thresholds)       |
|                     | `tau_list`            | `np.logspace(-3, -0.3, K).tolist()`         |
|                     | `truncation_mode`     | `"hard"`                                    |
|                     | `lambda_l1`           | `1e-3` (L1 regularization coefficient)      |
| **Training**        | `max_epochs`          | `5`                                         |
|                     | `batch_size`          | `32`                                        |
|                     | `learning_rate`       | `2e-4`                                      |
|                     | `weight_decay`        | `0.0`                                       |
|                     | `warmup_steps`        | `100`                                       |
|                     | `patience`            | `2`                                         |
| **Dataset**         | `dataset_type`        | `"alpaca"` (options: `"alpaca"`, `"gsm8k"`) |
|                     | `max_length`          | `512` (max sequence length for input)       |
|                     | `train_ratio`         | `0.7`                                       |
|                     | `val_ratio`           | `0.15`                                      |
|                     | `test_ratio`          | `0.15`                                      |
| **Training Opt.**   | `mixed_precision`     | `True`                                      |
|                     | `gradient_clipping`   | `1.0`                                       |
|                     | `save_every_n_epochs` | `1`                                         |
|                     | `log_every_n_steps`   | `200`                                       |
| **Reproducibility** | `seed`                | `42`                                        |
| **Device**          | `device`              | `"cuda"` if available, else `"cpu"`         |

## 设备运行情况

目前进行一个 epoch 用时大概 10 min 左右

![[Pasted image 20250718163945.png]]

## 训练日志重要摘录


- 2025-07-18 15:46:26,917 - data.dataset - INFO - Created data loaders: train=36054, val=7726, test=7726   
---
- 2025-07-18 15:46:26,921 - training.trainer - INFO - Starting training...                 
  **Epoch 1/5**: 100%|█| 1126/1126 [07:16<00:00,  2.58it/s, loss=11.7644, ce=11.7618, theta_nz=

- Validating: 100%|█| 242/242 [01:35<00:00,  2.54it/s]

- 2025-07-18 15:55:18,345 - training.trainer - INFO - New best validation loss: 11.7618    

- 2025-07-18 15:55:18,346 - training.trainer - INFO - 
  Epoch 1 Summary:                                                                         
  2025-07-18 15:55:18,346 - training.trainer - INFO - Train - CE: 11.7618, Total: 11.7648, KL: 8.2242                                                                               
  2025-07-18 15:55:18,346 - training.trainer - INFO - Val   - CE: 11.7618, Total: 11.7643, KL: 8.2466                                                                               

- 2025-07-18 15:55:18,346 - training.trainer - INFO - Theta - L1: 3.0171, NonZero: 5.0     

- 2025-07-18 15:55:18,346 - training.trainer - INFO - LR: 0.000186                         
---
- **Epoch 2/5**: 100%|█| 1126/1126 [07:17<00:00,  2.58it/s, loss=11.7634, ce=11.7618, theta_nz=

- Validating: 100%|█| 242/242 [01:35<00:00,  2.55it/s]

- 2025-07-18 16:04:10,547 - training.trainer - INFO -                                      
  Epoch 2 Summary:                                                                         
  2025-07-18 16:04:10,547 - training.trainer - INFO - Train - CE: 11.7618, Total: 11.7638, KL: 8.2243                                                                               
  2025-07-18 16:04:10,547 - training.trainer - INFO - Val   - CE: 11.7618, Total: 11.7633, KL: 8.2466                                                                               
  2025-07-18 16:04:10,547 - training.trainer - INFO - Theta - L1: 1.9879, NonZero: 4.994671403197158                                                                                
  2025-07-18 16:04:10,547 - training.trainer - INFO - LR: 0.000143                         
---
- **Epoch 3/5**: 100%|█| 1126/1126 [07:16<00:00,  2.58it/s, loss=11.7630, ce=11.7618, theta_nz=

- Validating: 100%|█| 242/242 [01:35<00:00,  2.55it/s]

- 2025-07-18 16:13:01,863 - training.trainer - INFO - Early stopping triggered after 3 epochs   
---
- 2025-07-18 16:13:01,863 - training.trainer - INFO - Training completed!                                                                                     
- 2025-07-18 16:13:03,355 - training.trainer - INFO - Final theta values: [ 3.5479364e-01  3.8752869e-01 -8.5525198e-07  4.3183061e-01   -1.4673337e-06]                                                                         
- 2025-07-18 16:13:03,355 - training.trainer - INFO - Non-zero theta count: 4              

- 2025-07-18 16:13:03,357 - __main__ - INFO - Evaluating on test set...                    
  Validating: 100%|█| 242/242 [01:33<00:00,  2.59it/s]

- 2025-07-18 16:14:36,893 - __main__ - INFO - Test metrics: {'ce_loss': 11.76178074277137, 'sparsity_loss': 1.1741552352905273, 'total_loss': 11.762954715854866, 'theta': tensor([ 3.5479e-01,  3.8753e-01, -8.5525e-07,  4.3183e-01, -1.4673e-06], device='cuda:0'), 'theta_l1': 1.1741552352905273, 'theta_nonzero': 4.0, 'kl_divergence': 8.22947494845745}                                                                 
- 2025-07-18 16:14:36,894 - __main__ - INFO - Training completed successfully!


## Final_results.json
```{json}
{
  "final_theta": [0.3548, 0.3875, -8.55e-07, 0.4318, -1.47e-06],
  "tau_list": [0.001, 0.0047, 0.0224, 0.1059, 0.5012],
  "best_val_loss": 11.7618,
  "train_metrics": [
    {
      "ce_loss": 11.7618,
      "sparsity_loss": 3.0171,
      "total_loss": 11.7648,
      "theta_l1": 3.0171,
      "theta_nonzero": 5.0,
      "kl_divergence": 8.2242
    },
    {
      "ce_loss": 11.7618,
      "sparsity_loss": 1.9879,
      "total_loss": 11.7638,
      "theta_l1": 1.9879,
      "theta_nonzero": 4.995,
      "kl_divergence": 8.2243
    },
    {
      "ce_loss": 11.7618,
      "sparsity_loss": 1.3528,
      "total_loss": 11.7631,
      "theta_l1": 1.3528,
      "theta_nonzero": 4.835,
      "kl_divergence": 8.2242
    }
  ],
  "val_metrics": [
    {
      "ce_loss": 11.7618,
      "sparsity_loss": 2.4741,
      "total_loss": 11.7643,
      "theta_l1": 2.4741,
      "theta_nonzero": 5.0,
      "kl_divergence": 8.2466
    },
    {
      "ce_loss": 11.7618,
      "sparsity_loss": 1.5625,
      "total_loss": 11.7633,
      "theta_l1": 1.5625,
      "theta_nonzero": 5.0,
      "kl_divergence": 8.2466
    },
    {
      "ce_loss": 11.7618,
      "sparsity_loss": 1.1742,
      "total_loss": 11.7630,
      "theta_l1": 1.1742,
      "theta_nonzero": 4.0,
      "kl_divergence": 8.2466
    }
  ]
}

```