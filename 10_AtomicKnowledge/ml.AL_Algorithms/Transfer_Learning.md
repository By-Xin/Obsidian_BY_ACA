---
aliases: ['迁移学习', 'Transfer Learning']
tags:
  - concept
  - method
  - ml/transfer-learning
related_concepts:
  - [[Fine-tuning]]
  - [[Domain_Adaptation]]
---

#TransferLearning

## 1. Introduction

Transfer learning is trying to see if we can use the knowledge learned from one task to another task, or one domain to another domain. For example:
- Similar domain, different tasks: Cat-vs-Dog classification -> Elephant-vs-Horse classification
- Different domain, similar tasks: Cat-vs-Dog classification -> Art-style Cat-vs-Dog classification

According to the level of knowledge transfer, transfer learning can be divided into:

| | Source Data (Labeled) | Target Data (Unlabeled) |
| --- | --- | --- |
| **Target Data (Labeled)** | Model Fine-tuning 
|**Target Data (Unlabeled)** | 

- Model Fine-tuning: 
  - Data status:
    - Target data $(x^t, y^t)$ is labeled, but very limited amount.
    - Source data $(x^s, y^s)$ is labeled, and much more than target data. But is different from target task.
  - Idea: Use source data to pre-train a model as an initialization, then fine-tune the model on target data.