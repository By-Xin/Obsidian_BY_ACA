---
aliases: [论文速查, Paper Quick Reference]
tags:
  - paper
  - reference
status: reading-list
---

# 论文速查 (Papers In a Word)

这是一个论文快速参考列表，记录值得深入阅读的文献。

## LLM

* **LIMA:Less is More for Alignment (2305.11206):** 强调高质量少训练集的作用会强于低质量的大水漫灌
* [混合专家模型 (MoE) 详解](https://zhuanlan.zhihu.com/p/674698482): 知乎上针对HuggingFace 的一个类似综述的博客
* **From Sparse to Soft Mixture of Experts (2308.00951, ICLR 2024)**: 提出了一个 sparse MoE 似乎有点火
* **Switch Transformers**: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (2101.03961): 虽然 MoE 在 1990s 早有提及, 但是这篇 Google 的文章一般认为是现代 LLM MoE 的关键起点
* **Language Models (Mostly) Know What They Know**: Anthropic 的22年一个文章, 长度很大, 提出了 P(True) (直接在模型输出后再prompt让模型判断正误) 和 P(I-Know) (改造结构加了一个分类模块输出Know/NotKnow), 分别衡量了先验和后验的模型对于输出的 confidence. 这个文章有点长, 涉及到的点很多, 最好仔细出一期解读.
* **Beyond the 80/20 Rule**: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning: Qwen 团队25的一个工作, 大概是 RL 作用在 sampling 等上面的. 请关注一下
* **Curious Case of Neural Text Degeneration**: 提出 top-p Sampling
* **Adaptive Decoding via Latent Preference Optimization (2411.09661):** Meta 的文章, 提出 Adaptive Decoding 通过 LPO 方法
* **Turning Up the Heat**: Min-p Sampling for Creative and Coherent LLM Outputs (2407.01082): Min-p Sampling


## Optimization

* **[CSC2541 Winter 2021 Topics in Machine Learning: Neural Net Training Dynamics](https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/)**: Toronto 大学的一门进阶课程. 主要的话题包括很多训练理论/优化部分的内容
* [ ] **Understanding warmup-scale-decay learning rates: A river valley loss landscape perspective**: River-Valley 模型

## Machine Learning
- [Advanced Topics in Machine Learning, STAT241B / CS281B, UCB](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/): **巨牛的一门课**, 核心内容包括
	- Stat/ML in a nutshell (review), Nearest neighbors and kernels, Splines and RKHS methods, Minimax theory, Empirical process theory, Lasso, Ridge, Ridgeless, Gradient flow, Buffer/spillover, Conformal prediction, Conformal under distribution shift, Scoring and calibration, 
- [Statistical Methods for Machine Learning, Larry Wasserman](https://www.stat.cmu.edu/~larry/=sml/): 写 all of statistics 的老师的课程. 
	- 主要包括 [**Density Estimation**](https://www.stat.cmu.edu/~larry/=sml/densityestimation.pdf) , [**Nonparametric Regression**](https://www.stat.cmu.edu/~larry/=sml/nonpar2019.pdf),  [**Linear Regression**](https://www.stat.cmu.edu/~larry/=sml/LinearRegression.pdf),   [**Sparsity**](https://www.stat.cmu.edu/~larry/=sml/sparsity.pdf) , [**Nonparametric Sparsity**](https://www.stat.cmu.edu/~larry/=sml/Spam.pdf), [**Linear Classifiers**](https://www.stat.cmu.edu/~larry/=sml/linearclassification.pdf),   [**Nonparametric Classifiers**](https://www.stat.cmu.edu/~larry/=sml/nonparclass.pdf) , [**Random Forests**](https://www.stat.cmu.edu/~larry/=sml/forests.pdf), [**Clustering**](https://www.stat.cmu.edu/~larry/=sml/clustering.pdf) , [**Graphical Models**](https://www.stat.cmu.edu/~larry/=sml/GraphicalModels.pdf) , [**Directed Graphical Models**](https://www.stat.cmu.edu/~larry/=sml/DAGs.pdf) , [**Causal Inference**](https://www.stat.cmu.edu/~larry/=sml/Causation.pdf) , [**Minimax Theory**](https://www.stat.cmu.edu/~larry/=sml/minimax.pdf) , [**Nonparametric Bayesian Inference**](https://www.stat.cmu.edu/~larry/=sml/nonparbayes) , [**Conformal Prediction**](https://www.stat.cmu.edu/~larry/=sml/Conformal) , [**Differential Privacy**](https://www.stat.cmu.edu/~larry/=sml/diffpriv.pdf) , [**Optimal Transport and Wasserstein Distance**](https://www.stat.cmu.edu/~larry/=sml/Opt.pdf) , [**Two Sample Testing**](https://www.stat.cmu.edu/~larry/=sml/TwoSample.pdf) , [**Dimension Reduction**](https://www.stat.cmu.edu/~larry/=sml/DimRed.pdf), [**Boosting**](https://www.stat.cmu.edu/~larry/=sml/Boosting.pdf), [**Support Vector Machines**](https://www.stat.cmu.edu/~larry/=sml/SVM.pdf), [**Online Learning**](https://www.stat.cmu.edu/~larry/=sml/Online.pdf)