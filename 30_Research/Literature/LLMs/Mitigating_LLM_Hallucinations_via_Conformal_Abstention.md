---
aliases: [LLM Hallucinations, Conformal Abstention, 幻觉问题, 弃权, Conformal Prediction]
tags:
  - concept
  - ml/deep-learning
  - cs/nlp
  - paper
related_concepts:
  - "[[Uncertainty_in_LLMs]]"
  - "[[LLM_Sampling_Methods]]"
  - "[[Deep_Reasoning_for_LLMs]]"
  - "[[Conformal_Prediction]]"
---

# Mitigating LLM Hallucinations via Conformal Abstention

- 通过在适当的时候进行弃权的方法改进幻觉问题. 且通过 conformal 来为 abstention 提供理论支持.

## Hallucination Problem

模型生成不遵循原文/ input content (faithfulness) 或不符合事实 (factualness) 的内容. 在传统任务里, 幻觉大多数指的是 faithfulness; 在 LLM 中, 更多的是 factualness. 

Hallucination 的识别本身也是模糊的, 即使对人类来说, 也并不是总能给出清晰的答案.

##  Problem Definition

### Abstention of LLMs

记 $\mathcal{X}$ 为输入空间 (可认为是 prompt), $\mathcal{Y}$ 为输出空间 (可认为是输出的结果). 则大语言的模型的输出可以看作是一个函数 $f: \mathcal{X} \to \mathcal{Y}$. 这里假设大模型的采样是 greedy 的, 即 $f(x) = \arg\max_{y \in \mathcal{Y}} \mathbb{P}(y|x)$.

Abstention 可以通过如下的方式来实现:
$$
a_\lambda(x) = \begin{cases}
1 & \text{if } g(X) \geq \lambda \\
0 & \text{otherwise}
\end{cases}
$$
其中 $g(x)$ 是一个 score function 表示其 confidence, $\lambda$ 是一个阈值. 这里的 $a_\lambda(x)$ 就是 abstention 的结果, 1 表示 abstain, 0 表示不 abstain. 并且其中这个 $\lambda$ 是与大语言模型 $f$ 相关的. 

此外再引入一个 match function $m(X; Y', Y) \in \{0, 1\}$, 其中 $Y'$ 是模型的输出, $Y$ 是 ground truth. 这里的 match function 用来表示模型的输出和 ground truth 是否匹配. 当 $m(X; Y', Y) = 1$ 时, 表示模型的输出和 ground truth "同等正确", 否则表示不匹配.

### 

可以构建如下的 loss function:
$$
\ell (X, Y; \lambda) = (1- a_\lambda(X)) (1- m(X; Y', Y)) 
$$

则可以得到如下优化目标:
$$\begin{align*}
& \arg\min_{\lambda\in\Lambda} T(\lambda) , ~\text{s.t. } R(\lambda) \leq \alpha\\
& T(\lambda) = \mathbb{E}[\alpha_\lambda(X)], ~ R(\lambda) = \mathbb{E}[\ell(X, Y; \lambda)]\\
\end{align*}$$