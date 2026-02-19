---
aliases: [Post-Training, Continual Learning, Catastrophic Forgetting, 后训练, 灾难性遗忘, 持续学习]
tags:
  - concept
  - ml/deep-learning
  - cs/nlp
related_concepts:
  - "[[Pretrain_and_Alignment_for_LLMs]]"
  - "[[Deep_Reasoning_for_LLMs]]"
  - "[[Transfer_Learning]]"
  - "[[LifeLong_Learning]]"
  - "[[Knowledge_Distillation]]"
source: "李宏毅"
---

# Post-Training and Forgetting

- Post-training (Continual Learning) 是指当下的许多前沿大模型已经有非常强的基础能力, 我们希望通过微调其参数来在特定领域 (如医疗、法律) 或特定语言等上进行更好的适应.
- 习惯上, 将 Post-training 前的模型称为 Foundation Model, Post-training 之后的模型称为 Fine-tuned Model. 这里并不局限 pre-training 的模型是一个基础模型, 也可以是一个已经经过 alignment 之后的具备一定能力的模型. 只要是我们想在其上进行微调的模型都可以称为 Post-training 的过程. 

## Post-training 的方法

一般而言, Post-training 的方法可以分为三种方向: 
- Pre-train Style: 直接通过搜集在特定领域的文本数据, 进行预训练 (文字接龙). 
  - 如果模型训练过分依赖, 可能会缺乏指令跟随的能力
  - *Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages* (arxiv.org/abs/2310.04799) 
    - 该文章尝试使用 LLaMA-2-Chat 模型进行中文的 Post-training, 主要采用的 Pre-train Style 进行微调. 然而初步发现, 尽管模型在中文的生成能力上有了提升, 但是其在原本的 alignment 能力上却有了明显的下降. 这说明了 Post-training 的过程可能会导致模型遗忘原本的能力, 也就是我们所说的 Forgetting.
- SFT (Supervised Fine-Tuning): 通过搜集特定领域的带有人类标注的文本数据构建明确的 input-output 组合进行有监督学习, 让模型学习在这种格式下生成目标答案
  - 关键用于提升模型可控性和指令响应性
  - *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* (arxiv.org/abs/2310.03693)
    - 该文章使用 SFT 的方法对模型进行微调, 实验发现即使对模型进行一些正常的 SFT 微调, 也会导致模型在一些安全性问题上出现问题.
  - *Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging* (arxiv.org/abs/2412.19512)
    - 该模型在 Llama-3-8B-Instruct 模型上进行 SFT 微调, 发现其在微调后在特定技能上的能力有了提升, 但是其在原本的能力 (如 safety alignment) 上却有了明显的下降. 
  - *Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning* (arxiv.org/abs/2402.13669)
    - 该文章用 SFT 微调后发现: Post-training 增强了模型在 target task 上的能力, 但是却削弱了在其他任务上的能力 (不只是安全性问题).
- RL Style: 通过强化学习( 如 PPO、DPO、RLAIF 等)来优化模型输出的偏好和质量，基于人类反馈或评分. 这时的数据没有固定的输出答案, 只有对模型输出的内容的好坏的评分
  - 存在多个候选输出，系统根据偏好反馈( 如人类打分或 reward model)更新策略


## Catastrophic Forgetting

上述当我们教模型一个对应任务而导致其他任务的能力下降的现象称为 Catastrophic Forgetting.
- *An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-Tuning* (arxiv.org/abs/2308.08747) 在最高 8B 的模型上实验发现, 更大的模型并没有更好的抵抗遗忘的能力.
- *Scaling Laws for Forgetting When Fine-Tuning LLMs* (arxiv.org/abs/2401.05605) 提出模型学的越多, 遗忘的程度越大.
- *LoRA Learns Less and Forgets Less* (arxiv.org/abs/2405.09673) 发现如果使用 LoRA, 其遗忘的程度会更小, 但是其代价是模型的学习能力的不足.

为了解决 Catastrophic Forgetting 的问题, 目前有一些方法可以尝试:
- 一个经典的解决 Catastrophic Forgetting 的方法是 **Experience Replay**, 也就是在微调新任务的过程中, 将之前的任务的数据也加入到当前的训练数据中. 
  - 在现代的情景下, 一个问题是我们并不总能获取到之前的任务的数据, 这时我们可以使用一些方法来进行数据的重建. 一个典型的方法是让语言模型自己先生成旧有任务的数据, 然后再混合到新的数据中进行训练. 这也称为 Pseudo Experience Replay.

- 还有一些其他类似的方法. 例如, 让 foundation model 对人类给定的正确答案进行 **paraphrase** 一个新的答案, 然后再利用这个答案进行训练. 其背后的思想是, 模型在 paraphrase 的过程中, 可能会保留一些原本的知识, 因此在微调的过程中可能会更好的避免遗忘.

- 再比如 **self-output** 的方法, 也就是让模型自己生成一个答案, 然后再利用这个答案进行训练. (此时往往训练的都是一些清晰的可分辨的任务如数学/编程等). 只要 foundation model 给出的答案是正确的, 那么这个答案就可以作为一个新的训练数据. 
  - 而这个方法和 RL 的方法也非常相像. 我们在进行 RL 的时候, 也会让模型自己生成一个答案, 然后再利用这个答案进行训练. 这可能也是为什么我们往往会把 RL 放在模型训练的最后一步的原因.
  - 即使生成答案的 Model 与微调的 Model 不同, 这种类似的 self-output 的方法也可以有效的避免遗忘. 而且这样的方法还能很好的避免由于 foundation model 能力太弱而导致生成的训练答案不够好的问题.
    - *I Learn Better If You Speak My Language* (arxiv.org/abs/2402.11192) 该文章通过对比人类提供的答案/Claude 生成的答案/GPT-4 生成的答案, 发现语言模型生成的答案 post-training 的效果要比人类提供的答案要好. 这里还进一步提出了一个 minimum change 的方法, 也就是让 foundation model 生成一个答案, 然后让更高级的例如 GPT-4 的模型仅仅对这个答案进行一些小的修改以确保其正确性. 然后用这个答案进行训练. 这样的方法可以有效的避免遗忘, 而且还可以提升模型的能力.

因此一个最后的 takeaway 是, 在当前的大语言模型中, 进行 post-training 以提高模型在某个领域的能力是一个非常重要的方向. 但是除去关注其在特定领域的能力, 我们也要关注其在习得这个能力时损失遗忘了多少原有的能力. 这时候, 我们可以使用一些方法来进行 Catastrophic Forgetting 的缓解. 而一个 general 的思想是, 我们要用模型"自己的话"来进行训练, 这样会更好的避免遗忘.

