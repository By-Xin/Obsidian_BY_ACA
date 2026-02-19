---
aliases: [Context Engineering, Prompt Engineering, In-Context Learning, ICL, 上下文工程, 提示工程]
tags:
  - concept
  - ml/deep-learning
  - cs/nlp
related_concepts:
  - "[[Deep_Reasoning_for_LLMs]]"
  - "[[Pretrain_and_Alignment_for_LLMs]]"
  - "[[BERT]]"
  - "[[Uncertainty_in_LLMs]]"
  - "[[Post-Training_and_Forgetting]]"
  - "[[Transfer_Learning]]"
source: "李宏毅; Anthropic Claude System Prompt"
---

# Context Engineering

> Ref: https://www.youtube.com/watch?v=lVdajtNpaGI

## Introduction

### 什么是 Context Engineering?

总的而言, 语言模型可以认为是一个 next-token-predictor. 给定输入 $x_t$, 语言模型 $f_{\text{LM}}$ 会输出下一个 token 的概率分布 $f_{\text{LM}}(x) = \mathbb{P}(x_{t+1}|x_t)$. 这时当我们的输出 $f_{\text{LM}}(x)$ 不是我们想要的, 我们有如下几种方式来调整输出:
- 调整模型参数, 即学习 $f_{\text{LM}}$. 在 LM 中通常包括 fine-tuning, LoRA, PEFT 等方法.
- 调整输入上下文 $x_t$ 而不是模型本身. 这就是 Context Engineering. 其不涉及任何模型的训练.


### Context Engineering v.s. Prompt Engineering 

- 其实本质上基本上是同样的! 只不过可能具体的关注细节不太一样.
  - **Prompt Engineering**: 更关注 prompt 的格式设计, "神奇咒语" (如 let's think step by step [arxiv.org/abs/2205.11916] ) 等 以让模型更好地理解任务. 不过随着技术的发展, 这类 prompt 的设计带来的提升越来越有限. 
  - **Context Engineering**: 当单纯的 prompt 设计已经不能满足需求时, 我们希望通过更自动化的上下文管理来提升模型的能力. 

## Context 里面需要有什么?

Context 一般包括如下几类信息:
- **User Prompt**: 提供前提, 任务范例等
  - 任务说明. 例如: "写一封给老板的请假邮件"
  - 详细指引. 例如: "开头先道歉, 然后说明请假原因, 最后表达感谢"
  - 额外条件. 例如: "100字以内"
  - 输出风格. 例如: "正式, 幽默"
- **System Prompt**: 由开发人员提供, 在每次进行对话时提供给语言模型的上下文信息. 
  - Claude Opus 4.1 公开了其 System Prompt (https://docs.anthropic.com/en/release-notes/system-prompts#august-5-2025).
      - 部分摘录如下 :
        ```
        The assistant is Claude, created by Anthropic.
        The current date is {{currentDateTime}}.
        Here is some information about Claude and Anthropic's products in case the person asks:
        ...
        ```
      - 该版本的 System Prompt 一共超过 2000 字, 包含信息众多:
        - 基本身份与产品咨询
        - 使用说明与限制
        - 互动态度与使用者回馈
        - 安全与禁止事项
        - 回应风格与格式
        - 知识与事实性
        - 自我定位与哲学原则 (Claude does not claim to be human or conscious)
        - 错误处理与互动细节 (If corrected, Claude first thinks carefully before acknowledging)
- **Dialogue History**: 相当于模型的短期记忆.
- **Long term memory**: OpenAI 在 2024 年 9 月的更新支持了 ChatGPT 的长期记忆功能. 

这样通过 prompt 来提供任务相关的信息, 让模型更好地理解任务的范式也称为 In-Context Learning 这种方式并没有任何模型的参数更新.

