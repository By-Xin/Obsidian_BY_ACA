---
aliases: [LLM推理, Deep Reasoning, LLM Reasoning, Chain of Thought, CoT, 思维链]
tags:
  - concept
  - cs/nlp
  - ml/deep-learning
related_concepts:
  - "[[Reinforcement_Learning]]"
  - "[[Knowledge_Distillation]]"
  - "[[Pretrain_and_Alignment_for_LLMs]]"
  - "[[Uncertainty_in_LLMs]]"
  - "[[Context_Engineering]]"
  - "[[Post-Training_and_Forgetting]]"
source: "李宏毅, DeepSeek-R1分析"
---

# Deep Reasoning for LLMs

## DeepSeek-R1 這類大型語言模型是如何進行「深度思考」（Reasoning）的？

## 介绍


在当下, 例如 ChatGPT o1/o3/o4, DeepSeek-R1, Gemini 2 Flash Thinking, Claude 3.7 Sonnet 等大型语言模型具有一定的长思考能力. 

对于这类的语言模型, 其深度思考具有一定的模式:
- 首先在给定问题后, 不会直接对问题进行回答, 而是会先在 `<think> ... </think>` 之间进行思考, 最后根据思考的结果进行回答. 
- 思考的主要内容可能包括:
  - Verification: e.g. "Let me check the answer ..."
  - Explore: e.g. "Let's  try a different approach ..."
  - Plan: e.g. "Let's first try to ..."

在 LLM 语境下, 我们也往往称呼这种思考为 Reasoning 推理 (注意要区分 Inference). 

这里还涉及到几个相关的概念: 
- Test-Time Compute: 这是指模型即使完成了训练已经在进行 inference (test-time) 了, 仍然会进行一系列的复杂计算.
- Test-Time Scaling: 这是指模型在进行 inference 时往往投入更多的计算资源会带来更好的效果.
  - *Scaling Scaling Laws with Board Games (arxiv.org/abs/2104.03113)* 研究了计算资源应当投入到训练过程中 (如训练一个更大的 Policy Network) 还是在推理过程中 (如使用更大的 Monte Carlo Tree Search) 的问题. 其发现, 用少量的 test-time compute 的资源投入就可以减少大量的训练资源投入.

对于让模型具有这样 Reasoning 的能力, 目前有几种方法:
- 更强的思维链 (Chain of Thought, CoT) (不需要微调)
- 给模型 Reasoning 的工作流程 (不需要微调)
- Imitation Learning, 教模型 Reasoning 的工作流程 (需要微调)
- 以结果为导向学习 Reasoning (需要微调)

## 更好的 Chain of Thought

Chain of Thought (CoT) 是一种让模型在进行推理时, 先列出思考的过程, 然后再给出最终的答案. 其在 2022 年被提出, 有两种主要的形式:
- **Few-Shot CoT** (*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (arxiv.org/abs/2201.11903)*)
  - 这种方法在给定的 prompt 中在提问之前, 会先给模型提供一些问答的范例, 其中回答的过程会包含 CoT 的过程. 
  - 例如: 
    ```
    Q: Roger has 5 tennis balls. He buys 3 more cans of tennis balls. Each can has 4 balls. How many tennis balls does Roger have now?
    A: Roger started with 5 tennis balls. He bought 3 cans of tennis balls, and each can has 4 balls. So he bought 3 * 4 = 12 more tennis balls. Now he has 5 + 12 = 17 tennis balls.
    Q: A juggler can juggle 16 balls. Half of the golf balls are red. How many red balls does the juggler have?
    A:
    ```
- **Zero-Shot CoT** (*Large Language Models are Zero-Shot Reasoners (arxiv.org/abs/2205.11916)*)
  - 这种方法在给定的 prompt 中, 只给出问题, 而不提供任何的范例. 研究发现仅仅是让模型 "think step by step" 就可以让模型进行 CoT 的推理.
  - 例如: 
    ```
    Q: A juggler can juggle 16 balls. Half of the golf balls are red. How many red balls does the juggler have?
    A: Let's think step by step. 
    ```

在当下的 Reasoning 研究中, 有时我们也称之为 "Long Chain of Thought" (LoT). 把之前的传统的 CoT 也称之为 "Short Chain of Thought" (SoT). 
- 进一步发现, 有时如果只是单纯的告诉模型 "think step by step", 其可能并没有办法进行深度的思考.  
  - 因此也有人提出了 **supervised CoT** (*Supervised Chain of Thought, arxiv.org/abs/2410.14918*), 其在给定的 prompt 中会一点一点告诉模型如何进行思考, 例如:
    ```
    請仔細思考並詳細回答以下問題。在回答前，請先深入分析題目的要求，
    訂出一個完整且清晰的解題計畫，明確列出你將如何分步完成這個問題。
    在執行每一個主要步驟前，請再次訂出該步驟的子計畫，
    仔細列出需要處理的細節，然後再按部就班地執行。
    每執行完一個步驟或子步驟後，請進行多次驗算，
    確保該步驟的答案絕對正確，並考量所有可能的解法。
    若在驗算過程中發現問題，請立即回到該步驟重新訂定或調整計畫。
    在進行以上過程時，務必將你詳細而完整的思考過程以及所有計畫、
    子計畫、驗算步驟，全部置於"<think>"和"</think>"這兩個符號之間。

    Q: 123 x 456 =
    ```



## 给模型 Reasoning 的工作流程

一个简单粗暴的思路是让模型尽可能多的进行 exploration, 当尝试的次数足够多时, 其可能会找到一个好的答案.
- *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling (arxiv.org/abs/2407.21787)* 研究发现, 对于数学问题, 让模型进行多次的采样, 只要尝试的足够多, 基本都可以找到正确的答案. 当然, 更好的模型所需的尝试次数往往会更少.

然而这样的 exploration 的一个问题是, 如何选出最好的答案? 可能的方法有:
- **Majority Vote** (Self-Consistency): (*Self-Consistency Improves Chain of Thought Reasoning in Language Models (arxiv.org/abs/2203.11171)*) 在所有模型出现的答案中, 选出出现次数最多的答案.
  - Majority Vote 的表现往往很好, 是一个常见的 baseline. (huggingface.co/HuggingFaceH4/blogpost-scaling-test-time-compute)
- **Confidence Score**: (*Chain-of-Thought Reasoning Without Prompting (arxiv.org/abs/2402.10200)*) 让模型给出每个答案的置信度, 选出置信度最高的答案.
- **Answer Verification**: (*Training Verifiers to Solve Math Word Problems (arxiv.org/abs/2110.14168)*) 通过引入另一个 Verifier (可能是另一个LLM), 给每个答案进行评分, 选出评分 Best-of-N 的的答案.
  - Verifier 甚至可以是通过给定 ground-truth 的数据进行训练的专用模型
- **Process Verification**: (*Let's Verify Step by Step (arxiv.org/2305.20050*)) 除了在最终给出答案时进行 verification, 我们也应当在思考的中间过程进行验证. 我们在每一步的思考中都给出一个 Verifier, 只对通过验证的思考继续后面的思考.
  - 一个实现的方法就是 **In-Context Learning**: 在 Prompt 中明确指定模型一步一步进行输出思考 `<step> ... </step>`, 然后在每次发现 `<\step>` 时就让模型暂停生成并使用 Process Verifier 进行验证.
  - 可以使用 Beam Search, DVTS 等搜索方法来保留最优的部分, 得到更好的答案.
    ![通过 Beam Search, 即使 1B 的模型也能有很好的效果](https://huggingface.co/datasets/HuggingFaceH4/blogpost-images/resolve/main/methods-all.png)
  

通常为了让更清楚的进行表达, 我们往往还会要求模型将最终的答案进行格式输出, 放置在 `<answer> ... </answer>` 之间.

## Imitation Learningn: 教模型 Reasoning 的工作流程

这相当于是一种 Post-Training 的方法, 相当于我们要让一个不会 Reasoning 的 Foundation Model 进行微调, 让其学会 Reasoning 的工作流程.

具体而言, 我们的训练数据要包含 `Input, Reasoning Process, Ground Truth` 三个部分. 通过这样的数据对模型进行微调, 让其学会 Reasoning 的工作流程. 而其中一个非常重要的问题就是: Reasoning Process 的数据应该如何得到?
- 一个直观的做法是: 通过引入另一个模型, 让其给出 Reasoning Process 和 Answer. 我们假设如果模型得到了正确的答案, 那么 Reasoning Process 也应该是正确的. 因此我们就保留这样的 Reasoning Process 作为训练数据. 
- 对于一些可能没有标准答案的问题, 我们也可以引入 Verifier 来验证答案的合理性以得到合理的 Reasoning Process.

*rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking (arxiv.org/abs/2501.04519)* 研究了在推论的每一步加入 verifier, 通过类似树状搜索的方法, 让模型在每一步都进行验证. 这样模型的 Reasoning 过程可能更为合理可靠. 

然而另一个问题是: Reasoning 的每一步一定必须要是正确的吗? 换言之, **纠错**或许也是 Reasoning 的重要能力. 换言之, 如果我们给模型训练的 Reasoning Process 一直是完美的, 模型可能就并不会学会纠错的能力, 这样其 Reasoning 时的 robustness 可能会很差.
- 因此也许我们可能需要在 Reasoning 的过程中加入一些噪声得到一些有可能错误的 Reasoning Process. 并且提供一个进行回退并修正的能力.

### Knowledge Distillation

刚刚提到的通过引入另一个模型来进行 Reasoning Process 的生成, 其实就是一种知识蒸馏 (Knowledge Distillation). 其主要的思路是: 我们有一个大型的模型, 其可以给出 Reasoning Process 和 Answer. 然后我们通过这个大型的模型来指导一个小型的模型进行 Reasoning 的工作流程.

*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arxiv.org/abs/2501.12948)* (即 DeepSeek-R1 的论文) 中也指出, 通过知识蒸馏的方式, 让小模型学习大型模型的 Reasoning 的工作流程, 其可以得到更好的效果.
![表格中框选的模型即为进行 Knowledge Distillation 的 Foundation Model, 通过 DS-R1 的 Distill 后其在各个测试项目上的表现均有所提升](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250505231720.png)

## 以结果为导向学习 Reasoning (Reinforcement Learning)

该方法的思路是 DeepSeek-R1 的主要训练思路. 其核心思想是: 我们有一系列 Input + Ground Truth 的训练数据, 我们让模型进行 Reasoning 并给出最终的答案. 如果答案正确我们就给模型正向的奖励, 如果答案错误我们就给模型负向的奖励. 不过事实上我们并不关注 Reasoning 的具体内容, 我们只关注最终的答案是否正确.

在 DeepSeek 中, 通过 DeepSeek-v3-base 作为 Foundation Model, 通过纯 RL 的方式进行训练, 得到的模型为 DeepSeek-R1-Zero. 在 *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arxiv.org/abs/2501.12948)* 中, 即使是纯粹的 RL 的训练, 其在 Reasoning 的任务上也有着很好的效果.
![红蓝两个折线即为 DeepSeek-R1-Zero 在不同训练阶段的表现](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250505232530.png)

### Aha Moment

在 *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arxiv.org/abs/2501.12948)* 中, 其提出了一个 Aha Moment 的概念, 指代其在推理过程中突然产生了纠错的能力.

![DeepSeek R1 中的 Aha Moment](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250505232833.png)

### R1 的训练过程

DeepSeek-R1-Zero 虽然具有了一定的 Reasoning 能力, 但是其并不是可以直接使用的模型. 其在 Reasoning 的过程中由于只关注结果的正确性, 其 Reasoning 的过程中可能会包含多语言混杂等一系列的问题. 这里, DeepSeek 通过一系列的手段对模型进行微调, 得到了我们最终在使用的 DeepSeek-R1 模型. 其具体的训练过程如下:

- 在得到 DeepSeek-R1-Zero 后, 用其生成 Reasoning 的过程, 并且通过大量的人力润色修订 (human annotation) 得到可用的 Reasoning 过程. 此外, DeepSeek 还通过 few-shot prompting with long CoT, directly prompting 等方法得到更多 detailed 的数据 (约几千条).
- 再利用这些 Reasoning 的数据, 对 DeepSeek-v3-base 进行 Imitation Learning 微调, 得到一个新的模型 Model A. 
- 在 Model A 的基础上, 再额外进行 RL 的训练. 其中, 除了对正确率的要求外, 还会要求模型在 Reasoning 的过程中要使用一致的语言. 在训练后又得到一个新的模型 Model B.
- 再使用这个新的模型对一系列数据进行 Reasoning, 得到许多 Reasoning 的数据和答案. 并且这里还引入 DeepSeek-v3 作为 verifier, 让其对 Reasoning 和答案进行评分, 最终得到一个新的 Reasoning 数据集. 这里得到约 60 万条数据. 此外又让 DeepSeek-v3 进行 Self-Output 得到了 20 万条数据混入其中 (避免遗忘).
- 让 DeepSeek-v3-base 在这组约 80 万条数据上进行 Imitation Learning, 得到一个新的模型 Model C.
- 对于 Model C 进行额外的 RL 训练, 着重处理 Safety 和 Helpfulness 的问题, 得到最终的模型 DeepSeek-R1.

> DeepSeek 也尝试使用了 MTCS 和 process verification 等方法, 但是没有最终使用. 

### Foundation Model 也很重要

![Foundation Model 会影响模型的水平 ](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250505235125.png)

以 Qwen-32B-Base 作为 Foundation Model 进行 RL 的效果就会弱于以 DeepSeek-v3-base 作为 Foundation Model 的效果. 而直接通过 Imitation Learning 的方式进行 Distillation 的效果微调 Qwen-32B-Base 的效果反而比较有效. 总而言之, **RL 是强化模型原有的能力, 而不是创造新的能力**.

因此 *Understanding R1-Zero-Like Training: A Critical Perspective (arxiv.org/abs/2503.20783)* 等研究指出, DeepSeek-v3-base 在 RL 之前本来就具有 Aha 的能力, RL 只是强化了这个能力.

