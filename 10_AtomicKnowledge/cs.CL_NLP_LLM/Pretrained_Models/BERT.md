---
aliases: [BERT, Bidirectional Encoder Representations from Transformers, 双向编码器表示]
tags:
  - concept
  - cs/nlp
  - ml/deep-learning
related_concepts:
  - "[[Transformer]]"
  - "[[Word_Embeddings]]"
  - "[[Self-Attention]]"
  - "[[Word2Vec]]"
  - "[[GloVe]]"
  - "[[Pretrain_and_Alignment_for_LLMs]]"
  - "[[Cross_Entropy]]"
source: "李宏毅 (机器学习,2021)"
---

# BERT

> Refs: 李宏毅 (机器学习,2021, https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php); https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT

## 从应用角度直观理解BERT

### BERT 的最核心功能: 语言理解

BERT (Bidirectional Encoder Representations from Transformers) 是一个由 Google 在2018年推出的基于 Transformers 架构的自然语言处理模型.  BERT 的本质是一个**语言理解模型**, 其本质核心是**学习文本的深层语义信息表示**. BERT 模型的输入是一个经过基本处理的文本序列 *(主要包括分词和位置编码等)*, 而其最基本的输出是对于每个 token 的一个高维的向量表示. 简单的讲, input 是文本序列, output 是文本序列中每个 token 的数学向量表示.

BERT 之所以强大, 是因为其能够完成许多下游的 NLP 任务, 如文本分类, 问答, 命名实体识别等. 不过这些任务并不是 BERT 本身 "直接" 完成的. BERT 本身只是一个**语言理解模型**, 它的核心是学习文本的**深度语义表示**. 为了完成具体的 NLP 任务，我们在 BERT 之上加一层额外的任务特定层(Head), 然后进行微调 (Fine-tuning) (可以理解为在得到一个很好的语义表示后, 针对具体任务进行后续的一些结构补充和训练). 

### BERT 与 word2vec, GloVe 等传统词向量模型的区别

从表象上看, BERT 似乎和传统的词向量模型 (如 word2vec, GloVe) 没有太大的区别, 都是将文本转换为向量表示. 不过在其原理和表达能力上, BERT 和传统的词向量模型有很大的区别  (以 word2vec 为例):
- **Word2vec** 
  - Word2vec 是基于相对简单的神经网络模型, 主要是通过训练一个浅层的神经网络来学习词的向量表示.
  - 词的向量表示是静态的, 即对于同一个词, 其在不同的上下文中的表示是相同的.
  - 无法区分同一个词在不同上下文中的含义 (如 "bank" 在 "bank account" 和 "river bank" 中分别表示"银行"和"河岸", 但在 word2vec 中是相同的)
  - 其训练是基于词的共现信息, 主要适用于词级别的任务 (如计算词与词之间的相似度, 词聚类等)
  - 优点是: 1) 计算成本较低, 只需要训练一个简单的神经网络(如 Skip-gram), 适合大规模数据快速训练; 2) 训练后存储的词向量较小(如 300 维), 计算效率高, 适用于低资源环境.
- **BERT**
  - BERT 是基于深度 Transformer 模型的, 具有更强的表达能力, 能够学习到文本的深层语义信息.
  - 词的向量表示是动态的, 即对于同一个词在不同上下文中的表示是不同的. (如上文提到的 "bank" 对于 BERT 就会有不同的表示)
  - BERT 因此可以更好地处理语境依赖和歧义词, 适用于下游 NLP 任务 (如情感分析、问答、阅读理解)
  - 缺点是: 1) 计算成本高, 预训练时需要使用大规模 Transformer 模型, 对硬件 (GPU)要求高; 2) 训练出的模型参数量巨大 (如 BERT-Base 110M 参数，BERT-Large 340M 参数)

总而言之, BERT 是一个和 Word2Vec 等类似于**同样一个任务处理工具的两个迭代版本**. BERT 更为复杂, 也更为强大. 如果任务仅需要静态的词向量(比如词相似度, 聚类), Word2Vec 是一个更轻量的选择. 如果你的任务需要理解上下文 (比如文本分类, 情感分析, 阅读理解), BERT 及其变体 (如 RoBERTa,ALBERT) 会有更好的效果. 

## BERT 的语义理解能力的实现 (Pre-training)

首先介绍 BERT 的基本语义理解能力是如何训练实现的. 

### BERT 的模型结构

**如不深究其原理, 一个简单的理解就是 BERT 的模型的输入是一串序列 (常为一串token), 输出也是一串序列 (每个 token 对应一个向量表示), 并且二者之间是一一对应的. 输入的序列有多长, 输出的序列也有多长.**

具体而言, BERT 的模型结构就是一个基于 Transformer 的深度神经网络. Generally, Transformer 可以分为 Encoder 和 Decoder 两部分, 而 BERT 只使用了 Transformer 的 Encoder 部分.

BERT 就相当于将多个 Transformer Encoder 串联在一起,  通过多层的 Transformer Encoder 来学习文本的深层语义信息.


### Self-Supervised Learning in BERT

#### Self-Supervised Learning 简介

Self-Supervised Learning 是 BERT 的核心训练方法.
- 传统的 Supervised Learning 是指在训练时我们的数据包括了feature $X$ 和 label $Y$, 模型通过训练来得到一个 $\hat{Y} = \text{model}(X)$, 使得 $\hat{Y}$ 尽可能接近 $Y$.
- 对于 Self-Supervised Learning, 我们的数据中没有 label $Y$, 我们需要一些方法来自己生成 label. 再通过模型训练来使得模型能够预测这个 label.

其在 BERT 中在 NLP 的文本数据上的具体实现 (即训练流程) 如下. 

#### Random Token Masking

BERT 对一串给定预料的学习的一个核心方法就是 **Random Token Masking**. 

Random Token Masking 是说, 在面对一串文本序列, 随机地将其中的一些 token "mask(遮盖)"掉, 然后让模型去预测这些被遮盖的 token.  具体的 mask 既可能是直接将 token 替换为一个特殊的 mask token `[MASK]`, 也可能是将选取的 token 替换为另一个随机的 token (也就是说 mask 哪个 token 是随机的, 用什么方法 mask 是随机的, mask 成什么 token 也是随机的).

不过不论如何, 对于 BERT 模型, 即使其 input 出现了一些 mask token, 其也会输出一个同样序列长度的 output. 因此我们定位到 mask token 的位置, 通过线性变化和 softmax 等操作, 将模型预测出的数学向量还原为具体的字符 token, 并且将这个 mask token 的预测结果与该位置的真实 token 进行比较, 通过交叉熵损函数来计算模型的预测误差, 从而进行模型的训练. 再稍微具体点说, softmax 的之后的结果是一个概率分布, 表示在现有的 token 词表中, 预测的这个位置的 token 的概率. 

#### Next Sentence Prediction

除了 Random Token Masking, BERT 还使用了 **Next Sentence Prediction** 来训练模型. 具体而言, 在训练时, BERT 会随机地从语料库中选取两个句子, 然后让模型判断这两个句子是否是连续的. 对于两个句子, 我们会用特殊的 token `[CLS]` 来标记句子的开始, 用 `[SEP]` 来标记每个单句的结束. 同样对于 BERT 模型, 其输出的结果是一个一样长度的序列. 不过这里我们只需要关注最开始的 `[CLS]` token 对应的输出结果. 我们希望模型在这个位置上输出一个二分类的结果 (YES/NO), 来表示模型的对于给定的这两个句子是否是连续的两句话的判断.

不过也有一些质疑者认为 Next Sentence Prediction 这个任务对于模型的训练效果并不是很大. 或许因为这个任务相对于 Random Token Masking 来说, 其实是一个相对简单的任务, 因此没有学到太多有用的信息. 
- Robustly optimized BERT approach (RoBERTa) (https://arxiv.org/abs/1907.11692)
- Sentence Order Prediction (SOP) (https://arxiv.org/abs/1902.00751) 提出了一个更复杂的任务替代 Next Sentence Prediction 进行 YES/NO 二分类的任务: SOP 会将两个相邻句子随机打乱顺序, 让模型判断这两个句子的顺序. 

## BERT 的下游任务处理 (Fine-tuning)

我们通过上面的 Self-Supervised Learning 的方式 (即通过 Random Token Masking 和 Next Sentence Prediction) 来 pre-train 了 BERT 模型具有了基本的语义理解能力.  

但是 BERT 为了完成具体的 NLP 下游任务 (down-stream tasks), 我们需要在 BERT 之上加一层额外的任务特定层(Head) (相当于针对各种不同的下游任务而附加的一些神经网络模型), 然后进行微调 (Fine-tuning). 

注意, 在 Fine-tuning 时, 我们不会再使用 Random Token Masking 和 Next Sentence Prediction 这两个任务, 而是直接使用下游任务的数据集进行训练. 因此 Fine-tuning 的过程是一个有监督的训练过程, 我们需要下游的任务提供的 label 信息来进行具体的训练.

而在这个训练的过程中, 我们还是要在 BERT 的预训练的基础上, 继续既更新 BERT 的参数, 又更新任务特定层 (Head) 的参数 (Head 的参数就是随机初始化进行训练即可).

下面针对不同种类的 NLP 任务, 我们会介绍如何在 BERT 之上进行 Fine-tuning.

### Sequence to Classification

第一种类别是 Sequence to Classification, 即输入是一个文本序列, 输出是一个分类结果. 常见的下游任务有文本分类, 情感分析等. 其实现方式是在 BERT 之上加一个全连接层 (Fully Connected Layer) 来进行分类. 我们主要就关注第一个 token `[CLS]` 经过模型后的输出结果, 然后通过全连接层来进行分类, 并且与真实的 label 进行比较, 通过交叉熵损失函数来进行训练.

### Sequence to Sequence (Seq2Seq) [Same Length]

第二种类别是一个 Seq2Seq 任务, 即输入是一个文本序列, 输出也是一个文本序列. 且特别地, 要求输入和输出的序列长度是相同的. 常见的下游任务有 POS tagging (词性标注).  这里可以直接使用 BERT 将每个 token 都对应了一个向量表示, 然后将这个向量表示输入到一个全连接层中, 来预测每个 token 的标签.

### 2-Sequences to Classification 

第三种类别是一个 2-Sequences to Classification 任务, 即输入是两个文本序列, 输出是一个分类结果. 常见的下游任务有 NLI (Natural Language Inference, 自然语言推理). 在 NLI 中, 我们需要判断两个句子之间的逻辑关系, 通常有三种关系: Entailment (蕴含), Neutral (中立), Contradiction (矛盾). 因此输入是两个句子, 输出是一个这三种关系的分类结果.  例如: *premise: "A person on a horse jumps over a broken down airplane. (假定: 一个人骑马跳过一架坠毁的飞机)"* 和 *hypothesis: "A person is at a diner. (猜想: 一个人在餐馆)"* 之间的关系是 Contradiction. 

对于 BERT 来说, 我们可以将两个句子的 token 序列拼接在一起, 然后通过一个全连接层, 根据 `[CLS]` token 的输出来进行分类.

### Extractive Question Answering

第四种类别是 Extractive Question Answering, 即输入是一个文本序列 $D$ 和一个基于这个文本的问题 $Q$ (即只能是一些信息提取的问题), 模型的输出是两个 token 的索引 $(s,e)$, 表示针对问题 $Q$ 在文本 $D$ 中的一个答案在文本中的起始和结束位置. 例如: *文本: "The capital of France is Paris."* 和 *问题: "What is the capital of France?"* 的答案是 $(4,5)$, 即 "Paris". 

## BERT 的 Hugging Face 库中的初步应用

Hugging Face 的 `transformers` 库 提供了简单易用的 API, 可以快速加载 BERT 并应用到各种 NLP 任务, 如文本分类, 命名实体识别, 文本生成等.  一个简单的操作流程如下 (该流程在 Colab 中可以有效运行):


1. **安装 `transformers` 库**
    ```bash
    pip install transformers torch
    ```

2. **加载预训练的 BERT 模型**
    ```python
    from transformers import BertTokenizer, BertModel

    # 加载 BERT 分词器和模型（BERT-Base） 
    # bert-base-uncased 是 BERT 的无区分大小写版本（适用于英语）
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
    model = BertModel.from_pretrained("bert-base-uncased")

    # 测试文本
    text = "BERT is a powerful model for NLP tasks."

    # 分词并转换为张量
    inputs = tokenizer(text, return_tensors="pt")

    # 获取 BERT 输出
    outputs = model(**inputs) 
    
    # 获取最后一层的隐藏状态, 它包含了 BERT 对输入文本的编码表示, 是我们最终需要的结果
    last_hidden_states = outputs.last_hidden_state # last_hidden_state 有三个维度: (batch_size, sequence_length, hidden_size) 
    print(last_hidden_states.shape)  # torch.Size([1, token_length, 768])
    ```

3. **调取 BERT 的词向量表示**
    ```python
    # 获取第i个 token 的词向量表示
    token_i = 2
    word_embedding = last_hidden_states[:, token_i, :]  
    print(word_embedding.shape)  
    print(word_embedding)
    ```

进一步我们还可以通过 `from transformers import BertForSequenceClassification` 等类来直接调用已经预训练+微调好的 BERT 模型来完成具体的 NLP 任务. 或者我们也可以在预训练好的 BERT 之上自己添加一些任务特定的层来完成自己的 NLP 任务. 总而言之, 只要我们得到了 BERT 词向量表示, 我们就可以在此基础上进行各种 NLP 任务的处理.

