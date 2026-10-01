#formulation 

> 说明: 本文档用来记录 MinPlus 项目的正式 Formulation

## LM 的解码过程
现有一个输入序列 $\mathbf{x_{1:t}} := (x_1, x_2, \ldots, x_t)^\top$, 其中每个 $x_i \in \mathcal{V}$ 是一个 token, $\mathcal{V}$ 是 token 的集合 (vocabulary). 一个典型的 Transformer-based 的自回归语言模型 (LLM) 主要包含如下几个板块:

1. **Input Embedding** $f_\text{emb}$: 将输入的 token 序列 $\mathbf{x_{1:t}}$ 映射到一个实数向量序列 $\mathbf{h_{1:t}} := (h_1, h_2, \ldots, h_t)^\top$, 其中每个 $h_i \in \mathbb{R}^d$ 是一个长度为 $d$ 的实数向量. 故:

$$ \mathbf{h_{1:t}} = f_\text{emb}(\mathbf{x_{1:t}}) \in \mathbb{R}^{t \times d} $$

2. **Decoder Layers** $f_\text{dec}$: 将输入的实数向量序列 $\mathbf{h_{1:t}}$ 通过一系列的解码层 (decoder layers) 进行处理，得到一个新的实数向量序列 $\mathbf{z_{1:t}} := (z_1, z_2, \ldots, z_t)^\top$, 其中每个 $z_i \in \mathbb{R}^d$. 故:

$$ \mathbf{z_{1:t}} = f_\text{dec}(\mathbf{h_{1:t}}) \in \mathbb{R}^{t \times d} $$

3. **Language Modeling Head** $f_\text{lm}$: 将解码层输出的实数向量序列 $\mathbf{z_{1:t}}$ 映射到一个长度为 $t$ 的 token 得分序列 (logits) $\mathbf{l_{1:t}} := (l_1, l_2, \ldots, l_t)^\top$, 其中每个 $l_i \in \mathbb{R}^{|\mathcal{V}|}$ 是一个长度为 $|\mathcal{V}|$ 的实数向量，表示每个 token 在该位置的得分. 故:

$$ \mathbf{l_{1:t}} = f_\text{lm}(\mathbf{z_{1:t}}) \in \mathbb{R}^{t \times |\mathcal{V}|} $$

4. **Decoding Function** $f_\text{decoding}$: 将 token 得分序列 $\mathbf{l_{1:t}}$ 通常取最后一个 token 的得分 $l_t$ 作为输入, 经过 softmax 等解码函数处理后, 得到最终选择输出的 token $x_{t+1}$, 即:

$$ x_{t+1} = f_\text{decoding}(l_t) \in \mathcal{V} $$

整个语言模型本质上也可以看作是对如下概率分布的建模 (假设序列长为 $T$):

$$ \mathcal P(x_{1:T}) = \prod_{t=1}^T \mathcal P(x_t | x_{1:t-1}) $$

由于我们要重点讨论模型的解码过程, 因此将从上述的输入嵌入 (input embedding) 直到解码函数前得到 logits 的部分抽象为一个函数 $\mathcal{M} : \mathcal{V}^t \to \mathbb{R}^t$，即:

$$ \mathbf{l_{1:t}} = \mathcal{M}(\mathbf{x_{1:t}}) \in \mathbb{R}^{t \times |\mathcal{V}|} $$

这时最后一个 token 的 logits 为 (在不引起歧义的情况下，省略下标 $t$):

$$ \mathrm l = \begin{bmatrix} \ell_{1} \\ \ell_{2} \\ \vdots \\ \ell_{|\mathcal{V}|} \end{bmatrix} \in \mathbb{R}^{|\mathcal{V}|} $$

传统的做法中我们会直接对这个 $l_t$ 进行 softmax 处理得到概率分布:

$$ \mathrm p = \begin{bmatrix} \vdots \\ p_i \\ \vdots \\ \end{bmatrix} = \text{softmax}(l) = \begin{bmatrix} \vdots \\ \frac{e^{\ell_{i}}}{\sum_{j=1}^{|\mathcal{V}|} e^{\ell_{j}}} \\ \vdots \\ \end{bmatrix}\in \mathbb{R}^{|\mathcal{V}|} $$

## 引入 Min-p Plus
我们这里参考 Min-p 的思路, 引入一个 hard-truncation 函数: $f_\tau : \mathbb{R} \to \mathbb{R}$, 其定义为:

$$ f_\tau(x) = \begin{cases} x, & \text{if } x \geq \tau \\ 0, & \text{if } x < \tau \end{cases} $$

对当前的概率分布 $\mathrm p$ 进行截断 (truncation) 处理并重新归一化 (normalization) 得到新的概率分布 $\mathrm p_\tau$:

$$ \mathrm p_\tau = \text{norm}(f_\tau(\mathrm p)) = \text{norm} \begin{bmatrix} \vdots \\ f_\tau(p_i) \\ \vdots \\ \end{bmatrix} = \begin{bmatrix} \vdots \\ \frac{f_\tau(p_i)}{\sum_{j=1}^{|\mathcal{V}|} f_\tau(p_j)} \\ \vdots \\ \end{bmatrix}:= \begin{bmatrix} \vdots \\ p_\tau^i \\ \vdots \\ \end{bmatrix} \in \mathbb{R}^{|\mathcal{V}|} $$

由于 truncation 函数的存在, 向量 $\mathrm p_\tau$ 是稀疏的.

推广上述情景, 给定超参数 $K$ 个已知的 truncation 阈值 $\{\tau_1, \tau_2, \ldots, \tau_K\}$, 我们可以以此类推得到 $k$ 个截断后的概率分布 $\{\mathrm p_{\tau_1}, \mathrm p_{\tau_2}, \ldots, \mathrm p_{\tau_K}\}$, 其中每个 $\mathrm p_{\tau_k} = \text{norm}(f_{\tau_k}(\mathrm p)) := F_k(\mathrm p)$.

因此我们可以将这 $K$ 个截断后的稀疏概率分布 $\{\mathrm p_{\tau_1}, \mathrm p_{\tau_2}, \ldots, \mathrm p_{\tau_K}\}$ _<font style="color:#DF2A3F;">这里有点类似于可以多选的分类问题，根据不同的任务选择合适的tau</font>_作为一组 pseudo-basis, 通过线性组合的方式得到一个新的映射 $\phi_\Theta : \mathbb{R}^{|\mathcal{V}|} \to \mathbb{R}^{|\mathcal{V}|}$, 其 element-wise 地作用在原始的概率分布 $\mathrm p$ 上:

$$ \phi_\Theta(\mathrm p) = \sum_{k=1}^K \theta_k F_k(\mathrm p)\in \mathbb{R}^{|\mathcal{V}|} $$

其中 $\Theta = [\theta_1, \theta_2, \ldots, \theta_K]^\top \in \mathbb{R}^K$ 是权重系数, 但是其本身可以是一个例如 MLP 等的可学习结构. 用向量形式, 记 $\mathcal{F} = [F_1, F_2, \ldots, F_K]^\top \in \mathbb{R}^{|\mathcal{V}| \times K}$, 则:

$$ \phi_\Theta(\mathrm p) = \mathcal{F}(\mathrm{p}) \Theta $$

此外我们还需要系数向量 $\Theta$ 满足一个稀疏性约束. 即给定超参数 $C > 0$, 使得 $\Theta$ 的 $l_0$ 范数 (即非零元素的个数) 不超过 $C$:

$$ \|\Theta\|_0 \leq C $$

<font style="background-color:#FBDE28;">最终得到的映射</font> $\phi_\Theta$ <font style="background-color:#FBDE28;">可以看作是对原始概率分布</font> $\mathrm p$ <font style="background-color:#FBDE28;">的一个变换. 为了使其输出的结果仍然是一个概率分布, 重新进行 norm 归一化处理: </font>

$$ \mathrm{\tilde p} = \text{norm}(\phi_\Theta(\mathrm p)) \in \mathbb{R}^{|\mathcal{V}|} $$

(*此外作为权重, 要求其总和为 1:

$$ \sum_{k=1}^K \theta_k = 1 $$

> Q: 我们似乎不需要这个和为1 的约束?因为我们已经把结果又经过了一个得到一个概率分布. (不过norm这个操作似乎更简单一些, 没有引入额外约束条件). 另外这里的 norm 似乎最好还是一个 softmax 的形式, 这样除了归一化还能保证其非负性, 以减少例如 reward hack 的风险.
)
>

> Q. 如果进一步想把 $\Theta$ 本身也当做一个网络的话, 可以用一个可学习的神经网络 $\mathcal{G}$ 来表示 $\Theta$, 相当于我们有输入 $\mathcal{F}(\mathrm p) \in \mathbb{R}^{|\mathcal{V}| \times K}$, 输出 $\Theta \in \mathbb{R}^K$, 通过一个映射 $\mathcal{G}: \mathbb{R}^{|\mathcal{V}| \times K} \to \mathbb{R}^K$, 使得:
$ \Theta = \mathcal{G}(\mathcal{F}(\mathrm p)) \in \mathbb{R}^K $. 然后进行加权: $\phi_{\Theta}(\mathrm p) = \mathcal{F}(\mathrm{p}) \Theta$. 而具体的这里的 $\mathcal{G}$ 如果是 MLP 则需要拉直 (flatten) 处理, 或者甚至可以考虑用 attention 机制进行处理. 再或者可以直接通过神经网络得到 $\phi_{\Theta}(\mathrm p) = \text{MLP}(\mathcal{F}(\mathrm p))$. 不过感觉 $\Theta$ 越复杂, 过拟合或者 reward hack 或者不符合约束的风险就越大.
>

需要指出, 我们这里想要做的是一种面向更大模型的 alignment.

## 通过 Min-p Plus 进行 Inference
在推理 (inference) 阶段, 我们认为已经得到一组最优的超参数 $K^* \in \mathbb{N}, C^* \in \mathbb{N}, \tau^*_k \in \mathbb{R}, \forall k \in \{1, 2, \ldots, K^*\}$, 以及一个最优的参数向量 $\Theta^* \in \mathbb{R}^{K^*}$.
因此我们可以得到一个最优的映射 $\phi_{\Theta^*} :\mathbb{R}^{|\mathcal{V}|} \to \mathbb{R}^{|\mathcal{V}|}$, 其逐元素地作用在原始的概率分布 $\mathrm p$ 上:

$$ \phi_{\Theta^*}(\mathrm p) = \sum_{k=1}^{K^*} \theta_k^* F_k(\mathrm p) \in \mathbb{R}^{|\mathcal{V}|} $$

这时我们可以将 $\phi_{\Theta^*}(\mathrm p)$ 作为一个新的概率分布, 通过归一化函数得到最终的输出概率分布:

$$ \mathrm{\tilde p} = \text{softmax}(\phi_{\Theta^*}(\mathrm p)) \in \mathbb{R}^{|\mathcal{V}|} $$

我们可以通过普通的抽样方法 (如 top-k, nucleus sampling 等) 从这个新的概率分布中采样得到最终的输出 token $x_{t+1}$.

## Min-p Plus 的训练
目前的训练目标的核心参数是 $\Theta = [\theta_1, \theta_2, \ldots, \theta_K]^\top \in \mathbb{R}^K$, 此外还希望通过 grid search 等方法调优超参数.

不过同时指出, $\|\Phi_\Theta\|_0 \leq C$ 似乎是一个 NP-hard 的问题, 可能的解决方法包括: 转化为 $\lambda \|\Phi_\Theta\|_1 + \mu \|\Phi_\Theta\|_2$ 等优化形式 (或者进行一些等价的松弛), 或者比如进行例如贪心等启发式的搜索方法.

目前有几种想到的训练思路 (各自直接也不一定是互斥的, 或许也可以相互结合借鉴)

**目前似乎最直接的一个办法是类似于在 预训练的时候做的, 进行词语接龙的训练. 我们准备一些 supervised 具体数据, 尝试去 align 这个问题. 就是当作一个简单的以vocabulary 进行分类的分类问题即可. **

****

_**SFT**_

例如 OpenAI WebGPT 等, 进行一般的指令微调 (SFT, Supervised Fine-Tuning), 通过最小化交叉熵损失函数进行训练. 不过这个似乎会有一些问题. 感觉不是很推荐.

_**RLHF**_

这个是最符合的训练思路, 例如 OpenAI InstructGPT ranking 等进行训练.

_**监督学习 v.s. 元学习**_

后期还可以做多任务适配, 进行类似于 MAML 的元学习过程.

_**certainty 等损失**_

还可以考虑引入一些额外的损失函数, 例如 entropy 等作为损失函数.

****
