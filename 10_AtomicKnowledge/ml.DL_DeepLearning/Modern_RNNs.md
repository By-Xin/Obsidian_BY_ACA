#DeepLearning 
# Modern RNNs

回顾, 对于 Sequential 数据, 我们会维护一个隐藏状态 $H_t$ 来存储过去的信息, 通过
$$
H_t = \phi(H_{t-1}W_{hh} + X_tW_{xh} + b_h)
$$
的形式来更新隐藏状态.

## Long Short-Term Memory (LSTM) Networks

RNN 在长序列上, 由于其梯度会层层累加, 造成 gradient explosion 或 gradient vanishing 问题. 因此我们需要一些更复杂的结构来维护长时间的依赖关系. LSTM 通过引入 *Gate* 来控制信息的流动, 使得信息可以在 cell state 中长期存储, 有效的解决了 Vanishing gradient 问题.



### LSTM Cell

在 LSTM 中, 我们会维护一个 cell state $C_t$ 来对信息进行长期存储. 在其中, 我们会使用三个 *Gate* 来控制信息的流动:
- Input Gate $\mathrm{I}_t$: 控制当前输入 $X_t$ 进入 cell state 的程度.
- Forget Gate $\mathrm{F}_t$: 控制 cell state 中的信息被遗忘 (趋近于 0) 的程度.
- Output Gate $\mathrm{O}_t$: 控制 cell state 中的信息被输出的程度.

![](https://d2l.ai/_images/lstm-3.svg)

下给出具体的公式:

假设当前的 batch size 为 $n$, 输入维度为 $d$, 隐藏状态维度为 $h$. 假设输入为 $\mathbf{X}_t \in \mathbb{R}^{n\times d}$, 隐藏状态为 $\mathbf{H}_{t-1} \in \mathbb{R}^{n\times h}$. 则在 $t$ 时刻, LSTM 的计算过程为:
$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_tW_{xi} + \mathbf{H}_{t-1}W_{hi} + b_i) \quad \text{(input gate)} \\
\mathbf{F}_t &= \sigma(\mathbf{X}_tW_{xf} + \mathbf{H}_{t-1}W_{hf} + b_f) \quad \text{(forget gate)} \\
\mathbf{O}_t &= \sigma(\mathbf{X}_tW_{xo} + \mathbf{H}_{t-1}W_{ho} + b_o) \quad \text{(output gate)} \\
\mathbf{\tilde{C}}_t &= \tanh(\mathbf{X}_tW_{xc} + \mathbf{H}_{t-1}W_{hc} + b_c) \quad \text{(candidate cell state)} \\
\mathbf{C}_t &= \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \mathbf{\tilde{C}}_t \quad \text{(cell state)}
\end{aligned}
$$

- 其中, $\mathbf{I}_t, \mathbf{F}_t, \mathbf{O}_t \in (0,1)$, 类似一个 soft-switch, 控制信息的流动. 取 1 时表示完全通过, 取 0 时表示完全不通过.
- Internal state $\mathbf{C}_t$ 表示了 LSTM 的最核心更新规则: 我们会将历史的信息 $\mathbf{C}_{t-1}$ 进行一定程度的遗忘 $\mathbf{F}_t\odot \mathbf{C}_{t-1}$, 然后将当前输入的信息 $\mathbf{\tilde{C}}_t$ 进行一定程度的保留 $\mathbf{I}_t \odot \mathbf{\tilde{C}}_t$, 最后将两者相加.

最终这个 cell state $\mathbf{C}_t$ 会通过一个输出 gate $\mathbf{O}_t$ 来控制输出:
$$
\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t)
$$

在 PyTorch 中, LSTM 已经被实现为一个内置的模块, 我们可以直接使用:
```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out
```

## Gated Recurrent Unit (GRU)

GRU 是一个更简化的 LSTM 变种, 它将 input gate 和 forget gate 合并成一个 update gate $\mathbf{U}_t$, 通过一个 reset gate $\mathbf{R}_t$ 来控制 cell state 的更新. 

在 $t$ 时刻, GRU 的计算过程为:
$$
\begin{aligned}
\mathbf{R}_t &= \sigma(\mathbf{X}_tW_{xr} + \mathbf{H}_{t-1}W_{hr} + b_r) \quad \text{(reset gate)} \\
\mathbf{Z}_t &= \sigma(\mathbf{X}_tW_{xz} + \mathbf{H}_{t-1}W_{hz} + b_z) \quad \text{(update gate)} \\
\bar{\mathbf{H}}_t &= \tanh(\mathbf{X}_tW_{xh} + \mathbf{R}_t \odot (\mathbf{H}_{t-1}W_{hh}) + b_h) \quad \text{(candidate hidden state)} \\
\end{aligned}
$$
- 当 Reset gate $\mathbf{R}_t$ 趋近于 1 时, $(\mathbf{R}_t \odot \mathbf{H}_{t-1}) \approx \mathbf{H}_{t-1}$, 使得当前的 hidden state 退化为: $\bar{\mathbf{H}}_t = \tanh(\mathbf{X}_tW_{xh} + \mathbf{H}_{t-1}W_{hh} + b_h)$, 也就是 GRU 退化为一个普通的 RNN.
- 当 Reset gate $\mathbf{R}_t$ 趋近于 0 时, 当前的 hidden state 会被重置为 0, 使得模型能够忘记之前的信息, 模型退化为 $\bar{\mathbf{H}}_t \approx \tanh(\mathbf{X}_tW_{xh} + b_h)$, 即一个普通的 MLP.
  
最终我们通过 update gate $\mathbf{Z}_t$ 来控制当前的 hidden state 的最终输出:
$$
\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1} + (1 - \mathbf{Z}_t) \odot \bar{\mathbf{H}}_t
$$
- 当 $\mathbf{Z}_t$ 趋近于 1 时, $\mathbf{H}_t \approx \mathbf{H}_{t-1}$, 使得当前的 hidden state 保持不变.
- 当 $\mathbf{Z}_t$ 趋近于 0 时, $\mathbf{H}_t \approx \bar{\mathbf{H}}_t$, 使得当前的 hidden state 被重置为 $\bar{\mathbf{H}}_t$.


![GRU 的计算图](https://d2l.ai/_images/gru-3.svg)

在 PyTorch 中, GRU 已经被实现为一个内置的模块, 我们可以直接使用:
```python
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, x):
        out, _ = self.gru(x)
        return out
```


***LSTM v.s. GRU***
- GRU 的模型设计更为简洁, 一般而言其训练速度更快
- 然而由于没有 LSTM 中的 internal state, 在一些具有非常强的长时间依赖关系的任务上, GRU 的性能可能会稍差一些.



## Deep RNNs

这里的 Deep RNNs 指的是在 RNN 的序列基础上, 我们把每一步的处理模块也换成一个更深层的处理网络. 一个一般的具有 $L$ 层的长度为 $T$ 的 RNN 的结构图如下:

![Deep RNNs 示意图](https://d2l.ai/_images/deep-rnn.svg)

可以看到, 这里的每一个 Hidden State (e.g. $H_3^{(L)}$) 都有两个输入: 同一时间步的上一层输入 (e.g. $H_3^{(L-1)}$) 和上一时间步的当前层输入 (e.g. $H_2^{(L)}$). 这样就形成了一个深度的 RNN 结构. 因此可以抽象为如下数学公式:
$$
\begin{aligned}
\mathbf{H}_t^{(l)} &= \phi_l (\mathbf{H}_{t}^{(l-1)}W_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)}W_{hh}^{(l)} + b_h^{(l)}) \quad (l=1,2,\ldots,L) \\
\mathbf{O}_t &= \mathbf{H}_t^{(L)}\mathbf{W}_{hq} + \mathbf{b}_q
\end{aligned}
$$
- 其中 $\mathbf{W}_{xh}^{(l)}, \mathbf{W}_{hh}^{(l)} \in \mathbb{R}^{h\times h}$, $\mathbf{b}_h^{(l)} \in \mathbb{R}^{h}$, $\mathbf{W}_{hq} \in \mathbb{R}^{h\times o}$, $\mathbf{b}_q \in \mathbb{R}^{o}$.
- 一般而言 $h\in(64,2056)$, $L\in(1,8)$.

## Bi-directional RNNs

Bi-directional RNNs 是在 RNN 的基础上, 将输入序列同时从前向和后向进行处理. 这样可以使得模型在每一个时间步都能获得前后文的信息. 这尤其在一些 NLP 任务中 (例如 Speech Detection, Missing Token Prediction 等) 非常有用, 我们常常需要前后文的信息来进行判断.


例如:
- *I am _____*.
- *I am _____ hungry*.
- *I am _____ hungry and I can eat half a cow*.


![Bi-directional RNNs](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250429125620.png)

具体而言, 我们会训练两个方向的 RNN:
$$
\begin{aligned}
\overset{\rightarrow}{\mathbf{H}}_t &= \phi(\overset{\rightarrow}{\mathbf{H}}_{t-1}W^{\text{forward}}_{hh} + \mathbf{X}_tW^{\text{forward}}_{xh} + b_h^{\text{forward}}) \\
\overset{\leftarrow}{\mathbf{H}}_t &= \phi(\overset{\leftarrow}{\mathbf{H}}_{t+1}W^{\text{backward}}_{hh} + \mathbf{X}_tW^{\text{backward}}_{xh} + b_h^{\text{backward}})
\end{aligned}
$$

其中注意, 这里的 $\overset{\leftarrow}{\mathbf{H}}_t, \overset{\rightarrow}{\mathbf{H}}_t$ 是两个不同的 hidden state, 它们的维度是相同的. 但是它们的权重矩阵和偏置项是不同的. 各自相当于是两个独立的 RNN. 训练完成后, 我们会将两个 hidden state 进行整合输出:
$$
\mathbf{H}_t = \begin{bmatrix}
\overset{\rightarrow}{\mathbf{H}}_t & \overset{\leftarrow}{\mathbf{H}}_t
\end{bmatrix}\\
\mathbf{O}_t = \mathbf{H}_t\mathbf{W}_{hq} + \mathbf{b}_q
$$

## Machine Translation
- 机器翻译是一个典型的 Seq2Seq 任务, 其发展和 RNN 密切相关. 翻译中, 我们有一个 Source Language 和一个 Target Language, 两个 sequence 之间的长度, 以及每个 token 的位置等信息都是不一样的. 这样的问题也称为 Unaligned Sequence Problem.

- 这里以 `Tatoeba` 数据集为例, 其数据集包含了多种语言的翻译对. 例如 English-French:
    ```
    # Go. Va!
    # Hi. Salut !
    # Run. Cous!
    # Run! Courez !
    ```

### Data Preprocessing

首先我们将数据集进行 tokenization:
```python
src, tgt = data._tokenize(text)
# [['go', '.', '<eos>'], ...]
# [['va', '!', '<eos>'], ...]
```
- 这里 `<eos>` 是一个特殊的 token, 用来表示 end of sequence.
- 为了进一步简化计算, 将稀有的 token 进行合并统一记为 `<unk>`.

### Loading Sequence Data of Same Length

我们希望把每个 sequence 的长度都统一为 $T$, 这样可以方便我们进行 batch 训练. 这里我们可以使用 `pad_sequence` 来实现. 具体而言其会通过 `<pad>` token 来填充每个 sequence, 使得每个 sequence 的长度都为 $T$.

此外, 在针对 target sequence 的时候, 我们会在每个 sequence 前面加一个 `<bos>` token 来表示开始, 并且标记了 target.

### Machine Translation 的 Encoder-Decoder 结构

对于机器翻译, 我们会使用一个 Encoder-Decoder 的结构. 其结构如下:

![Machine Translation 的 Encoder-Decoder 结构](https://d2l.ai/_images/seq2seq.svg)

- Encoder: 将输入的 source sequence 进行编码, 得到一个 hidden state $\mathbf{H}_T$.
- Decoder: 将 hidden state $\mathbf{H}_T$ 进行解码, 这个 hidden state 会被传入到 decoder 的每一个时间步中, 并联合 Target sequence 的上一个 token 作为输入, 进行解码预测. 模型在看到 `<bos>` token 后, 会开始进行预测, 直到预测到 `<eos>` token.

*Teacher Forcing:* Teacher forcing 会使用真实的 target token 作为输入 (当然是前移一个位置), 而不是使用模型预测的 token 作为下一个输入. 这样可以加快模型的收敛速度, 但是在测试时, 我们需要使用模型预测的 token 作为下一个输入.

#### Encoder

具体而言, 假设输入的序列为
$$
\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T]$$
使用如下的公式进行编码:
$$
\begin{aligned}
\mathbf{H}_t &= f(\mathbf{H}_{t-1}, \mathbf{x}_t) \\
\end{aligned}
$$

我们还会引入一个 `context vector` $\mathbf{c}_T$ 来表示当前的 hidden state (一般是最后一个步骤对应的):
$$
\begin{aligned}
\mathbf{c}_T &= \phi(\mathbf{H}_1, \mathbf{H}_2, \ldots, \mathbf{H}_T) \\
\end{aligned}
$$

Encoder 的训练权重有两个参数: 
- No. of rows: `vocab_size`, 表示输入的 token 的个数.
- No. of columns: `embed_size`, 表示每个 token 的 embedding 维度.

#### Decoder

在训练阶段, 我们希望 Decoder 能够和给定的 target sequence $\mathbf{Y} = [\mathbf{y}_1, \mathbf{y}_2, \ldots, \mathbf{y}_T']$ 进行对齐. 具体而言, Decoder 会输入三个变量: 上一步的 target token $\mathbf{y}_{t-1}$, 上一步的 hidden state $\mathbf{S}_{t-1}$, 以及 Encoder 的 context vector $\mathbf{c}_T$:
$$
\begin{aligned}
\mathbf{S}_t &= f(\mathbf{S}_{t-1}, \mathbf{y}_{t-1}, \mathbf{c}_T) 
\end{aligned}
$$
然后我们通过全连接层和 Softmax 层来进行预测:
$$
\mathbb{P}(\mathbf{y}_t | \mathbf{y}_{1},\cdots,\mathbf{y}_{t-1}, \mathbf{c}_T) 
$$

#### Prediction

在预测阶段, 我们只有输入序列, 并不知道 target sequence. 在预测时, 我们永远会从 `<bos>` token 开始进行解码，直到遇到 `<eos>` token 为止. 并且此时我们会用预测的 token 作为下一个输入. 

在理想状况中, 我们的目标是使得生成的 sequence 的联合概率分布最高:
$$
\prod_{t'=1}^{T'} \mathbb{P}(\mathbf{y}_{t'} | \mathbf{y}_{1},\cdots,\mathbf{y}_{t'-1}, \mathbf{c}_T)
$$

- Exhaustive Search: 想要枚举每一种可能进行计算是不现实的, 因为每一步我们都要从 $|\mathcal{V}|$ 个 token 中进行选择, 这会导致总的计算为 $\mathcal{O}(|\mathcal{V}|^{T'})$.

- Greedy Search: 在 Greedy Search 中, Decoder 会通过 Softmax 生成一个概率分布, 每一步会选择最高的概率对应的 token 作为输出, 直到遇到 `<eos>` token 为止. 
    $$
    y_{t'} = \argmax_{y\in\mathcal{Y}} \mathbb{P}(\mathbf{y} | \mathbf{y}_{1},\cdots,\mathbf{y}_{t'-1}, \mathbf{c}_T)
    $$
    - 然而 Greedy Search 并不保证能够收敛到全局最优. 
- Beam Search:
  - 在每个时间步, 选择出现概率最高的 $k$ 个 token 作为候选; 在生成的所有下一步的候选中, 再次选择$k$ 个 token 作为候选, 直到遇到 `<eos>` token 为止.
  - 在得到所有的候选后, 通过
    $$\frac{1}{L^\alpha} \log\mathbb{P}(\mathbf{y}_1,\cdots,\mathbf{y}_L| \mathbf{c}) = 
    \frac{1}{L^\alpha}\sum_{t'=1}^L\log\mathbb{P}(\mathbf{y}_{t'} | \mathbf{y}_{1},\cdots,\mathbf{y}_{t'-1}, \mathbf{c}_T) $$
  - Beam Search 的时间复杂度 $\mathcal{O}(k\mathcal{Y}T')$


#### BLEU

引入 BLEU (Bilingual Evaluation Understudy) 来评估翻译的质量 (此时假设我们是有标准答案的). BLEU 是一个基于 n-gram 的评估指标, 其计算方式如下:
$$
\text{BLEU} = \exp\left\{  \min(0,1-\frac{\text{len}_{\text{label}}}{\text{len}_{\text{prediction}}}) \right\}\prod_{n=1}^k p_n^{-2^n}
$$
- 其中 $p_n$ 是 n-gram 的精确度, 其计算方式如下:
    $$
    p_n = \frac{\text{number of matched n-grams in prediction and target}}{\text{number of n-grams in prediction}} \\
    $$
- $\text{len}_{\text{label}}$, $\text{len}_{\text{prediction}}$ 表示在 target 和prediction 中的 token 数量
- $k$ 是能够匹配到的最大 n-gram 的长度 (因为如果到 $k$-gram 时已经没有能够匹配到的模式了, 那么对于所有的 $n>k$ 的 n-gram 都是 0. 我们需要避免将0引入乘积中).


下面以 `A,B,C,D,E,F` 作为 target, `A,B,B,C,D` 作为 prediction 为例对 BLEU 的计算进行说明. 

- 首先计算 n-gram precision:
  - 1-gram $p_1$: 分母考察 prediction 中总的 1-gram 数量, 这里是5. 分子为依次遍历 target 的每个 1-gram, 看是否能够匹配到 prediction 中的 1-gram. 这里是 4 (因为 target 中只有一个 `B`, 因此对于 prediction 中的第二个 `B` 不会进行匹配). 因此 $p_1 = \frac{4}{5} = 0.8$.
  - 2-gram $p_2$: 分母考察 prediction 中总的 2-gram 数量, 这里为 `AB,BB,BC,CD` 共4 个. 分子为使用 target 的 `AB,BC,CD,DE,EF` 与 prediction 的 2-gram 进行匹配, 这里一共有 3 个匹配. 因此 $p_2 = \frac{3}{4} = 0.75$.
  - 3-gram $p_3$: Prediction 的 3-gram 为 `ABB,BBC,BCD`, 共 3 个.  Target 的 3-gram 为 `ABC,BCD,CDE,DEF`, 共 4 个. 这里 prediction 中的 `BCD` 和 target 中的 `BCD` 匹配, 因此 $p_3 = \frac{1}{3} = 0.33$.
  - 4-gram $p_4$: Prediction 的 4-gram 为 `ABBC,BBCD`, 共 2 个. Target 的 4-gram 为 `ABCD,BCDE,CDEF`, 共 3 个. 这里没有匹配, 因此 $p_4 = \frac{0}{2} = 0$.
- 然后计算长度惩罚项:
  - $\text{len}_{\text{label}} = 6$, $\text{len}_{\text{prediction}} = 5$, 因此 $1-\frac{\text{len}_{\text{label}}}{\text{len}_{\text{prediction}}} = 1-\frac{6}{5} = -0.2$.
- 此时$k=3$, 因为 $p_4=0$.

若 prediction 和 target 是完全相同的, 则所有 $p_n=1 ,\forall n$, 对应的 BLEU 为 1. 并且能够参与匹配的 n-gram 越长, 则 BLEU 越高. 前面的惩罚项是为了避免模型输出的长度过短, 使得 BLEU 过高.