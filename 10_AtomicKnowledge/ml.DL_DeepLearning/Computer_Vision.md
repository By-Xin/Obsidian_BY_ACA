# DL Applications: Computer Vision

> Ref: https://d2l.ai/chapter_computer-vision/index.html

在介绍完一系列的神经网络的基础架构后, 本节讲介绍其在计算机视觉中的应用.

## Image Augmentation

大型数据集是成功应用深度神经网络的先决条件. Image augmentation (图像增广) 是一种数据增强技术, 其在对训练图像进行一系列的随机变化之后, 生成相似但不同的训练样本, 从而扩大了训练集的规模. 此外, 我们还可以通过随机改变训练样本可以减少模型对某些属性的依赖, 从而提高模型的泛化能力. 例如, 我们可以以不同的方式裁剪图像, 调整亮度, 添加噪声等, 使得模型不依赖于图像的某些特征. 可以说, 图像增广技术对于AlexNet的成功是必不可少的.

常见的图像增广方法包括:
- Flipping (翻转): 水平翻转, 垂直翻转
- Cropping (裁剪): 截取图像的一部分, 并且可以决定是否缩放
- Color (颜色): 改变图像的亮度, 对比度, 饱和度等

## Fine-tuning

Fine-tuning (微调) 是迁移学习的一种方法, 其通过在一个(通常是较大的) 数据集上训练一个模型得到一个具有良好性能的模型, 然后在一个(通常是较小的) 数据集上对该模型进行微调使其适应新的数据集. 一般而言, 微调的步骤如下:
- 在一个大型数据集 (Source Dataset) 上训练一个源模型 (Source Model / Pre-trained Model)
- 将 Source Model 的除了最后输出层之外的所有层的架构设计和参数都复制到一个新的模型 (Target Model)
- 添加一个新的输出层, 已满足当前任务的要求 (如分类任务的类别数)
- 在新的数据集 (Target Dataset) 上训练 Target Model, 使其适应新的数据集. 这时新的输出层将会从头开始训练, 而其他层的参数将会被微调.  

![微调的一般步骤](https://d2l.ai/_images/finetune.svg)

## Object Detection

之前常来用作例子的各种图像任务其本质上属于 Image Classification (图像分类) 的任务, 也就是给定一张图像(通常只包含一个主体), 让模型判断出图像中包含的物体的类别. 

然而, 很多时候图像里有多个我们感兴趣的目标, 我们不仅想知道它们的类别, 还想得到它们在图像中的具体位置. 在计算机视觉里, 我们将这类任务称为目标检测 (object detection) / 目标识别 (object recognition). 

### Bounding Box

具体来说 Object Detection 需要我们解决两个问题:
- 目标定位 (object localization): 识别出图像中所有的目标, 并且给出它们在图像中的位置. 通常我们会给出一个边界框 (bounding box) 来表示目标在图像中的位置. 边界框通常用一个矩形来表示, 数学上有两种等价的表示方法:
    - 用矩形的左上角和右下角的坐标来表示  $(x_1, y_1, x_2, y_2)$
    - 用矩形的中心点的坐标和宽高来表示 $(x_c, y_c, w, h)$
- 目标分类 (object classification): 识别出图像中所有的目标, 并且给出它们的类别.

### Anchor Box

为了解决目标定位的问题, 我们需要给出一个边界框来表示目标在图像中的位置. 但是, 由于目标的大小和形状各不相同, 我们很难用一个固定的边界框来表示所有的目标. 为了解决这个问题, 我们可以使用 Anchor Box (锚框) 的方法.

Anchor Box 是一种预定义的边界框, 我们会以图像的每个像素为中心, 生成一系列不同长宽比和大小的边界框, 这些边界框就是 Anchor Box. 在训练时, 我们会依次检验每个 Anchor Box 中是否有我们感兴趣的目标. 

假设目标图像的大小为 $H \times W$. 我们预先定义缩放比为 $s_1, s_2, \ldots, s_n$ , 以及长宽比为 $r_1, r_2, \ldots, r_m$ 的共有 $n \times m$ 个 Anchor Box. 若对于每个像素点都生成这样 $mn$ 个 Anchor Box, 则总共会生成 $HWmn$ 个 Anchor Box. 

这有时会导致计算量过大. 在实践中, 往往只保留包含 $r_1$ 或者 $s_1$ 的 Anchor Box, 即对于每个像素点只生成 $m + n - 1$ 个 Anchor Box. 因此, 我们一般会总计生成 $HW(m + n - 1)$ 个 Anchor Box 作为下一步的输入.

![例: 在像素点(250, 250)处生成的所有 Anchor Box](https://zh.d2l.ai/_images/output_anchor_f592d1_63_0.svg)

> 在复杂的 CV 问题中, 我们像在基本的 classification 问题中把一个图片整体看成一个输入然后对每个像素点进行操作的方法是不可行的, 因为图片中包含的信息是非常复杂的, 我们不能简单地把一个图片看成一个输入. 这时我们需要对图片进行分割, 把图片分成多个小块, 然后对每个小块进行操作. 这里的 Anchor Box 起到了类似于 NLP 中的 Token 的作用, 其会成为我们进行后续训练的基本单元.


### Labeling the Anchor Boxes by Intersection over Union (IoU)

Spoiler, Object Detection 最终的目的是对图像进行 Localization 和 Classification 的训练. 而训练的对象 (基本单位就是) 这些 Anchor Box. 因此我们需要对其进行标记:
- 正样本 (positive sample): Anchor Box 与 Ground Truth Box (人工标注的包含目标的真实边界框) 得到了匹配
- 负样本 (negative sample): Anchor Box 没有匹配到任何 Ground Truth Box, 属于背景
- 忽略样本 (ignore sample): Anchor Box 有些模糊, 在训练时不考虑

因此我们首先需要定义 Anchor Box 和 Ground Truth Box 之间的匹配关系. 这时我们可以使用 Intersection over Union (IoU) 来判断.  IoU 的值在 $[0, 1]$ 之间, $\text{IoU} = 1$ 表示两个边界框完全重合, $\text{IoU} = 0$ 表示两个边界框没有交集.

![IoU 的计算](https://d2l.ai/_images/iou.svg)


进而对于一个图片, 我们预先定义了 $m$ 个 Anchor Box: $A_1, A_2, \ldots, A_{m}$, 以及 $n$ 个 Ground Truth Box: $G_1, G_2, \ldots, G_{n}$. 一般而言, $m \gg n$. 我们也可以把这样的对应关系记为一个 IoU 矩阵 $\mathbf{X} \in \mathbb{R}^{m \times n}$, 其中 $X_{ij} = \text{IoU}(A_i, G_j)$, 即第 $i$ 个 Anchor Box 和第 $j$ 个 Ground Truth Box 之间的重合度.

![IoU 矩阵](https://d2l.ai/_images/anchor-label.svg)

***Algorithm: Assigning Anchor Boxes to Ground Truth Boxes***

某种意义上, 上述的这个 IoU 矩阵就可以作为训练的数据了. 但是更直觉的做法是对该矩阵进行离散化处理, 给每个 Ground Truth Box 分配一个最优 (最大 IoU) 的 Anchor Box. 
- 因此我们希望能够通过这个匹配算法 $\mathcal{A}$ 来得到一组匹配关系组合: $\mathcal{M} \subseteq \{(A_i, G_j) | i = 1, 2, \ldots, m; j = 1, 2, \ldots, n\}$, 使得 $(A_i, G_j) \in \mathcal{M}$ 表示第 $i$ 个 Anchor Box是第 $j$ 个 Ground Truth Box 最优的匹配. 
- 在算法中, 由于 Anchor Box 的数量远大于 Ground Truth Box 的数量, 我们首先必须保证所有的 Ground Truth Box 都至少有一个最优 Anchor Box 和其匹配.
- 但在此基础上, 还会有很多的 Anchor Box 剩余没有得到匹配, 其可能也较好的包含了 Ground Truth (只是不是最优的). 我们可以考虑通过设置一个 IoU 的阈值 $\tau$ 使得足够好的 Anchor Box 也能得到匹配. 这样做会增加每个 Ground Truth Box 的匹配 Anchor Box 的数量, 使得模型的训练更加稳定 (否则只有一个正样本, 可能会导致数据过于稀疏, 训练不稳定).

因此上述样本的匹配算法可以描述为:

- 输入: IoU 矩阵 $\mathbf{X} \in \mathbb{R}^{m \times n}$, 以及阈值 $\tau$
- 输出: 匹配关系 $\mathcal{A}(\mathbf{X}, \tau) \rightarrow \mathcal{M} \subseteq \{(A_i, G_j) | i = 1, 2, \ldots, m; j = 1, 2, \ldots, n\}$

- **STEP 1**: 首次 IoU 匹配
  - 找出当前 IoU 矩阵 $\mathbf{X}$ 中的全局最大值 $X_{ij} = \max_{i,j} \mathbf{X}$
  - 将其对应的 Anchor Box $A_i$ 和 Ground Truth Box $G_j$ 进行匹配, 即 $(A_i, G_j) \in \mathcal{M}$
  - 丢弃该行和该列 (例如, 该行和该列的值都置为 $0$), 记为新的 IoU 矩阵 $\mathbf{X}'$
- **STEP 2**: 迭代 IoU 匹配 $n - 1$ 次
    - 在新的 IoU 矩阵 $\mathbf{X}'$ 中, 重复上述步骤, 找到新的全局最大值 $X_{ij} = \max_{i,j} \mathbf{X}'$
    - 分配 $(A_i, G_j) \in \mathcal{M}$, 并丢弃该行和该列
    - 重复上述过程直到所有的 Ground Truth Box 都唯一匹配了 Anchor Box
- **STEP 3**: 处理剩余的 Anchor Box
    - 对于剩余的 $m - n$ 个 Anchor Box, 逐行扫描 IoU 矩阵 $\mathbf{X}$ (即逐个 Anchor Box 的 IoU 值), 若该行的最大值大于阈值 $\tau$, 则将该行的 Anchor Box 和对应的 Ground Truth Box 进行匹配.

总的而言, 该算法最少会分配 $n$ 个 Anchor Box, 即每个 Ground Truth Box 1-1 匹配一个 Anchor Box. 最多会分配 $m$ 个 Anchor Box, 即每个 Anchor Box 都匹配到了一个 Ground Truth Box. 

***Labeling the Anchor Boxes***

在完成上述匹配后, 我们就可以对每个 Anchor Box 进行标记了. 根据上述所说, 我们会有三类标记:
- **正样本 (positive sample)**: Anchor Box 与 Ground Truth Box 得到了匹配. 这样的数据会的类别会与对应的 Ground Truth Box 一致. 并且将参与到分类和回归(位置)的训练中.
- **负样本 (negative sample)**: Anchor Box 没有匹配到任何 Ground Truth Box, 属于背景. 这样的数据会被标记为一个新的类别(背景), 并且将参与到分类的训练中. 其位置并不会参与到回归的训练中.
- **忽略样本 (ignore sample)**: Anchor Box 有些模糊, 在训练时不考虑. 这样的数据不会参与到分类和回归的训练中.

---

具体看一个例子. 如图所示, 我们有两个 Ground Truth Box (黑色的 dog 和 cat), 以及 $5$ 个 Anchor Box. 

![一个 Anchor Box 和 Ground Truth Box 的例子](https://zh.d2l.ai/_images/output_anchor_f592d1_123_0.svg)

我们可以计算出如下的 IoU 矩阵:
$$
\mathbf{X} = \begin{matrix}
\text{Dog} ~~~~\text{Cat} \\
\begin{bmatrix}
0.05 & 0.02 \\
0.14 & 0.00 \\
0.00 & 0.57 \\
0.00 & 0.21 \\
0.00 & 0.75 \\
\end{bmatrix}
\end{matrix}
$$

首先根据上述的算法, 我们可以得到如下的匹配关系:
- 全局最大值 $X_{ij} = 0.75$, 对应的关系为 $(A_4, G_{\text{Cat}})$
- 丢弃该行和该列, 得到新的 IoU 矩阵. 余下的矩阵最大值为 $X_{ij} = 0.14$, 对应的关系为 $(A_1, G_{\text{Dog}})$
- 规定 $\tau = 0.5$, 因此对于$\max A_0 = 0.05, \max A_3=0.21$, 都小于 $\tau$, 因此不进行匹配, 标记为背景. 对于 $\max A_2 = 0.57$, 大于 $\tau$, 因此进行匹配, 即 $(A_2, G_{\text{Cat}})$

***Regression for the Anchor Boxes***

在完成上述的 Anchor Box 的标记后, 我们就可以进行训练了. 对于位置的训练, 我们首先要计算出 Anchor Box 和 Ground Truth Box 之间的偏差 (offset). 假设 Anchor Box 的中心点坐标为 $(x_a, y_a)$, 宽高为 $(w_a, h_a)$, Ground Truth Box 的中心点坐标为 $(x_g, y_g)$, 宽高为 $(w_g, h_g)$. 则偏差可以表示为:
$$
\text{Offset} = \left( \frac{x_g - x_a}{\sigma_x w_a}, \frac{y_g - y_a}{\sigma_y h_a}, \frac{1}{\sigma_w} \log\frac{w_g}{w_a}, \frac{1}{\sigma_h} \log\frac{h_g}{h_a} \right)
$$
- 其中 $\sigma_x = \sigma_y = 0.1$, $\sigma_w = \sigma_h = 0.2$ 是超参数, 用于缩放偏差的值.

### Model Training and Inference by Non-Maximum Suppression (NMS)

在完成了 Anchor Box 的正负样本标记后, 我们就可以进行训练了. 这里不关心模型的具体架构, 只关心训练的过程. 假设我们已经得到了$N$ 个 Anchor Box, 每个 Anchor Box 都包含如下的信息:
- 自身位置 $(x_a, y_a, w_a, h_a)$
- 类别 $c_a$
- offset $\mathbf{t}_a = (t_{x_a}, t_{y_a}, t_{w_a}, t_{h_a})$
- 正负样本标记 mask $m_a \in \{0, 1\}$

***Head Network***

在训练时, 我们会对每个 Anchor Box 产生两个分支:
- Classification Head: 该分支用于对 Anchor Box 进行分类, 输出 $\hat p_a \in [0, 1]^{C}$, 其中 $C$ 是类别数. 表示 Anchor Box 属于每个类别的概率. 该分支的损失函数常用交叉熵损失函数 (cross-entropy loss), 记为 $\mathcal{L}_{\text{cls}}$.
- Regression Head: 该分支用于对 Anchor Box 中的正样本进行回归, 输出 $\hat t_a \in \mathbb{R}^{4}$, 其中 $4$ 是偏差的维度. 该分支的损失函数常用 Smooth L1 损失函数 / GIoU 等, 记为 $\mathcal{L}_{\text{reg}}$.
我们可以将这两个分支的损失函数结合起来, 得到最终的损失函数:
$$
\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{reg}}$$

此时我们便可以对模型进行训练了. 


***Inference by Non-Maximum Suppression (NMS)***

在训练完成后, 我们就可以对模型进行推理了. 对于输入的图像, 我们同样会得到 $N$ 个 Anchor Box. 将这些 Anchor Box 传入训练好的模型中, 我们会得到如下的信息:
- 分类概率 $\hat p_a \in [0, 1]^{C}$, 其中 $C$ 是类别数. 表示 Anchor Box 属于每个类别的概率.
- 偏差 $\hat t_a = (\hat t_{x_a}, \hat t_{y_a}, \hat t_{w_a}, \hat t_{h_a}) \in \mathbb{R}^{4}$, 表示 Anchor Box 的偏差.

对应传入的 Anchor Box 的自身信息 $(x_a, y_a, w_a, h_a)$, 我们可以进行 offset decoding, 得到修正后的 Anchor Box 的位置和类别, 即 Predicted Box:
$$\begin{aligned}
\hat x_{\text{pred}} &= x_a + \hat t_{x_a} \cdot \sigma_x w_a \\
\hat y_{\text{pred}} &=  y_a + \hat t_{y_a} \cdot \sigma_y h_a \\
\hat w_{\text{pred}} &=  w_a \cdot \exp(\hat t_{w_a}) \\
\hat h_{\text{pred}} &=  h_a \cdot \exp(\hat t_{h_a})
\end{aligned}$$
以及对应的类别 $\hat c_{\text{pred}} = \arg\max \hat p_a$ (有时若 $\hat p_a$ 的值小于某个阈值, 则该 Anchor Box 不属于任何类别) 及其置信度 $\hat p_{\text{pred}} = \max \hat p_a$.

***NMS***

在得到 Predicted Box 后, 我们需要对其进行后处理. 在实践中, 往往会得到很多重叠的 Predicted Box, 这时我们需要对其进行非极大值抑制 (Non-Maximum Suppression, NMS) 来去除冗余的 Predicted Box.

对于每个类别 $c$:
- 提取出所有的 Predicted Box 中, 属于该类别 $c$ 的 Predicted Box, 记为 $\mathcal B_c$
- 选择出 $\mathcal B_c$ 中置信度最高的 Predicted Box, 记为 $B_{\text{max}}$
- 计算 $B_{\text{max}}$ 和 $\mathcal B_c$ 中其他 Predicted Box 的 IoU 值, 记为 $\text{IoU}_{B_i \in \mathcal{B_c}\backslash B_{\text{max}}}(B_{\text{max}},B_i)$
- 将这些 IoU 值大于某个阈值 $\tau$ 的 Predicted Box 全部删除; 将 $B_{\text{max}}$ 加入到最终输出的 Predicted Box 列表中.
- 将剩余的 Predicted Box 记为 $\mathcal B_c'$ 重复上述步骤.

NMS 的作用相当于对于每个类别 $c$, 对于每个局部地区只保留置信度最高的 Predicted Box, 去掉所有在其附近的置信度较低的 Predicted Box. 

![NMS 之前 dog 有三个较为临近的 prediction box](https://d2l.ai/_images/output_anchor_f592d1_174_0.svg)

![NMS 之后只保留这个区域里置信度最高的 prediction box](https://d2l.ai/_images/output_anchor_f592d1_192_0.svg)


### Multiscale Object Detection

## Semantic Segmentation

