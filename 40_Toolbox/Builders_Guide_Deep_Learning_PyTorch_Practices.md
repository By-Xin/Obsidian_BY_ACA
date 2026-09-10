#DeepLearning 
# Builder's Guide (PyTorch Practices)

本文将主要包括神经网络在 PyTorch 中的最基本的构建方法, 包括
- 模型构建
- 参数的设置与初始化
- 自定义模型结构
- 模型的保存与加载
- GPU 加速

## Layers and Modules

Layers 和 Modules 是神经网路编程的基本组成部分. 我们通过组合不同的 Layers 和 Modules 来构建我们的模型. 在 Python 中, 为了能够模块化地构建神经网络, 我们通常会用 class 来定义不同的 modules. 这样我们就可以通过实例化不同的 modules 来构建我们的模型.

之前已经出现过的一个非常简单的例子是通过 `nn.Sequential` 来直接组合预定义的 modules. 例如:
```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(
    nn.LazyLinear(256),
    nn.ReLU(),
    nn.LazyLinear(10)
)

X = torch.randn(2, 20) # 2 个样本, 每个样本 20 个特征

net(X).shape #  torch.Size([2, 10])
```
  - 这里我们使用了 `nn.Sequential` 来将三个 modules 组合在一起: 一个线性层 `nn.LazyLinear` (输入维度为 256), 一个 ReLU 激活函数, 和一个线性层 `nn.LazyLinear` (输出维度为 10). 这可以理解为一个两层的全连接神经网络.

  - `X` 的 shape 是 `[2, 20]`, 也就是 2 个样本, 每个样本 20 个特征. 第一个参数是 batch size, 第二个参数是特征数. `randn` 生成的是标准正态分布的随机数.
  - `net(X)` 的 shape 是 `[2, 10]`, 也就是 2 个样本, 每个样本 10 个输出. 在这个网络中, 输入特征 `20 -> 256 -> 10` 输出特征. 样本量没有变化.

更一般的, 我们希望能够自定义一个 module. 这将通过定义一个继承自 `nn.Module` 的 class 来实现. 例如上面的例子可以通过自定义 module 来实现:
```python
class MyNet(nn.Module):
    def __init__(self): 
        '''
        定义模型的结构, 包括模型的各个 layers 及其参数.
        '''
        super(MyNet, self).__init__() # 继承父类nn.Module的初始化方法
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    def forward(self, X): 
        '''
        定义模型的前向传播过程, 即输入如何经过模型的各个 layers 得到输出.
        '''
        return self.out(F.relu(self.hidden(X)))

net =  MyNet() # 实例化模型 (instantiation)
net(X).shape #  torch.Size([2, 10])
```
  - 这里我们定义了一个自定义的 module  `MyNet`, 它包含了两个线性层 `hidden` 和 `out`. 在 `forward` 方法中, 我们定义了模型的前向传播过程. 这个模型的结构是 `20 -> 256 -> 10`, 与上面的例子相同. 
  - 通过 `nn.` 的方法定义的 layers 都是继承自 `nn.Module` 的, 其暗含着这些模型对应的参数都会被自动注册到模型的参数列表中. 

有时候我们需要在模型中使用一些不需要训练的参数. 这些参数应当被当作常量, 不应该在计算梯度时被更新.  例如我们希望在模型中使用一个常量参数 `weight`, 这时一种做法便是不要再通过 `nn.` 的方法定义, 而是直接在 `__init__` 方法中调用 `torch` 的方法定义:
```python
class MyNet(nn.Module):
    def __init__(self): 
        super(MyNet, self).__init__()
        self.weight = torch.rand((20, 20)) # 直接定义一个常量参数
        self.linear = nn.LazyLinear(20)
    def forward(self, X): 
        X = self.linear(X)
        X = F.relu(X @ self.weight + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
net = MyNet()
type(net.linear.weight) # <class 'torch.nn.parameter.Parameter'>
type(net.linear.bias) # <class 'torch.nn.parameter.Parameter'>
type(net.weight) # <class 'torch.Tensor'>
```
- 这里我们定义了一个常量参数 `weight` (shape 为 `[20, 20]`), 一个线性层 `linear`. 在 `forward` 方法中, 我们使用了这个常量参数 `weight` 来进行计算.
- `F.relu(X @ self.weight + 1)` 中的 `@` 是矩阵乘法, `X @ self.weight` 的结果是一个矩阵, `F.relu` 是逐元素的 ReLU 激活函数.


## Parameter Management

### Parameter Initialization

如果没有特别说明, 当我们通过继承`nn.Module`的方式定义模型时, 模型的参数都会被自动进行初始化 (默认为 Kaiming Initialization / He Initialization, 即 `torch.nn.init.kaiming_normal_`). 

PyTorch 中的 `nn.init` 模块也提供了很多可以用来初始化参数的方法:
- `nn.init.normal_(module.weight, mean=0, std=0.01)` 用正态分布初始化参数
- `nn.init.constant_(module.bias, val=0)` 用常数初始化参数
  - 如果是 0, 可以直接用 `module.init.zeros_()`
- `nn.init.xavier_uniform_(module.weight)` 用 Xavier 初始化参数

更进一步, 我们可以自定义初始化方法. 例如, 我们希望参数 $w$ 满足如下分布:
$$
w \sim \begin{cases}
\mathcal{U}(5,10) \quad \quad ~~\text{with prob.} ~ 0.25 \\
\mathcal{U}(-10,-5) \quad \text{with prob.} ~ 0.25 \\
0 \quad\quad\quad\quad\quad~~ \text{with prob.}  0.5
\end{cases}
$$

我们可以通过自定义一个初始化方法来实现:
```python
def my_init(module):
    nn.init.uniform_(module.weight, -10, 10)
    module.weight.data *= (module.weight.data.abs() >= 5)
net.apply(my_init)
```


### LazyLinear and Lazy Initialization

在刚刚的模型构建中, 我们多次调用了 `nn.LazyLinear` 来定义线性层. 在这个定义中, 我们并没有指定这个线性层的输入维度, 只指定了输出维度. 这是因为 `nn.LazyLinear` 具有 **Lazy Initialization** 的特性. 在定义模型时, 程序并不会具体定义其参数 (因此此时若通过代码查询参数维度, 其返回结果为 `<UninitializedParameter>`), 而只有在第一次调用时才会根据传入数据的维度来初始化参数 (这时若通过代码查询参数维度则会根据传入数据的维度返回具体的参数维度).

## Custom Layers and Layers with Parameters

有时候我们需要自定义一些 layers. 这时我们一般会在自定义的类中的 `def __init__(self):` 方法中定义这个 layer 的相关参数, 并在 `def forward(self, X):` 方法中定义这个 layer 的前向传播过程. 例如, 我们希望定义一个自定义的线性层 `MyLinear`:
```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    def forward(self, X):
        linear = X @ self.weight + self.bias
        return F.relu(linear)

net = MyLinear(20, 256)   
```
- 注意, 只有通过 `nn.Parameter` 定义的参数才会被自动注册到模型的参数列表中, 当作是需要训练的参数.
- 在实例化 `MyLinear` 时, 我们需要对照 `def __init__(self, in_features, out_features):` 方法中的参数来传入所有未被定义的参数.

## File I/O

模型的保存与加载是神经网络编程中的重要部分. 

- 对于单个 tensor: 
  - `torch.save(tensor, 'tensor.pt')` 保存 tensor
  - `torch.load('tensor.pt')` 加载 tensor
- 对于模型, 我们没有保存这个模型的全部结构, 而是保存了模型的参数. 因此当我们需要加载模型时, 我们需要先重新定义模型的结构, 然后再加载已经保存好的参数. 例如:
  ```python
    torch.save(net.state_dict(), 'net.pt') # 保存模型参数
    clone = MyNet() # 重新定义模型结构
    clone.load_state_dict(torch.load('net.pt')) # 加载模型参数
  ```

> `net.eval()` : 补充一个小知识点. 在定义模型时, 我们往往会涉及一些例如 `Dropout` 或者 `BatchNorm` 这样的层. 这些层在训练和测试时的行为是不同的. 例如在测试模式中, 我们希望用现在已经优化好的参数来进行预测, 而不希望再进行 Dropout. 因此我们需要在测试时将模型设置为测试模式, 即 `net.eval()`. 例如我们已经保存了一个优化好的模型, 我们可以通过以下方式来加载模型并利用模型进行预测:
> ```python
> clone = MyNet()
> clone.load_state_dict(torch.load('net.pt'))
> clone.eval() # 设置为测试模式
> Y_clone = clone(X)
> ```


## GPU Acceleration

在 PyTorch 中, 每一个 tensor 都有一个 `device` 属性, 用来表示这个 tensor 对应的设备.  默认情况下, tensor 都是在 CPU 上的. 例如:
```python
X = torch.randn(2, 20)
X.device # cpu
```

我们可以通过下面的方法查看当前设备:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
- 如果当前设备是 GPU, 则 `device` 为 `cuda`, 否则为 `cpu`.

还可以查看支持的GPU数量:
```python
torch.cuda.device_count()
```

如果我们希望将 tensor 和模型移动到 GPU 上, 可以通过 `to` 方法:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 检查当前设备
X = X.to(device)
net = net.to(device)
```
- 这样我们就将 tensor `X` 和模型 `net` 移动到了 GPU 上.
  
**注意:**
- 在实践中, 一定要确保每一个 tensor 和模型都在同一个设备上. 否则会报错.
- 并不是 GPU 就一定比 CPU 快. 因为我们还需要考虑到数据的传输成本. 因此往往只有在大规模的并行计算时, GPU 才会比 CPU 快.
