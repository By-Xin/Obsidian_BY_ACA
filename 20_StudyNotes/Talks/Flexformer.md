# Flexformer: Flexible Linear Transformer with Learnable Attention Kernel

## Kernel Linear Transformer

本质上, softmax based 的 transformer
$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
其实就是在寻找相似性. 

那么这种相似性完全可以通过一个 kernel function 来实现. 也就是说, 可以找到一个 kernel function $\Phi$ 使得
$$
\text{Attention}(Q, K, V) = \Phi(Q)(\Phi(K)^TV)
$$

也就是 kernel 中的理论也可以迁移到这个当中. 


把样本当作分布去采样. 相当于学的是 迪拉克测度. 