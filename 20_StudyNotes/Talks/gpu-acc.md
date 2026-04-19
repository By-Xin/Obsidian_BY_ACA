# Talk Notes: GPU-Accelerated Solver for Entropic-Regularized Optimal Transport

**Speaker: QIU, Yixuan (SUFE)**

**Software:**

- RegOT: github.com/yixuan/regot-python
- RegOT-CUDA: github.com/yixuan/regot-cuda

**Reference of RegOT:**

RegOT 是一个基于 SPLR 算法的一个工作, 而 cuRegOT 是近期基于 RegOT 的一个 GPU 加速的版本. 后者的有关内容暂时还没有正式发表, 但是前者的内容已经在 ICML 2025 发表了. 相关的论文引用如下:

```
@inproceedings{tang2024safe,
  title={Safe and sparse Newton method for entropic-regularized optimal transport},
  author={Tang, Zihao and Qiu, Yixuan},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  pages={129914--129943},
  year={2024}
}
```

```
@inproceedings{wang2025sparse,
  title={The Sparse-Plus-Low-Rank quasi-Newton method for entropic-regularized optimal transport},
  author={Wang, Chenrui and Qiu, Yixuan},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```

> - 会议录音ai 总结: https://drive.google.com/file/d/1h9SRUfNVnrtpBAx4qrrpwVsUxZDM8E_u/view?usp=drive_link
>
> - 会议录音转写: https://drive.google.com/file/d/1NPEcxo491tbiC7V7ik_e8IQvzYgTBoai/view?usp=drive_link

## Introduction to Optimal Transport (OT) & Computational Challenges

### What is OT?

- 经典教材: Optimal Transport: Old and New

- 对于两个概率分布 $P(x)$ 和 $Q(y)$，OT 旨在找到一个映射 $T$，使得将 $P$ 通过 $T$ 转换成 $Q$ 的代价最小化. (earth moving problem)
- 应用:
  - 定义 Wasserstein 距离: 衡量分布之间的距离
  - Generative modeling: Wasserstein GANs
  - Domain Adaptation: 当训练分布和测试分布不同时, 如何进行有效转换. 
  - NLP: word alignment
  - LLM: MHC (本质上其中一个部分也是在讨论 OT 问题)

### Computational Challenges: OT is HARD to Compute

对于一个典型的离散 OT 问题:
$$
\min_{P \in \Pi(a, b)} \langle M, P\rangle, \quad s.t. ~ \Pi(a, b) = \{P \in \mathbb{R}^{n \times m}_+: P \mathbf{1}_m = a, P^\top \mathbf{1}_n = b\}
$$
- 计算复杂度 $\mathcal{O}(n^3 \log n)$

传统解决方法:
$$
\min_{P \in \Pi(a, b)} \langle M, P\rangle - \epsilon H(P)
$$
- 相当于对 LP 问题进行 Smoothing 变成 strongly convex 等. 最终复杂度约为 $\mathcal{O}(n^2)$, 但其实其中还耦合了一些平滑因子, 以及本身收敛所需要的迭代次数 (或许收敛更慢). 

-  Sinkhorn 算法: 对于对偶问题进行分块下降等. 
   -  其优势: 基本只依赖于 dense matrix-vector multiplication, 非常适合 GPU 加速. 
   -  其劣势: 单次计算复杂度不高, 但是总体收敛迭代次数耦合在其中, 总体的计算效率不高. 另外存在一些 numerical instability, slow convergence 的问题. 并且其相当于一个一阶的算法, 忽略了二阶信息. 

## Background: SPLR Algorithm

提出了一个凸优化问题. 

$$
\min_{x\in \mathbb{R}^{n+m-1}}f(x) , f(x) = \eta \sum_{i=1}^{n+m-1} T_{ij} - \alpha^\top a - \beta^\top b, 
$$
where
$$
x = (\alpha_1, \cdots, \alpha_n, \beta_1, \cdots, \beta_{m-1})^\top, \beta_m = 0, T_{ij} = \exp\left(\frac{\alpha_i + \beta_j - M_{ij}}{\eta}\right)
$$

考虑使用二次牛顿法加速优化. 

$$
\nabla^2 f(x) = \eta^{-1} \begin{bmatrix}
\text{diag}(T \mathbf{1}_m) & T_{-m} \\
T_{-m}^\top & \text{diag}(T_{-m}^\top \mathbf{1}_n)
\end{bmatrix}
$$
- 其计算复杂度由于$T$ 是 exp 出来的, 仍然为 dense 的, 牛顿法的计算复杂度仍然为 $\mathcal{O}(n^3)$, 这也是其主要的计算瓶颈.  
- 这里也和 preconditioned 等有关. 

$$
x_{k+1} = x_k - \tau B_k^{-1} \nabla g_k 
$$
$$
B_k = H_\Omega + (s uu^\top + tvv^\top) + \tau I \approx H_k 
$$

那么进一步的对策为使用 Quasi-Newton 方法, 将上述 Hessian 拆成 sparse + low-rank 两大部分. 理由如下:
- 本身由于 OT 问题, 其 Hessian 本身经验上几乎都是稀疏的 (虽然数学上是非稀疏的, 但从数值上许多元素非常小, 可以近似为零).

Key Theory of SPLR:
- 几乎可以任意的对 Hessian 进行稀疏剔除 (只要保证一些 minimal 的结构不被破坏), 几乎可以无损的进行稀疏的剔除, 并且其条件数仍然是 bounded 不会变坏的. 其保证了算法的收敛性. 

## Moving to GPU: Algorithm-System Co-Design

### Why Not Just Run SPLR on GPU?

在得到 SPLR 之后, 直接将其移植到 GPU 上并不一定能获得理想的性能提升. 主要问题:
- 在求解 $H_\Omega$ 的逆的过程中, 要是用 sparse Cholesky decomposition, 其包含: 1. symbolic analysis (寻找 elimination trees, ordering); 2. numeric factorization. 
- 而在这个符号分析的时候, GPU 是完全空置的 (是 CPU 在做的). 这相当于一个串行的操作. 

### Introducing cuRegOT

进行了三方面改进:
1. Amortized symbolic analysis: 根据 SPLR 的定理, 无论我们如何进行 sparsity 的剔除, 我们都能保证算法的稳定性. 因此我们可以在第一次迭代的时候, 进行一次 symbolic analysis, 得到一个 sparsity pattern. 之后的迭代中, 我们都使用这个 sparsity pattern 来进行 numeric factorization. 这样就避免了每次迭代都要进行 symbolic analysis 的问题 (比如每十次迭代进行一次 symbolic analysis). 另一方面, 从经验上看, 本身的非零元素的稀疏变化也是缓慢的, 因此或许本身也不需要每次都进行完整的 symbolic analysis. 
2. Collaborative CPU-GPU Computing : 即使进行了 amortized symbolic analysis, 其 symbolic analysis 的过程 gpu 仍然是空置的. 此时则计算了一个独立的 Sinkhorn 算法在 GPU 上, 来充分利用 GPU 的计算资源. 然后在更新的的时候与这里的 SPLR 算法进行比较, 然后选择一个更快的算法来进行更新. 这样就实现了 CPU-GPU 的协同计算.
3. Fused CUDA Kernel: 本质在于, 让 GPU 一次通信进行更多的计算, 来稀释一次 CPU-GPU 的通信开销. 因此这里设计了一个 fused kernel, 来通过一个流式的读取, 只进行一次IO, 全部计算集中在 GPU 上进行. 这同时也利用了一些 GPU 有别于 CPU 的并行特性. 


### Results

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/benchmark.png)