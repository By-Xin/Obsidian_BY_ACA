# LinSATNet: The Positive Linear Satisfiablity Neural Network

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260607203914.png)

## Methodology

### Preliminary: Classic Sinkhorn Algorithm for Single Set of Marginal Distributions

考虑 optimial transport 问题. 给定 source $\mathbf{u} \in \mathbb{R}_{\geq 0}^n$ 和 target $\mathbf{v} \in \mathbb{R}_{\geq 0}^m$, 要求 $\sum_{i=1}^n u_i = \sum_{j=1}^m v_j := h$ (即 source 和 target 的总质量相同, 不过这里不要求归一化为概率分布). 

- 对应标准的 transport matrix $\mathbf{P} = \begin{pmatrix} P_{ij} \end{pmatrix} \in \mathbb{R}_{\geq 0}^{m \times n}$,其元素 $P_{ij}$ 表示从 source $j$ 向 target $i$ 传输的质量. 故自然有约束 $\mathbf{P} \mathbf{1}_n = \mathbf{v}$ (即 $P$ 的第 $i$ 行的元素之和等于 $v_i$, 表示 target $i$ 的总质量为 $v_i$) 和 $\mathbf{P}^\top \mathbf{1}_m = \mathbf{u}$ (即 $P$ 的第 $j$ 列的元素之和等于 $u_j$, 表示 source $j$ 的总质量为 $u_j$). 

- 另外定义 Score Matrix $\mathbf{S} \in \mathbb{R}_{\geq 0}^{m \times n}$, 表达从 source $j$ 向 target $i$ 传输的某种偏好. 该矩阵通常没有任何 marginal constrains. Optimal transport 的目标是找到一个 transport matrix $\mathbf{P}$, 使得在满足所有 marginal constrains 的前提下, 偏好 Score Matrix $\mathbf{S}$ 以某种方式被最大化满足. 

- 定义算法的输出 $\boldsymbol{\Gamma} \in [0,1]^{m \times n}$ 为 souce $j$ 分配给 target $i$ 的比例. 因此有关系 $P_{ij} = \Gamma_{ij} u_j$. 用矩阵形式表达为
    $$
    \mathbf{P} = \boldsymbol{\Gamma} \operatorname{Diag}(\mathbf{u}),
    $$
    且有约束 $\sum_i \Gamma_{ij} u_j = u_j$ (即每个 source 的分配比例之和为 1) 和 $\sum_j \Gamma_{ij} u_j = v_i$ (即每个 target 接收到的质量等于 $v_i$).

Sinkhorn 的算法流程如下. 
- 首先初始化, 对于 $\Gamma^{(0)}$, 令其初始值为
    $$
    \Gamma_{ij}^{(0)} = \frac{S_{ij}}{\sum_{i=1}^m S_{ij}}.
    $$
    因此 $\Gamma^{(0)}$ 是行归一化的, 即 $\sum_{j=1}^n \Gamma_{ij}^{(0)} = 1$.

- 在第 $t$ 轮迭代中, 进行交替归一化. 
    - 首先进行行缩放:
        $$
        \Gamma_{ij}^{'(t)} = \Gamma_{ij}^{(t)} \cdot \frac{v_i}{\sum_{j=1}^n \Gamma_{ij}^{(t)}}.
        $$
        归一化后的 $\Gamma^{'(t)}$ 满足 $\sum_{j=1}^n \Gamma_{ij}^{'(t)} = v_i$, 即第 $i$ 行的元素之和等于 $v_i$ (target $i$ 的总质量为 $v_i$).

    - 接着进行列缩放:
        $$
        \Gamma_{ij}^{(t+1)} = \Gamma_{ij}^{'(t)} \cdot \frac{u_j}{\sum_{i=1}^m \Gamma_{ij}^{'(t)}}.
        $$
        归一化后的 $\Gamma^{(t+1)}$ 满足 $\sum_{i=1}^m \Gamma_{ij}^{(t+1)} = u_j$, 即第 $j$ 列的元素之和等于 $u_j$ (source $j$ 的总质量为 $u_j$).

- 迭代进行上述交替归一化, 直到 $\boldsymbol\Gamma^{(t)}$ 收敛. 迭代结束后, 输出 $\boldsymbol\Gamma^{(t)}$ 作为最终的分配比例矩阵, 从而得到 transport matrix $\mathbf{P} = \boldsymbol\Gamma^{(t)} \operatorname{Diag}(\mathbf{u})$.

Sinkhorn 算法的核心在于: 进行行缩放时会破坏列归一化, 反之亦然. 因此通过不断的交替进行行缩放和列缩放, 最终可以同时满足行归一化和列归一化的约束. 该算法由于就是不断进行不断的行列求和, 故容易实现, 可微分, GPU friendly. 

对 Sinkhorn 算法的简要收敛性分析如下. 分别定义行归一化和列归一化的 L1 误差为
$$
\begin{aligned}
L_1(\boldsymbol\Gamma^{(t)}) & = \|\mathbf{v}^{(t)} - \mathbf{v}\|_1, \quad v_i^{(t)} = \sum_{j=1}^n \Gamma_{ij}^{(t)} u_j \\
L_2(\boldsymbol\Gamma^{(t)}) & = \|\mathbf{u}^{(t)} - \mathbf{u}\|_1, \quad u_j^{(t)} = \sum_{i=1}^m \Gamma_{ij}^{(t)} u_i
\end{aligned}
$$
在给定误差精度 $\varepsilon > 0$ 的前提下, Sinkhorn 算法的迭代次数 $T$ 满足
$$
\mathcal{O}\left(\frac{h^2 \log(\Delta / \alpha)}{\varepsilon^2}\right).
$$
其中 $\alpha = \min_{i,j} S_{ij}$ 即为 Score Matrix 中的最小非零元素, 该数值越小, 说明 Score Matrix 的条件数越差, 需要进行更多的迭代, $\Delta = \max_j | \{i: S_{ij} > 0\} |$ 即为 Score Matrix 的列稀疏度的最大值, 该数值越大, 说明 Score Matrix 的稀疏度越差, 同样代表问题更复杂, 需要进行更多的迭代.

### Extension: Sinkhorn Algorithm for Multiple Sets of Marginal Distributions

在 classic Sinkhorn 的基础上, 现在给定 $k$ 组 source-target pair $\{(\mathbf{u}_\eta, \mathbf{v}_\eta)\}_{\eta=1}^k$, 要求 $\sum_{i=1}^n u_{\eta,i} = \sum_{j=1}^m v_{\eta,j} := h_\eta$ (即每组 source 和 target 的总质量各自相同, 但并不要求彼此相等). 其共享同一个 Score Matrix $\mathbf{S} \in \mathbb{R}_{\geq 0}^{m \times n}$. 最终的目标还是找到一个分配比例矩阵 $\boldsymbol\Gamma \in [0,1]^{m \times n}$, 使得对于每组 source-target pair $(\mathbf{u}_\eta, \mathbf{v}_\eta)$, 都满足 $\boldsymbol\Gamma \mathbf{u}_\eta = \mathbf{v}_\eta$, 以及 $\boldsymbol\Gamma^\top \mathbf{v}_\eta = \mathbf{u}_\eta$.

针对这样的 multi-set 的情况, 在每轮迭代中都将轮流更新 $k$ 组 source-target pair 中的一组. 因此在第 $t$ 轮迭代中, 被更新的 source-target pair 为 $\eta = (t \mod k) + 1$. 例如, 假设 $k=3$, 则在第 $t = 0,1,2,3,4,5 \ldots$ 轮迭代中, 被更新的 source-target pair 分别为 $\eta = 1,2,3,1,2,3,\ldots$.

对于第 $t$ 次迭代 $\boldsymbol\Gamma^{(t)}$ 以及 $\boldsymbol\Gamma^{'(t)}$, 对应第 $\eta$ 组 source-target pair, 当前的迭代状态为:
$$
\begin{aligned}
\mathbf{v}_\eta^{(t)} & = \boldsymbol\Gamma^{(t)} \mathbf{u}_\eta, \quad \mathbf{u}_\eta^{(t)} = \boldsymbol\Gamma^{'(t)\top} \mathbf{v}_\eta.
% \boldsymbol\pi_{v_\eta}^{(t)} & = \frac{\mathbf{v}_\eta^{(t)}}{h_\eta}, \quad \boldsymbol\pi_{u_\eta}^{(t)} = \frac{\mathbf{u}_\eta^{(t)}}{h_\eta}.
\end{aligned}
$$

另外定义 probability marginals, 即对于每组 source-target pair $(\mathbf{u}_\eta, \mathbf{v}_\eta)$, 定义
$$
\text{Required Marginals: }    \quad \boldsymbol\pi_{u_\eta} = \frac{\mathbf{u}_\eta}{h_\eta}, \quad \boldsymbol\pi_{v_\eta} = \frac{\mathbf{v}_\eta}{h_\eta} , \\ 
\text{Current Marginals: } \quad \boldsymbol\pi_{u_\eta}^{(t)} = \frac{\mathbf{u}_\eta^{(t)}}{h_\eta}, \quad \boldsymbol\pi_{v_\eta}^{(t)} = \frac{\mathbf{v}_\eta^{(t)}}{h_\eta}. 
$$
是满足归一化的对应概率分布. 故可以进而定义当前迭代距离目标的 KL divergence 为
$$D_{\mathrm{KL}}
\left(
\boldsymbol\pi_{v_\eta}
\middle\|
\boldsymbol\pi_{v_\eta}^{(t)}
\right)
=
\frac1{h_\eta}
\sum_{i=1}^m
v_{\eta,i}
\log
\frac{v_{\eta,i}}{v_{\eta,i}^{(t)}}, \quad 
D_{\mathrm{KL}}(\boldsymbol\pi_{u_\eta} \| \boldsymbol\pi_{u_\eta}^{(t)}) = \frac1{h_\eta} \sum_{j=1}^n u_{\eta,j} \log \frac{u_{\eta,j}}{u_{\eta,j}^{(t)}}.
$$