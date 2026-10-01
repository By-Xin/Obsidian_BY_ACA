# 加速 truncation 计算

## 任务记号

首先给出 notation. 假设在当前时间步 $t$, 经过 LLM 后, 有 probability 向量 $\mathbf{p} = [p_1, p_2, \ldots, p_{|\mathcal{V}|}]^\top$, 其中 $p_i$ 表示词汇表中第 $i$ 个 token 的概率. 为了方便, 不妨将这个向量进行排序, 得到 $\mathbf{\hat p} = [p_{(1)}, p_{(2)}, \ldots, p_{(|\mathcal{V}|)}]^\top$, 其中 $p_{(i)}$ 表示排序后第 $i$ 个 token 的概率, 规定 $p_{(1)} \geq p_{(2)} \geq \ldots \geq p_{(|\mathcal{V}|)}$.

给定一个作用在这个概率向量上的函数 $f_\tau(\mathbf{\hat p}): \mathbb{R}^{|\mathcal{V}|} \to \mathbb{R}^{|\mathcal{V}|}$, 其输出是按照某个阈值 $\tau$ 进行截断后的概率向量. 具体地, 这个截断函数暂时设计如下 (待验证):
$$f_\tau(\mathbf{\hat p}) = [f_\tau(p_{(i)})] = [p_{(i)} \cdot  \boldsymbol{1} \{p_{(i)} / p_{\max} \ge \tau\}] \in \mathbb{R}^{|\mathcal{V}|}$$

现有 $K$ 个这样的 truncation 阈值, 同样从大到小排列为 $\tau_{(1)} \ge  \tau_{(2)} \ge \ldots \ge \tau_{(K)}$, 其各自对应了一个截断函数 $f_{\tau_{(k)}}(\mathbf{\hat p})$. 为方便起见, 也将这个阶段后的函数简记为 $f_{(i)}:= f_{\tau_{(i)}}(\mathbf{\hat p})$. 同时对这 $K$ 个函数进行加权, 得到一个聚合后的函数:
$$\phi = \sum_{i=1}^{K} \theta_{(i)} f_{(i)} \in \mathbb{R}^{|\mathcal{V}|}$$
最后对这个向量进行归一化, 并从中抽样, 作为最后的 decoding: 
$$x_{t+1} \sim \tilde{\phi} :=  \text{Norm}( {\phi} )$$


## 加速策略

上述过程如果直接实现的话, 会申请 $K$ 个大小为 $|\mathcal{V}|$ 的向量, 这在内存和计算上都是不小的开销.  因此需要进行加速.  这里只需要一个总的排序好的 vocabulary 概率向量 $\mathbf{\hat p}$, 以所有的从大到小排序好的截断阈值 $\tau_{(1)} \ge \tau_{(2)} \ge \cdots \ge \tau_{(K)}$ (且约定 $\tau_{(0)}=1$) 作为标记 (对应 $f_{(i)}, \theta_{(i)}$). 

作为一个截断映射, 事实上就是将一部分的向量元素进行 identity 保留, 其余截断为 0. 因此, 记保留的部分为 $\mathbf{p^{(1)}}$ (其中对于 $ \mathbf{p^{(1)}}$ 中的任意元素 $p_{(i)}$, 均有 $p_{(i)} / p_{\max} \ge \tau_{(1)}$). 故
$$f_{(1)} = \begin{bmatrix} \mathbf{p^{(1)}} \\ \boldsymbol{0}\end{bmatrix}$$


同理有
$$f_{(2)} = \begin{bmatrix}
\mathbf{p^{(1)}} \\
\mathbf{p^{(2)}}\\
\boldsymbol{0}
\end{bmatrix}, \cdots ,
f_{(K)} = \begin{bmatrix}
\mathbf{p^{(1)}} \\
\mathbf{p^{(2)}}\\
\vdots \\
\mathbf{p^{(K)}}\\
\boldsymbol{0}
\end{bmatrix}
$$

故可以总结为, 对任意$k \in \{1,\dots,K\}$, $\mathbf{p^{(k)}} := \left\{\, i : \tau_{(k)} \le \frac{p_{(i)}}{p_{\max}} < \tau_{(k-1)} \,\right\}$. 对应 $f_{(k)} = [\mathbf{p^{(1)}}; \mathbf{p^{(2)}}; \cdots; \mathbf{p^{(k)}}; \boldsymbol{0}]^\top$.

因此:
$$\begin{aligned}
\phi &= \sum_{i=1}^{K} \theta_{(i)} f_{(i)} \\
&= \sum_{i=1}^{K} \theta_{(i)} \begin{bmatrix}
\mathbf{p^{(1)}} \\
\mathbf{p^{(2)}}\\
\vdots \\
\mathbf{p^{(i)}}\\
\boldsymbol{0}
\end{bmatrix} \\
&= \begin{bmatrix}
(\theta_{(1)} + \theta_{(2)} + \cdots + \theta_{(K)}) \mathbf{p}^{(1)} \\
(\theta_{(2)} + \cdots + \theta_{(K)}) \mathbf{p}^{(2)}\\
\vdots \\
(\theta_{(K)}) \mathbf{p}^{(K)}\\
\boldsymbol{0}
\end{bmatrix}\\
&:= \begin{bmatrix}
s_{(1)} \mathbf{p}^{(1)} \\
s_{(2)} \mathbf{p}^{(2)}\\
\vdots \\
s_{(K)} \mathbf{p}^{(K)}\\
\boldsymbol{0}
\end{bmatrix}\end{aligned}$$

因此对于 $\mathbf{\hat p}$ 的每个元素 $p_{(i)}$, 需要找到其对应的 $\tau_{(k)}$ (即 $\exists k: \tau_{k} \leq {p_{(i)}}/{p_{\max}} < \tau_{k-1}$), 则此时, $\phi_i = (\sum_{j=k}^{K} \theta_{(j)}) p_{(i)}$. 这样就不需要存储 $K$ 个向量, 只需要存储一个向量和 $K$ 个阈值即可.

---

## 算法实现

事实上, 上述对 $\mathbf{p}$ 的排序操作也是多余的. 我们在具体算法实现上可以进一步优化. 这里采用 searchsorted 的方式, 直接在原始的 $\mathbf{p}$ 上进行操作. 具体步骤如下. 

**输入**: 未排序之概率向量 $\mathbf{p}$, 已排序好之截断阈值 $\tau_{(1)} \ge \tau_{(2)} \ge \cdots \ge \tau_{(K)}$. 
**计算流程**:
1. 计算 $p_{\max}$
2. 计算边界值 $b_{(k)}:=\tau_{(k)}\cdot p_{\max}, k\in\{1,\ldots,K\}$, 得到降序 $b_{\text{desc}}=[b_{(1)},\dots,b_{(K)}]$ . 翻转为升序 $b_{\text{asc}}=[b_{(K)},\dots,b_{(1)}]$.
3. 计算系数和 $s_k=\sum_{j=k}^K\theta_{(j)}, k\in\{1,\ldots,K\}$. 规定 $s_{K+1}=0$. 
4. 对向量 $\mathbf{p}$ 计算 `searchsorted(b_asc, p, right=True)`. 即遍历 $\mathbf{p}$ 中所有元素 $p_i\in \mathbf{p}$ , 求每个分量 $\gamma_i := \min\{ m \in \{0,1,\dots,K\} : p_i < b_{\text{asc}}[m+1] \}$, 即确认 $p_i$ 落在哪个区间.
5. 这样就可以将 $p_i$ 应落入的位置对应到已排序好的区间索引: $k_i = K - \gamma_i + 1$.  故可以对应权重系数 $s_{k_i}$.