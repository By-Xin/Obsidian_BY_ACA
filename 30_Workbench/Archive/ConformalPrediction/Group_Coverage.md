
**Group Coverage**

给定测试集合 ${(\mathrm{X}^{(i)}, Y^{(i)})}_{i=1}^N$，其中 $\mathrm{X}^{(i)} = [X^{(i)}_1, X^{(i)}_2, \ldots, X^{(i)}_d]^\top \in \mathbb{R}^d$, $Y^{(i)} \in \mathbb{R}$. 有某种预测方法在 $(\mathrm{X}^{(i)}, Y^{(i)})$ 上构造预测区间 $C^{(i)}$, 同时记成功覆盖的 indicator 为 $I^{(i)} = \boldsymbol{1}{Y^{(i)} \in C^{(i)}}$.

在当前样本空间, 对于每个维度 $j \in {1,2,\ldots,d}$, 将该维度的取值范围划分为 $K\in \mathbb{N}$ 个尽量等频率的区间, 记这些区间为

$$\mathcal{B}_j = {B_{j,1}, \ldots, B_{j, K}}$$

其中每个区间 $B_{j,k} = [b_{j,k}^{low}, b_{j,k}^{up}) \subseteq \mathbb{R}$, 相当于 $\mathrm{X}^{(i)}$ 在第 $j$ 个维度上的一个子区间.

对应的可以定义维度 $j$ 的分组函数 $g_j(X^{(i)}_j): \mathbb{R} \to {1,2,\ldots,K}$, 其将样本 $i$ 在第 $j$ 个维度的取值 $X^{(i)}_j$ 指派到对应的区间编号上:

$$g_j(X^{(i)}_j) = b^{(i)}_j\ \Leftrightarrow\ X^{(i)}_j \in B_{j,b}, \quad \forall i \in {1,\ldots,N}.$$ 其中 $b_j^{(i)} \in {1,\ldots,K}$ 表示样本 $i$ 在第 $j$ 个维度上所属的区间编号.

故对某个样本, 遍历其每个维度 $j$, 就可以得到该样本最终的分组编号为 $(b_1^{(i)}, b_2^{(i)}, \ldots, b_d^{(i)})$. 再进一步遍历每个样本 $i \in {1,\ldots,N}$ 的所有维度 $j \in {1,\ldots,d}$, 就可以得到该数据集在每个维度上的分组编号.

总的而言, 一共有 $d$ 个维度, 每个维度有 $K$ 个区间, 因此通过类似笛卡尔积的方式, 可以得到总共 $K^d$ 个分组. 我们可以站在分组的角度, 得到每个组对应的样本索引集合:

$$G_{b_1\cdots b_d} = {i\in {1,\ldots,N}: g_1(X^{(i)}_1) = b_1, \ldots, g_d(X^{(i)}_d) = b_d}, \quad \forall b_1,\ldots,b_d \in {1,\ldots,K}.$$ 其中 $G_{b_1\cdots b_d}$ 表示第1个维度在第 $b_1$ 个区间, 第2个维度在第 $b_2$ 个区间, ..., 第 $d$ 个维度在第 $b_d$ 个区间的样本集合, 其中 $b_j \in {1,\ldots,K}$.

接下来, 我们可以计算每个组的覆盖率. 对于某个组 $G_{b_1\cdots b_d}$, 其覆盖率定义为:

$$c_{b_1\cdots b_d} := \mathbb{P}(Y \in C | \mathrm{X} \in G_{b_1\cdots b_d}) \approx \frac{1}{|G_{b_1\cdots b_d}|} \sum_{i \in G_{b_1\cdots b_d}} I^{(i)}.$$

故定义总体的**最小分组覆盖率**为:

$$\text{MinGroupCov} := \min_{b_1,\ldots,b_d \in {1,\ldots,K}} c_{b_1\cdots b_d}.$$

即为所有维度之分组的可能性中覆盖率的下界.