好，我们把 **reranking** 过程彻底形式化一遍，用纯数学语言、带上维度标注。你就能清楚地看到每个对象的类型、索引范围和在 pipeline 中的位置。

---

## 一、符号总览

| 符号                                    | 含义                        | 典型维度/类型                            |
| ------------------------------------- | ------------------------- | ---------------------------------- |
| (\mathcal{I})                         | 全部商品集合                    | 大约 (10^6) 个 item id                |
| (\mathcal{S})                         | 训练集中的 session 集合          | 几百万个 session                       |
| (s \in \mathcal{S})                   | 一个具体的 session             | —                                  |
| (U_s = [i_1, i_2, \dots, i_{L_s}])    | session (s) 的历史行为序列（已发生）  | 长度 (L_s)                           |
| (U_s^{\text{set}})                    | 历史商品集合（去重）                | (                                  |
| (Y_s)                                 | session 的未来真实行为集合（用于打标）   | 一组 item id                         |
| (M^{(k)} \in \mathbb{R}_{+}^{         | \mathcal{I}               | \times                             |
| (K)                                   | 共现矩阵个数                    | 约 20                               |
| (\text{cand}(s)\subseteq \mathcal{I}) | 为 session (s) 召回的候选集合     | 大约 100 个 item                      |
| (x_{s,j}^{(k)})                       | 候选 (j) 在第 (k) 个共现矩阵下的强度特征 | 标量                                 |
| (\mathbf{x}_{s,j})                    | 候选 (j) 的特征向量              | (\mathbb{R}^{d})，通常 (d \approx 30) |
| (y_{s,j})                             | 标签：若 (j \in Y_s) 则 1，否则 0 | 标量                                 |

---

## 二、特征生成阶段（feature extraction）

1. **从历史行为构造候选池**  
    $$  
    \text{cand}(s)  
    ;=;  
    \bigcup_{k=1}^{K}  
    \operatorname{TopN}!\left(\sum_{i \in U_s^{\text{set}}} M^{(k)}_{i,\cdot}\right)  
    $$
    
    - 对每个矩阵 (M^{(k)})，取 session 内所有商品 (i) 的出边 (M^{(k)}_{i,\cdot})，累加权重；
        
    - 选取权重最高的前 (N_k) 个商品；
        
    - 把所有矩阵的结果合并并去重；
        
    - (N_k) 通常为 50。
        
    
    输出集合大小 (|\text{cand}(s)| \approx 100)。
    

---

2. **为每个候选计算交互强度特征**  
    对任意 (j \in \text{cand}(s))，定义  
    $$  
    x_{s,j}^{(k)}  
    ;=;  
    \sum_{i \in U_s^{\text{set}}} M^{(k)}_{i j}.  
    $$
    
    - 每个 (x_{s,j}^{(k)}) 是标量；
        
    - # 共得到 (K) 维向量：  
        [  
        \mathbf{x}_{s,j}^{\text{covisit}}
        
        \big(x_{s,j}^{(1)}, x_{s,j}^{(2)}, \dots, x_{s,j}^{(K)}\big)  
        \in \mathbb{R}^{K}.  
        ]
        

---

3. **拼接其它特征**
    
    - item-level 统计：点击次数、下单次数、最近热度等 → 维度 (d_1)；
        
    - session-level 统计：长度、时间段、点击/下单比例 → 维度 (d_2)；
        
    - time-diff 类特征 → 维度 (d_3)。
        
    
    # 最终特征向量：  
    $$  
    \mathbf{x}_{s,j}
    
    \big[,  
    \mathbf{x}_{s,j}^{\text{covisit}},  
    \mathbf{x}_{s,j}^{\text{item}},  
    \mathbf{x}_{s,j}^{\text{session}},  
    \mathbf{x}_{s,j}^{\text{time}}  
    ,\big]  
    \in \mathbb{R}^{d},  
    $$  
    其中 (d = K + d_1 + d_2 + d_3)，通常 30～50。
    

---

4. **打标签**  
    $$  
    y_{s,j} =  
    \begin{cases}  
    1, & \text{if } j \in Y_s,\  
    0, & \text{otherwise.}  
    \end{cases}  
    $$
    

---

## 三、训练集的矩阵形状

假设训练集中共 (N_{\text{session}}) 个 session，每个 session 召回 (N_{\text{cand}}) 个候选。

则训练样本总数约为  
[  
N = N_{\text{session}} \times N_{\text{cand}}.  
]

数据矩阵：

|矩阵|维度|含义|
|---|---|---|
|(\mathbf{X})|(N \times d)|所有特征|
|(\mathbf{y})|(N \times 1)|二元标签|
|(\mathbf{g})|(N_{\text{session}} \times 1)|group 向量（每组样本数）|

---

## 四、训练：XGBRanker 的目标

XGBoost Ranker 用 pairwise 或 NDCG 目标。以 pairwise 为例：

对每个 session (s)，定义候选集合 (\mathcal{C}_s = \text{cand}(s))。  
模型 (f(\mathbf{x};\theta)) 输出分数 (\hat{y}_{s,j}=f(\mathbf{x}_{s,j};\theta))。

# 排序损失：  
[  
\mathcal{L}(\theta)

\sum_{s\in\mathcal{S}}  
\sum_{\substack{(j_p, j_n)\in \mathcal{P}_s}}  
\ell\big(\hat{y}_{s,j_p} - \hat{y}_{s,j_n}\big),  
]  
其中  
(\mathcal{P}_s = {(j_p, j_n) : y_{s,j_p}=1, y_{s,j_n}=0})，  
(\ell(z)) 通常是 logistic loss (\log(1+\exp(-z)))。

模型参数 (\theta) 通过梯度提升树学习。

---

## 五、推理阶段（Inference）

对每个测试 session (s)：

1. 根据历史行为生成候选 (\text{cand}(s))；
    
2. 对每个 (j\in \text{cand}(s)) 计算 (\mathbf{x}_{s,j})；
    
3. 得到分数：  
    [  
    \hat{y}_{s,j} = f(\mathbf{x}_{s,j}; \hat{\theta}).  
    ]
    
4. 排序：  
    [  
    \text{rank}_s(j) = \operatorname{argsort}_{j\in \text{cand}(s)} (-\hat{y}_{s,j});  
    ]  
    取前 (K') 个作为最终推荐列表。
    

---

## 六、数据维度的直观量级示例

|名称|典型值|
|---|---|
|(|\mathcal{I}|
|(K)（共现矩阵数）|20|
|(N_{\text{cand}})|100|
|平均 (L_s)|10–40|
|特征维度 (d)|30–50|
|样本总数 (N)|几千万～上亿|

---

## 七、结果的含义

- 对于同一个 session (s)，模型输出一个向量  
    (\hat{\mathbf{y}}_s = (\hat{y}_{s,j_1}, \dots, \hat{y}_{s,j_{N_{\text{cand}}}}))。
    
- 排序后取 top 20，用于 clicks/carts/orders 各目标提交。
    
- 每个 (\hat{y}_{s,j}) 表示该用户 session 继续交互该商品的概率/倾向。
    

---

一句话总结这一阶段的数学结构：

> 训练阶段学函数  
> [  
> f: \mathbb{R}^d \to \mathbb{R}, \quad  
> \text{使得 } f(\mathbf{x}_{s,j_p}) > f(\mathbf{x}_{s,j_n}) \text{ 当 } y_{s,j_p}=1, y_{s,j_n}=0,  
> ]  
> 在推理阶段，对每个 session 内的候选集合 (\text{cand}(s))，按 (f(\mathbf{x}_{s,j})) 降序排列。

---

要不要我接着把这个写成一个小型数学 pipeline 图（比如矩阵与索引结构关系图），让你能看到每个量的箭头关系？