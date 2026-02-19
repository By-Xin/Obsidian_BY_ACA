# Multiple Comparisons

## ANOVA 回顾

Multiple comparisons 还是在一个 ANOVA 的场景下进行的. 这里简要回顾一下. 假设我们有 $a$ 个组. 若用 $Y_{ij}$ 表示第 $i$ 组的第 $j$ 个观测值, 那么我们有
$$
Y_{ij} = \mu_i + \epsilon_{ij}, \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)
$$

我们也可以用线性模型的形式来表示. 这里如果有 $a$ 个组, 那么我们可以用 $a-1$ 个虚拟变量来表示. 不妨令第一组为 baseline, 那么可以引入 $u_{i2}, u_{i3}, \ldots, u_{ia}$ 作为虚拟变量. 其含义为, 若第$i$个观测值属于第 $j$ 组, 则 $u_{ij} = 1$, 否则 $u_{ij} = 0$. 那么我们可以用线性模型来表示:
$$
Y_i  = \beta_0 + \beta_2 u_{i2} + \beta_3 u_{i3} + \ldots + \beta_a u_{ia} + \epsilon_i
$$
则此时:
- $\beta_0 = \mu_1$ 是第 1 组(基准组)的均值
- $\beta_j  = \mu_j - \mu_1$ 是第 $j$ 组的均值与基准组均值的差异



在这套模型中, 我们往往希望分析各组之间的差异以及其统计显著性. 
- 例如, 在生物实验中, 第一组可能是对照组, 其他组是不同的处理组. 我们希望知道例如第二种处理(第二组)与对照组的差异是否显著, 即 $\mu_2 - \mu_1 \neq 0$ 是否显著. 

因此引入 contrast (对比) 的概念. Contrast 是对组均值的线性组合:
$$
\sum_{j=1}^a c_j \mu_j := \mathbf{c}^\top \boldsymbol{\mu}
$$
其中 $\mathbf{c} = (c_1, c_2, \ldots, c_a)^\top$ 是一个 $a$ 维向量. 这里要求 $\sum_{j=1}^a c_j = 0$. 
- 例如(假设这里只有4组 $\boldsymbol{\mu} = (\mu_1, \mu_2, \mu_3, \mu_4)^\top$) , $\mu_2 - \mu_1$ 可以表示为 
$$
\begin{pmatrix}
-1 & 1 & 0 & 0
\end{pmatrix}
\begin{pmatrix}
\mu_1 \\
\mu_2 \\
\mu_3 \\
\mu_4 \\
\end{pmatrix}
$$
- 例如 $\mu_2 - (\mu_1 + \mu_3+\mu_4)/3$ 可以表示为
$$
\begin{pmatrix}
-1/3 & 1 & -1/3 & -1/3 
\end{pmatrix}
\begin{pmatrix}
\mu_1 \\
\mu_2 \\
\mu_3 \\
\mu_4  \\
\end{pmatrix}
$$

又由于我们刚刚在线性模型中有 $\beta_0 = \mu_1, \beta_j  = \mu_j - \mu_1 (j\ge2)$, 因此
$$\mu_j = 
\begin{cases}
\beta_0 \quad \text{if } j=1 \\
\beta_0 + \beta_j \quad \text{if } j \ge 2
\end{cases}
$$  
所以上述的 contrast 也可以用 beta 来表示, 记为 $\mathbf{a}^\top \boldsymbol{\beta}$ (即 $\mathbf{c}^\top \boldsymbol{\mu} \equiv \mathbf{a}^\top \boldsymbol{\beta}$). 例如
- $\mu_2 - \mu_4$ 可以表示为 (记$\boldsymbol{\beta} = \begin{pmatrix}
\beta_2, \beta_3, \beta_4
\end{pmatrix}^\top$)
$$\begin{align*}
\mu_2 - \mu_4 &= (\beta_0 + \beta_2) - (\beta_0 + \beta_4) = \beta_2 - \beta_4\\
\Rightarrow \mathbf{a} &= \begin{pmatrix}
-1 & 0 & 1
\end{pmatrix}
\end{align*}$$
- $(\mu_1+\mu_2)/2 - (\mu_3+\mu_4)/2$ 可以表示为
$$
\begin{align*}
\frac{\mu_1+\mu_2}{2} - \frac{\mu_3+\mu_4}{2} &= \frac{(\beta_0 + \beta_1) + (\beta_0 + \beta_2)}{2} - \frac{(\beta_0 + \beta_3) + (\beta_0 + \beta_4)}{2}\\
&= \frac{2\beta_0 + \beta_1 + \beta_2 - 2\beta_0 - \beta_3 - \beta_4}{2} \\&= \frac{\beta_1 + \beta_2 - \beta_3 - \beta_4}{2}\\
\Rightarrow \mathbf{a} &= \begin{pmatrix}
\frac{1}{2} & \frac{1}{2} & -\frac{1}{2} & -\frac{1}{2}
\end{pmatrix}
\end{align*}
$$

这里的 Multiple Comparisons 就是在此为基础进行的. 我们根据实际问题的需要有不同的 contrast, 我们希望能够假设检验:
$$
H_0: \mathbf{c}^\top \boldsymbol{\mu} = \mathbf{a}^\top \boldsymbol{\beta} = 0
$$
也就是本质上我们依然希望能够检验某种处理会不会对该组的结果(均值)产生显著影响, 而我们用来处理的手段就是引入上面的线性方程组并且通过对 $beta$ 的假设检验来进行分析. 其整体的设计思路依然是类似于 t 检验的构造统计量 $T = \frac{\mathbf{c}^\top \boldsymbol{\hat\beta}}{\sqrt{\mathbf{c}^\top \hat\Sigma \mathbf{c}}}$ 即估计值比上标准误. 但是由于具体情况的差异, 我们要对判断的标准等进行一系列调整. 

这里主要会介绍四种检验方法, 整体分为三组:
- General Exploring (**Scheffe 方法**): 
  - 我们在拿到数据进行具体分析之前没有特定的假设 (例如我们不会在实验设计的时候就想知道第二组treatment和对照组的差异), 只是想从宏观的目前手里的数据是否存在哪几对实验组之间存在差异, 有多少组都可以, 任意组之间存在差异都可以. 这种情况可以被叫做是 **post-hoc** 或者 **post-specfication** (因为我们是在拿到数据跑完回归之后, 再去根据系数判断是否有一些有差异的组). 
  - 并且我们参与比较的组也是任意的, 例如我们可以比较 $\mu_2 - \mu_1$ (和基准组对比), $\mu_2 - \mu_4$ (任意两组对比), 甚至 $(\mu_1 + \mu_2)/2 - (\mu_3+\mu_4)/2$ (任意多组构造的对比).
- Pairwise Comparisons (**Tukey 方法 / Dunnett 方法**): 这两种方法也是 post-hoc 的, 但是它们的假设是我们只比较两组之间的差异, 例如 $\mu_2 - \mu_1$, $\mu_3 - \mu_2$ 等等. 再具体而言,
  - Tukey 方法是比较任意两组之间的差异
  - Dunnett 方法是比较任意组与基准组之间的差异, 即只能是 $\mu_j - \mu_1, j \ge 2$ 的形式. 
- Pre-specified Comparisons (**Bonferroni 方法**): Bonferroni 方法是 pre-specified 的, 即我们在实验设计的时候就已经确定了要比较的组, 例如 $\mu_2 - \mu_1$, $(\mu_1 + \mu_2)/2 - (\mu_3+\mu_4)/2$ 等等 (Bonferroni 方法不限制参与比较的组的数量, 但是它们的形式是预先指定的).

上述的分类看上去可能有一些微妙, 因为我们唯一的区别是指定的比较形式不同. 但是其本质上会影响到我们检验的边界值的确定. 当我们的检验的形式指定的越具体 (例如Dunnett 方法, 明确说明我们会比较某一组和基准组的差异), 那么我们就可以用更精准的边界值来进行检验. 反之, 当我们指定的比较形式越模糊 (例如 Scheffe 方法, 我们可以比较任意两多组之间的差异), 那么我们就需要用足够保守的边界值来进行检验, 使得我们至少要让整体的概率满足 $\alpha = 0.05$ (或其他的显著性水平) 的要求.

## Scheffe 方法

Scheffe 方法不指定任何的比较形式, 也不限制参与比较的组的数量. 其整体流程如下.

**1. 建立模型**

这里以一个具体的例子来说明. 对于某肉制品, 我们有四种包装方式, 分别为 Package 1 ~ Package 4. 我们希望测定其细菌数量的差异. 这里我们有 $a=4$ 组, 每组分别取了3个样本测定其细菌数量. 具体数据为:

| 组别 | 样本1 | 样本2 | 样本3 |
|------|-------|-------|-------|
| package1    | 7.66 | 6.98 | 7.80 |
| package2   | 5.26 | 5.44 | 5.80 |
| package3    | 7.41 | 7.33 | 7.04 |
| package4    | 3.51 | 2.91 | 3.66 |

可以建立如下的线性模型:
$$
Y_i = \beta_0 + \beta_2 u_{i2} + \beta_3 u_{i3} + \beta_4 u_{i4} + \epsilon_i
$$

在 R 中可以用以下代码来建立模型:
```r
fit = lm( y ~ package, data = data)
```
这里的 `y` 是细菌数量, `package` 是组别.

拟合的结果为:

| Coefficient | Estimate | Std. Error | t value | Pr(>abs(t)) |
|------|-------|-------|-------|-------|
| (Intercept)  | 7.4800   | 0.1965     | 38.064  | 2.49e-10     |
| package2     | -1.9800  | 0.2779     | -7.125  | 9.95e-05     |
| package3     | -0.2200  | 0.2779     | -0.792  | 0.451        |
| package4     | -4.1200  | 0.2779     | -14.825 | 4.22e-07     |



**2. 进行 ANOVA 检验, 查看整体差异**

首先, 我们要进行 ANOVA 检验, 查看整体而言, 各组之间是否存在差异. 即:
$$
H_0: \mu_1 = \mu_2 = \mu_3 = \mu_4
$$

我们可以用以下代码来进行 ANOVA 检验:
```r
anova(fit)
```

其结果为:
| Source    | Df | Sum Sq | Mean Sq | F value | Pr(>F)       |
|-----------|----|--------|---------|---------|--------------|
| package   | 3  | 32.873 | 10.9576 | 94.584  | 1.376e-06    |
| Residuals | 8  | 0.927  | 0.1159  |         |              |

由于这里 `p-value` 非常小, **因此我们拒绝原假设, 认为各组之间存在显著差异.**

**3. 进行 Scheffe 检验, 查看具体差异**

*(1) 挑选希望进行比较的组*

这里我们要进行 Scheffe 检验, 查看具体是哪几对组之间存在差异. 

这里我们进行观察, 在上一步的 coefficient 中观察哪几组的系数是显著且取值差异较大的 (这也是为什么我们是 post-hoc 的原因, 因为我们可以根据数据的结果来进行判断). 例如我们这里观察到, $\beta_2, \beta_4$ 的系数是显著且绝对值较大的, 因此我们怀疑 P2组, P4组分别与基准组 P1会存在显著差异. 另外 $\beta_2 - \beta_3, \beta_2 - \beta_4, \beta_3 - \beta_4$ 之差的绝对值也较大, 因此我们也希望检验他们之间是否存在显著差异.

因此我们这里确定了我们希望比较的组为:
| 以 $\mu_k$ 的角度 | 以 $\beta_k$ 的角度 |
|------|------|
| $\mu_2 - \mu_1$ | $\beta_2$ |
| $\mu_4 - \mu_1$ | $\beta_4$ |
| $\mu_2 - \mu_3$ | $\beta_2 - \beta_3$ |
| $\mu_2 - \mu_4$ | $\beta_2 - \beta_4$ |
| $\mu_3 - \mu_4$ | $\beta_3 - \beta_4$ |


*(2) 写出对应的 contrast*

我们这里有5个检验要做, 分别写出各自的 contrast. 注意这里的 contrast 我们要以 $\beta$ 的角度来写, 即 $\mathbf{a}^\top \boldsymbol{\beta}$ 且注意这里 $\boldsymbol{\beta} = (\beta_2, \beta_3, \beta_4)^\top$, 一定要比总的组数少1维.


$$
\begin{align*}
\beta_2 &= \beta_2 + 0\beta_3 + 0\beta_4 \Rightarrow \mathbf{a}_1 = (1,0,0) \\
\beta_4 &= 0\beta_2 + 0\beta_3 + \beta_4 \Rightarrow \mathbf{a}_2 = (0,0,1) \\
\beta_2 - \beta_3 &= \beta_2 - \beta_3 + 0\beta_4 \Rightarrow \mathbf{a}_3 = (1,-1,0) \\
\beta_2 - \beta_4 &= \beta_2 + 0\beta_3 - \beta_4 \Rightarrow \mathbf{a}_4 = (1,0,-1) \\
\beta_3 - \beta_4 &= 0\beta_2 + 1\beta_3 - \beta_4 \Rightarrow \mathbf{a}_5 = (0,1,-1) \\
\end{align*}
$$

用 R 中可以用以下代码来记录 contrast:
```r
ctrst1 = c(1,0,0); ctrst2 = c(0,0,1) ; ctrst3 = c(1,-1,0);
ctrst4 = c(1,0,-1); ctrst5 = c(0,1,-1)
```

不过事实上上述内容可以合并为矩阵的形式一次性计算出来:
```r
conts = matrix(c(1,0,0, 0,0,1, 1,-1,0, 1,0,-1, 0,1,-1), nrow=5)
```

本质上对应着五个假设检验:
$$
H_0^{(i)}: \mathbf{a}_i^\top \boldsymbol{\beta} = 0, i=1,2,\ldots,5
$$

*(3) 计算检验统计量*

Scheffe 方法的检验统计量为 (其实和 t 检验是一样的):
$$
T_i = \frac{\mathbf{a}_i^\top \boldsymbol{\hat\beta}}{\sqrt{\text{Var}(\mathbf{a}_i^\top \boldsymbol{\hat\beta})}} = \frac{\mathbf{a}_i^\top \boldsymbol{\hat\beta}}{\sqrt{\mathbf{a}_i^\top \hat\Sigma \mathbf{a}_i}}
$$

例如, 对于 contrast 1 ($\mathbf{a}_1 = (1,0,0)$), 直接可以从 `summary(fit)$coefficients` 中得到:
$$
\begin{align*}
T_1 &= \frac{\hat\beta_2}{\sqrt{\text{Var}(\beta_2)}} = \frac{-1.9800}{{0.02779}}
\end{align*}
$$

再如, 对于 contrast 3 ($\mathbf{a}_3 = (1,-1,0)$), 我们可以得到:
$$
\begin{align*}
T_3 &= \frac{\hat\beta_2 - \hat\beta_3}{\sqrt{\text{Var}(\hat\beta_2 - \hat\beta_3)}} \\
&= \frac{\hat\beta_2 - \hat\beta_3}{\sqrt{\text{Var}(\hat\beta_2) + \text{Var}(\hat\beta_3) - 2\text{Cov}(\hat\beta_2, \hat\beta_3)}} \\
&= \frac{-1.9800 - (-0.2200)}{\sqrt{0.02779^2 + 0.2779^2 - 2 \times 0.039}} \\
\end{align*}
$$
其中 $\text{Cov}(\hat\beta_2, \hat\beta_3)$ 可以通过需要通过 `vcov(fit)` 来计算. 并对应提取:
  ```r
    > vcov(fit)
                (Intercept) packagepackage2 packagepackage3 packagepackage4
    (Intercept)      0.03861667     -0.03861667     -0.03861667     -0.03861667
    packagepackage2 -0.03861667      0.07723333      0.03861667      0.03861667
    packagepackage3 -0.03861667     [0.03861667]      0.07723333      0.03861667
    packagepackage4 -0.03861667      0.03861667      0.03861667      0.07723333
```

以此类推. 

*(4) 计算边界值*

Scheffe 方法的边界值为:
$$
\xi_\alpha = \sqrt{m_1 \mathcal{F}_{m_1, m_2}(1-\alpha)}
$$
其中, $m_1$ 是参数这里 $\mathbf{a}_i$ 的维度 (即参与比较的组数), $m_2$ 是残差的自由度 (即 $n - p$), $p$ 是模型的参数个数 (即组数 - 1).

从代码上看, 由于我们的 contrast 是设置如 `ctrst1 = c(1,0,0)` 等, 因此说明 $m_1 = 3$ (即参与比较的组数). 而 $m_2$ 可以直接观察 `summary(fit)` 中 Coefficients 下含 (Intercept) 在内的参数个数 (即 4), 因此 我们要用一个服从 $\mathcal{F}_{3,8}(1-\alpha)$ 的F分布分位数来进行检验. 

```r
> summary(fit)

Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
(Intercept)       7.4800     0.1965  38.064 2.49e-10 
packagepackage2  -1.9800     0.2779  -7.125 9.95e-05 
packagepackage3  -0.2200     0.2779  -0.792    0.451    
packagepackage4  -4.1200     0.2779 -14.825 4.22e-07 
```

在 ANOVA table 中也可以直接看到:
```r
> anova(fit)
Analysis of Variance Table

Response: y
          Df Sum Sq Mean Sq F value    Pr(>F)    
package    3 32.873 10.9576  94.584 1.376e-06 ***
Residuals  [8]  0.927  0.1159                      
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
```


*(5) 计算 Rejection Region 或 p-value 给出结论*

相应的拒绝域为:
$$
\mathcal{RR} = \{T_i > \xi_\alpha =  \sqrt{m_1 \mathcal{F}_{m_1, m_2}(1-\alpha) }\}
$$

或者
$$
p\text{-value} = \mathbb{P}(\mathcal{F}_{m_1, m_2} > T_i^2/m_1)
$$

若拒绝原假设, 则说明存在显著差异.

也可以顺便给出其置信区间:
$$
\mathbf{a}^\top \boldsymbol{\beta} \pm \xi_\alpha\cdot \text{s.e.}(\mathbf{a}^\top \boldsymbol{\beta}) \\= 
\mathbf{a}^\top \boldsymbol{\hat\beta} \pm  \sqrt{m_1 \mathcal{F}_{m_1, m_2}(1-\alpha) }\cdot \sqrt{\mathbf{a}^\top \hat\Sigma_{\boldsymbol{\beta}} \mathbf{a}}
$$


## Pairwise Comparisons

### Tukey 方法

Tukey 方法是 post-hoc 的, 但是它的假设是我们只比较两组之间的差异, 即 $\mu_2 - \mu_1$, $\mu_3 - \mu_2$ 等等. 其整体思路基本上和 Scheffe 方法是类似的, 只不过其 critic 边界值发生了变化. 

其原假设为: $\mathcal{H}_0: \mu_i - \mu_j = 0, j \neq i$.

同样还是拟合方程, 根据希望比较对象写出关于 $\beta$ 的 contrast, 并求出其检验统计量:
- 若比较 $\mu_j  - \mu_1 (j \ge 2)$, 则 $T_j = \frac{\hat\beta_j}{\sqrt{\text{Var}(\hat\beta_j)}}$
- 若比较 $\mu_j - \mu_k (j \neq k, j,k \neq 1)$, 则 $T_{jk} = \frac{\hat\beta_j - \hat\beta_k}{\sqrt{\text{Var}(\hat\beta_j - \hat\beta_k)}} = \frac{\hat\beta_j - \hat\beta_k}{\sqrt{\text{Var}(\hat\beta_j) + \text{Var}(\hat\beta_k) - 2\text{Cov}(\hat\beta_j, \hat\beta_k)}}$

Tukey 方法的边界值和对应拒绝域为:
$$
\mathcal{RR} = \{T_{ij} > \frac{q_{a, \text{df}, \alpha}}{\sqrt{2}} \}
$$
- $q_{a, \text{df}, \alpha}$ 是 Tukey HSD 的分位数, 其需要通过查表来获得. 其中: $a$ 是组数, $\text{df}$ 是残差的自由度 (即 $n - p$), $\alpha$ 是显著性水平.

对应的置信区间为:
$$
(\hat\beta_i - \hat\beta_j) \pm \frac{q_{a, \text{df}, \alpha}}{\sqrt{2}} \cdot \sqrt{\text{Var}(\hat\beta_i - \hat\beta_j)}\\
\text{or} ~~
\hat\beta_i \pm \frac{q_{a, \text{df}, \alpha}}{\sqrt{2}} \cdot \sqrt{\text{Var}(\hat\beta_i)}
$$

### Dunnett 方法

当我们明确只想要和基准组进行比较时, 我们可以使用 Dunnett 方法. 其可以给出更精准的边界值.

其原假设为: $\mathcal{H}_0: \mu_i - \mu_1 = 0, i \ge 2$. 因此对应的检验统计量为:
$$
T_i = \frac{\hat\beta_i}{\sqrt{\text{Var}(\hat\beta_i)}}$$

对应的拒绝域为:
$$
\mathcal{RR} = \{T_i > m_j d_{a, \text{df}, \alpha}\}
$$
- $d_{a, \text{df}, \alpha}$ 是 Dunnett 的分位数, 其需要通过查表来获得. 其中: $a$ 是组数, $\text{df}$ 是残差的自由度 (即 $n - p$), $\alpha$ 是显著性水平.
- $m_j$ 用来调节组内人数不平衡问题: $m_j = 1 + 0.07 (1 - \frac{n_j}{n_1})$, 其中 $n_j$ 是第 $j$ 组的样本量, $n_1$ 是基准组的样本量. 若每组样本量相等, 则 $m_j = 1$, 可以忽略.

对应的置信区间为:
$$
\hat\beta_i \pm m_j d_{a, \text{df}, \alpha} \cdot \sqrt{\text{Var}(\hat\beta_i)}
$$

## Bonferroni 方法

Bonferroni 方法是 pre-specified 的, 即我们在实验设计的时候就已经确定了要比较的组. 其基本思想是, 如果我们要进行 $k$ 次比较 (即 $k$ 个假设检验), 并且我们要求一共的显著性水平为 $\alpha$, 则每个比较的显著性水平应调整为 $\sum_{j=1}^k \alpha_j = \alpha$. 平均意义上, 每个假设检验的显著性水平为 $\alpha_j = \frac{\alpha}{k}$.

注意, 这里很明显的体现出了 Pre-specified 和 Post-hoc 的区别. 由于我们在实验设计的时候就已经确定了要比较的组, 因此我们可以明确的说出我们需要比较的次数 $k$ 和每个比较的显著性水平 $\alpha_j$. 反之, 在 Post-hoc 的情况下, 我们并不知道我们要比较的次数 $k$ 和每个比较的显著性水平 $\alpha_j$, 因此我们只能用一个保守的显著性水平 $\alpha$ (类似估计一个充分大的上界) 来确保我们整体的显著性水平为 $\alpha$.

因此在 Bonferroni 方法中, 我们需要首先确定要比较的组数 $k$, 然后进行假设检验, 其全过程和普通的 t 检验是一样的. 只不过我们要将显著性水平调整为 $\alpha_j = \frac{\alpha}{k}$ (即用更严格的显著性水平来进行检验).

下给出一个具体例子:

某食品公司的麦片有四种包装, 其分别具有特点:
|组别|配色|包装方式|
|------|------|------|
|$\mu_1$| 红色|袋装|
|$\mu_2$| 红色|盒装|
|$\mu_3$| 蓝色|袋装|
|$\mu_4$| 蓝色|盒装|

我们希望比较不同包装方式的销量差异. 这里我们有 $a=4$ 组. 这里我们关心如下两种因素的差异:
- 颜色: $\frac{\mu_1 + \mu_2}{2} - \frac{\mu_3 + \mu_4}{2}$
- 包装方式: $\frac{\mu_1 + \mu_3}{2} - \frac{\mu_2 + \mu_4}{2}$

这两个对比是提前指定的, 是典型的 Pre-specified Comparisons. 因此我们可以使用 Bonferroni 方法.

**将 contrast 写成 beta 的形式**

*颜色对比: *

对于 $\mathcal{H}_0: \frac{\mu_1 + \mu_2}{2} - \frac{\mu_3 + \mu_4}{2} = 0$, 可以忽略系数$1/2$, 直接写成:
$$\mathcal{H}_0: (\mu_1 + \mu_2) - (\mu_3 + \mu_4) = 0$$

由于 $\beta_1 = \mu_1$, $\beta_2 = \mu_2 - \mu_1$, $\beta_3 = \mu_3 - \mu_1$, $\beta_4 = \mu_4 - \mu_1$, 因此我们可以将上述的对比写成 beta 的形式:
$$\begin{align*}
(\mu_1 + \mu_2) - (\mu_3 + \mu_4) &=
\beta_1 +(\beta_2 + \beta_1) - (\beta_3 + \beta_1 + \beta_4 + \beta_1)\\
&= \beta_2 - \beta_3 - \beta_4 \\
\Rightarrow \mathbf{a}_1 = (1, -1, -1)^\top\\
\end{align*}$$


*包装对比:*
$$
\begin{align*}
(\mu_1 + \mu_3) - (\mu_2 + \mu_4) &=
\beta_1 + (\beta_3 + \beta_1) - (\beta_2 + \beta_4 + \beta_1)\\
&= \beta_3 - \beta_2 - \beta_4
\\
\Rightarrow \mathbf{a}_2 = (1, -1, -1)^\top\\
\end{align*}
$$

完全按照正常的 t 检验的流程进行检验:
$$
\begin{align*}
T_1 &= \frac{\mathbf{a}_1^\top \boldsymbol{\hat\beta}}{\sqrt{\mathbf{a}_1^\top \hat\Sigma_{\boldsymbol{\beta}} \mathbf{a}_1}}\\
&= \frac{\hat\beta_2 - \hat\beta_3 - \hat\beta_4}{\sqrt{\text{Var}(\hat\beta_2 - \hat\beta_3 - \hat\beta_4)}}\\
&= \cdots \\
&= -6.2456\\
T_2 &= \frac{\mathbf{a}_2^\top \boldsymbol{\hat\beta}}{\sqrt{\mathbf{a}_2^\top \hat\Sigma_{\boldsymbol{\beta}} \mathbf{a}_2}}\\
&= \frac{\hat\beta_3 - \hat\beta_2 - \hat\beta_4}{\sqrt{\text{Var}(\hat\beta_3 - \hat\beta_2 - \hat\beta_4)}}\\
&=\cdots \\
&= -2.1709
\end{align*}
$$

也可以直接求出其p-value:
$$
\begin{align*}
p\text{-value}_1 &= \mathbb{P}(\mathcal{T_{15}} > |T_1|)\\
p\text{-value}_2 &= \mathbb{P}(\mathcal{T_{15}} > |T_2|)
\end{align*}
$$
注意其服从的分布是 t 分布, 自由度为$n - p$.

但是最后进行比较的时候, 由于我们要进行两次比较, 因此我们要将显著性水平调整为 $\alpha_j = \frac{\alpha}{k} = \frac{0.05}{2} = 0.025$. 故对于每个假设检验, 当 $p\text{-value} < 0.025$ 时, 我们拒绝原假设.