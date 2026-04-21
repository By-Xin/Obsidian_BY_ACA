# Nestrov Acceleration Gradient Method for Holder Smooth Convex Optimization

## Problem Setup

考虑如下无约束优化问题:
$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})
$$
其中要求 $f: \mathbb{R}^n \to \mathbb{R}$ 凸且可微. 此外, 假设 $f$ 满足 $M_p$-Holder smooth, 即对于任意 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$, 都满足如下条件:
$$
\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq M_p \|\mathbf{x} - \mathbf{y}\|^{p-1}, \quad p \in [1, 2]
$$
- 当 $p=2$ 时, 就变成了经典的Lipschitz连续梯度条件, 也就是我们熟悉的光滑函数.

此外, 还假设原问题的最优值 $f^* = \min_{\mathbf{x}} f(\mathbf{x})$ 是可达的, 即存在 $\mathbf{x}^*$ 使得 $f(\mathbf{x}^*) = f^*$. 并且 $R_0 = \|\mathbf{x}^{(0)} - \mathbf{x}^*\| < +\infty$, 其中 $\mathbf{x}^{(0)}$ 是算法的初始点.


首先给出 Holder smooth 函数的一个重要性质. 这是我们后续分析的基础.

***Proposition* (Holder Descent)**: 对于满足 $M_p$-Holder smooth 的函数 $f$, 任意 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ 都满足如下不等式:
$$
f(\mathbf{y}) \leq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle + \frac{M_p}{p} \|\mathbf{y} - \mathbf{x}\|^p
$$

- *Proof*. 
  - 根据微积分基本定理及 Holder smooth 的定义, 可以得到如下表达式:
    $$
    \begin{aligned}
    f(\mathbf{y}) - f(\mathbf{x})  - \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle
        &= \int_0^1 \langle \nabla f(\mathbf{x} + t(\mathbf{y} - \mathbf{x})), \mathbf{y} - \mathbf{x} \rangle - \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle     \, \mathrm{d}t \\
        &= \int_0^1 \langle \nabla f(\mathbf{x} + t(\mathbf{y} - \mathbf{x})) - \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle \, \mathrm{d}t \\
        &\leq \int_0^1 \|\nabla f(\mathbf{x} + t(\mathbf{y} - \mathbf{x})) - \nabla f(\mathbf{x})\| \cdot \|\mathbf{y} - \mathbf{x}\| \, \mathrm{d}t \\
        &\leq \int_0^1 M_p \|t(\mathbf{y} - \mathbf{x})\|^{p-1} \cdot \|\mathbf{y} - \mathbf{x}\| \, \mathrm{d}t \\
        &= \int_0^1 M_p t^{p-1} \|\mathbf{y} - \mathbf{x}\|^p \, \mathrm{d}t \\
        &= \frac{M_p}{p} \|\mathbf{y} - \mathbf{x}\|^p
    \end{aligned}
    $$

$\square$

对于表达式 $\frac{M_p}{p} \|\mathbf{y} - \mathbf{x}\|^p$, 这一项是很好处理的二次项. 然而在更一般的 $p \in (1,2)$ 的情况下, 这一项就变成了一个非二次的项, 给标准的 accelerated gradient method 的分析带来了挑战. 因此一个总的目标是争取能够 relax 到如下形式:
$$
\frac{M_p}{p} \|\mathbf{y} - \mathbf{x}\|^p \leq \frac{H_\delta}{2} \|\mathbf{y} - \mathbf{x}\|^2 + \frac{\delta}{2}
$$  
在给出其具体分析之前, 先额外提出如下不等式引理.

***Lemma* (Young's Inequality)**: 对于任意 $t\geq 0$, $s>0$, $p \in (1,2)$, 都满足如下不等式:
$$
t^p \leq \frac{p}{2s} t^2 + \frac{2-p}{2} s^{\frac{p}{2-p}}.
$$

- *Proof*
  - 由 Young's Inequality,  给定对偶指数 $\alpha, \beta > 1$ 满足 $1 / \alpha + 1 / \beta = 1$, 则对于任意 $u,v \geq 0$, 都满足如下不等式:
    $$
    uv \leq \frac{u^\alpha}{\alpha} + \frac{v^\beta}{\beta}.
    $$
  - 在这里, 取 $\alpha = \frac{2}{p}$, $\beta = \frac{2}{2-p}$, 则满足 $1/\alpha + 1/\beta = 1$. 同时取 $u = t^p / s^{p/2}$, $v = s^{p/2}$, 则可以得到如下不等式:
    $$
    t^p = uv \leq \frac{u^\alpha}{\alpha} + \frac{v^\beta}{\beta} = \frac{p}{2s} t^2 + \frac{2-p}{2} s^{\frac{p}{2-p}}.
    $$

$\square$


下面这个定理便使用 Young's Inequality 来尝试将上述的 $p$ norm relax 成一个二次项加上一个常数项.

***Lemma* (Quadratic Relaxation for Holder Smoothness)**: 对于满足 $M_p$-Holder smooth 的函数 $f$, 任意 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$, 都满足如下不等式:
$$
f(\mathbf{y}) \leq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle + \frac{H_\delta}{2} \|\mathbf{y} - \mathbf{x}\|^2 + \frac{\delta}{2},
$$
或本质上:
$$
\frac{M_p}{p} \|\mathbf{y} - \mathbf{x}\|^p \leq \frac{H_\delta}{2} \|\mathbf{y} - \mathbf{x}\|^2 + \frac{\delta}{2},
$$
其中
$$
H_\delta = M_p^{\frac{2}{p}} \left(\frac{2-p}{p\delta}\right)^{\frac{2-p}{p}}.
$$

- *Proof*.
  - 令上述 Young's Inequality 中的 $t = \|\mathbf{y} - \mathbf{x}\|$, 则有:
    $$
    \begin{aligned}
        \|\mathbf{y} - \mathbf{x}\|^p &\leq \frac{p}{2s} \|\mathbf{y} - \mathbf{x}\|^2 + \frac{2-p}{2} s^{\frac{p}{2-p}} \\ 
        \iff \frac{M_p}{p} \|\mathbf{y} - \mathbf{x}\|^p &\leq \frac{M_p}{2s} \|\mathbf{y} - \mathbf{x}\|^2 + \frac{M_p(2-p)}{2p} s^{\frac{p}{2-p}}.
    \end{aligned}
    $$
  - 令 $s = \frac{M_p}{H_\delta}$, 则可以得到如下不等式:
    $$
    \frac{M_p}{p} \|\mathbf{y} - \mathbf{x}\|^p \leq \frac{H_\delta}{2} \|\mathbf{y} - \mathbf{x}\|^2 + \frac{M_p(2-p)}{2p} \left(\frac{M_p}{H_\delta}\right)^{\frac{p}{2-p}}.
    $$

  - 最后再令 
    $$
    \delta = \frac{M_p(2-p)}{p} \left(\frac{M_p}{H_\delta}\right)^{\frac{p}{2-p}},
    $$
    从中可以解出 $H_\delta$ 的表达式:
    $$
    H_\delta = M^{\frac{2}{p}} \left(\frac{2-p}{p\delta}\right)^{\frac{2-p}{p}}.
    $$
$\square$


## Nestrov Acceleration Gradient Method

### Algorithm Description

对于上述的 Holder smooth 函数, 可以使用如下的 Nestrov Acceleration Gradient Method 来进行优化. 其伪代码如下:

```
Algorithm: Hölder-Smooth NAG

Input: x0, p ∈ [1,2), Mp > 0, δ > 0, T
Set:
  H ← ((2-p)/(pδ))^((2-p)/p) * Mp^(2/p)
Initialize:
  y0 ← z0 ← x0
  α1 ← 1
  γ1 ← H

For t = 1 to T:
  xt ← (1-αt) yt-1 + αt zt-1
  zt ← zt-1 - (1/γt) ∇f(xt)
  yt ← (1-αt) yt-1 + αt zt

  If t < T:
    αt+1 ← (-αt^2 + sqrt(αt^4 + 4αt^2)) / 2
    γt+1 ← αt+1 H

Return yT
```

下面借此机会详细讲解一下 NAG 算法本身. 在凸光滑问题中, 一般的收敛速率为 $\mathcal{O}(1/T)$, 但是 NAG 可以将收敛速率提升到 $\mathcal{O}(1/T^2)$. 其核心思想是引入一个 momentum term 来加速梯度下降的过程. 在迭代过程中, 其符合如下三序列更新原则:
$$
\begin{aligned}
\mathbf{x}^{(t)} &= (1-\alpha_t) \mathbf{y}^{(t-1)} + \alpha_t \mathbf{z}^{(t-1)}, \\
\mathbf{z}^{(t)} &= \mathbf{z}^{(t-1)} - \frac{1}{\gamma_t} \nabla f(\mathbf{x}^{(t)}), \\
\mathbf{y}^{(t)} &= (1-\alpha_t) \mathbf{y}^{(t-1)} + \alpha_t \mathbf{z}^{(t)}.
\end{aligned}
$$
其中 $\alpha_t$ 和 $\gamma_t$ 是算法的参数, 其更新规则如下:
$$
\begin{aligned}
& \alpha_1 = 1,\\
& \alpha_{t+1} + \alpha_t^2 \alpha_{t+1} - \alpha_t^2 = 0, ~ \implies \alpha_{t+1} = \frac{-\alpha_t^2 + \sqrt{\alpha_t^4 + 4\alpha_t^2}}{2}, \\
& \gamma_{t+1} = \alpha_{t+1} H_\delta.
\end{aligned}
$$

下面依次理解这三个更新步骤的含义:
- $\mathbf{z}^{(t)}$ 的更新是一个标准的梯度下降步骤, 其步长为 $1/\gamma_t$. 不过这里注意, 更新的序列为 $\mathbf{z}$, 但采取的梯度是 $\nabla f(\mathbf{x}^{(t)})$. 这个点类似于主更新点, 或者说动量点.
- $\mathbf{y}^{(t)}$ 是最终被返回的序列, 其更新是 $\mathbf{y}^{(t-1)}$ 和 $\mathbf{z}^{(t)}$ 的一个 convex combination. 其中 $\alpha_t$ 控制了两者的权重. 这是真正被证明具有下降性质的序列.
- $\mathbf{x}^{(t)}$ 是一个 auxiliary sequence, 类似一个"前瞻点", 相当于在平衡动量点 $\mathbf{z}^{(t-1)}$ 和当前的输出点 $\mathbf{y}^{(t-1)}$ 之间进行一个 convex combination. 这个点是用来计算梯度的.

其一个模拟的示意图如下:

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20260421112154.png)

如果用 $\mathbf{y}^{(t)} - \mathbf{x}^{(t)}$, 则可以得到:
$$
\mathbf{y}^{(t)} - \mathbf{x}^{(t)} = \alpha_t (\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}).
$$
这说明: 从 $\mathbf{x}^{(t)}$ 到 $\mathbf{y}^{(t)}$ 位移, 是由 $\mathbf{z}^{(t)}$ 和 $\mathbf{z}^{(t-1)}$ 之间的差距所驱动的. 这也是为什么 $\mathbf{z}$ 被称为动量点的原因, 因为它在推动 $\mathbf{y}$ 的更新. 换言之, $\mathbf{x}^{(t)}$ 是一个"前瞻点", $\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}$ 是一个"动量项", 这个动量项在推动 $\mathbf{y}^{(t)}$ 向前更新. 这也是 NAG 能够加速收敛的核心机制.

### Algorithm Analysis

下面对算法的机制进行分析. 这里一个核心公式是第二个关于 $\mathbf{z}^{(t)}$ 的更新. 下面这个引理尝试说明, 即使我们是用 $f$ 在 $\mathbf{x}^{(t)}$ 处的梯度来更新 $\mathbf{z}^{(t)}$, 但是我们仍然可以得到一个关于 $\mathbf{z}^{(t)}$ 和任意 $\mathbf{x}$ 之间的关系. 这个关系是通过一个三点不等式来表达的, 因此被称为三点引理.


***Lemma* (Three-Point Lemma)**: 令 $\mathbf{z}^{(t)} = \mathbf{z}^{(t-1)} - \frac{1}{\gamma_t} \nabla f(\mathbf{x}^{(t)})$, 则对于任意 $\mathbf{x} \in \mathbb{R}^n$, 都满足如下不等式:
$$
\langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x} \rangle \leq \frac{\gamma_t}{2}
\Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2\Bigr).
$$

- *Proof*.
  - 由更新规则 $\mathbf{z}^{(t)} = \mathbf{z}^{(t-1)} - \frac{1}{\gamma_t} \nabla f(\mathbf{x}^{(t)})$, 其等价于 $\gamma_t (\mathbf{z}^{(t-1)} - \mathbf{z}^{(t)}) = \nabla f(\mathbf{x}^{(t)})$. 因此可以得到如下表达式:
    $$
    \begin{aligned}
    \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x} \rangle 
        &= \langle \gamma_t (\mathbf{z}^{(t-1)} - \mathbf{z}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x} \rangle \\
        &= \frac{\gamma_t}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2\Bigr).
    \end{aligned}
    $$

$\square$


- 该 Lemma 的 RHS 相当于进行两个层面的刻画.  对于任意一个给定的参考点 $\mathbf{x}$ (当然通常选取为最优解 $\mathbf{x}^\star$):
  - $\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2$ 刻画了更新后的 $\mathbf{z}^{(t)}$ 相比于更新前的 $\mathbf{z}^{(t-1)}$ 更接近 $\mathbf{x}$ 的程度. 
  - $\|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2$ 则刻画了 $\mathbf{z}$ 在两次迭代之间的移动幅度.


下给出其收敛率的分析. 对于满足 $M_p$-Holder smooth 的函数 $f$, 通过上述算法, 在第 $T$ 次迭代后, 可以得到如下的收敛率:
$$
f(\mathbf{y}^{(T)}) - f(\mathbf{x}^\star) \leq \underbrace{\frac{H_\delta R_0^2}{2\Gamma_T}}_{\scriptsize\text{main optimization}} + \underbrace{\frac{\delta}{2\Gamma_T} \sum_{t=1}^T \Gamma_t}_{\scriptsize\text{residual accumulation}},
$$
- 其中 $\Gamma_1 = 1$, $\Gamma_t = \dfrac{\Gamma_{t-1}}{1 - \alpha_t}$. 或可证明等价地,
  $$
  \Gamma_t = \prod_{k=2}^t \frac{1}{1 - \alpha_k} = \frac{1}{\alpha_t^2} \asymp t^2.
  $$
  此外, 
  $$
  H_\delta = M_p^{\frac{2}{p}} \left(\frac{2-p}{p\delta}\right)^{\frac{2-p}{p}},\quad R_0 = \|\mathbf{x}^{(0)} - \mathbf{x}^\star\|.
  $$

- 该定理总体而言, 由于量级上 $\Gamma_T \asymp T^2$, 因此总体而言可以看作是 $\mathcal{O}(1/T^2)$ 的收敛率. 在量级上看, 其大概的形式如下:
  $$
  f(\mathbf{y}^{(T)}) - f(\mathbf{x}^\star) \asymp \frac{H_\delta R_0^2}{T^2} + \delta T,
  $$
  具体而言, 这里分为了两大部分:
    - 第一个部分 $\frac{H_\delta R_0^2}{2\Gamma_T}$ 类似于标准 smooth NAG 中的 $\frac{L R_0^2}{T^2}$ 的形式, 是主要优化的收敛项. 
    - 第二个部分 $\frac{\delta}{2\Gamma_T} \sum_{t=1}^T \Gamma_t$ 则是由于我们实际处理的是一个 Holder smooth 的问题, 因此每一步的优化都相当于在进行一个二次项的 relax, 因此会引入一个 residual error. 这个 residual error 在每一步都会积累, 因此需要通过 $\sum_{t=1}^T \Gamma_t$ 来进行刻画. 



- *Proof*.
  - **首先证明单步的下降性质**.  先直接 claim 一下该步最终结论: 对于第 $t$ 次迭代, 上述的 NAG 算法满足如下单步下降性质:
    $$
    f(\mathbf{y}^{(t)}) \leq  f(\mathbf{x}^{(t)}) + \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{z}^{(t-1)} \rangle + \frac{\alpha_t^2 H_\delta}{2} \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2 + \frac{\delta}{2}.
    $$
    - 理由如下. 由 $\mathbf{y}^{(t)} = \mathbf{x}^{(t)} + \alpha_t (\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)})$ 的更新规则, 将其代入Holder smooth 本身的性质 (1):
      $$
      \begin{aligned}
          f(\mathbf{y}^{(t)}) &\leq f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{y}^{(t)} - \mathbf{x}^{(t)} \rangle + \frac{M_p}{p} \|\mathbf{y}^{(t)} - \mathbf{x}^{(t)}\|^p \quad \text{(Holder smoothness)} \\
          &= f(\mathbf{x}^{(t)}) + \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{z}^{(t-1)} \rangle + \frac{M_p}{p} \|\alpha_t (\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)})\|^p.
      \end{aligned}
      $$
    - 进一步处理这里的 $p$ norm 项, 根据 Young's Inequality, 我们先 claim 最终可以得到如下结果:
      $$
      \frac{M_p}{p} \|\alpha_t (\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)})\|^p \leq \frac{\alpha_t H_\delta}{2} \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2 + \frac{\delta}{2}.
      $$
      - 推导如下. 首先, 根据 Young's Inequality, 直接有:
        $$
        \begin{aligned}
         \frac{M_p}{p} \|\alpha_t (\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)})\|^p &\leq \frac{M_p}{2s} \alpha_t^p \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2 + \frac{M_p(2-p)\alpha_t^p}{2p} s^{\frac{p}{2-p}}.
        \end{aligned}
        $$
      - 令 $s = \frac{M_p \alpha_t^{p-1}}{\alpha_t H_\delta}$. (注意到 NAG 算法在设计时要求 $\mathbf{z}$ 的更新步长 $\gamma_t = \alpha_t H_\delta$, 因此 $s = \frac{M_p \alpha_t^{p-1}}{\gamma_t}$). 代入上式, 可以得到:
        $$
        \begin{aligned}
         \frac{M_p}{p} \|\alpha_t (\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)})\|^p &\leq \frac{\alpha_t^2 H_\delta}{2} \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2 + \frac{M_p(2-p)\alpha_t^p}{2p} \left(\frac{M_p \alpha_t^{p-2}}{H_\delta}\right)^{\frac{p}{2-p}} \\
         &= \frac{\alpha_t H_\delta}{2} \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2 + \frac{2 - p}{2p} M_p^{\frac{2}{2-p}} H_\delta^{-\frac{p}{2-p}} 
        \end{aligned}
        $$
      - 注意到, RHS 的最后一项, 根据最开始 $H_\delta$ 的定义: 
        $$
        H_\delta = M_p^{\frac{2}{p}} \left(\frac{2-p}{p\delta}\right)^{\frac{2-p}{p}} \implies \frac{2 - p}{2p} M_p^{\frac{2}{2-p}} H_\delta^{-\frac{p}{2-p}} = \frac{\delta}{2}.
        $$


  - **接着, 将上述结论中的 $f(\mathbf{x}^{(t)}) + \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{z}^{(t-1)} \rangle$ 进行进一步处理**. 最终我们预期得到的结论为:
    $$
    f(\mathbf{y}^{(t)}) \leq (1-\alpha_t) f(\mathbf{y}^{(t-1)}) + \alpha_t \Bigl( f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle \Bigr) + \frac{\alpha_t^2 H_\delta}{2} \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2 + \frac{\delta}{2}.
    $$
    - 理由如下. 首先, 将 $\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}$ 分解为 $\mathbf{z}^{(t)} - \mathbf{x}^{(t)} + \mathbf{x}^{(t)} - \mathbf{z}^{(t-1)}$, 则有:
      $$
      \begin{aligned}
      \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{z}^{(t-1)} \rangle 
          &= \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), (\mathbf{z}^{(t)} - \mathbf{x}^{(t)}) + (\mathbf{x}^{(t)} - \mathbf{z}^{(t-1)}) \rangle \\
          &= \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle + \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{x}^{(t)} - \mathbf{z}^{(t-1)} \rangle \quad (\dagger).
      \end{aligned}
      $$
      这个式子暂且保留.
    - 另一方面, 根据更新规则 $\mathbf{x}^{(t)} = (1-\alpha_t) \mathbf{y}^{(t-1)} + \alpha_t \mathbf{z}^{(t-1)}$, 可以直接变形得到 
      $$
      \begin{aligned}
      \mathbf{x}^{(t)} - \mathbf{z}^{(t-1)} &= (1-\alpha_t)(\mathbf{y}^{(t-1)} - \mathbf{z}^{(t-1)}),\\
      \iff \alpha_t (\mathbf{x}^{(t)} - \mathbf{z}^{(t-1)}) &= \alpha_t (1-\alpha_t)(\mathbf{y}^{(t-1)} - \mathbf{z}^{(t-1)}),\quad \text{(1)}.
      \end{aligned}
      $$
      此外, 将 $\mathbf{x}^{(t)}$ 的更新规则代入 $\mathbf{y}^{(t-1)} - \mathbf{x}^{(t)}$ 中, 则有
      $$
      \begin{aligned}
      \mathbf{y}^{(t-1)} - \mathbf{x}^{(t)} &= \mathbf{y}^{(t-1)} - ((1-\alpha_t) \mathbf{y}^{(t-1)} + \alpha_t \mathbf{z}^{(t-1)}) = \alpha_t (\mathbf{y}^{(t-1)} - \mathbf{z}^{(t-1)}), \\
      \iff (1-\alpha_t)(\mathbf{y}^{(t-1)} - \mathbf{x}^{(t)}) &= (1-\alpha_t)\alpha_t (\mathbf{y}^{(t-1)} - \mathbf{z}^{(t-1)}) \quad \text{(2)}.
      \end{aligned}
      $$ 
      比较 $(1)$ 和 $(2)$, 其 RHS 相同, 故
      $$
      \alpha_t (\mathbf{x}^{(t)} - \mathbf{z}^{(t-1)}) = (1-\alpha_t)(\mathbf{y}^{(t-1)} - \mathbf{x}^{(t)}).
      $$
      整理可得:
      $$
      \begin{aligned}
      \alpha_t (\mathbf{x}^{(t)} - \mathbf{z}^{(t-1)}) &= (1-\alpha_t)(\mathbf{y}^{(t-1)} - \mathbf{x}^{(t)}) \\
      \implies \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{x}^{(t)} - \mathbf{z}^{(t-1)} \rangle &= (1-\alpha_t) \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{y}^{(t-1)} - \mathbf{x}^{(t)} \rangle.
      \end{aligned}
      $$
      注意, 这个我们得到的内容就是 $(\dagger)$ 中的第二项. 

    - 故将 $(\dagger)$ 中的第二项进行替换, 并左右两侧同时加上 $f(\mathbf{x}^{(t)})$ , 有:
      $$
      \begin{aligned}
        f(\mathbf{x}^{(t)}) + \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{z}^{(t-1)} \rangle 
          &=\boxed{f(\mathbf{x}^{(t)})} + \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle + (1-\alpha_t) \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{y}^{(t-1)} - \mathbf{x}^{(t)} \rangle. \\
      \end{aligned}
      $$
      再拆分 $\boxed{f(\mathbf{x}^{(t)}) =(1-\alpha_t) f(\mathbf{x}^{(t)}) + \alpha_t f(\mathbf{x}^{(t)})}$ , 代入上式 RHS 中, 并合并同类项, 有:
      $$
      f(\mathbf{x}^{(t)}) + \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{z}^{(t-1)} \rangle 
          = (1-\alpha_t) \underbrace{\Bigl(f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{y}^{(t-1)} - \mathbf{x}^{(t)} \rangle\Bigr)}_{\text{convexity}}  + \alpha_t \Bigl(f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)}\rangle\Bigr).
      $$



    - 观察上面式子, 下利用 $f$ 之凸性对其进行进一步处理. 由凸函数的一阶条件:
      $$
      \begin{aligned}
        & \underbrace{f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{y}^{(t-1)} - \mathbf{x}^{(t)} \rangle}_{\text{convexity}} \leq f(\mathbf{y}^{(t-1)}), \\
      \end{aligned}
      $$
      可替换整理为
      $$
      \underline{f(\mathbf{x}^{(t)}) + \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{z}^{(t-1)} \rangle}
          \leq (1-\alpha_t) f(\mathbf{y}^{(t-1)}) + \alpha_t \Bigl(f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)}\rangle\Bigr).
      $$

    - Recall, 我们在第一步已经证明出来:     $f(\mathbf{y}^{(t)}) \leq  \underline{f(\mathbf{x}^{(t)}) + \alpha_t \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{z}^{(t-1)} \rangle} + \frac{\alpha_t^2 H_\delta}{2} \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2 + \frac{\delta}{2}.$ 故将上述的结论代入, 可以得到如下单步下降性质:
      $$
      f(\mathbf{y}^{(t)}) \leq (1-\alpha_t) f(\mathbf{y}^{(t-1)}) + \alpha_t \Bigl( f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle \Bigr) + \frac{\alpha_t^2 H_\delta}{2} \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2 + \frac{\delta}{2}.
      $$
      

  - **接着, 对于上述结论中的 $f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle$ 进行进一步处理**. 其最终预期得到的结论为:
    $$
    f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle \leq f(\mathbf{x}) + \frac{\alpha_t H_\delta}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2\Bigr).
    $$
    其中 $\mathbf{x}$ 是任意的参考点, 当然通常选取为最优解 $\mathbf{x}^\star$.

  
    - 理由如下.  注意到, 我们试图处理的 $f(\mathbf{x}^{(t)}) +\langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle$ 可以通过引入一个任意的参考点 $\mathbf{x}$ 来进行分解 $\langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle = \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x} \rangle + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{x} - \mathbf{x}^{(t)} \rangle$. 故:
      $$
      \begin{aligned}
        f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle 
          = \underbrace{f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{x} - \mathbf{x}^{(t)} \rangle}_{\text{from convexity}} +\underbrace{\langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x} \rangle}_{\text{from Three-Point Lemma}}.
      \end{aligned}
      $$
      
    - Recall, 由 $\mathbf{z}^{(t)} = \mathbf{z}^{(t-1)} - \frac{1}{\gamma_t} \nabla f(\mathbf{x}^{(t)})$ 的更新规则, 可得 *Three-Point Lemma*:
      $$
      \begin{aligned}
        \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x} \rangle&\leq \frac{\gamma_t}{2}
      \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2\Bigr) \\
      &= \frac{\alpha_t H_\delta}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2\Bigr).
      \end{aligned}
      $$
      其中, 第二个等式是因为 $\gamma_t = \alpha_t H_\delta$ 的更新规则.

    - 另一方面, 由 $f$ 的凸性, 可以得到:
      $$
      f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{x} - \mathbf{x}^{(t)} \rangle \leq f(\mathbf{x}).
      $$

    - 将上述两部分的结论代入, 可以得到如下的单步下降性质:
      $$
      f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle \leq \underbrace{f(\mathbf{x})}_{\text{from convexity}} + \underbrace{\frac{\alpha_t H_\delta}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2\Bigr)}_{\text{from Three-Point Lemma}}.
      $$


  - **然后我们将上述两步的结果进行合并整理, 我们预期得到如下性质**
    $$
    f(\mathbf{y}^{(t)}) \leq (1-\alpha_t) f(\mathbf{y}^{(t-1)}) + \alpha_t f(\mathbf{x}) + \frac{\alpha_t^2 H_\delta}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2\Bigr) + \frac{\delta}{2}.
    $$
    - 理由如下. 由第二步的结论, 我们有:
        $$
        f(\mathbf{y}^{(t)}) \leq (1-\alpha_t) f(\mathbf{y}^{(t-1)}) + \alpha_t \boxed{f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle} + \frac{\alpha_t^2 H_\delta}{2} \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2 + \frac{\delta}{2}.
        $$

    - 而注意到, 在第三步, 我们有
      $$
      \boxed{f(\mathbf{x}^{(t)}) + \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{z}^{(t)} - \mathbf{x}^{(t)} \rangle} \leq f(\mathbf{x}) + \frac{\alpha_t H_\delta}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2\Bigr).
      $$

    - 故将第三步的不等式代入第二步, 有:
      $$
      \begin{aligned}
      f(\mathbf{y}^{(t)}) &\leq (1-\alpha_t) f(\mathbf{y}^{(t-1)}) + \alpha_t \Bigl(f(\mathbf{x}) + \frac{\alpha_t H_\delta}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2\Bigr)\Bigr) + \frac{\alpha_t^2 H_\delta}{2} \|\mathbf{z}^{(t)} - \mathbf{z}^{(t-1)}\|^2 + \frac{\delta}{2} \\
      &= (1-\alpha_t) f(\mathbf{y}^{(t-1)}) + \alpha_t f(\mathbf{x}) + \frac{\alpha_t^2 H_\delta}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2\Bigr) + \frac{\delta}{2}.
      \end{aligned}
      $$

    - **最后, 将上述结论进行 telescoping 的累加处理, 就可以得到最终的收敛率**.  
      - 整理一下, 目前我们为止我们已经求出了一个比较 acceptable 的单步误差. 其形式大概是说, 第 $t$ 步的误差可以表示为前一步的误差, 加上一些关于 $\mathbf{z}$ 的距离项, 以及一个 residual error. 然而我们要考虑的是在一共进行 $T$ 次迭代之后的累积误差. 并且, 注意到这里的误差并不是给出的绝对误差, 而是相对上次迭代的迭代误差. 因此我们要对其进行一个 telescoping 的累加处理. 而在累加的过程中, 要进行一些特殊的处理, 确保中间的部分能够被有效的抵消掉. 
      - 我们在上一步已经得到 
        $$
        f(\mathbf{y}^{(t)}) \leq (1-\alpha_t) f(\mathbf{y}^{(t-1)}) + \alpha_t f(\mathbf{x}) + \frac{\alpha_t^2 H_\delta}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2\Bigr) + \frac{\delta}{2} \quad (\blacklozenge)
        $$
        我们期望将其整理为:
        $$
        \Gamma_t (f(\mathbf{y}^{(t)}) - f(\mathbf{x})) \leq \Gamma_{t-1} (f(\mathbf{y}^{(t-1)}) - f(\mathbf{x})) + \frac{\alpha_t^2 H_\delta \Gamma_t}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2\Bigr) + \frac{\delta \Gamma_t}{2}.
        $$
        并将其从 $t=1$ 累加到 $t=T$, 最终得到:
        $$
        \Gamma_T (f(\mathbf{y}^{(T)}) - f(\mathbf{x})) \leq \frac{H_\delta R_0^2}{2} + \frac{\delta}{2} \sum_{t=1}^T \Gamma_t.
        $$

      - 理由如下. 首先, 将 $(\blacklozenge)$ 中左右两侧同时减去 $f(\mathbf{x})$, 可以得到:
        $$
        \begin{aligned}
        f(\mathbf{y}^{(t)}) - f(\mathbf{x}) &\leq (1-\alpha_t) (f(\mathbf{y}^{(t-1)}) - f(\mathbf{x})) + \frac{\alpha_t^2 H_\delta}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2\Bigr) + \frac{\delta}{2}, \quad (1).
        \end{aligned}
        $$

      - 接着, 定义 $\Gamma_1 = 1, \Gamma_t = \frac{\Gamma_{t-1}}{1-\alpha_t}$. 故在 $(1)$ 中左右两侧同时乘以 $\Gamma_t$, 可以得到:
        $$
        \begin{aligned}
        \Gamma_t (f(\mathbf{y}^{(t)}) - f(\mathbf{x})) &\leq \Gamma_t (1-\alpha_t) (f(\mathbf{y}^{(t-1)}) - f(\mathbf{x})) + \frac{\alpha_t^2 H_\delta \Gamma_t}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2\Bigr) + \frac{\delta \Gamma_t}{2} \\
        &= \Gamma_{t-1} (f(\mathbf{y}^{(t-1)}) - f(\mathbf{x})) + \frac{\alpha_t^2 H_\delta \Gamma_t}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2\Bigr) + \frac{\delta \Gamma_t}{2} \qquad (2).
        \end{aligned} 
        $$

      - 注意到, $(2)$ 中包含一个系数 $\Gamma_t \alpha_t^2$. 事实上, $\Gamma_t \alpha_t^2 \equiv 1$. 理由如下. 回顾 $\alpha_t$ 的递推关系: 
        $$
        \alpha_{t+1}^2 + \alpha_{t+1} \alpha_t^2 - \alpha_t^2 = 0 \iff \alpha_{t+1}^2 = (1-\alpha_{t+1}) \alpha_t^2 \quad (\star).
        $$
        可以通过归纳法证明 $\Gamma_t \alpha_t^2 \equiv 1$. 首先, 当 $t=1$ 时, $\Gamma_1 \alpha_1^2 = 1$ (由于初始化 $\alpha_1 = 1$). 假设当 $t=k$ 时, $\Gamma_k \alpha_k^2 = 1$. 则当 $t=k+1$ 时, 
        $$
        \begin{aligned}
        \Gamma_{k+1} \alpha_{k+1}^2 &= \frac{\Gamma_k}{1-\alpha_{k+1}} \alpha_{k+1}^2 \stackrel{(\star)}{=} \frac{\Gamma_k}{1-\alpha_{k+1}} (1-\alpha_{k+1}) \alpha_k^2 = \Gamma_k \alpha_k^2 = 1.
        \end{aligned}
        $$
        故证毕. 因此 $(2)$ 可以进一步简化为:
        $$
        \Gamma_t (f(\mathbf{y}^{(t)}) - f(\mathbf{x})) - \Gamma_{t-1} (f(\mathbf{y}^{(t-1)}) - f(\mathbf{x})) \leq \frac{H_\delta}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2\Bigr) + \frac{\delta \Gamma_t}{2} \quad (3)
        $$

      - 最后, 将 $(3)$ 从 $t=1$ 累加到 $t=T$, 可以得到:
        $$
        \begin{aligned}
        \sum \text{LHS} &= \sum_{t=1}^T \Gamma_t (f(\mathbf{y}^{(t)}) - f(\mathbf{x})) = \Gamma_T (f(\mathbf{y}^{(T)}) - f(\mathbf{x})) - \Gamma_0 (f(\mathbf{y}^{(0)}) - f(\mathbf{x})) = \Gamma_T (f(\mathbf{y}^{(T)}) - f(\mathbf{x})), \\
        \end{aligned} 
        $$
        其中规定 $\Gamma_0 = 0$. 而对于 RHS, 则有:
        $$
        \begin{aligned}
        \sum \text{RHS} &= \sum_{t=1}^T \Bigl(\frac{H_\delta}{2} \Bigl(\|\mathbf{z}^{(t-1)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(t)} - \mathbf{x}\|^2\Bigr) + \frac{\delta \Gamma_t}{2}\Bigr) \\
        &= \frac{H_\delta}{2} \Bigl(\|\mathbf{z}^{(0)} - \mathbf{x}\|^2 - \|\mathbf{z}^{(T)} - \mathbf{x}\|^2\Bigr) + \frac{\delta}{2} \sum_{t=1}^T \Gamma_t \\
        &\leq \frac{H_\delta R_0^2}{2} + \frac{\delta}{2} \sum_{t=1}^T \Gamma_t.
        \end{aligned}
        $$
        故整体的不等式为:
        $$
        \Gamma_T (f(\mathbf{y}^{(T)}) - f(\mathbf{x})) \leq \frac{H_\delta R_0^2}{2} + \frac{\delta}{2} \sum_{t=1}^T \Gamma_t.
        $$



      - 最终, 由于 $\mathbf{x}$ 是任意的参考点, 因此我们可以将其选取为最优解 $\mathbf{x}^\star$, 从而得到最终的收敛率:
        $$
        \Gamma_T (f(\mathbf{y}^{(T)}) - f(\mathbf{x}^\star)) \leq \frac{H_\delta R_0^2}{2} + \frac{\delta}{2} \sum_{t=1}^T \Gamma_t.
        $$
        两侧同时除以 $\Gamma_T$, 就可以得到最终的收敛率
        $$
        f(\mathbf{y}^{(T)}) - f(\mathbf{x}^\star) \leq \frac{H_\delta R_0^2}{2\Gamma_T} + \frac{\delta}{2\Gamma_T} \sum_{t=1}^T \Gamma_t.
        $$

$\square !!!!!!!!!!!!!$
