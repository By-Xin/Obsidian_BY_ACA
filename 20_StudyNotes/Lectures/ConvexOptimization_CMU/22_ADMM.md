# Alternating Direction Method of Multipliers (ADMM)

>[!quote]
>
> - Lecture Reference: 
>   - <https://www.stat.cmu.edu/~ryantibs/convexopt-F18/>

## Recap

在 Dual Decomposition 中我们从对偶问题出发给出了基本的 ADMM 的方法说明. 这里进行回顾. 

考虑标准性问题形式:
$$
\min_{\mathbf{x} \in \mathbb{R}^n, \mathbf{z} \in \mathbb{R}^m} f(\mathbf{x}) + g(\mathbf{z}) \quad \text{s.t.} \quad A\mathbf{x} + B\mathbf{z} = \mathbf{c}\in \mathbb{R}^q.
$$

可以对该问题写出 Augmented Lagrangian:
$$
L_\rho(\mathbf{x}, \mathbf{z}, \mathbf{u}) = f(\mathbf{x}) + g(\mathbf{z}) + \mathbf{u}^\top (A\mathbf{x} + B\mathbf{z} - \mathbf{c}) + \frac{\rho}{2}\|A\mathbf{x} + B\mathbf{z} - \mathbf{c}\|_2^2
$$
则对应的 ADMM 的迭代更新为:
$$
\begin{aligned}
\mathbf{x}^{k} &= \arg\min_{\mathbf{x}}  \left\{f(\mathbf{x}) + \frac{\rho}{2} \|A\mathbf{x} + B\mathbf{z}^{k-1} - \mathbf{c} + \mathbf{u}^{k-1}\|_2^2 \right\}\\
\mathbf{z}^{k} &= \arg\min_{\mathbf{z}}  \left\{g(\mathbf{z}) + \frac{\rho}{2} \|A\mathbf{x}^{k} + B\mathbf{z} - \mathbf{c} + \mathbf{u}^{k-1}\|_2^2 \right\}\\
\mathbf{u}^{k} &= \mathbf{u}^{k-1} + \rho \cdot (A\mathbf{x}^{k} + B\mathbf{z}^{k} - \mathbf{c})
\end{aligned}
$$

若进一步 reparameterize $\mathbf{w} = \mathbf{u}/\rho$ 则有:
$$
L_\rho(\mathbf{x}, \mathbf{z}, \mathbf{w}) = f(\mathbf{x}) + g(\mathbf{z}) + \frac{\rho}{2}\|A\mathbf{x} + B\mathbf{z} - \mathbf{c} + \mathbf{w}\|_2^2 - \frac{\rho}{2}\|\mathbf{w}\|_2^2
$$
则对应的 ADMM 的迭代更新为:
$$
\begin{aligned}
\mathbf{x}^{k} &= \arg\min_{\mathbf{x}}  \left\{f(\mathbf{x}) + \frac{\rho}{2} \|A\mathbf{x + B\mathbf{z}^{k-1} - \mathbf{c} + \mathbf{w}^{k-1}}\|_2^2 \right\}\\
\mathbf{z}^{k} &= \arg\min_{\mathbf{z}}  \left\{g(\mathbf{z}) + \frac{\rho}{2} \|A\mathbf{x}^{k} + B\mathbf{z} - \mathbf{c} + \mathbf{w}^{k-1}\|_2^2 \right\}\\
\mathbf{w}^{k} &= \mathbf{w}^{k-1} + (A\mathbf{x}^{k} + B\mathbf{z}^{k} - \mathbf{c})
\end{aligned}
$$
其中 $\mathbf{w}^{k}$ 可以看作是对历史残差的累积:
$$
\mathbf{w}^{k} = \mathbf{w}^{0} + \sum_{i=1}^{k} (A\mathbf{x}^{i} + B\mathbf{z}^{i} - \mathbf{c}).
$$

总体而言 ADMM 的收敛特性和其他一阶算法类似. 其在 modest 的条件下有如下收敛特性:
- Residual Convergence: $\mathbf{r}^{k} = A\mathbf{x}^{k} + B\mathbf{z}^{k} - \mathbf{c} \to 0$. 随着迭代次数的增加, feasibility 会被逐渐满足.
- Objective Convergence: $f(\mathbf{x}^{k}) + g(\mathbf{z}^{k}) \to p^\star$. 随着迭代次数的增加,, objective value 会逐渐收敛到最优值.
- Dual Variable Convergence: $\mathbf{u}^{k} \to \mathbf{u}^\star$. 随着迭代次数的增加, dual variable 会逐渐收敛到最优值.

不过注意这里无法自动给出 primal iterate convergence, 也就是 $\mathbf{x}^{k} \to \mathbf{x}^\star$ 和 $\mathbf{z}^{k} \to \mathbf{z}^\star$, 这需要其他额外的例如强凸性等条件来保证.

## Discussions

### Connection to Proximal Operators

这里限定为无约束的 ADMM 问题:
$$
\min_{\mathbf{x} \in \mathbb{R}^n, \mathbf{z} \in \mathbb{R}^m} f(\mathbf{x}) + g(\mathbf{z}) \quad \text{s.t.} \quad \mathbf{x} = \mathbf{z}.
$$
即标准形式中对应 $A = I, B = -I, \mathbf{c} = 0$. 

回顾, proxy operator 的定义为:
$$
\operatorname{prox}_{f,t}(\mathbf{v}) = \arg\min_{\mathbf{x}} \left\{f(\mathbf{x}) + \frac{1}{2t}\|\mathbf{x} - \mathbf{v}\|_2^2\right\}
$$

因此, 若对照 scaled ADMM 的更新公式, 取 $t = 1/\rho$ , 则可以看出 ADMM 的更新步骤可以看作是对 $f$ 和 $g$ 的 proximal operator 的调用, 然后再对结果进行修正. 
$$
\begin{aligned}
\mathbf{x}^{k} &= \operatorname{prox}_{f, 1/\rho}(\mathbf{z}^{k-1} - \mathbf{w}^{k-1}) = \arg\min_{\mathbf{x}} \left\{f(\mathbf{x}) + \frac{\rho}{2}\|\mathbf{x} - (\mathbf{z}^{k-1} - \mathbf{w}^{k-1})\|_2^2\right\}
\\
\mathbf{z}^{k} &= \operatorname{prox}_{g, 1/\rho}(\mathbf{x}^{k} + \mathbf{w}^{k-1}) = \arg\min_{\mathbf{z}} \left\{g(\mathbf{z}) + \frac{\rho}{2}\|\mathbf{z} - (\mathbf{x}^{k} + \mathbf{w}^{k-1})\|_2^2\right\} \\
\mathbf{w}^{k} &= \mathbf{w}^{k-1} + (\mathbf{x}^{k} - \mathbf{z}^{k})
\end{aligned}
$$

### Practical Considerations

在实践当中, ADMM 有时会遇到如下问题.
1. ADMM 通常会快速收敛到一个还不错的解, 但是在追求高精度时会变慢;
2. ADMM 的收敛速度对 $\rho$ 的选择非常敏感, 需要进行调参;
3. 对同一个原问题采用不同的分解方式, 会得到不同的 ADMM 算法, 其收敛速度也会不同.

## Consensus  ADMM

## Special Decompositions