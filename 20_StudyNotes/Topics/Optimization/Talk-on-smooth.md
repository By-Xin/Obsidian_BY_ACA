# Smoothness

## Smoothing for weakly convex optimization

### Problem Settings: Weakly Convex, Non-Smooth

***Definition* ([[Weakly Convex]])** 一个函数 $g: \mathbb{R}^n \to \mathbb{R}$ 是弱凸的 (weakly convex) 若存在常数 $\lambda > 0$, 使得:
$$
g(x) + \frac{\lambda}{2}\|x\|_2^2
$$
是凸函数.

- 直观上, 弱凸函数允许有轻微的凹陷, 但凹陷程度弱于二次函数. 其中 $\lambda$ 越大, 凹陷程度越强.

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202603021104362.png)

可以证明, 对于 $\phi(x):=h(c(x))$, 若 $h: \mathbb{R}^m \to \mathbb{R}$ 是凸且 $L$-[[Lipschitz Continuous]]; $c: \mathbb{R}^d \to \mathbb{R}^m$ 是 [[$C^1$]] 光滑且其 Jacobian 是 $\beta$-[[Lipschitz Continuous]], 则 $\phi$ 是 $L\beta$-weakly convex 的.

---

许多问题都符合这样的结构. 例如:

- **Robust Phase Retrieval**:
  $$\min_{\mathbf{x} \in \mathbb{R}^n} \frac{1}{m} \sum_{i=1}^m |\langle \mathbf{a}_i, \mathbf{x}\rangle^2 - b_i|$$
  - 对于某种线性信号 $\langle \mathbf{a}_i, \mathbf{x}\rangle$, 我们无法观测其直接的取值, 而只能记录其强度 (平方值) $b_i$ (例如 X-ray 成像). 我们希望通过 $m$ 次观测, 恢复出原始信号 $\mathbf{x}$.
  
  - 如下图是一个二维的例子. 其中 $a_1 = (1/\sqrt{2}, 1/\sqrt{2})^\top$, $a_2 = (1/\sqrt{2}, -1/\sqrt{2})^\top$, $b_1 = 1$, $b_2 = 0.2$.
    ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202603021128768.png)

  - 其非凸性较易说明. 记 $\mathbf{z} = \langle \mathbf{a}_i, \mathbf{x}\rangle$, 则对于 $\mathbf{z}_1 = \sqrt{b_1}$, $\mathbf{z}_2 = -\sqrt{b_2}$, 均有 $|\mathbf{z}_i^2 - b_i| = 0$. 然而对于二者中点 $\mathbf{z} = 0$, 有 $|\mathbf{z}^2 - b| = |b_1 - b_2| > 0$. 因此该问题非凸.

  - 下说明其弱凸性. 记 $g_i(\mathbf{x}) = |\langle \mathbf{a}_i, \mathbf{x}\rangle^2 - b_i|$, 其中 $h_i(\mathbf{x}) = \langle \mathbf{a}_i, \mathbf{x}\rangle^2$ 是光滑函数, $\phi_i(t) = |t_i - b_i|$ 是非光滑但凸函数.
    -  对于 $h_i$, 其 Hessian 为 $\nabla^2 h_i(\mathbf{x}) = 2 \mathbf{a}_i \mathbf{a}_i^\top$, 为半正定, 故为凸. 且其特征值为 $0$ ($n-1$ 重) 和 $2\|\mathbf{a}_i\|_2^2$ (1 重). 
    -  分析可知, 对于 $\phi_i(t)$, 在 $t_i \ge b_i$ 的区域, $\phi_i(t)$ 为凸函数; 在 $t_i \le b_i$ 的区域, $\phi_i(t)$ 为凹函数. 在该区域, $-\nabla^2 h_i(\mathbf{x}) = -2 \mathbf{a}_i \mathbf{a}_i^\top$ 最小特征值为 $-2\|\mathbf{a}_i\|_2^2$. 
    -  因此, 故选取 $\lambda = \dfrac{2}{m}\sum_{i=1}^m \|\mathbf{a}_i\|_2^2$ 时, $g_i(\mathbf{x}) + \frac{\lambda}{2}\|x\|_2^2$ 为凸函数.

- **Blind Deconvolution**:
  $$\min_{\mathbf{x}, \mathbf{y} \in \mathbb{R}^n} \frac{1}{m} \sum_{i=1}^m |\langle \mathbf{u}_i, \mathbf{x}\rangle \langle \mathbf{v}_i, \mathbf{y}\rangle - b_i|$$


---

###  Moreau Envelope Smoothing

***Definition* (Moreau Envelope)** 对于给定的 $\rho$-[[Weakly Convex]]函数 $f(\mathbf{x})$, 其 Moreau Envelope 定义为:
$$
f^{\beta}(\mathbf{x}) = \inf_{\mathbf{z}} \left\{ f(\mathbf{z}) + \frac{1}{2\beta} \|\mathbf{z} - \mathbf{x}\|_2^2 \right\}
$$
- 其中 $\beta \in (0, 1/\rho)$ 是 proximal term 的权重. $\beta$ 越大, 则允许的距离越大.

其对应的 Proximal Operator 定义为:
$$
\hat{x}:=\text{prox}_{\beta f}(\mathbf{x}) = \arg\min_{\mathbf{z}} \left\{ f(\mathbf{z}) + \frac{1}{2\beta} \|\mathbf{z} - \mathbf{x}\|_2^2 \right\}
$$

对于 Moreau Envelope, 有如下性质:

1. $f^{\beta}(\mathbf{x})$ 是 $f(\mathbf{x})$ 的下界: $f^{\beta}(\mathbf{x}) \leq f(\mathbf{x}), ~\forall \mathbf{x}$.

2. 当 $f(\mathbf{x})$ 是 $\rho$-weakly convex, 且 $\beta \in (0, 1/\rho)$, 则 $f^{\beta}(\mathbf{x})$ 是 $1/\beta$-smooth 的.
   - $f^{\beta}(\mathbf{x})$ 是可微的, 其梯度为:
     $$
     \boxed{\nabla f^{\beta}(\mathbf{x}) =  \frac{1}{\beta} (\mathbf{x} - \text{prox}_{\beta f}(\mathbf{x}))} 
     $$
     - 直观上, Moreau Envelope 的梯度为从其 proximal 指向当前点的方向, 步长为 $1/\beta$
   - 由上述梯度, 又可立刻推得:
      $$
      \|\mathbf{x} - \hat{\mathbf{x}}\| = \|\mathbf{x} - \text{prox}_{\beta f}(\mathbf{x})\| = \beta \|\nabla f^{\beta}(\mathbf{x})\|
      $$
      - 即该 envelope 的梯度小等价于当前点与 proximal 点之间的距离小.
   - $f^{\beta}(\mathbf{x})$ 是 $1/\beta$-smooth 的, 即:
     $$
     \|\nabla f^{\beta}(\mathbf{x}) - \nabla f^{\beta}(\mathbf{y})\| \leq \frac{1}{\beta} \|\mathbf{x} - \mathbf{y}\|, ~\forall \mathbf{x}, \mathbf{y}.
     $$

3. 近似 Stationary
   $$
   \boxed{\text{dist}(0, \partial f(\hat{\mathbf{x}})) \leq \| \nabla f^{\beta}(\mathbf{x})\|}
   $$
   - 在 LHS, 我们通过衡量 $0$ 到次梯度集合的距离来衡量 stationary 的情况. 而这个距离又能被 Moreau Envelope 的梯度所控制.
   - 这说明, 我们在 $f^{\beta}(\mathbf{x})$ 上进行寻找 Stationary 优化的过程, 也能传到到作用在原始问题上的梯度上. 我们可以用连续函数来评估原先的非光滑函数.
   

其直观理解为: 既然 $f$ 是一个非光滑的函数, 不如在当前点 $\mathbf{x}$ 附近进行一个带有二次函数惩罚的局部最小化, 通过一个光滑的包络来消除非光滑性. 