# $F_\mu$ 光滑系数选择讨论

给定原函数 $F(\mathbf{x}) = \frac{1}{p} \|A\mathbf{x}-\mathbf{b}\|_p^p$, 其 max-conjugate form 为 $F(\mathbf{x}) = \max_{\mathbf{y} \in \mathbb{R}^m} \langle A\mathbf{x}-\mathbf{b}, \mathbf{y} \rangle - \frac{1}{q} \|\mathbf{y}\|_q^q$, 其中 $\frac{1}{p} + \frac{1}{q} = 1$, $p\in (1,2)$, $q\in (2, \infty)$.

并记唯一 Maximizer 为 $\mathbf{y}^\star(\mathbf{x}) = \operatorname{sign}(A\mathbf{x}-\mathbf{b}) \odot |A\mathbf{x}-\mathbf{b}|^{p-1}$.

考虑一个 general 的光滑策略:
$$
F_{\mu, s}(\mathbf{x}) = \max_{\mathbf{y} \in \mathbb{R}^m}  \quad \langle A\mathbf{x}-\mathbf{b}, \mathbf{y} \rangle - \frac{1}{q} \|\mathbf{y}\|_q^q - \frac{\mu}{s} \|\mathbf{y}\|_s^s, \quad 1<s<2.
$$

其 Maximizer $\mathbf{y}^\star_{\mu, s}(\mathbf{x})$ 满足一阶条件:
$$
A\mathbf{x}-\mathbf{b} = \nabla \left( \frac{1}{q} \|\mathbf{y}\|_q^q + \frac{\mu}{s} \|\mathbf{y}\|_s^s \right) = \mathbf{y} \odot |\mathbf{y}|^{q-2} + \mu \mathbf{y} \odot |\mathbf{y}|^{s-2}.
$$

---

因此关键是分析 $\Phi_{\mu,s}(\mathbf{y}) := \mathbf{y} \odot |\mathbf{y}|^{q-2} + \mu \mathbf{y} \odot |\mathbf{y}|^{s-2}$ 的性质. 

我们从标量情况出发. 考虑
$$
\psi_{\mu, s}(y) = y|y|^{q-2} + \mu y |y|^{s-2}.
$$
对 $y \neq 0$, 有
$$
\psi'_{\mu, s}(y) = (q-1)|y|^{q-2} + \mu (s-1) |y|^{s-2} 
$$

- 注意到, 当 $|y| \to 0$, $\psi'_{\mu, s}(y) \approx \mu (s-1) |y|^{s-2} \to +\infty$. 当 $|y| \to \infty$, $\psi'_{\mu, s}(y) \approx (q-1)|y|^{q-2} \to +\infty$. 由因为 $\psi'$ 是连续函数, 固定存在一个最小值, 记为:
    $$
    \kappa_{\mu, s} := \inf_{u>0} \psi'_{\mu, s}(u)
    $$
    
- 下求解这个最小值点的具体表达形式. 注意到该最小值点 $u_\star>0$ 满足
    $$
    \psi''_{\mu, s}(u_\star) = (q-1)(q-2)(u_\star)^{q-3} + \mu (s-1)(s-2) (u_\star)^{s-3} = 0.
    $$
    即
    $$
    (u_\star)^{q-s} = \mu \frac{(s-1)(2-s)}{(q-1)(q-2)}.
    $$
    再代回 $\psi'_{\mu, s}$ 的表达式, 可得
    $$
    \begin{aligned}
    \kappa_{\mu, s} &= \mu^{\frac{q-2}{q-s}} \left[(q-1)\left(\frac{(s-1)(2-s)}{(q-1)(q-2)}\right)^{\frac{q-2}{q-s}} + (s-1) \left(\frac{(s-1)(2-s)}{(q-1)(q-2)}\right)^{\frac{s-2}{q-s}}\right] := \mu^{\frac{q-2}{q-s}} C_{q,s}.
    \end{aligned}
    $$
    即
    $$
    \kappa_{\mu, s} \asymp \mu^{\frac{q-2}{q-s}}.
    $$
    - 注意到, 当 $s = p$ 时, 即有
        $$
        \kappa_{\mu, p} \asymp \mu^{\frac{1}{p}}.
        $$
- 因此, 几乎处处有 $\psi'_{\mu, s}(y) \geq \kappa_{\mu, s}$. 若不妨设 $a>b$, 则
    $$
    \psi_{\mu, s}(a) - \psi_{\mu, s}(b) = \int_b^a \psi'_{\mu, s}(y) dy \geq \int_b^a \kappa_{\mu, s} dy = \kappa_{\mu, s} (a-b).
    $$
    即
    $$
    (\psi_{\mu, s}(a) - \psi_{\mu, s}(b)) (a-b) \geq \kappa_{\mu, s} |a-b|^2.
    $$

---

进一步推高到向量. 由于 $\Phi_{\mu, s}$ 是逐分量作用的, 故对于任意的 $\mathbf{y}_1, \mathbf{y}_2 \in \mathbb{R}^m$, 都有
$$
\begin{aligned}
\langle \Phi_{\mu, s}(\mathbf{y}_1) - \Phi_{\mu, s}(\mathbf{y}_2), \mathbf{y}_1 - \mathbf{y}_2 \rangle  &= \sum_{i=1}^m (\psi_{\mu, s}(\mathbf{y}_{1,i}) - \psi_{\mu, s}(\mathbf{y}_{2,i})) (\mathbf{y}_{1,i} - \mathbf{y}_{2,i}) \\
&\geq \kappa_{\mu, s} \sum_{i=1}^m |\mathbf{y}_{1,i} - \mathbf{y}_{2,i}|^2 \\ 
&= \kappa_{\mu, s} \|\mathbf{y}_1 - \mathbf{y}_2\|_2^2.
\end{aligned}
$$

可以看到, 若令 $s=2$, 则 $\kappa_{\mu, 2} \asymp \mu^{\frac{q-2}{q-2}} = \mu$, 而 $\Phi_{\mu, 2}(\mathbf{y}) = \nabla \left( \frac{1}{q} \|\mathbf{y}\|_q^q \right) + \mu \mathbf{y}$, 故:
$$
\begin{aligned}
\mu \| \Delta \mathbf{y}\|_2^2 &\leq \langle \nabla \mathbb{y}, \mathbf{A} \Delta \mathbf{x} \rangle 
\end{aligned}
$$

---