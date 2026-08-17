---
aliases: [变量替换, Change of Variables, 概率分布变量替换, Jacobian变换]
tags:
  - concept
  - math/probability
related_concepts:
  - "[[Jacobian_Matrix]]"
  - "[[Determinant]]"
  - "[[Normal_Distribution]]"
  - "[[Normalizing_Flow]]"
  - "[[Box_Muller_Transformation]]"
source: "概率论; 生成模型"
---

# 概率分布的变量替换 (Change of Variables)

#Probability  

## Change of Variable Theorem

***Assume*** we have *INPUT* $z$, with a distribution $\pi(z)$; a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^n$; *OUTPUT* $x$ :$f(z) = x$; $x$'s distribution $p(x)$.

***E.g.***

$$ z \sim U(0, 1), x = f(z) = 2z+1 \\ \Rightarrow x \sim U(1, 3)$$

Given that

$$ \int p(x) dx = 1, \int \pi(z) dz = 1$$

Then we have

$$ p(x) = \frac12 \pi(z)$$

***More Generally :***

- Given point $z'$ from distribution $\pi(z')$, $x'$ from distribution $p(x')$, and $f$ is known.
- Take little volume $\Delta z$ around $z'$ to get $(z', z'+\Delta z)$ , and accordingly $(x', x'+\Delta x)$.
- Since $\Delta z$ and $\Delta x$ are small, we can assume that $(z', z'+\Delta z)$ is *UNIFORMLY DISTRIBUTED*, and so does $(x', x'+\Delta x)$.
- Moreover, since $x$ is transformed from $z$, from $(z', z'+\Delta z)$ to $(x', x'+\Delta x)$, the uniform distribution should be of the same volume
  - i.e. $$p(x')\Delta x = \pi(z')\Delta z $$ $$\boxed{p(x') = \pi(z') \frac{\Delta z}{\Delta x} = \pi(z')  \left|\frac{\partial z}{\partial x}\right|}$$
  - As long as $f$ is given, $\frac{\partial z}{\partial x}$ is fixed, so we can get $p(x)$ from $\pi(z)$

***Multi-dimensionally:***

$$\boxed{p(x) = \pi(z) \left|\det\left(\frac{\partial z}{\partial x}\right)\right| = \pi(z) \left|\det\left(J_{f^{-1}}\right)\right|}$$

#### Change of Variables Formula (More mathematically)

Assume that we have a mapping rule $z=f_\theta(x)$ over random variables $x$ and $z$. 

It can be proved that
$$
p_\theta(x) \mathrm{d}x = p(z) \mathrm{d}z,
$$

> **By NOTE**
> Personally I would say that although the random variable has been transformed from one to another, but in a grand view of a event from happening, the shift in space would not affect the probablities. Thus this euqation holds.
> Here, LI Hongyi's explanation is that if we scale down to a very small area $\mathrm{d}x$ and $\mathrm{d}z$, then both of them would approximately follow a uniform distribution, with their total area being the same (i.e. 1, to be precise).

and thus we've got the *Change of Variables* Formula:
$$
p_\theta(x) = p(f_\theta(x)) \left| \frac{\partial f_\theta(x)}{\partial x} \right|,
$$
where $f_\theta$ is required to be invertible and differentiable.
