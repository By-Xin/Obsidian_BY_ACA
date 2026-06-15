# Error Bound-based Analysis for Monotone Randomized Algorithms

余文忠 PolyU-HK (https://manchungyue.com/EBRA.pdf)

## Background 

$$
\min_{x \in \mathbb{R}^d} f(x) 
$$

$$
x^{k+1} = x^k - \alpha ...
$$

focus 在 smooth , diff 的 case

在期望意义上:

- 在 $f$ 的 convex 下保证 sublinear 的 $O(1/k)$ 的 rate. 

- 若 strongly convex 则保证 linear 的 rate. 几何收敛 $O(1 /c^k)$

- Polyak-Lojasiwicz property, 还是 linear 的


- Kurdyka-Lojasiewicz property. 在某一局部点的 growth 的速度控制在 grad norm 的某个函数. 在一些条件下, KL 是 PL. 


- essential cyclicity -> 强行要求在固定 step 内访问所有 coordinate. 例如 sample with replacement, shuffle 就可以. 


- Random Kransnoselskii-Mann 


RCD 

projecting


quanti / binary sgd

double stochastic sample coordinate

sign SGD

onebit sgd




