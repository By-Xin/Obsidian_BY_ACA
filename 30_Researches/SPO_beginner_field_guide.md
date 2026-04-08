# SPO beginner field guide

##### [**Undermind**](https://undermind.ai)

---


## Table of Contents

- [SPO field guide](#spo-field-guide)
- [What SPO is](#what-spo-is)
- [Terms that matter](#terms-that-matter)
- [The short history](#the-short-history)
- [The foundational papers](#the-foundational-papers)
  - [Smart Predict then Optimize](#smart-predict-then-optimize)
  - [Decision focused learning for combinatorial optimization](#decision-focused-learning-for-combinatorial-optimization)
  - [Hard combinatorial SPO](#hard-combinatorial-spo)
- [The theory core](#the-theory-core)
  - [Consistency and calibration](#consistency-and-calibration)
  - [Generalization](#generalization)
  - [Later refinements](#later-refinements)
- [The method map](#the-method-map)
- [The two main schools](#the-two-main-schools)
  - [School one](#school-one)
  - [School two](#school-two)
- [The wider academic map](#the-wider-academic-map)
- [Applications across the field](#applications-across-the-field)
- [Finance branch](#finance-branch)
  - [Mean variance portfolio optimization](#mean-variance-portfolio-optimization)
  - [Distributionally robust portfolio construction](#distributionally-robust-portfolio-construction)
  - [Risk budgeting and structured portfolios](#risk-budgeting-and-structured-portfolios)
  - [Sequential execution](#sequential-execution)
  - [What finance papers teach about the field](#what-finance-papers-teach-about-the-field)
- [What makes SPO methods work](#what-makes-spo-methods-work)
- [What can go wrong](#what-can-go-wrong)
- [Software and practical entry points](#software-and-practical-entry-points)
- [Recommended reading path](#recommended-reading-path)
  - [First layer](#first-layer)
  - [Second layer](#second-layer)
  - [Finance branch](#finance-branch-1)
- [Bottom line](#bottom-line)
- [References](#references)

## SPO field guide

Smart Predict then Optimize grew out of a simple complaint about the usual machine learning pipeline. In many decision problems, a model first predicts unknown quantities and an optimizer then chooses an action from those predictions. The usual training loss rewards accurate prediction. The real goal, though, is good decisions. SPO and the broader decision focused learning literature ask whether the model should instead be trained for downstream decision quality (Elmachtoub & Grigas, 2017; Mandi et al., 2023; Wilder et al., 2018).

The field is now broad. The original SPO line built a decision aware loss for linear objective optimization problems and developed theory for when its surrogate is sound (Balghiti et al., 2019; Elmachtoub & Grigas, 2017; H. Liu & Grigas, 2021). A closely related line in machine learning focused on differentiating through optimization layers and combinatorial solvers so neural networks could be trained end to end (Agrawal et al., 2019; Amos & Kolter, 2017; Wilder et al., 2018). Around both sits a wider contextual optimization literature that also includes stochastic programming, prescriptive analytics, robustness, online learning, and uncertainty sets (Bertsimas & Parys, 2017; Donti et al., 2017; Sadana et al., 2023). Finance is an important application area, but it is one branch among several rather than the field’s sole home (Butler & Kwon, 2021; Costa & Iyengar, 2022; Elmachtoub & Grigas, 2017; Uysal et al., 2021).

## What SPO is

The core setup is a contextual decision problem. Features $`x`$ are observed. Unknown problem parameters $`c`$ must be predicted. A feasible set $`S`$ is given by the optimization problem. The decision induced by a prediction $`\hat c`$ is

``` math
w^*(\hat c) \in \arg\min_{w \in S} \hat c^\top w
```

The true task loss is not pointwise prediction error. It is the decision regret under the true costs:

``` math
\ell_{\mathrm{SPO}}(\hat c, c) = c^\top w^*(\hat c) - z^*(c)
```

where $`z^*(c)`$ is the true optimal value. In words, SPO asks how much was lost because the decision was optimized for the wrong cost vector (Elmachtoub & Grigas, 2017).

This is the key mental shift for the whole field.

- **Prediction focused view** treats all forecast errors as equally important.
- **Decision focused view** treats forecast errors as important only insofar as they change the chosen action or its value.
- **Practical consequence** is that a model can have worse MSE and still make better decisions (Lee et al., 2024; Mandi et al., 2023; Wilder et al., 2018).

## Terms that matter

| Term | Main idea | Best linked papers | Why it matters |
|:---|:---|:---|:---|
| SPO | Train predictions for downstream regret | (Elmachtoub & Grigas, 2017), (H. Liu & Grigas, 2021), (Balghiti et al., 2019) | Foundational loss based view |
| SPO+ | Convex surrogate for the hard SPO loss | (Elmachtoub & Grigas, 2017), (H. Liu & Grigas, 2021) | Makes training tractable and theory possible |
| Decision focused learning | End to end training through the optimizer | (Wilder et al., 2018), (Mandi et al., 2023) | Broader ML framing of the same ambition |
| Differentiable optimization layers | Treat optimization as a neural layer | (Amos & Kolter, 2017), (Agrawal et al., 2019) | Key technical machinery |
| Contextual optimization | General family of decision under uncertainty methods | (Donti et al., 2017), (Sadana et al., 2023) | Places SPO in a wider OR landscape |
| Prescriptive analytics | Learn decisions from covariates under uncertainty | (Bertsimas & Parys, 2017), (Sadana et al., 2023) | Neighboring OR tradition |

## The short history

| Period | Landmark papers | What changed |
|:---|:---|:---|
| Early bridge between prediction and decisions | (Donti et al., 2017), (Amos & Kolter, 2017) | Showed end to end task based training and differentiable optimization layers |
| Foundational SPO period | (Elmachtoub & Grigas, 2017) | Defined SPO loss and SPO+ for linear objective problems |
| Early decision focused learning for discrete problems | (Wilder et al., 2018), (Ferber et al., 2019) | Brought combinatorial optimization and differentiable training together |
| Hard combinatorial SPO | (Mandi et al., 2019) | Showed SPO ideas can scale to harder discrete problems via relaxations and training tricks |
| Theory deepening | (Balghiti et al., 2019), (H. Liu & Grigas, 2021) | Added generalization, calibration, and finite sample guarantees |
| Consolidation and benchmarks | (Kotary et al., 2021), (Mandi et al., 2023), (Sadana et al., 2023), (Tang & Khalil, 2022) | Unified the field, compared methods, and lowered the tooling barrier |
| Recent expansion | (Schutte et al., 2023), (Huang & Gupta, 2024), (Hu et al., 2022; Hu et al., 2023; Hu et al., 2024) | Tackled robustness, gradient pathologies, and uncertainty in constraints |

## The foundational papers

### Smart Predict then Optimize

(Elmachtoub & Grigas, 2017) is the paper to read first. It introduced the SPO loss, derived the convex surrogate SPO+, and proved a Fisher consistency result under mild structural conditions. The paper also made an early and still important empirical point: when the predictive model is misspecified, training on a decision aware loss can beat standard prediction loss by a wide margin, even with simple linear models. The examples were shortest path and portfolio optimization, which helped establish both discrete operations and finance as natural use cases.

Two parts of this paper are especially foundational.

- **Loss design**. It defined regret in optimization terms instead of prediction terms.
- **Theory**. It showed the surrogate is not just a heuristic but aligned with the true decision loss in the population setting (Elmachtoub & Grigas, 2017).

### Decision focused learning for combinatorial optimization

(Wilder et al., 2018) is the other paper every newcomer should know. It came from the machine learning side and asked how to train predictive models through combinatorial decision problems. The method used continuous relaxations, quadratic regularization, and differentiation through KKT conditions. Its role in the field is different from (Elmachtoub & Grigas, 2017). SPO centers on a specially designed surrogate loss. This paper centers on backpropagating through an optimization layer. Those are now the two main instincts in the literature (Mandi et al., 2023; Wilder et al., 2018).

### Hard combinatorial SPO

(Mandi et al., 2019) extended the SPO line to genuinely hard discrete optimization. The main practical lesson was striking: during training, it is often enough to solve the continuous relaxation instead of the full NP hard problem. This kept the decision aware signal while making repeated training solves much cheaper. That result made SPO style methods feel practical beyond toy examples (Mandi et al., 2019).

## The theory core

The theory branch asks a simple question. If the model is trained on a tractable surrogate instead of the true decision regret, when can it still be trusted?

### Consistency and calibration

(Elmachtoub & Grigas, 2017) proved Fisher consistency of SPO+ for SPO loss. This is an asymptotic alignment result. It says that in the population limit, minimizing the surrogate can recover a minimizer of the true task loss.

(H. Liu & Grigas, 2021) goes further and is the main follow up theory paper for beginners. It provides calibration style bounds that convert excess SPO+ risk into excess true SPO risk. It also distinguishes two geometric regimes.

| Geometry of feasible set | Main takeaway from (H. Liu & Grigas, 2021) | Intuition |
|:---|:---|:---|
| Polyhedral sets | Weaker quadratic style calibration | Decisions can switch sharply at faces and vertices |
| Strongly convex sets | Stronger linear style calibration | Solutions move more smoothly with costs |

This matters because it ties optimization geometry to learnability. Smooth decision maps are easier to learn well than knife edge ones (H. Liu & Grigas, 2021).

### Generalization

(Balghiti et al., 2019) studies out of sample performance for SPO style training. The main challenge is that SPO loss is nonconvex, discontinuous, and not Lipschitz. Their solution is to analyze a margin based variant and exploit stability properties of the feasible region. The practical lesson is that not all optimization problems are equally friendly for learning. Problems with more stable optima should generalize better (Balghiti et al., 2019).

### Later refinements

Several later papers try to fix weak points exposed by early theory and practice.

- **Robust regret surrogates** in (Schutte et al., 2023) aim to better approximate expected regret under uncertainty.
- **Directional gradients** in (Huang & Gupta, 2024) address gradient quality and training signal.
- **Zero gradient diagnoses and fixes** in (Veviurko, Böhmer, et al., 2023; Veviurko, Bohmer, et al., 2023) show that exact differentiation can still stall even in convex settings if the solution map has large flat regions.

A good beginner summary is that the theory now says the idea is real, but the training signal can still be brittle. Much of recent work is about making that signal more informative.

## The method map

The field is easiest to understand as a set of technical strategies for getting useful gradients or useful surrogates.

| Method family | Main mechanism | Representative papers | Strengths | Weaknesses |
|:---|:---|:---|:---|:---|
| Decision aware surrogate losses | Optimize an upper bound or proxy for regret | (Elmachtoub & Grigas, 2017), (Mulamba et al., 2020), (Mandi et al., 2021), (Schutte et al., 2023) | Often general and solver agnostic | Surrogate quality varies by problem |
| Differentiate through relaxed optimization | Use KKT or implicit differentiation on a smooth relaxation | (Wilder et al., 2018), (Amos & Kolter, 2017), (Agrawal et al., 2019), (Ferber et al., 2019) | End to end and expressive | Relaxation gap and zero gradient issues |
| Perturbation and black box methods | Smooth decisions by noise or perturbation | (Berthet et al., 2020), (Mandi et al., 2023) | Can use harder solvers | More variance and weaker guarantees |
| Ranking and contrastive views | Learn to rank good solutions above bad ones | (Mulamba et al., 2020), (Mandi et al., 2021) | Efficient and intuitive for discrete spaces | Needs solution sets or caches |
| Global or bilevel reformulations | Solve the learning problem more exactly | (Jeong et al., 2022) | Strong optimality claims | Limited scalability |
| Constraint aware methods | Predict constraints safely and preserve feasibility | (Hu et al., 2022; Hu et al., 2023; Hu et al., 2024), (Mandi et al., 2025) | Handles realistic settings | Harder theory and training |

## The two main schools

### School one

The SPO school starts from regret as the main object and then designs a tractable surrogate. It is strongest when the unknown quantities sit in the objective and the optimization model has a linear objective form (Elmachtoub & Grigas, 2017; H. Liu & Grigas, 2021).

### School two

The decision focused learning school starts from an optimizer and asks how to backpropagate through it. It is strongest when differentiable relaxations or implicit differentiation are available and when richer neural predictors are wanted (Agrawal et al., 2019; Amos & Kolter, 2017; Wilder et al., 2018).

In practice, modern papers often mix the schools. A paper may use a differentiable optimizer and still train with a surrogate, or use ranking losses as a cheaper stand in for repeated exact optimization (Mandi et al., 2023).

## The wider academic map

SPO is not an isolated niche. It sits inside a larger family of learning for downstream decisions.

| Area | Relation to SPO | Good entry points |
|:---|:---|:---|
| Contextual stochastic optimization | Broad OR setting where decisions depend on covariates and uncertainty | (Donti et al., 2017), (Sadana et al., 2023) |
| Prescriptive analytics | Learns decisions or policies directly from data | (Bertsimas & Parys, 2017), (Sadana et al., 2023) |
| End to end constrained optimization | General ML plus optimizer viewpoint | (Kotary et al., 2021), (Mandi et al., 2023) |
| Differentiable optimization layers | Technical layer machinery for convex and quadratic problems | (Amos & Kolter, 2017), (Agrawal et al., 2019) |
| Combinatorial optimization layers | Discrete relaxations, perturbations, caches, ranking | (Wilder et al., 2018), (Ferber et al., 2019), (Mulamba et al., 2020), (Mandi et al., 2021) |
| Robust and distributionally robust optimization with learning | Adds ambiguity sets and robustness to misspecification | (Costa & Iyengar, 2022), (Patel et al., 2023), (Chenreddy & Delage, 2024) |
| Online and sequential decision focused learning | Repeated or streaming decisions | (H. Liu & Grigas, 2022), (Capitaine et al., 2025), (Kweon et al., 2024) |

The survey (Sadana et al., 2023) is useful because it makes clear that different communities rediscovered similar ideas under different names. The survey (Mandi et al., 2023) is useful because it maps those ideas specifically within decision focused learning and compares algorithms head to head.

## Applications across the field

Although finance matters, the literature is broader.

| Domain | What is predicted | What is optimized | Example papers |
|:---|:---|:---|:---|
| Portfolio construction | Returns, covariances, or risk budgets | Asset weights | (Elmachtoub & Grigas, 2017), (Butler & Kwon, 2021), (Costa & Iyengar, 2022), (Uysal et al., 2021), (Lee et al., 2024) |
| Execution and trading frictions | Future liquidity or market impact | Execution schedule | (Kweon et al., 2024) |
| Routing and shortest path | Edge costs | Path choice | (Elmachtoub & Grigas, 2017) |
| Packing and knapsack | Item values or rewards | Selection under capacity | (Demirovic et al., 2018), (Mandi et al., 2019) |
| Scheduling | Job costs or energy prices | Resource allocation schedule | (Mandi et al., 2019) |
| Power and energy | Demand, prices, renewable output | Dispatch and grid decisions | (Donti et al., 2017), (Chen & Hou, 2024) |
| Planning and graphs | Action costs, graph weights | Plans, matchings, or graph decisions | (Wilder et al., 2019), (Mandi et al., 2024), (Y. Liu et al., 2024) |

A useful correction for a beginner is that finance is not the dominant benchmark family in the foundational literature. Early core papers use finance as one example, but routing, knapsack, scheduling, and energy are at least as central to the field’s identity (Donti et al., 2017; Elmachtoub & Grigas, 2017; Mandi et al., 2019; Wilder et al., 2018).

## Finance branch

Finance fits decision focused learning unusually well because the whole pipeline is already prediction plus optimization. The catch is that finance is noisy, nonstationary, and sensitive to estimation error, so end to end methods can help but are not a free lunch.

### Mean variance portfolio optimization

(Butler & Kwon, 2021) is a strong finance entry point. It studies Markowitz mean variance optimization with integrated prediction. Instead of training a return model by ordinary regression and then plugging the forecasts into the optimizer, the paper chooses predictive parameters to minimize realized portfolio loss. For unconstrained and equality constrained cases, it derives closed form solutions. For more general constraints, it uses differentiable quadratic programming. Conceptually, the paper shows that end to end training can be read as learning forecasts that are better for portfolio choice, not necessarily better in raw statistical accuracy.

### Distributionally robust portfolio construction

(Costa & Iyengar, 2022) is important because it adds a finance specific concern that early SPO papers only touched lightly: model risk. The system learns not just the predictor but also the degree of robustness and risk tolerance inside a distributionally robust portfolio problem. This is a good example of how the field moved from plain end to end training toward decision aware training under ambiguity and regime uncertainty.

### Risk budgeting and structured portfolios

(Uysal et al., 2021) pushes the idea into risk budgeting rather than plain mean variance allocation. The key finding is that a model based end to end architecture, where a neural network feeds a structured risk budgeting layer, beats a model free black box allocator out of sample. For finance, that is a recurring lesson. Domain structure often helps more than unconstrained neural flexibility.

### Sequential execution

(Kweon et al., 2024) brings the paradigm into optimal execution. Here the prediction target is future market liquidity, and the downstream task is the liquidation schedule. This is a good example of a sequential and time series flavored decision focused problem rather than a static cross sectional allocation problem.

### What finance papers teach about the field

- **The target is not truth but usefulness.** A forecast that biases returns in a portfolio helpful way may beat a statistically cleaner forecast (Lee et al., 2024).
- **Robustness matters more than in many OR benchmarks.** Distribution shift and estimation error are first order concerns (Costa & Iyengar, 2022).
- **Structured optimization layers help.** Model based architectures often beat unconstrained black box predictors (Butler & Kwon, 2021; Uysal et al., 2021).
- **Classical portfolio theory remains the backbone.** Much of the literature extends Markowitz style ideas rather than replacing them (Butler & Kwon, 2021; Lee et al., 2024).

## What makes SPO methods work

Across the literature, the same pattern keeps appearing.

1.  A prediction model makes some errors.
2.  Only a few of those errors actually change the best action.
3.  Decision aware training learns to spend model capacity on those action changing directions.
4.  This can improve regret even when standard forecast metrics worsen.

That is why papers often report the same paradox: worse MSE, better decisions (Lee et al., 2024; Mandi et al., 2023; Wilder et al., 2018).

## What can go wrong

The literature is also unusually honest about failure modes.

| Failure mode | Why it happens | Representative papers |
|:---|:---|:---|
| Weak gradients | Solution maps are piecewise constant or flat | (Veviurko, Böhmer, et al., 2023; Veviurko, Bohmer, et al., 2023) |
| Bad relaxations | The relaxed problem is too different from the discrete one | (Mandi et al., 2023) |
| Solver cost in training | Every step may require solving a hard problem | (Mandi et al., 2019), (Mulamba et al., 2020) |
| Overfitting to surrogate quirks | Surrogate and true regret are not identical | (Schutte et al., 2023), (Mandi et al., 2023) |
| Constraint prediction errors | Predicted constraints can create infeasible decisions | (Hu et al., 2022; Hu et al., 2023; Hu et al., 2024) |
| Limited benefit under perfect specification | If prediction is already well specified, end to end gains may shrink | (Mandi et al., 2023) |

A beginner should not think of SPO as a universal replacement for prediction first training. It is best seen as a tool for settings where prediction errors matter unevenly and where optimization structure can reveal which errors are most costly.

## Software and practical entry points

(Tang & Khalil, 2022) is the most useful practical software paper. PyEPO provides a common interface for end to end predict then optimize methods over linear and integer programming problems. For someone who wants to move from reading to experimentation, it is the natural first stop.

The two best broad surveys are complementary.

- **For decision focused learning proper** read (Mandi et al., 2023). It gives taxonomy, benchmark results, and a grounded view of what works.
- **For the wider OR landscape** read (Sadana et al., 2023). It places SPO alongside contextual optimization, prescriptive analytics, and stochastic programming.

(Kotary et al., 2021) is helpful as a bridge survey on end to end constrained optimization and optimization layers.

## Recommended reading path

### First layer

| Read first | Why |
|:---|:---|
| (Elmachtoub & Grigas, 2017) | Original SPO formulation, SPO+, consistency, canonical examples |
| (Wilder et al., 2018) | End to end combinatorial optimization viewpoint |
| (Mandi et al., 2023) | Best single survey of the modern field |
| (Sadana et al., 2023) | Best broad map of neighboring OR literature |

### Second layer

| Read next | Why |
|:---|:---|
| (Balghiti et al., 2019) | Generalization theory |
| (H. Liu & Grigas, 2021) | Calibration and risk bounds for SPO+ |
| (Mandi et al., 2019) | Hard combinatorial training and relaxations |
| (Amos & Kolter, 2017), (Agrawal et al., 2019) | Differentiable optimization layers |
| (Tang & Khalil, 2022) | Practical tooling and algorithm menu |

### Finance branch

| Read after the basics | Why |
|:---|:---|
| (Butler & Kwon, 2021) | Clean mean variance integrated prediction setup |
| (Costa & Iyengar, 2022) | Robust end to end portfolio construction |
| (Uysal et al., 2021) | Model based end to end risk budgeting |
| (Lee et al., 2024) | Clear explanation of how DFL biases forecasts for portfolio choice |
| (Kweon et al., 2024) | Sequential execution application |

## Bottom line

SPO is best understood as the founding loss based formulation of a broader movement to train predictive models for the decisions they induce rather than for raw forecast accuracy. The field now has two mature cores. One builds decision aware surrogates with growing theory around calibration and generalization (Balghiti et al., 2019; Elmachtoub & Grigas, 2017; H. Liu & Grigas, 2021). The other differentiates through optimization layers and relaxed solvers to enable end to end learning with richer models (Agrawal et al., 2019; Amos & Kolter, 2017; Wilder et al., 2018). Around them lies a larger contextual optimization literature that connects OR, machine learning, and robust decision making (Donti et al., 2017; Sadana et al., 2023).

For finance, the most durable lesson is not that end to end methods replace portfolio theory. It is that they reshape how prediction should serve portfolio theory. Good financial forecasts are not simply accurate forecasts. They are forecasts that lead to better trades, better allocations, and better risk control under realistic uncertainty (Butler & Kwon, 2021; Costa & Iyengar, 2022; Lee et al., 2024; Uysal et al., 2021).

---

## References

Agrawal, A., Amos, B., Barratt, S. T., Boyd, S. P., Diamond, S., & Kolter, J. Z. (2019). Differentiable Convex Optimization Layers. *Neural Information Processing Systems*, 9558–9570.

Amos, B., & Kolter, J. Z. (2017). OptNet: Differentiable Optimization as a Layer in Neural Networks. *International Conference on Machine Learning*, 136–145.

Balghiti, O. E., Elmachtoub, A. N., Grigas, P., & Tewari, A. (2019). Generalization Bounds in the Predict-then-Optimize Framework. *Neural Information Processing Systems*, 14389–14398. <https://doi.org/10.1287/moor.2022.1330>

Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J.-P., & Bach, F. (2020). Learning with Differentiable Perturbed Optimizers. *ArXiv*, *abs/2002.08676*.

Bertsimas, D., & Parys, B. P. G. V. (2017). Bootstrap robust prescriptive analytics. *Mathematical Programming*, *195*, 39–78. <https://doi.org/10.1007/s10107-021-01679-2>

Butler, A., & Kwon, R. (2021). Integrating prediction in mean-variance portfolio optimization. In *Quantitative Finance* (Vol. 23, pp. 429–452). <https://doi.org/10.1080/14697688.2022.2162432>

Capitaine, A., Haddouche, M., Moulines, É., Jordan, M. I., Boursier, E., & Durmus, A. (2025). Online Decision-Focused Learning. *ArXiv*, *abs/2505.13564*. <https://doi.org/10.48550/arXiv.2505.13564>

Chen, Y., & Hou, Y. (2024). Close-loop Predict-and-Optimize Method for Power System Operation: Concepts, Rationality, Applications, and Prospects. *2024 IEEE 8th Conference on Energy Internet and Energy System Integration (EI2)*, 3289–3296. <https://doi.org/10.1109/EI264398.2024.10990799>

Chenreddy, A., & Delage, E. (2024). End-to-end Conditional Robust Optimization. *ArXiv*, *abs/2403.04670*. <https://doi.org/10.48550/arXiv.2403.04670>

Costa, G., & Iyengar, G. (2022). Distributionally robust end-to-end portfolio construction. *Quantitative Finance*, *23*, 1465–1482. <https://doi.org/10.1080/14697688.2023.2236148>

Demirovic, E., Guns, T., Stuckey, P. J., Bailey, J., Chan, J., Leckie, C., & Kotagiri, R. (2018). *Prediction + optimization for the knapsack problem*.

Donti, P., Kolter, J. Z., & Amos, B. (2017). Task-based End-to-end Model Learning in Stochastic Optimization. *Neural Information Processing Systems*, 5484–5494.

Elmachtoub, A. N., & Grigas, P. (2017). Smart “Predict, then Optimize.” *ArXiv*, *abs/1710.08005*. <https://doi.org/10.1287/MNSC.2020.3922>

Ferber, A., Wilder, B., Dilkina, B., & Tambe, M. (2019). MIPaaL: Mixed Integer Program as a Layer. *ArXiv*, *abs/1907.05912*. <https://doi.org/10.1609/AAAI.V34I02.5509>

Hu, X., Lee, J. C. H., & Lee, J. H. M. (2023). Two-Stage Predict+Optimize for Mixed Integer Linear Programs with Unknown Parameters in Constraints. *ArXiv*, *abs/2311.08022*. <https://doi.org/10.48550/arXiv.2311.08022>

Hu, X., Lee, J. C. H., Lee, J. H. M., & Stuckey, P. J. (2024). Multi-Stage Predict+Optimize for (Mixed Integer) Linear Programs. *Advances in Neural Information Processing Systems 37*. <https://doi.org/10.52202/079017-2068>

Hu, X., Lee, J. C. H., & Lee, J. H. (2022). Predict+Optimize for Packing and Covering LPs with Unknown Parameters in Constraints. *ArXiv*, *abs/2209.03668*. <https://doi.org/10.48550/arXiv.2209.03668>

Huang, M., & Gupta, V. (2024). Decision-Focused Learning with Directional Gradients. *Advances in Neural Information Processing Systems 37*. <https://doi.org/10.52202/079017-2514>

Jeong, J., Jaggi, P., Butler, A., & Sanner, S. (2022). An Exact Symbolic Reduction of Linear Smart Predict+Optimize to Mixed Integer Linear Programming. *International Conference on Machine Learning*, 10053–10067.

Kotary, J., Fioretto, F., Hentenryck, P. V., & Wilder, B. (2021). End-to-End Constrained Optimization Learning: A Survey. *ArXiv*, *abs/2103.16378*. <https://doi.org/10.24963/ijcai.2021/610>

Kweon, S., Yim, Y., & Min, S. (2024). Optimizing Sequential Predictions for Order Execution: a Decision Focused Learning Approach. In *Proceedings of the 5th ACM International Conference on AI in Finance*. <https://doi.org/10.1145/3677052.3698665>

Lee, J., Jeon, H., Bae, H., & Lee, Y. (2024). Return Prediction for Mean-Variance Portfolio Selection: How Decision-Focused Learning Shapes Forecasting Models. *Proceedings of the 6th ACM International Conference on AI in Finance*. <https://doi.org/10.1145/3768292.3770423>

Liu, H., & Grigas, P. (2021). Risk Bounds and Calibration for a Smart Predict-then-Optimize Method. *ArXiv*, *abs/2108.08887*.

Liu, H., & Grigas, P. (2022). Online Contextual Decision-Making with a Smart Predict-then-Optimize Method. *ArXiv*, *abs/2206.07316*. <https://doi.org/10.48550/arXiv.2206.07316>

Liu, Y., Zhou, C., Zhang, P., Pan, S., Li, Z., & Chen, H. (2024). Decision-focused Graph Neural Networks for Combinatorial Optimization. *ArXiv*, *abs/2406.03647*. <https://doi.org/10.48550/arXiv.2406.03647>

Mandi, J., Bucarey, V., Tchomba, M. M. K., & Guns, T. (2021). Decision-Focused Learning: Through the Lens of Learning to Rank. *International Conference on Machine Learning*, 14935–14947.

Mandi, J., Defresne, M., Berden, S., & Guns, T. (2025). Feasibility-Aware Decision-Focused Learning for Predicting Parameters in the Constraints. *ArXiv*, *abs/2510.04951*. <https://doi.org/10.48550/arXiv.2510.04951>

Mandi, J., Foschini, M., Höller, D., Thiébaux, S., Hoffmann, J., & Guns, T. (2024). Decision-Focused Learning to Predict Action Costs for Planning. *European Conference on Artificial Intelligence*, 4060–4067. <https://doi.org/10.48550/arXiv.2408.06876>

Mandi, J., Kotary, J., Berden, S., Mulamba, M., Bucarey, V., Guns, T., & Fioretto, F. (2023). Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities. *J. Artif. Intell. Res.*, *80*, 1623–1701. <https://doi.org/10.1613/jair.1.15320>

Mandi, J., Demirovi’c, E., Stuckey, P. J., & Guns, T. (2019). Smart Predict-and-Optimize for Hard Combinatorial Optimization Problems. *ArXiv*, *abs/1911.10092*. <https://doi.org/10.1609/AAAI.V34I02.5521>

Mulamba, M., Mandi, J., Diligenti, M., Lombardi, M., Bucarey, V., & Guns, T. (2020). Contrastive Losses and Solution Caching for Predict-and-Optimize. *International Joint Conference on Artificial Intelligence*, 2833–2840. <https://doi.org/10.24963/ijcai.2021/390>

Patel, Y. P., Rayan, S., & Tewari, A. (2023). Conformal Contextual Robust Optimization. *International Conference on Artificial Intelligence and Statistics*, 2485–2493. <https://doi.org/10.48550/arXiv.2310.10003>

Sadana, U., Chenreddy, A., Delage, E., Forel, A., Frejinger, E., & Vidal, T. (2023). A Survey of Contextual Optimization Methods for Decision Making under Uncertainty. *ArXiv*, *abs/2306.10374*. <https://doi.org/10.1016/j.ejor.2024.03.020>

Schutte, N., Postek, K., & Yorke-Smith, N. (2023). Robust Losses for Decision-Focused Learning. *ArXiv*, *abs/2310.04328*. <https://doi.org/10.24963/ijcai.2024/538>

Tang, B., & Khalil, E. B. (2022). PyEPO: a PyTorch-based end-to-end predict-then-optimize library for linear and integer programming. *Mathematical Programming Computation*, *16*, 297–335. <https://doi.org/10.1007/s12532-024-00255-x>

Uysal, A. S., Li, X., & Mulvey, J. M. (2021). End-to-end risk budgeting portfolio optimization with neural networks. *Annals of Operations Research*, *339*, 397–426. <https://doi.org/10.1007/s10479-023-05539-4>

Veviurko, G., Bohmer, W., & Weerdt, M. D. (2023). *You Shall Pass: Dealing with the Zero-Gradient Problem in Predict and Optimize for Convex Optimization*.

Veviurko, G., Böhmer, W., & Weerdt, M. D. (2023). You Shall not Pass: the Zero-Gradient Problem in Predict and Optimize for Convex Optimization. *ArXiv*, *abs/2307.16304*. <https://doi.org/10.48550/arXiv.2307.16304>

Wilder, B., Dilkina, B., & Tambe, M. (2018). Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization. *ArXiv*, *abs/1809.05504*. <https://doi.org/10.1609/AAAI.V33I01.33011658>

Wilder, B., Ewing, E., Dilkina, B., & Tambe, M. (2019). End to end learning and optimization on graphs. *ArXiv*, *abs/1905.13732*.
