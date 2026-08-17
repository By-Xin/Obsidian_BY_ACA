#StatisticalLearning 
## Introduction to Tree-based Models
### Decision Trees

- Pros
  - Explainable
  - Can handle both numerical and categorical data
- Cons
  - non-robust (easily to be affected by noise)
    - use ensemble methods to reduce variance
  - complex trees can be overfitting
    - prune trees to simplify
  - not easy to be parallelized in computing

### Random Forest 

- Train multiple decision trees to improve robustness
  - each tree is trained independently
  - majority voting for classification; average for regression
- *Randomness*
  - Bagging: random sampling training sets with replacement (有放回抽样)
  - random feature selection

### Gradient Boosting Decision Trees
- Train multiple decision trees sequentially:
  - $F_t(x)$ is the sum of all previous trees
  - Train a new tree $f_t(x)$ on the residual: $\{x_i, y_i - F_{t-1}(x_i)\}$
  - $F_{t+1} = F_t + f_t$