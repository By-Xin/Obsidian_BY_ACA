#MetaLearning

## Introduction to Hyperparameter Optimization

Typical methods for hyperparameter optimization:
- Gird Search
- Random Search 
  - Sample some hyperparameters randomly from the grids or distributions
  - **Theoratical assumptions:**
    - The relavent assumption is that: top $K$ results are good enough to approximate the global optimum. 
    - Then if there are totally $N$ hyperparameters(or its combinations), the probability of finding the best hyperparameters in the top $K$ results is $K/N$.
    - Thus if we sample $x$ times, the probability of finding the (almost) best hyperparameters is $1-(1-K/N)^x$.
    - Further control the probability of finding the best hyperparameters greater than $1-\epsilon$. For example $1-\epsilon=0.90$ and $N=1000, K=10$, then $x=230$.

- Classic method: *Bayesian Optimization*

## Auto ML

