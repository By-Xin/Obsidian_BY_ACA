---
aliases: ['集成学习', 'Ensemble Learning']
tags:
  - concept
  - method
  - ml/ensemble
related_concepts:
  - [[Boosting]]
  - [[Bagging]]
  - [[Random_Forest]]
---

#StatisticalLearning 
- In ensemble learning, multiple models are trained and combined to improve the performance of the model.
- weak & strong learners
  - *weak learner*: A model that performs slightly better than random guessing.
  - *strong learner*: A model that performs well on a given task.
- Ensemble learning performs better when models are diverse and independent.
- soft & hard voting
  - *hard voting*: The class with the most votes is predicted.
  - *soft voting*: The class with the highest average probability is predicted.
- Popular ensemble learning algorithms:
  - [[Bagging and Pasting]]
  - [[Boosting]]
  - [[Stacking]]
  - [[Random Forest]]
  - [[Gradient Boosting]]