#StatisticalLearning 

- [[Bagging and Pasting]] are two ensemble methods that build multiple instances of a model and combine them to get a more accurate and stable prediction.
- We train multiple models using same algorithm, but with different samples of the training data.
  - During sampling (to create a dataset), we can either:
    - sample with replacement (bagging)
    - sample without replacement (pasting)
  > *i.e. both bagging and pasting allow training instances to be sampled several times across multiple models, but only bagging allows training instances to be sampled several times for the same model.*

- *Out-of-Bag Evaluation*: In bagging, some instances may be sampled several times for some models, while others may not be sampled at all. The instances that are not sampled for a particular model are called out-of-bag instances. These instances can be used to evaluate the model without the need for a separate validation set.