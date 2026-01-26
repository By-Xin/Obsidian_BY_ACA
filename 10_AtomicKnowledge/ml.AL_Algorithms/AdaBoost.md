#StatisticalLearning 

- When training an AdaBoost model, the algorithm will start by training a base classifier on the original dataset.
- The algorithm will then increase the weights of the misclassified instances and train another classifier on the dataset.
- Continue this iterative process until a specified number of classifiers are trained or the model achieves a perfect score.
  - AdaBoost is similar to Gradient Descent, but instead of tweaking a single model's parameters to minimize a cost function, AdaBoost adds models to the ensemble, making the new model correct the errors of its predecessor.
- Once the training is complete, the model will make predictions via bagging or pasting.

- Algorithm:
  1. Initialize the weights of the instances: $w^{(1)}_i = 1/m$.
  2. Train a base classifier on the dataset, and compute the weighted error rate: $\epsilon_j = \frac{\sum_{i=1}^{m} w^{(j)}_i \times 1_\text{missclassified}}{\sum_{i=1}^{m} w^{(j)}_i}$.
  3. Compute the predictor's weight: $\alpha_j = \eta \log\left(\frac{1 - \epsilon_j}{\epsilon_j}\right)$.
  4. Update the weights of the instances: $w^{(j+1)}_i = \begin{cases} w^{(j)}_i & \text{if } h_j(x_i) = y_i \\ w^{(j)}_i \times \exp(\alpha_j) & \text{if } h_j(x_i) \neq y_i \end{cases}$.
  5. Normalize the weights: $w^{(j+1)}_i = \frac{w^{(j+1)}_i}{\sum_{i=1}^{m} w^{(j+1)}_i}$.
  6. Repeat steps 2-5 using the updated weights, until the desired number of classifiers is reached.
  7. Make predictions using the ensemble of classifiers: $\hat{y}(x) = \arg\max_k \sum_{j=1}^{N} \alpha_j \times 1_{h_j(x) = k}$. 