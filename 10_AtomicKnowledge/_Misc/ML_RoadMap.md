# Categories of Machine Learning

## 1. From Statistical Learning Aspect
### 1.1 [[Frequencist]]
*Statistical Learning* 
- [[OLS,Logistics]]
	-  [[Exponential Family (GLM)]]
- [[Demension Reduction]]
- [[SVM,Kernel]]
- [[Tree-based Models]]
	- [[Decision Trees]]
	- [[Ensemble Learning]]
		- [[Random Forest]]
		- [[Bagging and Pasting]]
		- [[Boosting]]
			- [[AdaBoost]]
			- [[Gradient Boosting]]
- [[Generalized Additive Models]]
	- [[Basis Functions for Splines]]
		- [[Polynomial Regression]]
		- [[Step Functions]]
		- [[Regression Splines]]
		- [[Smotting Splines]]
		- [[Local Regression]]
### 1.2 [[Baysian]]
- [[Generative Learning algorithms]]
	- [[Gaussian Discriminant Analysis (GDA)]]
	- [[Naive Bayes (Classifier)]]
- [[Probabilistic Graphical Model]]
  - [[Baysian Network]]
  - [[Markov Random Field]]
  - [[Baysian Inference]]
- [[EM Algorithm]]
### 1.3 Techniques in Machine Learning
- [[Regularization]]

## 2. Whether under supervision?
### 2.1 Supervised Learning
*including labels*
- [[KNN]]
- [[Linear Regression]]
- [[Logistics Regression]]
- [[SVM]]
- [[Decision Trees]], [[Random Forest]]
- [[Neural Forest]]

### 2.2 Unsupervised Learning
*without labels*
#### Clusterings:
- [[k-means]]
- [[DBSCAN]]
- [[HCA]]
#### Anomaly Detection & Novelty Detection 
*Anomaly detects outliers, Novelty detects new data*
- [[One Class SVM]]
- [[Isolation Forest]]
#### Visialization & Dimensionality
- [[PCA]]
- [[Kernal PCA]]
- [[LLE]]
- [[t-SNE]]
#### Association Rule Learning
 *finding patterns and relationships in data*
- [[Apriori]]
- [[Eclat]]

### 2.3 Semi-supervised Learning
*partially labeled, usually a combination of supervised and unsupervised learning*
- [[Deep Belief Networks]]
	- [[Restricted Boltzmann Machines]]

### 2.4 [[Reinforcement Learning]]
*learning from rewards or punishments*

## 3. Whether able to learn incrementally?
#### 3.1 Batch Learning
- unable to learn incrementally, each time trained on the whole dataset(e.g. Linear Regression)
- Requires a lot of computational resources

### 3.2 Online Learning
 - able to learn incrementally, used data can be discarded after learning(e.g. Gradient Descent)
 - *learning rate*: to control the learning speed for learning and forgetting
 - *out-of-core learning*: to train on huge datasets that cannot fit into a single machine's main memory, by using online learning
 - Challenge: if bad data is fed, the model's performance will degrade over time

#### 4. Whether able to generalize?
#### 4.1 Instance-based Learning
*make predictions by comparing new data to known data*
- [[KNN]]

#### 4.2 Model-based Learning
*define a model from the training data and use it to make predictions*
- [[Linear Regression]]
- [[SVM]]


# Main Challenges of Machine Learning 

- Insufficient Data
- Data lack of representativeness
- Poor Data Quality
  - Wrong Data: remove or correct
  - Missing Data: remove, fill, or ignore
- Irrelevant Features
  - Feature Engineering
    - Feature Selection
    - Feature Extraction
    - Feature Creation
- Overfitting
- Underfitting

# Model Evaluation & Testing

- Hyperparameter Tuning & Model Selection
  - For hyperparameter tuning, use *dev set* to tune hyperparameters, and *test set* to evaluate the model.
  - [[Cross-Validation]]
- Data Unmatching
  - *train-dev set*: to detect data mismatch between training and dev set