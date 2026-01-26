#StatisticalLearning 
- 基本记号
  - $(x^{(i)}, y^{(i)})$表示第$i$个训练样本(training example)
  - $x^{(i)}$表示第$i$个训练样本的输入变量(input variables), 也称为输入特征(input features)
  - $y^{(i)}$表示第$i$个训练样本的输出或目标变量(target variable)

- 有监督学习的general目标：给定一个训练集，学习一个函数$h: X \rightarrow Y$，使得$h(x)$能够对$y$进行准确预测 (这里也称这个函数$h$为一个假设(hypothesis))

- 根据被预测变量的连续/离散与否，可以将监督学习分为*回归(regression)* 和 *分类(classification)* 两种情况。