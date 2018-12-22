# XGBoost

## XGBoost简介 

在数据建模中，经常采用Boosting方法，该方法将成百上千个分类准确率较低的树模型组合起来，成为一个准确率很高的预测模型。这个模型会不断地迭代，每次迭代就生成一颗新的树。但在数据集较复杂的时候，可能需要几千次迭代运算，这将造成巨大的计算瓶颈。

针对这个问题。华盛顿大学的陈天奇博士开发的XGBoost（eXtreme Gradient Boosting）基于C++通过多线程实现了回归树的并行构建，并在原有Gradient Boosting算法基础上加以改进，从而极大地提升了模型训练速度和预测精度。在Kaggle的比赛中，多次有队伍借助XGBoost在比赛中夺得第一。其次，因为它的效果好，计算复杂度不高，在工业界中也有大量的应用。

## 监督学习的三要素

因为Boosting Tree本身是一种有监督学习算法，要讲Boosting Tree，先从监督学习讲起。在监督学习中有几个逻辑上的重要组成部件，粗略地可以分为：模型、参数、目标函数和优化算法。

### 模型

模型指的是给定输入 $$x_i$$ 如何去预测输出 $$y_i$$ 。我们比较常见的模型如线性模型（包括线性回归和Logistic Regression）采用线性加和的方式进行预测

                                                                    $$\hat{y_i}=\sum\limits_jw_j x_{ij}$$ 

这里的预测值 $$y$$ 可以有不同的解释，比如我们可以把它作为回归目标的输出，或者进行sigmoid变换得到概率（即用 $$\frac{1}{1+e^{-\hat{y_i}}}$$ 来预测正例的概率）

### 参数

### 目标函数：误差函数+正则化项

## Source

{% embed url="https://blog.csdn.net/anshuai\_aw1/article/details/82970489\#\_604" %}

{% embed url="https://blog.csdn.net/anshuai\_aw1/article/details/85093106" %}



