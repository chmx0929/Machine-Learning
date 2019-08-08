# 贝叶斯分类器

## 贝叶斯决策论

贝叶斯决策论是概率框架下实施决策的基本方法。对分类任务来说，在所有相关概率都已知的理想情形下，贝叶斯决策论考虑如何基于这些概率和误判损失来选择最优的类别标记。

假设有 $$N$$ 种可能的类别标记，即 $$\mathcal{Y}=\{c_1,c_2,\dots,c_N\}$$ ， $$\lambda_{ij}$$ 是将一个真实标记为 $$c_j$$ 的样本误分类为 $$c_i$$ 所产生的损失，基于后验概率 $$P(c_i|x)$$ 可获得将样本 $$x$$ 分类为 $$c_i$$ 所产生的期望损失，即在样本 $$x$$ 上的“条件风险”：

                                                             $$R(c_i|x)=\sum\limits_{j=1}^N\lambda_{ij}P(c_j|x)$$ 

我们的任务是寻找一个判定准则 $$h:\mathcal{X}\to\mathcal{Y}$$ 以最小化总体风险

                                                               $$R(h)=\mathbb{E}_x[R(h(x)|x)]$$ 

显然，对每个样本 $$x$$ ，若 $$h$$ 能最小化条件风险 $$R(h(x)|x)$$ ，则总体风险 $$R(h)$$ 也将被最小化。这就产生了贝叶斯判定准则：为最小化总体风险，只需在每个样本上选择那个能使条件风险 $$R(c|x)$$ 最小的类型标记，即

                                                              $$h^*(x)=\mathop{\arg\min}\limits_{c\in\mathcal{Y}}R(c|x)$$ 

此时， $$h^*$$ 称为贝叶斯最优分类器，与之对应的总体风险 $$R(h^*)$$ 称为贝叶斯风险。 $$1-R(h^*)$$ 反映了分类器所能达到的最好性能，即通过机器学习所能产生的模型精度的上限。具体来说，若目标是最小化分类错误率，则误判损失 $$\lambda_{ij}$$ 可写为

                                                                $$\lambda_{ij}=\begin{cases}0,\ \text{if}\ i=j\\ 1,\ \text{otherwiae}\end{cases}$$ 

此时条件风险

                                                                 $$R(c|x)=1-P(c|x)$$ 

于是，最小化分类错误率的贝叶斯最优分类器为

                                                              $$h^*(x)=\mathop{\arg\max}\limits_{c\in\mathcal{Y}}P(c|x)$$ 

即对每个样本 $$x$$ ，选择能使后验概率 $$P(c|x)$$ 最大的类别标记。

不难看出，欲使用贝叶斯判定准则来最小化决策风险，首先要获得后验概率 $$P(c|x)$$ 。然而，在现实任务中这通常难以直接获得。从这个角度来看，机器学习所要实现的是基于有限的训练样本集尽可能准确地估计出后验概率 $$P(c|x)$$ 。大体来说，主要有两种策略：

1. 1、给定 $$x$$ ，可通过直接建模 $$P(c|x)$$ 来预测 $$c$$ ，这样就得到的是“判别式模型”
2. 2、先对联合概率分布 $$P(x,c)$$ 建模，然后再由此获得 $$P(c|x)$$ ，这样得到的是“生成模型”

显然决策树，支持向量机等都可归入判别模型范畴。对生成模型来说，必然考虑

                                                                     $$P(c|x)=\frac{P(x,c)}{P(x)}$$ 

基于贝叶斯定理， $$P(c|x)$$ 可写为

                                                                   $$P(c|x)=\frac{P(c)P(x|c)}{P(x)}$$ 

其中， $$P(c)$$ 是类“先验”概率； $$P(x|c)$$ 是样本 $$x$$ 相对于标记 $$c$$ 的类条件概率，或称为“似然”； $$P(x)$$ 是用于归一化的“证据”因子。对给定样本 $$x$$ ，证据因子 $$P(x)$$ 与类标记无关，因此估计 $$P(c|x)$$ 的问题就转化为如何基于训练数据 $$D$$ 来估计先验概率 $$P(c)$$ 和似然 $$P(x|c)$$ 

类先验概率 $$P(c)$$ 表达了样本空间中各样本所占的比例，根据大数定律，当训练集包含充足的独立同分布样本时， $$P(c)$$ 可通过各类样本出现的频率来进行估计。

对类条件概率 $$P(x|c)$$ 来说，由于它涉及关于 $$x$$ 所有属性的联合概率，直接根据样本出现的频率来估计将会遇到严重的困难。例如，假设样本 $$d$$ 个属性值都是二值的，则样本空间将有 $$2^d$$ 种可能的取值，在现实应用中，这个值往往远大于训练样本数 $$m$$ ，也就是说，很多样本取值在训练集中根本没出现，直接用频率来估计 $$P(x|c)$$ 显然不可行，因为“未被观测到”与“出现概率为零”通常是不同的。估计类条件概率的一种常用策略是先假定其具有某种确定的概率分布形式，再基于训练样本对概率分布的参数进行估计。具体地，记关于类别 $$c$$ 的类条件概率为 $$P(x|c)$$ ，假设 $$P(x|c)$$ 具有确定的形式并且被参数向量 $$\theta_c$$ 唯一确定，则我们的任务就是利用训练集 $$D$$ 估计参数 $$\theta_c$$ 。为明确起见，我们将 $$P(x|c)$$ 记为 $$P(x|\theta_c)$$ 。事实上，概率模型的训练过程就是参数估计的过程，可用极大似然估计来确定估计概率分布参数。

## 朴素贝叶斯分类器

不难发现，基于贝叶斯公式 $$P(c|x)=\frac{P(c)P(x|c)}{P(x)}$$ 来估计后验概率 $$P(c|x)$$ 的主要困难在于：类条件概率 $$P(x|c)$$ 是所有属性上的联合概率，难以从有限的训练样本直接估计而得。为避开这个障碍，朴素贝叶斯分类器采用了“属性条件独立性假设”：对已知类别，假设所有属性独立。换言之，假设每个属性独立地对分类结果发生影响。基于属性条件独立性假设：

                                                    $$P(c|x)=\frac{P(c)P(x|c)}{P(x)}=\frac{P(c)}{P(x)}\prod\limits_{i=1}^dP(x_i|c)$$ 

其中 $$d$$ 为属性数目， $$x_i$$ 为 $$x$$ 在第 $$i$$ 个属性上的取值。

由于对所有类别来说 $$P(x)$$ 相同，因此基于 $$h^*(x)=\mathop{\arg\max}\limits_{c\in\mathcal{Y}}P(c|x)$$ 的贝叶斯判定准则：

                                                     $$h_{nb}(x)=\mathop{\arg\max}\limits_{c\in\mathcal{Y}}P(c)\prod_{i=1}^dP(x_i|c)$$ 

这就是朴素贝叶斯分类器的表达式。显然，朴素贝叶斯分类器的训练过程就是基于训练集 $$D$$ 来估计类先验概率 $$P(c)$$ ，并为每个属性估计条件概率 $$P(x_i|c)$$ 。

令 $$D_c$$ 表示训练集 $$D$$ 中第 $$c$$ 类样本的集合，若有充足的独立同分布样本，则可容易地估计出类先验概率

                                                                           $$P(c)=\frac{|D_c|}{|D|}$$ 

对离散属性而言，令 $$D_{c,x_i}$$ 表示 $$D_c$$ 中在第 $$i$$ 个属性上取值为 $$x_i$$ 的样本组成的集合，则条件概率 $$P(x_i,c)$$ 可估计为

                                                                        $$P(x_i,c)=\frac{D_{c,x_i}}{D_c}$$ 

对连续属性可考虑概率密度，假定 $$p(x_i,c)\sim\mathcal{N}(\mu_c,i,\sigma^2_{c,i})$$ ，其中 $$\mu_{c,i}$$ 和 $$\sigma^2_{c,i}$$ 分别是第 $$c$$ 类样本在第 $$i$$ 个属性上取值的均值和方差，则有

                                                         $$p(x_i|c)=\frac{1}{\sqrt{2\pi}\sigma_{c,i}}\exp(-\frac{(x_i-\mu_{c,i})^2}{2\sigma^2_{c,i}})$$ 

#### 举例说明

![](../../.gitbook/assets/1525765877800749.jpg)

假设我们对编号1进行分类（假设不知道其是否好瓜）

1、首先估计类先验概率 $$P(c)$$ ，显然有：

                                                $$P(好瓜=是)=\frac{8}{17}$$      $$P(好瓜=否)=\frac{9}{17}$$ 

2、然后，为每个属性估计条件概率 $$P(x_i|c)$$：

                                $$P(青绿|是)=P(色泽=青绿|好瓜=是)=\frac{3}{8}$$  

                                $$P(青绿|否)=P(色泽=青绿|好瓜=否)=\frac{3}{9}$$ 

                                $$P(蜷缩|是)=P(根蒂=蜷缩|好瓜=是)=\frac{5}{8}$$ 

                                $$P(蜷缩|否)=P(根蒂=蜷缩|好瓜=否)=\frac{3}{9}$$ 

                                                                   ......

                                  $$p_{密度:0.697|是}=p(密度=0.697|好瓜=是)$$ 

                                                          $$= \frac{1}{\sqrt{2\pi}\cdot0.129}\exp(-\frac{(0.697-0.574)^2}{2\cdot 0.129^2})$$ 

                                                                    ......

于是，有

$$P(好瓜=是)\times P_{青绿|是}\times P_{蜷缩|是}\times P_{浊响|是}\times P_{清晰|是}\times P_{凹陷|是}\times P_{硬滑|是}\times p_{密度:0.697|是}\times p_{含糖:0.460|是} \approx0.063$$ 

$$P(好瓜=否)\times P_{青绿|否}\times P_{蜷缩|否}\times P_{浊响|否}\times P_{清晰|否}\times P_{凹陷|否}\times P_{硬滑|否}\times p_{密度:0.697|否}\times p_{含糖:0.460|否} \approx6.80\times10^{-5}$$  

由于 $$0.063>6.80\times 10^{-5}$$ ，因此，朴素贝叶斯分类器将样本1判别为“好瓜”。

需注意，若某个属性值在训练集中没有与某个类同时出现过，则直接用上述方法进行概率估计会出现问题，假设对一个“敲声=清脆”的测试例，有 $$P_{清脆|是}=P(敲声=清脆|好瓜=是)=\frac{0}{8}$$ 。所以无论其他属性是什么，连乘计算出概率为零。为避免这种情况，在估计概率值时通常进行“平滑”，常用“拉普拉斯修正”。

## [Code实现](https://github.com/fengdu78/lihang-code/blob/master/code/%E7%AC%AC4%E7%AB%A0%20%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%28NaiveBayes%29/GaussianNB.ipynb)

基于贝叶斯定理与特征条件独立假设的分类方法。模型：

* 高斯模型
* 多项式模型
* 伯努利模型

### 数据

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:,:-1], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

### 手写实现

```python
class NaiveBayes:
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x-avg, 2) for x in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label:[] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))

model = NaiveBayes()
model.fit(X_train, y_train)
print(model.predict([4.4,  3.2,  1.3,  0.2]))
model.score(X_test, y_test)
```

### sklearn实现

{% embed url="https://scikit-learn.org/stable/modules/generated/sklearn.naive\_bayes.GaussianNB.html" %}

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB, MultinomialNB # 伯努利模型和多项式模型

clf = GaussianNB()
clf.fit(X_train, y_train)

clf.score(X_test, y_test)
clf.predict([[4.4,  3.2,  1.3,  0.2]])
```

