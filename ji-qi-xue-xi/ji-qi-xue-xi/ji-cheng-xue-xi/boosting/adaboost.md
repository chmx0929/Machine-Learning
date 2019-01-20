# AdaBoost

## AdaBoost算法

### 算法步骤

假设给定一个二分类的训练数据集 $$T = \{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$$ 其中，每个样本点由实例与标记组成。实例 $$x_i\in\mathcal{X}\subseteq R^n$$ ，标记 $$y_i \in \mathcal{Y}=\{-1,+1\}$$ ， $$\mathcal{X}$$ 是实例空间， $$\mathcal{Y}$$ 是标记集合。AdaBoost利用以下算法，从训练数据中学习一系列弱分类器或基本分类器，并将这些若分类器线性组合成为一个强分类器。

输入：训练数据集 $$T = \{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$$，其中 $$x_i\in\mathcal{X}\subseteq R^n$$ ，标记 $$y_i \in \mathcal{Y}=\{-1,+1\}$$ ，弱学习算法；

输出：最终分类器 $$G(x)$$ 

（1）初始化训练数据的权值分布

                        $$D_1=(w_{11},\dots,w_{1i},\dots,w_{1N}),\ \ \ w_{1i}=\frac{1}{N}, \ \ \ i=1,2,\dots,N$$ 

（2）对 $$m = 1,2,\dots,M$$ 

* （a）使用具有权值分布 $$D_m$$ 的训练数据集学习，得到基本分类器
*                         $$G_m(x):\mathcal{X}\to\{-1,+1\}$$ 
* （b）计算 $$G_m(x)$$ 在训练数据上的分类误差率
*                         $$e_m=\sum\limits_{i=1}^NP(G_m(x_i)\neq y_i)=\sum\limits_{i=1}^Nw_{mi}I(G_m(x_i)\neq y_i)$$ 
* （c）计算 $$G_m(x)$$ 的系数
*                          $$\alpha_m=\frac{1}{2}\ln\frac{1-e_m}{e_m}$$ 
* （d）更新训练数据集的权值分布
*                         $$D_{m+1}=(w_{m+1,1},\dots,w_{m+1,i},\dots,w_{m+1,N})$$ 
*                         $$w_{m+1,i}=\frac{w_{mi}}{Z_m}\exp(-\alpha_my_iG_m(x_i)),\ \ \ i=1,2,\dots,N$$ 
*                         $$Z_m=\sum\limits_{i=1}^Nw_{mi}\exp(-\alpha_my_iG_m(x_i))$$ 

（3）构建基本分类器的线性组合

                           $$f(x)=\sum\limits_{m=1}^M\alpha_mG_m(x)$$ 

           得到最终分类器

                            $$G(x)=sign(f(x))=sign(\sum\limits_{m=1}^M\alpha_mG_m(x))$$ 

### 步骤说明

步骤（1）：假设训练数据集具有均匀的权值分布，即每个训练样本在基本分类器的学习中作用相同，这一假设保证第1步能够在原始数据上学习基本分类器 $$G_1(x)$$ 

步骤（2）：AdaBoost反复学习基本分类器，在每一轮 $$m=1,2,\dots,M$$ 顺次地执行下列操作

* （a）使用当前分布 $$D_m$$ 加权的训练数据集，学习基本分类器 $$G_m(x)$$ 
* （b）计算基本分类器 $$G_m(x)$$ 在加权训练数据集上的分类误差率：
*                   $$e_m=\sum\limits_{i=1}^NP(G_m(x_i)\neq y_i)=\sum\limits_{G_m(x_i)\neq y_i}w_{mi}$$ 
*            这里， $$w_{mi}$$ 表示第 $$m$$ 轮中第 $$i$$ 个实例的权值， $$\sum\limits_{i=1}^Nw_{mi}=1$$ 。这表明， $$G_m(x)$$ 在加权的
*            训练数据集上的分类误差率是被 $$G_m(x)$$ 误分类样本的权值之和，由此可以看出数据权值分布
*            $$D_m$$ 与基本分类器 $$G_m(x)$$ 的分类误差率的关系。
* （c）计算基本分类器 $$G_m(x)$$ 的系数 $$\alpha_m$$。 $$\alpha_m$$ 表示 $$G_m(x)$$ 在最终分类器中的重要性。由
*            $$\alpha_m=\frac{1}{2}\ln\frac{1-e_m}{e_m}$$ 可知，当 $$e_m\leq\frac{1}{2}$$ 时， $$\alpha_m\geq0$$ ，并且 $$\alpha_m$$ 随着 $$e_m$$ 的减小而增大，所以
*             分类误差率越小的基本分类器在最终分类器中的作用越大。
* （d）更新训练数据的权值分布为下一轮作准备。
*            $$w_{m+1,i}=\frac{w_{mi}}{Z_m}\exp(-\alpha_my_iG_m(x_i)),\ \ \ i=1,2,\dots,N$$ 可写为
*            $$w_{m+1,i} = \begin{cases}\frac{w_{mi}}{Z_m}e^{-\alpha_m},\ \ \ G_m(x_i)=y_i\\ \frac{w_{mi}}{Z_m}e^{a_m},\ \ \ \ \ G_m(x_i)\neq y_i\end{cases}$$ 
*             由此可知，被基本分类器 $$G_m(x)$$ 误分类样本的权值得以扩大，而被正确分类的样本的权值却
*             得以缩小。两相比较，由$$\alpha_m=\frac{1}{2}\ln\frac{1-e_m}{e_m}$$ 知误分类样本的权值被放大 $$e^{2\alpha_m}=\frac{1-e_m}{e_m}$$ 被。因
*             此，误分类样本在下一轮学习中起更大的作用。不改变所给的训练数据，而不断改变训练数据
*             权值的分布，使得训练数据在基本分类器的学习中起不同的作用，这就是AdaBoost的一个特点

步骤（3）：线性组合 $$f(x)$$ 实现 $$M$$ 个基本分类器的加权表决。系数 $$\alpha_m$$ 表示了基本分类器 $$G_m(x)$$ 的重要性，这里，所有 $$\alpha_m$$ 之和并不为 $$1$$ 。 $$f(x)$$ 的符号决定实例 $$x$$ 的类， $$f(x)$$ 的绝对值表示分类的确信度。利用基本分类的线性组合构建最终分类器是AdaBoost的另一特点。

### 例子

| 序号 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| x | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| y | 1 | 1 | 1 | -1 | -1 | -1 | 1 | 1 | 1 | -1 |

步骤（1）：初始化数据权值分布

                        $$D_1=(w_{11},w_{12},\dots,w_{110}),\ \ \ w_{1i}=0.1,\ i=1,2,\dots,10$$ 

步骤（2）：

对 $$m = 1$$ ：

* （a）在权值分布为 $$D_1$$ 的训练数据上，阈值 $$v$$ 取 $$2.5$$ 时分类误差率最低，故基本分类器为
*            $$G_1(x)=\begin{cases}1,\ \ \ \ \ x<2.5\\-1,\ \ x>2.5\end{cases}$$ 
* （b） $$G_1(x)$$ 在训练数据集上的误差率 $$e_1=P(G_1(x_i)\neq y_i)=0.3$$ \(6,7,8样本错分，权值均为1\)
* （c）计算 $$G_1(x)$$ 的系数 $$\alpha_1 = \frac{1}{2}\log\frac{1-e_1}{e_1}=0.4236$$ 
* （d）更新训练数据的权值分布
*           $$D_2=(w_{21},w_{22},\dots,w_{210})$$ 
*           $$w_{2i} = \frac{w_{1i}}{Z_1}\exp(-\alpha_1y_iG_1(x_i)),\ \ \ i=1,2,\dots,10$$ 
*           $$D_2 = (0.07143,0.07143,0.07143,0.07143,0.07143,0.07143,0.16667,0.16667,0.16667,0.07143)$$           $$f_1(x)=0.4236G_1(x)$$ 
* 分类器 $$sign[f_1(x)]$$ 在训练数据集上有 $$3$$ 个误分类点。

对 $$m = 2$$ ：

*  （a）在权值分布为 $$D_2$$ 的训练数据上，阈值 $$v$$ 取 $$8.5$$ 时分类误差率最低，故基本分类器为
*            $$G_2(x)=\begin{cases}1,\ \ \ \ \ x<8.5\\-1,\ \ x>8.5\end{cases}$$ 
* （b） $$G_2(x)$$ 在训练数据集上的误差率 $$e_2=0.2143$$ \(4,5,6样本错分，权值均为0.07143\)
* （c）计算 $$G_2(x)$$ 的系数 $$\alpha_2 = \frac{1}{2}\log\frac{1-e_2}{e_2}=0.6496$$ 
* （d）更新训练数据的权值分布
*           $$D_3 = (0.0455,0.0455,0.0455,0.1667,0.1667,0.1667,0.1060,0.1060,0.1060,0.0455)$$     
*           $$f_2(x)=0.4236G_1(x)+0.6496G_2(x)$$ 
* 分类器 $$sign[f_2(x)]$$ 在训练数据集上有 $$3$$ 个误分类点。

对 $$m = 3$$ ：

*  （a）在权值分布为 $$D_3$$ 的训练数据上，阈值 $$v$$ 取 $$5.5$$ 时分类误差率最低，故基本分类器为
*            $$G_3(x)=\begin{cases}1,\ \ \ \ \ x<5.5\\-1,\ \ x>5.5\end{cases}$$ 
* （b） $$G_3(x)$$ 在训练数据集上的误差率 $$e_3=0.1820$$ \(4-9样本错分，权值见 $$D_3$$ \)
* （c）计算 $$G_3(x)$$ 的系数 $$\alpha_3 = \frac{1}{2}\log\frac{1-e_3}{e_3}=0.7514$$ 
* （d）更新训练数据的权值分布
*           $$D_4 = (0.125,0.125,0.125,0.102,0.102,0.102,0.065,0.065,0.065,0.125)$$     
*           $$f_3(x)=0.4236G_1(x)+0.6496G_2(x)+0.7514G_3(x)$$ 
* 分类器 $$sign[f_3(x)]$$ 在训练数据集上有 $$0$$ 个误分类点。

步骤（3）：于是最终分类器为

             $$G(x)=sign[f_3(x)]=sign[0.4236G_1(x)+0.6496G_2(x)+0.7514G_3(x)]$$ 

## AdaBoost训练误差分析

AdaBoost最基本的性质是它能在学习过程中不断减少训练误差，即在训练数据集上的分类学习误差率。关于这个问题有下面的定理：

_AdaBoost的训练误差界_：AdaBoost算法最终分类器的训练误差界为

                                 $$\frac{1}{N}\sum\limits_{i=1}^NI(G(x_i)\neq y_i)\leq\frac{1}{N}\sum\limits_{i}\exp(-y_if(x_i))=\prod\limits_mZ_m$$ 

其中，$$G(x)=sign(f(x))$$，$$f(x)=\sum\limits_{m=1}^M\alpha_mG_m(x)$$， $$Z_m=\sum\limits_{i=1}^Nw_{mi}\exp(-\alpha_my_iG_m(x_i))$$ 

#### 证明：

（1）当 $$G(x_i)\neq y_i$$ 时，不等式左边每个误分权值为 $$1$$ ，不等式右边因为 $$y_if(x_i)<0$$ ，所以每个误分权值 $$\exp(-y_if(x_i))\geq1$$  ，所以不等式 $$\frac{1}{N}\sum\limits_{i=1}^NI(G(x_i)\neq y_i)\leq\frac{1}{N}\sum\limits_{i}\exp(-y_if(x_i))$$ 得证

（2）证等式部分 $$\frac{1}{N}\sum\limits_{i}\exp(-y_if(x_i))=\prod\limits_mZ_m$$

                                $$\frac{1}{N}\sum\limits_{i}\exp(-y_if(x_i))$$

                                  $$=\frac{1}{N}\sum\limits_{i}\exp(-\sum\limits_{m=1}^M\alpha_my_iG_m(x_i))$$ 

由$$w_{m+1,i}=\frac{w_{mi}}{Z_m}\exp(-\alpha_my_iG_m(x_i)),\ \ \ i=1,2,\dots,N$$和$$Z_m=\sum\limits_{i=1}^Nw_{mi}\exp(-\alpha_my_iG_m(x_i))$$ 

                代入移项得到 $$w_{mi}\exp(-\alpha_my_iG_m(x_i))=Z_mw_{m+1,i}$$ ，代入需要证明式子得

                                  $$=\sum\limits_iw_{1i}\prod\limits_{m=1}^M\exp(-\alpha_my_iG_m(x_i))$$ 

                                  $$=Z_1\sum\limits_iw_{2i}\prod\limits_{m=2}^M\exp(-\alpha_my_iG_m(x_i))$$ 

                                  $$=Z_1Z_2\sum\limits_iw_{3i}\prod\limits_{m=3}^M\exp(-\alpha_my_iG_m(x_i))$$ 

                                                         ... ...

                                  $$=Z_1Z_2\dots Z_{M-1}\sum\limits_iw_{Mi}\exp(-\alpha_My_iG_M(x_i))$$ 

                                  $$=\prod\limits_{m=1}^MZ_m$$ 

这一定理说明，可以再每一轮选取适当的 $$G_m$$ 使得 $$Z_m$$ 最小，从而使训练误差下降最快。

## [Code实现](https://github.com/fengdu78/lihang-code/blob/master/code/%E7%AC%AC8%E7%AB%A0%20%E6%8F%90%E5%8D%87%E6%96%B9%E6%B3%95%28AdaBoost%29/Adaboost.ipynb)

### 数据

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

plt.scatter(X[:50,0],X[:50,1], label='0')
plt.scatter(X[50:,0],X[50:,1], label='1')
plt.legend()
```

![](../../../../.gitbook/assets/image%20%2810%29.png)

### 手写实现

```python
class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate
    
    def init_args(self, datasets, labels):
        
        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape
        
        # 弱分类器数目和集合
        self.clf_sets = []
        
        # 初始化weights
        self.weights = [1.0/self.M]*self.M
        
        # G(x)系数 alpha
        self.alpha = []
        
    def _G(self, features, labels, weights):
        m = len(features)
        error = 100000.0 # 无穷大
        best_v = 0.0
        # 单维features
        features_min = min(features)
        features_max = max(features)
        n_step = (features_max - features_min + self.learning_rate) // self.learning_rate
        # print('n_step:{}'.format(n_step))
        direct, compare_array = None, None
        for i in range(1, int(n_step)):
            v = features_min + self.learning_rate * i
            
            if v not in features:
                # 误分类计算
                compare_array_positive = np.array([1 if features[k] > v else -1 for k in range(m)])
                weight_error_positive = sum([weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])
                
                compare_array_nagetive = np.array([-1 if features[k] > v else 1 for k in range(m)])
                weight_error_nagetive = sum([weights[k] for k in range(m) if compare_array_nagetive[k] != labels[k]])

                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'
                    
                # print('v:{} error:{}'.format(v, weight_error))
                if weight_error < error:
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v
        return best_v, direct, error, compare_array
        
    # 计算alpha
    def _alpha(self, error):
        return 0.5 * np.log((1-error)/error)
    
    # 规范化因子
    def _Z(self, weights, a, clf):
        return sum([weights[i]*np.exp(-1*a*self.Y[i]*clf[i]) for i in range(self.M)])
        
    # 权值更新
    def _w(self, a, clf, Z):
        for i in range(self.M):
            self.weights[i] = self.weights[i]*np.exp(-1*a*self.Y[i]*clf[i])/ Z
    
    # G(x)的线性组合
    def _f(self, alpha, clf_sets):
        pass
    
    def G(self, x, v, direct):
        if direct == 'positive':
            return 1 if x > v else -1 
        else:
            return -1 if x > v else 1 
    
    def fit(self, X, y):
        self.init_args(X, y)
        
        for epoch in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None
            # 根据特征维度, 选择误差最小的
            for j in range(self.N):
                features = self.X[:, j]
                # 分类阈值，分类误差，分类结果
                v, direct, error, compare_array = self._G(features, self.Y, self.weights)
                
                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j
                    
                # print('epoch:{}/{} feature:{} error:{} v:{}'.format(epoch, self.clf_num, j, error, best_v))
                if best_clf_error == 0:
                    break
                
            # 计算G(x)系数a
            a = self._alpha(best_clf_error)
            self.alpha.append(a)
            # 记录分类器
            self.clf_sets.append((axis, best_v, final_direct))
            # 规范化因子
            Z = self._Z(self.weights, a, clf_result)
            # 权值更新
            self._w(a, clf_result, Z)
            
#             print('classifier:{}/{} error:{:.3f} v:{} direct:{} a:{:.5f}'.format(epoch+1, self.clf_num, error, best_v, final_direct, a))
#             print('weight:{}'.format(self.weights))
#             print('\n')
    
    def predict(self, feature):
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = feature[axis]
            result += self.alpha[i] * self.G(f_input, clf_v, direct)
        # sign
        return 1 if result > 0 else -1
    
    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            feature = X_test[i]
            if self.predict(feature) == y_test[i]:
                right_count += 1
        
        return right_count / len(X_test)

X = np.arange(10).reshape(10, 1)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

clf = AdaBoost(n_estimators=3, learning_rate=0.5)
clf.fit(X, y)

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = AdaBoost(n_estimators=10, learning_rate=0.2)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# 100次结果
result = []
for i in range(1, 101):
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = AdaBoost(n_estimators=100, learning_rate=0.2)
    clf.fit(X_train, y_train)
    r = clf.score(X_test, y_test)
    # print('{}/100 score：{}'.format(i, r))
    result.append(r)

print('average score:{:.3f}%'.format(sum(result)))
```

### sklearn实现

{% embed url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html" %}

```text
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
clf.fit(X_train, y_train)

clf.score(X_test, y_test)
```

