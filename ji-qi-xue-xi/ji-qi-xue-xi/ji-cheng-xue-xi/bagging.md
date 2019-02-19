# Bagging

Bagging是指的个体学习器间不存在强依赖关系、可同时生成的并行化方法。

Bagging基于自助采样法，给定包含 $$m$$ 个样本的数据集，我们随机取出一个样本放入采样集中，再把该样本放回初始数据集，使得下次采样时该样本仍有可能被选中，这样，经过 $$m$$ 次操作，我们得到含 $$m$$ 个样本的采样集，初始训练集里有的样本在采样集里多次出现，有的则从未出现。由 $$\lim\limits_{m\to\infty}(1-\frac{1}{m})^m\to\frac{1}{e}\approx0.368$$ 可知初始训练集约有 $$63.2\%$$ 的样本出现在采样集中， $$36.8\%$$ 未出现在采样集中。

照这样，我们可采集出 $$T$$ 个含 $$m$$ 个训练样本的训练集，然后基于每个采样集训练出一个基学习器，再将这些学习器进行结合。Bagging通常对分类任务使用简单投票法，对回归任务使用简单平均法。

值得一提的是，自助采样过程还给Bagging带来了另一个有点：由于每个基学习器只使用了初始训练集中约 $$63.2\%$$ 的样本，剩下的样本可用作验证集来对泛化性能进行“包外估计”。

## 随机森林\(Random Forest\)

随机森林在以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程中引入了随机属性选择。具体来说，在随机森林中，对基决策树的每个结点，先从该结点的属性集合中随机选择一个包含 $$k$$ 个属性的子集，然后再从这个子集中选择一个最优属性用于划分。这里的参数 $$k$$ 控制了随机性的引入程度：若 $$k = d$$ ，则基决策树的构建与传统决策树相同；若令 $$k = 1$$ ，则是随机选择一个属性用于划分；一般情况下，推荐值 $$k = \log_2d$$ 。

随机森林的训练效率常常优于纯Bagging思想的决策树群，因为在个体决策树的构建过程中，Bagging使用的是“确定型”决策树，在选择划分属性时要对结点的所有属性进行考察，而随即森林使用的“随机型”决策树则只考察一个属性子集。

可以看出，随机森林对Bagging只做了小改动，但是与Bagging中基学习器的“多样性”仅通过样本扰动（通过对初始训练样本采样）而来不同，随机森林中基学习器的多样性不仅来自样本扰动，还来自属性扰动，这就使得集成的泛化性能可通过个体学习器之间差异度的增加而进一步提升。

### [Code实现](https://github.com/chmx0929/UIUCclasses/blob/master/412DataMining/Assignment/Assignment4/haow4_assign4/code/RandomForest.py)

```python
# -*- coding: UTF-8 -*-
import sys
import math
from DecisionTree import DecisionTreeClassifier

def mymode(X, n_tree, n_samples):
    predictions = []
    for i in range(n_samples):
        d = {}
        for j in range(n_tree):
            x = X[j][i]
            if x not in d:
                d[x] = 0
            d[x] += 1
        res = [ (x, d[x]) for x in d ]
        res = sorted(res, key=lambda x:x[1], reverse=True)
        predictions.append( res[0][0] )
    return predictions 

def load_data(file_name):
    X = []
    y = []
    with open(file_name, "r") as f:
        for l in f:
            sp = l.strip("\n").split(" ")
            label = int(sp[0])
            y.append(label)
            fea = []
            for i in range(1, len(sp)):
                fea.append( float( sp[i].split(":")[1] ) )
            X.append(fea)
    return (X), (y)

def shuffle_in_unison(a, b):
    import random
    random.seed(100)
    all = [ (a[i], b[i]) for i in range(len(a)) ]
    random.shuffle(all)
    na = [ x[0] for x in all ]
    nb = [ x[1] for x in all ]
    return na, nb

def gini(Y):
    distribution = Counter(Y)
    s = 0.0
    total = len(Y)
    for y, num_y in distribution.items():
        probability_y = float (num_y/total)
        s += (probability_y)*math.log(probability_y)
    return -s

def gini_gain(y, y_true, y_false):
    return gini(y) - (gini(y_true)*len(y_true) + gini(y_false)*len(y_false))/len(y)

class RandomForestClassifier(object):
    def __init__(self, n_estimators=32, max_features=lambda x: x, max_depth=20,
        min_samples_split=2, bootstrap=0.632):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.forest = []

    def fit(self, X, y):
        self.forest = []
        n_samples = len(y)
        n_sub_samples = int(round(n_samples*self.bootstrap))
        for i in xrange(self.n_estimators):
            X, y = shuffle_in_unison(X, y)
            X_subset = [ X[i] for i in range(n_sub_samples) ]
            y_subset = [ y[i] for i in range(n_sub_samples) ]

            tree = DecisionTreeClassifier(self.max_features, self.max_depth,
                                            self.min_samples_split)
            tree.fit(X_subset, y_subset)
            self.forest.append(tree)

    def predict(self, X):
        n_samples = len(X)
        n_trees = len(self.forest)
        predictions = [ [ 0 for i in range(n_samples) ] for j in range(n_trees) ]
        for i in xrange(n_trees):
            predictions[i] = self.forest[i].predict(X)
        return mymode(predictions, n_trees, n_samples)

    def score(self, X, y):
        y_predict = self.predict(X)
        n_samples = len(y)
        correct = 0
        for i in xrange(n_samples):
            if y_predict[i] == y[i]:
                correct = correct + 1
        accuracy = correct/float(n_samples)
        return accuracy
    
if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    train_x, train_y = load_data( train_file )
    test_x, test_y = load_data(test_file)
    # print "Load data finish..."
    rf = RandomForestClassifier(n_estimators=32, max_depth=20)
    rf.fit(train_x, train_y)
    # print "test acc:", rf.score(test_x, test_y) 
    preds = rf.predict(test_x)

    # print "confusion matrix:"
    class_num = 0
    class_map = {}
    class_index = 0
    for i in train_y:
        if i not in class_map:
            class_map[i] = class_index
            class_index += 1
            class_num += 1
    for i in preds:
        if i not in class_map:
            class_map[i] = class_index
            class_index += 1
            class_num += 1
    matrix = [ [ 0 for i in range(class_num) ] for j in range( class_num ) ]
    for i in range( len(test_y) ):
        actual = test_y[i]
        pred = preds[i]
        matrix[ class_map[ actual ] ] [ class_map[pred] ] += 1
    for i in matrix:
        print " ".join( [ str(x) for x in i ] )
```

