# 推荐系统

## 推荐系统概述

推荐系统是由系统来根据用户过去的行为、用户的属性（年龄、性别、职业等）、上下文等来猜测用户的兴趣，给用户推送物品（包括电商的宝贝、新闻、电影等）。推荐一般是非主动触发的，和搜索的主动触发不一样（搜索一般有主动提供的query）；也有小部分情况，用户有提供query：例如用户在电商里搜索一个商品后，结果页最底下推荐给用户的商品，这种情况算是有提供query。

直观写成公式，即 $$f(X_{\text{User Feature}},Y_{\text{Item Feature}})\to 0/1$$ 。通过用户特征信息和候选推荐品找到这样一个映射， $$0$$ 表示未点击或未购买等， $$1$$ 表示点击有浏览了或下单购买了。

![](../../../.gitbook/assets/640.jpeg)

推荐系统涉及到的两大实体：user和item往往是不同类型的东西，例如user是人，item是电商的宝贝，他们表面上的特征可能是没有任何的重叠的，这不同于搜索引擎里的query和doc都是文本：

![](../../../.gitbook/assets/640.jpg)

正是由于user和item不同类型的东西，所以推荐里匹配可能比搜索里的更难。

## 传统匹配模型

###  **Collaborative Filtering Models**

协同过滤（CF）是推荐里最通用、最著名的算法了。CF的基本假设是：一个用户的行为，可以由跟他行为相似的用户进行推测。协同过滤一般分为user-based和item-based、model-based三类方法。user-based和item-based、model-based的协同过滤的算法的思想通俗的讲，就是：

* user-based：两个用户A和B很相似，那么可以把B购买过的商品推荐给A（如果A没买过）；例如你和你的师兄都是学机器学习的，那么你的师兄喜欢的书籍，你也很有可能喜欢
* item-based: 两个item：A和B很相似，那么某个用户购买过A，则可以给该用户推荐B。例如一个用户购买过《模式识别》这本书，它很有可能也喜欢《推荐系统》这本书。计算两个item是否相似的一种简单方法是，看看他们的共现概率，即他们被用户同时购买的概率
* model-based: 用机器学习的思想来建模解决，例如聚类、分类、PLSA、矩阵分解等

CF要解决的问题用数学化的语言来表达就是：一个矩阵的填充问题，已经打分的item为observed data，未打分的是missing data：

![](../../../.gitbook/assets/timline-jie-tu-20190318101806.png)



###  **Generic Feature-based Models**

## **深度学习匹配模型**

###  **基于representation learning的方法**

####  **基于Collaborative Filtering的方法**

####  **基于Collaborative Filtering + Side Info的方法**

###  **基于matching function learning的方法**

###  **representation learning和matching function learning的融合**

## Source

{% embed url="https://zhuanlan.zhihu.com/p/45849695" %}

{% embed url="https://www.comp.nus.edu.sg/~xiangnan/sigir18-deep.pdf" %}



