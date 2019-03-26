# DIN

Deep Interest Network\(DIN\)模型是阿里在2017年发表的用户兴趣分布网络。与上面的FNN,PNN等引入低阶代数范式不同，DIN的核心是基于数据的内在特点，引入了更高阶的学习范式。互联网上用户兴趣是多种多样的，从数学的角度来看，用户的兴趣在兴趣空间是一个多峰分布。在预测多兴趣的用户点击某个商品的概率时，其实用户的很多兴趣跟候选商品是无关的，也就是说我们只需要考虑用户跟商品相关的局部兴趣。所以DIN网络结构引入了兴趣局部激活单元，它受attention机制启发，从用户大量的行为集合中捕获到与candidate商品相关的行为子簇，对于用户行为子簇，通过Embedding操作，做weighted sum便可很好的预估出用户与candidate相关的兴趣度。传统的GwEN、WDL、FNN等模型在刻画用户兴趣分布时，会简单的将用户兴趣特征组做sum或average的pooling操作，这会把用户真正相关的兴趣淹没在pooling过程中。DIN模型的具体细节可以参考论文：[Zhou et al, “Deep Interest Network for click-through rate prediction”](https://arxiv.org/pdf/1706.06978.pdf) 。

![](../../../../../.gitbook/assets/v2-8c386a725488fd5013c37af0d052ffdd_r.jpg)

## Source

{% embed url="https://zhuanlan.zhihu.com/p/51623339" %}

{% embed url="https://zhuanlan.zhihu.com/p/54822778" %}

{% embed url="https://mp.weixin.qq.com/s/MtnHYmPVoDAid9SNHnlzUw" %}

{% embed url="https://zhuanlan.zhihu.com/p/37562283" %}

{% embed url="https://zhuanlan.zhihu.com/p/34940250" %}



