# Deep Crossing

微软于2016年提出的[Deep Crossing](https://github.com/wzhe06/Reco-papers/blob/master/Deep%20Learning%20Recommender%20System/%5BDeep%20Crossing%5D%20Deep%20Crossing%20-%20Web-Scale%20Modeling%20without%20Manually%20Crafted%20Combinatorial%20Features%20%28Microsoft%202016%29.pdf)可以说是**深度学习CTR模型的最典型和基础性的模型**。如图2的模型结构图所示，它涵盖了深度CTR模型最典型的要素，即通过加入embedding层将稀疏特征转化为低维稠密特征，用stacking layer，或者叫做concat layer将分段的特征向量连接起来，再通过多层神经网络完成特征的组合、转换，最终用scoring layer完成CTR的计算。跟经典DNN有所不同的是，Deep crossing采用的multilayer perceptron是由残差网络组成的，这无疑得益于MSRA著名研究员何恺明提出的著名的152层ResNet。

![](../../../../../.gitbook/assets/v2-cef2b96858c05b98d698786884cfe891_hd.jpg)

## Source

{% embed url="https://zhuanlan.zhihu.com/p/63186101" %}



