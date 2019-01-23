# fastText

word2vec是一种无监督模型，而fastText则是对应的有监督模型，都属于Tomas Mikolov的杰作。fastText由Facebook在2016年[开源](https://github.com/facebookresearch/fastText)，学习的目标不再是词语内在的共现，而是人工标注的label。

fastText模型架构如下所示，与word2vec中的CBOW模型类似，只是中间的单词被替换成了标签。其中 $$x_1,x_2,\dots,x_{N-1},x_N$$ 表示一个句子对应的n-gram特征，相比词袋模型，n-gram特征会关注词语的顺序，这些特征被转化为词向量并进行平均从而形成隐层变量，最终的输出层采样Softmax计算相应的概率。

![](../../../.gitbook/assets/v2-7f38f23e98ee89d21fd16e34d5f07d69_hd.jpg)

对于 $$N$$ 个文档的集合，fastText对应的损失函数为：

                                                         $$-\frac{1}{N}\sum\limits_{n=1}^Ny_n\log(f(BAx_n))$$ 

其中 $$x_n$$ 是第 $$n$$ 个文档对应的归一化统计特征， $$A$$ 和 $$B$$ 是权重矩阵，这个模型可以在多个CPU上使用梯度下降方法并行计算。

与word2vec类似，fastText也采用了层次式的分类器，只是word2vec是针对单词的，fastText则针对label

## Source

{% embed url="https://github.com/facebookresearch/fastText" %}

{% embed url="https://zhuanlan.zhihu.com/p/32965521" %}



