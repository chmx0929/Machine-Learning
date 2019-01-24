# 文档维度情感分析

文档级别的情感分类是将文档确定其整体情绪方向/极性，即确定文档（例如，完整的在线评论）是否传达整体正面或负面意见。在此设置中，它是二分类任务（正面/负面）。它也可以被表述为回归任务，例如，推断用于评价的1到5星的总体评分（1分负面一直至5分正面）。一些研究人员还将此视为5分类任务。

情感分类通常被视为文档分类的特例。在这种分类中，文档表示起着重要作用，所以文档表示应该反映文档中单词或句子传达的原始信息。传统上，词袋模型（BoW）用于在NLP和文本挖掘中生成文本表示。基于BoW，文档被转换为具有固定长度的数字特征向量，其每个元素可以是单词出现与否（缺席或存在, 0/1），单词频率或TF-IDF分数。这个向量的维度等于词汇量的大小。来自BoW的文档向量通常非常稀疏，因为单个文档仅包含词汇表中的少量单词。早期神经网络采用了这种特征设置。

尽管上述方法很受欢迎，但BoW有一些缺点。首先，忽略单词顺序，这意味着只要它们共享相同的单词，两个文档就可以具有完全相同的表示。 Bag-of-N-Grams是BoW的扩展，可以在短语境（n-gram）中考虑单词顺序，但它也会受到数据稀疏性和高维度的影响。其次，BoW几乎不能编码单词的语义。例如，“smart”，“clever”和“book”等词在BoW中它们之间的距离相等，但“smart”应该在语义上更接近“clever”而不是“book”。

为了解决BoW的缺点，提出了基于神经网络的文字嵌入技术（CBOW、Skip-Gram、word2vec等）来生成用于单词表示的密集向量（或低维向量），这在某种程度上能够编码一些单词的属性（语义和句法等）。利用单词嵌入作为单词的输入，可以使用神经网络导出作为密集向量（或称为密集文档向量）的文档表示。

注意，除了上述两种方法，即使用BoW和通过字嵌入学习文档的密集向量之外，还可以直接从BoW学习密集文档向量。我们区分下图中展示相关研究中使用的不同方法。

![](../../../.gitbook/assets/timline-jie-tu-20190123145417.png)

当文档被恰当地表示时，可以遵循传统的监督学习设置使用各种神经网络模型来进行情感分类。在一些情况下，神经网络可以仅用于提取文本特征/文本表示，并且这些特征被馈送到一些其他非神经分类器（例如，SVM）以获得最终的全局最佳分类器。神经网络和SVM的特性以它们的优点相结合的方式相互补充。

除了复杂的文档/文本表示外，研究人员还利用数据的特征 - 产品评论，进行情感分类。对于产品评论，一些研究人员发现共同模拟情绪和一些额外信息（例如，用户信息和产品信息）进行分类是有益的。另外，由于文档通常包含长依赖关系，因此注意机制也经常用于文档级情感分类。我们总结了上面的图表2中的现有技术。

## 各技术简介

#### [Document-level sentiment classification: An empirical comparison between SVM and ANN](https://www.sciencedirect.com/science/article/pii/S0957417412009153)

Moraes等人对支持向量机（SVM）和人工神经网络（ANN）进行了文档级情感分类的实证比较，证明了ANN在大多数情况下和SVM结果可以媲美。

#### [Distributed Representations of Sentences and Documents](https://arxiv.org/abs/1405.4053)

为了克服BoW的弱点，Le和Mikolov提出了Paragraph Vector，这是一种无监督学习算法，可以学习句子、段落和文档等可变长度文本的矢量表示。通过预测从段落中采样的上下文中的周围单词来学习矢量表示。

#### [Domain adaptation for large-scale sentiment classification a deep learning approach](http://www.icml-2011.org/papers/342_icmlpaper.pdf)

Glorot等人研究了情绪分类的领域适应问题。他们提出了一种基于具有稀疏整流器单元的Stacked Deoising Autoencoder的深度学习系统，该系统可以使用标记和未标记的数据执行无监督的文本特征/表示提取。这些特征非常有利于情感分类器的域适应。

#### [Semi-supervised autoencoder for sentiment analysis](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12059/11750)

Zhai和Zhang介绍了一种半监督自动编码器，它进一步考虑了学习阶段的情绪信息，以获得更好的文档向量，用于情感分类。更具体地，该模型通过将自动编码器中的损失函数放宽到[Bregman divergence](https://en.wikipedia.org/wiki/Bregman_divergence)并且还从标签信息导出判别性损失函数来学习文本数据的任务特定表示。



## Source

{% embed url="https://arxiv.org/abs/1801.07883" %}

