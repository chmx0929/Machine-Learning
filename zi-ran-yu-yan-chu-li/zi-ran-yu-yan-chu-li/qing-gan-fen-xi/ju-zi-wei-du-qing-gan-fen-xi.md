# 句子维度情感分析

句子级别情绪分类是确定单个给定句子中表达的情绪。如前所述，句子的情感可以用主观性分类和极性分类来推断，其中前者分类句子是主观的还是客观的（即是否含有情感），后者决定主观句子是否表达消极或积极的情绪。在现有的深度学习模型中，句子情感分类通常被制定为联合三向分类问题，即将句子预测为正向，中立和否定。

与文档级别情感分类相同，由神经网络产生的句子表示对于句子级别情感分类也很重要。另外，由于句子通常与文档相比较短，因此可以使用一些句法和语义信息（例如，解析树，意见词典和词性标签）来帮助。还可以考虑其他信息，例如评论评级，社交关系和跨域信息。例如，社交关系被用于发现社交媒体数据（如tweet）中的情绪。

在早期研究中，解析树（提供一些语义和句法信息）与原始单词一起用作神经模型的输入，从而可以更好地推断情感构成。但最近，CNN和RNN变得越来越流行，他们不需要解析树来从句子中提取特征。相反，CNN和RNN使用单词嵌入作为输入，它已经编码了一些语义和句法信息。此外，CNN或RNN的模型体系结构也可以帮助学习句子中单词之间的内在关系。相关工作将在下面详细介绍。

## 各技术简介

#### [Semi-supervised recursive autoencoders for predicting sentiment distributions](http://www.aclweb.org/anthology/D11-1014)

Socher首先提出了一种用于句子级别情感分类的半监督递归自动编码器网络（RAE），其获得了句子的简化向量表示。后来，Socher等人提出了一种矩阵向量递归神经网络（MVRNN），其中每个字还与树结构中的矩阵表示（除了矢量表示）相关联。树结构从外部解析器获得。在[Socher之后的工作中](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)，作者进一步介绍了递归神经张量网络（RNTN），其中基于张量的组合函数用于更好地捕获元素之间的相互作用。[Qian等人提出了两种更先进的模型](http://www.aclweb.org/anthology/P15-1132)，Tag引导递归神经网络（TG-RNN），它根据短语的词性标签和Tagembedded递归神经网络/递归神经网络（TE-）选择组合函数。 RNN / RNTN），它学习标签嵌入，然后将标签和字嵌入组合在一起。

#### [A Convolutional Neural Network for Modelling Sentences](http://www.aclweb.org/anthology/P14-1062)

Kalchbrenner等人提出了一种动态CNN（称为DCNN）用于句子的语义建模。 DCNN使用动态K-Max池算子作为非线性子采样函数。网络引起的特征图能够捕获单词关系。[Kim的论文](https://www.aclweb.org/anthology/D14-1181)中建议使用CNN进行句子级别的情感分类，并尝试使用几种变体，即CNN-rand（其中字嵌入被随机初始化），CNN-static（其中字嵌入是预训练和固定的），CNN-非静态（其中字嵌入是预先训练和微调的）和CNN-多通道（其中使用多组字嵌入）。

#### [Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts](http://anthology.aclweb.org/C/C14/C14-1008.pdf)

dos Santos和Gatti提出了一个字符到句子CNN（CharSCNN）模型。CharSCNN使用两个卷积层从任何大小的单词和句子中提取相关特征，以执行短文本的情感分析。Wang等人通过模拟组成过程中单词的相互作用，利用LSTM进行Twitter情绪分类。通过门结构进行字嵌入之间的乘法运算用于提供更大的灵活性，并且与简单的递归神经网络中的加性结果相比产生更好的组成结果。与双向RNN类似，通过允许隐藏层中的双向连接，可以将单向LSTM扩展为[双向LSTM](ftp://ftp.idsia.ch/pub/juergen/nn_2005.pdf)。

#### [Dimensional Sentiment Analysis Using a Regional CNN-LSTM Model](http://anthology.aclweb.org/P16-2037)

Wang等人提出了一个区域CNN-LSTM模型，它由两部分组成：区域CNN和LSTM，用于预测文本的价值唤醒评级。

#### [Combination of Convolutional and Recurrent Neural Network for Sentiment Analysis of Short Texts](http://www.aclweb.org/anthology/C16-1229)

Wang等人描述了用于短文本情感分类的联合CNN和RNN架构，其利用了CNN生成的粗粒度局部特征和通过RNN学习的长距离依赖性。

#### [CNN- and LSTM-based Claim Classification in Online User Comments](https://pdfs.semanticscholar.org/c250/b11a5909baebe5de3195d6ddcdacc809fda7.pdf)

Guggilla等人提出了一种基于LSTM和CNN的深度神经网络模型，该模型利用word2vec和语言嵌入进行声明分类（将句子分类为事实或感觉）。

#### [Encoding Syntactic Knowledge in Neural Networks for Sentiment Classification](https://dl.acm.org/citation.cfm?id=3052770)

Huang等人提出在树结构化LSTM中编码句法知识（例如，词性标签）以增强短语和句子表示。

#### [A Multilayer Perceptron based Ensemble Technique for Fine-grained Financial Sentiment Analysis](https://www.aclweb.org/anthology/D17-1057)

Akhtar等人提出了几种基于多层感知器的集成模型，用于金融微博和新闻的精细情感分类。

#### [Weakly-supervised deep learning for customer review sentiment classification](https://dl.acm.org/citation.cfm?id=3061139)

Guan等人使用弱监督的CNN进行句子（以及方面）级别情绪分类。它包含一个两步学习过程：它首先学习一个由整体评论评级弱监督的句子表示，然后使用句子（和方面）级别标签进行微调。

#### [Context-Sensitive Lexicon Features for Neural Sentiment Analysis](https://aclweb.org/anthology/D16-1169)

Teng等人提出了一种基于上下文敏感词典的情感分类方法，该方法基于简单的加权和模型，使用双向LSTM来学习构成句子情感值时词汇情绪的情感强度，强化和否定。

#### [Learning Sentence Embeddings with Auxiliary Tasks for Cross-Domain Sentiment Classification](https://aclweb.org/anthology/D16-1023)

Yu和Jiang研究了跨域句子情感分类中学习广义句嵌入的问题，并设计了一个包含两个分离的CNN的神经网络模型，共同学习标记和未标记数据的两个隐藏特征表示。

#### [Microblog Sentiment Classification via Recurrent Random Walk Network Learning](https://www.ijcai.org/proceedings/2017/0494.pdf)

Zhao等人通过利用用户发布的推文及其社交关系的深层语义表示，介绍了一种针对固定推文的情感分类的循环随机游走网络学习方法。

#### [Learning Cognitive Features from Gaze Data for Sentiment and Sarcasm Classification using Convolutional Neural Network](https://aclanthology.info/papers/P17-1035/p17-1035)

Mishra等人利用CNN从阅读文本的人类读者的眼动（或凝视）数据中自动提取认知特征，并将其用作丰富的特征以及用于情感分类的文本特征。

#### [Linguistically Regularized LSTMs for Sentiment Classification](https://arxiv.org/abs/1611.03949)

Qian等人提出了一个语言规范化的LSTM来完成这项任务。所提出的模型将情感词典，否定词和强度词等语言资源纳入LSTM，以更准确地捕捉句子中的情感效应。

## Source

{% embed url="https://arxiv.org/abs/1801.07883" %}

