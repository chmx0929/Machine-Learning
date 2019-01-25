# 方面维度情感分析

与文档级别和句子级别情绪分类不同，方面级别情绪分类同时考虑情绪和目标信息，因为情绪总是有目标。如前所述，目标通常是实体或实体方面。为简单起见，实体和方面通常都称为方面。给定句子和目标方面，方面级别情绪分类旨在推断句子朝向目标方面的情感极性/方向。例如，在句子“屏幕非常清晰，但电池寿命太短。”如果目标方面是“屏幕”，则情绪是积极的，但如果目标方面是“电池寿命”，则情绪是否定的。

方面级别情绪分类具有挑战性，因为难以将目标的语义相关性与其周围的上下文单词建模。不同的语境词对句子对目标的情感极性有不同的影响。因此，在使用神经网络构建学习模型时，必须捕获目标词和上下文词之间的语义连接。

使用神经网络在方面级别情感分类中有三个重要任务。第一个任务是表示目标的上下文，其中上下文表示句子或文档中的上下文单词。使用上面两节中提到的文本表示方法可以类似地解决该问题。第二个任务是生成目标表示，该表示可以与其上下文正确交互。一般解决方案是学习目标嵌入，类似于嵌入字。第三项任务是确定指定目标的重要情绪背景（单词）。例如，在“iPhone的屏幕清晰但电池寿命短”的句子中，“清晰”是“屏幕”的重要背景词，“短”是“电池寿命”的重要背景。注意机制最近解决了这项任务。尽管已经提出了许多深度学习技术来处理方面级别的情感分类，但据我们所知，文献中仍然没有主导技术。相关工作及其主要重点介绍如下。

## 技术简介

#### [Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification](http://www.aclweb.org/anthology/P14-2009)

Dong等人提出了一种自适应递归神经网络（AdaRNN），用于依赖于目标的推特情绪分类，它学习根据语境和句法结构将词语的情感传播到目标。它使用根节点的表示作为特征，并将它们提供给softmax分类器以预测类的分布。

#### [Target-Dependent Twitter Sentiment Classification with Rich Automatic Features](https://www.ijcai.org/Proceedings/15/Papers/194.pdf)

Vo和Zhang通过利用丰富的自动功能来研究基于方面的Twitter情绪分类，这是使用无监督学习方法获得的附加功能。该论文表明，多个嵌入，多个池功能和情感词典可以提供丰富的特征信息来源，并有助于实现性能提升。

#### [Effective LSTMs for Target-Dependent Sentiment Classification](https://www.aclweb.org/anthology/C/C16/C16-1311.pdf)

由于LSTM可以更灵活地捕获目标与其上下文单词之间的语义关系，因此Tang等人提出了目标依赖LSTM（TD-LSTM）和目标连接LSTM（TC-LSTM），通过将目标纳入LSTM来扩展LSTM考虑。他们将给定目标视为一个特征，并将其与方面情感分类的上下文特征连接起来。

#### [A Hierarchical Model of Reviews for Aspect-based Sentiment Analysis](https://arxiv.org/abs/1609.02745)

Ruder等人提出使用分层和双向LSTM模型进行方面级别情感分类，该模型能够利用句内和句子间关系。在评论中对句子及其结构的唯一依赖使得所提出的模型语言无关。 Word嵌入被送入句子级双向LSTM。前向和后向LSTM的最终状态与目标嵌入连接在一起，并被送入双向检查级LSTM。在每个时间步骤，前向和后向LSTM的输出被连接并馈送到最终层，其输出情绪的概率分布。

#### [Gated Neural Networks for Targeted Sentiment Analysis](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12074/12065)

考虑到Dong等人和Vo和Zhang的模型的局限性，Zhang等人提出了一种句子级神经模型来解决池函数的弱点，它没有明确地模拟推文级语义。为此，提出了两个门控神经网络。首先，双向门控神经网络用于连接推文中的单词，以便可以在隐藏层上应用池化功能而不是单词，以更好地表示目标及其上下文。其次，使用三向门控神经网络结构来模拟目标提及与其周围环境之间的相互作用，通过使用门控神经网络结构来模拟封闭推文的语法和语义以及之间的交互来解决限制。周围环境和目标分别。已经显示门控神经网络通过更好的梯度传播来减少标准递归神经网络对序列末端的偏差。

#### [Attention-based LSTM for Aspect-level Sentiment Classification](https://aclweb.org/anthology/D16-1058)

Wang等人提出了一种基于注意力的LSTM方法，该方法具有目标嵌入，这被证明是一种有效的方法来强制神经模型来处理句子的相关部分。注意机制用于强制模型以响应特定方面的句子的重要部分。同样，Yang等人提出了两种基于注意的双向LSTM来改善分类性能。 [Liu和Zhang](http://leoncrashcode.github.io/Documents/EACL2017.pdf)通过区分从左上下文获得的注意力和给定目标/方面的正确上下文来扩展注意力建模。他们通过增加多个门来进一步控制他们的注意力。

#### [Aspect Level Sentiment Classification with Deep Memory Network](https://arxiv.org/abs/1605.08900)

Tang等人介绍了用于方面级别情感分类的端到端存储器网络，其采用具有外部存储器的注意机制来捕获关于给定目标方面的每个上下文字的重要性。在推断方面的情感极性时，该方法明确地捕获每个上下文单词的重要性。这种重要度和文本表示是用多个计算层计算的，每个计算层都是外部存储器上的神经注意模型。

#### [Rationalizing Neural Predictions](https://people.csail.mit.edu/taolei/papers/emnlp16_rationale.pdf)

Lei等人提出使用神经网络方法来提取输入文本的片段作为评论评级的理由（原因）。该模型由发生器和解码器组成。生成器指定可能的基本原理（提取的文本）的分布，并且编码器将任何这样的文本映射到任务特定的目标向量。对于多方面情绪分析，目标向量的每个坐标表示与相关方面有关的响应或评级。

#### [Deep Memory Networks for Attitude Identification](https://arxiv.org/abs/1701.04189)

LI等人将目标识别任务整合到情感分类任务中，以更好地模拟情感 - 情感交互。他们表明，情感识别可以通过端到端机器学习架构来解决，其中两个子任务由深存储器网络交织。以这种方式，在目标检测中产生的信号提供用于极性分类的线索，并且相反地，预测的极性提供对目标的识别的反馈。

#### [Interactive Attention Networks for Aspect-Level Sentiment Classification](https://www.ijcai.org/proceedings/2017/0568.pdf)

Ma等人提出了一种交互式注意网络（IAN），它既考虑了目标，也考虑了上下文。也就是说，它使用两个注意网络来交互地检测目标表达/描述的重要单词以及其完整上下文的重要单词。

#### [Recurrent Attention Network on Memory for Aspect Sentiment Analysis](https://www.aclweb.org/anthology/D/D17/D17-1047.pdf)

Chen等人提出利用反复关注网络来更好地捕捉复杂情境的情绪。为实现这一目标，他们提出的模型使用循环/动态注意结构，并学习GRU中注意力的非线性组合。

#### [Dyadic Memory Networks for Aspect-based Sentiment Analysis](https://dl.acm.org/citation.cfm?id=3132936)

Tay等人设计了一种二元记忆网络（DyMemNN），通过使用神经张量成分或全息成分进行记忆选择操作来模拟方面和背景之间的二元相互作用。

## 目标提取及分类

要进行方面级别情绪分类，需要具有可以手动给定或自动提取的方面（或目标）。在本节中，我们将使用深度学习模型从句子或文档中讨论自动方面提取（或方面术语提取）的现有工作。让我们用一个例子说明问题。例如，在句子“图像非常清晰”中，单词“图像”是一个方面术语（或情感目标）。方面分类的相关问题是将相同的方面表达式分组到类别中。例如，方面术语“图像”，“照片”和“图片”可以被分组为名为Image的一个方面类别。在下面的回顾中，我们简单概括一下与意见相关的方面和实体的提取方法。

深度学习模型可以帮助完成这项任务的一个原因是，深度学习本质上擅长学习（复杂）特征表示。当在一些特征空间中适当地表征方面时，例如，在一个或一些隐藏层中，可以利用它们对应的特征表示之间的相互作用来捕获方面与其上下文之间的语义或相关性。换句话说，深度学习提供了一种可行的自动化特征工程方法，无需人为参与。

### 技术简介

[Katiyar和Cardie](http://www.aclweb.org/anthology/P16-1087)研究了使用深度双向LSTM来联合提取意见实体以及连接实体的IS-FORM和IS-ABOUT关系。[Wang等人](https://arxiv.org/abs/1603.06679)进一步提出了一种集成RNN和条件随机场（CRF）的联合模型，以共同提取方面和意见术语或表达。所提出的模型可以同时学习高级判别特征并在方面和意见术语之间双重传播信息。[Wang等人](http://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/15-Wang-W-14441.pdf)又进一步提出了一种耦合多层注意模型（CMLA），用于共同提取方面和意见术语。该模型包括使用GRU单元的方面注意和意见关注。[Li和Lam](https://www.aclweb.org/anthology/D/D17/D17-1310.pdf)提出了一种改进的基于LSTM的方法，专门用于方面术语提取。它由三个LSTM组成，其中两个LSTM用于捕获方面和情感交互。第三个LSTM将使用情绪极性信息作为附加指导。

[He等人](https://www.comp.nus.edu.sg/~leews/publications/acl17.pdf)提出了一种基于注意力的无监督方面提取模型。主要的直觉是利用注意机制更多地关注与方面相关的单词，同时在学习方面嵌入期间强调与方面无关的单词，类似于自动编码器框架。

[Zhang等人](http://aclweb.org/anthology/D15-1073)利用神经网络扩展了CRF模型，共同提取方面和相应的情绪。所提出的CRF变体用连续字嵌入替换CRF中的原始离散特征，并在输入和输出节点之间添加神经层。

[Zhou等人](https://pdfs.semanticscholar.org/08bc/c29f8c827550c64061917d7f3bc4e57d0e69.pdf)提出了一种半监督的单词嵌入学习方法，用于在具有噪声标签的大量评论中获得连续的单词表示。通过学习单词矢量，通过神经网络堆叠单词矢量来学习更深和混合的特征。最后，使用利用混合特征训练的逻辑回归分类器来预测方面类别。

[Yin等人](https://arxiv.org/pdf/1605.07843.pdf)首先通过考虑连接词的依赖路径来学习单词嵌入。然后他们设计了一些嵌入功能，这些功能考虑了基于CRF的方面术语提取的线性上下文和依赖关系上下文信息。

[Xiong等人](https://arxiv.org/abs/1604.08672)提出了一种基于注意力的深度距离度量学习模型来对方面短语进行分组。基于注意力的模型是学习上下文的特征表示。方面短语嵌入和上下文嵌入都用于学习Kmeans聚类的深度特征子空间度量。

[Poria等人](http://ww.w.sentic.net/aspect-extraction-for-opinion-mining.pdf)提出使用CNN进行方面提取。他们开发了一个七层深度卷积神经网络，将固定句子中的每个单词标记为方面或非方位词。一些语言模式也被整合到模型中以进一步改进。

[Ying等人](https://pdfs.semanticscholar.org/d083/41562091ac6777f613a68a0d59eb600b5c57.pdf)提出了两种基于RNN的跨域方面提取模型。他们首先使用基于规则的方法为每个句子生成辅助标签序列。然后，他们使用真实标签和辅助标签训练模型，这显示了有希望的结果。

## Source

{% embed url="https://arxiv.org/abs/1801.07883" %}

