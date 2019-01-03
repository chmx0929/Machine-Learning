# 长短期记忆

## RNN长期记忆的依赖问题

RNN的一个特点就是可以结合之前的信息解决当前的任务，比如利用视频前几帧来帮助理解视频当前帧的内容。有时，我们只需要关注近邻的信息就可以解决当前任务，比如我们预测“云彩飘在天”这句话最后一个词，我们很容易得到“天”。这种情况下，相关信息与当前任务距离很小，RNN完全能利用之前信息解决问题

![](../../../.gitbook/assets/rnn-shorttermdepdencies.png)

然而，有很多情况是需要更多地信息才能解决任务。比如我们还是预测最后一个词，但是这次是一大段话“我成长在法国。.... 我能说一口流利的法语”。近邻相关信息确定最后这个词要给一种语言，然而需要这一段的第一句话，一开头的“法国”作为有效信息才能预测出“法语”。这个距离间隔就很大了。很不幸的是，随着间隔距离增大，RNN有时不能学到那些关联的有效信息。

![](../../../.gitbook/assets/rnn-longtermdependencies.png)

理论上，RNN完全能解决这个长期记忆的依赖问题，只需要设置合理的参数就可以。然而实际中，很困难，具体原因可见这两篇paper： [Hochreiter \(1991\) \[German\]](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf) and [Bengio, et al. \(1994\)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf)。

## LSTM网络

长短期记忆网络（Long Short Term Memory networks\(LSTMs\)）是一种特殊的RNN，旨在避免长期记忆的依赖问题。所有循环神经网络都具有神经网络重复模块链的形式。在标准RNNs中，这个重复模块有很简单的结构，比如tanh层，如下图。

![The repeating module in a standard RNN contains a single layer.](../../../.gitbook/assets/lstm3-simplernn.png)

LSTMs也有这样的链式结构，但是重复模块也些许区别。不同于标准RNN的一层（比如上例tanh），LSTM的更新模块具有4个不同的层相互作用，如下图。

![The repeating module in an LSTM contains four interacting layers.](../../../.gitbook/assets/lstm3-chain.png)

![](../../../.gitbook/assets/lstm2-notation.png)

在上图中，每一行都携带一个完整的向量，从一个节点的输出到其他节点的输入。粉色圆圈表示逐点运算，如矢量加法，而黄色框表示神经网络层。行合并表示连接，而行分叉表示其内容被复制，副本将转移到不同的位置。

## LSTM的核心思想

从上一部分的图中可以看出，在每个序列索引位置 $$t$$ 时刻向前传播的除了和RNN一样的隐藏状态 $$h(t)$$ ，还多了另一个隐藏状态，如下图中上面的长横线。这个隐藏状态我们一般称为细胞状态\(Cell State\)，是LSTM的关键。细胞状态有点像传送带。它直接沿着整个链运行，只有一些微小的线性相互作用。信息很容易沿着它不变地流动。

![](../../../.gitbook/assets/lstm3-c-line.png)

LSTM能够移除或添加信息到细胞状态，由称为门\(Gates\)的结构进行调节。门会选择性的让信息穿过。它们是由sigmoid神经网络层和逐点乘法运算组成。sigmoid层输出 $$0$$ 到 $$1$$ 之间的数字，描述每个组件应该通过多少。值为 $$0$$ 意味着“不让任何东西通过”，而值为 $$1$$ 则意味着“让一切都通过！”

![](../../../.gitbook/assets/lstm3-gate.png)

LSTM在在每个序列索引位置 $$t$$ 的门一般包括遗忘门，输入门和输出门三种来保护和控制细胞状态。

## LSTM整体流程



## Source

{% embed url="http://colah.github.io/posts/2015-08-Understanding-LSTMs/" %}

{% embed url="https://blog.csdn.net/anshuai\_aw1/article/details/85168486" %}



