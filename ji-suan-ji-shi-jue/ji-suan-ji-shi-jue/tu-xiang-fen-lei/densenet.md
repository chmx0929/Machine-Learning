# DenseNet

ResNet在一定程度上解决了过深模型（比如几百甚至上千层）梯度发散导致无法训练的问题，其关键之处在于层间的快速连接。受此启发，能否进一步增加连接，充分利用所有层的特征呢？[DenseNe](https://github.com/liuzhuang13/DenseNet)t就是这样的模型。

## DenseNet结构

DenseNet模块中的核心模块Dense Block如下图所示，相比ResNet的残差模块，DenseNet具有更多的跨层快捷连接，从输入层开始，每层都作为后面各层的输入。

![](../../../.gitbook/assets/f838717a-6ad1-11e6-9391-f0906c80bc1d.jpg)

在具体实现上，在ResNet中，第 $$l$$ 层的输入 $$x_{l-1}$$ 经过层的转换函数 $$H_l$$ 后得到对应的输出 $$H_l(x_{l-1})$$ ，该输出与输入 $$x_{l-1}$$ 的线性组合就成了下一层的输入 $$x_l$$ 。即：

                                                                $$x_l = H_l(x_{l-1})+x_{l-1}$$ 

而在Dense Block中，第 $$l$$ 层的新增输入 $$x_{l-1}$$ 与之前的所有输入 $$x_0,x_1,\dots,x_{l-3},x_{l-2}$$ 按照通道拼接在一起组成真正的输入，即 $$[x_0,x_1,\dots,x_{l-2},x_{l-1}]$$ ，该输入经过一个Batch Normalization层、ReLU和卷积层得到对应的隐层输出 $$H_l$$ ，该隐层输出就是下一层的新增输入 $$x_l$$ ，即：

                                                     $$x_l=H_l([x_0,x_1,\dots,x_{l-2},x_{l-1}])$$ 

$$x_l$$ 再与之前的所有输入拼接为 $$[x_0,x_1,\dots,x_{l-1},x_{l}]$$ 作为下一层的输入。一般来说，每层新增输入 $$x_l$$ 的通道数量 $$k$$ 都很小，在上图中为 $$4$$ ，原论文中的模型一般取 $$k = 12$$ 。这个新增通道数量 $$k$$ 有一个专门的名字叫增长率（Growth Rate）。由于采用这种拼接方式，同时每个隐层特别瘦（即增长率 $$k$$ 较小），使得DenseNet看起来连接很密集，但实际参数数量及对应运算量反而较少。DenseNet相比ResNet在性能上有一定的优势，在ImageNet分类数据集上达到同样的准确率，DenseNet的参数数量及运算量可能只需要ResNet的一半左右。

最终的DenseNet由Dense Block以及转换层（Transition Layer）组成，转换层一般由一个Batch Normalization层、卷积核大小为1\*1的卷积层和池化层组成，其中1\*1的卷积主要用于瘦身，即降低通道数量。如下图所示，是包含三个Dense Block的DenseNet模型。

![](../../../.gitbook/assets/fa648b32-6ad1-11e6-9625-02fdd72fdcd3.jpg)

## Source

{% embed url="https://github.com/liuzhuang13/DenseNet" %}





