# 图像分类

CS231n的博客里的描述网络结构的layer pattern，一般常见的网络都可以表示为：

           $$\text{Input}\to [[\text{Conv}\to \text{ReLU}]*N\to \text{Pool}?]*M\to [\text{FC}\to \text{ReLU}]*K\to \text{FC}$$ 

其中 $$\text{Pool}$$ 后面的 $$?$$ 表示 $$\text{Pool}$$ 是一个可选项。这样的pattern因为可以对小卷积核堆叠，很自然也更适合描述深层网络的构建，例如 $$\text{Input}\to\text{FC}$$ 表示一个线性分类器。

串联和串联中带有并联的网络架构。近年来，GoogLeNet在其网络结构中引入了Inception模块，ResNet中引入了Residual Block，这些模块都有自己复杂的操作。换句话说，传统一味地去串联网络可能并不如这样串联为主线，带有一些并联同类操作但不同参数的模块可能在特征提取上更好。 所以这里本质上依旧是在做特征工程，只不过把这个过程放在block或者module的小的网络结构里，毕竟kernel、stride、output的大小等等超参数都要自己设置，目的还是产生更多丰富多样的特征。

在这里推荐[何恺明18年CVPR的讲座](http://kaiminghe.com/cvpr18tutorial/cvpr2018_tutorial_kaiminghe.pdf)，串讲了几个经典模型和batch normalization。比较高屋建瓴，可以点链接看一下。总体来说：

* （1）第一个得到广泛关注的 AlexNet，它本质上就是扩展 LeNet 的深度，并应用一些 ReLU、Dropout 等技巧。AlexNet 有 5 个卷积层和 3 个最大池化层，它可分为上下两个完全相同的分支，这两个分支在第三个卷积层和全连接层上可以相互交换信息。与Inception同年提出的优秀网络还有VGG-Net，它相比于AlexNet有更小的卷积核和更深的层级。
* （2）VGG-Net的泛化性能非常好，常用于图像特征的抽取目标检测候选框生成等。VGG最大的问题就在于参数数量，VGG-19基本上是参数量最多的卷积网络架构。这一问题也是第一次提出 Inception 结构的GoogLeNet所重点关注的，它没有如同VGG-Net那样大量使用全连接网络，因此参数量非常小。
* （3）GoogLeNet最大的特点就是使用了Inception模块，它的目的是设计一种具有优良局部拓扑结构的网络，即对输入图像并行地执行多个卷积运算或池化操作，并将所有输出结果拼接为一个非常深的特征图。因为1\*1、3\*3或 5\*5 等不同的卷积运算与池化操作可以获得输入图像的不同信息，并行处理这些运算并结合所有结果将获得更好的图像表征。

LeNet-5告诉我们深度学习在图像上可行；AlexNet告诉我们深度学习大体框架；VGGNet告诉我们用小感受野（小的卷积核）只要深度足够依旧能达到大感受野（大卷积核）的效果；GoogLeNet告诉我们越接近低层（输入层）使用小卷积核捕捉细节，越远离低层使用大卷积核捕捉更抽象特征，另外采用不同尺度会更符合直觉（图片内关键区域大部分不是等大的）；ResNet告诉我们如何解决超深神经网络梯度爆炸/消失的方案。

用在ImageNet上pre-trained过的模型。设计自己模型架构很浪费时间，尤其是不同的模型架构需要跑数据来验证性能，所以不妨使用别人在ImageNet上训练好的模型，然后在自己的数据和问题上在进行参数微调，收敛快精度更好。 我认为只要性能好精度高，选择什么样的模型架构都可以，但是有时候要结合应用场景，对实时性能速度有要求的，可能需要多小网络，或者分级小网络，或者级联的模型，或者做大网络的知识蒸馏得到小网络，甚至对速度高精度不要求很高的，可以用传统方法。

ZFNet，DPN这两个个网络（13、17年ILSVRC冠军）后期有时间我会继续进行整理。

## Source

{% embed url="http://cs231n.stanford.edu/" %}

{% embed url="https://mp.weixin.qq.com/s?\_\_biz=MzAwNDI4ODcxNA==&mid=2652246142&idx=1&sn=4e479a9b7f8be21b657efc997eb841e6&scene=0" %}

{% embed url="https://www.jiqizhixin.com/articles/2018-05-30-7?from=synced&keyword=resnet" %}

{% embed url="http://kaiminghe.com/cvpr18tutorial/cvpr2018\_tutorial\_kaiminghe.pdf" %}





