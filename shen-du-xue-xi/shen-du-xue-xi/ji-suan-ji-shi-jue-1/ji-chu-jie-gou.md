# 基础结构

## 基本组成单元

卷积\(Convolution\)、激活函数\(Active Function\)、池化\(Pooling\)、全连接\(Softmax\)

## 神经网络需要解决的问题

1、特征表达

2、模型优化（过拟合、欠拟合）

#### AlexNet提出的方案 <a id="alexnet-ti-chu-de-fang-an"></a>

1、修正线性单元\(Rectified Linear Unit, ReLU\) --&gt; 加速训练和收敛

2、Dropout --&gt; 降低过拟合

3、数据增强 --&gt; 降低过拟合

4、双GPU训练 --&gt; 加快训练速度

## 卷积神经网络 <a id="juan-ji-shen-jing-wang-luo"></a>

### 卷积\(Convolution\) <a id="juan-ji-convolution"></a>

![&#x4E8C;&#x7EF4;&#x5377;&#x79EF;&#x793A;&#x610F;&#x56FE;](../../../.gitbook/assets/image%20%284%29.png)

![padding                                                strides                                                        dilation](../../../.gitbook/assets/timline-jie-tu-20180911092534.png)

![transposed                             padding+strides                   padding+strides+transposed](../../../.gitbook/assets/timline-jie-tu-20180911092703.png)

### 激活函数\(Active Function\) <a id="ji-huo-han-shu-active-function"></a>

模仿人类神经元、非线性变换增强特征的表达能力、同时考虑优化时梯度消失问题 \(轻轻抚摸一个人可能感觉不到，但重击一个人会有明显感觉，激活函数作相似事情，增加特征表达能力\)

![sigmoid&amp;tanh](../../../.gitbook/assets/timline-jie-tu-20180911092931.png)

![&#x4E8C;&#x7EF4;&#x60C5;&#x51B5;&#x4E0B;&#xFF0C;&#x4F7F;&#x7528;ReLU&#x4E4B;&#x540E;&#x7684;&#x6548;&#x679C;](../../../.gitbook/assets/timline-jie-tu-20180911093043.png)

![&#x6FC0;&#x6D3B;&#x51FD;&#x6570;&#x7684;&#x53D1;&#x5C55;](../../../.gitbook/assets/timline-jie-tu-20180911093135.png)

### 池化\(Pooling\) <a id="chi-hua-pooling"></a>

 降低计算复杂度，增强特征的空间变换不变性

![Max&#x3001;Min&#x3001;Random pooling](../../../.gitbook/assets/timline-jie-tu-20180911093221.png)

### 全连接\(Softmax\) <a id="quan-lian-jie-softmax"></a>

 全连接操作的特点：需要固定维度、参数多，计算量大，占整个网络的参数量、计算量的一半以上

                                    Softmax： $$f(z_j) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k} } , \ \ for j= 1,2\dots, K$$ 

![&#x4E00;&#x7EF4;&#x5168;&#x8FDE;&#x63A5;                                                                                     &#x4E8C;&#x7EF4;&#x5168;&#x8FDE;&#x63A5;](../../../.gitbook/assets/timline-jie-tu-20180911093406.png)

### Dropout <a id="dropout"></a>

以一定概率丢弃全连接层中的节点，用以对抗过拟合问题

![Dropout&#x7B80;&#x56FE;&#x793A;&#x4F8B;](../../../.gitbook/assets/tu-pian-1%20%281%29.png)

![Dropout\(&#x6FC0;&#x6D3B;&#x51FD;&#x6570;&#x4E4B;&#x540E;\) Dropconnect\(&#x6FC0;&#x6D3B;&#x51FD;&#x6570;&#x4E4B;&#x524D;\)](../../../.gitbook/assets/tu-pian-2%20%283%29.png)

