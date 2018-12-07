# 文本向量化

文本向量化就是将文本表示成一系列能够表达文本语义的向量。

## Continuous Bag-of-Word Model

Continuous Bag-of-Word\(CBOW\)顾名思义，即连续词袋模型，即文本以单个词为最小单位，像“support vector machine”词组也会被当做三个独立的词考虑，且是连续词袋，即目标词的前后词也作为因素考虑。

### One-word context

#### 模型结构

下图为网络模型例子，词汇表大小为 $$V$$；隐藏层宽度为 $$N$$\(即我们想要的词向量维度\)，各层连接为全连接方式；输入为one-hot编码的向量，即词汇表出现的 $$V$$ 个非重复词，一个词 $$w$$ 的向量 $$(x_1,x_2,\dots,x_V)$$ 为对应 $$x_w$$ 的位置为 $$1$$ ，其他位置都为 $$0$$ ；真实的 $$y$$ 为文本中输入词的下一个词的one-hot编码的向量。

![](../../.gitbook/assets/timline-jie-tu-20181204100447.png)

输入层和隐藏层间的权重可由一个 $$V\times N$$ 的矩阵 $$W$$ 表示。 $$W$$ 的每一行是一个 $$N$$ 维向量，表示输入层对应的词向量 $$v_w$$ 。

                                                       $$W =  \left[  \begin{matrix}    w_{11} \ \  w_{12} \ \ \dots\ \ w_{1N}\\  w_{21} \ \  w_{22} \ \ \dots\ \ w_{2N}\\  \dots \ \  \dots \ \ \dots\ \ \dots\\  w_{V1} \ \  w_{V2} \ \ \dots\ \ w_{VN}\\  \end{matrix}   \right] $$ 

$$W$$ 的第 $$i$$ 行是 $$v_w^T$$ ，给定一个词 $$x_k=1$$ 且 $$x_{k'}=0$$ 对于 $$k'\neq k$$ （即这个词的one-hot向量只有 $$k$$ 位置为 $$1$$ ），我们可得：

                                                           $$h=W^Tx=W^T_{(k,\cdot)}:=v^T_{w_I}$$                                                                 （1）

其实就是将 $$W$$ 的第 $$k$$ 行复制给了 $$h$$ ，因为 $$x$$ 只有在第 $$k$$ 位置是 $$1$$ （因为输入是one-hot，经过矩阵相乘其实就是把权重 $$W$$ 对应行的值传递给下一层）。 $$v_{w_I}$$ 即是输入词 $$w_I$$ 的向量表示。（这就意味着隐藏层的激活函数是线性的即可，不需要使用ReLU之类的对它们进行非线性变换。比如Multi-word context model中直接把这层的输入进行加权求和传给下层）

隐藏层到输出层的权重可用一个 $$N\times V$$ 的矩阵 $$W'=\{w'_{ij}\}$$ 表示：

                                                     $$W' =  \left[  \begin{matrix}    w'_{11} \ \  w'_{12} \ \ \dots\ \ w'_{1N}\\  w'_{21} \ \  w'_{22} \ \ \dots\ \ w'_{2N}\\  \dots \ \  \dots \ \ \dots\ \ \dots\\  w'_{V1} \ \  w'_{V2} \ \ \dots\ \ w'_{VN}\\  \end{matrix}   \right] $$ 

基于权重，我们对于每一个词汇表里的词可计算一个分数 $$u_j$$：

                                                                          $$u_j=v_{w_j}'^T h$$                                                                                  （2）

其中 $$v'_{w_j}$$ 是 $$W'$$ 第 $$j$$ 列。然后我们用softmax去获得这个词的后验分布，是一个多项式分布：

                                                        $$p(w_j|w_I)=y_j=\frac{\exp(u_j)}{\sum\limits_{j'=1}^V\exp (u_{j'})}$$                                                                 （3）

其中 $$y_j$$ 是输出层第 $$j$$ 个单元的输出。结合输入层到隐藏层 $$h=W^Tx=W^T_{(k,\cdot)}:=v^T_{w_I}$$ 和隐藏层到输出层 $$u_j=v_{w_j}'^T h$$ 公式代入softmax，我们得到：

                                                          $$p(w_j|w_I)=\frac{\exp(v_{w_j}'^T)v_{w_I}}{\sum\limits_{j'=1}^V\exp(v_{w_{j'}'}'^Tv_{w_I})}$$ 

这里 $$v_w$$ 和 $$v_w'$$ 是词 $$w$$ 的两种表达形式。 $$v_w$$ 源自输入层到隐藏层权重矩阵 $$W$$ 的行， $$v_w'$$ 源自隐藏层到输出层权重矩阵 $$W'$$ 的列。我们将 $$v_w$$ 和 $$v'_w$$ 分别称为“输入向量”和“输出向量”。

模型目标是最大化 $$p(w_j|w_I)=\frac{\exp(v_{w_j}'^T)v_{w_I}}{\sum\limits_{j'=1}^V\exp(v_{w_{j'}'}'^Tv_{w_I})}$$ ，即模型输入 $$w_I$$，模型输出$$w_O$$\(表示它的index在输出层为$$j^*$$\) 与真实$$y$$\(输入词的下一个词的one-hot向量\)一致。即 $$y$$ 向量第 $$k$$ 位为 $$1$$，其他为 $$0$$ ，我们期望的最佳模型是输出层第 $$k$$ 个单元为 $$1$$ ，其他为 $$0$$ 。模型使用反向传播进行训练。

#### 模型训练

1）隐藏层到输出层权重更新

训练目标即最大化 $$p(w_j|w_I)=\frac{\exp(v_{w_j}'^T)v_{w_I}}{\sum\limits_{j'=1}^V\exp(v_{w_{j'}'}'^Tv_{w_I})}$$ ，即给定输入词 $$w_I$$ 的情况下和权重的情况下输出 $$w_O$$ 的条件概率最大：

                $$\max p(w_O|w_I)=\max y_{j^*}=\max \log y_{j^*} = u_{j^*}-\log \sum\limits_{j'=1}^V\exp(u_{j'}):= -E$$ 

上式给了损失函数的定义，即 $$E = -\log p(w_O|w_I)$$ ，我们旨在最小化 $$E$$ 。还有其中 $$j^*$$ 是输出层实际输出词的index。

通过对输出层第 $$j$$ 个单元的输入 $$u_j$$ 的偏导数，我们可得

                                                                      $$\frac{\partial E}{\partial u_j}=y_j-t_j:=e_j$$ 

上式给出了 $$e_j$$ 的定义，其中 $$t_j$$ 只有在第 $$j$$ 个单元是所期待的输出词（即真实的 $$y$$ ）时才为 $$1$$ ，其他情况下为 $$0$$ 。这个导偏数其实就是表示在输出层的预测误差 $$e_j$$ 。

下一步我们取 $$w'_{ij}$$ 的偏导数以获得隐藏层到输出层权重的梯度

                                                                  $$\frac{\partial E}{\partial w'_{ij}}=\frac{\partial E}{\partial u_j}\frac{\partial u_j}{\partial w'_{ij}}=e_j\cdot h_i$$ 

因此，用随机梯度下降法，我们可以得到隐藏层到输出层的权重更新公式：

                                                               $$w'^{(new)}_{ij}=w'^{(old)}_{ij}-\eta\cdot e_j\cdot h_i$$ 

或者                                    $$v'^{(new)}_{w_j}=v'^{(old)}_{w_j}-\eta\cdot e_j\cdot h\ \ \ \ for\ j=1,2,\dots,V$$  

其中 $$\eta>0$$ 是学习率， $$e_j=y_j-t_j$$ ， $$h_i$$ 是隐藏层第 $$i$$ 个单元； $$v'_{w_j}$$ 是 $$w_j$$ 的输出向量。这个更新公式其实就表明了我们需要查看词汇表中每一个可能的词，比较网络的输出$$y_j$$与期望的输出\(实际值\) $$t_j$$：

* 如果 $$y_j>t_j$$ （估计过高），

### Multi-word context

## Source

{% embed url="https://arxiv.org/pdf/1411.2738.pdf" %}

{% embed url="https://blog.csdn.net/lanyu\_01/article/details/80097350" %}





