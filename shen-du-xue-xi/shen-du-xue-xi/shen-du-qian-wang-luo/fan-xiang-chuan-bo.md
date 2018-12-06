# 反向传播

当我们使用前馈神经网络接收输入 $$x$$ 并产生输出 $$\hat{y}$$ 时，信息通过网络向前流动。输入 $$x$$ 提供初始信息，然后传播到每一层的隐藏单元，最终产生输出 $$\hat{y}$$ ，这称之为前向传播。在训练过程中，前向传播可以持续向前直到它产生一个标量代价函数 $$J(\theta)$$ 。反向传播算法\(Back Propagation\)，允许来自代价函数的信息通过网络向后流动，以便计算梯度。

## 链式法则

微积分中的链式法则（为了不与概率中的链式法则相混淆）用于计算复合函数的导数。反向传播是一种计算链式法则的算法，使用高效的特定运算顺序。

设 $$x$$ 是实数， $$f$$ 和 $$g$$ 是从实数映射到实数的函数。 $$y=g(x)$$ 并且 $$z=f(g(x))=f(y)$$ 。那么链式法则是说：

                                                                         $$\frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}$$ 

我们可以将这种标量情况进行拓展。假设假设 $$x\in \mathbb{R}^n$$ ， $$y\in \mathbb{R}^n$$ ， $$g$$ 是从 $$\mathbb{R}^n$$ 到 $$\mathbb{R}^n$$ 的映射， $$f$$ 是 $$\mathbb{R}^n$$ 到 $$\mathbb{R}^n$$ 的映射。如果 $$y=g(x)$$ 并且 $$z=f(y)$$ ，那么

                                                                      $$\frac{\partial z}{\partial x_i}=\sum\limits_j\frac{\partial z}{\partial y_i}\frac{\partial y_j}{\partial x_i}$$ 

使用向量记法，可以等价地写成

                                                                     $$\Delta x^z=(\frac{\partial y}{\partial x})^T\Delta y^z$$ 

这里 $$\frac{\partial y}{\partial x}$$ 是 $$g$$ 的 $$n\times m$$ 的Jacobian矩阵。

通常我们将反向传播算法应用于任意维度的张量，而不仅仅用于向量。从概念上讲，这与使用向量的方向传播完全相同。唯一的区别是如何将数字排列称网格以形成张量。我们可以想象，在运行反向传播之前，将每个张量扁平为一个向量，计算一个向量值梯度，然后将该梯度重新构造成一个张量。从这种重新排列的观点上看，反向传播仍然只是将Jacobian乘以梯度。

## 反向传播

在进行DNN反向传播算法前，我们需要选择一个损失函数，来度量训练样本计算出的输出和真实的训练样本输出之间的损失。DNN可选择的损失函数有不少，为了专注算法，这里我们使用最常见的均方差来度量损失。当然，针对不同的任务，可以选择不同的损失函数。即对于每个样本，我们期望最小化下式：

                                                             $$J(W,b,x,y)=\frac{1}{2}||a^{(L)}-y||_2^2$$ 

其中， $$a^{(L)}$$ 和 $$y$$ 为 $$n_{out}$$ 维度的向量，而 $$||S||_2$$ 为 $$S$$ 的 $$L_2$$ 范数。损失函数有了，现在我们开始用梯度下降法迭代求解每一层的 $$W$$ 和 $$b$$ 

#### 第一步

首先是输出层（第 $$L$$ 层）。输出层的 $$W$$ 和 $$b$$ 满足下式：

                                                      $$a^{(L)}=\sigma(z^{(L)})=\sigma(W^{(L)}a^{(L-1)}+b^{(L)})$$ 

这样对于输出层的参数，我们的损失函数变为：

                              $$J(W,b,x,y)=\frac{1}{2}||a^{(L)}-y||_2^2=\frac{1}{2}||\sigma(W^{(L)}a^{(L-1)}+b^{(L)})-y||_2^2$$ 

这样求解 $$W$$ 和 $$b$$ 的梯度就简单了：

      $$\frac{\partial J(W,b,x,y)}{\partial W^{(L)}}=\frac{\partial J(W,b,x,y)}{\partial a^{(L)}}\frac{\partial a^{(L)}}{\partial W^{(L)}}=\frac{\partial J(W,b,x,y)}{\partial a^{(L)}}\frac{\partial a^{(L)}}{\partial z^{(L)}}\frac{\partial z^{(L)}}{\partial W^{(L)}}=(a^{(L)}-y)\odot\sigma'(z^{(L)})(a^{(L-1)})^T$$ 

          $$\frac{\partial J(W,b,x,y)}{\partial b^{(L)}}=\frac{\partial J(W,b,x,y)}{\partial a^{(L)}}\frac{\partial a^{(L)}}{\partial b^{(L)}}=\frac{\partial J(W,b,x,y)}{\partial a^{(L)}}\frac{\partial a^{(L)}}{\partial z^{(L)}}\frac{\partial z^{(L)}}{\partial b^{(L)}}=(a^{(L)}-y)\odot\sigma'(z^{(L)})$$ 

上面式子前两项之所以是Hadamard积 $$\odot$$ 形式，是因为 $$\frac{\partial J(W,b,x,y)}{\partial a^{(L)}}\frac{\partial a^{(L)}}{\partial z^{(L)}}$$ 都是针对同一层的神经元。如果我们考虑对于 $$L$$ 层的第 $$j$$ 个神经元，即 $$\frac{\partial J(W,b,x,y)}{\partial a^{(L)}}\sigma'(z_j^{(L)})$$ ，那么整合这一层的神经元，自然是 $$(a^{(L)}-y)\odot\sigma'(z^{(L)})$$ 这样Hadamard积的形式。

$$(a^{(L-1)})^T$$ 在第一个式子的最后是因为若 $$Y=WX+B$$ ，那么 $$\frac{\partial C}{\partial W}=\frac{\partial C}{\partial Y}X^T$$ 

#### 第二步

我们注意到在求解输出层的 $$W$$ 和 $$b$$ 时，有公共的部分 $$\frac{\partial J(W,b,x,y)}{\partial a^{(L)}}\frac{\partial a^{(L)}}{\partial z^{(L)}}=\frac{\partial J(W,b,x,y)}{\partial z^{(L)}}$$ ，因此我们可以把公共的部分即对 $$z^{(L)}$$ 先算出来，记为

                                                $$\delta^{(L)}=\frac{\partial J(W,b,x,y)}{\partial z^{(L)}}=(a^{(L)}-y)\odot\sigma'(z^{(L)})$$ 

根据第一步的公式我们可以把输出层的梯度计算出来，计算上一层 $$L-1$$ ，上上层 $$L-2$$ ...的梯度就需要步步递推了：对于第 $$l$$ 层的未激活输出 $$z^{(l)}$$ ，它的梯度可以表示为

                                       $$\delta^{(l)}=\frac{\partial J(W,b,x,y)}{\partial z^{(l)}}=\frac{\partial J(W,b,x,y)}{\partial z^{(L)}}\frac{\partial z^{(L)}}{\partial z^{(L-1)}}\frac{\partial z^{(L-1)}}{\partial z^{(L-2)}}\cdots\frac{\partial z^{(l+1)}}{\partial z^{(l)}}$$ 

如果我们可以依次计算出第 $$l$$ 层的 $$\delta^{(l)}$$ ，则该层的 $$W^{(l)}$$ 和 $$b^{(l)}$$ 就很好计算了，因为根据前向传播：

                                                                  $$z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$$ 

所以我们可以很方便的计算出第 $$l$$ 层的 $$W^{(l)}$$ 和 $$b^{(l)}$$ 的梯度如下

                                               $$\frac{\partial J(W,b,x,y)}{\partial W^{(l)}}=\frac{\partial J(W,b,x,y)}{\partial z^{(l)}}\frac{\partial z^{(l)}}{\partial W^{(l)}}=\delta^{(l)}(a^{(l-1)})^T$$ 

                                                        $$\frac{\partial J(W,b,x,y)}{\partial b^{(l)}}=\frac{\partial J(W,b,x,y)}{\partial z^{(l)}}\frac{\partial z^{(l)}}{\partial b^{(l)}}=\delta^{(l)}$$ 

#### 第三步

现在问题的关键就是求 $$\delta^{(l)}$$ 了。这里我们使用数学归纳法，假设第 $$l+1$$ 层的 $$\delta^{(l+1)}$$ 已经求出，那我们如何求第 $$l$$ 层的 $$\delta^{(l)}$$ 呢：

                                         $$\delta^{(l)}=\frac{\partial J(W,b,x,y)}{\partial z^{(l)}}=\frac{\partial J(W,b,x,y)}{\partial z^{(l+1)}}\frac{\partial z^{(l+1)}}{\partial z^{(l)}}=\delta^{(l+1)}\frac{\partial z^{(l+1)}}{\partial z^{(l)}}$$ 

可见，关键在于求解 $$\frac{\partial z^{(l+1)}}{\partial z^{(l)}}$$ ，而 $$z^{(l+1)}$$ 和 $$z^{(l)}$$ 的关系很容易求出：

                                        $$z^{(l+1)}=W^{(l+1)}a^{l}+b^{(l+1)}=W^{(l+1)}\sigma(z^{(l)})+b^{(l+1)}$$ 

这样可得

                                              $$\frac{\partial z^{(l+1)}}{\partial z^{(l)}}=(W^{(l+1)})^T\odot\underbrace{\sigma'(z^{(l)},\dots,\sigma'(z^{(l)})}_{n^{l+1}}$$ 

上式的意思是 $$(W^{l+1})^T$$ 的每一列都是Hadamard积 $$\sigma'(z^{(l)})$$ ，将上式代入 $$\delta^{(l)}=\delta^{(l+1)}\frac{\partial z^{(l+1)}}{\partial z^{(l)}}$$ ，得

             $$\delta^{(l)}=\delta^{(l+1)}\frac{\partial z^{(l+1)}}{\partial z^{(l)}}=\frac{(\delta^{(l+1)})^T z^{(l+1)}}{\partial z^{(l)}}=\frac{(\delta^{(l+1)})^T(W^{(l+1)}\sigma(z^{(l)}+b^{(l+1)})) }{\partial z^{(l)}}=\frac{(\delta^{(l+1)})^TW^{(l+1)}\sigma(z^{(l)}) }{\partial z^{(l)}}$$    

                     $$=((\delta^{l+1})^TW^{(l+1)})^T\odot\sigma'(z^{(l)})=(W^{(l+1)})^T\delta^{(l+1)}\odot\sigma'(z^{(l)})$$ 

#### 总结

其实，对于更新每一层的 $$W^{l},b^{l}$$ 的对应梯度，我们仔细观察整个过程，发现只需要四个公式就可以完整地进行更新。这就是著名的反向传播的四个公式。我们稍加改动，使其可以适用于多种损失函数，即：

                                            $$\delta^{(L)}=\frac{\partial J}{\partial a^{(L)}}\odot\sigma'(z^{(L)})\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ BP(1)$$ 

                                            $$\delta^{(l)}=(W^{(l+1)})^T\delta^{(l+1)}\odot\sigma'(z^{(l)})\ \ \ \ \ \ \ \ \ BP(2)$$ 

                                            $$\frac{\partial J}{\partial W^{(l)}}=\delta^{(l)}(a^{(l-1)})^T\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ BP(3)$$ 

                                            $$\frac{\partial J}{\partial b^{(l)}}=\delta^{(l)}\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ BP(4)$$ 

现在我们总结下DNN反向传播算法的过程。由于梯度下降法有批量（Batch），小批量\(mini-Batch\)，随机三个变种，为了简化描述，这里我们以最基本的批量梯度下降法为例来描述反向传播算法。实际上在业界使用最多的是mini-Batch的梯度下降法。不过区别仅仅在于迭代时训练样本的选择而已。

输入：总层数 $$L$$ ，以及各隐藏层与输出层的神经元个数，激活函数，损失函数，迭代步长 $$\alpha$$ ，最大迭代次数MAX与停止迭代阈值 $$\epsilon$$ ，输入的 $$m$$ 个训练样本 $$\{(x_1,y_1),(x_2,y_2),\dots,(x_m,y_m)\}$$ 

输出：各隐藏层与输出层的线性关系系数矩阵 $$W$$ 和偏倚向量 $$b$$ （即输出模型）

1\) 初始化各隐藏层与输出层的线性关系系数矩阵 $$W$$ 和偏倚向量 $$b$$ 的值为一个随机值

2\) for 1 to MAX：

              2-1\) for $$i = 1\ to\ m$$ ：

                                a\) 将DNN输入 $$a^{1}$$ 设置为 $$x_i$$ 

                                b\) for $$l = 2\ to\ L$$ ，进行前向传播算法计算 $$a^{i,l}=\sigma(z^{i,l})=\sigma(W^la^{i,l-1}+b^l)$$ 

                                c\) 通过损失函数计算输出层的 $$\delta^{i,L}$$ （BP1）

                                d\) for $$l = 2\ to\ L$$ ，进行反向传播算法计算 $$\delta^{i,l}=(W^{l+1})^T\delta^{i,l+1}\odot\sigma'(z^{i,l})$$ （BP2）

              2-2\) for $$l = 2\ to\ L$$，更新第 $$l$$ 层的 $$W^l,b^l$$ ：

                                  $$W^l=W^l-\alpha\sum\limits_{i=1}^m\delta^{i,l}(a^{i,l-1})^T$$ （BP3）

                                  $$b^l=b^l-\alpha\sum\limits_{i=1}^m\delta^{i,l}$$ （BP4）

              2-3\) 如果所有 $$W,b$$ 的变化值都小于停止迭代阈值 $$\epsilon$$ ，则跳出迭代循环到步骤3

3\) 输出各隐藏层与输出层的线性关系系数矩阵 $$W$$ 和偏倚向量 $$b$$ （即输出模型）

## Source

{% embed url="https://blog.csdn.net/anshuai\_aw1/article/details/84666595" %}



