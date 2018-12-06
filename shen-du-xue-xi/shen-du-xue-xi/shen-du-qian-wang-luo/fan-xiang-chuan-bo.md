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

上面式子前两项之所以是Hadamard积（ $$\odot$$ ）

## Source

{% embed url="https://blog.csdn.net/anshuai\_aw1/article/details/84666595" %}



