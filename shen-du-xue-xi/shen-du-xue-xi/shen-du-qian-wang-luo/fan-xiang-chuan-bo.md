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

## Source

{% embed url="https://blog.csdn.net/anshuai\_aw1/article/details/84666595" %}



