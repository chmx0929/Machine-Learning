# 前向传播

假设我们选择的激活函数是 $$\sigma(z)$$ ；隐藏层和输出层的输入值为 $$z$$ ，上标表示第几层，下标表示本层单元的index；隐藏层和输出层的输出值为 $$a$$ ，上标表示第几层，下标表示本层单元的index。则对于下图的三层DNN，利用和感知机一样的思路，我们可以利用上一层的输出计算下一层的输出，也就是所谓的DNN前向传播算法。

![](../../../.gitbook/assets/1042406-20170220142015116-1152957081.png)

对于第二层的输出 $$a_1^{(2)},a_2^{(2)},a_3^{(2)}$$， $$w$$ 的下标第一个数表示当前层的index，第二个数表示前一层的index，上标 $$(2)$$ 代表第一层到第二层，我们有

                                     $$a_1^{(2)}=\sigma(z_1^{(2)})=\sigma(w^{(2)}_{11}x_1+w_{12}^{(2)}x_2+w_{13}^{(2)}x_3+b_1^{(2)})$$ 

                                     $$a_2^{(2)}=\sigma(z_1^{(2)})=\sigma(w^{(2)}_{21}x_1+w_{22}^{(2)}x_2+w_{23}^{(2)}x_3+b_3^{(2)})$$ 

                                     $$a_3^{(2)}=\sigma(z_1^{(2)})=\sigma(w^{(2)}_{31}x_1+w_{32}^{(2)}x_2+w_{33}^{(2)}x_3+b_3^{(2)})$$ 

对于第三层的输出 $$a_1^{(3)}$$ ，我们有

                                      $$a_1^{(3)}=\sigma(z_1^{(3)})=\sigma(w^{(3)}_{11}x_1+w_{12}^{(3)}x_2+w_{13}^{(3)}x_3+b_1^{(3)})$$ 

将上面的例子一般化，假设第 $$l-1$$ 层共有 $$m$$ 个神经元，则对于第 $$l$$ 层的第 $$j$$ 个神经元的输出 $$a^{l}_j$$ ：

                                                  $$a^{(l)}_j=\sigma(z^{(l)}_j)=\sigma(\sum\limits_{k=1}^mw^{(l)}_{jk}a^{(l-1)}_k+b^{(l)}_j)$$ 

可以看出，使用代数法一个个的表示输出比较复杂，而如果使用矩阵法则比较的简洁。假设第 $$l-1$$ 层共有 $$m$$ 个神经元，而第 $$l$$ 层共有 $$n$$ 个神经元，则第 $$l$$ 层的线性系数 $$w$$ 组成了一个 $$n\times m$$ 的矩阵 $$W^l$$ ，第 $$l$$ 层的偏倚 $$b$$ 组成了一个 $$n\times 1$$ 的向量 $$b^l$$ ，第 $$l-1$$ 层的输出 $$a$$ 组成了一个 $$m \times 1$$ 的向量 $$a^{(l-1)}$$ ，第 $$l$$ 层的未激活前线性输出 $$z$$ 组成了一个 $$n\times 1$$ 的向量 $$z^{(l)}$$ ，第 $$l$$ 层的输出 $$a$$ 组成了一个 $$n\times 1$$ 的向量 $$a^{(l)}$$ 。则用矩阵法表示，第 $$l$$ 层的输出为：

                                                    $$a^{(l)}=\sigma(z^{(l)})=\sigma(W^{(l)}a^{(l-1)}+b^{(l)})$$ 

所谓的DNN的前向传播算法也就是利用我们的若干个权重系数矩阵 $$W$$ 和偏倚向量 $$b$$ 来和输入值向量 $$x$$ 进行一系列线性运算和激活运算，从输入层开始，一层层的向后计算，一直到输出层，得到输出结果为止。

输入：总层数 $$L$$ ，所有隐藏层和输出层对应的矩阵 $$W$$ ，偏倚向量 $$b$$ ，输入值向量 $$x$$ 

输出：输出层的输出 $$a^{(L)}$$ 

* （1）初始化 $$a^{(1)}=x$$ 
* （2）for $$l$$ in $$(2,L)$$ ：
*                      $$a^{(l)}=\sigma(z^{(l)})=\sigma(W^{(l)}a^{(l-1)}+b^{(l)})$$ 

最后结果即输出 $$a^{(L)}$$ 

## Source

{% embed url="https://blog.csdn.net/anshuai\_aw1/article/details/84615935" %}

