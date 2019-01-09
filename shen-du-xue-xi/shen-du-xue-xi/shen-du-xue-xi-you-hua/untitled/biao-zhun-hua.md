# 标准化

Normalization，即标准化，和普通的数据标准化类似，是将分散的数据统一的一种做法，也是优化神经网络的一种方法。Normalization可以将数据统一规格，能让机器学习更容易学习到数据之中的规律。

如果我们仅仅停留在使用Normalization上，那么现成的框架只需要一行就可以添加到模型中。我们真正想知道的是，隐藏在BN的背后深度学习的问题，以及这样简单的操作是如何奏效的。

## 深度学习的Normalization具体做什么

由于有多种Normalization的方法，如Batch Normalization（BN），Layer Norm（LN），Weight Norm（WN），Cosine Norm\(CN\)等，我这里以最经典的Batch Normalization为例说明Normalization到底是做的什么操作。需要知道的是，BN可以在激活函数 $$\sigma$$ 之前，也可以在 $$\sigma$$ 之后。 有如下图的神经网络：

![](../../../../.gitbook/assets/1042406-20170220142015116-1152957081%20%281%29.png)

## 为什么深度学习中需要Normalization

## Normalization 的通用框架与基本思想

## 主流 Normalization 方法梳理

## Normalization 为什么会有效

## 对Batch Normalization的实践及分析

### Batch Normalization对前向传播影响

Batch Normalization在前向传播时有三个主要任务：

* 计算出每批训练数据的统计量
* 对数据进行标准化
* 对标准化后的数据进行扭转，将其映射到表征能力更大的空间上

#### Batch Normalization在Mini-Batch上的变换算法

输入：每个Mini-Batch上的 $$x$$ 值 $$X=\{x_{1},x_2\dots ,x_m\}$$ ，需要学习的参数 $$\gamma,\beta$$ 

输出： $$\{y_n=BN_{\gamma,\beta}(X_n)\}$$ 

              （1）计算每个Mini-Batch均值： $$\text{for} \ n=1\ \text{to}\ N\$$： $$ \mu_n \gets \frac{1}{m}\sum\limits_{i=1}^mx_{n,i}$$ 

              （2）计算每个Mini-Batch方差： $$\text{for} \ n=1\ \text{to}\ N\$$： $$\sigma^2_n = \frac{1}{m}\sum\limits_{i=1}^m(x_{n,i}-\mu_n)^2$$ 

              （3）每个Mini-Batch标准化： $$\text{for} \ n=1\ \text{to}\ N\$$： $$\hat{x}_{n,i} = \frac{x_{n,i}-\mu_n}{\sqrt{\sigma^2_n+\epsilon}}$$ 

              （4）缩放与位移： $$y_n \gets \gamma\hat{X_n}+\beta$$ 即 $$BN_{\gamma,\beta}(X_n)$$ 

#### 1、从Mini-Batch计算均值与方差

SGD通过最小化以下损失函数来求解神经网络的最优值：

                                                         $$\Theta = \mathop{\arg\min}\limits_{\Theta}\frac{1}{N}\sum\limits_{i=1}^Nl(x_i,\Theta)$$ 

其中 $$\Theta$$ 是神经网络中的参数； $$\{x_1,x_2,\dots,x_N\}$$ 是训练数据集。

为了综合随机梯度下降和批量梯度下降这两种算法的优点，Mini-Batch梯度下降在单个样本迭代和全部样本迭代之间找到了一个折中点，既加快了参数的迭代速度，也避免了单个样本数据带来的波动性。当Mini-Batch中的数据量为 $$m$$ 时，可以通过如下公式计算出梯度：

                                                           $$\Delta\Theta=\eta\cdot \frac{1}{m}\sum\limits_{i=1}^m\frac{\partial l(x_i,\Theta)}{\partial \Theta}$$ 

其中 $$\eta$$ 为学习率。

可以证明，通过小批量样本计算出的梯度可以正确地表示全部训练数据的梯度，平切没皮的数据量越大，样本梯度越接近于总体梯度。另外，得益于现代平行计算技术，使用Mini-Batch比 $$m$$ 词单样本的计算效率更高，因此比原始的随机梯度下降方法更快。

#### 2、从批量样本推断总体均值与方差

网络模型在训练阶段与测试阶段使用的统计量是不同的。训练时，每个batch使用本批的统计量来进行标准化，测试时则需要总体的统计量对数据进行转换。总体统计量可以通过如下公式从每批的样本统计量的移动平均数中推断出来。值得注意的是，推断方差时需要加上 $$m/(m-1)$$ 的校正，因为样本方差均值的 $$\frac{m}{m-1}$$ 倍才是总体方差的无偏估计。

对于多个Mini-Batch的训练集 $$X$$ ，每个batch大小为 $$m$$ ：

                       总体均值： $$E[x]\gets E_X[\mu_X]$$       总体方差： $$\text{Var}[x]\gets \frac{m}{m-1}E_X[\sigma^2_X]$$ 

无偏估计是指估计量的期望值与估计参数的真实值偏差为零，我们可以通过如下推导证明 $$E_X[\mu_X]$$ 是 $$E[x]$$ 的无偏估计， $$\frac{m}{m-1}E_X[\sigma^2_X]$$ 是 $$\text{Var}[x]$$ 的无偏估计。

         $$E_X[\mu_X]=E[\frac{1}{n}\sum\limits_{i=1}^nX_i]=\frac{1}{mn}\sum\limits_{i=1}^{mn}x_i=E[x]$$ 这里设有 $$n$$ 个batch，每个batch有 $$m$$ 样本

在推导方差期望之前，我们首先要了解样本均值的方差公式。

                          $$\text{Var}[\mu_X]=\text{Var}[\frac{1}{m}\sum\limits_{i=1}^mX_i]$$ 

## Source

{% embed url="https://blog.csdn.net/anshuai\_aw1/article/details/84975689\#Batch\_Normalization\_284" %}





