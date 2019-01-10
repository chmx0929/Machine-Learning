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

## Batch Normalization的分析

### 对前向传播影响

Batch Normalization在前向传播时有三个主要任务：

* 计算出每批训练数据的统计量
* 对数据进行标准化
* 对标准化后的数据进行扭转，将其映射到表征能力更大的空间上

#### Batch Normalization在Mini-Batch上的变换算法

![](../../../../.gitbook/assets/bn_algorithm.PNG)

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

#### 3、数据标准化

数据标准化又叫作数据归一化，是数据挖掘过程中常用的数据预处理方式。当我们使用真实世界中的数据进行分析时，会遇到两个问题：

* 特征变量之间的量纲单位不同
* 特征变量之间的变化尺度\(scale\)不同

特征变量的尺度不同导致参数尺度规模也不同，带来的最大问题就是在优化阶段，梯度变化会产生震荡，减慢收敛速度。经过标准化的数据，各个特征变量对梯度的影响变得统一，梯度的变化会更加稳定。

![](../../../../.gitbook/assets/1670644-2b87fba30d7d8c39.webp)

总结起来，数据标准化有以下三个优点：

* 数据标准化能够是数值变化更稳定，从而使梯度的数量级不会变化过大。
* 在某些算法中，标准化后的数据允许使用更大的步长，以提高收敛地速度。
* 数据标准化可以提高被特征尺度影响较大的算法的精度，比如k-means、kNN、带有正则的线性回归。

#### 4、缩放与偏移\(Scale-Shift\)

标准化后的数据还需要进行一定的缩放与偏移等变换，即在标准化后的数据上放大或缩小一定的比例并向上或向下平移一定的距离，变换公式为 $$y_i= \gamma\hat{x}_i+\beta$$ 。

### 对后向传播影响

#### 1、后向传播过程

后向传播是一种在训练时与梯度下降方法结合使用的优化算法，是训练人工神经网络的常用方法。该方法计算网络中所有权重的损失函数的梯度。计算出的梯度会被送给优化方法，优化方法又使用梯度来更新权重，以试图使损失函数最小化。

通过前面的介绍可知，前向传播可以分为三步：将数据标准化、对数据进行线性偏移、将数据输出到下一层，即 $$\hat{x}(\mu,\sigma^2,x)\to y(\hat{x},\gamma,\beta)\to l$$ （其中， $$\hat{x}$$ 是标准化后的输入， $$y$$ 是 $$\hat{x}$$ 的线性变换， $$l$$  代表BN的下一层）。同样地，反向传播就是按照相反地三步传递误差，即 $$l \to y(\hat{x},\gamma,\beta)\to \hat{x}(\mu,\sigma^2,x)$$ 。所以求导得过程也依次为 $$\frac{\partial l}{\partial y}\to \frac{\partial y}{\partial \hat{x}},\frac{\partial y}{\partial \gamma},\frac{\partial y}{\partial \beta}\to \frac{\partial \hat{x}}{\partial \sigma^2},\frac{\partial \hat{x}}{\partial \mu},\frac{\partial l}{\partial x_i}$$ 。如下图所示：

![](../../../../.gitbook/assets/bncircuit.png)

#### 2、梯度的推导

以下是链式法则的基本公式。

假设 $$u = u(x,y),\ x=x(r,t),\ y=y(r,t)$$，则 $$\frac{\partial u}{\partial r}=\frac{\partial u}{\partial x}\cdot\frac{\partial x}{\partial r}+\frac{\partial u}{\partial y}\cdot\frac{\partial y}{\partial r}$$ 

假设从下一层产生的误差为 $$l$$ ，则本层产生的误差为 $$\frac{\partial l}{\partial y}$$ 

（1）计算 $$\frac{\partial y}{\partial \hat{x}},\frac{\partial y}{\partial \gamma},\frac{\partial y}{\partial \beta}$$ 

* （a）$$\frac{\partial l}{\partial \gamma}=\sum\limits_{i=1}^m\frac{\partial l}{\partial y_i}\cdot\frac{\partial y_i}{\partial \gamma}=\sum\limits_{i=1}^m\frac{\partial l}{\partial y_i}\cdot \hat{x}_i$$ 
* （b）$$\frac{\partial l}{\partial \beta}=\sum\limits_{i=1}^m\frac{\partial l}{\partial y_i}\cdot \frac{\partial y_i}{\partial \beta}=\sum\limits_{i=1}^m\frac{\partial l}{\partial y_i}$$ 
* （c）$$\frac{\partial l}{\partial \hat{x}_i}=\frac{\partial l}{\partial y_i}\cdot\frac{\partial y_i}{\partial \hat{x}_i}=\frac{\partial l}{\partial y_i}\cdot \gamma$$ 

（2）计算 $$\frac{\partial \hat{x}}{\partial \sigma^2},\frac{\partial \hat{x}}{\partial \mu},\frac{\partial l}{\partial x_i}$$ 

* （a）$$\frac{\partial l}{\partial \sigma^2}=\frac{\partial l}{\partial \hat{x}}\cdot \frac{\partial \hat{x}}{\partial \sigma^2}$$ ，其中 $$\hat{x}_i=(x_i-\mu)(\sigma^2+\epsilon)^{-0.5}$$ 

  （b）$$\frac{\partial l}{\partial \mu}=\frac{\partial l}{\partial \hat{x}_i}\cdot\frac{\partial \hat{x}_i}{\partial \mu}+\frac{\partial l}{\partial \sigma^2}\cdot\frac{\partial \sigma^2}{\partial \mu}$$ ，其中 $$\hat{x}_i=\frac{(x_i-\mu)}{\sqrt{\sigma^2+\epsilon}}$$ ， $$\frac{\partial \hat{x}_i}{\partial \mu}=\frac{1}{\sqrt{\sigma^2+\epsilon}}\cdot(-1)$$ ， $$\sigma^2=\frac{1}{m}\sum\limits_{i=1}^m(x_i-\mu)^2$$ ， $$\frac{\partial \sigma^2}{\partial \mu}=\frac{1}{m}\sum\limits_{i=1}^m2\cdot(x_i-\mu)\cdot(-1)$$ 

* （c） $$\frac{\partial l}{\partial x_i}=\frac{\partial l}{\partial \hat{x}_i}\cdot\frac{\partial \hat{x}_i}{\partial x_i}+\frac{\partial l}{\partial \mu}\cdot\frac{\partial \mu}{\partial x_i}+\frac{\partial l}{\partial \sigma^2}\cdot\frac{\partial \sigma^2}{\partial x_i}$$ ，其中 $$\frac{\partial \hat{x}_i}{\partial x_i}=\frac{1}{\sqrt{\sigma^2+\epsilon}}$$ ， $$\frac{\partial \mu}{\partial x_i}=\frac{1}{m}$$ ， $$\frac{\partial \sigma^2}{\partial x_i}=\frac{2(x_i-\mu)}{m}x_i$$ 

### 有效性分析

从前面的算法介绍中可以看出，BN的原理十分简单，就是对每批的训练数据进行标准化，再做适当的线性变换，但是却可以显著提高深度神经网络的训练效果。一方面，BN可以有效地减轻神经网络的内部协移\(Internal Covariate Shift\)；另一方面，BN可以有效地改善训练过程中的梯度流的变化，解决梯度消失的问题，加快收敛。同时，BN也在一定程度上起到了 $$L^2$$ 正则的作用。

#### 1、内部协移

内部协移\(Internal Covariate Shift\)是由于神经网络中每层的输入发生了变化，造成每层的参数要不断地适应新分布的问题。传统的Covariate Shift问题是指经过学习系统后的输入数据的分布发生改变，是典型的迁移学习的问题。Internal Covariate Shift与其相似，数据分布变化的来源从学习系统变成神经网络层。假设有一个两层的神经网络模型，第一层的映射函数是 $$F_1$$ ，第二层的映射函数是 $$F_2$$ ，由此得到的输出为： $$l=F_2(F_1(u,\Theta_1),\Theta_2)$$ 。于是，我们可以得到梯度下降的公式

                                                             $$\Theta_2\gets \Theta_2-\frac{a}{m}\sum\limits_{i=1}^m\frac{\partial F_2(x_i,\Theta_2)}{\partial \Theta_2}$$ 

其中， $$x = F_1(u,\Theta_1)$$ ， $$m$$ 依然是批量数据的大小， $$\alpha$$ 是学习率。

可以看出，经过两层神经网络之后，数据已经发生很复杂的变化，这将导致每层神经元的参数都要不断地调整适应这种输入数据分布的变化，不仅使网络的收敛速度变慢，也使得每层超参数的设定变得更加复杂

BN可以在数据经过多层神经网络后，重新回到均值为 $$0$$ 、方差为 $$1$$ 的分布上，解决了以上问题，使数据的变化分布变得稳定，训练过程也随之变得平稳，超参数的调整变得简单。

#### 2、梯度流

梯度流\(Gradient Flow\)，是指梯度按照最陡峭的路径逐渐减小的流动变化，在梯度下降方法中用来描述梯度的变化过程。

（1）梯度去哪了

深度神经网络主要是通过反向传播算法来寻找最优解的，即将梯度从损失函数层向各层反向传递。在传递过程中，如果梯度没有稳定地下降，就有可能导致产生梯度爆炸或梯度消失的现象。这个问题是由于梯度通过多层神经网络传递导致最后一层产生级数积累的结果。假设网络模型中的每一层接收的梯度都是上一层的 $$K$$ 倍，那么 $$L$$ 层神经网络将会产生相对原有输入 $$K^L$$ 倍的变化，当 $$K>1$$ 时，最后一层的输入会非常大；而当 $$K<1$$ 时，最后一层的输入会变得非常小，比如 $$0.9^{10}\approx0.34867844$$ 。所以经过多层的影响后，前面的神经网络层接收到梯度可能会趋近于零，消失了。

传统的网络模型有三种方法来解决梯度消失的问题：

（a）使用ReLU激活函数。Sigmoid函数在数值过大或过小时都会进入梯度饱和区域，而且梯度的计算 $$\text{output}*(1-\text{output})$$ 衰减得特别快。相对地，ReLU采用分段激活的方法，令参数结果变得稀疏，在激活阶段，梯度稳定为 $$1$$ ，有助于网络模型的收敛。不过ReLU存在两个问题：

* 网络模型的结果并不是越稀疏越好，过于稀疏的参数会导致欠拟合
* ReLU可能会过早地关闭一些输入的神经元，使其得不到更新

（b）仔细地初始化与调试参数。参数的初始化选择对模型训练的影响很大，适合的初始值可以使模型较快地收敛到理想的结果。但由于深度学习理论尚处于快速发展的阶段，很多深度神经网络模型的实际应用没有有效的理论解释，参数的初始化与调试在一定程度上依赖于经验，仍属于复杂的黑盒问题。

（c）使用较小的学习率。当使用Sigmoid激活时，较大的学习率会使权重参数变大，导致层间输入变大，较大的数值位于激活函数的饱和区域，从而使梯度趋近于零。另外，学习率属于网络模型中的超参数，完全依靠外部设置，也会带来其他的影响。过大的学习率会使梯度产生过多的抖动，无法收敛；而较小的学习率会减少参数更新的幅度，拖慢收敛速度。当使用ReLU激活时，过大的学习率还会使模型在训练初期产生梯度爆炸，也完全无法收敛。

（2）Batch Normalization带来更好的梯度流

与之前提及的数据标准化的道理类似，BN能够减少训练时每层梯度的变化幅度，使得梯度稳定在理想的变化范围内，从而改善梯度流，产生以下收益：

* BN能够减少梯度对参数的尺度或初始值的依赖，使得调参更加容易。
* BN允许网络接受更大的学习率，这是因为 $$\text{BN}(Wu)=\text{BN}((\alpha W)u) \Rightarrow \frac{\partial \text{BN}((\alpha W)u)}{\partial u}=\frac{\partial W u}{\partial u}\Rightarrow\frac{\partial \text{BN}((\alpha W)u)}{\partial \alpha W}=\frac{1}{\alpha}\cdot\frac{\partial Wu}{\partial W}$$ ，所以学习率的尺度不会明显影响所产生的梯度的尺度。
* 由于梯度流的改善，模型能够更快地达到较高的精度。

## Source

{% embed url="https://blog.csdn.net/anshuai\_aw1/article/details/84975689\#Batch\_Normalization\_284" %}

{% embed url="http://cs231n.github.io/neural-networks-2/" %}





