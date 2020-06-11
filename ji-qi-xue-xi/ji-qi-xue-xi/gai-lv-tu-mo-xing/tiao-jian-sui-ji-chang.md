# 条件随机场

## 条件随机场

条件随机场\(Conditional Random Field, CRF\)是一种判别式无向图模型。生成式模型是直接对联合分布进行建模，而判别式模型则是对条件分布进行建模。条件随机场是给定随机变量 $$X$$ 条件下，随机变量 $$Y$$ 的马尔可夫随机场。 

设 $$X$$ 与 $$Y$$ 是随机变量， $$P(Y|X)$$ 是在给 $$X$$ 的条件下 $$Y$$ 的条件概率分布。若随机变量 $$Y$$ 构成一个无向图 $$G = (V,E)$$ 表示的马尔可夫随机场，即

                                                 $$P(Y_v|X,Y_w,w\neq v) = P(Y_v|X,Y_w,w\sim v)$$ 

对任意结点 $$v$$ 成立，则称条件概率分布 $$P(Y|X)$$ 为条件随机场。式中 $$w\sim v$$ 表示在图 $$G=(V,E)$$ 中与结点 $$v$$ 有边连接的所有结点 $$w$$ ， $$w\neq v$$ 表示结点 $$v$$ 以外所有结点， $$Y_v,Y_u,Y_w$$ 为结点 $$v,u,w$$ 对应的随机变量。

## 链式条件随机场

这里主要介绍定义在线性链上的特殊的条件随机场，称为线性链条件随机场\(Linear Chain Conditional Random Field\)。线性链条件随机场可以用于标注问题。这时，在条件随机概率模型 $$P(Y|X)$$ 中， $$Y$$ 是输出变量，表示标记序列， $$X$$ 是输入变量，表示需要标注的观测序列。也把标记序列称为状态序列。学习时，利用训练数据集通过极大似然估计或正则化的极大似然估计得到条件概率模型 $$\hat{P}(Y|X)$$；预测时，对于给定的输入序列 $$x$$ ，求出条件概率 $$\hat{P}(y|x)$$ 最大的输出序列 $$\hat{y}$$。

![](../../../.gitbook/assets/4155986-800a14625dea5575.webp)

设 $$X=(X_1,X_2,\dots,X_n)$$ ， $$Y = (Y_1,Y_2,\dots,Y_n)$$ 均为线性链表示的随机变量序列，若在给定随机变量序列 $$X$$ 的条件下，随机变量序列 $$Y$$ 的条件概率分布 $$P(Y|X)$$ 构成条件随机场，即满足马尔可夫性

                         $$P(Y_i|X,Y_1,\dots,Y_{i-1},Y_{i+1},\dots,Y_n)=P(Y_i|X,Y_{i-1},Y_{i+1})$$ 

                                      $$ i=1,2,\dots,n(在i=1和n时只考虑单边)$$ 

则称 $$P(Y|X)$$ 为线性链条件随机场。在标注问题中， $$X$$ 表示输入观测序列， $$Y$$ 表示对应的输出标记序列或状态序列。

![](../../../.gitbook/assets/4155986-c1ae06cfa161c166.webp)

## [条件随机场运行过程](https://www.zhihu.com/question/35866596/answer/236886066)

请看第一张概率图模型构架图，CRF上面是马尔科夫随机场（马尔科夫网络），而条件随机场是在给定的随机变量 ![X](https://www.zhihu.com/equation?tex=X) （具体，对应观测序列 ![o\_{1}, \cdots, o\_{i}](https://www.zhihu.com/equation?tex=o_%7B1%7D%2C+%5Ccdots%2C+o_%7Bi%7D) ）条件下，随机变量 ![Y](https://www.zhihu.com/equation?tex=Y) （具体，对应隐状态序列 ![i\_{1}, \cdots, i\_{i}](https://www.zhihu.com/equation?tex=i_%7B1%7D%2C+%5Ccdots%2C+i_%7Bi%7D) 的马尔科夫随机场。  
广义的CRF的定义是： 满足 ![P\(Y\_{v}\|X,Y\_{w},w \neq v\) = P\(Y\_{v}\|X,Y\_{w},w \sim v\) ](https://www.zhihu.com/equation?tex=P%28Y_%7Bv%7D%7CX%2CY_%7Bw%7D%2Cw+%5Cneq+v%29+%3D+P%28Y_%7Bv%7D%7CX%2CY_%7Bw%7D%2Cw+%5Csim+v%29+) 的马尔科夫随机场叫做条件随机场

不过一般说CRF为序列建模，就专指CRF线性链（linear chain CRF）：![](https://pic3.zhimg.com/v2-c5e2e782e35f6412ed65e58cdda0964e_b.jpg)

![](../../../.gitbook/assets/linear-chain-crf.jpg)

概率无向图的联合概率分布可以在因子分解下表示为：

![](https://www.zhihu.com/equation?tex=P%28Y+%7C+X%29%3D%5Cfrac%7B1%7D%7BZ%28x%29%7D+%5Cprod_%7Bc%7D%5Cpsi_%7Bc%7D%28Y_%7Bc%7D%7CX+%29+%3D+%5Cfrac%7B1%7D%7BZ%28x%29%7D+%5Cprod_%7Bc%7D+e%5E%7B%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28c%2Cy%7Cc%2Cx%29%7D+%3D+%5Cfrac%7B1%7D%7BZ%28x%29%7D+e%5E%7B%5Csum_%7Bc%7D%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28y_%7Bi%7D%2Cy_%7Bi-1%7D%2Cx%2Ci%29%7D)

而在线性链CRF示意图中，每一个（ ![I\_{i} \sim O\_{i}](https://www.zhihu.com/equation?tex=I_%7Bi%7D+%5Csim+O_%7Bi%7D) ）对为一个最大团,即在上式中 ![c = i](https://www.zhihu.com/equation?tex=c+%3D+i) 。并且线性链CRF满足 ![P\(I\_{i}\|O,I\_{1},\cdots, I\_{n}\) = P\(I\_{i}\|O,I\_{i-1},I\_{i+1}\) ](https://www.zhihu.com/equation?tex=P%28I_%7Bi%7D%7CO%2CI_%7B1%7D%2C%5Ccdots%2C+I_%7Bn%7D%29+%3D+P%28I_%7Bi%7D%7CO%2CI_%7Bi-1%7D%2CI_%7Bi%2B1%7D%29+) 。

**所以CRF的建模公式如下：**

![](https://www.zhihu.com/equation?tex=P%28I+%7C+O%29%3D%5Cfrac%7B1%7D%7BZ%28O%29%7D+%5Cprod_%7Bi%7D%5Cpsi_%7Bi%7D%28I_%7Bi%7D%7CO+%29+%3D+%5Cfrac%7B1%7D%7BZ%28O%29%7D+%5Cprod_%7Bi%7D+e%5E%7B%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29%7D+%3D+%5Cfrac%7B1%7D%7BZ%28O%29%7D+e%5E%7B%5Csum_%7Bi%7D%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29%7D)

我要敲黑板了，这个公式是非常非常关键的，注意递推过程啊，我是怎么从 ![&#x220F;](https://www.zhihu.com/equation?tex=%E2%88%8F) 跳到 ![e^{\sum}](https://www.zhihu.com/equation?tex=e%5E%7B%5Csum%7D) 的

不过还是要多啰嗦一句，想要理解CRF，必须判别式模型的概念要深入你心。正因为是判别模型，所以不废话，我上来就直接为了确定边界而去建模，因为我创造出来就是为了这个分边界的目的的。比如说序列求概率（分类）问题，我直接考虑找出函数分类边界。所以才为什么会有这个公式。所以再看到这个公式也别懵逼了，he was born for discriminating the given data from different classes. 就这样。不过待会还会具体介绍特征函数部分的东西。除了建模总公式，关键的CRF重点概念在MEMM中已强调过：**判别式模型**、**特征函数**。

**特征函数**

上面给出了CRF的建模公式：

![](https://www.zhihu.com/equation?tex=P%28I+%7C+O%29%3D%5Cfrac%7B1%7D%7BZ%28O%29%7D+e%5E%7B%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bk%7D%5E%7BM%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29%7D)

* 下标i表示我当前所在的节点（token）位置。
* 下标k表示我这是第几个特征函数，并且每个特征函数都附属一个权重 ![\lambda\_{k}](https://www.zhihu.com/equation?tex=%5Clambda_%7Bk%7D) ，也就是这么回事，每个团里面，我将为 ![token\_{i}](https://www.zhihu.com/equation?tex=token_%7Bi%7D) 构造M个特征，每个特征执行一定的限定作用，然后建模时我再为每个特征函数加权求和。
* ![Z\(O\)](https://www.zhihu.com/equation?tex=Z%28O%29) 是用来归一化的，为什么？想想LR以及softmax为何有归一化呢，一样的嘛，形成概率值。
* 再来个重要的理解。 ![P\(I\|O\)](https://www.zhihu.com/equation?tex=P%28I%7CO%29) 这个表示什么？具体地，表示了在给定的一条观测序列 ![O=\(o\_{1},\cdots, o\_{i}\)](https://www.zhihu.com/equation?tex=O%3D%28o_%7B1%7D%2C%5Ccdots%2C+o_%7Bi%7D%29) 条件下，我用CRF所求出来的隐状态序列 ![I=\(i\_{1},\cdots, i\_{i}\)](https://www.zhihu.com/equation?tex=I%3D%28i_%7B1%7D%2C%5Ccdots%2C+i_%7Bi%7D%29) 的概率，注意，这里的I是一条序列，有多个元素（一组随机变量），而至于观测序列 ![O=\(o\_{1},\cdots, o\_{i}\)](https://www.zhihu.com/equation?tex=O%3D%28o_%7B1%7D%2C%5Ccdots%2C+o_%7Bi%7D%29) ，它可以是一整个训练语料的所有的观测序列；也可以是在inference阶段的一句sample，比如说对于序列标注问题，我对一条sample进行预测，可能能得到 ![P\_{j}\(I \| O\)&#xFF08;j=1,&#x2026;,J&#xFF09;](https://www.zhihu.com/equation?tex=P_%7Bj%7D%28I+%7C+O%29%EF%BC%88j%3D1%2C%E2%80%A6%2CJ%EF%BC%89)J条隐状态I，但我肯定最终选的是最优概率的那条（by viterbi）。这一点希望你能理解。

对于CRF，可以为他定义两款特征函数：转移特征&状态特征。 我们将建模总公式展开：

![](https://www.zhihu.com/equation?tex=P%28I+%7C+O%29%3D%5Cfrac%7B1%7D%7BZ%28O%29%7D+e%5E%7B%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bk%7D%5E%7BM%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29%7D%3D%5Cfrac%7B1%7D%7BZ%28O%29%7D+e%5E%7B+%5B+%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bj%7D%5E%7BJ%7D%5Clambda_%7Bj%7Dt_%7Bj%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29+%2B+%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bl%7D%5E%7BL%7D%5Cmu_%7Bl%7Ds_%7Bl%7D%28O%2CI_%7Bi%7D%2Ci%29+%5D+%7D)

其中：![t\_{j}](https://www.zhihu.com/equation?tex=t_%7Bj%7D) 为i处的转移特征，对应权重 ![\lambda\_{j}](https://www.zhihu.com/equation?tex=%5Clambda_%7Bj%7D) ,每个 ![token\_{i}](https://www.zhihu.com/equation?tex=token_%7Bi%7D) 都有J个特征,转移特征针对的是前后token之间的限定。举个例子：

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+t_%7Bk%3D1%7D%28o%2Ci%29+%3D+%5Cbegin%7Bcases%7D+1%26+%5Ctext%7B%E6%BB%A1%E8%B6%B3%E7%89%B9%E5%AE%9A%E8%BD%AC%E7%A7%BB%E6%9D%A1%E4%BB%B6%EF%BC%8C%E6%AF%94%E5%A6%82%E5%89%8D%E4%B8%80%E4%B8%AAtoken%E6%98%AF%E2%80%98I%E2%80%99%7D%EF%BC%8C%5C%5C+0%26+%5Ctext%7Bother%7D+%5Cend%7Bcases%7D+%5Cend%7Bequation%7D)

sl为i处的状态特征，对应权重μl,每个tokeni都有L个特征。举个例子：

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+s_%7Bl%3D1%7D%28o%2Ci%29+%3D+%5Cbegin%7Bcases%7D+1%26+%5Ctext%7B%E6%BB%A1%E8%B6%B3%E7%89%B9%E5%AE%9A%E7%8A%B6%E6%80%81%E6%9D%A1%E4%BB%B6%EF%BC%8C%E6%AF%94%E5%A6%82%E5%BD%93%E5%89%8Dtoken%E7%9A%84POS%E6%98%AF%E2%80%98V%E2%80%99%7D%EF%BC%8C%5C%5C+0%26+%5Ctext%7Bother%7D+%5Cend%7Bcases%7D+%5Cend%7Bequation%7D)

不过一般情况下，我们不把两种特征区别的那么开，合在一起：

![](https://www.zhihu.com/equation?tex=P%28I+%7C+O%29%3D%5Cfrac%7B1%7D%7BZ%28O%29%7D+e%5E%7B%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bk%7D%5E%7BM%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29%7D)

满足特征条件就取值为1，否则没贡献，甚至你还可以让他打负分，充分惩罚。

再进一步理解的话，我们需要把特征函数部分抠出来：

![](https://www.zhihu.com/equation?tex=Score+%3D+%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bk%7D%5E%7BM%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29+)

是的，我们为 ![token\_{i}](https://www.zhihu.com/equation?tex=token_%7Bi%7D) 打分，满足条件的就有所贡献。最后将所得的分数进行log线性表示，求和后归一化，即可得到概率值……完了又扯到了log线性模型。现在稍作解释：

> log-linear models take the following form:  
>
>
> ![P\(y\|x;\omega\) = \frac{ exp\(\omega&#xB7;\phi\(x,y\)\) }{ \sum\_{y^{&apos;}\in Y }exp\(\omega&#xB7;\phi\(x,y^{&#x2018;}\)\) }](https://www.zhihu.com/equation?tex=P%28y%7Cx%3B%5Comega%29+%3D+%5Cfrac%7B+exp%28%5Comega%C2%B7%5Cphi%28x%2Cy%29%29+%7D%7B+%5Csum_%7By%5E%7B%27%7D%5Cin+Y+%7Dexp%28%5Comega%C2%B7%5Cphi%28x%2Cy%5E%7B%E2%80%98%7D%29%29+%7D)

我觉得对LR或者sotfmax熟悉的对这个应该秒懂。然后CRF完美地满足这个形式，所以又可以归入到了log-linear models之中。

### **模型运行过程**

模型的工作流程，跟MEMM是一样的：

* step1. 先预定义特征函数 ![ f\_{a}\(o,i\)](https://www.zhihu.com/equation?tex=+f_%7Ba%7D%28o%2Ci%29) 
* step2. 在给定的数据上，训练模型，确定参数 ![\lambda\_{k}](https://www.zhihu.com/equation?tex=%5Clambda_%7Bk%7D)
* step3. 用确定的模型做`序列标注问题`或者`序列求概率问题`

### **1. 学习训练过程**

一套CRF由一套参数λ唯一确定（先定义好各种特征函数）。

同样，CRF用极大似然估计方法、梯度下降、牛顿迭代、拟牛顿下降、IIS、BFGS、L-BFGS等等。各位应该对各种优化方法有所了解的。其实能用在log-linear models上的求参方法都可以用过来。

### **2. 序列标注过程**

还是跟HMM一样的，用学习好的CRF模型，在新的sample（观测序列 ![o\_{1}, \cdots, o\_{i}](https://www.zhihu.com/equation?tex=o_%7B1%7D%2C+%5Ccdots%2C+o_%7Bi%7D) ）上找出一条概率最大最可能的隐状态序列 ![i\_{1}, \cdots, i\_{i}](https://www.zhihu.com/equation?tex=i_%7B1%7D%2C+%5Ccdots%2C+i_%7Bi%7D) 。

只是现在的图中的每个隐状态节点的概率求法有一些差异而已,正确将每个节点的概率表示清楚，路径求解过程还是一样，采用viterbi算法。

啰嗦一下，我们就定义i处的局部状态为 ![\delta\_{i}\(I\)](https://www.zhihu.com/equation?tex=%5Cdelta_%7Bi%7D%28I%29) ,表示在位置i处的隐状态的各种取值可能为I，然后递推位置i+1处的隐状态，写出来的DP转移公式为：

![](https://www.zhihu.com/equation?tex=%5Cdelta_%7Bi%2B1%7D+%3D+max_%7B1+%5Cle+j+%5Cle+m%7D%5Clbrace+%5Cdelta_%7Bi%7D%28I%29+%2B+%5Csum_%7Bi%7D%5E%7BT%7D%5Csum_%7Bk%7D%5E%7BM%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28O%2CI_%7Bi-1%7D%2CI_%7Bi%7D%2Ci%29+%5Crbrace)

这里没写规范因子 ![Z\(O\)](https://www.zhihu.com/equation?tex=Z%28O%29) 是因为不规范化不会影响取最大值后的比较。

### **3. 序列求概率过程**

跟HMM举的例子一样的，也是分别去为每一批数据训练构建特定的CRF，然后根据序列在每个MEMM模型的不同得分概率，选择最高分数的模型为wanted类别。

## Source

{% embed url="https://www.zhihu.com/question/35866596/answer/236886066" %}



