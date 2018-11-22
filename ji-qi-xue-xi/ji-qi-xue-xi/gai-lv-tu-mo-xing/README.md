# 概率图模型

![](../../../.gitbook/assets/v2-714c1843f78b6aecdb0c57cdd08e1c6a_hd.jpg)

## [有向图vs.无向图](https://www.zhihu.com/question/35866596/answer/236886066)

上图可以看到，贝叶斯网络（信念网络）都是有向的，马尔科夫网络无向。所以，贝叶斯网络适合为有单向依赖的数据建模，马尔科夫网络适合实体之间互相依赖的建模。具体地，他们的核心差异表现在如何求 ![P=\(Y\)](https://www.zhihu.com/equation?tex=P%3D%28Y%29) ，即怎么表示 ![Y=&#xFF08;y\_{1},\cdots,y\_{n}&#xFF09;](https://www.zhihu.com/equation?tex=Y%3D%EF%BC%88y_%7B1%7D%2C%5Ccdots%2Cy_%7Bn%7D%EF%BC%89) 这个的联合概率。

### **有向图**

对于有向图模型，这么求联合概率： ![P\(x\_{1}, {\cdots}, x\_{n} \)=\prod\_{i=0}P\(x\_{i} \| \pi\(x\_{i}\)\)](https://www.zhihu.com/equation?tex=P%28x_%7B1%7D%2C+%7B%5Ccdots%7D%2C+x_%7Bn%7D+%29%3D%5Cprod_%7Bi%3D0%7DP%28x_%7Bi%7D+%7C+%5Cpi%28x_%7Bi%7D%29%29)

举个例子，对于下面的这个有向图的随机变量\(注意，这个图我画的还是比较广义的\)：![](https://pic1.zhimg.com/v2-5b3f6b4a2d905297b7f73a89e92ee618_b.jpg)![](https://pic1.zhimg.com/80/v2-5b3f6b4a2d905297b7f73a89e92ee618_hd.jpg)

应该这样表示他们的联合概率:

![](https://www.zhihu.com/equation?tex=P%28x_%7B1%7D%2C+%7B%5Ccdots%7D%2C+x_%7Bn%7D+%29%3DP%28x_%7B1%7D%29%C2%B7P%28x_%7B2%7D%7Cx_%7B1%7D+%29%C2%B7P%28x_%7B3%7D%7Cx_%7B2%7D+%29%C2%B7P%28x_%7B4%7D%7Cx_%7B2%7D+%29%C2%B7P%28x_%7B5%7D%7Cx_%7B3%7D%2Cx_%7B4%7D+%29+)

### **无向图**

对于无向图，我看资料一般就指马尔科夫网络\(注意，这个图我画的也是比较广义的\)。![](https://pic4.zhimg.com/v2-1d8faeb71d690d02e110c7cd1d39eed3_b.jpg)![](https://pic4.zhimg.com/80/v2-1d8faeb71d690d02e110c7cd1d39eed3_hd.jpg)

如果一个graph太大，可以用因子分解将 ![P=\(Y\)](https://www.zhihu.com/equation?tex=P%3D%28Y%29) 写为若干个联合概率的乘积。咋分解呢，将一个图分为若干个“小团”，注意每个团必须是“最大团”（就是里面任何两个点连在了一块，具体……算了不解释，有点“最大连通子图”的感觉），则有：

![ ](https://www.zhihu.com/equation?tex=P%28Y+%29%3D%5Cfrac%7B1%7D%7BZ%28x%29%7D+%5Cprod_%7Bc%7D%5Cpsi_%7Bc%7D%28Y_%7Bc%7D+%29+)

, 其中 ![Z\(x\) = \sum\_{Y} \prod\_{c}\psi\_{c}\(Y\_{c} \)](https://www.zhihu.com/equation?tex=Z%28x%29+%3D+%5Csum_%7BY%7D+%5Cprod_%7Bc%7D%5Cpsi_%7Bc%7D%28Y_%7Bc%7D+%29) ，公式应该不难理解吧，归一化是为了让结果算作概率。

所以像上面的无向图：

![](https://www.zhihu.com/equation?tex=P%28Y+%29%3D%5Cfrac%7B1%7D%7BZ%28x%29%7D+%28+%5Cpsi_%7B1%7D%28X_%7B1%7D%2C+X_%7B3%7D%2C+X_%7B4%7D+%29+%C2%B7+%5Cpsi_%7B2%7D%28X_%7B2%7D%2C+X_%7B3%7D%2C+X_%7B4%7D+%29+%29)

其中， ![ \psi\_{c}\(Y\_{c} \)](https://www.zhihu.com/equation?tex=+%5Cpsi_%7Bc%7D%28Y_%7Bc%7D+%29) 是一个最大团 ![C](https://www.zhihu.com/equation?tex=C) 上随机变量们的联合概率，一般取指数函数的：

![](https://www.zhihu.com/equation?tex=%5Cpsi_%7Bc%7D%28Y_%7Bc%7D+%29+%3D+e%5E%7B-E%28Y_%7Bc%7D%29%7D+%3De%5E%7B%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28c%2Cy%7Cc%2Cx%29%7D)

好了，管这个东西叫做[势函数](https://baike.baidu.com/item/%E5%8A%BF%E5%87%BD%E6%95%B0)。注意 ![e^{\sum\_{k}\lambda\_{k}f\_{k}\(c,y\|c,x\)}](https://www.zhihu.com/equation?tex=e%5E%7B%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28c%2Cy%7Cc%2Cx%29%7D) 是否有看到CRF的影子。

那么概率无向图的联合概率分布可以在因子分解下表示为：

![](https://www.zhihu.com/equation?tex=P%28Y+%29%3D%5Cfrac%7B1%7D%7BZ%28x%29%7D+%5Cprod_%7Bc%7D%5Cpsi_%7Bc%7D%28Y_%7Bc%7D+%29+%3D+%5Cfrac%7B1%7D%7BZ%28x%29%7D+%5Cprod_%7Bc%7D+e%5E%7B%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28c%2Cy%7Cc%2Cx%29%7D+%3D+%5Cfrac%7B1%7D%7BZ%28x%29%7D+e%5E%7B%5Csum_%7Bc%7D%5Csum_%7Bk%7D%5Clambda_%7Bk%7Df_%7Bk%7D%28y_%7Bi%7D%2Cy_%7Bi-1%7D%2Cx%2Ci%29%7D)

## Source

{% embed url="https://www.zhihu.com/question/35866596/answer/236886066" %}



