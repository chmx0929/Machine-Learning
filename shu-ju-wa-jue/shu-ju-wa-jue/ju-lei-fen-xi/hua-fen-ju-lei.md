# 划分聚类

## K- clustering

### K-Means

K-Means即每一簇类使用类的均值来表示

输入：样本集 $$D=\{x_1,x_2,\dots,x_m\}$$ ，聚类簇数 $$k$$ 

过程：

* 从 $$D$$ 中随机选取 $$k$$ 个样本作为初始均值向量 $$\{\mu_1,\mu_2,\dots,\mu_k\}$$ 
* repeat：
*             令 $$C_i=\varnothing(1\leq i \leq k)$$ 
*             for $$j = 1,2,\dots,m$$：
*                         计算样本 $$x_j$$ 与各均值向量 $$\mu_i(1\leq i \leq k)$$ 的距离 $$d_{ji}=||x_j-x_i||_2$$ 
*                         根据距离最近的均值向量确定 $$x_j$$ 的簇标记 $$\lambda_j=\arg\min_{i\in \{1,2,\dots,k\}}d_{ji}$$ 
*                         将样本 $$x_j$$ 划入相应的簇 $$C_{\lambda_j}=C_{\lambda_j}\cup\{x_j\}$$ 
*             for $$i = 1,2,\dots,k$$ ：
*                         计算新均值向量 $$\mu'_i=\frac{1}{|C_i|}\sum_{x\in C_i}x$$ 
*                         if $$\mu'_i \neq \mu_i$$ ：
*                                     将当前均值向量 $$\mu_i$$ 更新为 $$\mu'_i$$ 
*                         else：
*                                     保持当前均值向量不变
* until：当前均指向量均未更新

![](../../../.gitbook/assets/timline-jie-tu-20181126192739.png)

### K-Medoids

K-Means对异常值敏感，如果一个极端值会干扰结果。这时可以使用K-Medoids

```text
初始化：从数据点集中随机选择k个点，作为初始中心点；将待聚类的数据点集中的点，指派到最近的中心点
迭代(收敛或变化小于阈值停止)：
    对每中心点，和其每一个非中心点交换，计算交换后划分所生成的代价值，若交换造成代价增加，则取消交换
```

![](../../../.gitbook/assets/timline-jie-tu-20181126194248.png)

### K-Medians&K-Modes

相较均值\(mean\)，中位数\(median\)更不易被异常值影响，所以也可使用K-Medians，即将均值换为中位数

K-Means不好处理类别数据\(不好计算距离\)，此时可使用K-Modes\(众数\)。 K-Modes是用每个聚类中的众数（mode）做中心点。距离的定义也不同，通常K-Means较多使用欧式距离，K-Modes一般是汉明距离，也就是对于每个特征来说，如果不同记为1，相同则为0。

#### [K-Modes例子](http://sofasofa.io/forum_main_post.php?postid=1000500)

| 手机 | 国家 | 人群 | 颜色 |
| :---: | :---: | :---: | :---: |
| 1 | 中 | 青年 | 白 |
| 2 | 日 | 青年 | 黑 |
| 3 | 中 | 青年 | 蓝 |
| 4 | 中 | 青年 | 黑 |
| 5 | 日 | 青年 | 白 |
| 6 | 日 | 中年 | 黑 |
| 7 | 美 | 中年 | 蓝 |
| 8 | 美 | 中年 | 白 |
| 9 | 中 | 中年 | 黑 |
| 10 | 美 | 中年 | 黑 |

 假定我们选择聚类的数量K=2，初始点为手机1（中，青年，白）和手机6（日，中年，黑），得到

|  | 与手机1距离 | 与手机6距离 |
| :---: | :---: | :---: |
| 2 | 2 | 1 |
| 3 | 1 | 3 |
| 4 | 1 | 2 |
| 5 | 1 | 2 |
| 7 | 3 | 2 |
| 8 | 2 | 2 |
| 9 | 2 | 1 |
| 10 | 3 | 1 |

距离越小属于同一类别，若到距离相同则可以随机归于一类。

### Kernel K-Means

对于非凸簇类，可使用核函数映射到高维，再进行聚类。

![](../../../.gitbook/assets/timline-jie-tu-20181126195624.png)

## 高斯混合模型

高斯混合模型（Gaussian Mixture Model）通常简称GMM，是一种业界广泛使用的聚类算法，该方法使用了高斯分布作为参数模型，并使用了期望最大（Expectation Maximization，简称EM）算法进行训练。 在特定约束条件下，K-means算法可以被看作是高斯混合模型（GMM）的一种特殊形式。

高斯混合模型（Gaussian Mixed Model）指的是多个高斯分布函数的线性组合，理论上GMM可以拟合出任意类型的分布，通常用于解决同一集合下的数据包含多个不同的分布的情况（或者是同一类分布但参数不一样，或者是不同类型的分布，比如正态分布和伯努利分布）。

### 例子一

如下图，图中的点在我们看来明显分成两个聚类。这两个聚类中的点分别通过两个不同的正态分布随机生成而来。但是如果没有GMM，那么只能用一个的二维高斯分布来描述下图中的数据。图中的椭圆即为二倍标准差的正态分布椭圆。这显然不太合理，毕竟肉眼一看就觉得应该把它们分成两类。

![](../../../.gitbook/assets/20170302175442272.png)

这时候就可以使用GMM了！如下图，数据在平面上的空间分布和上图一样，这时使用两个二维高斯分布来描述图中的数据，分别记为 $$\mathcal{N}(\mu_1,\sum_1)$$ 和 $$\mathcal{N}(\mu_2,\sum_2)$$ 。图中的两个椭圆分别是这两个高斯分布的二倍标准差椭圆。可以看到使用两个二维高斯分布来描述图中的数据显然更合理。实际上图中的两个聚类的中的点是通过两个不同的正态分布随机生成而来。如果将两个二维高斯分布 $$\mathcal{N}(\mu_1,\sum_1)$$ 和 $$\mathcal{N}(\mu_2,\sum_2)$$合成一个二维的分布，那么就可以用合成后的分布来描述图中的所有点。最直观的方法就是对这两个二维高斯分布做线性组合，用线性组合后的分布来描述整个集合中的数据。这就是高斯混合模型（GMM）。

![](../../../.gitbook/assets/20170302175549877.png)

### 例子二

![](../../../.gitbook/assets/v2-5cc4a35306b5b1f3176188e04d786c86_hd.jpg)

上图为男性（蓝）和女性（红）的身高采样分布图。上图的y-轴所示的概率值，是在已知每个用户性别的前提下计算出来的。但通常情况下我们并不能掌握这个信息（也许在采集数据时没记录），因此不仅要学出每种分布的参数，还需要生成性别的划分情况 $$\varphi_i$$ 。 当决定期望值时，需要将权重值分别生成男性和女性的相应身高概率值并相加。

### 高斯混合模型定义

高斯混合模型，顾名思义，就是数据可以看作是从多个高斯分布中生成出来的。从[中心极限定理](https://en.wikipedia.org/wiki/Central_limit_theorem)可以看出，高斯分布这个假设其实是比较合理的。 为什么我们要假设数据是由若干个高斯分布组合而成的，而不假设是其他分布呢？实际上不管是什么分布，只 $$k$$ 取得足够大，这个XX Mixture Model就会变得足够复杂，就可以用来逼近任意连续的概率密度分布。只是因为高斯函数具有良好的计算性能，所GMM被广泛地应用。

每个GMM由K个高斯分布组成，每个高斯分布称为一个组件（Component），这些组件线性加成在一起就组成了GMM的概率密度函数。高斯混合模型具有如下形式的概率分布模型：

                                                             $$P(y|\theta)=\sum\limits_{k=1}^K\alpha_k\phi(y|\theta_k)$$ 

其中， $$\alpha_k$$ 是系数， $$\alpha_k\geq 0$$ ， $$\sum\limits_{k=1}^K\alpha_k=1$$ ； $$\phi(y|\theta_k)$$ 是高斯分布密度， $$\theta_k=(\mu_k,\sigma^2_k)$$ ，

                                                      $$\phi(y|\theta_k) = \frac{1}{\sqrt{2\pi}\sigma_k}\exp(-\frac{(y-\mu_k)^2}{2\sigma_k^2})$$ 

称为第 $$k$$ 个分模型。

### 高斯混合参数估计

假设观测数据 $$y_1,y_2,\dots,y_N$$ 由告诉混合模型生成

                                                                $$P(y|\theta)=\sum\limits_{k=1}^K\alpha_k\phi(y|\theta_k)$$ 

其中， $$\theta=(\alpha_1,\alpha_2,\dots,\alpha_K;\theta_1,\theta_2,\dots,\theta_K)$$ 。我们用EM算法估计高斯混合模型的参数 $$\theta$$ 。

#### 1、明确变量，写出完全数据的对数似然函数

可以设想观测数据 $$y_j,\ j=1,2,\dots,N$$ ，是这样产生的：首先依概率 $$\alpha_k$$ 选择第 $$k$$ 个高斯分布分模型 $$\phi(y|\theta_k)$$ ；然后依第 $$k$$ 个分模型的概率分布 $$\phi(y|\theta_k)$$ 生成观测数据 $$y_j$$ 。这时观测数据 $$y_j,\ j=1,2,\dots,N$$ 是已知的；反映观测数据 $$y_j$$ 来自第 $$k$$ 个分模型的数据是未知的， $$k=1,2,\dots,K$$ ，以隐变量 $$\gamma_{jk}$$ 表示，其定义如下：

                           $$\gamma_{jk}=\begin{cases}1,\ \ \ 第j个观测来自第k个分模型\\ 0,\ \ \ 否则\end{cases}, \ \ j=1,2,\dots,N;k=1,2,\dots,K$$  $$\gamma_{jk}$$ 是0-1随机变量。

有了观测数据 $$y_j$$ 及未观测数据 $$\gamma_{jk}$$ ，那么完全数据是

                                                $$(y_j,\gamma_{j1},\gamma_{j2},\dots,\gamma_{jK}),\ j=1,2,\dots,N$$ 

于是，可以写出完全数据的似然函数：

                                              $$P(y,\gamma|\theta)=\prod\limits_{j=1}^NP(y_j,\gamma_{j1},\gamma_{j2},\dots,\gamma_{jK}|\theta)$$ 

                                                                  $$= \prod\limits_{k=1}^K\prod\limits_{j=1}^N[\alpha_k\phi(y_j|\theta_k)]^{\gamma_{jk}}$$ 

                                                                  $$= \prod\limits_{k=1}^K\alpha_k^{n_k}\prod\limits_{j=1}^N[\phi(y_j|\theta_k)]^{\gamma_{jk}}$$ 

                                                                  $$= \prod\limits_{k=1}^K\alpha_k^{n_k}\prod\limits_{j=1}^N[\frac{1}{\sqrt{2\pi}\sigma_k}\exp(-\frac{(y_j-\mu_k)^2}{2\sigma^2_k})]^{\gamma_{jk}}$$ 

式中， $$n_k = \sum\limits_{j=1}^N\gamma_{jk}$$ ， $$\sum\limits_{k=1}^Kn_k = N$$ 。

那么，完全数据的对数似然函数为：

                     $$\log P(y,\gamma|\theta)=\sum\limits_{k=1}^K\{n_k\log\alpha_k+\sum\limits_{j=1}^N\gamma_{jk}[\log(\frac{1}{\sqrt{2\pi}})-\log\sigma_k-\frac{1}{2\sigma_k^2}(y_j-\mu_k)^2]\}$$ 

#### 2、EM算法的E步：确定Q函数

                $$Q(\theta,\theta^{(i)})=E[\log P(y,\gamma|\theta)|y,\theta^{(i)}]$$ 

                                    $$ = E\{\sum\limits_{k=1}^K\{n_k\log\alpha_k+\sum\limits_{j=1}^N\gamma_{jk}[\log(\frac{1}{\sqrt{2\pi}})-\log\sigma_k-\frac{1}{2\sigma_k^2}(y_j-\mu_k)^2]\}\}$$ 

                                    $$=\sum\limits_{k=1}^K\{\sum\limits_{j=1}^N(E\gamma_{jk})\log\alpha_k+\sum\limits_{j=1}^N(E\gamma_{jk})[\log(\frac{1}{\sqrt{2\pi}})-\log\sigma_k-\frac{1}{2\sigma^2_k}(y_j-\mu_k)^2]\}$$ 

这里需要计算 $$E(\gamma_{jk}|y,\theta)$$ ，记为 $$\hat{\gamma}_{jk}$$ 

                             $$\hat{\gamma}_{jk}=E(\gamma_{jk}|y,\theta)=P(\gamma_{jk}=1|y,\theta)$$ 

                                     $$=\frac{P(\gamma_{jk}=1,y_j|\theta)}{\sum\limits_{k=1}^KP(\gamma_{jk}=1,y_j|\theta)}$$ 

                                     $$=\frac{P(y_j|\gamma_{jk}=1,\theta)P(\gamma_{jk}=1|\theta)}{\sum\limits_{k=1}^KP(y_j|\gamma_{jk}=1,\theta)P(\gamma_{jk}=1|\theta)}$$ 

                                     $$= \frac{\alpha_k\phi(y_j|\theta_k)}{\sum\limits_{k=1}^K\alpha_k\phi(y_j|\theta_k)}, \ \ j=1,2,\dots,N;k=1,2,\dots,K$$ 

$$\hat{\gamma}_{jk}$$ 是在当前模型参数下第 $$j$$ 个观测数据来自第 $$k$$ 个分模型的概率，称为分模型 $$k$$ 对观测数据 $$y_i$$ 的响应度。

将 $$\hat{\gamma}_{jk}=E\gamma_{jk}$$ 及 $$n_k=\sum\limits_{j=1}^NE\gamma_{jk}$$ 代入 $$Q(\theta,\theta^{(i)})$$ 即得

               $$Q(\theta,\theta^{(i)})=\sum\limits_{k=1}^K\{n_k\log\alpha_k+\sum\limits_{j=1}^N\hat{\gamma}_{jk}[\log(\frac{1}{\sqrt{2\pi}})-\log\sigma_k-\frac{1}{2\sigma^2_k}(y_j-\mu_k)^2]\}$$ 

#### 3、确定EM算法的M步

迭代的 $$M$$ 步是球函数 $$Q(\theta,\theta^{(i)})$$ 对 $$\theta$$ 的极大值，即求新一轮迭代的模型参数：

                                                            $$\theta^{(i+1)} = \mathop{\arg\max}\limits_\theta Q(\theta,\theta^{(i)})$$ 

用 $$\hat{\mu}_k$$ ， $$\hat{\sigma}^2_k$$ 及 $$\hat{\alpha}_k,\ k=1,2,\dots,K$$ ，表示 $$\theta^{(i+1)}$$ 的各参数。求 $$\hat{\mu}_k$$ ， $$\hat{\sigma}^2_k$$ 只需将 $$Q(\theta,\theta^{(i)})=\sum\limits_{k=1}^K\{n_k\log\alpha_k+\sum\limits_{j=1}^N\hat{\gamma}_{jk}[\log(\frac{1}{\sqrt{2\pi}})-\log\sigma_k-\frac{1}{2\sigma^2_k}(y_j-\mu_k)^2]\}$$ 分别对 $$\mu_k$$ ， $$\sigma^2_k$$ 求偏导数并令其为 $$0$$ ，即可得到；求 $$\hat{\alpha}_k$$ 是在 $$\sum\limits_{k=1}^K\alpha_k=1$$ 条件下求偏导数并令其为 $$0$$ 得到的。结果如下：

                                                      $$\hat{\mu}_k=\frac{\sum\limits_{j=1}^N\hat{\gamma}_{jk}y_j}{\sum\limits_{j=1}^N\hat{\gamma}_{jk}},\ \ k=1,2,\dots,K$$ 

                                                 $$\hat{\sigma}^2_k=\frac{\sum\limits_{j=1}^N\hat{\gamma}_{jk}(y_j-\mu_k)^2}{\sum\limits_{j=1}^N\hat{\gamma}_{jk}},\ \ k=1,2,\dots,K$$ 

                                                    $$\hat{\alpha}_k=\frac{n_k}{N}=\frac{\sum\limits_{j=1}^N\hat{\gamma}_{jk}}{N},\ \ k=1,2,\dots,K$$ 

重复以上计算，直到对数似然函数值不再有明显的变化为止。

### 高斯混合模型EM算法

输入：观测数据 $$y_1,y_2,\dots,y_N$$ ，高斯混合模型；

输出：高斯混合模型参数

（1）取参数的初始值开始迭代

（2）E步：依据当前模型参数，计算分模型 $$k$$ 对观测数据 $$y_j$$ 的响应度

                                 $$\hat{\gamma}_{jk}= \frac{\alpha_k\phi(y_j|\theta_k)}{\sum\limits_{k=1}^K\alpha_k\phi(y_j|\theta_k)}, \ \ j=1,2,\dots,N;k=1,2,\dots,K$$ 

（3）M步：计算新一轮迭代的模型参数

                                                      $$\hat{\mu}_k=\frac{\sum\limits_{j=1}^N\hat{\gamma}_{jk}y_j}{\sum\limits_{j=1}^N\hat{\gamma}_{jk}},\ \ k=1,2,\dots,K$$ 

                                                 $$\hat{\sigma}^2_k=\frac{\sum\limits_{j=1}^N\hat{\gamma}_{jk}(y_j-\mu_k)^2}{\sum\limits_{j=1}^N\hat{\gamma}_{jk}},\ \ k=1,2,\dots,K$$ 

                                                    $$\hat{\alpha}_k=\frac{n_k}{N}=\frac{\sum\limits_{j=1}^N\hat{\gamma}_{jk}}{N},\ \ k=1,2,\dots,K$$ 

（4）重复第（2）步和第（3）步，直到收敛。

## Source

{% embed url="http://sofasofa.io/forum\_main\_post.php?postid=1000500" %}

{% embed url="https://zhuanlan.zhihu.com/p/31103654" %}

{% embed url="https://blog.csdn.net/jinping\_shi/article/details/59613054" %}

{% embed url="https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E8%81%9A%E7%B1%BB/gaussian-mixture/gaussian-mixture.md" %}

