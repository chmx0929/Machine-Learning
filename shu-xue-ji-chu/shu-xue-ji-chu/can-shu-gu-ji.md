# 参数估计

根据一系列独立样本 $$X=\{x_1,x_2, \dots , x_N\}$$，以及模型 $$P(X;\theta)$$，找到参数 $$\theta$$ 

## [最大似然估计\(Maximum likelihood estimation\)](https://www.jianshu.com/p/f1d3906e4a3e)

        $$P(X;\theta) = P(x_1,x_2,\dots,x_N;\theta) = \prod \limits_i P(x_i;\theta)$$

$$\to \theta_{ML} = \mathop{argmax}\limits_\theta \prod\limits_i P(x_i;\theta)$$ 

#### 例子1：抽球

假设一个袋子装有白球与红球，比例未知，现在抽取10次（每次抽完都放回，保证事件独立性），假设抽到了7次白球和3次红球，在此数据样本条件下，可以采用最大似然估计法求解袋子中白球的比例（最大似然估计是一种“模型已定，参数未知”的方法）。当然，这种数据情况下很明显，白球的比例是70%，但如何通过理论的方法得到这个答案呢？一些复杂的条件下，是很难通过直观的方式获得答案的，这时候理论分析就尤为重要了，这也是学者们为何要提出最大似然估计的原因。我们可以定义从袋子中抽取白球和红球的概率如下：

$$x_1$$ 为第一次采样， $$x_2$$ 为第二次， $$f$$ 为模型， $$\theta$$ 模型参数

                                              $$f(x_1,x_2|\theta)=f(x_1|\theta)\times f(x_2|\theta)$$ 

其中 $$a = b$$ 是未知的，因此，我们定义似然 $$L$$ 为：

                                        $$L(\theta|x_1,x_2)=f(x_1,x_2|\theta)=\prod\limits_{i=1}^2f(x_i|\theta)$$ 

两边取 $$ln$$ ，取 $$ln$$ 是为了将右边的乘号变为加号，方便求导，左边的通常称之为对数似然：

                                 $$\ln L(\theta|x_1,x_2)=\ln \sum\limits_{i=1}^2f(x_i|\theta)=\sum\limits_{i=1}^2\ln f(x_i|\theta)$$ 

平均似然对数：

                                                             $$\hat{l}=\frac{1}{2}\ln L(\theta|x_1,x_2)$$ 

最大似然估计的过程，就是找一个合适的 $$\theta$$ ，使得平均对数似然的值为最大。因此，可以得到以下最大估计的公式：

                                                       $$\hat{\theta}_{mle}=\mathop{\arg \max}\limits_{\theta\in \Theta}\hat{l}(\theta|x_1,x_2)$$ 

这里讨论的是2次采样的情况，拓展到多次采样的情况，n次采样最大似然估计公式：

                                                $$\hat{\theta}_{mle}=\mathop{\arg \max}\limits_{\theta\in \Theta}\hat{l}(\theta|x_1,x_2,\dots,x_n)$$ 

我们定义 $$M$$ 为模型（也就是之前公式中的 $$f$$ ），表示抽到白球的概率为 $$\theta$$ ，而抽到红球的概率为 $$1-\theta$$ ，因此10次抽取抽到白球7次的概率可以表示为：

           $$P(x_1,x_2,\dots,x_{10}|M)=P(x_1|M)\times P(x_2|M)\dots P(x_{10}|M)=\theta^7(1-\theta)^3$$ 

将其描述为平均似然可得：

                               $$\hat{l}=\frac{1}{10}\ln P(x_1,x_2,\dots, x_{10}|M)=\frac{1}{10}\ln[\theta^7(1-\theta)^3]$$           

那么最大似然就是找到一个合适的 $$\theta$$ ，获得最大的平均似然。因此我们可以对平均似然的公式对 $$\theta$$ 求导，并令导数为0。

                                $$\hat{l}'(\theta)=7\theta^6(1-\theta)^3-3\theta^7(1-\theta)^2=0 \Rightarrow \theta=0.7$$ 

由此可得，当抽取白球的概率 $$\theta$$ 为0.7时，最可能产生10次抽取抽到白球7次的事件。

#### 例子2：正态分布

假如有一组采样值 $$(x_1,\dots,x_n)$$ ，我们知道其服从正态分布，且标准差已知。当这个正态分布的期望为多少时，产生这个采样数据的概率为最大？

这个例子中正态分布就是模型 $$M$$ ，而期望就是前文提到的 $$\theta$$ ，似然函数如下：

                           $$L(\theta|x_1,x_2,\dots,x_n)=f(x_1,x_2,\dots,x_n|\theta)=\prod\limits_{i=1}^nf(x_i|\theta)$$ 

正态分布的公式，当第一参数（期望）为0，第二参数（方差）为1时，分布为标准正态分布：

                                    $$M=f(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2}),\ N(\mu,\sigma^2)$$ 

所以似然值为：

                          $$P(x_1,x_2,\dots,x_n|M)=(\frac{1}{\sqrt{2\pi}\sigma})^n\exp(-\frac{1}{2\sigma^2}\sum\limits_{i=1}^n(x-\mu)^2)$$ 

对上式求导可得：

                                    $$\hat{l}'(\theta)=0\Rightarrow\sum\limits_{i=1}^nx_i-n\mu=0\Rightarrow\mu=\frac{1}{n}\sum\limits_{i=1}^nx_i$$ 

#### 综上所述，可得求解最大似然估计的一般过程为：

1. 写出似然函数；

2. 如果无法直接求导的话，对似然函数取对数；

3. 求导数 ；

4. 求解模型中参数的最优值。

## [最大后验概率估计\(Maximum a posteriori\)](https://www.cnblogs.com/liliu/archive/2010/11/24/1886110.html)

假设有五个袋子，各袋中都有无限量的饼干\(樱桃口味或柠檬口味\)，已知五个袋子中两种口味的比例分别是

1.樱桃 100%    2.樱桃 75% + 柠檬 25%    3.樱桃 50% + 柠檬 50%    4.樱桃 25% + 柠檬 75%    5.柠檬 100%

如果只有上所述条件，从同一个袋子中连续拿到2个柠檬饼干，那这个袋子最有可能是上述五个的哪一个？

      我们首先采用最大似然估计来解这个问题，写出似然函数。假设从袋子中能拿出柠檬饼干的概率为 $$p$$ \(我们通过这个概率 $$p$$ 来确定是从哪个袋子中拿出来的\)，则似然函数可以写作

                                                                  $$p(两个柠檬饼干|袋子) = p^2$$ 

由于 $$p$$ 的取值是一个离散值，即上面描述中的0,25%，50%，75%，1。我们只需要评估一下这五个值哪个值使得似然函数最大即可，得到为袋子5。这里便是最大似然估计的结果。

上述最大似然估计有一个问题，就是没有考虑到模型本身的概率分布，下面我们扩展这个饼干的问题。

假设拿到袋子1或5的概率 $$g$$ 都是0.1，拿到2或4的概率都是0.2，拿到3的概率是0.4，那同样上述问题的答案呢？这个时候就变 $$MAP$$ 了（若本例中拿到五个袋子概率都是0.2，即 $$\theta$$ 为均匀分布时， $$MAL = MLE$$ ；当 $$p(\theta)$$不同时，即本例子， $$MAP \neq MLE$$ ）。我们根据公式

                                                                $$\theta_{MAP} = \mathop{argmax}\limits_\theta P(\theta)P(x|\theta)$$ 

写出我们的

                                                                               $$MAP = p^2 \times g$$ 

 根据题意的描述可知， $$p$$ 的取值分别为0,25%, 50%, 75%, 1， $$g$$ 的取值分别为0.1, 0.2, 0.4, 0.2, 0.1.分别计算出 $$MAP$$ 函数的结果为：0, 0.0125, 0.125, 0.28125, 0.1.由上可知，通过 $$MAP$$ 估计可得结果是从第四个袋子中取得的最高。

## EM算法\(Expectation–maximization Algorithm\)

EM算法是一种迭代算法，用于含有隐变量的概率模型参数的极大似然估计，或极大后验概率估计。EM算法的每次迭代由两步组成：E步：求期望\(Expectation\)；M步求极大\(Maximization\)。

#### 算法步骤

输入：观测变量数据 $$Y$$ ，隐变量数据 $$Z$$ ，联合分布 $$P(Y,Z|\theta)$$ ，条件概率分布 $$P(Z|Y,\theta)$$ 

输出：模型参数 $$\theta$$ 

（1）选择参数的初值 $$\theta^{(0)}$$ ，开始迭代

（2）E步：记 $$\theta^{(i)}$$ 为第 $$i$$ 次迭代参数 $$\theta$$ 的估计值，在第 $$i+1$$ 次迭代的E步，计算

                   $$Q(\theta,\theta^{(i)})=E_Z[\log P(Y,Z|\theta)|Y,\theta^{(i)}]=\sum\limits_Z\log P(Y,Z|\theta)P(Z|Y,\theta^{(i)})$$ 

                    这里， $$P(Z|Y,\theta^{(i)})$$ 是在给定观测数据 $$Y$$ 和当前的参数估计 $$\theta^{(i)}$$ 下隐变量数据 $$Z$$ 的条件概

                    率分布。

（3）M步：求使 $$Q(\theta,\theta ^{(i)})$$ 极大化的 $$\theta$$ ，确定第 $$i+1$$ 次迭代的参数的估计值 $$\theta^{(i+1)}$$ 

                    $$\theta^{(i+1)}=\arg\max\limits_\theta Q(\theta,\theta^{(i)})$$ 

（4）重复第（2）步和第（3）步，直至收敛。

#### 步骤说明

步骤（1）参数的初值可以任意选择，但需注意EM算法对初值是敏感的

步骤（2）E步求 $$Q(\theta,\theta^{(i)})$$ 。 $$Q$$ 函数式中 $$Z$$ 是未观测数据， $$Y$$ 是观测数据。注意 $$Q(\theta,\theta^{(i)})$$ 的第 $$1$$ 个变元表示要极大化的参数，第 $$2$$ 个变元表示参数的当前估计值。每次迭代实际在求 $$Q$$ 函数及其极大。

步骤（3）M步求 $$Q(\theta,\theta^{(i)})$$ 得极大化，得到 $$\theta^{(i+1)}$$ ，完成一次迭代 $$\theta^{(i)}\to\theta^{(i+1)}$$ ，每次迭代使似然函数增大或达到局部极值。

步骤（4）给出停止迭代的条件，一般是对较小的正数 $$\varepsilon_1,\varepsilon_2$$ ，满足 $$||\theta^{(i+1)}-\theta^{(i)}||<\varepsilon_1$$ 或 $$||Q(\theta^{i+1},\theta^{(i)})-Q(\theta^{i},\theta^{(i)})||<\varepsilon_2$$ 则停止迭代。

#### Q函数

E步中的 $$Q(\theta,\theta^{(i)})$$ 是EM算法的核心，称为 $$Q$$ 函数。完全数据的对数似然函数 $$\log P(Y,Z|\theta)$$ 关于在给定观测数据 $$Y$$ 和当前参数 $$\theta^{(i)}$$ 下对未观测数据 $$Z$$ 的条件概率分布 $$P(Z|Y,\theta^{(i)})$$ 的期望称为 $$Q$$ 函数

                    $$Q(\theta,\theta^{(i)})=E_Z[\log P(Y,Z|\theta)|Y,\theta^{(i)}]=\sum\limits_Z\log P(Y,Z|\theta)P(Z|Y,\theta^{(i)})$$ 

#### 算法举例

假设有 $$3$$ 枚硬币，分别记作 $$A,B,C$$ 。这些硬币正面出现的概率分别是 $$\pi,p,q$$ 。进行如下掷硬币试验：先掷硬币 $$A$$ ，根据其结果选出硬币 $$B$$ 或硬币 $$C$$ ，正面选硬币 $$B$$ ，反面选 $$C$$ ；然后掷选出的硬币，掷硬币的结果，出现正面记作 $$1$$ ，出现反面记作 $$0$$ ；独立地重复 $$n$$ 次试验（本例 $$n = 10$$ ），结果如下

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 结果 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 |

假设只能观测到掷硬币的结果，不能观测掷硬币的过程。问如何估计三枚硬币正面出现的概率，即三硬币模型的参数。

三硬币模型可以写作： $$P(y|\theta)=\sum\limits_zP(y,z|\theta)=\sum\limits_zP(z|\theta)P(y|z,\theta)$$ 

                                                        $$=\pi p^y(1-p)^{1-y}+(1-\pi)q^y(1-q)^{1-y}$$ 

这里，随机变量 $$y$$ 是观测变量，表示一次试验观测的结果是 $$1$$ 或 $$0$$ ；随机变量 $$z$$ 是隐变量，表示未观测到的掷硬币 $$A$$ 的结果； $$\theta=(\pi,p,q)$$ 是模型参数。这一模型是以上数据的生成模型。注意，随机变量 $$y$$ 的数据可以观测，随机变量 $$z$$ 的数据不可观测。将观测数据表示为 $$Y = (Y_1,Y_2,\dots,Y_n)^T$$ ，未观测数据表示为 $$Z = (Z_1,Z_2,\dots,Z_n)^T$$ ，则观测数据的似然函数为

                                                       $$P(Y|\theta)=\sum\limits_ZP(Z|\theta)P(Y|Z,\theta)$$ 

即

                                  $$P(Y|\theta)=\prod\limits_{j=1}^n[\pi p^{y_j}(1-p)^{1-{y_j}}+(1-\pi)q^{y_j}(1-q)^{1-{y_j}}]$$ 

考虑求模型参数 $$\theta=(\pi,p,q)$$ 的极大似然估计，即

                                                                 $$\hat{\theta}=\arg\max\limits_\theta\log P(Y|\theta)$$ 

这个问题没有解析解，只有通过迭代的方法求解。EM算法就是可以用于求解这个问题的一种迭代算法。

EM算法首先选取参数的初值，记作 $$\theta^{(0)}=(\pi^{(0)},p^{(0)},q^{(0)})$$ ，然后通过下面的迭代计算参数的估计值，直到收敛为止。第 $$i$$ 次迭代参数的估计值为 $$\theta^{(i)}=(\pi^{(i)},p^{(i)},q^{(i)})$$ 。EM算法的第 $$i+1$$ 次迭代如下

E步：计算在模型参数 $$\pi^{(i)},p^{(i)},q^{(i)}$$ 下观测数据 $$y_i$$ 来自掷硬币 $$B$$ 的概率

                                          $$\mu_j^{(i+1)}=\frac{\pi^{(i)}(p^{(i)})^{y_j}(1-p^{(i)})^{1-y_j}}{\pi^{(i)}(p^{(i)})^{y_j}(1-p^{(i)})^{1-y_j}+(1-\pi^{(i)})(q^{(i)})^{y_j}(1-q^{(i)})^{1-y_j}}$$ 

M步：计算模型参数的新估计值

                              $$\pi^{(i+1)} = \frac{1}{n}\sum\limits_{j=1}^n\mu_j^{(i+1)}$$     $$p^{(i+1)} = \frac{\sum\limits_{j=1}^n\mu_j^{(i+1)}y_j}{\sum\limits_{j=1}^n\mu_j^{(i+1)}}$$     $$q^{(i+1)} = \frac{\sum\limits_{j=1}^n(1-\mu_j^{(i+1)})y_j}{\sum\limits_{j=1}^n(1-\mu_j^{(i+1)})}$$ 

进行数字计算。假设模型参数的初值取为

                                                   $$\pi^{(0)}=0.5,\ \ \ p^{(0)}=0.5,\ \ \ q^{(0)}=0.5$$ 

由E步得到对于 $$y_j=1$$ 与 $$y_j=0$$ 均有 $$\mu_j^{(1)}=0.5$$ 

由M步迭代公式得到

                                                   $$\pi^{(1)}=0.5,\ \ \ p^{(1)}=0.6,\ \ \ q^{(1)}=0.6$$ 

再回到E步，得到 $$\mu_j^{(2)}=0.5,\ \ \ j=1,2,\dots,10$$ 

继续M步，得

                                                   $$\pi^{(2)}=0.5,\ \ \ p^{(2)}=0.6,\ \ \ q^{(2)}=0.6$$ 

收敛，于是得到模型的参数 $$\theta$$ 的极大似然估计

                                                          $$\hat{\pi}=0.5,\ \ \ \hat{p}=0.6,\ \ \ \hat{q}=0.6$$ 

如果取初值 $$\pi^{(0)}=0.4,\ \ \ p^{(0)}=0.6,\ \ \ q^{(0)}=0.7$$ ，那么得到的模型参数的极大似然估计是 $$\hat{\pi}=0.4064,\ \ \ \hat{p}=0.5368,\ \ \ \hat{q}=0.6432$$ 。这就是说，EM算法与初值的选取有关，选择不同的初值可能得到不同的参数估计值。

### [Code实现](https://github.com/fengdu78/lihang-code/blob/master/code/%E7%AC%AC9%E7%AB%A0%20EM%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E6%8E%A8%E5%B9%BF%28EM%29/em.ipynb)

E-step： $$\mu^{i+1}=\frac{\pi (p^i)^{y_i}(1-(p^i))^{1-y_i}}{\pi (p^i)^{y_i}(1-(p^i))^{1-y_i}+(1-\pi) (q^i)^{y_i}(1-(q^i))^{1-y_i}}$$ 

```python
import numpy as np
import math

pro_A, pro_B, por_C = 0.5, 0.5, 0.5

def pmf(i, pro_A, pro_B, por_C):
    pro_1 = pro_A * math.pow(pro_B, data[i]) * math.pow((1-pro_B), 1-data[i])
    pro_2 = pro_A * math.pow(pro_C, data[i]) * math.pow((1-pro_C), 1-data[i])
    return pro_1 / (pro_1 + pro_2)
```

M-step： $$\pi^{i+1}=\frac{1}{n}\sum_{j=1}^n\mu^{i+1}_j， p^{i+1}=\frac{\sum_{j=1}^n\mu^{i+1}_jy_i}{\sum_{j=1}^n\mu^{i+1}_j}， q^{i+1}=\frac{\sum_{j=1}^n(1-\mu^{i+1}_jy_i)}{\sum_{j=1}^n(1-\mu^{i+1}_j)}$$ 

```python
class EM:
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob
        
    # e_step
    def pmf(self, i):
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i]) * math.pow((1-self.pro_B), 1-data[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, data[i]) * math.pow((1-self.pro_C), 1-data[i])
        return pro_1 / (pro_1 + pro_2)
    
    # m_step
    def fit(self, data):
        count = len(data)
        print('init prob:{}, {}, {}'.format(self.pro_A, self.pro_B, self.pro_C))
        for d in range(count):
            _ = yield
            _pmf = [self.pmf(k) for k in range(count)]
            pro_A = 1/ count * sum(_pmf)
            pro_B = sum([_pmf[k]*data[k] for k in range(count)]) / sum([_pmf[k] for k in range(count)])
            pro_C = sum([(1-_pmf[k])*data[k] for k in range(count)]) / sum([(1-_pmf[k]) for k in range(count)])
            print('{}/{}  pro_a:{:.3f}, pro_b:{:.3f}, pro_c:{:.3f}'.format(d+1, count, pro_A, pro_B, pro_C))
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C
```

测试

```python
data=[1,1,0,1,0,0,1,0,1,1]

em = EM(prob=[0.5, 0.5, 0.5])
f = em.fit(data)
next(f)

# 第一次迭代
f.send(1)

# 第二次
f.send(2)

em = EM(prob=[0.4, 0.6, 0.7])
f2 = em.fit(data)
next(f2)

f2.send(1)

f2.send(2)
```

