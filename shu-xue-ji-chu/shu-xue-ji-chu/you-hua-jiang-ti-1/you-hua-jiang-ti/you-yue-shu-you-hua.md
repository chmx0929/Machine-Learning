# 有约束优化

## [拉格朗日乘子法\(Lagrange multiplier\)](https://zh.wikipedia.org/wiki/%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E4%B9%98%E6%95%B0)

在数学中的最优化问题中，拉格朗日乘数法是一种寻找多元函数在其变量受到一个或多个条件的约束时的极值的方法，这里的条件约束是等式约束，不等式约束使用KKT解决。这种方法可以将一个有 $$n$$ 个变量与k个约束条件的最优化问题转换为一个解有 $$n+k$$ 个变量的方程组的解的问题。这种方法中引入了一个或一组新的未知数，即拉格朗日乘数，又称拉格朗日乘子，或拉氏乘子，它们是在转换后的方程，即约束方程中作为梯度\(gradient\)的线性组合中各个向量的系数。

比如，要求$$f(x,y)$$在 $$g(x,y)=c$$ 时的最大值时，我们可以引入新变量拉格朗日乘数 $$\lambda$$ ，这时我们只需要下列拉格朗日函数的极值：

                                             $$\mathcal{L}(x,y,\lambda) = f(x,y)+\lambda\cdot(g(x,y)-c)$$ 

 更一般地，对含 $$n$$ 个变量和 $$k$$ 个约束的情况，有：

                      $$\mathcal{L}(x_1,\dots,x_n,\lambda_1,\dots,\lambda_k) = f(x_1,\dots,x_n)-\sum\limits_{i=1}^k\lambda_ig_i(x_1,\dots,x_n)$$ 

拉格朗日乘数法所得的极点会包含原问题的所有极值点，但并不保证每个极值点都是原问题的极值点。

假设有函数 $$f(x,y)$$ ，要求其极值\(最大值/最小值\)，且满足条件： $$g(x,y)=c$$ ， $$c$$ 是常数。对不同 $$d_n$$ 的值，不难想象出 $$f(x,y)=d_n$$ 的等高线。而方程 $$g$$ 的可行集所构成的线正好是 $$g(x,y) = c$$ 。想像我们沿着 $$g = c$$ 的可行集走；因为大部分情况下 $$f$$ 的等高线和 $$g$$ 的可行集线不会重合，但在有解的情况下，这两条线会相交。想像此时我们移动 $$g = c$$ 上的点，因为 $$f$$ 是连续的方程，我们因此能走到 $$f(x,y)=d_n$$ 更高或更低的等高线上，也就是说 $$d_n$$ 可以变大或变小。只有当 $$g = c$$ 和 $$f(x,y)=d_n$$ 相切，也就是说，此时，我们正同时沿着 $$g = c$$ 和 $$f(x,y) = d_n$$ 走。这种情况下，会出现极值或鞍点。

![](../../../../.gitbook/assets/lagrange_multiplier.png)

用向量的形式来表达的话，我们说相切的性质在此意味着 $$f$$ 和 $$g$$ 的切线在某点上平行，同时也意味着两者的梯度平行。此时引入一个未知标量 $$\lambda$$ ，并求解：

                                                 $$\nabla[f(x,y)+\lambda(g(x,y)-c)] = 0$$ 

一旦求出 $$\lambda$$ 的值，将其套入下式，易求在无约束条件下的极值和对应的极值点：

                                              $$F(x,y,\lambda) = f(x,y)+\lambda(g(x,y)-c)$$ 

新方程 $$F(x,y,\lambda)$$ 在达到极值时与 $$f(x,y)$$ 相等，因为 $$F(x,y,\lambda)$$ 达到极值时 $$g(x,y)-c$$ 总等于零

#### Example

求 $$f(x,y)=x^2y$$ 满足 $$x^2+y^2 = 1$$ 的最小值。

因为只有一个限制条件，我们只需要用一个乘数 $$\lambda$$ 

$$\Phi(x,y,\lambda) = f(x,y)+\lambda(g(x,y)-c) = x^2y+\lambda((x^2+y^2)-1)$$ 

将所有 $$\Phi$$ 方程的偏微分设为零，得到一个方程组，最小值是以下方程组的解中的一个：

                     $$2xy+2\lambda x = 0$$ 

                     $$x^2+2\lambda y =0$$ 

                     $$x^2+y^2-1 = 0$$ 

## [卡罗需－库恩－塔克条件\(KKT\)](https://www.zhihu.com/question/23311674)

卡罗需-库恩-塔克条件\(Kuhn-Tuhn, KKT\)是在满足一些规则的条件\(可以为不等式\)下，一个非线性规划\(Nonlinear Programming\)问题有最优化解法的一个必要和充分条件，是一个广义化拉格朗日乘数的成果。

现在考虑不等式 $$g(x)\leq 0$$ ，此时最优点要不在 $$g(x)<0$$ 的区域中，或在边界 $$g(x) = 0$$ 上。

1. 1、对于 $$g(x)<0$$ 的情形，约束 $$g(x)\leq0$$ 不起作用，可直接通过条件 $$\nabla f(x) = 0$$ 来获得最优点；这等价于将 $$\lambda$$ 置零然后对 $$\nabla_x\mathcal{L}(x,\lambda)$$ 置零得到最优点。
2. 2、 $$g(x) =0$$ 的情形类似于上面等式约束的分析，但需要注意的是，此时 $$\nabla f(x^*) $$ 的方向必须与 $$\nabla g(x^*) $$ 相反\(即一个增大另一个必须减小，才能使两者和为零\)，即存在常数 $$\lambda>0$$ \(若 $$\lambda<0$$则会出现 $$g(x)>0$$，不符合约束 \)使得 $$\nabla f(x^*) +\lambda\nabla g(x^*) =0$$整合这两种情形，必满足 $$\lambda g(x)=0$$ 

因此，在约束 $$g(x)\leq 0 $$ 下最小化 $$f(x)$$ ，可转化为在如下约束下最小化 $$\mathcal{L}(x,\lambda)=f(x)+\lambda g(x)$$ 的拉格朗日函数：

                                                                     $$\begin{cases}h(x)=0\\g(x)\leq0\\ \lambda\geq 0 \\ \lambda g(x) = 0\end{cases}$$ 

上式即称为Karush-Kuhn-Tucker\(KKT\)条件。上式可推广到多个约束，比如问题：

                         $$\max_xf(x)\ \ \ s.t.\ \ \ h_j(x)=0,j=1,\dots,q\ ;\ \ g_i(x)\leq0,i=1,\dots,p$$ 

也就是说，自变量 $$x$$ 是一个 $$n$$ 维向量，要最大化一个目标函数 $$f$$ ，满足若干等式和不等式约束。KKT条件宣称，如果有一个点 $$x^*$$ 是满足所有约束的极值点，则

                          $$\nabla f(x^*)=\sum\limits_j\lambda_j\nabla h_j(x^*)+\sum\limits_i\mu_i\nabla g_j(x^*)\ \ \ \ \mu_i\geq0,\ \mu_ig_i(x^*) = 0$$ 

简单说，就是在极值处， $$f$$ 的梯度是一系列等式约束 $$h_j$$ 的梯度和不等式约束 $$g_i$$ 的梯度的线性组合。在这个线性组合中，等式约束梯度的权值 $$\lambda_j$$ 没有要求；不等式约束梯度的权值 $$\mu_i$$ 是非负的，并且如果每个 $$g_i(x^*)$$ 严格小于 $$0$$ ，那这个约束不会出现在加权式子中，因为对应的权值 $$\mu_i$$ ，必须为 $$0$$ .换句话说，只有 $$x^*$$ 恰好在边界 $$g_i=0$$ 上的那些 $$g_i$$ 的梯度才会出现在加权式中。如果去掉不等式约束部分，那么上式就是拉格朗日乘子法的精确表述。

给定一个优化问题，我们把满足所有约束条件的n维空间区域称为可行域。从可行域中的每一个点 $$x$$ 朝某个方向 $$v$$ 出发走一点点，如果还在可行域中，或者偏离可行域的程度很小，准确地说，偏移量是行进距离的高阶无穷小量，那么我们就说 $$v$$ 是一个可行方向。我们用 $$F(x)$$ 表示点 $$x$$ 的所有可行方向的集合。对于 可行域中的一个极大值点 $$x^*$$ ，它的可行方向集合为 $$F(x^*)$$ ，从 $$x^*$$ 朝 $$F(x^*)$$ 中某个方向走一小步，那么落点仍然\(近似\)在可行域中。 $$x^*$$ 是局部最大值点就意味着在这些可行方向上目标函数 $$f(x)$$ 不能增大，从而我们得到这样一个结论： 在极值点 $$x^*$$ ，让目标函数增大的方向不能在 $$F(x^*)$$ 中。

#### Example

求解    $$\mathop{\min} (x_1^2+x_2^2) \ \ \ s.t.\ x_1+x_2=1, x_2\leq \alpha$$ 

写出拉格朗日函数：

                          $$\mathcal{L}(x_1,x_2,\lambda,\mu) = x_1^2+x_2^2+\lambda(1-x_1-x_2)+\mu(x_2-\alpha)$$ 

KKT方程组：

                         $$\begin{cases}\frac{\partial L}{\partial x_i} = 0 \to \frac{\partial L}{\partial x_1} = 2x_1-\lambda = 0, \ \ \frac{\partial L}{\partial x_2} = 2x_2-\lambda+\mu = 0\\ x_1+x_2 - 1=0\\ x_2-\alpha\leq0\\ \mu\geq0\\\mu(x_2-\alpha) = 0 \end{cases}$$ 

## 拉格朗日对偶性\(Lagrange duality\)

在约束最优化问题中，常常利用拉格朗日对偶性将原始问题转换为对偶问题，通过解对偶问题而得到原始问题的解。

#### 原始问题

假设 $$f(x),\ c_i(x),\ h_j(x)$$ 是定义在 $$R^n$$ 上的连续可微函数。考虑约束最优化问题：

               $$\mathop{\min}\limits_{x\in R^n}f(x)\ \ \ s.t.\ \  c_i(x)\leq 0,\ i=1,2,\dots,k\ \ \ \ h_j(x)=0,\ j=1,2,\dots,l$$ 

称此约束最优化问题为原始最优化问题或原始问题。

首先，引入拉格朗日函数：

                                $$\mathcal{L}(x,\alpha,\beta) = f(x)+\sum\limits_{i=1}^k\alpha_ic_i(x)+\sum\limits_{j=1}^l\beta_jh_j(x)$$ 

这里， $$x = (x^{(1)},x^{(2)},\cdots,x^{(n)})^T\in R^n,\ \alpha_i,\ \beta_j$$ 是拉格朗日乘子， $$\alpha\geq0$$ 。考虑 $$x$$ 的函数

                                                $$\theta_P(x)=\mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\mathcal{L}(x,\alpha,\beta)$$ 

这里，下标 $$P$$ 表示原始问题。假定给定某个 $$x$$ 。如果 $$x$$ 违反原始问题的约束条件，即存在某个 $$i$$ 使得 $$c_i(w)>0$$ 或者存在某个 $$j$$ 使得 $$h_j(w)\neq0$$ ，那么就有

                           $$\theta_P(x)=\mathop{\max}\limits_{\alpha,\beta:\alpha\geq0}[f(x)+\sum\limits_{i=1}^k\alpha_ic_i(x)+\sum\limits_{j=1}^l\beta_jh_j(x)]=+\infty$$ 

因为若某个 $$i$$ 使约束 $$c_i(x)>0$$ ，则可令 $$\alpha_i\to+\infty$$ ，若某个 $$j$$ 使约束 $$h_j(x)\neq0$$ ，则可令 $$\beta_jh_j(x)\to+\infty$$ ，而降将其余各项 $$\alpha_i,\ \beta_j$$ 均取值为 $$0$$ 

相反地，若 $$x$$ 满足等式和不等式约束，则可知 $$\theta_P(x)=f(x)$$ ，因此

                                            $$\theta_P(x)=\begin{cases}f(x),\ x满足原始约束\\+\infty,\ 其他\end{cases}$$ 

所以如果考虑极小化问题

                                            $$\mathop{\min}\limits_x\theta_P(x)=\mathop{\min}\limits_x\mathop{\max}\limits_{\alpha,\beta:\alpha\geq0}\mathcal{L}(x,\alpha,\beta)$$ 

它是与原始最优化问题等价的，即他们有相同解。问题 $$\mathop{min}\limits_x\mathop{max}\limits_{\alpha,\beta:\alpha\geq0}\mathcal{L}(x,\alpha,\beta)$$ 称为广义拉格朗日函数的极小极大问题。这样一来，就把原始最优化问题表示为拉格朗日函数的极小极大问题。为了方便，定义原始问题的最优值

                                                         $$p^* = \mathop{\min}\limits_x\theta_P(x)$$ 

#### 对偶问题

定义 $$\theta_D(\alpha,\beta)=\mathop{\min}\limits_x\mathcal{L}(x,\alpha,\beta)$$ 再考虑极大化，即

                                          $$\mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\theta_D(\alpha,\beta) = \mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\mathop{\min}\limits_x\mathcal{L}(x,\alpha,\beta)$$ 

可以将广义拉格朗日函数的极大极小为表示为约束最优化问题：

                     $$\mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\theta_D(\alpha,\beta) = \mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\mathop{\min}\limits_x\mathcal{L}(x,\alpha,\beta)\ \ \ \ s.t.\ \alpha_i\geq0,\ i=1,2,\dots,k$$ 

称为原始问题的对偶问题。定义对偶问题的最优值

                                                        $$d^* = \mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\theta_D(\alpha,\beta)$$ 

#### 原始问题和对偶问题关系

若原始问题和对偶问题都有最优值，则

                           $$d^* = \mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\mathop{\min}\limits_x\mathcal{L}(x,\alpha,\beta)\leq \mathop{\min}\limits_x\mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\mathcal{L}(x,\alpha,\beta)=p^*$$ 

证明：

由 $$\mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\theta_D(\alpha,\beta) = \mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\mathop{\min}\limits_x\mathcal{L}(x,\alpha,\beta)$$ 和 $$\theta_P(x)=\mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\mathcal{L}(x,\alpha,\beta)$$ ，对任意 $$\alpha,\beta,x$$ 

                  $$\theta_D(\alpha,\beta)=\mathop{\min}\limits_x\mathcal{L}(x,\alpha,\beta)\leq\mathcal{L}(x,\alpha,\beta)\leq\mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\mathcal{L}(x,\alpha,\beta)=\theta_P(x)$$ 

即 $$\theta_D(\alpha,\beta)\leq\theta_P(x)$$ ，由于原始问题和对偶问题均有最优解，所以

                                                 $$\mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\theta_D(\alpha,\beta)\leq\mathop{\min}\limits_x\theta_P(x)$$ 

即

                           $$d^* = \mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\mathop{\min}\limits_x\mathcal{L}(x,\alpha,\beta)\leq \mathop{\min}\limits_x\mathop{\max}\limits_{\alpha,\beta:\alpha_i\geq0}\mathcal{L}(x,\alpha,\beta)=p^*$$ 

推论：设 $$x^*,\ \alpha^*,\ \beta^*$$ 分别是原始问题和对偶问题的可行解，并且 $$d^*=p^*$$ ，则 $$x^*,\ \alpha^*,\ \beta^*$$ 分别是原始问题和对偶问题的最优解。

## [线性规划\(Linear programming\)](https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92)

在数学中，线性规划（Linear Programming，简称LP）特指目标函数和约束条件皆为线性的最优化问题。描述线性规划问题的常用和最直观形式是标准型。标准型包括以下三个部分：

1. 1、一个需要极大化的线性函数，例如 $$c_1x_1+c_2x_2$$ 
2. 2、以下形式的问题约束，例如 $$a_{11}x_1+a_{12}x_2\leq b_1\ \ \ a_{21}x_1+a_{22}x_2\leq b_2$$ 
3. 3、和非负变量，例如 $$x_1\geq 0 , x_2\geq 0$$ 

线性规划问题通常可以用矩阵形式表达成：

                                             $$\mathop{\max}c^Tx\ \ \ \ \ s.t.Ax\leq b, x\geq0$$ 

其他类型的问题，例如极小化问题，不同形式的约束问题，和有负变量的问题，都可以改写成其等价问题的标准型。

![](../../../../.gitbook/assets/9373629.png)

## 二次规划\(Quadratic programming\)

二次规划包括凸二次优化和非凸二次优化。在此类问题中，目标函数是变量的二次函数，约束条件是变量的线性不等式。假定变量个数为 $$d$$ ，约束条件的个数为 $$m$$ ，则标准的二次规划问题形如下：

                                                         $$\mathop{\min}\limits_x\frac{1}{2}x^TQx+c^Tx\ \ s.t.\ Ax\leq b$$ 

其中 $$x$$ 为 $$d$$ 维向量， $$Q\in \mathbb{R}^{d\times d}$$ 为[实对称矩阵](https://baike.baidu.com/item/%E5%AE%9E%E5%AF%B9%E7%A7%B0%E7%9F%A9%E9%98%B5)， $$A\in \mathbb{R}^{m\times d}$$ 为[实矩阵](https://baike.baidu.com/item/%E5%AE%9E%E7%9F%A9%E9%98%B5)， $$b\in \mathbb{R}^m$$ 和 $$c\in \mathbb{R}^d$$ 为实向量， $$Ax\leq b$$ 的每一行对应一个约束。

1. 1、若 $$Q$$ 为[半正定矩阵](https://blog.csdn.net/asd136912/article/details/79146151)，则上式是凸函数，相应的二次规划是凸二次优化问题；此时若约束条件 $$Ax\leq b$$ 定义的可行域不为空，且目标函数在此可行域有下界，则该问题将有全局最小值。
2. 2、若 $$Q$$ 为[正定矩阵](https://blog.csdn.net/asd136912/article/details/79146151)，则该问题有唯一的全局最小解
3. 3、若 $$Q$$ 为非正定矩阵，则上式有多个平稳点和局部极小点的NP-hard问题

常用的二次规划解法有椭球法，内点法，增广拉格朗日法、梯度投影法等。

## 半正定规划\(Semi-Definite programming\)

半正定规划\(SDP\)是一类凸优化问题，其中的变量可组织成半正定对称矩阵形式，且优化问题的目标函数和约束都是这些变量的线性函数。给定 $$d\times d$$ 的对称矩阵 $$X、C$$ 

                                                        $$C\cdot X = \sum\limits_{i=1}^{d}\sum\limits_{j=1}^{d}C_{ij}X_{ij}$$ 

若 $$A_i(i=1,2,\dots,m)$$ 也是 $$d\times d$$ 的对称矩阵， $$b_i(i=1,2,\dots,m)$$ 为 $$m$$ 个实数，则半正定规划问题形如：

                                     $$\mathop{\min}\limits_{X} C\cdot X  \ \ \ s.t.\ A_i\cdot X=b_i,\ i=1,2,\dots,m\ \  X\succeq 0$$ 

半正定规划与线性规划都拥有线性的目标函数和约束，但半正定规划中的约束 $$X\succeq 0$$ 是一个非线性、非光滑约束条件。在优化理论中，半正定规划具有的一般性，能将几种标准的优化问题\(如线性规划、二次规划\)统一起来。

常见的用于求解线性规划的内点法经过少许改造即可求解半正定规划问题，但半正定规划的计算复杂度较高，难以直接用于大规模的问题。

