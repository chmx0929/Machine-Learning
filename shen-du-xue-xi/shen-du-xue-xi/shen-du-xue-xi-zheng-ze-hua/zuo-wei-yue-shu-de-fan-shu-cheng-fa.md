# 作为约束的范数惩罚

考虑经过参数范数正则化的代价函数

                                                    $$\tilde{J}(\theta;X,y) = J(\theta;X,y)+\alpha\Omega(\theta)$$ 

我们可以构造广义拉格朗日函数来最小化带约束的函数，即在原始目标函数上添加一系列惩罚项。每个惩罚使一个KKT乘子的系数以及一个表示约束是否满足的函数之间的乘积。如果想约束 $$\Omega(\theta)$$ 小于某个常数 $$k$$ ，我们可以构建广义拉格朗日函数

                                           $$\mathcal{L}(\theta,\alpha;X,y) = J(\theta;X,y)+\alpha(\Omega(\theta)-k)$$ 

这个约束问题的解：

                                                          $$\theta^*=\mathop{\arg\min}\limits_{\theta}\max\limits_{\alpha,\alpha\geq0}\mathcal{L}(\theta,\alpha)$$ 

要解决这个问题，我们需要对 $$\theta$$ 和 $$\alpha$$ 都做出调整。有许多不同的优化方法，有些可能会使用梯度下降而其他可能会使用梯度为 $$0$$ 的解析解，但在所有过程中 $$\alpha$$ 在 $$\Omega(\theta)>k$$ 时必须增加，在 $$\Omega(\theta)<k$$ 时必须减小。所有正值的 $$\alpha$$ 都鼓励 $$\Omega(\theta)$$ 收缩。最优值 $$\alpha^*$$ 也将鼓励 $$\Omega(\theta)$$ 收缩，但不会强到使得 $$\Omega(\theta)$$ 小于 $$k$$ 

为了洞察约束的影响，我们可以固定 $$\alpha^*$$ ，把这个问题看成只跟 $$\theta$$ 有关的函数：

                                   $$\theta^*=\mathop{\arg\min}\limits_{\theta}\mathcal{L}(\theta,\alpha^*)=\mathop{\arg\min}\limits_{\theta}J(\theta;X,y)+\alpha^*\Omega(\theta)$$ 

这和最小化 $$\tilde{J}$$ 的正则化训练问题是完全一样的。因此，我们可以把参数范数惩罚看作对权重强加的约束。

