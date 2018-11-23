# AdaBoost

## AdaBoost算法步骤

假设给定一个二分类的训练数据集 $$T = \{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$$ 其中，每个样本点由实例与标记组成。实例 $$x_i\in\mathcal{X}\subseteq R^n$$ ，标记 $$y_i \in \mathcal{Y}=\{-1,+1\}$$ ， $$\mathcal{X}$$ 是实例空间， $$\mathcal{Y}$$ 是标记集合。AdaBoost利用以下算法，从训练数据中学习一系列弱分类器或基本分类器，并将这些若分类器线性组合成为一个强分类器。

输入：训练数据集 $$T = \{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$$，其中 $$x_i\in\mathcal{X}\subseteq R^n$$ ，标记 $$y_i \in \mathcal{Y}=\{-1,+1\}$$ ，弱学习算法；

输出：最终分类器 $$G(x)$$ 

（1）初始化训练数据的权值分布

                        $$D_1=(w_{11},\dots,w_{1i},\dots,w_{1N}),\ \ \ w_{1i}=\frac{1}{N}, \ \ \ i=1,2,\dots,N$$ 

（2）对 $$m = 1,2,\dots,M$$ 

* （a）使用具有权值分布 $$D_m$$ 的训练数据集学习，得到基本分类器
*                                $$G_m(x):\mathcal{X}\to\{-1,+1\}$$ 
* （b）计算 $$G_m(x)$$ 在训练数据上的分类误差率
*            $$e_m=\sum\limits_{i=1}^NP(G_m(x_i)\neq y_i)=\sum\limits_{i=1}^Nw_{mi}I(G_m(x_i)\neq y_i)$$ 
* （c）
* （d）



