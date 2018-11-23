# AdaBoost

## AdaBoost算法

### 算法步骤

假设给定一个二分类的训练数据集 $$T = \{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$$ 其中，每个样本点由实例与标记组成。实例 $$x_i\in\mathcal{X}\subseteq R^n$$ ，标记 $$y_i \in \mathcal{Y}=\{-1,+1\}$$ ， $$\mathcal{X}$$ 是实例空间， $$\mathcal{Y}$$ 是标记集合。AdaBoost利用以下算法，从训练数据中学习一系列弱分类器或基本分类器，并将这些若分类器线性组合成为一个强分类器。

输入：训练数据集 $$T = \{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$$，其中 $$x_i\in\mathcal{X}\subseteq R^n$$ ，标记 $$y_i \in \mathcal{Y}=\{-1,+1\}$$ ，弱学习算法；

输出：最终分类器 $$G(x)$$ 

（1）初始化训练数据的权值分布

                        $$D_1=(w_{11},\dots,w_{1i},\dots,w_{1N}),\ \ \ w_{1i}=\frac{1}{N}, \ \ \ i=1,2,\dots,N$$ 

（2）对 $$m = 1,2,\dots,M$$ 

* （a）使用具有权值分布 $$D_m$$ 的训练数据集学习，得到基本分类器
*                         $$G_m(x):\mathcal{X}\to\{-1,+1\}$$ 
* （b）计算 $$G_m(x)$$ 在训练数据上的分类误差率
*                         $$e_m=\sum\limits_{i=1}^NP(G_m(x_i)\neq y_i)=\sum\limits_{i=1}^Nw_{mi}I(G_m(x_i)\neq y_i)$$ 
* （c）计算 $$G_m(x)$$ 的系数
*                          $$\alpha_m=\frac{1}{2}\ln\frac{1-e_m}{e_m}$$ 
* （d）更新训练数据集的权值分布
*                         $$D_{m+1}=(w_{m+1,1},\dots,w_{m+1,i},\dots,w_{m+1,N})$$ 
*                         $$w_{m+1,i}=\frac{w_{mi}}{Z_m}\exp(-\alpha_my_iG_m(x_i)),\ \ \ i=1,2,\dots,N$$ 
*                         $$Z_m=\sum\limits_{i=1}^Nw_{mi}\exp(-\alpha_my_iG_m(x_i))$$ 

（3）构建基本分类器的线性组合

                           $$f(x)=\sum\limits_{m=1}^M\alpha_mG_m(x)$$ 

           得到最终分类器

                            $$G(x)=sign(f(x))=sign(\sum\limits_{m=1}^M\alpha_mG_m(x))$$ 

### 步骤说明

步骤（1）：假设训练数据集具有均匀的权值分布，即每个训练样本在基本分类器的学习中作用相同，这一假设保证第1步能够在原始数据上学习基本分类器 $$G_1(x)$$ 

步骤（2）：AdaBoost反复学习基本分类器，在每一轮 $$m=1,2,\dots,M$$ 顺次地执行下列操作

* （a）使用当前分布 $$D_m$$ 加权的训练数据集，学习基本分类器 $$G_m(x)$$ 
* （b）计算基本分类器 $$G_m(x)$$ 在加权训练数据集上的分类误差率：
*                   $$e_m=\sum\limits_{i=1}^NP(G_m(x_i)\neq y_i)=\sum\limits_{G_m(x_i)\neq y_i}w_{mi}$$ 
*            这里， $$w_{mi}$$ 表示第 $$m$$ 轮中第 $$i$$ 个实例的权值， $$\sum\limits_{i=1}^Nw_{mi}=1$$ 。这表明， $$G_m(x)$$ 在加权的
*            训练数据集上的分类误差率是被 $$G_m(x)$$ 误分类样本的权值之和，由此可以看出数据权值分布
*            $$D_m$$ 与基本分类器 $$G_m(x)$$ 的分类误差率的关系。
* （c）计算基本分类器 $$G_m(x)$$ 的系数 $$\alpha_m\cdot \alpha_m$$ 

### 举例

## AdaBoost训练误差分析





